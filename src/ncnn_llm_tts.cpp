/**
 * @file ncnn_llm_tts.cpp
 * @brief TTS 语音合成运行时实现
 *
 * 构造函数从 model.json 加载网络权重和配置；
 * synthesize() 执行 prefill → 自回归解码 → 音频生成的完整流程。
 */

#include "ncnn_llm_tts.h"
#include "sampling.h"

/**
 * @brief 构造函数：加载 model.json 和所有 ncnn 网络权重
 *
 * 加载顺序：
 *   1. 读取 model.json 配置
 *   2. 创建 embed / decoder / lm_head 网络（两种模式都需要）
 *   3. 加载 BPE 分词器
 *   4. 根据 tts_mode 加载额外的 codec / flow / vocoder 网络
 */
ncnn_llm_tts::ncnn_llm_tts(const std::string& model_path,
                            bool use_vulkan,
                            int num_threads,
                            int vulkan_device)
    : ncnn_llm_base(use_vulkan, num_threads > 0 ? num_threads : 4) {
    try {
        // 读取 model.json 配置文件
        json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            ifs >> config;
        }

        // 创建核心网络：embedding、decoder、lm_head
        embed_net_ = std::make_shared<ncnn::Net>();
        decoder_net_ = std::make_shared<ncnn::Net>();
        lm_head_net_ = std::make_shared<ncnn::Net>();

        if (num_threads > 0) {
            embed_net_->opt.num_threads = num_threads;
            decoder_net_->opt.num_threads = num_threads;
            lm_head_net_->opt.num_threads = num_threads;
        }

        // Vulkan 配置：仅 decoder 启用 Vulkan（计算密集型），embed/lm_head 走 CPU
        if (use_vulkan) {
            printf("[ncnn_llm_tts] Vulkan enabled, using device %d\n", vulkan_device >= 0 ? vulkan_device : 0);
            if (vulkan_device >= 0) {
                decoder_net_->opt.vulkan_device_index = vulkan_device;
            }
            decoder_net_->opt.use_bf16_storage = true;
            decoder_net_->opt.use_fp16_arithmetic = false;
            decoder_net_->opt.use_fp16_storage = false;
            decoder_net_->opt.use_vulkan_compute = true;
        } else {
            printf("[ncnn_llm_tts] Vulkan disabled, using CPU only\n");
        }

        std::string embed_param = model_path + "/" + config["params"]["embed_token_param"].get<std::string>();
        std::string embed_bin = model_path + "/" + config["params"]["embed_token_bin"].get<std::string>();
        std::string decoder_param = model_path + "/" + config["params"]["decoder_param"].get<std::string>();
        std::string decoder_bin = model_path + "/" + config["params"]["decoder_bin"].get<std::string>();
        std::string lm_head_param = model_path + "/" + config["params"]["lm_head_param"].get<std::string>();
        std::string lm_head_bin = model_path + "/" + config["params"]["lm_head_bin"].get<std::string>();

        printf("Loading TTS model from %s\n", model_path.c_str());
        printf("  embed param: %s\n", embed_param.c_str());
        printf("  embed bin: %s\n", embed_bin.c_str());
        printf("  decoder param: %s\n", decoder_param.c_str());
        printf("  decoder bin: %s\n", decoder_bin.c_str());
        printf("  lm_head param: %s\n", lm_head_param.c_str());
        printf("  lm_head bin: %s\n", lm_head_bin.c_str());

        embed_net_->load_param(embed_param.c_str());
        embed_net_->load_model(embed_bin.c_str());
        decoder_net_->load_param(decoder_param.c_str());
        decoder_net_->load_model(decoder_bin.c_str());
        lm_head_net_->load_param(lm_head_param.c_str());
        lm_head_net_->load_model(lm_head_bin.c_str());

        // 加载 BPE 分词器（支持 bbpe 和普通 bpe）
        std::string type = "bpe";
        if (config["tokenizer"].contains("type")) {
            type = config["tokenizer"]["type"].get<std::string>();
        }
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        bpe_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe"
        ));

        if (config["tokenizer"].contains("additional_special_tokens")) {
            std::vector<std::string> additional_special_tokens = config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>();
            for (const auto& token : additional_special_tokens) {
                bpe_->AddAdditionalSpecialToken(token);
            }
        }

        auto eos_token = config["tokenizer"]["eos"].get<std::string>();
        eos_ = (eos_token != "") ? bpe_->token_to_id().at(eos_token) : -1;

        auto bos_token = config["tokenizer"].value("bos", "");
        bos_ = (bos_token != "") ? bpe_->token_to_id().at(bos_token) : -1;

        if (config["setting"].contains("attn_cnt")) {
            attn_cnt_ = config["setting"]["attn_cnt"].get<int>();
        }
        if (config["setting"].contains("rope")) {
            auto rope_cfg = config["setting"]["rope"];
            if (rope_cfg.contains("rope_head_dim")) {
                rope_head_dim_ = rope_cfg["rope_head_dim"].get<int>();
            }
            if (rope_cfg.contains("rope_theta")) {
                rope_theta_ = rope_cfg["rope_theta"].get<float>();
            }
        }

        vocab_size_ = (int)bpe_->vocab_size();

        // 解析 TTS 模型类型
        if (config["setting"].contains("tts_model_type")) {
            tts_model_type_ = config["setting"]["tts_model_type"].get<std::string>();
        }

        // 解析 TTS 模式：codec（离散 token 解码）或 flow（flow-matching + vocoder）
        std::string tts_mode_str = "codec";
        if (config["setting"].contains("tts_mode")) {
            tts_mode_str = config["setting"]["tts_mode"].get<std::string>();
        }
        if (tts_mode_str == "flow") {
            tts_mode_ = TTS_FLOW;
        } else {
            tts_mode_ = TTS_CODEC;
        }

        // 读取音频配置参数
        if (config["setting"].contains("audio")) {
            auto audio_cfg = config["setting"]["audio"];
            if (audio_cfg.contains("num_codebooks")) {
                num_codebooks_ = audio_cfg["num_codebooks"].get<int>();
            }
            if (audio_cfg.contains("codec_vocab_size")) {
                codec_vocab_size_ = audio_cfg["codec_vocab_size"].get<int>();
            }
            if (audio_cfg.contains("sample_rate")) {
                sample_rate_ = audio_cfg["sample_rate"].get<int>();
            }
            if (audio_cfg.contains("frame_rate")) {
                frame_rate_ = audio_cfg["frame_rate"].get<int>();
            }
            if (audio_cfg.contains("mel_dim")) {
                mel_dim_ = audio_cfg["mel_dim"].get<int>();
            }
        }

        // Codec 模式：加载音频 codec 解码网络
        // 支持两种配置字段名：
        //   - tokenizer_decoder_param/bin  (Qwen3-TTS-Tokenizer-12Hz)
        //   - codec_param/bin              (旧版 SNAC/EnCodec)
        if (tts_mode_ == TTS_CODEC) {
            if (config["params"].contains("tokenizer_decoder_param")) {
                // Qwen3-TTS-Tokenizer-12Hz: 替换 AudioCodec 的配置键
                json codec_config = config["params"];
                codec_config["codec_param"] = config["params"]["tokenizer_decoder_param"];
                codec_config["codec_bin"] = config["params"]["tokenizer_decoder_bin"];
                audio_codec_ = std::make_unique<AudioCodec>(
                    model_path, codec_config, use_vulkan, num_threads);
                if (!audio_codec_->ok()) {
                    fprintf(stderr, "[ncnn_llm_tts] Tokenizer decoder load failed\n");
                    audio_codec_.reset();
                } else {
                    sample_rate_ = audio_codec_->sample_rate();
                    num_codebooks_ = audio_codec_->num_codebooks();
                }
            } else if (config["params"].contains("codec_param")) {
                audio_codec_ = std::make_unique<AudioCodec>(
                    model_path, config["params"], use_vulkan, num_threads);
                if (!audio_codec_->ok()) {
                    fprintf(stderr, "[ncnn_llm_tts] Audio codec load failed\n");
                    audio_codec_.reset();
                } else {
                    sample_rate_ = audio_codec_->sample_rate();
                    num_codebooks_ = audio_codec_->num_codebooks();
                }
            }
        }

        // Flow 模式：加载 flow-matching 网络和 vocoder
        if (tts_mode_ == TTS_FLOW) {
            if (config["params"].contains("flow_param")) {
                std::string flow_param = model_path + "/" + config["params"]["flow_param"].get<std::string>();
                std::string flow_bin = model_path + "/" + config["params"]["flow_bin"].get<std::string>();
                printf("  flow param: %s\n", flow_param.c_str());
                printf("  flow bin: %s\n", flow_bin.c_str());
                flow_net_ = std::make_shared<ncnn::Net>();
                if (num_threads > 0) flow_net_->opt.num_threads = num_threads;
                if (use_vulkan) {
                    flow_net_->opt.use_vulkan_compute = true;
                    flow_net_->opt.use_bf16_storage = true;
                }
                flow_net_->load_param(flow_param.c_str());
                flow_net_->load_model(flow_bin.c_str());
            }

            if (config["params"].contains("vocoder_param")) {
                std::string vocoder_param = model_path + "/" + config["params"]["vocoder_param"].get<std::string>();
                std::string vocoder_bin = model_path + "/" + config["params"]["vocoder_bin"].get<std::string>();
                printf("  vocoder param: %s\n", vocoder_param.c_str());
                printf("  vocoder bin: %s\n", vocoder_bin.c_str());
                vocoder_net_ = std::make_shared<ncnn::Net>();
                if (num_threads > 0) vocoder_net_->opt.num_threads = num_threads;
                if (use_vulkan) {
                    vocoder_net_->opt.use_vulkan_compute = true;
                    vocoder_net_->opt.use_bf16_storage = true;
                }
                vocoder_net_->load_param(vocoder_param.c_str());
                vocoder_net_->load_model(vocoder_bin.c_str());
            }
        }

        printf("  attn_cnt: %d, rope_head_dim: %d, rope_theta: %.1f\n", attn_cnt_, rope_head_dim_, rope_theta_);
        printf("  tts_mode: %s, num_codebooks: %d, sample_rate: %d, frame_rate: %d\n",
               tts_mode_ == TTS_CODEC ? "codec" : "flow", num_codebooks_, sample_rate_, frame_rate_);
        printf("  vocab_size: %d, eos: %d, model_type: %s\n", vocab_size_, eos_, tts_model_type_.c_str());

    } catch (std::exception& e) {
        ok_ = false;
        throw std::runtime_error(std::string("ncnn_llm_tts load model failed: ") + e.what());
    }
}

/**
 * @brief 生成 RoPE 位置编码缓存
 *        委托给 utils/rope_embed.h 中的 generate_rope_embed_cache
 */
void ncnn_llm_tts::generate_rope_cache(int seq_len, int position_id,
                                        ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const {
    generate_rope_embed_cache(seq_len, rope_head_dim_, position_id, cos_cache, sin_cache, rope_theta_);
}

/**
 * @brief 构建因果注意力掩码（下三角矩阵）
 *        mask[i][j] = 0.0 (j <= i, 可见), -1e38 (j > i, 屏蔽)
 */
ncnn::Mat ncnn_llm_tts::build_causal_mask(int seq_len) const {
    ncnn::Mat mask(seq_len, seq_len);
    mask.fill(0.0f);
    for (int i = 0; i < seq_len; ++i) {
        float* row = mask.row(i);
        for (int j = i + 1; j < seq_len; ++j) {
            row[j] = -1e38f;
        }
    }
    return mask;
}

/**
 * @brief Prefill 阶段：文本编码 + 首 token 生成
 *
 * 流程：
 *   1. 应用 chat template 构造 prompt
 *   2. BPE 分词
 *   3. token embedding 查表
 *   4. 因果掩码 + RoPE 缓存
 *   5. decoder prefill（一次性处理全部 prompt token，初始化 KV cache）
 *   6. lm_head + argmax 得到首 token
 *
 * @return 上下文，包含 KV cache、首 token、position_id
 */
std::shared_ptr<ncnn_llm_gpt_ctx> ncnn_llm_tts::prefill(const std::string& text, const TtsConfig& cfg) {
    // Qwen3-TTS prompt format:
    //   <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    // For CustomVoice/VoiceDesign, instruct is prepended as user message:
    //   <|im_start|>user\n{instruct}<|im_end|>\n
    std::string full_prompt;

    if (!cfg.instruct.empty()) {
        full_prompt += "<|im_start|>user\n" + cfg.instruct + "<|im_end|>\n";
    }

    full_prompt += "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> token_ids = bpe_->encode(full_prompt, false, false);

    if (cfg.debug) {
        printf("[ncnn_llm_tts] Prompt tokens: %d\n", (int)token_ids.size());
    }

    ncnn::Mat token_embed = llm_run_text_embed(*embed_net_, token_ids);

    int seq_len = (int)token_ids.size();
    ncnn::Mat mask = build_causal_mask(seq_len);

    ncnn::Mat cos_cache, sin_cache;
    generate_rope_cache(seq_len, 0, cos_cache, sin_cache);

    KVCache kv_cache;
    ncnn::Mat decode_out = llm_run_decoder_with_kv(*decoder_net_, token_embed, mask,
                                                    cos_cache, sin_cache,
                                                    kv_cache, attn_cnt_, true);

    ncnn::Mat last_hidden = decode_out.row_range(seq_len - 1, 1);
    ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, last_hidden);

    int next_token_id = argmax1d(logits);

    auto ctx = std::make_shared<ncnn_llm_gpt_base_ctx>();
    ctx->kv_cache = std::move(kv_cache);
    ctx->cur_token = next_token_id;
    ctx->position_id = seq_len;
    return ctx;
}

/**
 * @brief 自回归解码阶段：逐 token 生成音频 codec token
 *
 * 每步流程：
 *   1. 将上一步 token 通过 embed_net 转为 embedding
 *   2. 生成单步 RoPE 缓存
 *   3. decoder 单步前向（更新 KV cache，非 prefill 模式）
 *   4. lm_head 得到 logits
 *   5. 采样（贪心 / top-k / top-p / temperature）得到下一个 token
 *
 * 生成的 flat token 序列随后按 num_codebooks 拆分为多层 codebook token
 *
 * @param ctx prefill 返回的上下文
 * @param cfg 采样配置
 * @return 多层 codebook token
 */
std::vector<std::vector<int>> ncnn_llm_tts::generate_codec_tokens(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const TtsConfig& cfg) {

    auto ctx = ctx_in->clone();
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);

    std::vector<int> flat_tokens;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (ctx->cur_token == eos_ && eos_ >= 0) break;

        flat_tokens.push_back(ctx->cur_token);

        if (cfg.debug && step % 100 == 0) {
            printf("[ncnn_llm_tts] Step %d, token=%d, total=%d\n", step, ctx->cur_token, (int)flat_tokens.size());
        }

        ncnn::Mat cur_embed = llm_run_text_embed(*embed_net_, ctx->cur_token);

        ncnn::Mat cos_cache, sin_cache;
        generate_rope_cache(1, ctx->position_id, cos_cache, sin_cache);
        ctx->position_id++;

        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out = llm_run_decoder_with_kv(*decoder_net_, cur_embed, mask,
                                                        cos_cache, sin_cache,
                                                        ctx->kv_cache, attn_cnt_, false);

        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);

        LlmTokenSampleConfig sample_cfg;
        sample_cfg.vocab_size = vocab_size_;
        sample_cfg.temperature = cfg.temperature;
        sample_cfg.top_p = cfg.top_p;
        sample_cfg.top_k = cfg.top_k;
        sample_cfg.repetition_penalty = cfg.repetition_penalty;
        sample_cfg.do_sample = cfg.do_sample;
        int next_id = llm_select_next_token(logits, history, sample_cfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }

    printf("[ncnn_llm_tts] Generated %d codec tokens\n", (int)flat_tokens.size());

    // Qwen3-TTS-12Hz: LLM 生成交错的多码本 token 序列
    // 每 num_codebooks_ 个 token 构成一帧，对应 (T, Q) 矩阵的一行
    // token[0]→(t=0,q=0), token[1]→(t=0,q=1), ..., token[Q-1]→(t=0,Q-1)
    // token[Q]→(t=1,q=0), ...
    int num_frames = (int)flat_tokens.size() / num_codebooks_;
    std::vector<std::vector<int>> codebook_tokens(num_codebooks_);
    for (int cb = 0; cb < num_codebooks_; ++cb) {
        for (int t = 0; t < num_frames; ++t) {
            codebook_tokens[cb].push_back(flat_tokens[t * num_codebooks_ + cb]);
        }
    }

    return codebook_tokens;
}

/**
 * @brief Codec 模式：通过 AudioCodec 将多层 codec token 解码为 PCM
 */
std::vector<float> ncnn_llm_tts::codec_to_pcm(const std::vector<std::vector<int>>& codec_tokens,
                                                const TtsConfig& cfg) {
    if (!audio_codec_ || !audio_codec_->ok()) {
        fprintf(stderr, "[ncnn_llm_tts] Audio codec not available\n");
        return {};
    }

    printf("[ncnn_llm_tts] Decoding %d codebooks via codec decoder...\n", (int)codec_tokens.size());
    return audio_codec_->decode(codec_tokens);
}

/**
 * @brief Flow 模式：通过 flow-matching ODE + vocoder 将条件向量转为 PCM
 *
 * 流程：
 *   1. 构建长度掩码
 *   2. flow_match_decode：从噪声出发，ODE 迭代求解生成 mel
 *   3. vocoder_decode：mel → 时域 PCM 波形
 *
 * @param condition LLM 隐状态条件 [mel_dim, mel_len]
 */
std::vector<float> ncnn_llm_tts::flow_to_pcm(const ncnn::Mat& condition, const TtsConfig& cfg) {
    if (!flow_net_) {
        fprintf(stderr, "[ncnn_llm_tts] Flow network not available\n");
        return {};
    }

    int mel_len = condition.h;
    ncnn::Mat x_mask(mel_len);
    x_mask.fill(1.0f);

    FlowMatchingConfig fm_cfg;
    fm_cfg.num_steps = cfg.flow_steps;
    fm_cfg.sigma = cfg.flow_sigma;
    fm_cfg.temperature = cfg.temperature;
    fm_cfg.ode_solver = cfg.flow_ode_solver;

    printf("[ncnn_llm_tts] Flow-matching decode (%d steps, %s solver)...\n",
           fm_cfg.num_steps, fm_cfg.ode_solver == 1 ? "Heun" : "Euler");

    ncnn::Mat mel = flow_match_decode(*flow_net_, condition, x_mask, fm_cfg, cfg.num_threads);
    if (mel.empty()) {
        fprintf(stderr, "[ncnn_llm_tts] Flow-matching produced empty mel\n");
        return {};
    }

    if (!vocoder_net_) {
        fprintf(stderr, "[ncnn_llm_tts] Vocoder not available\n");
        return {};
    }

    printf("[ncnn_llm_tts] Vocoder decode (mel: %dx%d)...\n", mel.w, mel.h);
    ncnn::Mat pcm_mat = vocoder_decode(*vocoder_net_, mel, cfg.num_threads);
    if (pcm_mat.empty()) {
        fprintf(stderr, "[ncnn_llm_tts] Vocoder produced empty output\n");
        return {};
    }

    int total = (int)pcm_mat.total();
    std::vector<float> pcm(total);
    const float* p = pcm_mat;
    std::memcpy(pcm.data(), p, total * sizeof(float));
    return pcm;
}

/**
 * @brief 文本转语音（一次性返回完整 PCM）
 *
 * 完整流程：
 *   1. prefill：文本编码 + 首 token 生成
 *   2. generate_codec_tokens：自回归解码
 *   3. 根据 tts_mode 调用 codec_to_pcm 或 flow_to_pcm
 *   4. float PCM → int16_t PCM（clamp + 量化）
 */
TtsResult ncnn_llm_tts::synthesize(const std::string& text, const TtsConfig& cfg) {
    TtsResult result;
    result.sample_rate = sample_rate_;

    if (!ok_) return result;

    printf("[ncnn_llm_tts] Synthesizing: \"%s\"\n", text.c_str());

    auto ctx = prefill(text, cfg);
    if (!ctx) {
        fprintf(stderr, "[ncnn_llm_tts] Prefill failed\n");
        return result;
    }

    auto codec_tokens = generate_codec_tokens(ctx, cfg);
    result.codec_tokens = codec_tokens;

    // Flow 模式：将 codec token 构建为条件矩阵，通过 flow-matching 生成 mel 再转 PCM
    std::vector<float> pcm;
    if (tts_mode_ == TTS_CODEC) {
        pcm = codec_to_pcm(codec_tokens, cfg);
    } else {
        // 计算总帧数（取所有 codebook 中最长的）
        int total_tokens = 0;
        for (const auto& cb : codec_tokens) {
            total_tokens = std::max(total_tokens, (int)cb.size());
        }

        // 构建 flow-matching 条件矩阵 [mel_dim, total_tokens]
        ncnn::Mat condition(mel_dim_, total_tokens);
        condition.fill(0.0f);
        for (int cb = 0; cb < (int)codec_tokens.size() && cb < mel_dim_; ++cb) {
            float* row = condition.row(cb);
            for (int i = 0; i < (int)codec_tokens[cb].size(); ++i) {
                row[i] = static_cast<float>(codec_tokens[cb][i]);
            }
        }

        pcm = flow_to_pcm(condition, cfg);
    }

    if (pcm.empty()) {
        fprintf(stderr, "[ncnn_llm_tts] PCM generation failed\n");
        return result;
    }

    // 将 float PCM [-1.0, 1.0] 量化为 int16_t [-32768, 32767]
    result.pcm.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        result.pcm[i] = (int16_t)lroundf(v * 32767.0f);
    }

    printf("[ncnn_llm_tts] Synthesis complete: %d samples (%.2f sec)\n",
           (int)result.pcm.size(), (float)result.pcm.size() / sample_rate_);

    return result;
}

/**
 * @brief 文本转语音（流式回调版）
 *        先执行完整合成，再按 50ms chunk 分块回调 PCM 数据
 */
bool ncnn_llm_tts::synthesize(const std::string& text,
                               const TtsConfig& cfg,
                               std::function<void(const int16_t*, size_t)> callback) {
    if (!ok_) return false;

    TtsResult result = synthesize(text, cfg);
    if (result.pcm.empty()) return false;

    const int chunk_size = sample_rate_ / 20;
    for (size_t i = 0; i < result.pcm.size(); i += chunk_size) {
        size_t remaining = result.pcm.size() - i;
        size_t this_chunk = remaining < (size_t)chunk_size ? remaining : (size_t)chunk_size;
        callback(result.pcm.data() + i, this_chunk);
    }
    return true;
}
