/**
 * @file ncnn_llm_tts.cpp
 * @brief TTS 语音合成运行时实现
 *
 * 构造函数从 model.json 加载网络权重和配置；
 * synthesize() 执行 prefill → 自回归解码 → 音频生成的完整流程。
 *
 * Qwen3-TTS 架构：
 *   Talker (20层 Transformer) → 每步生成 1 个主 codebook token (vocab=3072)
 *   CodePredictor (5层 Transformer) → 每步为 talker 生成 31 个子 codebook token (vocab=2048)
 *   SpeakerEncoder (ECAPA-TDNN) → 从参考音频 mel 提取 speaker embedding (dim=1024)
 *   num_code_groups = 32 (1 主 + 31 子)
 */

#include "ncnn_llm_tts.h"
#include "sampling.h"

#include <cmath>
#include <fstream>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
            if (audio_cfg.contains("cp_vocab_size")) {
                cp_vocab_size_ = audio_cfg["cp_vocab_size"].get<int>();
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

        // 读取 Code Predictor 配置
        if (config["setting"].contains("cp_attn_cnt")) {
            cp_attn_cnt_ = config["setting"]["cp_attn_cnt"].get<int>();
        }
        if (config["setting"].contains("cp_num_kv_heads")) {
            cp_num_kv_heads_ = config["setting"]["cp_num_kv_heads"].get<int>();
        }
        if (config["setting"].contains("cp_num_heads")) {
            cp_num_heads_ = config["setting"]["cp_num_heads"].get<int>();
        }
        if (config["setting"].contains("cp_head_dim")) {
            cp_head_dim_ = config["setting"]["cp_head_dim"].get<int>();
        }

        // 加载 Speaker Encoder (ECAPA-TDNN)
        if (config["params"].contains("speaker_encoder_param")) {
            std::string se_param = model_path + "/" + config["params"]["speaker_encoder_param"].get<std::string>();
            std::string se_bin = model_path + "/" + config["params"]["speaker_encoder_bin"].get<std::string>();
            printf("  speaker_encoder param: %s\n", se_param.c_str());
            printf("  speaker_encoder bin: %s\n", se_bin.c_str());
            speaker_encoder_net_ = std::make_shared<ncnn::Net>();
            if (num_threads > 0) speaker_encoder_net_->opt.num_threads = num_threads;
            if (use_vulkan) {
                speaker_encoder_net_->opt.use_vulkan_compute = true;
                speaker_encoder_net_->opt.use_bf16_storage = true;
            }
            if (speaker_encoder_net_->load_param(se_param.c_str()) == 0 &&
                speaker_encoder_net_->load_model(se_bin.c_str()) == 0) {
                has_speaker_encoder_ = true;
                printf("  Speaker encoder loaded successfully\n");
            } else {
                fprintf(stderr, "[ncnn_llm_tts] Speaker encoder load failed\n");
                speaker_encoder_net_.reset();
            }
        }

        // 加载 Text Projection (ResizeMLP)
        if (config["params"].contains("text_projection_param")) {
            std::string tp_param = model_path + "/" + config["params"]["text_projection_param"].get<std::string>();
            std::string tp_bin = model_path + "/" + config["params"]["text_projection_bin"].get<std::string>();
            printf("  text_projection param: %s\n", tp_param.c_str());
            text_projection_net_ = std::make_shared<ncnn::Net>();
            if (num_threads > 0) text_projection_net_->opt.num_threads = num_threads;
            if (text_projection_net_->load_param(tp_param.c_str()) == 0 &&
                text_projection_net_->load_model(tp_bin.c_str()) == 0) {
                has_text_projection_ = true;
                printf("  Text projection loaded successfully\n");
            } else {
                fprintf(stderr, "[ncnn_llm_tts] Text projection load failed\n");
                text_projection_net_.reset();
            }
        }

        // 加载 Code Predictor 组件
        if (config["params"].contains("cp_decoder_param")) {
            std::string cpd_param = model_path + "/" + config["params"]["cp_decoder_param"].get<std::string>();
            std::string cpd_bin = model_path + "/" + config["params"]["cp_decoder_bin"].get<std::string>();
            printf("  cp_decoder param: %s\n", cpd_param.c_str());
            cp_decoder_net_ = std::make_shared<ncnn::Net>();
            if (num_threads > 0) cp_decoder_net_->opt.num_threads = num_threads;
            if (use_vulkan) {
                cp_decoder_net_->opt.use_vulkan_compute = true;
                cp_decoder_net_->opt.use_bf16_storage = true;
            }
            if (cp_decoder_net_->load_param(cpd_param.c_str()) != 0 ||
                cp_decoder_net_->load_model(cpd_bin.c_str()) != 0) {
                fprintf(stderr, "[ncnn_llm_tts] CP decoder load failed\n");
                cp_decoder_net_.reset();
            }
        }

        if (config["params"].contains("cp_lm_heads_param")) {
            std::string cpl_param = model_path + "/" + config["params"]["cp_lm_heads_param"].get<std::string>();
            std::string cpl_bin = model_path + "/" + config["params"]["cp_lm_heads_bin"].get<std::string>();
            printf("  cp_lm_heads param: %s\n", cpl_param.c_str());
            cp_lm_heads_net_ = std::make_shared<ncnn::Net>();
            if (num_threads > 0) cp_lm_heads_net_->opt.num_threads = num_threads;
            if (cp_lm_heads_net_->load_param(cpl_param.c_str()) != 0 ||
                cp_lm_heads_net_->load_model(cpl_bin.c_str()) != 0) {
                fprintf(stderr, "[ncnn_llm_tts] CP lm_heads load failed\n");
                cp_lm_heads_net_.reset();
            }
        }

        if (config["params"].contains("cp_codec_embeds_param")) {
            std::string cpc_param = model_path + "/" + config["params"]["cp_codec_embeds_param"].get<std::string>();
            std::string cpc_bin = model_path + "/" + config["params"]["cp_codec_embeds_bin"].get<std::string>();
            printf("  cp_codec_embeds param: %s\n", cpc_param.c_str());
            cp_codec_embeds_net_ = std::make_shared<ncnn::Net>();
            if (num_threads > 0) cp_codec_embeds_net_->opt.num_threads = num_threads;
            if (cp_codec_embeds_net_->load_param(cpc_param.c_str()) != 0 ||
                cp_codec_embeds_net_->load_model(cpc_bin.c_str()) != 0) {
                fprintf(stderr, "[ncnn_llm_tts] CP codec_embeds load failed\n");
                cp_codec_embeds_net_.reset();
            }
        }

        has_code_predictor_ = (cp_decoder_net_ != nullptr && cp_lm_heads_net_ != nullptr);
        if (has_code_predictor_) {
            printf("  Code predictor loaded (decoder + lm_heads + codec_embeds)\n");
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
        printf("  codec_vocab_size: %d, cp_vocab_size: %d, vocab_size: %d, eos: %d, model_type: %s\n",
               codec_vocab_size_, cp_vocab_size_, vocab_size_, eos_, tts_model_type_.c_str());
        printf("  has_speaker_encoder: %d, has_code_predictor: %d, has_text_projection: %d\n",
               has_speaker_encoder_ ? 1 : 0, has_code_predictor_ ? 1 : 0, has_text_projection_ ? 1 : 0);
        if (has_code_predictor_) {
            printf("  cp_attn_cnt: %d, cp_num_kv_heads: %d, cp_num_heads: %d, cp_head_dim: %d\n",
                   cp_attn_cnt_, cp_num_kv_heads_, cp_num_heads_, cp_head_dim_);
        }

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
 * @brief 构建带 KV cache 的注意力掩码
 *
 * Prefill: mask is (seq_len, seq_len) causal matrix
 * Decode:  mask is (1, past_len + 1) all zeros (single token sees all history)
 *
 * When KV cache is present, total key length = past_len + seq_len.
 * For prefill with empty cache, past_len=0, so mask is (seq_len, seq_len).
 */

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
    ctx->past_hidden = last_hidden;  // Save for code predictor
    return ctx;
}

/**
 * @brief 自回归解码阶段：talker 生成主 codebook + code predictor 生成子 codebook
 *
 * Qwen3-TTS 每步生成流程：
 *   1. Talker decoder 前向 → codec_head → 主 codebook token (vocab=3072)
 *   2. Code predictor 31 步自回归 → 31 个子 codebook token (vocab=2048)
 *   3. 32 个 codebook embedding 求和 → 作为下一步 talker 输入
 *
 * @param ctx prefill 返回的上下文
 * @param cfg 采样配置
 * @return 多层 codebook token [32][T]，第 0 层为主 codebook
 */
std::vector<std::vector<int>> ncnn_llm_tts::generate_codec_tokens(
    const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx_in,
    const TtsConfig& cfg) {

    auto ctx = ctx_in->clone();
    std::unordered_set<int> history;
    history.insert(ctx->cur_token);

    std::vector<int> main_tokens;
    std::vector<std::vector<int>> sub_tokens_per_step;

    int num_sub_codebooks = num_codebooks_ - 1;

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (ctx->cur_token == eos_ && eos_ >= 0) break;

        int main_token = ctx->cur_token;
        main_tokens.push_back(main_token);

        if (cfg.debug && step % 50 == 0) {
            printf("[ncnn_llm_tts] Step %d, main_token=%d, total=%d\n", step, main_token, (int)main_tokens.size());
        }

        // Step 1: Run code predictor to get 31 sub-codebook tokens
        std::vector<int> sub_tokens;
        if (has_code_predictor_ && cp_decoder_net_ && cp_lm_heads_net_) {
            // Get talker's last hidden state for code predictor prefill
            // We need the hidden state from the talker's last forward pass.
            // Since we don't store it, we re-run lm_head to get it... 
            // Actually, we need the hidden state BEFORE lm_head.
            // The talker decoder output IS the hidden state.
            // We store it from the previous decode step.
            
            // For the first step after prefill, we need the prefill's last hidden.
            // We'll use a simplified approach: re-embed the current token and run a single
            // talker step to get hidden, then feed to code predictor.
            // But actually, the code predictor needs the hidden from the SAME step
            // that produced main_token. So we need to capture it.
            
            // The talker forward already happened in the previous iteration (or prefill).
            // We stored the hidden in ctx as an extension. For now, use a workaround:
            // re-embed main_token and run through talker to get hidden.
            // This is not ideal but works for correctness.
            
            // Better approach: modify the decode loop to save hidden.
            // For now, we use the hidden from the decode step (stored in ctx).
            // See the decode loop below where we capture hidden.
            
            sub_tokens = run_code_predictor(ctx->past_hidden, main_token, cfg);
        }
        sub_tokens_per_step.push_back(sub_tokens);

        // Step 2: Compute combined embedding for next talker step
        // combined_embed = codec_embed(main_token) + sum(cp_codec_embed[i](sub_token[i]))
        // In the original model, all 32 codebook embeddings are summed.
        // We use embed_net_ for main codebook (codec_embedding) and cp_codec_embeds_net_ for sub-codebooks.
        
        ncnn::Mat main_embed = llm_run_text_embed(*embed_net_, main_token);

        if (!sub_tokens.empty() && cp_codec_embeds_net_) {
            // Add sub-codebook embeddings
            // cp_codec_embeds_net_ outputs [31, 1, 1, 1024] for a single token
            // We need to sum the appropriate step's embedding
            for (int i = 0; i < (int)sub_tokens.size() && i < num_sub_codebooks; ++i) {
                ncnn::Mat sub_id_mat(1);
                ((float*)sub_id_mat.data)[0] = (float)sub_tokens[i];
                
                ncnn::Extractor ex = cp_codec_embeds_net_->create_extractor();
                ex.input("in0", sub_id_mat);
                ncnn::Mat all_embeds;
                ex.extract("out0", all_embeds);
                
                if (!all_embeds.empty()) {
                    // Extract embedding for step i
                    ncnn::Mat step_embed = extract_cp_embedding(all_embeds, i);
                    if (!step_embed.empty() && !main_embed.empty()) {
                        add_mats_inplace(main_embed, step_embed);
                    }
                }
            }
        }

        // Step 3: Run talker decoder for next token
        ncnn::Mat cos_cache, sin_cache;
        generate_rope_cache(1, ctx->position_id, cos_cache, sin_cache);
        ctx->position_id++;

        ncnn::Mat mask(1, ctx->kv_cache[0].first.h + 1);
        mask.fill(0.0f);

        ncnn::Mat decode_out = llm_run_decoder_with_kv(*decoder_net_, main_embed, mask,
                                                        cos_cache, sin_cache,
                                                        ctx->kv_cache, attn_cnt_, false);

        // Save hidden state for code predictor in next step
        ctx->past_hidden = decode_out;

        ncnn::Mat logits = llm_run_lm_head(*lm_head_net_, decode_out);

        LlmTokenSampleConfig sample_cfg;
        sample_cfg.vocab_size = codec_vocab_size_;
        sample_cfg.temperature = cfg.temperature;
        sample_cfg.top_p = cfg.top_p;
        sample_cfg.top_k = cfg.top_k;
        sample_cfg.repetition_penalty = cfg.repetition_penalty;
        sample_cfg.do_sample = cfg.do_sample;
        int next_id = llm_select_next_token(logits, history, sample_cfg);

        ctx->cur_token = next_id;
        history.insert(next_id);
    }

    printf("[ncnn_llm_tts] Generated %d main codebook tokens\n", (int)main_tokens.size());

    // Assemble [32][T] codebook token matrix
    // codebook[0] = main_tokens (talker)
    // codebook[1..31] = sub_tokens from code predictor, transposed
    std::vector<std::vector<int>> codebook_tokens(num_codebooks_);
    codebook_tokens[0] = main_tokens;

    int T = (int)main_tokens.size();
    for (int cb = 1; cb < num_codebooks_; ++cb) {
        codebook_tokens[cb].resize(T);
        for (int t = 0; t < T; ++t) {
            if (t < (int)sub_tokens_per_step.size() && cb - 1 < (int)sub_tokens_per_step[t].size()) {
                codebook_tokens[cb][t] = sub_tokens_per_step[t][cb - 1];
            } else {
                codebook_tokens[cb][t] = 0;
            }
        }
    }

    return codebook_tokens;
}

/**
 * @brief Code predictor 自回归生成 31 个子 codebook token
 *
 * Prefill: 输入 cat(talker_hidden, main_codebook_embed) → [1, 2, 1024]
 *          通过 5 层 transformer → lm_head[0] → 第 1 个子 codebook token
 * Generate: 逐 token 生成，每步：
 *   - 用上一步的子 codebook token 查 cp_codec_embedding[step-1]
 *   - 通过 5 层 transformer (增量) → lm_head[step] → 下一个子 codebook token
 */
std::vector<int> ncnn_llm_tts::run_code_predictor(const ncnn::Mat& talker_hidden,
                                                     int main_codebook_token,
                                                     const TtsConfig& cfg) {
    int num_sub = num_codebooks_ - 1;
    std::vector<int> sub_tokens;

    if (!cp_decoder_net_ || !cp_lm_heads_net_) {
        return sub_tokens;
    }

    // Prefill: cat(talker_hidden, main_codebook_embed) → [1, 2, 1024]
    ncnn::Mat main_embed = llm_run_text_embed(*embed_net_, main_codebook_token);

    // Concatenate talker_hidden [1, 1, 1024] and main_embed [1, 1, 1024] → [1, 2, 1024]
    int hidden_dim = talker_hidden.w;
    ncnn::Mat prefill_input(hidden_dim, 2, 1);
    const float* th_ptr = (const float*)talker_hidden.data;
    const float* me_ptr = (const float*)main_embed.data;
    float* pi_ptr = (float*)prefill_input.data;
    memcpy(pi_ptr, th_ptr, hidden_dim * sizeof(float));
    memcpy(pi_ptr + hidden_dim, me_ptr, hidden_dim * sizeof(float));

    // Build mask for prefill [2, 2] causal
    ncnn::Mat cp_mask(2, 2);
    cp_mask.fill(0.0f);
    ((float*)cp_mask.data)[1] = -1e38f;  // mask[0][1]

    // RoPE for prefill
    ncnn::Mat cp_cos, cp_sin;
    generate_rope_cache(2, 0, cp_cos, cp_sin);

    // Run code predictor decoder prefill
    KVCache cp_kv_cache;
    ncnn::Mat cp_hidden = llm_run_decoder_with_kv(*cp_decoder_net_, prefill_input, cp_mask,
                                                    cp_cos, cp_sin,
                                                    cp_kv_cache, cp_attn_cnt_, true);

    // Get last hidden for lm_head[0]
    ncnn::Mat cp_last_hidden = cp_hidden.row_range(1, 1);

    // Run merged lm_heads → [31, 1, 1, 2048]
    ncnn::Mat all_logits;
    {
        ncnn::Extractor ex = cp_lm_heads_net_->create_extractor();
        ex.input("in0", cp_last_hidden);
        ex.extract("out0", all_logits);
    }

    // Extract step 0 logits → first sub-codebook token
    ncnn::Mat step0_logits = extract_cp_logits(all_logits, 0);
    int sub_token_0 = argmax1d(step0_logits);
    sub_tokens.push_back(sub_token_0);

    // Generate remaining 30 sub-codebook tokens (steps 1..30)
    for (int step = 1; step < num_sub; ++step) {
        // Embed previous sub-codebook token using cp_codec_embedding[step-1]
        ncnn::Mat prev_id_mat(1);
        ((float*)prev_id_mat.data)[0] = (float)sub_tokens.back();

        ncnn::Mat all_embeds;
        {
            ncnn::Extractor ex = cp_codec_embeds_net_->create_extractor();
            ex.input("in0", prev_id_mat);
            ex.extract("out0", all_embeds);
        }
        ncnn::Mat step_embed = extract_cp_embedding(all_embeds, step - 1);
        if (step_embed.empty()) {
            step_embed = ncnn::Mat(hidden_dim, 1, 1);
            step_embed.fill(0.0f);
        }

        // Single token decode
        ncnn::Mat cp_decode_mask(1, cp_kv_cache[0].first.h + 1);
        cp_decode_mask.fill(0.0f);

        ncnn::Mat cp_cos_step, cp_sin_step;
        generate_rope_cache(1, step + 1, cp_cos_step, cp_sin_step);

        ncnn::Mat cp_dec_out = llm_run_decoder_with_kv(*cp_decoder_net_, step_embed,
                                                         cp_decode_mask, cp_cos_step, cp_sin_step,
                                                         cp_kv_cache, cp_attn_cnt_, false);

        // Run merged lm_heads
        ncnn::Mat all_logits_step;
        {
            ncnn::Extractor ex = cp_lm_heads_net_->create_extractor();
            ex.input("in0", cp_dec_out);
            ex.extract("out0", all_logits_step);
        }

        // Extract step logits
        ncnn::Mat step_logits = extract_cp_logits(all_logits_step, step);
        int sub_token = argmax1d(step_logits);
        sub_tokens.push_back(sub_token);
    }

    return sub_tokens;
}

/**
 * @brief 从 merged cp_lm_heads 输出中提取指定 step 的 logits
 *
 * merged lm_heads 输出 shape: [31, 1, 1, cp_vocab_size]
 * ncnn::Mat layout: w=cp_vocab_size, h=1, c=1, d=31 (或类似)
 */
ncnn::Mat ncnn_llm_tts::extract_cp_logits(const ncnn::Mat& all_logits, int step) const {
    if (all_logits.empty()) return ncnn::Mat();

    int vocab = cp_vocab_size_;
    // ncnn::Mat for [31, B, T, vocab] with B=1, T=1:
    // total = 31 * vocab, step i starts at i * vocab
    const float* base = (const float*)all_logits.data;
    int total = all_logits.total();

    if (total == 31 * vocab) {
        ncnn::Mat out(vocab);
        memcpy(out.data, base + step * vocab, vocab * sizeof(float));
        return out;
    } else if (total == vocab) {
        return all_logits;  // single step output
    }

    int per_step = total / 31;
    ncnn::Mat out(per_step);
    memcpy(out.data, base + step * per_step, per_step * sizeof(float));
    return out;
}

/**
 * @brief 从 merged cp_codec_embeds 输出中提取指定 step 的 embedding
 */
ncnn::Mat ncnn_llm_tts::extract_cp_embedding(const ncnn::Mat& all_embeds, int step) const {
    if (all_embeds.empty()) return ncnn::Mat();

    int embed_dim = 1024;
    const float* base = (const float*)all_embeds.data;
    int total = all_embeds.total();

    if (total == 31 * embed_dim) {
        ncnn::Mat out(embed_dim);
        memcpy(out.data, base + step * embed_dim, embed_dim * sizeof(float));
        return out;
    } else if (total == embed_dim) {
        return all_embeds;
    }

    int per_step = total / 31;
    ncnn::Mat out(per_step);
    memcpy(out.data, base + step * per_step, per_step * sizeof(float));
    return out;
}

/**
 * @brief 通过 speaker encoder 从参考音频提取 speaker embedding
 */
std::vector<float> ncnn_llm_tts::run_speaker_encoder(const std::string& wav_path) {
    if (!has_speaker_encoder_ || !speaker_encoder_net_) {
        fprintf(stderr, "[ncnn_llm_tts] Speaker encoder not available\n");
        return {};
    }

    int sr = 0;
    std::vector<float> wav = read_wav(wav_path, sr);
    if (wav.empty()) {
        fprintf(stderr, "[ncnn_llm_tts] Failed to read reference audio: %s\n", wav_path.c_str());
        return {};
    }

    // Resample to 24kHz if needed
    if (sr != sample_rate_) {
        fprintf(stderr, "[ncnn_llm_tts] Warning: reference audio sr=%d, expected %d. "
                "Resampling not implemented, using as-is.\n", sr, sample_rate_);
    }

    // Compute log-mel spectrogram
    ncnn::Mat mel = compute_mel_spectrogram(wav.data(), (int)wav.size(), sample_rate_);
    if (mel.empty()) {
        fprintf(stderr, "[ncnn_llm_tts] Mel spectrogram computation failed\n");
        return {};
    }

    // Run speaker encoder
    ncnn::Mat spk_embed;
    ncnn::Extractor ex = speaker_encoder_net_->create_extractor();
    ex.input("in0", mel);
    ex.extract("out0", spk_embed);

    if (spk_embed.empty()) {
        fprintf(stderr, "[ncnn_llm_tts] Speaker encoder produced empty output\n");
        return {};
    }

    int dim = (int)spk_embed.total();
    std::vector<float> embedding(dim);
    memcpy(embedding.data(), spk_embed.data, dim * sizeof(float));

    printf("[ncnn_llm_tts] Speaker embedding extracted: dim=%d\n", dim);
    return embedding;
}

/**
 * @brief 计算 log-mel spectrogram
 *
 * 参数对齐 Qwen3-TTS: n_fft=1024, hop=256, win=1024, fmin=0, fmax=12000, num_mels=128
 */
ncnn::Mat ncnn_llm_tts::compute_mel_spectrogram(const float* wav, int num_samples, int sample_rate) const {
    const int n_fft = 1024;
    const int hop_size = 256;
    const int win_size = 1024;
    const int num_mels = mel_dim_;
    const float fmin = 0.0f;
    const float fmax = 12000.0f;

    int num_frames = (num_samples - win_size) / hop_size + 1;
    if (num_frames <= 0) {
        fprintf(stderr, "[ncnn_llm_tts] Audio too short for mel: %d samples\n", num_samples);
        return ncnn::Mat();
    }

    // Pre-compute mel filterbank [num_mels, n_fft/2+1]
    int n_freqs = n_fft / 2 + 1;
    std::vector<std::vector<float>> mel_basis(num_mels, std::vector<float>(n_freqs, 0.0f));

    // Build mel filterbank (HTK formula)
    auto hz_to_mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);
    std::vector<float> mel_points(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (num_mels + 1));
    }

    std::vector<int> bin_points(num_mels + 2);
    for (int i = 0; i < num_mels + 2; ++i) {
        bin_points[i] = (int)floorf(mel_points[i] / sample_rate * n_fft + 0.5f);
    }

    for (int m = 0; m < num_mels; ++m) {
        int left = bin_points[m];
        int center = bin_points[m + 1];
        int right = bin_points[m + 2];
        for (int k = left; k < center && k < n_freqs; ++k) {
            if (center > left) {
                mel_basis[m][k] = (float)(k - left) / (center - left);
            }
        }
        for (int k = center; k < right && k < n_freqs; ++k) {
            if (right > center) {
                mel_basis[m][k] = (float)(right - k) / (right - center);
            }
        }
    }

    // Hann window
    std::vector<float> window(win_size);
    for (int i = 0; i < win_size; ++i) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (win_size - 1)));
    }

    // Output: [num_mels, num_frames] as ncnn::Mat (w=num_mels, h=num_frames)
    // Speaker encoder expects [1, T_mel, 128] which is w=128, h=T_mel
    ncnn::Mat mel_out(num_mels, num_frames, 1);
    mel_out.fill(0.0f);

    // FFT buffer
    std::vector<float> fft_real(n_fft, 0.0f);
    std::vector<float> fft_imag(n_fft, 0.0f);

    for (int frame = 0; frame < num_frames; ++frame) {
        int start = frame * hop_size;

        // Apply window
        for (int i = 0; i < win_size; ++i) {
            fft_real[i] = wav[start + i] * window[i];
            fft_imag[i] = 0.0f;
        }
        for (int i = win_size; i < n_fft; ++i) {
            fft_real[i] = 0.0f;
            fft_imag[i] = 0.0f;
        }

        // Simple DFT (O(n^2) - can be optimized with FFTW later)
        std::vector<float> power(n_freqs);
        for (int k = 0; k < n_freqs; ++k) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < n_fft; ++n) {
                float angle = -2.0f * (float)M_PI * k * n / n_fft;
                real += fft_real[n] * cosf(angle);
                imag += fft_real[n] * sinf(angle);
            }
            power[k] = real * real + imag * imag;
        }

        // Apply mel filterbank
        float* mel_row = mel_out.row(frame);
        for (int m = 0; m < num_mels; ++m) {
            float val = 0.0f;
            for (int k = 0; k < n_freqs; ++k) {
                val += power[k] * mel_basis[m][k];
            }
            // Log compression
            val = logf(val + 1e-10f);
            mel_row[m] = val;
        }
    }

    return mel_out;
}

/**
 * @brief 读取 WAV 文件为 float 波形
 */
std::vector<float> ncnn_llm_tts::read_wav(const std::string& path, int& out_sample_rate) const {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "[ncnn_llm_tts] Cannot open wav: %s\n", path.c_str());
        return {};
    }

    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "[ncnn_llm_tts] Not a RIFF file: %s\n", path.c_str());
        return {};
    }

    uint32_t file_size;
    file.read((char*)&file_size, 4);

    char wave[4];
    file.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "[ncnn_llm_tts] Not a WAVE file: %s\n", path.c_str());
        return {};
    }

    // Parse chunks
    out_sample_rate = 24000;
    int num_channels = 1;
    int bits_per_sample = 16;
    std::vector<int16_t> pcm_data;

    while (file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;
        file.read(chunk_id, 4);
        if (!file.good()) break;
        file.read((char*)&chunk_size, 4);

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format;
            file.read((char*)&audio_format, 2);
            file.read((char*)&num_channels, 2);
            file.read((char*)&out_sample_rate, 4);
            uint32_t byte_rate;
            file.read((char*)&byte_rate, 4);
            uint16_t block_align;
            file.read((char*)&block_align, 2);
            file.read((char*)&bits_per_sample, 2);
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            int num_samples = chunk_size / (bits_per_sample / 8) / num_channels;
            pcm_data.resize(num_samples);
            file.read((char*)pcm_data.data(), chunk_size);

            // Convert to float and de-interleave if stereo
            std::vector<float> wav(num_samples);
            for (int i = 0; i < num_samples; ++i) {
                wav[i] = (float)pcm_data[i] / 32768.0f;
            }
            return wav;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    return {};
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
