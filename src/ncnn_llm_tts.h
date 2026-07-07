#pragma once

/**
 * @file ncnn_llm_tts.h
 * @brief TTS（Text-to-Speech）语音合成运行时
 *
 * 继承 ncnn_llm_base，复用 KV cache、Vulkan 生命周期管理和采样逻辑。
 * 支持两种合成模式：
 *
 * 1. TTS_CODEC 模式：
 *    文本 → LLM 自回归生成离散音频 codec token → AudioCodec 解码为 PCM
 *    适用于 SNAC/EnCodec 等离散编解码器架构
 *
 * 2. TTS_FLOW 模式：
 *    文本 → LLM 生成隐状态条件 → Flow-matching ODE 求解生成 mel → Vocoder 转 PCM
 *    适用于 CosyVoice/Qwen2.5-TTS 等基于 flow-matching 的架构
 *
 * Qwen3-TTS 推理流程（codec 模式）：
 *   prefill（文本编码 + 首 token 生成）
 *   → generate_codec_tokens（talker 自回归 + code predictor 生成 32 码本）
 *   → codec_to_pcm（tokenizer decoder 解码为 PCM）
 *
 * Qwen3-TTS 架构要点：
 *   - Talker: 20 层 Transformer，每步生成 1 个主 codebook token (vocab=3072)
 *   - CodePredictor: 5 层 Transformer，每步为 talker 生成 31 个子 codebook token (vocab=2048)
 *   - SpeakerEncoder: ECAPA-TDNN，从参考音频 log-mel 提取 speaker embedding (dim=1024)
 *   - num_code_groups = 32 (1 主 + 31 子)
 */

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "ncnn_llm_gpt.h"
#include "ncnn_text_runtime.h"
#include "flow_matching.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"
#include "utils/prompt.h"
#include "utils/audio/audio_codec.h"
#include "utils/audio/wav_writer.h"

using nlohmann::json;

/**
 * @brief TTS 合成配置
 */
struct TtsConfig {
    // --- LLM 自回归采样参数 (对齐 Qwen3-TTS generate_config.json 默认值) ---
    int max_new_tokens = 2048;       // 最大生成 token 数
    float temperature = 0.9f;        // 采样温度
    float top_p = 1.0f;              // Top-p 核采样阈值
    int top_k = 50;                  // Top-k 采样
    float repetition_penalty = 1.05f;// 重复惩罚
    int do_sample = 1;               // 1 = 随机采样，0 = 贪心解码
    int num_threads = 4;             // CPU 线程数

    // --- Flow-matching 参数（仅 TTS_FLOW 模式使用，Qwen3-TTS 默认 codec 模式不需要）---
    int flow_steps = 10;             // ODE 求解步数
    float flow_sigma = 0.0f;         // 初始噪声标准差
    int flow_ode_solver = 0;         // 0 = Euler, 1 = Heun

    // --- Qwen3-TTS 特有参数 ---
    std::string language = "Auto";   // 语言: Auto/Chinese/English/...
    std::string speaker;             // 说话人 (CustomVoice 模式)
    std::string instruct;            // 指令文本 (VoiceDesign/CustomVoice 模式)

    // --- 音色克隆（Base 模型）---
    std::string reference_text;      // 参考文本
    std::string reference_audio_path;// 参考音频路径
    bool x_vector_only_mode = false; // true=仅用speaker embedding, false=ICL模式

    bool non_streaming_mode = true;  // 非流式模式
    bool debug = false;              // 调试输出
};

/**
 * @brief TTS 合成结果
 */
struct TtsResult {
    std::vector<int16_t> pcm;                    // PCM 采样数据（16-bit）
    int sample_rate = 24000;                      // 采样率
    std::vector<std::vector<int>> codec_tokens;   // 生成的 codec token（多层 codebook, [32][T]）
};

/**
 * @brief TTS 语音合成运行时，继承 ncnn_llm_base 复用基础设施
 */
class ncnn_llm_tts : public ncnn_llm_base {
public:
    /**
     * @brief 构造并加载 TTS 模型
     * @param model_path    模型目录路径，需包含 model.json 和 ncnn 权重文件
     * @param use_vulkan    是否启用 Vulkan GPU 加速
     * @param num_threads   CPU 线程数，0 表示使用默认值
     * @param vulkan_device Vulkan 设备索引
     */
    ncnn_llm_tts(const std::string& model_path,
                 bool use_vulkan = false,
                 int num_threads = 0,
                 int vulkan_device = 0);

    /**
     * @brief 文本转语音（一次性返回完整 PCM）
     * @param text 输入文本
     * @param cfg  合成配置
     * @return TtsResult，包含 PCM 数据和元信息
     */
    TtsResult synthesize(const std::string& text, const TtsConfig& cfg);

    /**
     * @brief 文本转语音（流式回调，按 chunk 输出 PCM）
     * @param text     输入文本
     * @param cfg      合成配置
     * @param callback 每个 chunk 的回调函数，参数为 PCM 数据指针和采样点数
     * @return true 合成成功
     */
    bool synthesize(const std::string& text,
                    const TtsConfig& cfg,
                    std::function<void(const int16_t*, size_t)> callback);

    bool ok() const { return ok_; }
    int sample_rate() const { return sample_rate_; }
    int num_codebooks() const { return num_codebooks_; }

private:
    // --- ncnn 网络（核心三段图）---
    std::shared_ptr<ncnn::Net> embed_net_;      // token embedding 网络
    std::shared_ptr<ncnn::Net> decoder_net_;    // talker transformer decoder（带 KV cache）
    std::shared_ptr<ncnn::Net> lm_head_net_;    // codec_head 输出投影 (hidden→3072)

    // --- ncnn 网络（Qwen3-TTS 扩展组件）---
    std::shared_ptr<ncnn::Net> text_projection_net_;  // text_hidden→hidden MLP
    std::shared_ptr<ncnn::Net> speaker_encoder_net_;  // ECAPA-TDNN: mel→speaker embedding
    std::shared_ptr<ncnn::Net> cp_decoder_net_;       // code predictor 5-layer transformer
    std::shared_ptr<ncnn::Net> cp_lm_heads_net_;      // 31 merged lm_heads (hidden→2048×31)
    std::shared_ptr<ncnn::Net> cp_codec_embeds_net_;  // 31 merged codec embeddings

    // --- ncnn 网络（flow 模式，非 Qwen3-TTS）---
    std::shared_ptr<ncnn::Net> flow_net_;       // flow-matching 速度场网络（仅 TTS_FLOW）
    std::shared_ptr<ncnn::Net> vocoder_net_;    // vocoder 网络（仅 TTS_FLOW）

    std::shared_ptr<BpeTokenizer> bpe_;          // 文本分词器
    std::unique_ptr<AudioCodec> audio_codec_;   // 音频 codec 解码器（仅 TTS_CODEC）

    // --- 合成模式 ---
    enum TtsMode {
        TTS_CODEC = 0,  // codec 模式：LLM → codec token → AudioCodec → PCM
        TTS_FLOW = 1    // flow 模式：LLM → 隐状态 → flow-matching → mel → vocoder → PCM
    } tts_mode_ = TTS_CODEC;

    // --- 模型参数 ---
    int bos_ = 0;               // BOS token id
    int eos_ = 0;               // EOS token id
    int attn_cnt_ = 20;         // talker attention 层数（KV cache 组数）
    int rope_head_dim_ = 128;   // RoPE 头维度
    float rope_theta_ = 1000000.0f;  // RoPE 基频

    // Code Predictor 参数
    int cp_attn_cnt_ = 5;       // code predictor 层数
    int cp_num_kv_heads_ = 8;   // code predictor KV heads
    int cp_num_heads_ = 16;     // code predictor attention heads
    int cp_head_dim_ = 128;     // code predictor head dim

    int num_codebooks_ = 32;       // 音频 codebook 层数 (1 主 + 31 子)
    int codec_vocab_size_ = 3072;  // talker codec 词表大小
    int cp_vocab_size_ = 2048;     // code predictor 子 codebook 词表大小
    int sample_rate_ = 24000;      // 输出采样率
    int frame_rate_ = 12;          // 音频帧率 (12Hz)
    int mel_dim_ = 128;            // mel 维度（speaker encoder 输入 + flow 模式）
    int speaker_enc_dim_ = 1024;   // speaker embedding 维度

    int vocab_size_ = 0;          // 文本词表大小
    std::string tts_model_type_ = "base"; // base / custom_voice / voice_design

    // 组件可用性标志
    bool has_speaker_encoder_ = false;
    bool has_code_predictor_ = false;
    bool has_text_projection_ = false;

    /**
     * @brief Prefill 阶段：文本编码 + 首 token 生成
     *        复用 ncnn_text_runtime 的 llm_run_text_embed / llm_run_decoder_with_kv
     * @return 包含 KV cache、首 token、position_id 的上下文
     */
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& text, const TtsConfig& cfg);

    /**
     * @brief 自回归解码阶段：talker 生成主 codebook + code predictor 生成子 codebook
     *
     * 每步流程：
     *   1. Talker decoder 前向 → codec_head → 主 codebook token (vocab=3072)
     *   2. Code predictor 31 步自回归 → 31 个子 codebook token (vocab=2048)
     *   3. 32 个 codebook embedding 求和 → 作为下一步 talker 输入
     *
     * @param ctx prefill 返回的上下文
     * @param cfg 采样配置
     * @return 多层 codebook token [32][T]，第 0 层为主 codebook
     */
    std::vector<std::vector<int>> generate_codec_tokens(
        const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx,
        const TtsConfig& cfg);

    /**
     * @brief Code predictor 自回归生成 31 个子 codebook token
     *
     * Prefill: 输入 cat(talker_past_hidden, main_codebook_embed) → [B, 2, 1024]
     * Generate: 逐 token 生成，每步用不同 lm_head，独立 KV cache
     *
     * @param talker_hidden talker 上一步的 hidden state [1, 1024]
     * @param main_codebook_token talker 生成的主 codebook token id
     * @return 31 个子 codebook token id
     */
    std::vector<int> run_code_predictor(const ncnn::Mat& talker_hidden,
                                         int main_codebook_token,
                                         const TtsConfig& cfg);

    /**
     * @brief 通过 speaker encoder 从参考音频提取 speaker embedding
     *
     * 流程：wav → log-mel → speaker_encoder ncnn → [1024]
     *
     * @param wav_path 参考音频文件路径
     * @return speaker embedding [1024]
     */
    std::vector<float> run_speaker_encoder(const std::string& wav_path);

    /**
     * @brief 计算 log-mel spectrogram（C++ 实现）
     *
     * 参数对齐 Qwen3-TTS: n_fft=1024, hop=256, win=1024, fmin=0, fmax=12000, num_mels=128
     *
     * @param wav 输入波形 float 数据
     * @param num_samples 采样点数
     * @param sample_rate 采样率
     * @return log-mel [T_mel, 128] 的 ncnn::Mat
     */
    ncnn::Mat compute_mel_spectrogram(const float* wav, int num_samples, int sample_rate) const;

    /**
     * @brief 读取 WAV 文件为 float 波形
     */
    std::vector<float> read_wav(const std::string& path, int& out_sample_rate) const;

    /**
     * @brief Codec 模式：通过 AudioCodec 将 codec token 解码为 PCM
     */
    std::vector<float> codec_to_pcm(const std::vector<std::vector<int>>& codec_tokens, const TtsConfig& cfg);

    /**
     * @brief Flow 模式：通过 flow-matching + vocoder 将条件向量转为 PCM
     * @param condition LLM 隐状态条件 [mel_dim, mel_len]
     */
    std::vector<float> flow_to_pcm(const ncnn::Mat& condition, const TtsConfig& cfg);

    /**
     * @brief 生成 RoPE 位置编码缓存
     */
    void generate_rope_cache(int seq_len, int position_id, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache) const;

    /**
     * @brief 构建因果注意力掩码（下三角矩阵）
     *        mask[i][j] = 0 (j <= i), -inf (j > i)
     */
    ncnn::Mat build_causal_mask(int seq_len) const;

    /**
     * @brief 从 merged cp_lm_heads 输出中提取指定 step 的 logits
     */
    ncnn::Mat extract_cp_logits(const ncnn::Mat& all_logits, int step) const;

    /**
     * @brief 从 merged cp_codec_embeds 输出中提取指定 step 的 embedding
     */
    ncnn::Mat extract_cp_embedding(const ncnn::Mat& all_embeds, int step) const;
};
