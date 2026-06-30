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
 * 推理流程：
 *   prefill（文本编码 + 首 token 生成）
 *   → generate_codec_tokens（自回归解码循环，复用 ncnn_text_runtime）
 *   → codec_to_pcm / flow_to_pcm（音频生成）
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
    // --- LLM 自回归采样参数 ---
    int max_new_tokens = 4096;       // 最大生成 token 数
    float temperature = 0.3f;        // 采样温度
    float top_p = 0.8f;              // Top-p 核采样阈值
    int top_k = 50;                  // Top-k 采样
    float repetition_penalty = 1.1f; // 重复惩罚
    int do_sample = 1;               // 1 = 随机采样，0 = 贪心解码
    int num_threads = 4;             // CPU 线程数

    // --- Flow-matching 参数（仅 TTS_FLOW 模式使用）---
    int flow_steps = 10;             // ODE 求解步数
    float flow_sigma = 0.0f;         // 初始噪声标准差
    int flow_ode_solver = 0;         // 0 = Euler, 1 = Heun

    // --- 音色克隆（预留）---
    std::string reference_text;      // 参考文本
    std::string reference_audio_path;// 参考音频路径

    bool debug = false;              // 调试输出
};

/**
 * @brief TTS 合成结果
 */
struct TtsResult {
    std::vector<int16_t> pcm;                    // PCM 采样数据（16-bit）
    int sample_rate = 24000;                      // 采样率
    std::vector<std::vector<int>> codec_tokens;   // 生成的 codec token（多层 codebook）
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
    // --- ncnn 网络 ---
    std::shared_ptr<ncnn::Net> embed_net_;      // token embedding 网络
    std::shared_ptr<ncnn::Net> decoder_net_;    // transformer decoder（带 KV cache）
    std::shared_ptr<ncnn::Net> lm_head_net_;    // lm_head 输出投影
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
    int attn_cnt_ = 32;         // attention 层数（KV cache 组数）
    int rope_head_dim_ = 64;    // RoPE 头维度
    float rope_theta_ = 1000000.0f;  // RoPE 基频

    int num_codebooks_ = 4;       // 音频 codebook 层数
    int codec_vocab_size_ = 4096; // codec 词表大小
    int sample_rate_ = 24000;     // 输出采样率
    int mel_dim_ = 128;           // mel 维度（仅 TTS_FLOW）

    int vocab_size_ = 0;          // 文本词表大小

    /**
     * @brief Prefill 阶段：文本编码 + 首个 token 生成
     *        复用 ncnn_text_runtime 的 llm_run_text_embed / llm_run_decoder_with_kv
     * @return 包含 KV cache、首 token、position_id 的上下文
     */
    std::shared_ptr<ncnn_llm_gpt_ctx> prefill(const std::string& text, const TtsConfig& cfg);

    /**
     * @brief 自回归解码阶段：逐 token 生成音频 codec token
     *        复用 ncnn_text_runtime 的 decoder + lm_head + 采样逻辑
     * @param ctx prefill 返回的上下文
     * @param cfg 采样配置
     * @return 多层 codebook token，外层 = codebook 层数，内层 = token 序列
     */
    std::vector<std::vector<int>> generate_codec_tokens(
        const std::shared_ptr<ncnn_llm_gpt_ctx>& ctx,
        const TtsConfig& cfg);

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
};
