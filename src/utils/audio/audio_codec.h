#pragma once

/**
 * @file audio_codec.h
 * @brief 音频 codec 解码器，将离散音频 token 解码为 PCM 波形
 *
 * 支持 SNAC / EnCodec 等多层 codebook 架构的音频编解码器。
 * 每层 codebook 产生一组离散 token，所有 codebook 的 token 组合后
 * 通过 ncnn 网络解码为连续的 PCM 音频数据。
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

using nlohmann::json;

/**
 * @brief 音频 codec 配置参数
 */
struct CodecConfig {
    int num_codebooks = 32;          // codebook 层数 (Qwen3-TTS-12Hz: 1主+31子=32)
    int vocab_size = 3072;           // 每个 codebook 的词表大小
    int sample_rate = 24000;         // 输出音频采样率
    int frame_rate = 12;             // 帧率 (12Hz)
    int hop_length = 2000;           // 每帧对应的采样点数 (24000/12=2000)
};

/**
 * @brief 音频 codec 解码器，基于 ncnn 实现
 *
 * 工作流程：
 *   1. 将多层 codebook token 转换为 ncnn::Mat 输入 [max_frames, num_codebooks]
 *   2. 通过 ncnn 网络前向推理，得到 PCM 输出
 *   3. 将输出 Mat 转换为 float 向量返回
 */
class AudioCodec {
public:
    /**
     * @brief 构造并加载 codec 解码网络
     * @param model_path  模型目录路径
     * @param config      model.json 中的 params 配置段，需包含 codec_param / codec_bin 字段
     * @param use_vulkan  是否启用 Vulkan GPU 加速
     * @param num_threads CPU 线程数，0 表示使用默认值
     */
    AudioCodec(const std::string& model_path,
               const json& config,
               bool use_vulkan = false,
               int num_threads = 0);

    /**
     * @brief 将多层 codebook token 解码为 PCM 音频
     * @param codec_tokens 外层维度 = codebook 层数，内层维度 = 该 codebook 的 token 序列
     * @return PCM float 数据，值域 [-1.0, 1.0]，采样率由 sample_rate() 获取
     */
    std::vector<float> decode(const std::vector<std::vector<int>>& codec_tokens);

    bool ok() const { return ok_; }
    int sample_rate() const { return codec_config_.sample_rate; }
    int num_codebooks() const { return codec_config_.num_codebooks; }

private:
    std::shared_ptr<ncnn::Net> decoder_net_;   // codec 解码网络
    CodecConfig codec_config_;
    bool use_vulkan_ = false;
    int num_threads_ = 4;
    bool ok_ = true;

    /**
     * @brief 将多层 codebook token 转换为 ncnn::Mat
     *        输出形状 [max_frames, num_codebooks]，不足的帧补零
     */
    ncnn::Mat tokens_to_mat(const std::vector<std::vector<int>>& codec_tokens) const;

    /**
     * @brief 将 ncnn 网络输出 Mat 转换为 PCM float 向量
     */
    std::vector<float> mat_to_pcm(const ncnn::Mat& output) const;
};
