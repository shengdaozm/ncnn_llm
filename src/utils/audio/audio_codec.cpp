/**
 * @file audio_codec.cpp
 * @brief 音频 codec 解码器实现
 */

#include "audio_codec.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

AudioCodec::AudioCodec(const std::string& model_path,
                       const json& config,
                       bool use_vulkan,
                       int num_threads)
    : use_vulkan_(use_vulkan),
      num_threads_(num_threads > 0 ? num_threads : 4) {
    try {
        // 从 model.json 的 params 段读取 codec 配置
        if (config.contains("num_codebooks")) {
            codec_config_.num_codebooks = config["num_codebooks"].get<int>();
        }
        if (config.contains("vocab_size")) {
            codec_config_.vocab_size = config["vocab_size"].get<int>();
        }
        if (config.contains("sample_rate")) {
            codec_config_.sample_rate = config["sample_rate"].get<int>();
        }
        if (config.contains("hop_length")) {
            codec_config_.hop_length = config["hop_length"].get<int>();
        }
        if (config.contains("codebook_rates")) {
            codec_config_.codebook_rates = config["codebook_rates"].get<std::vector<int>>();
        }
        // 若 codebook_rates 长度与 num_codebooks 不匹配，则填充默认值 1
        if ((int)codec_config_.codebook_rates.size() != codec_config_.num_codebooks) {
            codec_config_.codebook_rates.resize(codec_config_.num_codebooks, 1);
        }

        // 构建 codec 解码网络的 param/bin 文件路径
        std::string decoder_param = model_path + "/" + config["codec_param"].get<std::string>();
        std::string decoder_bin = model_path + "/" + config["codec_bin"].get<std::string>();

        printf("[audio_codec] Loading codec decoder from %s\n", model_path.c_str());
        printf("  codec param: %s\n", decoder_param.c_str());
        printf("  codec bin: %s\n", decoder_bin.c_str());
        printf("  num_codebooks: %d, vocab_size: %d, sample_rate: %d, hop_length: %d\n",
               codec_config_.num_codebooks, codec_config_.vocab_size,
               codec_config_.sample_rate, codec_config_.hop_length);

        // 创建 ncnn 网络并设置选项
        decoder_net_ = std::make_shared<ncnn::Net>();
        if (num_threads > 0) {
            decoder_net_->opt.num_threads = num_threads;
        }
        if (use_vulkan) {
            printf("[audio_codec] Vulkan enabled\n");
            decoder_net_->opt.use_vulkan_compute = true;
            decoder_net_->opt.use_bf16_storage = true;
        }

        // 加载模型文件，失败时标记 ok_ = false
        if (decoder_net_->load_param(decoder_param.c_str()) != 0 ||
            decoder_net_->load_model(decoder_bin.c_str()) != 0) {
            ok_ = false;
            return;
        }
    } catch (std::exception& e) {
        ok_ = false;
        fprintf(stderr, "[audio_codec] Load failed: %s\n", e.what());
    }
}

ncnn::Mat AudioCodec::tokens_to_mat(const std::vector<std::vector<int>>& codec_tokens) const {
    int num_codebooks = (int)codec_tokens.size();
    if (num_codebooks == 0) return ncnn::Mat();

    // 计算所有 codebook 中最长的帧数，不足的帧补零
    int max_frames = 0;
    for (const auto& cb : codec_tokens) {
        max_frames = std::max(max_frames, (int)cb.size());
    }

    // 构建 ncnn::Mat，形状 [max_frames, num_codebooks]
    ncnn::Mat mat(max_frames, num_codebooks);
    mat.fill(0.0f);

    // 将每层 codebook 的 token 填入对应行
    for (int cb = 0; cb < num_codebooks; ++cb) {
        float* row = mat.row(cb);
        for (int i = 0; i < (int)codec_tokens[cb].size(); ++i) {
            row[i] = static_cast<float>(codec_tokens[cb][i]);
        }
    }
    return mat;
}

std::vector<float> AudioCodec::mat_to_pcm(const ncnn::Mat& output) const {
    // 将 ncnn::Mat 输出连续拷贝为 float 向量
    int total = output.w * output.h;
    const float* p = output;
    std::vector<float> pcm(total);
    std::memcpy(pcm.data(), p, total * sizeof(float));
    return pcm;
}

std::vector<float> AudioCodec::decode(const std::vector<std::vector<int>>& codec_tokens) {
    if (!ok_ || !decoder_net_ || codec_tokens.empty()) {
        return {};
    }

    // 将多层 codebook token 转换为输入 Mat
    ncnn::Mat input = tokens_to_mat(codec_tokens);

    // 通过 ncnn 网络前向推理
    ncnn::Mat output;
    ncnn::Extractor ex = decoder_net_->create_extractor();
    ex.input("in0", input);
    ex.extract("out0", output);

    if (output.empty()) {
        fprintf(stderr, "[audio_codec] Decoder produced empty output\n");
        return {};
    }

    return mat_to_pcm(output);
}
