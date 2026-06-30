/**
 * @file wav_writer.cpp
 * @brief WAV 文件写入实现
 *
 * 生成标准的 RIFF/WAVE 格式 16-bit PCM 文件。
 * WAV 文件结构：RIFF header + fmt chunk + data chunk。
 */

#include "wav_writer.h"

#include <cmath>
#include <cstdio>
#include <cstring>

/**
 * @brief 标准 WAV 文件头（44 字节），使用默认值初始化
 *        字段排列严格遵循 RIFF/WAVE 规范，可直接 fwrite 写入文件
 */
struct WavHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};       // RIFF 标识
    uint32_t riff_size = 0;                      // 文件大小 - 8
    char wave[4] = {'W', 'A', 'V', 'E'};        // WAVE 标识
    char fmt[4] = {'f', 'm', 't', ' '};         // fmt 子块标识
    uint32_t fmt_size = 16;                      // fmt 子块大小（PCM 固定为 16）
    uint16_t audio_format = 1;                   // 1 = PCM（线性量化）
    uint16_t channels = 1;                       // 声道数
    uint32_t sample_rate = 24000;                // 采样率
    uint32_t byte_rate = 0;                      // 每秒字节数 = sample_rate * block_align
    uint16_t block_align = 0;                    // 每帧字节数 = channels * bits_per_sample / 8
    uint16_t bits_per_sample = 16;              // 位深度
    char data[4] = {'d', 'a', 't', 'a'};        // data 子块标识
    uint32_t data_size = 0;                      // PCM 数据字节数
};

/**
 * @brief 内部函数：将 int16_t PCM 原始数据写入 WAV 文件
 *        先写 44 字节 WAV 头，再写 PCM 数据体
 */
static bool write_wav_raw(const std::string& path,
                          const int16_t* pcm,
                          size_t num_samples,
                          int sample_rate,
                          int channels) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "[wav_writer] Failed to open %s for writing\n", path.c_str());
        return false;
    }

    // 根据参数填充 WAV 头
    WavHeader hdr;
    hdr.channels = (uint16_t)channels;
    hdr.sample_rate = (uint32_t)sample_rate;
    hdr.bits_per_sample = 16;
    hdr.block_align = hdr.channels * hdr.bits_per_sample / 8;
    hdr.byte_rate = hdr.sample_rate * hdr.block_align;
    hdr.data_size = (uint32_t)(num_samples * sizeof(int16_t));
    hdr.riff_size = 36 + hdr.data_size;

    fwrite(&hdr, sizeof(WavHeader), 1, fp);
    fwrite(pcm, sizeof(int16_t), num_samples, fp);
    fclose(fp);
    return true;
}

bool write_wav(const std::string& path,
               const std::vector<int16_t>& pcm,
               int sample_rate,
               int channels) {
    return write_wav_raw(path, pcm.data(), pcm.size(), sample_rate, channels);
}

bool write_wav(const std::string& path,
               const float* data,
               size_t num_samples,
               int sample_rate,
               int channels) {
    // 将 float [-1.0, 1.0] 转换为 int16_t [-32768, 32767]
    std::vector<int16_t> pcm(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = (int16_t)lroundf(v * 32767.0f);
    }
    return write_wav_raw(path, pcm.data(), pcm.size(), sample_rate, channels);
}

bool write_wav(const std::string& path,
               const std::vector<float>& data,
               int sample_rate,
               int channels) {
    return write_wav(path, data.data(), data.size(), sample_rate, channels);
}
