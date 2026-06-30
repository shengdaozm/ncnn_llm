#pragma once

/**
 * @file wav_writer.h
 * @brief WAV 文件写入工具，将 PCM 数据写为标准 16-bit PCM WAV 文件
 *
 * 提供三个重载：
 *  - int16_t 向量直接写入
 *  - float 指针（[-1, 1] 归一化）→ int16_t → WAV
 *  - float 向量（[-1, 1] 归一化）→ int16_t → WAV
 */

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief 将 int16_t PCM 数据写入 WAV 文件
 * @param path      输出文件路径
 * @param pcm       PCM 采样数据（交错排列，多声道时按 L,R,L,R... 排列）
 * @param sample_rate 采样率，如 24000
 * @param channels  声道数，默认 1（单声道）
 * @return true 写入成功，false 写入失败
 */
bool write_wav(const std::string& path,
               const std::vector<int16_t>& pcm,
               int sample_rate,
               int channels = 1);

/**
 * @brief 将 float 归一化 PCM 数据写入 WAV 文件
 * @param path        输出文件路径
 * @param data        float 数据指针，值域 [-1.0, 1.0]，超出范围会被 clamp
 * @param num_samples 采样点总数
 * @param sample_rate 采样率
 * @param channels    声道数，默认 1
 * @return true 写入成功
 */
bool write_wav(const std::string& path,
               const float* data,
               size_t num_samples,
               int sample_rate,
               int channels = 1);

/**
 * @brief 将 float 向量形式的归一化 PCM 数据写入 WAV 文件
 *        等价于 write_wav(path, data.data(), data.size(), sample_rate, channels)
 */
bool write_wav(const std::string& path,
               const std::vector<float>& data,
               int sample_rate,
               int channels = 1);
