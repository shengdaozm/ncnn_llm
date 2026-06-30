#pragma once

/**
 * @file flow_matching.h
 * @brief Flow-matching 解码器和 Vocoder 推理接口
 *
 * Flow-matching 是一种基于 ODE 的生成方法（类似 CosyVoice / Qwen2.5-TTS），
 * 从高斯噪声出发，通过迭代求解 ODE 逐步变换为目标 mel-spectrogram。
 * 随后通过 vocoder（如 HiFiGAN）将 mel 转换为时域 PCM 波形。
 */

#include <mat.h>
#include <net.h>

/**
 * @brief Flow-matching 解码配置
 */
struct FlowMatchingConfig {
    int num_steps = 10;         // ODE 求解步数，越多则质量越高但速度越慢
    float sigma = 0.0f;         // 初始噪声标准差，0 表示确定性
    float temperature = 1.0f;   // 采样温度，影响随机性
    int ode_solver = 0;         // 0 = Euler（一阶，快），1 = Heun（二阶，更精确）
};

/**
 * @brief Flow-matching 解码：从噪声出发，通过 ODE 求解生成 mel-spectrogram
 *
 * 求解的 ODE 形式为：dx/dt = v(x, t, condition)
 * 其中 v 是 flow 网络预测的速度场，condition 是文本/codec 条件。
 *
 * Euler 法：x_{n+1} = x_n + dt * v(x_n, t_n)
 * Heun 法（改进 Euler）：x_{n+1} = x_n + 0.5 * dt * (v(x_n, t_n) + v(x_{n+1}, t_{n+1}))
 *
 * @param flow_net    flow-matching 速度场网络
 * @param condition   条件向量 [mel_dim, mel_len]，来自 LLM 隐状态或 codec token
 * @param x_mask      长度掩码 [mel_len]，标记有效帧位置
 * @param cfg         flow-matching 配置
 * @param num_threads CPU 线程数
 * @return 生成的 mel-spectrogram [mel_dim, mel_len]
 */
ncnn::Mat flow_match_decode(ncnn::Net& flow_net,
                            const ncnn::Mat& condition,
                            const ncnn::Mat& x_mask,
                            const FlowMatchingConfig& cfg,
                            int num_threads);

/**
 * @brief Vocoder 解码：将 mel-spectrogram 转换为时域 PCM 波形
 *
 * @param vocoder_net  vocoder 网络（如 HiFiGAN）
 * @param mel          输入 mel-spectrogram
 * @param num_threads  CPU 线程数
 * @return PCM 波形 Mat，float 值域 [-1.0, 1.0]
 */
ncnn::Mat vocoder_decode(ncnn::Net& vocoder_net,
                         const ncnn::Mat& mel,
                         int num_threads);
