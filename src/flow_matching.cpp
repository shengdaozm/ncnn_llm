/**
 * @file flow_matching.cpp
 * @brief Flow-matching ODE 求解器和 Vocoder 推理实现
 */

#include "flow_matching.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

/**
 * @brief 执行 flow 网络的单步前向推理
 *
 * 输入：当前状态 x，条件向量 condition，掩码 x_mask，时间步 t
 * 输出：速度场 v(x, t, condition)，指导 ODE 的演化方向
 *
 * ncnn 网络接口约定：
 *   in0 = x（当前状态），in1 = condition，in2 = x_mask，in3 = t（时间步标量）
 *   out0 = 速度场预测
 */
static ncnn::Mat run_flow_step(ncnn::Net& net,
                               const ncnn::Mat& x,
                               const ncnn::Mat& condition,
                               const ncnn::Mat& x_mask,
                               float t,
                               int num_threads) {
    ncnn::Mat t_mat(1);
    t_mat[0] = t;

    ncnn::Mat output;
    ncnn::Extractor ex = net.create_extractor();
    if (num_threads > 0) {
        ex.set_num_threads(num_threads);
    }
    ex.input("in0", x);
    ex.input("in1", condition);
    ex.input("in2", x_mask);
    ex.input("in3", t_mat);
    ex.extract("out0", output);
    return output;
}

/**
 * @brief 采样高斯噪声作为 ODE 初始状态 x_0
 *
 * 当 sigma > 0 时使用 sigma 作为标准差；
 * 当 temperature > 0 且 sigma <= 0 时使用 temperature 作为标准差；
 * 两者都 <= 0 时返回全零（确定性解码）。
 *
 * @param w           mel 维度
 * @param c           mel 帧数
 * @param sigma       噪声标准差
 * @param temperature 采样温度
 * @return 噪声 Mat，形状 [w, 1, c]
 */
static ncnn::Mat sample_gaussian(int w, int c, float sigma, float temperature) {
    ncnn::Mat noise(w, 1, c);
    noise.fill(0.0f);

    if (sigma <= 0.0f && temperature <= 0.0f) {
        return noise;
    }

    std::mt19937 rng(std::random_device{}());
    float scale = sigma > 0.0f ? sigma : temperature;
    std::normal_distribution<float> dist(0.0f, scale);

    float* p = noise;
    for (int i = 0; i < noise.total(); ++i) {
        p[i] = dist(rng);
    }
    return noise;
}

ncnn::Mat flow_match_decode(ncnn::Net& flow_net,
                            const ncnn::Mat& condition,
                            const ncnn::Mat& x_mask,
                            const FlowMatchingConfig& cfg,
                            int num_threads) {
    if (condition.empty() || x_mask.empty()) {
        fprintf(stderr, "[flow_matching] Empty condition or mask\n");
        return ncnn::Mat();
    }

    int mel_dim = condition.w;
    int mel_len = condition.h;

    // 从高斯噪声开始
    ncnn::Mat x = sample_gaussian(mel_dim, mel_len, cfg.sigma, cfg.temperature);

    // ODE 时间步长 dt = 1 / num_steps，从 t=0 积分到 t=1
    float dt = 1.0f / static_cast<float>(cfg.num_steps);

    if (cfg.ode_solver == 1) {
        // ---- Heun 方法（二阶改进 Euler）----
        // 每步执行两次网络前向，精度更高但计算量翻倍
        for (int step = 0; step < cfg.num_steps; ++step) {
            float t = static_cast<float>(step) * dt;
            float t_next = t + dt;

            // 第一次前向：计算当前点的速度 v1 = v(x, t)
            ncnn::Mat v1 = run_flow_step(flow_net, x, condition, x_mask, t, num_threads);
            if (v1.empty()) {
                fprintf(stderr, "[flow_matching] Heun step %d: v1 empty\n", step);
                return x;
            }

            // 用 Euler 法预估下一步状态 x_next = x + dt * v1
            ncnn::Mat x_next(mel_dim, 1, mel_len);
            const float* px = x;
            const float* pv = v1;
            float* pdst = x_next;
            for (int i = 0; i < x.total(); ++i) {
                pdst[i] = px[i] + dt * pv[i];
            }

            // 第二次前向：计算预估点的速度 v2 = v(x_next, t_next)
            ncnn::Mat v2 = run_flow_step(flow_net, x_next, condition, x_mask, t_next, num_threads);
            if (v2.empty()) {
                // 若第二次前向失败，退回 Euler 结果
                x = x_next;
                continue;
            }

            // 校正：用 v1 和 v2 的平均值作为该步的速度
            const float* pv2 = v2;
            for (int i = 0; i < x.total(); ++i) {
                pdst[i] = px[i] + 0.5f * dt * (pv[i] + pv2[i]);
            }
            x = x_next;
        }
    } else {
        // ---- Euler 方法（一阶）----
        // 每步只执行一次网络前向，速度快
        for (int step = 0; step < cfg.num_steps; ++step) {
            float t = static_cast<float>(step) * dt;

            ncnn::Mat v = run_flow_step(flow_net, x, condition, x_mask, t, num_threads);
            if (v.empty()) {
                fprintf(stderr, "[flow_matching] Euler step %d: v empty\n", step);
                return x;
            }

            // x_{n+1} = x_n + dt * v(x_n, t_n)
            float* px = x;
            const float* pv = v;
            for (int i = 0; i < x.total(); ++i) {
                px[i] += dt * pv[i];
            }
        }
    }

    return x;
}

ncnn::Mat vocoder_decode(ncnn::Net& vocoder_net,
                         const ncnn::Mat& mel,
                         int num_threads) {
    if (mel.empty()) {
        fprintf(stderr, "[flow_matching] Empty mel for vocoder\n");
        return ncnn::Mat();
    }

    // Vocoder 前向：mel-spectrogram → 时域 PCM 波形
    ncnn::Mat output;
    ncnn::Extractor ex = vocoder_net.create_extractor();
    if (num_threads > 0) {
        ex.set_num_threads(num_threads);
    }
    ex.input("in0", mel);
    ex.extract("out0", output);
    return output;
}
