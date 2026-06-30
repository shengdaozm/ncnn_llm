/**
 * @file tts_main.cpp
 * @brief TTS 语音合成 CLI 示例程序
 *
 * 用法：
 *   xmake build tts_main
 *   xmake run tts_main --model ./assets/qwen_tts --text "Hello world" --output output.wav
 *
 * 支持的参数：
 *   --model          模型目录路径
 *   --text           要合成的文本
 *   --output         输出 WAV 文件路径
 *   --vulkan         启用 Vulkan GPU 加速
 *   --vulkan-device  Vulkan 设备索引
 *   --threads        CPU 线程数
 *   --max-tokens     最大生成 token 数
 *   --temperature    采样温度
 *   --top-p          Top-p 核采样
 *   --top-k          Top-k 采样
 *   --repetition-penalty  重复惩罚
 *   --greedy         贪心解码（确定性输出）
 *   --flow-steps     Flow-matching ODE 步数
 *   --debug          调试输出
 */

#include <iostream>
#include <string>
#include "ncnn_llm_tts.h"

int main(int argc, char** argv) {
    // 默认参数
    std::string model_path = "assets/qwen_tts";
    std::string text = "Hello, this is a text to speech test.";
    std::string output_path = "tts_output.wav";
    bool use_vulkan = false;
    int num_threads = 0;
    int vulkan_device = 0;
    int max_new_tokens = 4096;
    float temperature = 0.3f;
    float top_p = 0.8f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int do_sample = 1;
    int flow_steps = 10;
    bool debug = false;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--text" && i + 1 < argc) {
            text = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--vulkan") {
            use_vulkan = true;
        } else if (arg == "--vulkan-device" && i + 1 < argc) {
            vulkan_device = atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (arg == "--repetition-penalty" && i + 1 < argc) {
            repetition_penalty = atof(argv[++i]);
        } else if (arg == "--greedy") {
            do_sample = 0;
            temperature = 0.0f;
        } else if (arg == "--flow-steps" && i + 1 < argc) {
            flow_steps = atoi(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        } else {
            fprintf(stderr, "Usage: %s --model <model_path> --text <text> [--output <wav_path>] [--vulkan] [--threads N] [--max-tokens N] [--temperature F] [--top-p F] [--top-k N] [--repetition-penalty F] [--greedy] [--flow-steps N] [--debug]\n", argv[0]);
            return 1;
        }
    }

    // 加载 TTS 模型
    printf("Loading TTS model from %s\n", model_path.c_str());

    ncnn_llm_tts tts(model_path, use_vulkan, num_threads, vulkan_device);
    if (!tts.ok()) {
        fprintf(stderr, "Failed to load TTS model\n");
        return 1;
    }

    printf("Synthesizing: \"%s\"\n", text.c_str());

    // 构建合成配置
    TtsConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = temperature;
    cfg.top_p = top_p;
    cfg.top_k = top_k;
    cfg.repetition_penalty = repetition_penalty;
    cfg.do_sample = do_sample;
    cfg.num_threads = num_threads > 0 ? num_threads : 4;
    cfg.flow_steps = flow_steps;
    cfg.debug = debug;

    // 执行语音合成
    TtsResult result = tts.synthesize(text, cfg);

    if (result.pcm.empty()) {
        fprintf(stderr, "Synthesis failed\n");
        return 1;
    }

    // 将 PCM 数据写入 WAV 文件
    printf("Writing WAV to %s (%d samples, %.2f sec, %d Hz)\n",
           output_path.c_str(), (int)result.pcm.size(),
           (float)result.pcm.size() / result.sample_rate, result.sample_rate);

    if (!write_wav(output_path, result.pcm, result.sample_rate)) {
        fprintf(stderr, "Failed to write WAV file\n");
        return 1;
    }

    printf("Done.\n");
    return 0;
}
