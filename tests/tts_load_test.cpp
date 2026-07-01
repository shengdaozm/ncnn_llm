/**
 * @file tts_load_test.cpp
 * @brief TTS 模型加载测试 — 验证 ncnn 网络和分词器可正常初始化，不执行推理
 *
 * 用法:
 *   tts_load_test --model <model_dir>
 *
 * 退出码:
 *   0 = 所有组件加载成功
 *   1 = 加载失败
 */

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <net.h>

using nlohmann::json;

int main(int argc, char** argv) {
    std::string model_dir = "assets/qwen3_tts";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else {
            fprintf(stderr, "Usage: %s --model <model_dir>\n", argv[0]);
            return 1;
        }
    }

    if (!std::filesystem::exists(model_dir)) {
        fprintf(stderr, "Model directory does not exist: %s\n", model_dir.c_str());
        return 1;
    }

    // 1. 读取 model.json
    std::string model_json_path = model_dir + "/model.json";
    std::ifstream ifs(model_json_path);
    if (!ifs.is_open()) {
        fprintf(stderr, "FAIL: Cannot open model.json: %s\n", model_json_path.c_str());
        return 1;
    }

    json config;
    ifs >> config;

    printf("=== TTS Model Load Test ===\n");
    printf("Model directory: %s\n\n", model_dir.c_str());

    int total_checks = 0;
    int passed_checks = 0;

    auto check = [&](const char* name, bool ok) {
        total_checks++;
        if (ok) {
            passed_checks++;
            printf("  [PASS] %s\n", name);
        } else {
            printf("  [FAIL] %s\n", name);
        }
    };

    // 2. 检查 param/bin 文件是否存在
    auto params = config["params"];
    auto check_file = [&](const std::string& key) -> bool {
        if (!params.contains(key)) return false;
        std::string fname = model_dir + "/" + params[key].get<std::string>();
        return std::filesystem::exists(fname);
    };

    check("embed_token_param exists", check_file("embed_token_param"));
    check("embed_token_bin exists", check_file("embed_token_bin"));
    check("decoder_param exists", check_file("decoder_param"));
    check("decoder_bin exists", check_file("decoder_bin"));
    check("lm_head_param exists", check_file("lm_head_param"));
    check("lm_head_bin exists", check_file("lm_head_bin"));

    // Tokenizer decoder (optional)
    bool has_tok_dec = params.contains("tokenizer_decoder_param");
    if (has_tok_dec) {
        check("tokenizer_decoder_param exists", check_file("tokenizer_decoder_param"));
        check("tokenizer_decoder_bin exists", check_file("tokenizer_decoder_bin"));
    } else {
        printf("  [SKIP] tokenizer_decoder not configured (optional)\n");
    }

    // 3. 尝试加载 ncnn 网络
    printf("\nLoading ncnn networks...\n");

    // embed
    {
        ncnn::Net net;
        std::string p = model_dir + "/" + params["embed_token_param"].get<std::string>();
        std::string b = model_dir + "/" + params["embed_token_bin"].get<std::string>();
        int r1 = net.load_param(p.c_str());
        int r2 = net.load_model(b.c_str());
        check("embed_net load", r1 == 0 && r2 == 0);
    }

    // decoder
    {
        ncnn::Net net;
        std::string p = model_dir + "/" + params["decoder_param"].get<std::string>();
        std::string b = model_dir + "/" + params["decoder_bin"].get<std::string>();
        int r1 = net.load_param(p.c_str());
        int r2 = net.load_model(b.c_str());
        check("decoder_net load", r1 == 0 && r2 == 0);
    }

    // lm_head
    {
        ncnn::Net net;
        std::string p = model_dir + "/" + params["lm_head_param"].get<std::string>();
        std::string b = model_dir + "/" + params["lm_head_bin"].get<std::string>();
        int r1 = net.load_param(p.c_str());
        int r2 = net.load_model(b.c_str());
        check("lm_head_net load", r1 == 0 && r2 == 0);
    }

    // tokenizer decoder (if present)
    if (has_tok_dec) {
        ncnn::Net net;
        std::string p = model_dir + "/" + params["tokenizer_decoder_param"].get<std::string>();
        std::string b = model_dir + "/" + params["tokenizer_decoder_bin"].get<std::string>();
        int r1 = net.load_param(p.c_str());
        int r2 = net.load_model(b.c_str());
        check("tokenizer_decoder_net load", r1 == 0 && r2 == 0);
    }

    // 4. 检查分词器文件
    printf("\nChecking tokenizer files...\n");
    auto tok = config["tokenizer"];
    if (tok.contains("vocab_file")) {
        std::string vf = model_dir + "/" + tok["vocab_file"].get<std::string>();
        check("vocab.txt exists", std::filesystem::exists(vf));
    }
    if (tok.contains("merges_file")) {
        std::string mf = model_dir + "/" + tok["merges_file"].get<std::string>();
        check("merges.txt exists", std::filesystem::exists(mf));
    }

    // 5. 输出配置摘要
    printf("\n=== Configuration Summary ===\n");
    if (config["setting"].contains("attn_cnt"))
        printf("  attn_cnt:        %d\n", config["setting"]["attn_cnt"].get<int>());
    if (config["setting"].contains("tts_mode"))
        printf("  tts_mode:        %s\n", config["setting"]["tts_mode"].get<std::string>().c_str());
    if (config["setting"].contains("tts_model_type"))
        printf("  tts_model_type:  %s\n", config["setting"]["tts_model_type"].get<std::string>().c_str());
    if (config["setting"].contains("rope")) {
        auto rope = config["setting"]["rope"];
        printf("  rope_type:       %s\n", rope.value("type", "RoPE").c_str());
        printf("  rope_head_dim:   %d\n", rope.value("rope_head_dim", 128));
        printf("  rope_theta:      %.1f\n", rope.value("rope_theta", 1000000.0f));
    }
    if (config["setting"].contains("audio")) {
        auto audio = config["setting"]["audio"];
        printf("  num_codebooks:   %d\n", audio.value("num_codebooks", 8));
        printf("  codec_vocab_size: %d\n", audio.value("codec_vocab_size", 32768));
        printf("  sample_rate:     %d\n", audio.value("sample_rate", 24000));
        printf("  frame_rate:      %d\n", audio.value("frame_rate", 12));
    }

    // 6. 结果
    printf("\n=== Result: %d/%d checks passed ===\n", passed_checks, total_checks);
    if (passed_checks == total_checks) {
        printf("ALL CHECKS PASSED\n");
        return 0;
    } else {
        printf("SOME CHECKS FAILED\n");
        return 1;
    }
}
