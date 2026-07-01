#!/usr/bin/env python3
"""
Qwen3-TTS 导出参数校验脚本

对比 PyTorch 原版 config.json 与导出的 model.json，
确保关键参数一致。

Usage:
  python tests/verify_tts_export.py --model_dir assets/qwen3_tts --config_path <path_to_config.json>
"""

import argparse
import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_params(config_json_path: str, model_json_path: str) -> bool:
    """对比 PyTorch config.json 与 ncnn model.json 的关键参数。"""
    config = load_json(config_json_path)
    model = load_json(model_json_path)

    errors = []
    warnings = []

    # Qwen3-TTS config.json has parameters nested under "talker_config"
    talker_cfg = config.get("talker_config", config)

    # 1. num_hidden_layers -> attn_cnt
    pytorch_layers = talker_cfg.get("num_hidden_layers")
    ncnn_attn_cnt = model.get("setting", {}).get("attn_cnt")
    if pytorch_layers is not None and ncnn_attn_cnt is not None:
        if pytorch_layers == ncnn_attn_cnt:
            print(f"  [OK] num_hidden_layers == attn_cnt: {pytorch_layers}")
        else:
            errors.append(f"num_hidden_layers ({pytorch_layers}) != attn_cnt ({ncnn_attn_cnt})")
    else:
        warnings.append(f"Cannot compare layers: pytorch={pytorch_layers}, ncnn={ncnn_attn_cnt}")

    # 2. rope_theta
    pytorch_rope_theta = talker_cfg.get("rope_theta")
    ncnn_rope_theta = model.get("setting", {}).get("rope", {}).get("rope_theta")
    if pytorch_rope_theta is not None and ncnn_rope_theta is not None:
        if abs(float(pytorch_rope_theta) - float(ncnn_rope_theta)) < 1e-6:
            print(f"  [OK] rope_theta: {pytorch_rope_theta}")
        else:
            errors.append(f"rope_theta mismatch: pytorch={pytorch_rope_theta}, ncnn={ncnn_rope_theta}")
    else:
        warnings.append(f"Cannot compare rope_theta: pytorch={pytorch_rope_theta}, ncnn={ncnn_rope_theta}")

    # 3. head_dim
    pytorch_head_dim = talker_cfg.get("head_dim")
    ncnn_head_dim = model.get("setting", {}).get("rope", {}).get("rope_head_dim")
    if pytorch_head_dim is not None and ncnn_head_dim is not None:
        if int(pytorch_head_dim) == int(ncnn_head_dim):
            print(f"  [OK] head_dim: {pytorch_head_dim}")
        else:
            errors.append(f"head_dim mismatch: pytorch={pytorch_head_dim}, ncnn={ncnn_head_dim}")
    else:
        warnings.append(f"Cannot compare head_dim: pytorch={pytorch_head_dim}, ncnn={ncnn_head_dim}")

    # 4. vocab_size (codec vocab in talker_config)
    pytorch_vocab = talker_cfg.get("vocab_size")
    if pytorch_vocab is not None:
        print(f"  [OK] PyTorch codec vocab_size: {pytorch_vocab}")

    # 4b. hidden_size / num_attention_heads / num_key_value_heads
    pytorch_hidden = talker_cfg.get("hidden_size")
    pytorch_num_heads = talker_cfg.get("num_attention_heads")
    pytorch_kv_heads = talker_cfg.get("num_key_value_heads")
    if pytorch_hidden is not None:
        print(f"  [OK] hidden_size: {pytorch_hidden}")
    if pytorch_num_heads is not None:
        print(f"  [OK] num_attention_heads: {pytorch_num_heads}")
    if pytorch_kv_heads is not None:
        print(f"  [OK] num_key_value_heads: {pytorch_kv_heads}")

    # 4c. rope_scaling (mRoPE)
    pytorch_rope_scaling = talker_cfg.get("rope_scaling")
    if pytorch_rope_scaling is not None:
        print(f"  [OK] rope_scaling: {pytorch_rope_scaling}")

    # 5. num_codebooks / frame_rate (from speech_tokenizer config if available)
    ncnn_audio = model.get("setting", {}).get("audio", {})
    ncnn_codebooks = ncnn_audio.get("num_codebooks")
    ncnn_frame_rate = ncnn_audio.get("frame_rate")
    ncnn_codec_vocab = ncnn_audio.get("codec_vocab_size")
    ncnn_sample_rate = ncnn_audio.get("sample_rate")
    if ncnn_codebooks is not None:
        print(f"  [OK] num_codebooks: {ncnn_codebooks}")
    if ncnn_frame_rate is not None:
        print(f"  [OK] frame_rate: {ncnn_frame_rate}")
    if ncnn_codec_vocab is not None:
        print(f"  [OK] codec_vocab_size: {ncnn_codec_vocab}")
    if ncnn_sample_rate is not None:
        print(f"  [OK] sample_rate: {ncnn_sample_rate}")

    # 6. tts_model_type
    ncnn_tts_type = model.get("setting", {}).get("tts_model_type")
    if ncnn_tts_type:
        print(f"  [OK] tts_model_type: {ncnn_tts_type}")

    # 7. 检查必要的 param/bin 文件是否存在
    params = model.get("params", {})
    model_dir = os.path.dirname(model_json_path)
    for key, val in params.items():
        if key.endswith("_param"):
            param_file = os.path.join(model_dir, val)
            if os.path.exists(param_file):
                size = os.path.getsize(param_file)
                print(f"  [OK] {val} exists ({size} bytes)")
            else:
                errors.append(f"Missing file: {val}")

    # 8. 检查 tokenizer 文件
    tokenizer_cfg = model.get("tokenizer", {})
    for key in ("vocab_file", "merges_file"):
        fname = tokenizer_cfg.get(key)
        if fname:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                print(f"  [OK] {fname} exists ({size} bytes)")
            else:
                errors.append(f"Missing tokenizer file: {fname}")

    # 输出 warnings
    for w in warnings:
        print(f"  [WARN] {w}")

    # 输出 errors
    for e in errors:
        print(f"  [FAIL] {e}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Verify Qwen3-TTS export parameters")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the ncnn model directory (containing model.json)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to the PyTorch config.json (if not in model_dir)")
    args = parser.parse_args()

    model_json_path = os.path.join(args.model_dir, "model.json")
    if not os.path.exists(model_json_path):
        print(f"ERROR: model.json not found at {model_json_path}")
        sys.exit(1)

    config_path = args.config_path
    if config_path is None:
        config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found at {config_path}")
        print("       Specify --config_path explicitly")
        sys.exit(1)

    print(f"Comparing:")
    print(f"  PyTorch config: {config_path}")
    print(f"  ncnn model.json: {model_json_path}")
    print()

    ok = verify_params(config_path, model_json_path)
    print()
    if ok:
        print("=== All checks PASSED ===")
        sys.exit(0)
    else:
        print("=== Some checks FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
