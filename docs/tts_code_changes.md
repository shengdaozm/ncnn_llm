# 代码修改笔记: Qwen3-TTS 适配

## 修改时间: 2026-07-01

## 1. 导出脚本 (`export/qwen3_tts_export.py`)

**完全重写**，从 Qwen2.5-TTS 改为 Qwen3-TTS。

### 关键变更
- 目标模型改为 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- 支持分别导出 LLM 和 Tokenizer-12Hz 两个独立模型
- 导出脚本支持 `--skip-tokenizer` 跳过 tokenizer 导出
- 支持 `--tts_model_type` 指定模型类型 (base/custom_voice/voice_design)
- `model.json` 中新增 `frame_rate`, `tts_model_type` 字段
- decoder 导出去掉了 attention_mask 输入（由 ncnn 内部处理 causal mask）

### 文件命名变更
| 旧名称 | 新名称 | 说明 |
|---|---|---|
| `codec.ncnn.param/bin` | `tokenizer_decoder.ncnn.param/bin` | Qwen3-TTS-Tokenizer-12Hz 解码器 |

## 2. C++ 推理代码修改

### 2.1 `src/ncnn_llm_tts.h`

- `TtsConfig` 默认参数对齐 Qwen3-TTS `generate_config.json`:
  - `max_new_tokens`: 4096 → 2048
  - `temperature`: 0.3 → 0.9
  - `top_p`: 0.8 → 1.0
  - `repetition_penalty`: 1.1 → 1.05
- 新增 `language`, `speaker`, `instruct`, `non_streaming_mode` 字段
- `num_codebooks_`: 4 → 8 (Qwen3-TTS-12Hz 默认 8 层量化)
- `codec_vocab_size_`: 4096 → 32768
- 新增 `frame_rate_` (12Hz), `tts_model_type_` 成员

### 2.2 `src/ncnn_llm_tts.cpp`

#### Prompt 格式修正
旧:
```cpp
apply_chat_template(messages, {}, true, false)
```
新 (对齐 Qwen3-TTS 源码):
```cpp
// CustomVoice/VoiceDesign: instruct 作为 user 消息
if (!cfg.instruct.empty()) {
    full_prompt += "<|im_start|>user\n" + cfg.instruct + "<|im_end|>\n";
}
// 目标文本作为 assistant 消息
full_prompt += "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
```

#### 多码本 token 拆分修正
旧（简单交错，但 `row(cb)` 取法不对）:
```cpp
for (size_t i = cb; i < flat_tokens.size(); i += num_codebooks_) {
    codebook_tokens[cb].push_back(flat_tokens[i]);
}
```
新（按帧分组，每 `num_codebooks_` 个 token 为一帧）:
```cpp
int num_frames = flat_tokens.size() / num_codebooks_;
for (int cb = 0; cb < num_codebooks_; ++cb) {
    for (int t = 0; t < num_frames; ++t) {
        codebook_tokens[cb].push_back(flat_tokens[t * num_codebooks_ + cb]);
    }
}
```

#### Codec 加载逻辑
支持两种配置字段:
- `tokenizer_decoder_param/bin` (Qwen3-TTS-Tokenizer-12Hz, 新)
- `codec_param/bin` (旧版 SNAC/EnCodec, 向后兼容)

### 2.3 `src/utils/audio/audio_codec.h`

- `CodecConfig` 默认值更新:
  - `num_codebooks`: 4 → 8
  - `vocab_size`: 4096 → 32768
  - 新增 `frame_rate` (12Hz)
  - `hop_length`: 480 → 2000 (24000/12)
  - 移除 `codebook_rates`（Qwen3-TTS 所有 codebook 帧率相同）

### 2.4 `src/utils/audio/audio_codec.cpp`

#### `tokens_to_mat` 矩阵布局修正
旧: `[max_frames, num_codebooks]`，按 codebook 行排列
新: `(T, Q)` 矩阵，`mat.row(t) = [codebook_0[t], ..., codebook_Q-1[t]]`

```cpp
// ncnn::Mat: w = num_codebooks (Q), h = max_frames (T)
ncnn::Mat mat(num_codebooks, max_frames);
for (int cb = 0; cb < num_codebooks; ++cb) {
    for (int t = 0; t < codec_tokens[cb].size(); ++t) {
        mat.row(t)[cb] = codec_tokens[cb][t];
    }
}
```

#### 配置解析
新增 `frame_rate` 解析，自动计算 `hop_length = sample_rate / frame_rate`

### 2.5 `examples/tts_main.cpp`

- 默认参数对齐 Qwen3-TTS
- 新增 `--language`, `--speaker`, `--instruct`, `--streaming-mode` 参数

## 3. 已知限制

1. **Tokenizer-12Hz 导出未验证**: Qwen3-TTS-Tokenizer-12Hz 的具体网络结构需要进一步研究 `core/tokenizer_12hz/` 源码
2. **语音克隆未实现**: `reference_text` / `reference_audio_path` 仍为预留字段
3. **流式生成未实现**: `non_streaming_mode` 参数已支持但实际推理仍为全量模式
4. **精度验证未完成**: 需要实际模型权重进行 PyTorch vs ncnn 对比
