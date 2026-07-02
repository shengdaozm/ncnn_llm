# 代码修改笔记: Qwen3-TTS 适配

## 修改时间: 2026-07-02 (KV cache 支持)

## 0. KV cache 导出修复 (2026-07-02)

### 问题

旧版 `DecoderTS` 传 `past_key_values=None, use_cache=False` 绕过 `DynamicCache`，
导致导出的 ncnn decoder 网络没有 KV cache 输入/输出，C++ 端无法进行自回归推理。

### 根因

Qwen2/3 的 `DecoderLayer.forward()` 使用 `DynamicCache`（mutable 对象）管理 KV cache：
- `past_key_values.update(key_states, value_states)` 内部拼接并返回完整 KV
- `torch.jit.trace` 无法序列化非 tensor 对象
- layer 的返回值只有 `hidden_states`（一个 tensor），不包含 cache

### 解决方案

不再调用 `DecoderLayer.forward()`，改为直接调用子模块并手动实现完整的 attention + KV cache：

1. **`DecoderLayerTS`**: 直接使用 `input_layernorm` / `q_proj` / `k_proj` / `v_proj` / `o_proj` /
   `post_attention_layernorm` / `mlp`，手动完成：
   - QKV 投影 + reshape
   - RoPE (`_rotate_half` + `_apply_rotary_pos_emb`，自包含实现)
   - `torch.cat([cache_k, key_states])` 拼接 KV cache
   - `_repeat_kv` GQA 展开
   - Eager attention (matmul + softmax + matmul)
   - 残差连接 + MLP

2. **`DecoderTS`**: 管理 28 层 KV cache 传递，接口对齐 C++ 端 `llm_run_decoder_with_kv`：
   - Inputs: `in0`=embeds, `in1`=mask, `in2`=cos, `in3`=sin, `cache_k{i}`, `cache_v{i}`
   - Outputs: `out0`=hidden, `out_cache_k{i}`, `out_cache_v{i}`

3. **导出时**使用非空 cache tensor（`past_len=4`）trace，确保 `torch.cat` 分支被固化。
   C++ 端 prefill 时传入空 `ncnn::Mat`，Concat 层自动处理零维度拼接。

4. **删除** `export/qwen_tts_export.py`（Qwen2.5-TTS 导出脚本，已废弃）

5. **`src/ncnn_text_runtime.cpp`**: prefill 时也传入 cache blob（空 `ncnn::Mat`），
   因为新网络中 cache 是 `torch.cat` 的操作数，不输入会导致 ncnn 报错。

6. **`tests/test_export_logic.py`**: 更新 `FakeDecoderLayer` 结构对齐新 wrapper，
   新增 `test_decoder_layer_ts` 测试。

---

## 修改时间: 2026-07-01 (初始适配)

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
