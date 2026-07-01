# Qwen3-TTS 架构研究笔记

## 研究时间: 2026-07-01

## 1. 模型总览

Qwen3-TTS 是阿里通义千问团队的开源 TTS 模型，核心特点：

- **离散多码本 LM 架构**（非 DiT、非 flow-matching）
- 使用自研 **Qwen3-TTS-Tokenizer-12Hz** 作为音频编解码器
- 支持 10 种语言（中英日韩德法俄葡西意）
- 三种模型类型：Base（语音克隆）、CustomVoice（自定义音色）、VoiceDesign（声音设计）

## 2. 架构拆解

### 2.1 整体推理流程

```
文本 → Qwen3 LLM (多码本自回归) → 离散音频 codes (T, Q)
                                     ↓
                     Qwen3-TTS-Tokenizer-12Hz (解码器)
                                     ↓
                                   PCM 音频
```

### 2.2 Tokenizer-12Hz (Qwen3TTSTokenizerV2Model)

从源码 `qwen_tts/inference/qwen3_tts_tokenizer.py` 分析：

- **编码**: 音频波形 → `audio_codes`，shape 为 `(codes_len, num_quantizers)` — **多码本！**
- **解码**: `audio_codes` → PCM 波形
- 12Hz 帧率（每秒 12 个 codec frame）
- 与 25Hz tokenizer 的区别：
  - 25Hz: codes shape `(codes_len,)` 单码本，额外需要 `xvectors` 和 `ref_mels`
  - 12Hz: codes shape `(codes_len, num_quantizers)` 多码本，解码只需 codes

### 2.3 LLM 模型 (Qwen3TTSForConditionalGeneration)

从源码 `qwen_tts/inference/qwen3_tts_model.py` 分析：

- 继承自 transformers 的模型体系，通过 `AutoModel` 加载
- 内部包含 `speech_tokenizer`（Tokenizer-12Hz 实例）
- `generate()` 方法返回 `talker_codes_list`（多码本 token 列表）
- 支持 `non_streaming_mode` 参数（Dual-Track 流式架构）

### 2.4 三种生成模式

#### CustomVoice (自定义音色)
```python
model.generate(
    input_ids=input_ids,          # assistant text tokens
    instruct_ids=instruct_ids,    # optional instruct text tokens
    languages=languages,
    speakers=speakers,            # 预设说话人名
    non_streaming_mode=True,
)
```

#### VoiceDesign (声音设计)
```python
model.generate(
    input_ids=input_ids,
    instruct_ids=instruct_ids,    # 自然语言音色描述
    languages=languages,
    non_streaming_mode=True,
)
```

#### Voice Clone (语音克隆, Base 模型)
```python
model.generate(
    input_ids=input_ids,
    ref_ids=ref_ids,              # 参考文本 tokens
    voice_clone_prompt={
        "ref_code": [...],        # 参考音频 codes
        "ref_spk_embedding": [...],  # 说话人嵌入
        "x_vector_only_mode": [...],
        "icl_mode": [...],
    },
    languages=languages,
    non_streaming_mode=False,
)
```

### 2.5 Prompt 格式

从 `qwen3_tts_model.py` 的 `_build_*` 方法：

```python
# Assistant text (要合成的文本)
f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

# Ref text (语音克隆的参考文本)
f"<|im_start|>assistant\n{text}<|im_end|>\n"

# Instruct text (指令文本)
f"<|im_start|>user\n{instruct}<|im_end|>\n"
```

使用标准 ChatML 格式（`<|im_start|>` / `<|im_end|>`）。

### 2.6 生成参数默认值

```python
hard_defaults = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    # subtalker 参数 (仅 tokenizer-v2)
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
}
```

### 2.7 解码后处理

```python
# Voice Clone 模式：拼接参考 codes + 生成 codes，解码后裁剪参考段
codes_for_decode = torch.cat([ref_code, generated_codes], dim=0)
wavs, fs = speech_tokenizer.decode(codes_for_decode)
# 裁剪参考音频长度
cut = int(ref_len / total_len * wav.shape[0])
wav_out = wav[cut:]
```

## 3. 与当前 ncnn_llm 实现的差距

| 维度 | Qwen3-TTS 实际 | 当前实现 | 需要修改 |
|---|---|---|---|
| 音频编解码器 | Tokenizer-12Hz (多码本, shape `(T,Q)`) | 假设 SNAC/EnCodec (简单交错) | **重写** |
| 多码本 token 排列 | `(T, Q)` 矩阵，T=时间步, Q=量化层数 | 简单交错拆分 `flat[i%Q]` | **修正** |
| Flow-matching | **不需要** | 已实现但 Qwen3-TTS 用不到 | 保留但非默认 |
| Prompt 格式 | `<\|im_start\|>assistant\n{text}<\|im_end\|>\n<\|im_start\|>assistant\n` | 通用 ChatML | **适配** |
| 语音克隆 | ref_code + spk_embedding + ICL | 仅预留字段 | **需要实现** |
| 流式生成 | Dual-Track hybrid streaming | 全量生成 | 后续考虑 |
| 导出目标 | Qwen/Qwen3-TTS-12Hz-* + Tokenizer-12Hz | Qwen/Qwen2.5-TTS | **更新** |

## 4. 导出策略

需要导出 **两个独立模型**：

### 4.1 LLM 模型 (Qwen3-TTS-12Hz-0.6B/1.7B)
- `embed.ncnn.param/bin` — token embedding
- `decoder.ncnn.param/bin` — Transformer decoder (带 KV cache)
- `lm_head.ncnn.param/bin` — 输出投影

### 4.2 Tokenizer-12Hz 模型 (Qwen3-TTS-Tokenizer-12Hz)
- `tokenizer_decoder.ncnn.param/bin` — codes → PCM 解码器
- 这是独立模型，需要单独用 PNNX 导出

## 5. 关键待确认问题

1. Tokenizer-12Hz 的具体网络结构（需要读 `core/tokenizer_12hz/` 源码）
2. 多码本 token 的具体排列方式（是 `(T, Q)` 还是交错？）
3. LLM 输出的是 logits 还是直接 token id？多码本如何从单个 vocab 映射？
4. Qwen3-TTS 的 LLM 是否基于 Qwen3 架构（RoPE、attention pattern 等）？
