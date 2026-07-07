# Qwen3-TTS ncnn 移植实现笔记

## 日期: 2026-07-07

## 1. 背景

Qwen3-TTS 是阿里通义千问团队的开源 TTS 模型，采用 **Talker + CodePredictor + SpeakerEncoder** 三模块架构，辅以独立的 **SpeechTokenizer V2 (12Hz)** 进行音频编解码。

本文档记录将 Qwen3-TTS-12Hz-0.6B-Base 适配到 ncnn 推理框架的完整实现过程。

## 2. 模型架构

### 2.1 整体推理流程

```
输入文本 + 参考音频
    │
    ├── SpeakerEncoder (ECAPA-TDNN) ──→ speaker_embedding [1024]
    │
    ├── Text Embedding + Codec Embedding (prompt 组装)
    │         │
    ▼         ▼
  Talker (20 层 Transformer, hidden=1024)
    │   ├── codec_head → 主 codebook logits (vocab=3072)
    │   └── 每步: 主 codebook token → CodePredictor → 31 个子 codebook token
    │
    ▼
  32 个 codebook 的离散 token 矩阵 [T, 32]
    │
    ▼
  SpeechTokenizer V2 Decoder → waveform (24kHz)
```

### 2.2 核心参数

| 组件 | 参数 | 值 |
|------|------|----|
| Talker | hidden_size | 1024 |
| Talker | num_hidden_layers | 20 |
| Talker | num_attention_heads | 16 |
| Talker | num_key_value_heads | 2 (GQA group=8) |
| Talker | head_dim | 128 |
| Talker | codec_head vocab_size | 3072 |
| CodePredictor | num_hidden_layers | 5 |
| CodePredictor | num_attention_heads | 16 |
| CodePredictor | num_key_value_heads | 8 |
| CodePredictor | vocab_size (子 codebook) | 2048 |
| CodePredictor | lm_heads 数量 | 31 |
| SpeakerEncoder | enc_dim | 1024 |
| SpeakerEncoder | mel_dim | 128 |
| 音频 | num_code_groups | 32 (1 主 + 31 子) |
| 音频 | sample_rate | 24000 |
| 音频 | frame_rate | 12Hz |

### 2.3 Talker 每步生成流程

1. Talker decoder 前向 → codec_head → 主 codebook token (vocab=3072)
2. Code predictor 31 步自回归 → 31 个子 codebook token (vocab=2048)
3. 32 个 codebook embedding **求和** → 作为下一步 talker 输入

关键点：每步生成时，32 个 codebook 的 embedding 求和后作为 Talker 的输入，这是一种多码本融合策略。

### 2.4 Code Predictor 自回归机制

Code predictor 有 31 个独立的 lm_head，按 generation_step 顺序使用：
- step 0 → lm_head[0] → 第 1 个子 codebook token
- step 1 → lm_head[1] → 第 2 个子 codebook token
- ...
- step 30 → lm_head[30] → 第 31 个子 codebook token

Prefill 输入: `cat(talker_past_hidden, main_codebook_embed)` → `[1, 2, 1024]`

## 3. ncnn 子图拆分策略

### 3.1 设计原则

保留标准三段图（embed + decoder + lm_head）作为 Talker 的核心推理图，在此之外新增独立子图：

| 子图 | 文件 | 说明 |
|------|------|------|
| Talker Embedding | embed.ncnn | text/codec token → embedding |
| Talker Decoder | decoder.ncnn | 20 层 Transformer（带 KV cache） |
| Talker LM Head | lm_head.ncnn | hidden → 主 codebook logits (3072) |
| Text Projection | text_projection.ncnn | text_hidden(2048) → hidden(1024) MLP |
| Speaker Encoder | speaker_encoder.ncnn | ECAPA-TDNN: mel → speaker embedding (1024) |
| CP Decoder | cp_decoder.ncnn | 5 层 Transformer（带 KV cache） |
| CP LM Heads | cp_lm_heads.ncnn | 31 个 merged lm_heads → 子 codebook logits (2048) |
| CP Codec Embeds | cp_codec_embeds.ncnn | 31 个 merged codec embeddings |
| Tokenizer Decoder | tokenizer_decoder.ncnn | codes [32, T] → PCM |

### 3.2 C++ Runtime 负责的部分

| 功能 | 原因 |
|------|------|
| BPE tokenizer | 与原版 prompt 构造强相关 |
| WAV 读取/写出 | runtime 基础能力 |
| log-mel 计算 | 避免 STFT 混入 ncnn 图 |
| Prompt 组装 | 动态序列拼接、条件分支 |
| 生成循环调度 | 自回归控制流 |
| 多码本 embedding 求和 | 32 个 codebook 逐个查表后累加 |
| 采样 (top-k/top-p/temperature) | 动态概率操作 |

### 3.3 与三段图的关系

三段图（embed + decoder + lm_head）是 Talker 的正确设计，不需要拆分。新增的子图（speaker encoder、code predictor、text projection）是三段图之外的独立组件，不影响现有架构。

参考别人的实现，其子图拆分本质上也是给 Qwen3-TTS 的多组件系统各自独立导出 ncnn graph，核心 LLM 仍然是标准三段图。

## 4. 代码结构

### 4.1 导出脚本 (export/qwen3_tts_export.py)

新增 wrapper 类：
- `SpeakerEncoderTS` — ECAPA-TDNN 导出包装
- `TextProjectionTS` — ResizeMLP 导出包装
- `CodePredictorDecoderTS` — 5 层 Transformer + KV cache 导出包装
- `CodePredictorLmHeadsTS` — 31 个 lm_head 合并为单个权重张量
- `CodePredictorCodecEmbedsTS` — 31 个 codec embedding 合并

导出流程：
1. 导出 embed / lm_head / text_projection / speaker_encoder / cp_decoder / cp_lm_heads / cp_codec_embeds / decoder
2. 导出 tokenizer_decoder
3. 生成 model.json（包含所有子图路径和配置）

### 4.2 C++ Runtime (src/ncnn_llm_tts.h / .cpp)

新增成员变量：
- `text_projection_net_` — text_hidden → hidden MLP
- `speaker_encoder_net_` — ECAPA-TDNN
- `cp_decoder_net_` — code predictor 5 层 Transformer
- `cp_lm_heads_net_` — 31 merged lm_heads
- `cp_codec_embeds_net_` — 31 merged codec embeddings

新增方法：
- `run_code_predictor()` — 31 步自回归生成子 codebook
- `run_speaker_encoder()` — 参考音频 → speaker embedding
- `compute_mel_spectrogram()` — C++ 实现 log-mel（n_fft=1024, hop=256, num_mels=128）
- `read_wav()` — WAV 文件读取
- `extract_cp_logits()` — 从 merged 输出中提取指定 step 的 logits
- `extract_cp_embedding()` — 从 merged 输出中提取指定 step 的 embedding

修正的核心逻辑：
- `generate_codec_tokens()` — 重写为 talker + code predictor 联合生成
- `num_codebooks_` 默认值从 8 改为 32
- `codec_vocab_size_` 默认值从 32768 改为 3072
- `attn_cnt_` 默认值从 32 改为 20

### 4.3 上下文扩展 (src/ncnn_llm_gpt.h)

`ncnn_llm_gpt_ctx` 新增 `past_hidden` 字段，用于在 talker 和 code predictor 之间传递 hidden state。

### 4.4 model.json 结构

```json
{
  "params": {
    "embed_token_param": "embed.ncnn.param",
    "decoder_param": "decoder.ncnn.param",
    "lm_head_param": "lm_head.ncnn.param",
    "text_projection_param": "text_projection.ncnn.param",
    "speaker_encoder_param": "speaker_encoder.ncnn.param",
    "cp_decoder_param": "cp_decoder.ncnn.param",
    "cp_lm_heads_param": "cp_lm_heads.ncnn.param",
    "cp_codec_embeds_param": "cp_codec_embeds.ncnn.param",
    "tokenizer_decoder_param": "tokenizer_decoder.ncnn.param"
  },
  "setting": {
    "attn_cnt": 20,
    "cp_attn_cnt": 5,
    "cp_num_kv_heads": 8,
    "cp_num_heads": 16,
    "cp_head_dim": 128,
    "audio": {
      "num_codebooks": 32,
      "codec_vocab_size": 3072,
      "cp_vocab_size": 2048,
      "sample_rate": 24000,
      "frame_rate": 12
    }
  }
}
```

## 5. Voice Clone 流程

```
1. 读取参考音频 WAV → float 波形
2. 计算 log-mel spectrogram (n_fft=1024, hop=256, num_mels=128, fmax=12000)
3. mel → speaker_encoder.ncnn → speaker_embedding [1024]
4. 组装 prompt: text_embed + speaker_embed + codec_prompt
5. Talker prefill → 首 token
6. 逐步生成: talker → code predictor → 32 码本
7. tokenizer_decoder → PCM → WAV
```

CLI 用法：
```bash
xmake run tts_main --model assets/qwen3_tts \
    --text "你好世界" \
    --ref-audio ref.wav \
    --ref-text "这是参考音频的文本" \
    --output output.wav
```

## 6. 待完成事项

### P0 — 立即需要
- [ ] 实际导出测试：运行 `python export/qwen3_tts_export.py` 验证所有子图导出
- [ ] Tokenizer Encoder 导出：参考音频 → ref_code（当前缺少，需要 C++ RVQ 或导出 encoder latent）
- [ ] Prompt 组装完善：speaker embedding 注入、ref_code ICL 拼接

### P1 — 工程化
- [ ] Parity 验证框架：PyTorch baseline → ncnn → 逐级 compare
- [ ] Profile/Bucket 系统：按输入长度选择静态窗口 graph
- [ ] DFT → FFT 优化：mel 计算中的 O(n^2) DFT 替换为 FFTW/KISSFFT

### P2 — 后续
- [ ] Tokenizer Decoder 拆分导出（如果整图 trace 失败）
- [ ] Flow-matching 模式适配（CosyVoice 等）
- [ ] 流式生成支持
