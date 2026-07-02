# Qwen3-TTS 模型导出完整笔记

## 日期: 2026-07-01
## 状态: 已本地验证通过（Qwen3-TTS-12Hz-0.6B-Base）

---

## 1. 导出概览

### 目标

将 HuggingFace 上的 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 模型转换为 ncnn 格式，供 C++ 运行时推理。

### 导出的 4 个子网络

| 子网络 | 文件 | 大小 | 用途 |
|---|---|---|---|
| embed | `embed.ncnn.param/bin` | 1.2GB | 文本 token → embedding |
| decoder | `decoder.ncnn.param/bin` | 1.6GB | 28 层 Transformer decoder（带 KV cache） |
| lm_head | `lm_head.ncnn.param/bin` | 12MB | hidden state → codec logits（输出投影） |
| tokenizer_decoder | `tokenizer_decoder.ncnn.param/bin` | 435MB | 多码本音频 token → PCM 波形 |

### 导出路径

不同子网络因模型复杂度不同，使用了不同的导出路径：

```
embed:         TorchScript trace → pnnx CLI → ncnn
lm_head:       TorchScript trace → pnnx CLI → ncnn
decoder:       TorchScript trace → pnnx CLI → ncnn
tokenizer_decoder: ONNX (dynamo) → inline weights → pnnx CLI → ncnn
```

---

## 2. 环境准备

### Python 依赖

```bash
pip install torch transformers 'huggingface_hub[cli]' soundfile librosa numpy
pip install qwen-tts          # 注册 qwen3_tts 模型类型
pip install pnnx              # ncnn 转换工具
pip install onnxscript onnx   # ONNX 导出（tokenizer_decoder 需要）
```

### 系统依赖

```bash
# Ubuntu
sudo apt-get install -y sox libsox-dev

# macOS
brew install sox
```

### 模型下载

```bash
# 下载 LLM 模型（含 speech_tokenizer 子目录）
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./qwen3_tts_llm

# 只下载权重文件（更快）
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./qwen3_tts_llm --include "*.safetensors"
```

模型目录结构：
```
qwen3_tts_llm/
├── config.json                    # 顶层配置（含 talker_config, decoder_config 等）
├── model.safetensors              # LLM 权重（1.7GB）
├── generation_config.json
├── merges.txt                     # BPE merges
├── vocab.json                     # 词汇表
├── tokenizer_config.json
├── preprocessor_config.json
└── speech_tokenizer/              # Tokenizer-12Hz 独立模型
    ├── config.json
    ├── model.safetensors          # codec 权重（651MB）
    └── preprocessor_config.json
```

---

## 3. 模型类型注册

### 问题

Qwen3-TTS 使用自定义模型类型 `qwen3_tts`，transformers 默认不识别。直接调用 `AutoConfig.from_pretrained()` 会报错：

```
ValueError: The checkpoint you are trying to load has model type `qwen3_tts`
but Transformers does not recognize this architecture.
```

### 解决方案

`qwen_tts` 包**不会在 import 时自动注册**模型类型。必须显式调用 `AutoConfig.register()` 和 `AutoModel.register()`：

```python
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
```

对于 Tokenizer-12Hz，需要注册两个版本：

```python
from qwen_tts.core import (
    Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model,   # 25Hz
    Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model,   # 12Hz
)
AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)
AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
```

**注意**：类在 `qwen_tts.core.models` 中，不在 `qwen_tts.core` 中（tokenizer 的类在 `qwen_tts.core` 中）。

---

## 4. 模型结构分析

### Qwen3TTSForConditionalGeneration 层级结构

通过阅读源码 `qwen_tts/core/models/modeling_qwen3_tts.py` 确认：

```
Qwen3TTSForConditionalGeneration
├── talker: Qwen3TTSTalkerForConditionalGeneration
│   ├── model: Qwen3TTSTalkerModel
│   │   ├── text_embedding: nn.Embedding(151643, 1024)    ← 文本 token embedding
│   │   ├── codec_embedding: nn.Embedding(3072, 1024)      ← 音频 codec token embedding
│   │   ├── layers: nn.ModuleList[28 × Qwen3TTSTalkerDecoderLayer]
│   │   ├── norm: Qwen3TTSRMSNorm
│   │   └── rotary_emb: Qwen3TTSTalkerRotaryEmbedding      ← 使用 mRoPE
│   ├── codec_head: nn.Linear(1024, 3072, bias=False)      ← 输出投影（非 lm_head!）
│   ├── text_projection: Qwen3TTSTalkerResizeMLP
│   └── code_predictor: Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
├── speaker_encoder: Qwen3TTSSpeakerEncoder (仅 Base 模型)
└── speech_tokenizer: None (运行时单独加载)
```

### 关键发现

1. **embed_tokens 不存在**：`Qwen3TTSForConditionalGeneration` 没有标准 HF 模型的 `embed_tokens` 属性。文本 embedding 在 `model.talker.model.text_embedding`。

2. **lm_head 不存在**：实际输出投影是 `model.talker.codec_head`，不是 `lm_head`。虽然代码里声明了 `_tied_weights_keys = ["lm_head.weight"]`，但 `__init__` 中并未创建 `lm_head`。

3. **vocab_size 是 3072**：这是 codec 词表大小，不是文本词表（文本词表 151643 在 tokenizer 中）。

4. **RoPE 使用 mRoPE**：`rope_scaling` 包含 `mrope_section: [24, 20, 20]` 和 `interleaved: True`。

### 配置参数（来自 config.json）

```
talker_config:
  hidden_size: 1024
  num_hidden_layers: 28
  num_attention_heads: 16
  num_key_value_heads: 8
  head_dim: 128
  rope_theta: 1000000
  vocab_size: 3072 (codec)
  rope_scaling:
    interleaved: true
    mrope_section: [24, 20, 20]
    rope_type: default
```

---

## 5. 各子网络导出详情

### 5.1 Embed (token embedding)

**Wrapper:**
```python
class EmbedTS(nn.Module):
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed
    def forward(self, input_ids):
        return self.embed(input_ids)
```

**源模块:** `model.talker.model.text_embedding` (nn.Embedding(151643, 1024))

**Example input:** `torch.tensor([[1,2,3,4,5]], dtype=torch.long)` shape `[1,5]`

**导出路径:** TorchScript trace → pnnx CLI

**pnnx 命令:**
```bash
pnnx embed.pt inputshape=[1,5]i64 ncnnparam=embed.ncnn.param ncnnbin=embed.ncnn.bin fp16=0 optlevel=2 device=cpu
```

**结果:** 成功，165B param + 1.2GB bin

---

### 5.2 LM Head (输出投影)

**Wrapper:**
```python
class LmHeadTS(nn.Module):
    def __init__(self, lm_head: nn.Linear):
        super().__init__()
        self.lm_head = lm_head
    def forward(self, hidden_states):
        return self.lm_head(hidden_states)
```

**源模块:** `model.talker.codec_head` (nn.Linear(1024, 3072, bias=False))

**注意:** 是 `codec_head` 不是 `lm_head`！

**Example input:** `torch.randn(1, 1, 1024)` shape `[1,1,1024]`

**导出路径:** TorchScript trace → pnnx CLI

**pnnx 命令:**
```bash
pnnx lm_head.pt inputshape=[1,1,1024]f32 ncnnparam=lm_head.ncnn.param ncnnbin=lm_head.ncnn.bin fp16=0 optlevel=2 device=cpu
```

**结果:** 成功，177B param + 12MB bin

---

### 5.3 Decoder (Transformer)

#### 问题

直接用 `torch.jit.trace` 跟踪整个 `Qwen3TTSTalkerModel.forward()` 会失败：
1. `DynamicCache` 对象无法被 trace（`Only tensors and tuples of tensors are supported`）
2. 模型内部有条件分支（`if position_ids.ndim == 3`），trace 时会报 `unordered_map::at: key not found`
3. `DecoderLayer.forward()` 只返回 `hidden_states`，KV cache 通过 `DynamicCache` 内部管理，无法通过 `out[1]` 获取

#### 解决方案

完全绕过 `DecoderLayer.forward()`，直接调用子模块，手动实现 attention + KV cache：

```python
class DecoderLayerTS(nn.Module):
    def __init__(self, layer, num_kv_heads, num_heads, head_dim):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.q_proj = layer.self_attn.q_proj
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj
        self.o_proj = layer.self_attn.o_proj
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        # ...

    def forward(self, hidden_states, attention_mask, cos_cache, sin_cache, cache_k, cache_v):
        # 手动 QKV 投影 + RoPE + KV cache concat + eager attention
        ...
        key_states = torch.cat([cache_k, key_states], dim=2)   # KV cache 拼接
        value_states = torch.cat([cache_v, value_states], dim=2)
        new_k = key_states
        new_v = value_states
        ...
        return hidden_states, new_k, new_v
```

KV cache 以显式 tensor 输入/输出，对齐 C++ 端 `llm_run_decoder_with_kv` 接口：
- Inputs: `in0`=embeds, `in1`=mask, `in2`=cos, `in3`=sin, `cache_k{i}`, `cache_v{i}`
- Outputs: `out0`=hidden, `out_cache_k{i}`, `out_cache_v{i}`

导出时使用非空 cache（`past_len=4`）trace，确保 `torch.cat` 分支被固化。
C++ prefill 时传空 `ncnn::Mat`，Concat 层自动处理零维度拼接。
```

#### mRoPE 的 position_ids 形状

Qwen3-TTS 使用多模态 RoPE (mRoPE)，`position_ids` 需要 **3D 形状** `(3, batch, seq)`，不是标准的 2D `(batch, seq)`：

```python
# 错误: (1, 8) — 会导致 "too many indices for tensor of dimension 2"
ex_position_ids = torch.arange(seq_len).unsqueeze(0)

# 正确: (3, 1, 8) — mRoPE 需要 3 个维度的位置编码
ex_position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
```

rotary_emb 的输出也是 3D：
```
cos shape: (3, 1, 8, 128)
sin shape: (3, 1, 8, 128)
```

#### pnnx 命令

```bash
# 导出时使用非空 cache (past_len=4) 确保 torch.cat 分支被 trace
# 输入: embeds[1,8,1024]f32, mask[8,12]f32, cos[8,128]f32, sin[8,128]f32,
#       cache_k0[8,4,128]f32, cache_v0[8,4,128]f32, ... (28 层 × 2)
pnnx decoder.pt inputshape=[1,8,1024]f32,[8,12]f32,[8,128]f32,[8,128]f32,[8,4,128]f32,[8,4,128]f32,... ncnnparam=decoder.ncnn.param ncnnbin=decoder.ncnn.bin fp16=0 optlevel=2 device=cpu
```

**结果:** 成功，95KB param + 1.6GB bin

---

### 5.4 Tokenizer-12Hz Decoder (音频 codec)

这是最复杂的部分，三种导出方式中只有 ONNX dynamo 路径成功。

#### 模型结构

```
Qwen3TTSTokenizerV2Model
├── encoder: Qwen3TTSTokenizerV2Encoder (编码器，不需要导出)
└── decoder: Qwen3TTSTokenizerV2Decoder (解码器，需要导出)
    ├── pre_transformer: nn.Module
    ├── quantizer: nn.Module
    ├── pre_conv: nn.Module
    ├── upsample: nn.Module
    └── decoder: nn.Module (内部 Transformer decoder)
```

#### 配置参数

```
decoder_config:
  num_quantizers: 16        ← 不是 8！
  codebook_size: 2048       ← 不是 32768！
  codebook_dim: 512
  decoder_dim: 1536
  hidden_size: 512
  num_hidden_layers: 8
  num_attention_heads: 16
  head_dim: 64
  semantic_codebook_size: 4096
  upsample_rates: [8, 5, 4, 3]
  upsample_ratios: [2, 2]

encoder_valid_num_quantizers: 16
decode_upsample_rate: 1920
```

#### Wrapper

直接包装 `model.decoder` 子模块（不包装 `model.decode()` 方法，因为后者有额外的输出封装）：

```python
class TokenizerDecoderTS(nn.Module):
    def __init__(self, tokenizer_model: nn.Module):
        super().__init__()
        self.decoder = tokenizer_model.decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)
```

#### 输入形状

codes 的形状是 `(batch, num_quantizers, T)`，**不是** `(batch, T, num_quantizers)`：

```python
# 错误: (1, 100, 16) — 会报 "Expected 16 layer of codes, got 100"
ex_codes = torch.randint(0, 2048, (1, 100, 16), dtype=torch.long)

# 正确: (1, 16, 100) — (batch, num_quantizers, T)
ex_codes = torch.randint(0, 2048, (1, 16, 100), dtype=torch.long)
```

#### 导出路径：ONNX dynamo → inline → pnnx

**TorchScript trace 失败原因:** 模型内部使用 `vmap`（向量化映射）来创建 causal mask，这会导致 `unordered_map::at: key not found` 错误。`torch.jit.trace` 无法处理这种动态控制流。

**ONNX legacy 导出失败原因:** 同样的 vmap 问题。

**ONNX dynamo 导出成功:** `torch.onnx.export(dynamo=True)` 使用 `torch.export` 而非 `torch.jit.trace`，可以处理更复杂的控制流。

但 dynamo 导出会产生**外部数据文件** (`.onnx.data`)，pnnx 不支持读取：

```
ort CreateSession failed graph.cc:4143
InjectExternalInitializersFromFilesInMemory
External file: tokenizer_decoder.onnx.data not found
```

**解决：** 用 `onnx` 库重新加载（含外部数据），然后以内联方式重新保存：

```python
import onnx

# 加载（自动合并外部数据）
onnx_model = onnx.load(onnx_path, load_external_data=True)

# 清除 data_location 标记
for tensor in onnx_model.graph.initializer:
    tensor.ClearField('data_location')

# 保存为单文件（无外部数据）
onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)

# 删除外部数据文件
ext_data = onnx_path + ".data"
if os.path.exists(ext_data):
    os.remove(ext_data)
```

**pnnx 命令:**
```bash
pnnx tokenizer_decoder.onnx inputshape=[1,16,100]i64 ncnnparam=tokenizer_decoder.ncnn.param ncnnbin=tokenizer_decoder.ncnn.bin fp16=0 optlevel=2 device=cpu
```

**结果:** 成功，69KB param + 435MB bin

---

## 6. Tokenizer 文件提取

### vocab.txt

从 `vocab.json` 转换为按 ID 排序的 token 列表：

```python
import json
with open(vocab_json_path) as f:
    vocab = json.load(f)
id_to_token = sorted(vocab.items(), key=lambda x: x[1])
with open(vocab_path, 'w') as f:
    for token, _ in id_to_token:
        f.write(token + '\n')
```

### merges.txt

直接从模型目录复制（tokenizer Python API 不暴露 merges）：

```python
import shutil
shutil.copy2(merges_txt_src, merges_path)
```

### 注意事项

- Qwen3-TTS 的 tokenizer 是 Qwen2TokenizerFast，加载时会警告 Mistral regex 问题，需要在 `AutoTokenizer.from_pretrained()` 时传 `fix_mistral_regex=True`
- vocab.txt 有 151643 个 token
- merges.txt 有 151387 行
- 额外 special tokens 包括 `<tts_pad>`, `<tts_text_bos>`, `<|audio_pad|>` 等

---

## 7. model.json 配置

最终生成的 `model.json`:

```json
{
  "model_type": "tts",
  "params": {
    "embed_token_param": "embed.ncnn.param",
    "embed_token_bin": "embed.ncnn.bin",
    "decoder_param": "decoder.ncnn.param",
    "decoder_bin": "decoder.ncnn.bin",
    "lm_head_param": "lm_head.ncnn.param",
    "lm_head_bin": "lm_head.ncnn.bin",
    "tokenizer_decoder_param": "tokenizer_decoder.ncnn.param",
    "tokenizer_decoder_bin": "tokenizer_decoder.ncnn.bin"
  },
  "tokenizer": {
    "type": "bbpe",
    "vocab_file": "vocab.txt",
    "merges_file": "merges.txt",
    "eos": "<|im_end|>",
    "bos": "",
    "additional_special_tokens": [...]
  },
  "setting": {
    "attn_cnt": 28,
    "tts_mode": "codec",
    "tts_model_type": "base",
    "rope": {
      "type": "RoPE",
      "rope_head_dim": 128,
      "rope_theta": 1000000
    },
    "audio": {
      "num_codebooks": 16,
      "codec_vocab_size": 2048,
      "sample_rate": 24000,
      "frame_rate": 12
    }
  }
}
```

### 关键参数来源

| 参数 | 值 | 来源 |
|---|---|---|
| attn_cnt | 28 | `config.talker_config.num_hidden_layers` |
| rope_head_dim | 128 | `config.talker_config.head_dim` |
| rope_theta | 1000000 | `config.talker_config.rope_theta` |
| num_codebooks | 16 | `speech_tokenizer/config.json` → `decoder_config.num_quantizers` |
| codec_vocab_size | 2048 | `speech_tokenizer/config.json` → `decoder_config.codebook_size` |
| sample_rate | 24000 | `speech_tokenizer/config.json` → `output_sample_rate` |
| frame_rate | 12 | 12Hz tokenizer（近似 12.5） |

---

## 8. 导出命令

### 完整导出（LLM + Tokenizer）

```bash
python export/qwen3_tts_export.py \
  --llm_model_id /path/to/qwen3_tts_llm \
  --tokenizer_model_id /path/to/qwen3_tts_llm/speech_tokenizer \
  --out_dir assets/qwen3_tts \
  --device cpu \
  --num_codebooks 16 \
  --tts_model_type base
```

### 仅导出 LLM（跳过 tokenizer）

```bash
python export/qwen3_tts_export.py \
  --llm_model_id /path/to/qwen3_tts_llm \
  --out_dir assets/qwen3_tts \
  --device cpu \
  --skip-tokenizer \
  --num_codebooks 16 \
  --tts_model_type base
```

---

## 9. 遇到的问题与解决方案汇总

| # | 问题 | 原因 | 解决方案 |
|---|---|---|---|
| 1 | `model type qwen3_tts not recognized` | transformers 不内置 Qwen3-TTS | 显式 `AutoConfig.register()` + `AutoModel.register()` |
| 2 | `import qwen_tts` 不注册模型类型 | 包设计如此，不会自动注册 | 从 `qwen_tts.core.models` 导入类并手动注册 |
| 3 | `Qwen3TTSForConditionalGeneration has no attribute embed_tokens` | 非标准 HF 结构 | 使用 `model.talker.model.text_embedding` |
| 4 | `lm_head not found` | 实际输出头叫 `codec_head` | 使用 `model.talker.codec_head` |
| 5 | `vocab_size: 3072` (不是 151936) | 这是 codec 词表 | 文本词表在 tokenizer 中，codec 词表在 talker_config 中 |
| 6 | decoder trace: `DynamicCache not supported` | torch.jit.trace 不支持非 tensor 返回 | 绕过 `DecoderLayer.forward()`，直接调用子模块，手动实现 attention + KV cache |
| 7 | decoder trace: `unordered_map::at: key not found` | 条件分支 `if position_ids.ndim == 3` | 同上，不传 position_ids，直接传 cos/sin |
| 8 | decoder 无 KV cache 输入/输出 | `use_cache=False` 绕过 DynamicCache | 手动 `torch.cat([cache_k, new_k])` 管理显式 tensor cache |
| 9 | C++ prefill 传空 cache 时 ncnn 报错 | cache 是 `torch.cat` 操作数，必须输入 | C++ 端传空 `ncnn::Mat`，Concat 层处理零维度 |
| 9 | tokenizer trace: `unordered_map::at: key not found` | vmap 用于 causal mask 创建 | 改用 ONNX dynamo 导出 |
| 10 | tokenizer ONNX: `External file .onnx.data not found` | dynamo 导出使用外部数据 | 用 onnx 库重新内联保存 |
| 11 | tokenizer: `Expected 16 layer of codes, got 8` | num_quantizers 默认值错误 | 从 `decoder_config.num_quantizers` 读取 (16) |
| 12 | tokenizer: `Expected 16 layer of codes, got 100` | codes 形状错误 | 使用 `(batch, num_quantizers, T)` 而非 `(batch, T, num_quantizers)` |
| 13 | `merges.txt` 空文件 | tokenizer API 不暴露 merges | 直接从模型目录复制文件 |
| 14 | `torch_dtype deprecated` | 新版 transformers | 改用 `dtype=torch.float32` |
| 15 | `huggingface-cli deprecated` | 新版 huggingface_hub | 改用 `hf download` |
| 16 | `sox not found` | qwen-tts 依赖 sox | `apt-get install sox` 或 `brew install sox` |
| 17 | pnnx Python API: `unexpected keyword argument outputdir` | API 签名不同 | 改用 pnnx CLI (subprocess) |

---

## 10. 导出后的文件清单

```
assets/qwen3_tts/
├── model.json                    1.3KB   配置文件
├── embed.ncnn.param              165B    embedding 网络结构
├── embed.ncnn.bin                1.2GB   embedding 权重
├── decoder.ncnn.param            95KB    decoder 网络结构
├── decoder.ncnn.bin              1.6GB   decoder 权重
├── lm_head.ncnn.param            177B    lm_head 网络结构
├── lm_head.ncnn.bin              12MB    lm_head 权重
├── tokenizer_decoder.ncnn.param  69KB    codec 解码器结构
├── tokenizer_decoder.ncnn.bin    435MB   codec 解码器权重
├── vocab.txt                     1.5MB   词汇表 (151643 tokens)
└── merges.txt                    1.6MB   BPE merges (151387 行)

总计: ~3.3GB
```

---

## 11. CI 适配

在 GitHub Actions 中运行导出需要：

1. **安装依赖**: `torch`, `transformers`, `qwen-tts`, `pnnx`, `onnxscript`, `onnx`, `sox`
2. **下载模型**: `hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base`
3. **运行导出**: `python export/qwen3_tts_export.py ...`
4. **参数校验**: `python tests/verify_tts_export.py ...`
5. **C++ 加载测试**: `tts_load_test --model assets/qwen3_tts`

CI 中可能的问题：
- 模型下载耗时 ~10min（1.7GB + 651MB）
- 导出耗时 ~5min（CPU）
- pnnx 转换耗时 ~5min
- 总计 ~20min

---

## 12. 后续工作

1. ~~**C++ 推理适配**~~: 已完成 — decoder 支持 KV cache 输入/输出，`llm_run_decoder_with_kv` 已适配
2. **精度验证**: 对比 PyTorch 原版和 ncnn 版的输出是否一致
3. **多码本生成逻辑**: LLM 生成的 codec token 序列如何映射到 `(batch, 16, T)` 矩阵
4. **流式生成**: Dual-Track hybrid streaming 架构的 C++ 实现
5. **语音克隆**: 参考音频编码 + speaker embedding 提取
