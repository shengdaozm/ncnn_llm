# LLM 推理全流程解析

本文档基于 `ncnn_llm` 源码，详细解析从程序启动到生成回复的完整链路。

## 目录

- [整体流程概览](#整体流程概览)
- [1. 程序入口与选项解析](#1-程序入口与选项解析)
- [2. 模型加载](#2-模型加载)
- [3. CLI 主循环](#3-cli-主循环)
- [4. Prefill 预填充](#4-prefill-预填充)
- [5. Generate 自回归解码](#5-generate-自回归解码)
- [6. 采样策略](#6-采样策略)
- [7. 工具调用（Tool Call）](#7-工具调用tool-call)
- [8. 视觉模型（VLM）流程](#8-视觉模型vlm流程)
- [总结：数据流](#总结数据流)

---

## 整体流程概览

```
main()
  → parse_options()              解析命令行参数
  → detect_template_type()       判断 ChatML / YouTu 模板
  → ncnn_llm_gpt(model_path)     加载模型（3 个 ncnn 网络 + 分词器）
  → run_cli()
      ├── prefill(system_prompt)     首次预填充，建立初始 KV cache
      ├── define_tools()             可选：注入工具定义
      └── while loop:
            ├── prefill(user_msg, ctx)  追加用户输入到上下文
            └── generate(ctx, cfg, cb)  自回归解码生成回复
```

---

## 1. 程序入口与选项解析

**入口文件**: `examples/llm_ncnn_run/main.cpp`

```cpp
int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    opt.model_path = normalize_model_path(opt.model_path);
    TemplateType template_type = detect_template_type(opt.model_path);
    ncnn_llm_gpt model(opt.model_path, opt.use_vulkan, opt.num_threads, opt.vulkan_device);
    // ...
    return run_cli(opt, model, builtin_tools, builtin_router, template_type, image);
}
```

### 关键步骤

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 解析选项 | `options.cpp:34` | `--model`, `--image`, `--threads`, `--use-vulkan`, `--vulkan-device` |
| 路径归一化 | `main.cpp:13` | 无路径前缀时自动补 `./assets/` |
| 模板检测 | `cli_runner.cpp:9` | 读 `model.json` 的 `type` 字段，区分 ChatML（Qwen3/MiniCPM4）和 YouTu |

### 支持的模板类型

```cpp
enum class TemplateType {
    CHATML,   // Qwen3, MiniCPM4 风格: <|im_start|>...<|im_end|>
    YOUTU     // YouTu LLM 风格: <|User|>...<|Assistant|>
};
```

---

## 2. 模型加载

**代码位置**: `src/ncnn_llm_gpt.cpp:18`（构造函数）

### 加载流程

```
读取 model.json
  ├── 加载 3 个核心 ncnn 网络:
  │     embed_net      token id → token embedding
  │     decoder_net    Transformer decoder（主计算图）
  │     proj_out_net   hidden state → logits（即 lm_head）
  │
  ├── [VLM] 额外加载视觉网络:
  │     vision_embed_patch   图像 patch → patch embedding
  │     vision_embed_pos     位置编码
  │     vision_encoder       ViT Transformer 编码器
  │
  ├── 加载 BPE 分词器 (vocab.txt + merges.txt)
  │     支持 bpe 和 bbpe 两种模式
  │     注册 additional_special_tokens
  │     解析 eos / bos token id
  │
  └── 解析模型配置:
        attn_cnt          注意力层数
        sconv_cnt / gdr_cnt   Qwen3.5 特有的状态缓存
        rope_type         RoPE 类型: RoPE / LongRoPE / NTK_RoPE / YaRN_RoPE
        rope_theta        RoPE 基频
        tool_call_id      工具调用 token id
        think_id          <think> token id
        vision 配置       patch_size, patch_dim, spatial_merge_size 等
```

### Vulkan / CPU 设置

```cpp
if (use_vulkan) {
    decoder_net->opt.use_vulkan_compute = true;
    decoder_net->opt.use_bf16_storage = true;
    // 注意：fp16 在 Vulkan 下关闭，避免精度问题
    decoder_net->opt.use_fp16_arithmetic = false;
    decoder_net->opt.use_fp16_storage = false;
} else {
    // 纯 CPU 推理
}
```

---

## 3. CLI 主循环

**代码位置**: `examples/llm_ncnn_run/cli_runner.cpp:31`

```cpp
int run_cli(...) {
    // 1. 首次 prefill：注入 system prompt，建立初始 KV cache
    std::string system_prompt = "You are a helpful assistant.";
    std::string prompt = apply_chat_template(template_type, {{"system", system_prompt}}, {}, false, false);
    auto ctx = model.prefill(prompt);

    // 2. 可选：注入工具定义
    if (!builtin_tools.empty()) {
        ctx = model.define_tools(ctx, builtin_tools, system_prompt);
    }

    // 3. 对话循环
    while (true) {
        std::string input;
        std::getline(std::cin, input);
        if (input == "exit" || input == "quit") break;

        // 拼接 chat template 并 prefill
        std::string user_message = apply_chat_template(...);
        ctx = model.prefill(user_message, ctx);          // 追加到已有上下文

        // 自回归生成
        GenerateConfig cfg;
        cfg.top_k = 40;
        cfg.top_p = 0.9f;
        cfg.temperature = 0.7f;
        cfg.do_sample = false;

        ctx = model.generate(ctx, cfg, [](const std::string& token) {
            std::cout << token << std::flush;             // 流式输出
        });
    }
}
```

### 上下文对象 `ctx`

`ctx` 是 `shared_ptr<ncnn_llm_gpt_ctx>`，携带跨轮次的状态：

```cpp
class ncnn_llm_gpt_ctx {
    KVCache kv_cache;      // 每层的 (K, V) 缓存
    int cur_token;         // 当前 token id（上次生成的）
    int position_id;       // RoPE 位置计数器
};

// Qwen3.5 额外有:
class qwen3_5_ctx : public ncnn_llm_gpt_ctx {
    std::vector<ncnn::Mat> sconv_cache;   // 状态卷积缓存
    std::vector<ncnn::Mat> gdr_cache;     // GDR 缓存
};
```

**多轮对话原理**: 每轮 prefill/generate 都会 clone 一份 ctx，在其基础上追加新 token 的 KV cache，实现增量推理。旧的 ctx 不被修改，可用于回溯。

---

## 4. Prefill 预填充

**代码位置**: `src/ncnn_llm_gpt.cpp:248`（首次）、`:639`（追加）

Prefill 的作用是**批量处理 prompt 中的所有 token，建立 KV cache，并预测第一个生成 token**。

### 核心步骤

```
① 分词
   token_ids = bpe->encode(input_text)
   ↓
② 拆分最后一个 token
   last_token = token_ids.back()
   token_ids.pop_back()
   （前 n-1 个 token 批量建 cache，最后一个单独跑出 hidden state）
   ↓
③ 生成 RoPE 位置编码 (cos_cache, sin_cache)
   根据 rope_type 选择:
     - RoPE:      generate_rope_embed_cache()
     - LongRoPE:  generate_rope_embed_cache_LongRoPE()
     - NTK_RoPE:  generate_ntk_rope_embed_cache()
     - YaRN_RoPE: generate_yarn_rope_embed_cache()
   ↓
④ Token Embedding
   embed_net(input_ids) → token_embed   [shape: (seq_len, hidden_size)]
   ↓
⑤ 构建 Causal Mask（因果注意力掩码）
   mask[i][j] = 0      if j ≤ i
   mask[i][j] = -1e38  if j > i    （上三角屏蔽）
   ↓
⑥ Decoder 前向（批量处理 seq_len-1 个 token）
   decoder_net(
     input  = token_embed,
     mask   = causal_mask,
     cos    = cos_cache,
     sin    = sin_cache,
     [input = kv_cache]    // 追加模式时传入已有 cache
   )
   → 输出 out_cache_k{i} / out_cache_v{i}  → 建立/更新 KV cache
   ↓
⑦ 单独处理 last_token
   embed_net(last_token_id) → last_token_embed
   decoder_net(
     input = last_token_embed,
     mask  = [kv_len+1, 1] (全 0, 可看到所有历史),
     cos/sin = last_position,
     input  = kv_cache
   )
   → 更新 KV cache → 得到 decode_out（hidden state）
   ↓
⑧ LM Head
   proj_out_net(decode_out) → logits   [shape: (vocab_size,)]
   ↓
⑨ Argmax
   next_token_id = argmax(logits)
   ↓
⑩ 返回更新后的 ctx
   { kv_cache, cur_token = next_token_id, position_id += seq_len }
```

### 为什么拆最后一个 token？

前 `n-1` 个 token 批量走 decoder 建立 KV cache，最后一个 token 单独跑一遍 decoder 获取其 hidden state 用于预测。这样可以：
- 避免对最后一个 token 做不必要的 mask 处理
- 保持 batch prefill 的效率

### 追加模式 Prefill（多轮对话）

与首次 prefill 的区别：
1. **clone ctx**：复制上一轮的 KV cache
2. **RoPE 位置**：从 `ctx->position_id` 开始，而非 0
3. **Mask 维度**：`(kv_cache_len + new_seq_len, new_seq_len)`，已有 cache 部分全可见
4. **Decoder 输入**：需要传入 `cache_k{i}` / `cache_v{i}` 作为历史

---

## 5. Generate 自回归解码

**代码位置**: `src/ncnn_llm_gpt.cpp:843`

Generate 在 prefill 建立的 KV cache 基础上，**逐 token 生成回复**。

```cpp
for (int step = 0; step < cfg.max_new_tokens; ++step) {
    if (ctx->cur_token == eos) break;          // 遇到 EOS 停止

    // ① Tool call 检测
    if (ctx->cur_token == tool_call_id) {
        flag_in_tool_call = true;              // 开始捕获工具调用 JSON
    } else if (ctx->cur_token == tool_call_end_id) {
        flag_in_tool_call = false;
        handle_tool(tool_call_content, ctx);   // 执行工具 → prefill 结果
        tool_call_content.clear();
        continue;
    } else if (flag_in_tool_call) {
        tool_call_content += bpe->decode({cur_token});
    } else {
        callback(bpe->decode({cur_token}));    // 正常输出到终端
    }

    // ② Embed 当前 token
    ncnn::Mat cur_embed = llm_run_text_embed(*embed_net, ctx->cur_token);

    // ③ 生成 RoPE (position = ctx->position_id)
    generate_rope_embed_cache(1, rope_head_dim, ctx->position_id, cos, sin);
    ctx->position_id++;

    // ④ 构建 Mask（单 token, 可看到所有历史）
    ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
    mask.fill(0.f);

    // ⑤ Decoder 前向（单 token, 带 KV cache）
    decode_out = llm_run_decoder_with_kv(
        decoder_net, cur_embed, mask, cos, sin,
        ctx->kv_cache, attn_cnt, false /* is_decode */
    );
    // → KV cache 每层追加当前 token 的 K/V

    // ⑥ LM Head
    ncnn::Mat logits = llm_run_lm_head(*proj_out_net, decode_out);

    // ⑦ 采样
    int next_id = llm_select_next_token(logits, history, sample_cfg);
    ctx->cur_token = next_id;
    history.insert(next_id);
}
```

### 单步解码的 ncnn 调用

```cpp
// ncnn_text_runtime.cpp:34
ncnn::Mat llm_run_decoder_with_kv(...) {
    ncnn::Extractor ex = decoder_net.create_extractor();
    ex.input("in0", embeds);      // 当前 token embedding
    ex.input("in1", mask);        // 注意力掩码
    ex.input("in2", cos_cache);   // RoPE cos
    ex.input("in3", sin_cache);   // RoPE sin

    // 传入历史 KV cache
    for (int i = 0; i < attn_cnt; i++) {
        ex.input("cache_k{i}", kv_cache[i].first);
        ex.input("cache_v{i}", kv_cache[i].second);
    }

    // 提取更新后的 KV cache
    for (int i = 0; i < attn_cnt; i++) {
        ex.extract("out_cache_k{i}", k_cache);
        ex.extract("out_cache_v{i}", v_cache);
        kv_cache[i] = {k_cache, v_cache};
    }

    ex.extract("out0", decode_out);  // hidden state
    return decode_out;
}
```

---

## 6. 采样策略

**代码位置**: `src/ncnn_text_runtime.cpp:85` + `src/sampling.cpp`

```
logits (vocab_size,)
  │
  ├── ① 重复惩罚 (Repetition Penalty)
  │     对 history 中的每个 token t:
  │       if scores[t] > 0: scores[t] /= penalty
  │       else:             scores[t] *= penalty
  │
  ├── ② 贪心模式 (do_sample = false 或 temperature ≤ 0)
  │     → 直接 argmax(scores)
  │
  └── ③ 采样模式 (do_sample = true)
        ├── softmax(scores / temperature)    温度缩放
        ├── top_k: 只保留概率最高的 k 个      截断
        ├── top_p: 保留累积概率 ≥ p 的 token  核采样截断
        └── sample_from_probs()              按概率随机采样
              → std::discrete_distribution
```

### 默认参数

```cpp
struct GenerateConfig {
    int   max_new_tokens    = 4096;
    float temperature       = 0.3;
    float top_p             = 0.8;
    int   top_k             = 50;
    float repetition_penalty = 1.1;
    int   do_sample         = 1;
};
```

CLI 中默认 `do_sample = false`（贪心解码）。

---

## 7. 工具调用（Tool Call）

**代码位置**: `src/ncnn_llm_gpt.cpp:846`

当模型配置了 `tool_call_id` 和 `tool_call_end_id` 时，generate 循环会检测特殊 token：

```
模型生成 <tool_call> token
  → flag_in_tool_call = true
  → 后续 token 累积到 tool_call_content（逐 token decode）

模型生成 </tool_call> token
  → handle_tool():
      ① json::parse(tool_call_content)         解析工具调用 JSON
      ② cfg.tool_callback(json) → result       执行回调函数
      ③ prefill(                                把工具结果喂回模型:
           "<|im_end|>\n"
           "<|im_start|>user\n"
           "<tool_response>\n"
           "{result}\n"
           "</tool_response>"
           "<|im_end|>\n"
           "<|im_start|>assistant\n"
           "<think>\n</think>\n\n"
         )
      ④ 继续生成（模型看到工具结果后继续回答）
```

### Chat Template 中的工具定义

```cpp
// prompt.cpp:31
"<|im_start|>system\n"
"You may call one or more functions..."
"<tools>"
  {"type":"function","function":{"name":"...","parameters":{...}}}
"</tools>"
"For each function call, return a json object..."
"<tool_call>\n{\"name\": ..., \"arguments\": ...}\n</tool_call>"
```

---

## 8. 视觉模型（VLM）流程

**代码位置**: `src/ncnn_llm_gpt.cpp:438`（VLM prefill）+ `:1158`（视觉特征提取）

当传入 `--image` 参数时，prefill 会额外处理图像：

```
① 图像预处理
   bgr_to_pixel_values()
     ├── 图片按 patch_size (默认 14) 切块
     ├── 每个像素归一化: (pixel/255 - mean) / std
     └── 输出: pixel_values [shape: (patch_size²×3, num_patches)]
   ↓
② Patch 重排
   reorder_patches_for_merge()
     按 spatial_merge_size (默认 2) 重排 patch 顺序
   ↓
③ Patch Embedding
   vision_embed_patch(patch) → patch_embed
   每个 patch 独立过一次网络
   ↓
④ [可选] 位置编码
   vision_embed_pos(grid) → pos_embeds
   ↓
⑤ ViT Encoder
   vision_encoder(patch_embeds, pos_embeds, emb_cos, emb_sin)
   → image_embeds  [shape: (patch_dim, num_patches)]
   ↓
⑥ 注入到文本序列
   inject_image_embeds()
     把 token 序列中 <|image_pad|> 的 embedding 替换为 image_embeds
   ↓
⑦ 多模态 RoPE (mRoPE)
   按 (时间, 高度, 宽度) 分维度生成位置编码
   generate_rope_embed_cache_vision_mrope()
```

### 支持的视觉类型

| 类型 | 说明 |
|---|---|
| `VISION_CLOSE` | 纯文本模型，不处理图像 |
| `VISION_VIT` | 标准 ViT 视觉编码器（Qwen2.5-VL） |
| `VISION_QWEN3_5_VL` | Qwen3.5-VL 风格，使用不同的归一化和 mRoPE |

---

## 总结：数据流

```
用户文本
  → apply_chat_template()           拼接 prompt（ChatML / YouTu 格式）
  → BPE encode()                    token_ids[]
  → embed_net()                     token embeddings
  → [VLM: vision_encoder() → inject]  图像嵌入注入
  → decoder_net()                   Transformer 前向（RoPE + causal mask + KV cache）
  → proj_out_net()                  logits
  → sampling()                      rep_penalty → temp → top_k → top_p → sample
  → next_token
  → BPE decode()                    输出文本
  → 循环直到 EOS
```

### 核心设计要点

1. **KV Cache 共享**: prefill 批量建立 cache，generate 每步追加 1 个 token 的 cache，避免重复计算历史 token
2. **Context Clone**: 每轮对话 clone ctx，实现不可变历史 + 增量推理
3. **三个 ncnn 网络分离**: embed / decoder / lm_head 独立，便于不同模型复用 decoder 逻辑
4. **RoPE 在 CPU 端计算**: cos/sin cache 在 C++ 端生成后作为 blob 输入 ncnn，而非在图中计算
5. **VLM 复用文本 decoder**: 图像 embedding 注入到 token embedding 后，走相同的 decoder 前向路径
