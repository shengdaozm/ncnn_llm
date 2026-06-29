# ncnn_llm 使用说明

`ncnn_llm` 是一个基于 `ncnn` 的 C++ 推理项目，主要用于本地运行 LLM、VLM、OCR、翻译和 embedding 模型。项目使用 `xmake` 构建。

## 依赖

需要先准备：

- `xmake`
- `ncnn master`
- `nlohmann_json`

当前 `xmake.lua` 中依赖声明如下：

```lua
add_requires("ncnn master", {
    configs = {
        vulkan=true
    }
})

add_requires("nlohmann_json")
```

因此默认情况下，`xmake` 会通过自己的包管理流程查找或安装 `ncnn` 和 `nlohmann_json`。如果希望使用本机已经安装到 `~/.local` 的 `ncnn`，需要额外调整 xmake 配置或包查找方式。

## 编译

进入项目目录：

```bash
cd ../ncnn_llm
```

编译全部目标：

```bash
xmake
```

或：

```bash
xmake build
```

只编译单个目标，例如主 CLI：

```bash
xmake build llm_ncnn_run
```

## LSP 补全

项目已经在 `xmake.lua` 中启用了 `compile_commands.json` 自动生成：

```lua
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
```

执行 `xmake` 后会生成：

```text
.vscode/compile_commands.json
```

如果 `clangd` 没有自动识别，可以在项目根目录创建软链接：

```bash
ln -sf .vscode/compile_commands.json compile_commands.json
```

或者添加 `.clangd`：

```yaml
CompileFlags:
  CompilationDatabase: .vscode
```

## 编译目标

当前 `xmake.lua` 中实际定义了这些 target：

| Target | 类型 | 说明 |
| --- | --- | --- |
| `ncnn_tokenizer` | 静态库 | tokenizer 相关实现 |
| `ncnn_llm` | 静态库 | 核心 LLM / VLM / OCR / embedding 运行时 |
| `llm_ncnn_run` | 可执行程序 | 主入口，支持聊天和 VLM 图文推理 |
| `benchllm` | 可执行程序 | LLM benchmark |
| `test_llm` | 可执行程序 | 测试程序 |
| `nllb_main` | 可执行程序 | NLLB 翻译示例 |
| `embedding_main` | 可执行程序 | 文本 embedding 示例 |
| `clip_main` | 可执行程序 | CLIP 图文 embedding 示例 |
| `ocr_main` | 可执行程序 | OCR 示例 |

## 运行示例

### 文本聊天

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b
```

指定 CPU 线程数：

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b --threads 4
```

启用 Vulkan：

```bash
xmake run llm_ncnn_run --model ./assets/qwen3_0.6b --vulkan --vulkan-device 0
```

### VLM 图文输入

```bash
xmake run llm_ncnn_run --model ./assets/qwen2.5_vl_3b --image ./assets/test.jpg
```

### OCR

```bash
xmake build ocr_main
xmake run ocr_main --model ./assets/glm_ocr --image ./test_ocr.png --prompt "Read the text in the image."
```

### 文本 Embedding

```bash
xmake build embedding_main
xmake run embedding_main --model ./assets/jina-embeddings-v5-text-nano
```

### CLIP 多模态 Embedding

```bash
xmake build clip_main
xmake run clip_main --model ./assets/jina_clip_v2 --image ./assets/ganyu.jpg
```

### NLLB 翻译

```bash
xmake build nllb_main
xmake run nllb_main --model ./assets/nllb
```

### 测试

```bash
xmake build test_llm
xmake run test_llm
```

### Benchmark

```bash
xmake build benchllm
xmake run benchllm [loop_count] [num_threads] [powersave] [gpu_device] [cooling_down] [seqlen]
```

## 支持能力

项目 README 中标注支持以下模型类型：

| 类型 | 模型 | 说明 |
| --- | --- | --- |
| LLM | YoutuLLM | 聊天 / 文本生成 |
| LLM | MiniCPM4 | 聊天 / 文本生成 |
| LLM | Qwen3 | 聊天 / 文本生成 |
| VLM | Qwen3.5 | 图像 + 文本输入 |
| VLM | Qwen2.5-VL | 图像 + 文本输入 |
| OCR | GLM-OCR | 图片文字识别 |
| 翻译 | NLLB | 翻译示例 |
| Embedding | Jina-Embeddings-v5-Text-Nano | 768 维文本嵌入 |
| Embedding | Jina-CLIP-v2 | 1024 维文本 + 图像嵌入 |

## 模型目录

模型需要使用已经转换成 ncnn 格式的模型目录。README 中给出的模型下载地址：

```text
https://mirrors.sdu.edu.cn/ncnn_modelzoo/
```

下载后建议放到 `assets/` 下，例如：

```text
assets/
└── qwen3_0.6b/
    ├── model.json
    ├── *.ncnn.param
    ├── *.ncnn.bin
    └── tokenizer files
```

运行时通过 `--model` 指定模型目录。

当前本地 `assets/` 目录里基本还没有模型文件，只有 tokenizer 相关文件，因此要实际运行推理，需要先下载对应模型目录。

## 常用 CLI 参数

`llm_ncnn_run` 常用参数：

| 参数 | 说明 |
| --- | --- |
| `--model` | 模型目录 |
| `--threads` | CPU 线程数 |
| `--vulkan` | 启用 Vulkan 计算 |
| `--vulkan-device` | Vulkan 设备编号 |
| `--image` | VLM 输入图像路径 |
| `--builtin-tools` | 启用内置演示工具 |

## 项目结构

```text
ncnn_llm/
├── assets/                 # 本地模型目录和演示资源
├── benchmark/              # Benchmark 入口
├── examples/               # CLI 和功能示例
│   ├── llm_ncnn_run/       # 统一聊天 / VLM 运行器
│   ├── ocr_main.cpp        # OCR 示例
│   ├── embedding_main.cpp  # 文本嵌入示例
│   ├── clip_main.cpp       # CLIP 示例
│   └── nllb_main.cpp       # 翻译示例
├── export/                 # 导出脚本
├── src/                    # 核心运行时
│   ├── ncnn_llm_gpt.*      # LLM / VLM 运行时
│   ├── ncnn_llm_ocr.*      # OCR 图像 prefill + 共享解码
│   ├── ncnn_embedding.*    # 嵌入模型运行时
│   ├── ncnn_text_runtime.* # 共享文本解码辅助函数
│   └── utils/              # 分词器、图像、RoPE、prompt 工具
├── tests/                  # 单元测试
└── xmake.lua               # 构建配置
```
