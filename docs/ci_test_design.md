# CI 自动化测试设计笔记

## 设计时间: 2026-07-01

## 目标

在 GitHub Actions (Ubuntu CPU runner) 上自动完成：
1. 下载 Qwen3-TTS 模型（LLM + Tokenizer-12Hz）
2. 运行 PNNX 导出脚本，生成 ncnn 格式模型
3. 校验导出的 model.json 参数与 PyTorch 原版一致
4. C++ 加载 ncnn 模型，验证网络可正常初始化（不要求完整推理）

## 两个 Workflow 文件

### 1. `tts-ci.yml` — 快速编译验证

**触发**: push/PR 修改 TTS 相关代码时
**平台**: Ubuntu only
**内容**:
- `build-tts` job: xmake 编译 `tts_main` + `tts_load_test`，验证二进制可启动
- `python-lint` job: 语法检查导出脚本和验证脚本

### 2. `tts-model-ci.yml` — 完整模型测试

**触发**: push/PR + `workflow_dispatch`（手动触发）
**平台**: Ubuntu only
**内容**:

#### Job 1: `export-and-verify`
1. 安装 Python 依赖 (torch, transformers, pnnx)
2. `huggingface-cli download` 拉取模型（缓存）
3. 运行 `export/qwen3_tts_export.py` 导出
4. 运行 `tests/verify_tts_export.py` 参数校验
5. 上传导出的 ncnn 模型为 artifact

#### Job 2: `cpp-load-test` (依赖 Job 1)
1. 下载 Job 1 的模型 artifact
2. xmake 编译 `tts_load_test` + `tts_main`
3. 运行 `tts_load_test --model assets/qwen3_tts` — 验证网络加载
4. 运行 `tts_main --model ... --text "Hello"` — 冒烟测试（60s 超时）

## 资源评估

| 模型 | 大小(估) | HuggingFace ID |
|---|---|---|
| Qwen3-TTS-12Hz-0.6B-Base | ~2.4GB (fp32) | Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| Qwen3-TTS-Tokenizer-12Hz | ~300MB (估) | Qwen/Qwen3-TTS-Tokenizer-12Hz |

GitHub Actions runner: 2 核 CPU, 7GB RAM, 14GB SSD

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 模型下载超时 | 缓存 + 只下载 0.6B (最小) |
| PNNX 导出失败 | 保留 TorchScript fallback，CI 不阻塞 |
| C++ 推理 OOM/超时 | 只测试模型加载 + 60s 超时冒烟测试 |
| 模型需要授权 | 下载失败时 CI 跳过后续步骤，不阻塞 |

## 新增文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `.github/workflows/tts-ci.yml` | CI | 快速编译验证 |
| `.github/workflows/tts-model-ci.yml` | CI | 完整模型测试 |
| `tests/tts_load_test.cpp` | C++ | ncnn 网络加载测试程序 |
| `tests/verify_tts_export.py` | Python | 参数校验脚本 |

## xmake.lua 变更

新增 `tts_load_test` target:
```lua
target("tts_load_test")
    set_kind("binary")
    add_files("tests/tts_load_test.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")
    set_rundir("$(projectdir)/")
```
