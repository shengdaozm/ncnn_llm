#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "========================================="
echo "  ncnn_llm LSP 一键配置脚本"
echo "========================================="

# ---- 1. 检查依赖 ----
echo ""
echo "[1/4] 检查依赖..."

check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "  [x] 未找到 $1，正在安装..."
    brew install "$1"
  else
    echo "  [v] $1 已安装: $(command -v "$1")"
  fi
}

check_cmd xmake

# clangd 优先用 homebrew 的版本
if ! command -v clangd &>/dev/null; then
  echo "  [x] 未找到 clangd，正在安装..."
  brew install llvm
  echo "  请将 /opt/homebrew/opt/llvm/bin 加入 PATH"
else
  echo "  [v] clangd 已安装: $(command -v clangd)"
fi

# ---- 2. 生成 compile_commands.json ----
echo ""
echo "[2/4] 生成 compile_commands.json..."

if [ -f ".vscode/compile_commands.json" ]; then
  echo "  [v] compile_commands.json 已存在，重新生成..."
fi

xmake project -k compile_commands .vscode
echo "  [v] compile_commands.json 生成完成"

# ---- 3. 可选：下载 ncnn 头文件（仅用于 LSP，不编译）----
echo ""
echo "[3/4] 检查 ncnn 头文件..."

if [ -d "refer/ncnn/src" ]; then
  echo "  [v] ncnn 头文件已存在于 refer/ncnn/"
else
  echo "  ncnn 头文件缺失，clangd 将无法解析 ncnn 相关符号。"
  read -p "  是否浅克隆 ncnn 仓库到 refer/ncnn 供 LSP 使用? (y/N) " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p refer
    git clone --depth 1 https://github.com/Tencent/ncnn.git refer/ncnn
    echo "  [v] ncnn 头文件已下载到 refer/ncnn/"
  else
    echo "  [!] 跳过。ncnn 相关头文件将无法被 LSP 解析。"
  fi
fi

# ---- 4. 配置文件就绪检查 ----
echo ""
echo "[4/4] 检查配置文件..."

if [ -f ".clangd" ]; then
  echo "  [v] .clangd 已就绪"
else
  echo "  [x] .clangd 缺失!"
  exit 1
fi

if [ -f ".vscode/settings.json" ]; then
  echo "  [v] .vscode/settings.json 已就绪"
else
  echo "  [x] .vscode/settings.json 缺失!"
  exit 1
fi

# ---- 完成 ----
echo ""
echo "========================================="
echo "  配置完成!"
echo "========================================="
echo ""
echo "使用方式:"
echo "  1. 用 VS Code 打开本仓库"
echo "  2. 安装 clangd 扩展 (llvm-vs-code-extensions.vscode-clangd)"
echo "  3. 不要安装/禁用 C/C++ 扩展的 IntelliSense (已在 settings.json 中禁用)"
echo "  4. clangd 会自动加载 .vscode/compile_commands.json"
echo ""
echo "如果修改了 xmake.lua 或增删源文件，重新运行本脚本即可更新编译数据库。"
echo ""
