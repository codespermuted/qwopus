#!/bin/bash
# ──────────────────────────────────────────────
# Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
# Environment setup script (Ubuntu + CUDA 12.8)
# ──────────────────────────────────────────────

set -e

echo "================================================"
echo "  Setup: Qwen3.5-27B Opus Distilled (Q5_K_M)"
echo "  Target: 2x RTX 5060 Ti (16GB each)"
echo "================================================"

# 1. GPU check
echo ""
echo "Checking GPUs..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "nvidia-smi not found. Check your CUDA driver installation."
    exit 1
fi

# 2. Install huggingface_hub
echo ""
echo "Installing huggingface_hub..."
pip install huggingface_hub --break-system-packages -q

# 3. Install llama-cpp-python CUDA build
echo ""
echo "Installing llama-cpp-python (CUDA build)..."
echo "   This includes a compile step and may take a few minutes..."
CMAKE_ARGS="-DGGML_CUDA=on" \
pip install llama-cpp-python \
    --force-reinstall \
    --no-cache-dir \
    --break-system-packages

# 4. Verify install
echo ""
echo "Verifying installation..."
python3 -c "
from llama_cpp import Llama
print('llama-cpp-python installed')
import llama_cpp
print(f'   version: {llama_cpp.__version__}')
"

# 5. Pre-download model (optional)
echo ""
read -p "Download the model now? (Q5_K_M ~19.4GB) [y/N]: " download_now

if [[ "$download_now" =~ ^[Yy]$ ]]; then
    python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path

model_dir = Path.home() / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

print('Starting download...')
path = hf_hub_download(
    repo_id='mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF',
    filename='Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf',
    local_dir=str(model_dir),
    local_dir_use_symlinks=False,
)
print(f'Download complete: {path}')
"
fi

echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  How to run:"
echo "    python3 run_qwopus.py          # Interactive chat"
echo "    python3 run_qwopus.py --demo   # One-shot inference example"
echo "================================================"
