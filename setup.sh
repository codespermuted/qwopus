#!/bin/bash
# ──────────────────────────────────────────────
# Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
# 환경 설정 스크립트 (Ubuntu + CUDA 12.8)
# ──────────────────────────────────────────────

set -e

echo "================================================"
echo "  Setup: Qwen3.5-27B Opus Distilled (Q5_K_M)"
echo "  Target: 2x RTX 5060 Ti (16GB each)"
echo "================================================"

# 1. GPU 확인
echo ""
echo "🔍 GPU 확인..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi를 찾을 수 없습니다. CUDA 드라이버를 확인하세요."
    exit 1
fi

# 2. huggingface_hub 설치
echo ""
echo "📦 huggingface_hub 설치..."
pip install huggingface_hub --break-system-packages -q

# 3. llama-cpp-python CUDA 빌드 설치
echo ""
echo "📦 llama-cpp-python (CUDA) 빌드 설치..."
echo "   ⏳ 컴파일이 포함되어 몇 분 걸릴 수 있습니다..."
CMAKE_ARGS="-DGGML_CUDA=on" \
pip install llama-cpp-python \
    --force-reinstall \
    --no-cache-dir \
    --break-system-packages

# 4. 설치 확인
echo ""
echo "🔍 설치 확인..."
python3 -c "
from llama_cpp import Llama
print('✅ llama-cpp-python 설치 완료')
import llama_cpp
print(f'   버전: {llama_cpp.__version__}')
"

# 5. 모델 사전 다운로드 (선택)
echo ""
read -p "🔽 모델을 지금 다운로드하시겠습니까? (Q5_K_M ≈ 19.4GB) [y/N]: " download_now

if [[ "$download_now" =~ ^[Yy]$ ]]; then
    python3 -c "
from huggingface_hub import hf_hub_download
from pathlib import Path

model_dir = Path.home() / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

print('⬇️  다운로드 시작...')
path = hf_hub_download(
    repo_id='mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF',
    filename='Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf',
    local_dir=str(model_dir),
    local_dir_use_symlinks=False,
)
print(f'✅ 다운로드 완료: {path}')
"
fi

echo ""
echo "================================================"
echo "  ✅ 설정 완료!"
echo ""
echo "  실행 방법:"
echo "    python3 run_qwopus.py          # 인터랙티브 채팅"
echo "    python3 run_qwopus.py --demo   # 단발성 추론 예시"
echo "================================================"
