"""GPU 감지 및 모델 설정."""
from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger(__name__)

# Q5_K_M 모델 (~19.4GB)에 필요한 최소 VRAM (MB)
MIN_TOTAL_VRAM_MB = 18_000


def detect_gpus() -> list[dict]:
    """NVIDIA GPU와 VRAM을 감지한다. {name, total_mb, free_mb} 리스트를 반환한다."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "name": parts[0],
                "total_mb": int(parts[1]),
                "free_mb": int(parts[2]),
            })
    return gpus


def build_llama_config(gpus: list[dict]) -> dict:
    """
    사용 가능한 GPU를 기반으로 llama.cpp 설정을 구성한다.
    n_gpu_layers, tensor_split, n_ctx 등이 포함된 딕셔너리를 반환한다.
    VRAM이 부족하면 RuntimeError를 발생시킨다.
    """
    if not gpus:
        raise RuntimeError(
            "NVIDIA GPU가 감지되지 않았습니다.\n"
            "Qwopus는 총 VRAM 18GB 이상의 GPU가 필요합니다.\n"
            "27B 모델은 CPU 전용 모드를 지원하지 않습니다."
        )

    total_vram = sum(g["total_mb"] for g in gpus)
    total_free = sum(g["free_mb"] for g in gpus)

    if total_vram < MIN_TOTAL_VRAM_MB:
        gpu_info = ", ".join(f"{g['name']} ({g['total_mb']}MB)" for g in gpus)
        raise RuntimeError(
            f"VRAM 부족: GPU {len(gpus)}개에서 총 {total_vram:,}MB.\n"
            f"  GPU: {gpu_info}\n"
            f"  필요량: Q5_K_M (27B 모델)에 ~{MIN_TOTAL_VRAM_MB:,}MB.\n"
            f"\n"
            f"대안:\n"
            f"  - 더 작은 양자화 사용 (Q4_K_M, Q3_K_M)\n"
            f"  - 더 작은 모델 사용 (14B, 7B)\n"
            f"  - GPU VRAM 추가"
        )

    if total_free < MIN_TOTAL_VRAM_MB:
        gpu_info = ", ".join(f"{g['name']} (여유 {g['free_mb']}MB)" for g in gpus)
        raise RuntimeError(
            f"여유 VRAM 부족: GPU {len(gpus)}개에서 여유 {total_free:,}MB.\n"
            f"  GPU: {gpu_info}\n"
            f"  필요량: 여유 ~{MIN_TOTAL_VRAM_MB:,}MB.\n"
            f"\n"
            f"다른 GPU 프로세스를 종료하여 VRAM을 확보하세요 (nvidia-smi 확인)."
        )

    # 각 GPU의 VRAM에 비례하여 tensor_split 구성
    if len(gpus) == 1:
        tensor_split = None
    else:
        total = sum(g["total_mb"] for g in gpus)
        tensor_split = [g["total_mb"] / total for g in gpus]

    # 여유 VRAM에 따라 컨텍스트 크기 조정
    headroom = total_free - MIN_TOTAL_VRAM_MB
    if headroom > 8000:
        n_ctx = 16384
    elif headroom > 4000:
        n_ctx = 8192
    else:
        n_ctx = 4096

    config = {
        "n_gpu_layers": -1,
        "n_ctx": n_ctx,
        "n_batch": 512,
        "flash_attn": True,
        "verbose": False,
    }
    if tensor_split:
        config["tensor_split"] = tensor_split

    return config


def print_gpu_summary(gpus: list[dict]):
    """GPU 요약 정보를 보기 좋게 출력한다."""
    for i, g in enumerate(gpus):
        print(f"  GPU {i}: {g['name']} — 총 {g['total_mb']:,}MB, 여유 {g['free_mb']:,}MB")


def format_gpu_info(gpus: list[dict]) -> str:
    """GPU 정보를 한 줄 문자열로 포맷한다."""
    if not gpus:
        return ""
    parts = []
    for g in gpus:
        parts.append(f"{g['name']} ({g['total_mb']:,}MB)")
    return "GPU: " + " + ".join(parts)
