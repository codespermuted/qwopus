"""GPU detection and model configuration."""
from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger(__name__)

# Minimum VRAM (MB) required for the Q5_K_M model (~19.4GB)
MIN_TOTAL_VRAM_MB = 18_000


def detect_gpus() -> list[dict]:
    """Detect NVIDIA GPUs and their VRAM. Returns a list of {name, total_mb, free_mb}."""
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
    Build a llama.cpp config based on available GPUs.
    Returns a dict containing n_gpu_layers, tensor_split, n_ctx, etc.
    Raises RuntimeError if there is not enough VRAM.
    """
    if not gpus:
        raise RuntimeError(
            "No NVIDIA GPU detected.\n"
            "Qwopus requires GPUs with at least 18GB of total VRAM.\n"
            "The 27B model does not support CPU-only mode."
        )

    total_vram = sum(g["total_mb"] for g in gpus)
    total_free = sum(g["free_mb"] for g in gpus)

    if total_vram < MIN_TOTAL_VRAM_MB:
        gpu_info = ", ".join(f"{g['name']} ({g['total_mb']}MB)" for g in gpus)
        raise RuntimeError(
            f"Insufficient VRAM: {total_vram:,}MB across {len(gpus)} GPU(s).\n"
            f"  GPU: {gpu_info}\n"
            f"  Required: ~{MIN_TOTAL_VRAM_MB:,}MB for Q5_K_M (27B model).\n"
            f"\n"
            f"Alternatives:\n"
            f"  - Use a smaller quantization (Q4_K_M, Q3_K_M)\n"
            f"  - Use a smaller model (14B, 7B)\n"
            f"  - Add more GPU VRAM"
        )

    if total_free < MIN_TOTAL_VRAM_MB:
        gpu_info = ", ".join(f"{g['name']} (free {g['free_mb']}MB)" for g in gpus)
        raise RuntimeError(
            f"Insufficient free VRAM: {total_free:,}MB free across {len(gpus)} GPU(s).\n"
            f"  GPU: {gpu_info}\n"
            f"  Required: ~{MIN_TOTAL_VRAM_MB:,}MB free.\n"
            f"\n"
            f"Stop other GPU processes to free VRAM (check with nvidia-smi)."
        )

    # Build tensor_split proportional to each GPU's VRAM
    if len(gpus) == 1:
        tensor_split = None
    else:
        total = sum(g["total_mb"] for g in gpus)
        tensor_split = [g["total_mb"] / total for g in gpus]

    # Adjust context size based on free VRAM
    headroom = total_free - MIN_TOTAL_VRAM_MB
    if headroom > 12000:
        n_ctx = 32768
    elif headroom > 8000:
        n_ctx = 24576
    elif headroom > 4000:
        n_ctx = 16384
    else:
        n_ctx = 8192

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
    """Pretty-print a GPU summary."""
    for i, g in enumerate(gpus):
        print(f"  GPU {i}: {g['name']} — total {g['total_mb']:,}MB, free {g['free_mb']:,}MB")


def format_gpu_info(gpus: list[dict]) -> str:
    """Format GPU info as a single-line string."""
    if not gpus:
        return ""
    parts = []
    for g in gpus:
        parts.append(f"{g['name']} ({g['total_mb']:,}MB)")
    return "GPU: " + " + ".join(parts)
