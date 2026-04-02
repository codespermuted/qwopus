"""Local LLM engine — wraps llama-cpp-python for the harness."""
from __future__ import annotations

import logging
from pathlib import Path

from .gpu import detect_gpus, build_llama_config, print_gpu_summary

logger = logging.getLogger(__name__)

_llm = None

MODEL_REPO = "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
MODEL_FILE = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf"
MODEL_DIR = Path.home() / "models"


def get_llm():
    """Load the local model (singleton). Auto-detects GPU config."""
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    # Detect GPUs and build config
    gpus = detect_gpus()
    print("🔍 GPU detection:")
    if gpus:
        print_gpu_summary(gpus)
    config = build_llama_config(gpus)  # Raises RuntimeError if insufficient

    # Download model if needed
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILE

    if not model_path.exists():
        print(f"⬇️  Downloading model (~19.4GB): {MODEL_FILE}")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("✅ Download complete.")

    # Load model
    n_gpus = len(gpus)
    split_info = f", tensor_split={config.get('tensor_split', 'auto')}" if n_gpus > 1 else ""
    print(f"🚀 Loading model (n_ctx={config['n_ctx']}, {n_gpus} GPU(s){split_info})...")

    _llm = Llama(model_path=str(model_path), **config)
    print("✅ Model loaded.\n")
    return _llm


def chat_completion(messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3) -> dict:
    """Run chat completion and return the raw response dict."""
    llm = get_llm()
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )


def strip_thinking(text: str) -> tuple[str, str]:
    """Remove <think>...</think> block and internal reasoning, return (thinking, answer)."""
    import re

    # Handle <think>...</think> blocks (possibly multiple)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_parts = pattern.findall(text)
    if thinking_parts:
        thinking = "\n".join(p.strip() for p in thinking_parts)
        answer = pattern.sub("", text).strip()
        return thinking, answer

    # Handle </think> at start (model sometimes omits opening tag)
    if text.lstrip().startswith("</think>"):
        answer = text.split("</think>", 1)[1].strip()
        return "", answer

    # Heuristic: strip leading internal reasoning lines before the actual answer.
    # The model often starts with "The user wants..." / "Let me..." / "I need to..."
    # We detect these and move them to thinking.
    lines = text.split("\n")
    reasoning_lines = []
    answer_start = 0
    reasoning_prefixes = (
        "the user ", "let me ", "i need to ", "i should ", "i'll ",
        "i will ", "looking at ", "based on ", "now i ", "first,",
        "사용자가 ", "먼저 ", "확인해", "살펴보",
    )
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped and any(stripped.startswith(p) for p in reasoning_prefixes):
            reasoning_lines.append(line.strip())
            answer_start = i + 1
        else:
            break

    if reasoning_lines and answer_start < len(lines):
        thinking = "\n".join(reasoning_lines)
        answer = "\n".join(lines[answer_start:]).strip()
        if answer:
            return thinking, answer

    return "", text
