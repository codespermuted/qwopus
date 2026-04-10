"""Local LLM engine — llama-cpp-python wrapper for the harness."""
from __future__ import annotations

import logging
from pathlib import Path

from .gpu import detect_gpus, build_llama_config, print_gpu_summary

logger = logging.getLogger(__name__)

_llm = None
_n_ctx = 8192  # Set when the model is loaded

# Defaults — can be overridden in settings.json
DEFAULT_REPO = "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
DEFAULT_FILE = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf"
MODEL_DIR = Path.home() / "models"


def get_llm():
    """Load the local model (singleton). Auto-detects GPU configuration."""
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    # Load model settings from settings.json
    from .config import Settings
    settings = Settings.load()
    model_repo = settings.get("model.repo", DEFAULT_REPO)
    model_file = settings.get("model.file", DEFAULT_FILE)

    # Detect GPUs and build config
    gpus = detect_gpus()
    print("Detecting GPUs:")
    if gpus:
        print_gpu_summary(gpus)
    config = build_llama_config(gpus)

    # Download model if needed
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        print(f"Downloading model: {model_file}")
        hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("Download complete.")

    # Load the model
    n_gpus = len(gpus)
    split_info = f", tensor_split={config.get('tensor_split', 'auto')}" if n_gpus > 1 else ""
    print(f"Loading model (n_ctx={config['n_ctx']}, {n_gpus} GPU(s){split_info})...")

    global _n_ctx
    _n_ctx = config["n_ctx"]
    _llm = Llama(model_path=str(model_path), **config)
    print("Model loaded.\n")
    return _llm


def get_n_ctx() -> int:
    """Return the context window size (use after get_llm() has been called)."""
    return _n_ctx


def chat_completion(messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3) -> dict:
    """Run a chat completion and return the raw response dict."""
    llm = get_llm()
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )


def chat_completion_stream(messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3):
    """Streaming chat completion. Yields tokens one at a time."""
    llm = get_llm()
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            yield content


def strip_thinking(text: str) -> tuple[str, str]:
    """Remove <think>...</think> blocks and internal reasoning, returning (thinking, answer)."""
    import re

    # Handle <think>...</think> blocks (there may be several)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_parts = pattern.findall(text)
    if thinking_parts:
        thinking = "\n".join(p.strip() for p in thinking_parts)
        answer = pattern.sub("", text).strip()
        return thinking, answer

    # Handle a leading </think> (the model sometimes omits the opening tag)
    if text.lstrip().startswith("</think>"):
        answer = text.split("</think>", 1)[1].strip()
        return "", answer

    # Heuristic: strip internal reasoning lines that appear before the real answer.
    # The model often starts with "The user wants..." / "Let me..." / "I need to..." etc.
    # Detect these patterns and move them into the thinking section.
    lines = text.split("\n")
    reasoning_lines = []
    answer_start = 0
    reasoning_prefixes = (
        "the user ", "let me ", "i need to ", "i should ", "i'll ",
        "i will ", "looking at ", "based on ", "now i ", "first,",
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
