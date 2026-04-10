"""
Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (Q5_K_M GGUF)
Multi-GPU inference script for RTX 5060 Ti x2 (16GB each)

Prerequisites:
  pip install huggingface_hub llama-cpp-python --break-system-packages

  # Requires a CUDA build (llama-cpp-python GPU acceleration):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir --break-system-packages
"""

import sys
import os
from pathlib import Path

# ──────────────────────────────────────────────
# 1. Model download (runs only on first invocation)
# ──────────────────────────────────────────────
MODEL_REPO = "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
MODEL_FILE = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf"
MODEL_DIR = Path.home() / "models"


def download_model() -> str:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / MODEL_FILE

    if local_path.exists():
        print(f"Model already exists: {local_path}")
        return str(local_path)

    print(f"Downloading model... ({MODEL_REPO}/{MODEL_FILE})")
    print("   Q5_K_M ~19.4GB — this may take a while depending on your connection.")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Download complete: {path}")
    return path


# ──────────────────────────────────────────────
# 2. Load the model (dual GPU configuration)
# ──────────────────────────────────────────────
def load_model(model_path: str):
    from llama_cpp import Llama

    print("Loading model (split across 2x RTX 5060 Ti)...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,           # Offload all layers to GPU
        n_ctx=8192,                # Context length (tune based on VRAM headroom)
        n_batch=512,               # Batch size
        tensor_split=[0.5, 0.5],   # Even split across GPU 0 and GPU 1
        flash_attn=True,           # Enable Flash Attention
        verbose=False,
    )
    print("Model loaded.")
    return llm


# ──────────────────────────────────────────────
# 3. Chat function
# ──────────────────────────────────────────────
def chat(llm, user_message: str, system_prompt: str = None, enable_thinking: bool = True):
    """Single-turn chat. Supports thinking mode."""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        # Qwen3.5 thinking mode: produces <think>...</think> before the answer
    )

    assistant_msg = response["choices"][0]["message"]["content"]
    return assistant_msg


def parse_thinking(response: str) -> tuple[str, str]:
    """Split out the <think>...</think> block from the final answer."""
    if "<think>" in response and "</think>" in response:
        think_start = response.index("<think>") + len("<think>")
        think_end = response.index("</think>")
        thinking = response[think_start:think_end].strip()
        answer = response[think_end + len("</think>"):].strip()
        return thinking, answer
    return "", response


# ──────────────────────────────────────────────
# 4. Interactive chat loop
# ──────────────────────────────────────────────
def interactive_chat(llm):
    system_prompt = "You are a helpful assistant. Think step by step."
    history = []

    print("\n" + "=" * 60)
    print("Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")
    print("   Q5_K_M | 2x RTX 5060 Ti")
    print("=" * 60)
    print("Commands: /quit (exit) | /clear (reset chat) | /think off|on (toggle thinking)")
    print("=" * 60 + "\n")

    show_thinking = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("Goodbye.")
            break
        if user_input == "/clear":
            history.clear()
            print("Conversation cleared.\n")
            continue
        if user_input.startswith("/think"):
            arg = user_input.split()[-1] if len(user_input.split()) > 1 else ""
            show_thinking = arg != "off"
            print(f"Thinking display: {'ON' if show_thinking else 'OFF'}\n")
            continue

        # Build multi-turn history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        print("Generating...", end="", flush=True)

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
        )

        assistant_msg = response["choices"][0]["message"]["content"]
        thinking, answer = parse_thinking(assistant_msg)

        print("\r" + " " * 20 + "\r", end="")  # Clear the "Generating..." line

        if thinking and show_thinking:
            print(f"Thinking:\n{thinking}\n")
        print(f"Assistant: {answer}\n")

        # Store only the final answer in history (per the model card recommendation)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        # Drop old messages if the history gets too long (context management)
        if len(history) > 20:
            history = history[-16:]


# ──────────────────────────────────────────────
# 5. One-shot inference example (for scripting)
# ──────────────────────────────────────────────
def single_inference_example(llm):
    """One-shot inference example — usable directly from a script."""
    prompt = "Analyze the role of VPPs (virtual power plants) in South Korea's energy transition policy and the outlook for their future."

    print(f"Prompt: {prompt}\n")
    response = chat(llm, prompt, system_prompt="You are a helpful energy domain expert.")
    thinking, answer = parse_thinking(response)

    if thinking:
        print(f"Thinking:\n{thinking}\n")
    print(f"Answer:\n{answer}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model_path = download_model()
    llm = load_model(model_path)

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        single_inference_example(llm)
    else:
        interactive_chat(llm)
