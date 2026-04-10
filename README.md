# 🐙 Qwopus

**Local AI coding agent** — built on Qwen3.5-27B, a Claude 4.6 Opus reasoning knowledge-distillation model.
---

## Installation

```bash
# 1. Install Qwopus
pip install git+https://github.com/codespermuted/qwopus.git

# 2. Build llama-cpp-python with CUDA (required for GPU acceleration)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

The model (~19.4GB) is downloaded automatically on first run.

## System Requirements

- **GPU**: NVIDIA, **18GB+ total VRAM** (e.g. RTX 3090, 4090, dual 5060 Ti, etc.)
- **CUDA**: 12.0+
- **Python**: 3.10+
- **OS**: Linux

GPUs are auto-detected, and you'll see a clear message if VRAM is insufficient.

---

## Usage

```bash
# Run inside any project folder
cd your-project/
qwopus

# One-shot mode
qwopus "Explain the structure of this project"

# Specify a working directory
qwopus --cwd /path/to/project
```

### Commands

Run `/help` to see the full list.

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/status` | Session info |
| `/clear` | Reset the conversation |
| `/save` | Save the session |
| `/resume <id>` | Resume a session |
| `/quit` | Exit |
| `!command` | Run a shell command directly |

### Tools

Tools Qwopus uses autonomously:

**Bash** · **FileRead** · **FileWrite** · **FileEdit** · **Glob** · **Grep**

---

## How It Works

```
User input → LLM reasoning → Tool call → Feed result back → Re-reason → Answer
```

- Local GPU inference via llama.cpp
- Automatic multi-GPU sharding
- Dangerous commands require user confirmation
- Repeated identical tool calls are auto-stopped (hallucination guard)

---

## Credits

### Model
- **[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3-27B)** — Qwen Team (Alibaba) · Apache 2.0
- **[GGUF quantization](https://huggingface.co/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF)** — mradermacher

### Inference Engine
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — Georgi Gerganov · MIT
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Andrei Betlen · MIT

### Architectural References
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — Anthropic
- **[Claw Code](https://github.com/instructkr/claw-code)** — instructkr

### Other
- **[Rich](https://github.com/Textualize/rich)** · **[prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)** · **[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)**

---

## Disclaimer

- This is a **personal project** unaffiliated with Anthropic or Alibaba/Qwen.
- No Claude Code source code was copied; the architecture was independently implemented based on publicly documented patterns.
- Model output always requires verification. Production use is at the user's own risk.

## License

MIT — [LICENSE](LICENSE)
