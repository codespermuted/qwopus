"""로컬 LLM 엔진 — 하네스를 위한 llama-cpp-python 래퍼."""
from __future__ import annotations

import logging
from pathlib import Path

from .gpu import detect_gpus, build_llama_config, print_gpu_summary

logger = logging.getLogger(__name__)

_llm = None
_n_ctx = 8192  # 모델 로드 시 설정됨

# 기본값 — settings.json에서 오버라이드 가능
DEFAULT_REPO = "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
DEFAULT_FILE = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf"
MODEL_DIR = Path.home() / "models"


def get_llm():
    """로컬 모델을 로드한다 (싱글턴). GPU 설정을 자동 감지한다."""
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    # settings.json에서 모델 설정 로드
    from .config import Settings
    settings = Settings.load()
    model_repo = settings.get("model.repo", DEFAULT_REPO)
    model_file = settings.get("model.file", DEFAULT_FILE)

    # GPU 감지 및 설정 구성
    gpus = detect_gpus()
    print("🔍 GPU 감지:")
    if gpus:
        print_gpu_summary(gpus)
    config = build_llama_config(gpus)

    # 필요 시 모델 다운로드
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        print(f"⬇️  모델 다운로드 중: {model_file}")
        hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("✅ 다운로드 완료.")

    # 모델 로드
    n_gpus = len(gpus)
    split_info = f", tensor_split={config.get('tensor_split', 'auto')}" if n_gpus > 1 else ""
    print(f"🚀 모델 로드 중 (n_ctx={config['n_ctx']}, GPU {n_gpus}개{split_info})...")

    global _n_ctx
    _n_ctx = config["n_ctx"]
    _llm = Llama(model_path=str(model_path), **config)
    print("✅ 모델 로드 완료.\n")
    return _llm


def get_n_ctx() -> int:
    """컨텍스트 윈도우 크기를 반환한다 (get_llm() 호출 후 사용)."""
    return _n_ctx


def chat_completion(messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3) -> dict:
    """채팅 완성을 실행하고 원시 응답 딕셔너리를 반환한다."""
    llm = get_llm()
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )


def chat_completion_stream(messages: list[dict], max_tokens: int = 4096, temperature: float = 0.3):
    """스트리밍 채팅 완성. 토큰 단위로 yield한다."""
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
    """<think>...</think> 블록과 내부 추론을 제거하고 (사고 과정, 답변)을 반환한다."""
    import re

    # <think>...</think> 블록 처리 (여러 개일 수 있음)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_parts = pattern.findall(text)
    if thinking_parts:
        thinking = "\n".join(p.strip() for p in thinking_parts)
        answer = pattern.sub("", text).strip()
        return thinking, answer

    # 시작 부분의 </think> 처리 (모델이 여는 태그를 생략하는 경우가 있음)
    if text.lstrip().startswith("</think>"):
        answer = text.split("</think>", 1)[1].strip()
        return "", answer

    # 휴리스틱: 실제 답변 전에 나오는 내부 추론 줄을 제거한다.
    # 모델이 종종 "The user wants..." / "Let me..." / "I need to..." 등으로 시작한다.
    # 이러한 패턴을 감지하여 사고 과정으로 이동시킨다.
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
