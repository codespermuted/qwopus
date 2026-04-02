"""
Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled (Q5_K_M GGUF)
RTX 5060 Ti x2 (16GB each) 멀티GPU 추론 스크립트

사전 설치:
  pip install huggingface_hub llama-cpp-python --break-system-packages

  # CUDA 빌드가 필요 (llama-cpp-python GPU 가속):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir --break-system-packages
"""

import sys
import os
from pathlib import Path

# ──────────────────────────────────────────────
# 1. 모델 다운로드 (최초 1회만 실행됨)
# ──────────────────────────────────────────────
MODEL_REPO = "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF"
MODEL_FILE = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf"
MODEL_DIR = Path.home() / "models"


def download_model() -> str:
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / MODEL_FILE

    if local_path.exists():
        print(f"✅ 모델이 이미 존재합니다: {local_path}")
        return str(local_path)

    print(f"⬇️  모델 다운로드 중... ({MODEL_REPO}/{MODEL_FILE})")
    print("   Q5_K_M ≈ 19.4GB — 네트워크 상태에 따라 시간이 걸릴 수 있습니다.")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"✅ 다운로드 완료: {path}")
    return path


# ──────────────────────────────────────────────
# 2. 모델 로드 (듀얼 GPU 설정)
# ──────────────────────────────────────────────
def load_model(model_path: str):
    from llama_cpp import Llama

    print("🚀 모델 로딩 중 (2x RTX 5060 Ti 분할)...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,           # 전체 레이어 GPU 오프로드
        n_ctx=8192,                # 컨텍스트 길이 (VRAM 여유에 따라 조정)
        n_batch=512,               # 배치 크기
        tensor_split=[0.5, 0.5],   # GPU 0, GPU 1 균등 분할
        flash_attn=True,           # Flash Attention 활성화
        verbose=False,
    )
    print("✅ 모델 로드 완료!")
    return llm


# ──────────────────────────────────────────────
# 3. 채팅 함수
# ──────────────────────────────────────────────
def chat(llm, user_message: str, system_prompt: str = None, enable_thinking: bool = True):
    """단일 턴 채팅. thinking 모드 지원."""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        # Qwen3.5 thinking 모드: <think>...</think> 후 답변 생성
    )

    assistant_msg = response["choices"][0]["message"]["content"]
    return assistant_msg


def parse_thinking(response: str) -> tuple[str, str]:
    """<think>...</think> 블록과 최종 답변을 분리"""
    if "<think>" in response and "</think>" in response:
        think_start = response.index("<think>") + len("<think>")
        think_end = response.index("</think>")
        thinking = response[think_start:think_end].strip()
        answer = response[think_end + len("</think>"):].strip()
        return thinking, answer
    return "", response


# ──────────────────────────────────────────────
# 4. 인터랙티브 채팅 루프
# ──────────────────────────────────────────────
def interactive_chat(llm):
    system_prompt = "You are a helpful assistant. Think step by step."
    history = []

    print("\n" + "=" * 60)
    print("🤖 Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled")
    print("   Q5_K_M | 2x RTX 5060 Ti")
    print("=" * 60)
    print("명령어: /quit (종료) | /clear (대화 초기화) | /think off|on (사고과정 표시)")
    print("=" * 60 + "\n")

    show_thinking = True

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("👋 종료합니다.")
            break
        if user_input == "/clear":
            history.clear()
            print("🗑️  대화 기록 초기화됨.\n")
            continue
        if user_input.startswith("/think"):
            arg = user_input.split()[-1] if len(user_input.split()) > 1 else ""
            show_thinking = arg != "off"
            print(f"💭 사고과정 표시: {'ON' if show_thinking else 'OFF'}\n")
            continue

        # 멀티턴 히스토리 구성
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        print("🤔 생성 중...", end="", flush=True)

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
        )

        assistant_msg = response["choices"][0]["message"]["content"]
        thinking, answer = parse_thinking(assistant_msg)

        print("\r" + " " * 20 + "\r", end="")  # "생성 중..." 지우기

        if thinking and show_thinking:
            print(f"💭 Thinking:\n{thinking}\n")
        print(f"🤖 Assistant: {answer}\n")

        # 히스토리에는 최종 답변만 저장 (모델 카드 권장사항)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        # 히스토리가 너무 길면 오래된 것 제거 (컨텍스트 관리)
        if len(history) > 20:
            history = history[-16:]


# ──────────────────────────────────────────────
# 5. 단발성 추론 예시 (스크립트 용도)
# ──────────────────────────────────────────────
def single_inference_example(llm):
    """단발성 추론 예시 — 스크립트에서 바로 사용 가능"""
    prompt = "대한민국의 에너지 전환 정책에서 VPP(가상발전소)의 역할과 향후 전망을 분석해줘."

    print(f"📝 Prompt: {prompt}\n")
    response = chat(llm, prompt, system_prompt="You are a helpful energy domain expert.")
    thinking, answer = parse_thinking(response)

    if thinking:
        print(f"💭 Thinking:\n{thinking}\n")
    print(f"🤖 Answer:\n{answer}")


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
