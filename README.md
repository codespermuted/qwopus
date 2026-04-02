# 🐙 Qwopus

**로컬 AI 코딩 에이전트** — Qwen3.5-27B 기반, Claude 4.6 Opus reasoning 증류 모델

Claude Code처럼 동작하지만, **완전히 로컬에서** 실행됩니다. API 키 없음. 클라우드 없음. 비용 없음.

---

## 설치

```bash
# 1. Qwopus 설치
pip install git+https://github.com/codespermuted/qwopus.git

# 2. llama-cpp-python CUDA 빌드 (GPU 가속 필수)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

모델(~19.4GB)은 첫 실행 시 자동 다운로드됩니다.

## 시스템 요구사항

- **GPU**: NVIDIA, 총 VRAM **18GB 이상** (예: RTX 3090, 4090, 5060 Ti x2 등)
- **CUDA**: 12.0+
- **Python**: 3.10+
- **OS**: Linux

GPU는 자동 감지되며, VRAM 부족 시 안내 메시지가 나옵니다.

---

## 사용법

```bash
# 아무 프로젝트 폴더에서 실행
cd your-project/
qwopus

# 원샷 모드
qwopus "이 프로젝트 구조 설명해줘"

# 작업 디렉토리 지정
qwopus --cwd /path/to/project
```

### 명령어

`/help`로 전체 목록 확인 가능

| 명령어 | 설명 |
|--------|------|
| `/help` | 도움말 |
| `/status` | 세션 정보 |
| `/clear` | 대화 초기화 |
| `/save` | 세션 저장 |
| `/resume <id>` | 세션 재개 |
| `/quit` | 종료 |
| `!명령어` | 셸 명령어 직접 실행 |

### 도구

Qwopus가 자율적으로 사용하는 도구:

**Bash** · **FileRead** · **FileWrite** · **FileEdit** · **Glob** · **Grep**

---

## 동작 원리

```
사용자 입력 → LLM 추론 → 도구 호출 → 결과 반영 → 재추론 → 답변
```

- 로컬 GPU에서 llama.cpp로 추론
- 멀티 GPU 자동 분할
- 위험 명령은 사용자 확인 필요
- 동일 도구 반복 호출 시 자동 중단 (hallucination 방지)

---

## 출처 및 크레딧

### 모델
- **[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3-27B)** — Qwen Team (Alibaba) · Apache 2.0
- **[GGUF 양자화](https://huggingface.co/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF)** — mradermacher

### 추론 엔진
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — Georgi Gerganov · MIT
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** — Andrei Betlen · MIT

### 아키텍처 참고
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — Anthropic
- **[Claw Code](https://github.com/instructkr/claw-code)** — instructkr

### 기타
- **[Rich](https://github.com/Textualize/rich)** · **[prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)** · **[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)**

---

## 면책 사항

- Anthropic, Alibaba/Qwen과 **무관한 개인 프로젝트**입니다.
- Claude Code 소스를 직접 복사하지 않았으며, 공개된 아키텍처 패턴을 참고하여 독립 구현했습니다.
- 모델 출력은 항상 검증이 필요하며, 프로덕션 사용은 사용자 책임입니다.

## 라이선스

MIT — [LICENSE](LICENSE)
