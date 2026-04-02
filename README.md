# 🐙 Qwopus

**Local AI coding agent** powered by Qwen3.5-27B, distilled from Claude 4.6 Opus reasoning.

Claude Code처럼 동작하지만, **완전히 로컬**에서 실행됩니다. API 키 없음, 클라우드 없음, 비용 없음.

> **"입만 있던 챗봇에 손(도구)을 달았습니다."**
> 파일을 읽고, 수정하고, 명령어를 실행하는 코딩 에이전트를 로컬 GPU에서.

---

## Demo

```
╔══════════════════════════════════════════════════════════════╗
║  🐙 Qwopus — Local AI Coding Agent                         ║
║     Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled        ║
║     Q5_K_M · Auto GPU Detection · llama.cpp                 ║
╠══════════════════════════════════════════════════════════════╣
║  /help for commands · /quit to exit · !cmd for shell        ║
╚══════════════════════════════════════════════════════════════╝

📁 Working directory: /home/user/my-project

👤 You: 이 프로젝트에 어떤 파일들이 있어?

🔧 Bash: ls -la
   ✅ total 28 ...

현재 디렉토리에 있는 파일은 다음과 같습니다:
  - src/       소스 코드
  - tests/     테스트 코드
  - README.md  프로젝트 설명
  ...
```

## Requirements

| 항목 | 최소 사양 |
|------|----------|
| **GPU** | NVIDIA GPU(s), 총 VRAM **18GB 이상** |
| **CUDA** | 12.0+ |
| **Python** | 3.10+ |
| **OS** | Linux (Ubuntu 22.04+ 테스트 완료) |
| **디스크** | ~20GB (모델 저장 공간) |

**GPU 예시:**
- RTX 3090 (24GB) — 단일 GPU
- RTX 4090 (24GB) — 단일 GPU
- RTX 5060 Ti 16GB x2 — 멀티 GPU (자동 분할)
- RTX 4070 Ti Super 16GB x2 — 멀티 GPU

> GPU가 부족하면 시작 시 명확한 에러 메시지와 대안을 안내합니다.

---

## Install

### 방법 1: pip (권장)

```bash
# 1. Qwopus 설치
pip install git+https://github.com/codespermuted/qwopus.git

# 2. llama-cpp-python CUDA 빌드 (GPU 가속 필수)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 방법 2: 소스에서 설치

```bash
git clone https://github.com/codespermuted/qwopus.git
cd qwopus
pip install -e .

# CUDA 빌드
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 모델 다운로드

모델(~19.4GB)은 **첫 실행 시 자동으로 다운로드**됩니다.
`~/models/` 디렉토리에 저장되며, 이후 재다운로드 없이 바로 로드됩니다.

수동 다운로드도 가능합니다:
```bash
huggingface-cli download \
  mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF \
  Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf \
  --local-dir ~/models/
```

---

## Usage

```bash
# 아무 프로젝트 폴더에서 실행 — Claude Code처럼
cd your-project/
qwopus

# 원샷 모드 — 질문 하나만 던지고 결과 받기
qwopus "이 프로젝트의 구조를 설명해줘"

# 작업 디렉토리 지정
qwopus --cwd /path/to/project

# 세션 이어서 하기
qwopus --resume <session_id>
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | 도움말 표시 |
| `/status` | 세션 정보 (메시지 수, 토큰 사용량) |
| `/clear` | 대화 기록 초기화 |
| `/compact` | 오래된 메시지 정리 |
| `/save` | 세션을 디스크에 저장 |
| `/sessions` | 저장된 세션 목록 |
| `/resume <id>` | 저장된 세션 재개 |
| `/quit` | 종료 |
| `!command` | 셸 명령어 직접 실행 |

### Tools

Qwopus가 자율적으로 사용하는 도구:

| Tool | Description |
|------|-------------|
| **Bash** | 셸 명령어 실행 |
| **FileRead** | 파일 읽기 (줄 번호 포함) |
| **FileWrite** | 파일 생성/덮어쓰기 |
| **FileEdit** | 파일 수정 (정확한 문자열 치환) |
| **Glob** | 패턴으로 파일 찾기 |
| **Grep** | 정규식으로 파일 내용 검색 |

---

## How It Works

```
사용자 입력
  ↓
시스템 프롬프트 + 도구 정의 + 대화 히스토리 구성
  ↓
로컬 LLM 추론 (Qwen3.5-27B, llama.cpp)
  ↓
도구 호출 파싱 (```tool {...}``` 블록)
  ↓
도구 실행 (Bash, FileRead, etc.) ←── 권한 검사
  ↓
결과를 컨텍스트에 추가
  ↓
LLM 재추론 (도구 결과 반영)
  ↓
반복 (최대 10라운드) 또는 최종 답변 출력
```

### GPU Auto-Detection

시작 시 GPU를 자동으로 감지하고 최적 설정을 적용합니다:

- **GPU 없음** → 에러 + 안내 메시지
- **VRAM 부족** → 에러 + 대안 제시 (더 작은 양자화/모델)
- **단일 GPU** (≥18GB) → 단일 GPU 모드
- **멀티 GPU** → VRAM 비율에 따라 자동 `tensor_split`
- **여유 VRAM에 따라** `n_ctx` 자동 조절 (4K ~ 16K)

### Safety

- `rm -rf`, `git push --force` 등 위험한 명령은 **사용자 확인 필요**
- 도구별 권한 차단 가능 (deny by name/prefix)
- 모든 실행은 **로컬 머신 내**에서만 이루어짐

---

## Architecture

```
harness/
├── cli.py           # 진입점 — REPL, 인자 파싱
├── engine.py        # LLM 로드 및 추론 (llama-cpp-python)
├── gpu.py           # GPU 감지, VRAM 검증, 자동 설정
├── runtime.py       # 턴 루프 — 사용자 → LLM → 도구 → LLM 반복
├── tools.py         # 6개 내장 도구 정의 및 실행
├── commands.py      # /slash 명령어 처리
├── permissions.py   # 위험 명령 탐지, 도구 차단
├── session.py       # 대화 히스토리, JSON 영속화
└── models.py        # 데이터 모델 (ToolCall, ToolResult, etc.)
```

---

## Credits & Acknowledgments

이 프로젝트는 여러 오픈소스 프로젝트와 연구의 결과물 위에 만들어졌습니다.

### Core Model

- **[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3-27B)** by [Qwen Team (Alibaba)](https://github.com/QwenLM/Qwen3)
  - Apache 2.0 License
  - 기반 LLM 아키텍처

- **[Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF)** — GGUF quantization by [mradermacher](https://huggingface.co/mradermacher)
  - Claude 4.6 Opus의 reasoning 능력을 증류(distillation)한 모델

### Inference Engine

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** by Georgi Gerganov et al.
  - MIT License
  - 로컬 GGUF 모델 추론 엔진

- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** by Andrei Betlen
  - MIT License
  - llama.cpp의 Python 바인딩

### Architecture Inspiration

- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** by [Anthropic](https://www.anthropic.com/)
  - AI 코딩 에이전트 하네스의 아키텍처 참고
  - 도구 시스템, REPL 루프, 세션 관리 등의 설계 패턴 참조

- **[Claw Code](https://github.com/instructkr/claw-code)** by [instructkr](https://github.com/instructkr)
  - Claude Code를 Python/Rust로 재구현한 클린룸 프로젝트
  - 하네스 아키텍처 (도구 레지스트리, 턴 루프, 권한 시스템) 연구 참고

### Other Dependencies

- **[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)** — 모델 다운로드
  - Apache 2.0 License

---

## Disclaimer

- 이 프로젝트는 Anthropic, Alibaba/Qwen과 **무관한 개인 프로젝트**입니다.
- Claude Code의 소스 코드를 직접 복사하지 않았으며, 공개된 아키텍처 패턴을 참고하여 독립적으로 구현했습니다.
- 모델의 출력은 항상 검증이 필요하며, 프로덕션 환경에서의 사용은 사용자의 책임입니다.

## License

MIT License — see [LICENSE](LICENSE) file.
