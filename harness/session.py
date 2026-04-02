"""세션 관리 — 대화 기록 및 영속화."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SESSION_DIR = Path.home() / ".qwopus" / "sessions"

# 한국어/영어 혼합 텍스트의 대략적인 문자당 토큰 추정치
CHARS_PER_TOKEN = 3


@dataclass
class Session:
    """대화 기록과 토큰 추적을 관리한다."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, str]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    max_context_tokens: int = 14000  # n_ctx 이하로 여유분 확보

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_for_context(self, system_prompt: str) -> list[dict]:
        """LLM에 전달할 메시지 목록을 구성하며, 오래된 메시지를 잘라낸다."""
        system_tokens = self._estimate_tokens(system_prompt)
        budget = self.max_context_tokens - system_tokens

        # 뒤에서부터 역순으로 메시지를 추가하되 예산 초과 시 중단
        selected = []
        used = 0
        for msg in reversed(self.messages):
            msg_tokens = self._estimate_tokens(msg["content"])
            if used + msg_tokens > budget:
                break
            selected.append(msg)
            used += msg_tokens

        selected.reverse()

        result = [{"role": "system", "content": system_prompt}]
        result.extend(selected)
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """대략적인 토큰 수 추정."""
        return max(1, len(text) // CHARS_PER_TOKEN)

    def save(self):
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSION_DIR / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "messages": self.messages,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, session_id: str) -> Session:
        path = SESSION_DIR / f"{session_id}.json"
        data = json.loads(path.read_text())
        session = cls(
            session_id=data["session_id"],
            messages=data["messages"],
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
        )
        return session

    @classmethod
    def list_sessions(cls) -> list[str]:
        if not SESSION_DIR.exists():
            return []
        return [p.stem for p in SESSION_DIR.glob("*.json")]
