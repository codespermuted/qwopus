"""Session management — conversation history and persistence."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SESSION_DIR = Path.home() / ".qwopus" / "sessions"

# Rough chars-per-token estimate for mixed CJK/English text
CHARS_PER_TOKEN = 3


@dataclass
class Session:
    """Manages conversation history and token tracking."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, str]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    max_context_tokens: int = 14000  # Leave headroom below n_ctx

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_for_context(self, system_prompt: str) -> list[dict]:
        """Build the messages list for the LLM, trimming old messages to fit."""
        system_tokens = self._estimate_tokens(system_prompt)
        budget = self.max_context_tokens - system_tokens

        # Walk backwards, adding messages until budget is exhausted
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
        """Rough token estimate."""
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
