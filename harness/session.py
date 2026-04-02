"""Session management — conversation history and persistence."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SESSION_DIR = Path.home() / ".qwopus" / "sessions"


@dataclass
class Session:
    """Manages conversation history and token tracking."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, str]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    max_history_messages: int = 40  # Keep last N messages for context

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._compact()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._compact()

    def get_messages_for_context(self, system_prompt: str) -> list[dict]:
        """Build the messages list for the LLM, with system prompt."""
        result = [{"role": "system", "content": system_prompt}]
        result.extend(self.messages)
        return result

    def _compact(self):
        """Keep only the last N messages to stay within context limits."""
        if len(self.messages) > self.max_history_messages:
            self.messages = self.messages[-self.max_history_messages:]

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
