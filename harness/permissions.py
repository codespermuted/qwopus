"""도구 실행 권한 시스템."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# bash 실행 전 사용자 확인이 필요한 위험 패턴
DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"\brm\s+-r\b"),
    re.compile(r"\bgit\s+push\b.*--force"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+checkout\s+--\s"),
    re.compile(r"\bgit\s+clean\s+-f"),
    re.compile(r"\bgit\s+branch\s+-D\b"),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bkill\s+-9\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r">\s*/dev/sd"),
]


@dataclass(frozen=True)
class ToolPermissionContext:
    """허용되는 도구를 제어하는 컨텍스트."""
    deny_names: frozenset[str] = field(default_factory=frozenset)
    deny_prefixes: tuple[str, ...] = ()

    def blocks(self, tool_name: str) -> bool:
        lower = tool_name.lower()
        if lower in {n.lower() for n in self.deny_names}:
            return True
        return any(lower.startswith(p.lower()) for p in self.deny_prefixes)


def check_bash_safety(command: str) -> str | None:
    """명령어가 위험해 보이면 경고 메시지를 반환하고, 그렇지 않으면 None을 반환한다."""
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return f"위험한 패턴 감지: {pattern.pattern}"
    return None
