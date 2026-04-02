"""Permission system for tool execution."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Patterns that require user confirmation before bash execution
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
    """Controls which tools are allowed."""
    deny_names: frozenset[str] = field(default_factory=frozenset)
    deny_prefixes: tuple[str, ...] = ()

    def blocks(self, tool_name: str) -> bool:
        lower = tool_name.lower()
        if lower in {n.lower() for n in self.deny_names}:
            return True
        return any(lower.startswith(p.lower()) for p in self.deny_prefixes)


def check_bash_safety(command: str) -> str | None:
    """Return a warning message if the command looks dangerous, else None."""
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return f"Dangerous pattern detected: {pattern.pattern}"
    return None
