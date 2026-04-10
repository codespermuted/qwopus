"""Core data models for the harness."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a tool the LLM can call."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A tool call parsed from the LLM output."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool."""
    name: str
    output: str
    success: bool = True


@dataclass
class UsageSummary:
    """Tracks token usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TurnResult:
    """Result of a single conversation turn."""
    user_prompt: str
    assistant_response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: UsageSummary = field(default_factory=UsageSummary)
    stop_reason: str = "completed"  # completed | tool_use | max_turns | error


@dataclass(frozen=True)
class PermissionDenial:
    """A denied tool call."""
    tool_name: str
    reason: str
