"""하네스의 핵심 데이터 모델."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """LLM이 호출할 수 있는 도구 정의."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """LLM 출력에서 파싱된 도구 호출."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """도구 실행 결과."""
    name: str
    output: str
    success: bool = True


@dataclass
class UsageSummary:
    """토큰 사용량 추적."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TurnResult:
    """대화 한 턴의 결과."""
    user_prompt: str
    assistant_response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: UsageSummary = field(default_factory=UsageSummary)
    stop_reason: str = "completed"  # completed | tool_use | max_turns | error


@dataclass(frozen=True)
class PermissionDenial:
    """거부된 도구 호출."""
    tool_name: str
    reason: str
