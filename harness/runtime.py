"""Conversation runtime — the core turn loop with tool dispatch."""
from __future__ import annotations

import json
import re
from typing import Callable

from .engine import chat_completion, strip_thinking, get_n_ctx
from .indexer import build_project_index
from .models import ToolCall, ToolResult, TurnResult, UsageSummary
from .session import Session
from .tools import (
    TOOL_REGISTRY,
    execute_tool,
    get_tool_definitions_for_prompt,
)
from . import ui

# Maximum tool-use iterations per turn to prevent infinite loops
# 도구 반복 최대 횟수 — 너무 높으면 hallucination, 너무 낮으면 복잡한 작업 불가
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT_TEMPLATE = """\
You are Qwopus, a powerful AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, file manipulation, \
running commands, and answering questions about codebases.

Current working directory: {cwd}

# Project Index
Below is a summary of files in the current project. Use this to navigate efficiently \
without reading every file. Use Grep/Glob to find specific content, then FileRead only \
what you need.

{project_index}

# Available Tools

You can use tools by responding with a JSON block in this exact format:
```tool
{{"tool": "<ToolName>", "arguments": {{...}}}}
```

You may call multiple tools in sequence. After each tool call, you will receive the \
result and can continue your response.

When you are done and have no more tools to call, just respond with normal text.

{tool_definitions}

# How to behave

## Accuracy
- NEVER fabricate information, file contents, or tool results. If you don't know, say so.
- ALWAYS read a file before referencing its contents. Do not guess what's in a file.
- Use the Project Index above to know what files exist. Don't scan the whole project.
- Use Grep to find specific content, then FileRead only the relevant lines (use offset/limit).
- Verify your claims by using tools. If you say "this file has X", show it with FileRead.

## Focus
- Match the scope of your action to what was asked:
  - "설명해줘" / "explain" → read and explain only, do NOT create or modify files.
  - "만들어줘" / "create" / "build" → you CAN proactively create files and code.
  - "고쳐줘" / "fix" → read, diagnose, then fix.
- When reading the codebase, read only the files you need. Don't read every single file.
- Stay on topic. If the user asks about X, don't start building Y.
- Be concise. Lead with the answer, not the reasoning.

## Proactivity
- You CAN suggest useful improvements, tools, or features if they are relevant.
- You CAN create files when the user's request implies it (e.g. "이거 추가해줘").
- But always make sure your suggestion is relevant to what the user is working on.

## Safety
- If a command is dangerous (rm -rf, git push --force, etc.), explain and ask first.
- Do NOT include internal reasoning ("The user wants...", "Let me think...") in your response.
"""

# Regex to extract ```tool ... ``` blocks from LLM output
TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(\{.*?\})\s*\n```", re.DOTALL)


def build_system_prompt(cwd: str) -> str:
    index = build_project_index(cwd)
    return SYSTEM_PROMPT_TEMPLATE.format(
        cwd=cwd,
        project_index=index,
        tool_definitions=get_tool_definitions_for_prompt(),
    )


def parse_tool_calls(text: str) -> list[ToolCall]:
    """LLM 응답에서 도구 호출을 추출한다. 여러 포맷을 지원."""
    calls = []

    # 1차: ```tool ``` 블록
    for match in TOOL_BLOCK_RE.finditer(text):
        tc = _try_parse_tool_json(match.group(1))
        if tc:
            calls.append(tc)

    # 2차: ```tool 없이 JSON만 있는 경우 (모델이 포맷을 어긴 경우)
    if not calls:
        for match in re.finditer(r'\{\s*"tool"\s*:\s*"(\w+)".*?\}', text, re.DOTALL):
            tc = _try_parse_tool_json(match.group(0))
            if tc:
                calls.append(tc)

    return calls


def _try_parse_tool_json(raw: str) -> ToolCall | None:
    """JSON 문자열을 ToolCall로 파싱 시도."""
    try:
        data = json.loads(raw)
        name = data.get("tool", "")
        args = data.get("arguments", {})
        if name and name in TOOL_REGISTRY:
            return ToolCall(name=name, arguments=args)
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def strip_tool_blocks(text: str) -> str:
    """Remove tool call blocks from text for display."""
    return TOOL_BLOCK_RE.sub("", text).strip()


class ConversationRuntime:
    """Manages the conversation loop: user -> LLM -> tools -> LLM -> ..."""

    def __init__(self, cwd: str, session: Session | None = None):
        self.cwd = cwd
        self.session = session or Session()
        self.system_prompt = build_system_prompt(cwd)

    def run_turn(self, user_input: str) -> TurnResult:
        """Execute a full turn: submit user message, handle tool calls in a loop."""
        # Sync context limit with actual model n_ctx (set after first load)
        try:
            self.session.max_context_tokens = get_n_ctx() - 2048  # headroom for output
        except Exception:
            pass

        self.session.add_user_message(user_input)

        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        final_text = ""
        prev_tool_sig = ""  # 반복 감지용

        for round_idx in range(MAX_TOOL_ROUNDS):
            # Build messages for the LLM
            messages = self.session.get_messages_for_context(self.system_prompt)

            # Run inference with spinner
            with ui.tool_spinner("model", "thinking..."):
                response = chat_completion(messages, max_tokens=4096, temperature=0.3)

            raw_content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})

            self.session.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.session.total_completion_tokens += usage.get("completion_tokens", 0)

            # Strip thinking
            thinking, content = strip_thinking(raw_content)
            if thinking:
                ui.print_thinking(thinking)

            # Check for tool calls
            tool_calls = parse_tool_calls(content)
            display_text = strip_tool_blocks(content)

            if display_text:
                final_text += display_text + "\n"

            if not tool_calls:
                # No tools requested — turn is done
                if display_text:
                    ui.print_response(display_text)
                self.session.add_assistant_message(content)
                return TurnResult(
                    user_prompt=user_input,
                    assistant_response=final_text.strip(),
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    usage=UsageSummary(
                        self.session.total_prompt_tokens,
                        self.session.total_completion_tokens,
                    ),
                    stop_reason="completed",
                )

            # 반복 감지: 같은 도구를 같은 인자로 다시 호출하면 루프 종료
            current_sig = str([(tc.name, tc.arguments) for tc in tool_calls])
            if current_sig == prev_tool_sig:
                ui.print_warning("동일한 도구 호출 반복 감지 — 루프를 종료합니다.")
                if display_text:
                    ui.print_response(display_text)
                self.session.add_assistant_message(content)
                return TurnResult(
                    user_prompt=user_input,
                    assistant_response=final_text.strip(),
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    usage=UsageSummary(
                        self.session.total_prompt_tokens,
                        self.session.total_completion_tokens,
                    ),
                    stop_reason="completed",
                )
            prev_tool_sig = current_sig

            # 도구 실행 및 결과 피드백
            tool_output_parts = []
            for tc in tool_calls:
                summary = _summarize_args(tc)
                ui.print_tool_call(tc.name, summary)

                with ui.tool_spinner(tc.name, summary):
                    result = execute_tool(tc, self.cwd, ui.confirm)

                all_tool_calls.append(tc)
                all_tool_results.append(result)

                ui.print_tool_result(tc.name, result.output, result.success)

                # 컨텍스트 절약을 위해 도구 출력 길이 제한
                output_for_context = result.output
                if len(output_for_context) > 1500:
                    output_for_context = (
                        output_for_context[:1000]
                        + "\n\n... (잘림 — 전체 내용이 필요하면 FileRead로 다시 읽으세요) ...\n\n"
                        + output_for_context[-400:]
                    )
                tool_output_parts.append(
                    f"[Tool Result: {result.name}]\n{output_for_context}"
                )

            # Don't display intermediate text when tools are being called
            # — the final response will come in the next round

            # Add assistant message (with tool calls) and tool results to history
            self.session.add_assistant_message(content)
            self.session.add_user_message(
                "Tool results:\n\n" + "\n\n".join(tool_output_parts)
            )

        # Exceeded max tool rounds
        ui.print_warning("Max tool rounds reached.")
        self.session.add_assistant_message("(max tool rounds reached)")
        return TurnResult(
            user_prompt=user_input,
            assistant_response=final_text.strip(),
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            usage=UsageSummary(
                self.session.total_prompt_tokens,
                self.session.total_completion_tokens,
            ),
            stop_reason="max_turns",
        )


def _summarize_args(tc: ToolCall) -> str:
    """Short summary of tool arguments for display."""
    args = tc.arguments
    if tc.name == "Bash":
        return args.get("command", "")[:80]
    elif tc.name in ("FileRead", "FileWrite", "FileEdit"):
        return args.get("path", "")
    elif tc.name == "Glob":
        return args.get("pattern", "")
    elif tc.name == "Grep":
        return f"/{args.get('pattern', '')}/ in {args.get('path', '.')}"
    return json.dumps(args)[:80]
