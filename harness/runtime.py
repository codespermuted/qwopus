"""Conversation runtime — the core turn loop with tool dispatch."""
from __future__ import annotations

import json
import re
from typing import Callable

from .engine import chat_completion, strip_thinking, get_n_ctx
from .models import ToolCall, ToolResult, TurnResult, UsageSummary
from .session import Session
from .tools import (
    TOOL_REGISTRY,
    execute_tool,
    get_tool_definitions_for_prompt,
)
from . import ui

# Maximum tool-use iterations per turn to prevent infinite loops
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT_TEMPLATE = """\
You are Qwopus, an AI coding assistant running locally on the user's machine.
You help with software engineering tasks: reading code, debugging, running commands, \
and answering questions about codebases.

Current working directory: {cwd}

# Available Tools

You can use tools by responding with a JSON block in this exact format:
```tool
{{"tool": "<ToolName>", "arguments": {{...}}}}
```

You may call multiple tools in sequence. After each tool call, you will receive the \
result and can continue your response.

When you are done and have no more tools to call, just respond with normal text.

{tool_definitions}

# STRICT RULES — NEVER VIOLATE THESE

1. ONLY do what the user explicitly asked. Do NOT add extra tasks, features, or files.
2. NEVER create, write, or modify files unless the user specifically requests it.
3. NEVER fabricate information. If you don't know, say so.
4. NEVER generate code that the user didn't ask for.
5. If asked to "explain" or "describe", ONLY read and explain — do NOT create anything.
6. Do NOT include internal reasoning ("The user wants...", "Let me...") in your response.
7. Be concise. Answer directly.
8. When reading files, read only what's needed — don't read every file in the project.
9. If a command is dangerous, explain why before running it.
10. Stay focused on the user's question. Do NOT go off-topic.
"""

# Regex to extract ```tool ... ``` blocks from LLM output
TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(\{.*?\})\s*\n```", re.DOTALL)


def build_system_prompt(cwd: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        cwd=cwd,
        tool_definitions=get_tool_definitions_for_prompt(),
    )


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from LLM response text."""
    calls = []
    for match in TOOL_BLOCK_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("tool", "")
            args = data.get("arguments", {})
            if name and name in TOOL_REGISTRY:
                calls.append(ToolCall(name=name, arguments=args))
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


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

            # Execute tools and feed results back
            tool_output_parts = []
            for tc in tool_calls:
                summary = _summarize_args(tc)
                ui.print_tool_call(tc.name, summary)

                with ui.tool_spinner(tc.name, summary):
                    result = execute_tool(tc, self.cwd, ui.confirm)

                all_tool_calls.append(tc)
                all_tool_results.append(result)

                ui.print_tool_result(tc.name, result.output, result.success)

                # Truncate tool output for context to avoid blowing up tokens
                output_for_context = result.output
                if len(output_for_context) > 3000:
                    output_for_context = (
                        output_for_context[:2000]
                        + "\n\n... (truncated) ...\n\n"
                        + output_for_context[-800:]
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
