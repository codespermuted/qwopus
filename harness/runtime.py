"""Conversation runtime — the core turn loop with tool dispatch."""
from __future__ import annotations

import json
import re
from typing import Callable

from .engine import chat_completion, strip_thinking
from .models import ToolCall, ToolResult, TurnResult, UsageSummary
from .session import Session
from .tools import (
    TOOL_REGISTRY,
    execute_tool,
    get_tool_definitions_for_prompt,
)

# Maximum tool-use iterations per turn to prevent infinite loops
MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT_TEMPLATE = """\
You are Qwopus, a powerful AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, file manipulation, \
running commands, and answering questions about codebases.

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

# Guidelines
- Read files before editing them.
- Use Bash for system commands, Grep/Glob for searching.
- Be concise. Lead with the answer.
- If a command is dangerous, explain why before running it.
- Do NOT fabricate tool results. Always actually call the tool.
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
    """Manages the conversation loop: user → LLM → tools → LLM → ..."""

    def __init__(self, cwd: str, session: Session | None = None, confirm_fn: Callable | None = None):
        self.cwd = cwd
        self.session = session or Session()
        self.system_prompt = build_system_prompt(cwd)
        self.confirm_fn = confirm_fn or (lambda msg: input(msg).strip().lower() in ("y", "yes"))

    def run_turn(self, user_input: str) -> TurnResult:
        """Execute a full turn: submit user message, handle tool calls in a loop."""
        self.session.add_user_message(user_input)

        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        final_text = ""

        for round_idx in range(MAX_TOOL_ROUNDS):
            # Build messages for the LLM
            messages = self.session.get_messages_for_context(self.system_prompt)

            # Run inference
            response = chat_completion(messages, max_tokens=4096, temperature=0.3)
            raw_content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})

            self.session.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.session.total_completion_tokens += usage.get("completion_tokens", 0)

            # Strip thinking
            thinking, content = strip_thinking(raw_content)
            if thinking:
                print(f"\n💭 Thinking:\n{thinking}\n")

            # Check for tool calls
            tool_calls = parse_tool_calls(content)
            display_text = strip_tool_blocks(content)

            if display_text:
                print(display_text)
                final_text += display_text + "\n"

            if not tool_calls:
                # No tools requested — turn is done
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
                print(f"\n🔧 {tc.name}: {_summarize_args(tc)}")
                result = execute_tool(tc, self.cwd, self.confirm_fn)
                all_tool_calls.append(tc)
                all_tool_results.append(result)

                status = "✅" if result.success else "❌"
                # Show truncated output
                preview = result.output[:300] + "..." if len(result.output) > 300 else result.output
                print(f"   {status} {preview}")

                tool_output_parts.append(
                    f"[Tool Result: {result.name}]\n{result.output}"
                )

            # Add assistant message (with tool calls) and tool results to history
            self.session.add_assistant_message(content)
            self.session.add_user_message(
                "Tool results:\n\n" + "\n\n".join(tool_output_parts)
            )

        # Exceeded max tool rounds
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
