"""Conversation runtime — turn loop, tool dispatch, and hook execution."""
from __future__ import annotations

import json
import re

from .config import Settings
from .engine import chat_completion, chat_completion_stream, strip_thinking, get_n_ctx
from .hooks import HookRunner
from .indexer import build_project_index
from .models import ToolCall, ToolResult, TurnResult, UsageSummary
from .session import Session
from .tools import (
    TOOL_REGISTRY,
    execute_tool,
    get_tool_definitions_for_prompt,
)
from . import ui

SYSTEM_PROMPT_TEMPLATE = """\
You are Qwopus, a powerful AI coding assistant running locally on the user's machine.
You help with software engineering tasks: writing code, debugging, file manipulation, \
running commands, and answering questions about codebases.

Current date: {current_date}
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
  - "explain" → read and explain only, do NOT create or modify files.
  - "create" / "build" → you CAN proactively create files and code.
  - "fix" → read, diagnose, then fix.
- When reading the codebase, read only the files you need. Don't read every single file.
- Stay on topic. If the user asks about X, don't start building Y.
- Be concise. Lead with the answer, not the reasoning.

## Proactivity
- You CAN suggest useful improvements, tools, or features if they are relevant.
- You CAN create files when the user's request implies it.
- But always make sure your suggestion is relevant to what the user is working on.

## Safety
- If a command is dangerous (rm -rf, git push --force, etc.), explain and ask first.
- Do NOT include internal reasoning ("The user wants...", "Let me think...") in your response.
"""

# Regex for extracting ```tool ... ``` blocks
TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(\{.*?\})\s*\n```", re.DOTALL)


def build_system_prompt(cwd: str) -> str:
    from datetime import date
    index = build_project_index(cwd)
    return SYSTEM_PROMPT_TEMPLATE.format(
        cwd=cwd,
        current_date=date.today().isoformat(),
        project_index=index,
        tool_definitions=get_tool_definitions_for_prompt(),
    )


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from an LLM response."""
    calls = []
    for match in TOOL_BLOCK_RE.finditer(text):
        tc = _try_parse_tool_json(match.group(1))
        if tc:
            calls.append(tc)
    # Fallback: bare JSON without a ```tool block
    if not calls:
        for match in re.finditer(r'\{\s*"tool"\s*:\s*"(\w+)".*?\}', text, re.DOTALL):
            tc = _try_parse_tool_json(match.group(0))
            if tc:
                calls.append(tc)
    return calls


def _try_parse_tool_json(raw: str) -> ToolCall | None:
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
    return TOOL_BLOCK_RE.sub("", text).strip()


class ConversationRuntime:
    """Manages the conversation loop: user -> LLM -> tool -> LLM -> ..."""

    def __init__(self, cwd: str, session: Session | None = None, settings: Settings | None = None):
        self.cwd = cwd
        self.session = session or Session()
        self.settings = settings or Settings.load()
        self.system_prompt = build_system_prompt(cwd)
        self.hooks = HookRunner(self.settings, cwd)
        self.use_streaming = self.settings.get("ui.streaming", True)
        self.max_tool_rounds = self.settings.get("tools.max_tool_rounds", 8)
        self.tool_output_limit = self.settings.get("tools.tool_output_limit", 1500)

    def run_turn(self, user_input: str) -> TurnResult:
        """Run a turn: user message -> tool loop -> final response."""
        # Align the context limit with the actual n_ctx
        try:
            self.session.max_context_tokens = get_n_ctx() - 2048
        except Exception:
            pass

        # pre_turn hook
        self.hooks.run_pre_turn(user_input)

        self.session.add_user_message(user_input)

        all_tool_calls: list[ToolCall] = []
        all_tool_results: list[ToolResult] = []
        final_text = ""
        prev_tool_sig = ""

        for round_idx in range(self.max_tool_rounds):
            messages = self.session.get_messages_for_context(self.system_prompt)

            # LLM inference — streaming or batch
            if self.use_streaming:
                raw_content = self._run_streaming(messages)
            else:
                raw_content = self._run_batch(messages)

            # Separate out thinking (in streaming mode the UI has already shown it)
            thinking, content = strip_thinking(raw_content)
            if thinking and not self.use_streaming and self.settings.get("ui.show_thinking", True):
                ui.print_thinking(thinking)

            # Parse tool calls
            tool_calls = parse_tool_calls(content)
            display_text = strip_tool_blocks(content)

            if display_text:
                final_text += display_text + "\n"

            # No tool calls means the turn is done
            if not tool_calls:
                # Only the batch mode needs a panel (streaming already printed live)
                if display_text and not self.use_streaming:
                    ui.print_response(display_text)
                self.session.add_assistant_message(content)
                self.hooks.run_post_turn(final_text)
                return self._build_result(user_input, final_text, all_tool_calls, all_tool_results, "completed")

            # Loop detection
            current_sig = str([(tc.name, tc.arguments) for tc in tool_calls])
            if current_sig == prev_tool_sig:
                ui.print_warning("Detected repeated identical tool calls — exiting loop")
                if display_text and not self.use_streaming:
                    ui.print_response(display_text)
                self.session.add_assistant_message(content)
                return self._build_result(user_input, final_text, all_tool_calls, all_tool_results, "completed")
            prev_tool_sig = current_sig

            # Execute tools
            tool_output_parts = []
            for tc in tool_calls:
                summary = _summarize_args(tc)
                ui.print_tool_call(tc.name, summary)

                # pre_tool hook
                hook_result = self.hooks.run_pre_tool(tc.name, tc.arguments)
                if hook_result == "BLOCK":
                    ui.print_warning(f"Blocked by hook: {tc.name}")
                    result = ToolResult(name=tc.name, output="Blocked by hook", success=False)
                else:
                    with ui.tool_spinner(tc.name, summary):
                        result = execute_tool(tc, self.cwd, ui.confirm)

                    # post_tool hook
                    self.hooks.run_post_tool(tc.name, result.output, result.success)

                all_tool_calls.append(tc)
                all_tool_results.append(result)
                ui.print_tool_result(tc.name, result.output, result.success)

                # Truncate for context savings
                output_for_ctx = result.output
                if len(output_for_ctx) > self.tool_output_limit:
                    output_for_ctx = (
                        output_for_ctx[:self.tool_output_limit - 500]
                        + "\n\n... (truncated) ...\n\n"
                        + output_for_ctx[-400:]
                    )
                tool_output_parts.append(f"[Tool Result: {result.name}]\n{output_for_ctx}")

            self.session.add_assistant_message(content)
            self.session.add_user_message("Tool results:\n\n" + "\n\n".join(tool_output_parts))

        ui.print_warning("Reached maximum tool rounds")
        self.session.add_assistant_message("(max tool rounds reached)")
        return self._build_result(user_input, final_text, all_tool_calls, all_tool_results, "max_turns")

    def _run_streaming(self, messages: list[dict]) -> str:
        """Run LLM inference in streaming mode."""
        max_tokens = self.settings.get("model.max_tokens", 4096)
        temperature = self.settings.get("model.temperature", 0.3)
        token_iter = chat_completion_stream(messages, max_tokens=max_tokens, temperature=temperature)
        raw = ui.stream_response(token_iter, strip_thinking)
        return raw

    def _run_batch(self, messages: list[dict]) -> str:
        """Run LLM inference in batch mode."""
        max_tokens = self.settings.get("model.max_tokens", 4096)
        temperature = self.settings.get("model.temperature", 0.3)
        with ui.tool_spinner("model", "thinking..."):
            response = chat_completion(messages, max_tokens=max_tokens, temperature=temperature)
        raw_content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        self.session.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.session.total_completion_tokens += usage.get("completion_tokens", 0)
        return raw_content

    def _build_result(self, user_input, final_text, tool_calls, tool_results, stop_reason) -> TurnResult:
        return TurnResult(
            user_prompt=user_input,
            assistant_response=final_text.strip(),
            tool_calls=tool_calls,
            tool_results=tool_results,
            usage=UsageSummary(self.session.total_prompt_tokens, self.session.total_completion_tokens),
            stop_reason=stop_reason,
        )


def _summarize_args(tc: ToolCall) -> str:
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
