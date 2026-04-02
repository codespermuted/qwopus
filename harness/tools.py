"""Built-in tool definitions and execution."""
from __future__ import annotations

import glob as glob_mod
import json
import os
import subprocess
from pathlib import Path

from .models import ToolCall, ToolDefinition, ToolResult
from .permissions import check_bash_safety

# ── Tool Definitions ─────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolDefinition] = {}


def _register(name: str, description: str, parameters: dict):
    TOOL_REGISTRY[name] = ToolDefinition(name=name, description=description, parameters=parameters)


_register("Bash", "Execute a shell command and return its output.", {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "The shell command to execute."}
    },
    "required": ["command"],
})

_register("FileRead", "Read a file and return its contents.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Absolute path to the file."},
        "offset": {"type": "integer", "description": "Line number to start reading from (0-based)."},
        "limit": {"type": "integer", "description": "Number of lines to read."},
    },
    "required": ["path"],
})

_register("FileWrite", "Write content to a file (creates or overwrites).", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Absolute path to the file."},
        "content": {"type": "string", "description": "Content to write."},
    },
    "required": ["path", "content"],
})

_register("FileEdit", "Replace an exact string in a file.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Absolute path to the file."},
        "old_string": {"type": "string", "description": "Exact text to find."},
        "new_string": {"type": "string", "description": "Text to replace with."},
    },
    "required": ["path", "old_string", "new_string"],
})

_register("Glob", "Find files matching a glob pattern.", {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')."},
        "path": {"type": "string", "description": "Directory to search in."},
    },
    "required": ["pattern"],
})

_register("Grep", "Search file contents with a regex pattern.", {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regex pattern to search for."},
        "path": {"type": "string", "description": "File or directory to search in."},
        "glob": {"type": "string", "description": "Glob to filter files (e.g. '*.py')."},
    },
    "required": ["pattern"],
})


# ── Tool Execution ───────────────────────────────────────────

def execute_tool(call: ToolCall, cwd: str, confirm_fn=None) -> ToolResult:
    """Execute a tool call and return the result."""
    try:
        if call.name == "Bash":
            return _exec_bash(call, cwd, confirm_fn)
        elif call.name == "FileRead":
            return _exec_file_read(call)
        elif call.name == "FileWrite":
            return _exec_file_write(call)
        elif call.name == "FileEdit":
            return _exec_file_edit(call)
        elif call.name == "Glob":
            return _exec_glob(call, cwd)
        elif call.name == "Grep":
            return _exec_grep(call, cwd)
        else:
            return ToolResult(name=call.name, output=f"Unknown tool: {call.name}", success=False)
    except Exception as e:
        return ToolResult(name=call.name, output=f"Error: {e}", success=False)


def _exec_bash(call: ToolCall, cwd: str, confirm_fn) -> ToolResult:
    command = call.arguments.get("command", "")
    warning = check_bash_safety(command)
    if warning and confirm_fn:
        if not confirm_fn(f"⚠️  {warning}\nCommand: {command}\nAllow? [y/N]: "):
            return ToolResult(name="Bash", output="Command denied by user.", success=False)

    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=cwd, timeout=120,
    )
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    if not output:
        output = "(no output)"
    # Truncate very long output
    if len(output) > 8000:
        output = output[:4000] + "\n\n... (truncated) ...\n\n" + output[-2000:]
    return ToolResult(name="Bash", output=output, success=result.returncode == 0)


def _exec_file_read(call: ToolCall) -> ToolResult:
    path = call.arguments["path"]
    offset = call.arguments.get("offset", 0)
    limit = call.arguments.get("limit", 2000)
    with open(path, "r") as f:
        lines = f.readlines()
    selected = lines[offset:offset + limit]
    numbered = "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(selected))
    return ToolResult(name="FileRead", output=numbered or "(empty file)")


def _exec_file_write(call: ToolCall) -> ToolResult:
    path = call.arguments["path"]
    content = call.arguments["content"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return ToolResult(name="FileWrite", output=f"Wrote {len(content)} bytes to {path}")


def _exec_file_edit(call: ToolCall) -> ToolResult:
    path = call.arguments["path"]
    old = call.arguments["old_string"]
    new = call.arguments["new_string"]
    with open(path, "r") as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        return ToolResult(name="FileEdit", output="old_string not found in file.", success=False)
    if count > 1:
        return ToolResult(name="FileEdit", output=f"old_string found {count} times; must be unique.", success=False)
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    return ToolResult(name="FileEdit", output=f"Edited {path}")


def _exec_glob(call: ToolCall, cwd: str) -> ToolResult:
    pattern = call.arguments["pattern"]
    search_path = call.arguments.get("path", cwd)
    full_pattern = os.path.join(search_path, pattern)
    matches = sorted(glob_mod.glob(full_pattern, recursive=True))
    if not matches:
        return ToolResult(name="Glob", output="No matches found.")
    # Limit output
    if len(matches) > 200:
        matches = matches[:200]
        matches.append(f"... and more (truncated at 200)")
    return ToolResult(name="Glob", output="\n".join(matches))


def _exec_grep(call: ToolCall, cwd: str) -> ToolResult:
    pattern = call.arguments["pattern"]
    search_path = call.arguments.get("path", cwd)
    file_glob = call.arguments.get("glob", "")

    cmd = ["grep", "-rn", "--color=never"]
    if file_glob:
        cmd += ["--include", file_glob]
    cmd += [pattern, search_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout.strip()
    if not output:
        return ToolResult(name="Grep", output="No matches found.")
    lines = output.split("\n")
    if len(lines) > 100:
        output = "\n".join(lines[:100]) + f"\n... ({len(lines)} total matches, showing first 100)"
    return ToolResult(name="Grep", output=output)


def get_tool_definitions_for_prompt() -> str:
    """Format tool definitions for injection into system prompt."""
    parts = []
    for tool in TOOL_REGISTRY.values():
        props = tool.parameters.get("properties", {})
        params_desc = []
        for pname, pdef in props.items():
            req = " (required)" if pname in tool.parameters.get("required", []) else ""
            params_desc.append(f"    - {pname}: {pdef.get('description', '')}{req}")
        params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"
        parts.append(f"## {tool.name}\n{tool.description}\nParameters:\n{params_str}")
    return "\n\n".join(parts)
