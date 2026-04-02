"""내장 도구 정의 및 실행."""
from __future__ import annotations

import glob as glob_mod
import json
import os
import subprocess
from pathlib import Path

from .models import ToolCall, ToolDefinition, ToolResult
from .permissions import check_bash_safety

# ── 도구 정의 ─────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolDefinition] = {}


def _register(name: str, description: str, parameters: dict):
    TOOL_REGISTRY[name] = ToolDefinition(name=name, description=description, parameters=parameters)


_register("Bash", "셸 명령어를 실행하고 결과를 반환한다.", {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "실행할 셸 명령어."}
    },
    "required": ["command"],
})

_register("FileRead", "파일을 읽고 내용을 반환한다.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "파일의 절대 경로."},
        "offset": {"type": "integer", "description": "읽기 시작할 줄 번호 (0 기반)."},
        "limit": {"type": "integer", "description": "읽을 줄 수."},
    },
    "required": ["path"],
})

_register("FileWrite", "파일에 내용을 쓴다 (생성 또는 덮어쓰기).", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "파일의 절대 경로."},
        "content": {"type": "string", "description": "쓸 내용."},
    },
    "required": ["path", "content"],
})

_register("FileEdit", "파일에서 정확한 문자열을 치환한다.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "파일의 절대 경로."},
        "old_string": {"type": "string", "description": "찾을 정확한 텍스트."},
        "new_string": {"type": "string", "description": "대체할 텍스트."},
    },
    "required": ["path", "old_string", "new_string"],
})

_register("Glob", "글롭 패턴과 일치하는 파일을 찾는다.", {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "글롭 패턴 (예: '**/*.py')."},
        "path": {"type": "string", "description": "검색할 디렉토리."},
    },
    "required": ["pattern"],
})

_register("Grep", "정규식 패턴으로 파일 내용을 검색한다.", {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "검색할 정규식 패턴."},
        "path": {"type": "string", "description": "검색할 파일 또는 디렉토리."},
        "glob": {"type": "string", "description": "파일 필터링 글롭 (예: '*.py')."},
    },
    "required": ["pattern"],
})


# ── 도구 실행 ─────────────────────────────────────────────────

def execute_tool(call: ToolCall, cwd: str, confirm_fn=None) -> ToolResult:
    """도구 호출을 실행하고 결과를 반환한다."""
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
            return ToolResult(name=call.name, output=f"알 수 없는 도구: {call.name}", success=False)
    except Exception as e:
        return ToolResult(name=call.name, output=f"오류: {e}", success=False)


def _exec_bash(call: ToolCall, cwd: str, confirm_fn) -> ToolResult:
    command = call.arguments.get("command", "")
    warning = check_bash_safety(command)
    if warning and confirm_fn:
        if not confirm_fn(f"⚠️  {warning}\n명령어: {command}\n허용하시겠습니까? [y/N]: "):
            return ToolResult(name="Bash", output="사용자가 명령어를 거부했습니다.", success=False)

    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=cwd, timeout=120,
    )
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    if not output:
        output = "(출력 없음)"
    # 너무 긴 출력 잘라내기
    if len(output) > 8000:
        output = output[:4000] + "\n\n... (잘림) ...\n\n" + output[-2000:]
    return ToolResult(name="Bash", output=output, success=result.returncode == 0)


def _exec_file_read(call: ToolCall) -> ToolResult:
    path = call.arguments["path"]
    offset = call.arguments.get("offset", 0)
    limit = call.arguments.get("limit", 2000)
    with open(path, "r") as f:
        lines = f.readlines()
    selected = lines[offset:offset + limit]
    numbered = "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(selected))
    return ToolResult(name="FileRead", output=numbered or "(빈 파일)")


def _exec_file_write(call: ToolCall) -> ToolResult:
    path = call.arguments["path"]
    content = call.arguments["content"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return ToolResult(name="FileWrite", output=f"{path}에 {len(content)}바이트를 기록했습니다")


def _exec_file_edit(call: ToolCall) -> ToolResult:
    path = call.arguments["path"]
    old = call.arguments["old_string"]
    new = call.arguments["new_string"]
    with open(path, "r") as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        return ToolResult(name="FileEdit", output="파일에서 old_string을 찾을 수 없습니다.", success=False)
    if count > 1:
        return ToolResult(name="FileEdit", output=f"old_string이 {count}번 발견되었습니다. 고유해야 합니다.", success=False)
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    return ToolResult(name="FileEdit", output=f"{path} 수정 완료")


def _exec_glob(call: ToolCall, cwd: str) -> ToolResult:
    pattern = call.arguments["pattern"]
    search_path = call.arguments.get("path", cwd)
    full_pattern = os.path.join(search_path, pattern)
    matches = sorted(glob_mod.glob(full_pattern, recursive=True))
    if not matches:
        return ToolResult(name="Glob", output="일치하는 항목이 없습니다.")
    # 출력 제한
    if len(matches) > 200:
        matches = matches[:200]
        matches.append(f"... 외 다수 (200개에서 잘림)")
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
        return ToolResult(name="Grep", output="일치하는 항목이 없습니다.")
    lines = output.split("\n")
    if len(lines) > 100:
        output = "\n".join(lines[:100]) + f"\n... (전체 {len(lines)}건 중 처음 100건 표시)"
    return ToolResult(name="Grep", output=output)


def get_tool_definitions_for_prompt() -> str:
    """시스템 프롬프트에 삽입할 도구 정의를 포맷한다."""
    parts = []
    for tool in TOOL_REGISTRY.values():
        props = tool.parameters.get("properties", {})
        params_desc = []
        for pname, pdef in props.items():
            req = " (필수)" if pname in tool.parameters.get("required", []) else ""
            params_desc.append(f"    - {pname}: {pdef.get('description', '')}{req}")
        params_str = "\n".join(params_desc) if params_desc else "    (매개변수 없음)"
        parts.append(f"## {tool.name}\n{tool.description}\n매개변수:\n{params_str}")
    return "\n\n".join(parts)
