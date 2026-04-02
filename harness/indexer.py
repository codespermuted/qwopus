"""프로젝트 인덱서 — 파일 구조와 요약을 자동 생성하여 컨텍스트에 주입."""
from __future__ import annotations

import os
from pathlib import Path

# 인덱싱 대상 확장자
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".rb", ".php", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json",
    ".md", ".txt",
}

# 무시할 디렉토리
IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "egg-info", ".tox", ".claude",
}

# 인덱스 최대 길이 (토큰 절약)
MAX_INDEX_CHARS = 3000


def build_project_index(cwd: str) -> str:
    """프로젝트의 파일 구조와 각 파일 요약을 생성한다."""
    root = Path(cwd)
    entries: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 무시할 디렉토리 필터링
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]

        rel_dir = os.path.relpath(dirpath, root)
        depth = rel_dir.count(os.sep)

        # 너무 깊은 디렉토리 스킵
        if depth > 4:
            dirnames.clear()
            continue

        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in CODE_EXTENSIONS:
                continue

            rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
            summary = _extract_summary(os.path.join(dirpath, fname))

            if summary:
                entries.append(f"  {rel_path} — {summary}")
            else:
                entries.append(f"  {rel_path}")

            # 길이 제한
            if sum(len(e) for e in entries) > MAX_INDEX_CHARS:
                entries.append(f"  ... (파일이 더 있음)")
                return "\n".join(entries)

    if not entries:
        return "  (코드 파일 없음)"

    return "\n".join(entries)


def _extract_summary(filepath: str) -> str:
    """파일의 첫 docstring 또는 주석을 추출하여 한 줄 요약을 반환한다."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 10:  # 처음 10줄만 확인
                    break
                lines.append(line)
    except (OSError, PermissionError):
        return ""

    text = "".join(lines)

    # Python docstring: """...""" 또는 '''...'''
    for delim in ('"""', "'''"):
        if delim in text:
            start = text.index(delim) + 3
            end = text.find(delim, start)
            if end > start:
                doc = text[start:end].strip().split("\n")[0].strip()
                if doc:
                    return doc[:80]

    # 주석 기반 (#, //)
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment = stripped.lstrip("# ").strip()
            if comment and len(comment) > 5:
                return comment[:80]
        elif stripped.startswith("//"):
            comment = stripped.lstrip("/ ").strip()
            if comment and len(comment) > 5:
                return comment[:80]

    return ""
