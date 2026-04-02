"""프로젝트 인덱서 — 파일 구조, 요약, 핵심 키워드를 자동 추출하여 컨텍스트에 주입."""
from __future__ import annotations

import os
import re
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
MAX_INDEX_CHARS = 4000

# 핵심 키워드 패턴 — 파일에서 이 패턴이 보이면 태그로 추출
KEYWORD_PATTERNS = [
    (re.compile(r'target[_\s]*=.*?["\'](\w+)["\']', re.IGNORECASE), "target"),
    (re.compile(r'(?:predict|forecast|output)[_\s]*(?:col|column|var|name).*?["\'](\w+)["\']', re.IGNORECASE), "predict"),
    (re.compile(r'item_id.*?["\'](\w+)["\']'), "item"),
    (re.compile(r'prediction_length\s*=\s*(\d+)'), "horizon"),
    (re.compile(r'class\s+(\w+)\s*[\(:]'), "class"),
    (re.compile(r'def\s+(main|train|fit|predict|evaluate|run)\s*\('), "entry"),
]


def build_project_index(cwd: str) -> str:
    """프로젝트의 파일 구조, 요약, 핵심 키워드를 생성한다."""
    root = Path(cwd)
    entries: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]

        rel_dir = os.path.relpath(dirpath, root)
        depth = rel_dir.count(os.sep)

        if depth > 4:
            dirnames.clear()
            continue

        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in CODE_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)
            summary = _extract_summary(full_path)
            tags = _extract_tags(full_path) if ext == ".py" else ""

            parts = [f"  {rel_path}"]
            if summary:
                parts.append(f"— {summary}")
            if tags:
                parts.append(f"[{tags}]")
            entries.append(" ".join(parts))

            if sum(len(e) for e in entries) > MAX_INDEX_CHARS:
                entries.append(f"  ... (파일이 더 있음)")
                return "\n".join(entries)

    if not entries:
        return "  (코드 파일 없음)"

    return "\n".join(entries)


def _extract_summary(filepath: str) -> str:
    """파일의 첫 docstring 또는 주석을 한 줄 요약으로 추출한다."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 10:
                    break
                lines.append(line)
    except (OSError, PermissionError):
        return ""

    text = "".join(lines)

    for delim in ('"""', "'''"):
        if delim in text:
            start = text.index(delim) + 3
            end = text.find(delim, start)
            if end > start:
                doc = text[start:end].strip().split("\n")[0].strip()
                if doc:
                    return doc[:80]

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


def _extract_tags(filepath: str) -> str:
    """Python 파일에서 핵심 키워드를 태그로 추출한다."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read(5000)  # 처음 5KB만 스캔
    except (OSError, PermissionError):
        return ""

    tags = []
    seen = set()

    for pattern, label in KEYWORD_PATTERNS:
        for match in pattern.finditer(content):
            value = match.group(1) if match.lastindex else match.group(0)
            tag = f"{label}:{value}"
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
            if len(tags) >= 6:  # 태그 수 제한
                break

    return ", ".join(tags) if tags else ""


def scan_project_targets(cwd: str) -> str:
    """프로젝트 전체에서 타겟/예측 변수를 스캔하여 요약한다.
    모델이 핵심 정보를 놓치지 않도록 돕는 보조 도구."""
    root = Path(cwd)
    findings: list[str] = []

    # 타겟 관련 패턴
    target_patterns = [
        re.compile(r'(?:target|Target|TARGET)\s*[=:]\s*["\']?(\w+)["\']?'),
        re.compile(r'rename.*?(?:columns|col).*?["\'](\w+)["\']\s*:\s*["\']target["\']'),
        re.compile(r'["\'](\w+)["\']\s*:\s*["\']target["\']'),
        re.compile(r'prediction_length\s*=\s*(\d+)'),
        re.compile(r'item_id.*?["\'](\w+)["\']'),
        re.compile(r'(?:y_col|target_col|label_col)\s*=\s*["\'](\w+)["\']'),
    ]

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]

        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)

            try:
                with open(full_path, "r", errors="ignore") as f:
                    content = f.read(8000)
            except (OSError, PermissionError):
                continue

            file_findings = []
            for pattern in target_patterns:
                for match in pattern.finditer(content):
                    file_findings.append(match.group(0).strip()[:80])

            if file_findings:
                findings.append(f"\n  📄 {rel_path}:")
                for ff in dict.fromkeys(file_findings):  # 중복 제거
                    findings.append(f"    - {ff}")

    if not findings:
        return "타겟/예측 변수를 찾을 수 없습니다."

    return "프로젝트 타겟 변수 스캔 결과:" + "\n".join(findings)
