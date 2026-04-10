"""Project indexer — auto-extracts file structure, summaries, and key keywords into the context."""
from __future__ import annotations

import os
import re
from pathlib import Path

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".rb", ".php", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json",
    ".md", ".txt",
}

# Directories to skip
IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "egg-info", ".tox", ".claude",
}

# Max index length (to save tokens)
MAX_INDEX_CHARS = 4000

# Key keyword patterns — when a file matches one, extract it as a tag
KEYWORD_PATTERNS = [
    (re.compile(r'target[_\s]*=.*?["\'](\w+)["\']', re.IGNORECASE), "target"),
    (re.compile(r'(?:predict|forecast|output)[_\s]*(?:col|column|var|name).*?["\'](\w+)["\']', re.IGNORECASE), "predict"),
    (re.compile(r'item_id.*?["\'](\w+)["\']'), "item"),
    (re.compile(r'prediction_length\s*=\s*(\d+)'), "horizon"),
    (re.compile(r'class\s+(\w+)\s*[\(:]'), "class"),
    (re.compile(r'def\s+(main|train|fit|predict|evaluate|run)\s*\('), "entry"),
]


def build_project_index(cwd: str) -> str:
    """Build a summary of the project's file structure, summaries, and key keywords."""
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
                entries.append(f"  ... (more files)")
                return "\n".join(entries)

    if not entries:
        return "  (no code files)"

    return "\n".join(entries)


def _extract_summary(filepath: str) -> str:
    """Extract the first docstring or comment from a file as a one-line summary."""
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
    """Extract key keywords from a Python file as tags."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read(5000)  # Scan only the first 5KB
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
            if len(tags) >= 6:  # Cap the number of tags
                break

    return ", ".join(tags) if tags else ""


def scan_project_targets(cwd: str) -> str:
    """Scan the entire project for target/prediction variables and summarize them.
    A helper tool to make sure the model doesn't miss key information."""
    root = Path(cwd)
    findings: list[str] = []

    # Target-related patterns
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
                findings.append(f"\n  {rel_path}:")
                for ff in dict.fromkeys(file_findings):  # Dedupe
                    findings.append(f"    - {ff}")

    if not findings:
        return "No target/prediction variables found."

    return "Project target variable scan results:" + "\n".join(findings)
