"""Terminal UI — rich rendering, spinners, and formatted output."""
from __future__ import annotations

import time
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.rule import Rule
from rich.columns import Columns

# ── Theme ────────────────────────────────────────────────────

THEME = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "tool.name": "bold magenta",
    "tool.args": "dim",
    "thinking": "dim italic",
    "user": "bold blue",
    "assistant": "bold green",
    "muted": "dim",
})

console = Console(theme=THEME)

# ── Banner ───────────────────────────────────────────────────

BANNER_TEXT = """\
[bold cyan]   ____                                [/]
[bold cyan]  / __ \\__      ______  ____  __  _____[/]
[bold cyan] / / / / | /| / / __ \\/ __ \\/ / / / ___/[/]
[bold cyan]/ /_/ /| |/ |/ / /_/ / /_/ / /_/ (__  )[/]
[bold cyan]\\___\\_\\ |__/|__/\\____/ .___/\\__,_/____/ [/]
[bold cyan]                    /_/                 [/]"""


def print_banner(cwd: str, gpu_info: str = ""):
    """Print the startup banner."""
    console.print()
    console.print(BANNER_TEXT)
    console.print()
    console.print(
        Panel(
            f"[bold]Local AI Coding Agent[/]\n"
            f"[dim]Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled[/]\n"
            f"\n"
            f"[bold blue]>[/] Working directory: [bold]{cwd}[/]\n"
            f"{f'[bold blue]>[/] {gpu_info}' if gpu_info else ''}"
            f"\n"
            f"[dim]/help for commands · /quit to exit · !cmd for shell[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


# ── User Input ───────────────────────────────────────────────

def get_user_input() -> str | None:
    """Get user input with styled prompt. Returns None on EOF/interrupt."""
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import InMemoryHistory

        if not hasattr(get_user_input, "_history"):
            get_user_input._history = InMemoryHistory()

        result = prompt(
            HTML("<ansicyan><b>❯ </b></ansicyan>"),
            history=get_user_input._history,
            multiline=False,
        )
        return result.strip()
    except (EOFError, KeyboardInterrupt):
        return None
    except ImportError:
        # Fallback if prompt_toolkit unavailable
        try:
            return input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            return None


# ── Thinking Display ─────────────────────────────────────────

def print_thinking(text: str):
    """Display thinking in a collapsed-style panel."""
    if not text.strip():
        return
    # Truncate very long thinking
    lines = text.strip().split("\n")
    if len(lines) > 15:
        display = "\n".join(lines[:12]) + f"\n  ... ({len(lines) - 12} more lines)"
    else:
        display = text.strip()

    console.print(
        Panel(
            Text(display, style="thinking"),
            title="[dim italic]thinking[/]",
            title_align="left",
            border_style="dim",
            padding=(0, 1),
        )
    )


# ── Tool Execution Display ───────────────────────────────────

@contextmanager
def tool_spinner(tool_name: str, summary: str):
    """Show a spinner while a tool executes."""
    display_text = Text()
    display_text.append(f"  {tool_name}", style="tool.name")
    display_text.append(f"  {summary}", style="tool.args")

    spinner = Spinner("dots", text=display_text, style="cyan")
    with Live(spinner, console=console, transient=True):
        yield


def print_tool_call(tool_name: str, summary: str):
    """Print a tool call header."""
    console.print(
        Text.assemble(
            ("  ▶ ", "bold cyan"),
            (tool_name, "tool.name"),
            ("  ", ""),
            (summary, "tool.args"),
        )
    )


def print_tool_result(tool_name: str, output: str, success: bool):
    """Print a tool result with syntax highlighting."""
    icon = "[success]✓[/]" if success else "[error]✗[/]"
    console.print(f"  {icon} ", end="")

    # Truncate long output for display
    lines = output.split("\n")
    if len(lines) > 25:
        display = "\n".join(lines[:20]) + f"\n    ... ({len(lines) - 20} more lines)"
    else:
        display = output

    if tool_name == "Bash" and success:
        console.print(
            Panel(
                Syntax(display, "bash", theme="monokai", line_numbers=False, word_wrap=True)
                if _looks_like_code(display) else Text(display, style="dim"),
                border_style="dim green" if success else "dim red",
                padding=(0, 1),
            )
        )
    elif tool_name in ("FileRead", "Grep"):
        console.print(
            Panel(
                Text(display, style="dim"),
                border_style="dim green" if success else "dim red",
                padding=(0, 1),
            )
        )
    else:
        console.print(f"[dim]{display}[/]")


def _looks_like_code(text: str) -> bool:
    """Heuristic: does the text look like code output?"""
    indicators = ["total ", "drwx", "-rw-", "  File ", "Traceback", "Error:", ">>>"]
    return any(ind in text for ind in indicators)


# ── Assistant Response ───────────────────────────────────────

def print_response(text: str):
    """Print the assistant's response with markdown rendering."""
    if not text.strip():
        return
    console.print()
    console.print(
        Panel(
            Markdown(text, code_theme="monokai"),
            title="[bold green]qwopus[/]",
            title_align="left",
            border_style="green",
            padding=(0, 2),
        )
    )
    console.print()


# ── Status & Info ────────────────────────────────────────────

def print_info(msg: str):
    console.print(f"[info]  {msg}[/]")


def print_warning(msg: str):
    console.print(f"[warning]  ⚠ {msg}[/]")


def print_error(msg: str):
    console.print(f"[error]  ✗ {msg}[/]")


def print_success(msg: str):
    console.print(f"[success]  ✓ {msg}[/]")


def print_command_response(text: str):
    """Print a slash command response."""
    console.print(
        Panel(text, border_style="dim cyan", padding=(0, 2))
    )


def print_separator():
    console.print(Rule(style="dim"))


# ── Confirmation ─────────────────────────────────────────────

def confirm(msg: str) -> bool:
    """Ask user for confirmation."""
    console.print(f"[warning]{msg}[/]", end="")
    try:
        answer = input(" ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ── GPU Info ─────────────────────────────────────────────────

def format_gpu_info(gpus: list[dict]) -> str:
    """Format GPU info for the banner."""
    if not gpus:
        return ""
    parts = []
    for g in gpus:
        parts.append(f"{g['name']} ({g['total_mb']:,}MB)")
    return "GPU: " + " + ".join(parts)
