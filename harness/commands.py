"""Slash command handling."""
from __future__ import annotations

from .session import Session
from . import ui

from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def handle_slash_command(cmd: str, session: Session, cwd: str) -> str | None:
    """
    Handle a /slash command. Returns "__EXIT__" to quit, "" for handled, None if unknown.
    """
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        _show_help()
        return ""

    elif command == "/status":
        _show_status(session, cwd)
        return ""

    elif command == "/clear":
        session.messages.clear()
        ui.print_success("Conversation cleared.")
        return ""

    elif command == "/save":
        session.save()
        ui.print_success(f"Session saved: {session.session_id}")
        return ""

    elif command == "/sessions":
        sessions = Session.list_sessions()
        if not sessions:
            ui.print_info("No saved sessions.")
        else:
            ui.console.print("[bold]Saved sessions:[/]")
            for s in sessions:
                ui.console.print(f"  [cyan]{s}[/]")
        return ""

    elif command == "/resume":
        if not arg:
            ui.print_warning("Usage: /resume <session_id>")
            return ""
        try:
            loaded = Session.load(arg.strip())
            session.session_id = loaded.session_id
            session.messages = loaded.messages
            session.total_prompt_tokens = loaded.total_prompt_tokens
            session.total_completion_tokens = loaded.total_completion_tokens
            ui.print_success(f"Resumed session: {loaded.session_id} ({len(loaded.messages)} messages)")
        except FileNotFoundError:
            ui.print_error(f"Session not found: {arg}")
        return ""

    elif command in ("/quit", "/exit"):
        return "__EXIT__"

    elif command == "/compact":
        before = len(session.messages)
        # 최근 10개 메시지만 유지
        if len(session.messages) > 10:
            session.messages = session.messages[-10:]
        ui.print_info(f"Compacted: {before} → {len(session.messages)} messages")
        return ""

    return None  # Not a known command


def _show_help():
    """Display help with rich formatting."""
    ui.console.print()

    # Commands table
    table = Table(
        title="[bold cyan]Commands[/]",
        show_header=True,
        header_style="bold",
        border_style="cyan",
        padding=(0, 2),
    )
    table.add_column("Command", style="bold cyan", no_wrap=True)
    table.add_column("Description")

    table.add_row("/help", "Show this help")
    table.add_row("/status", "Session info & token usage")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/compact", "Trim old messages")
    table.add_row("/save", "Save session to disk")
    table.add_row("/sessions", "List saved sessions")
    table.add_row("/resume [dim]<id>[/]", "Resume a saved session")
    table.add_row("/quit", "Exit Qwopus")

    ui.console.print(table)
    ui.console.print()

    # Tools table
    tools_table = Table(
        title="[bold magenta]Available Tools[/]",
        show_header=True,
        header_style="bold",
        border_style="magenta",
        padding=(0, 2),
    )
    tools_table.add_column("Tool", style="bold magenta", no_wrap=True)
    tools_table.add_column("Description")

    tools_table.add_row("Bash", "Execute shell commands")
    tools_table.add_row("FileRead", "Read files with line numbers")
    tools_table.add_row("FileWrite", "Create or overwrite files")
    tools_table.add_row("FileEdit", "Edit files (find & replace)")
    tools_table.add_row("Glob", "Find files by pattern")
    tools_table.add_row("Grep", "Search file contents with regex")

    ui.console.print(tools_table)
    ui.console.print()

    # Tips
    ui.console.print(
        Panel(
            "[bold]Tips[/]\n"
            "  [cyan]!command[/]    Run a shell command directly\n"
            "  [cyan]↑ / ↓[/]      Browse input history\n"
            "  [cyan]Ctrl+C[/]     Cancel / Exit",
            border_style="dim",
            padding=(0, 2),
        )
    )
    ui.console.print()


def _show_status(session: Session, cwd: str):
    """Display session status with rich formatting."""
    total = session.total_prompt_tokens + session.total_completion_tokens

    table = Table(show_header=False, border_style="cyan", padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Session ID", f"[cyan]{session.session_id}[/]")
    table.add_row("Working Dir", cwd)
    table.add_row("Messages", str(len(session.messages)))
    table.add_row("Prompt Tokens", f"{session.total_prompt_tokens:,}")
    table.add_row("Completion Tokens", f"{session.total_completion_tokens:,}")
    table.add_row("Total Tokens", f"[bold]{total:,}[/]")

    ui.console.print()
    ui.console.print(table)
    ui.console.print()
