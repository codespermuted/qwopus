"""Slash command handling."""
from __future__ import annotations

from .session import Session


def handle_slash_command(cmd: str, session: Session, cwd: str) -> str | None:
    """
    Handle a /slash command. Returns response text, or None if not a known command.
    """
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        return HELP_TEXT

    elif command == "/status":
        return (
            f"Session: {session.session_id}\n"
            f"Messages: {len(session.messages)}\n"
            f"Tokens: {session.total_prompt_tokens + session.total_completion_tokens:,}\n"
            f"CWD: {cwd}"
        )

    elif command == "/clear":
        session.messages.clear()
        return "🗑️  Conversation cleared."

    elif command == "/save":
        session.save()
        return f"💾 Session saved: {session.session_id}"

    elif command == "/sessions":
        sessions = Session.list_sessions()
        if not sessions:
            return "No saved sessions."
        return "Saved sessions:\n" + "\n".join(f"  - {s}" for s in sessions)

    elif command == "/resume":
        if not arg:
            return "Usage: /resume <session_id>"
        try:
            loaded = Session.load(arg.strip())
            session.session_id = loaded.session_id
            session.messages = loaded.messages
            session.total_prompt_tokens = loaded.total_prompt_tokens
            session.total_completion_tokens = loaded.total_completion_tokens
            return f"📂 Resumed session: {loaded.session_id} ({len(loaded.messages)} messages)"
        except FileNotFoundError:
            return f"Session not found: {arg}"

    elif command == "/quit" or command == "/exit":
        return "__EXIT__"

    elif command == "/compact":
        before = len(session.messages)
        session._compact()
        return f"Compacted: {before} → {len(session.messages)} messages"

    return None  # Not a known command


HELP_TEXT = """\
╭──────────────────────────────────────╮
│          Qwopus Commands             │
├──────────────────────────────────────┤
│  /help       Show this help          │
│  /status     Session info            │
│  /clear      Clear conversation      │
│  /compact    Trim old messages       │
│  /save       Save session to disk    │
│  /sessions   List saved sessions     │
│  /resume ID  Resume a session        │
│  /quit       Exit                    │
│                                      │
│  !command    Run shell command        │
╰──────────────────────────────────────╯\
"""
