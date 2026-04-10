"""Interactive REPL — main entry point."""
from __future__ import annotations

import os
import subprocess
import sys

from .commands import handle_slash_command
from .config import Settings
from .runtime import ConversationRuntime
from .session import Session
from . import ui


def main():
    cwd = os.getcwd()
    resume_id = None
    rest_args = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--cwd" and i + 1 < len(args):
            cwd = args[i + 1]
            i += 2
        elif args[i] == "--resume" and i + 1 < len(args):
            resume_id = args[i + 1]
            i += 2
        elif args[i] == "--no-stream":
            # Disable streaming (for debugging)
            os.environ["QWOPUS_NO_STREAM"] = "1"
            i += 1
        else:
            rest_args.append(args[i])
            i += 1

    # Load settings
    settings = Settings.load()
    if os.environ.get("QWOPUS_NO_STREAM"):
        settings.set("ui.streaming", False)

    # Load or create a session
    if resume_id:
        session = Session.load(resume_id)
        ui.print_success(f"Resumed session: {session.session_id}")
    else:
        session = Session()

    runtime = ConversationRuntime(cwd=cwd, session=session, settings=settings)

    # One-shot mode
    if rest_args:
        prompt = " ".join(rest_args)
        ui.console.print(f"[bold blue]>[/] {prompt}\n")
        runtime.run_turn(prompt)
        _print_usage(session, settings)
        return

    # Interactive mode
    from .gpu import detect_gpus, format_gpu_info
    from .indexer import build_project_index
    gpus = detect_gpus()

    ui.print_banner(cwd, format_gpu_info(gpus))

    index = build_project_index(cwd)
    file_count = len([l for l in index.split("\n") if l.strip() and not l.strip().startswith("...")])
    ui.print_info(f"Project index: {file_count} files")
    stream_status = "on" if settings.get("ui.streaming", True) else "off"
    ui.print_info(f"Streaming: {stream_status}")

    auto_save = settings.get("session.auto_save_interval", 10)

    while True:
        user_input = ui.get_user_input()

        if user_input is None:
            ui.console.print("\n[dim]Bye![/]")
            break

        if not user_input:
            continue

        # !shell command
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if cmd:
                ui.console.print(f"[dim]$ {cmd}[/]")
                try:
                    subprocess.run(cmd, shell=True, cwd=cwd)
                except Exception as e:
                    ui.print_error(str(e))
            continue

        # /slash command
        if user_input.startswith("/"):
            response = handle_slash_command(user_input, session, cwd)
            if response == "__EXIT__":
                ui.console.print("[dim]Bye![/]")
                break
            if response is not None:
                if response:
                    ui.print_command_response(response)
                continue
            ui.print_warning(f"Unknown command: {user_input.split()[0]}")
            continue

        # Normal prompt -> run a turn
        result = runtime.run_turn(user_input)
        _print_usage(session, settings)

        # Auto-save
        if auto_save and len(session.messages) % auto_save == 0:
            session.save()


def _print_usage(session: Session, settings: Settings):
    if not settings.get("ui.show_token_usage", True):
        return
    total = session.total_prompt_tokens + session.total_completion_tokens
    if total > 0:
        ui.console.print(
            f"[dim]  tokens: {total:,} "
            f"(prompt {session.total_prompt_tokens:,} + "
            f"completion {session.total_completion_tokens:,})[/]"
        )


if __name__ == "__main__":
    main()
