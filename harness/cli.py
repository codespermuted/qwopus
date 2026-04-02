"""Interactive REPL — the main entry point."""
from __future__ import annotations

import os
import subprocess
import sys

from .commands import handle_slash_command
from .runtime import ConversationRuntime
from .session import Session
from . import ui


def main():
    cwd = os.getcwd()
    resume_id = None
    rest_args = []

    # Parse known flags, collect the rest
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--cwd" and i + 1 < len(args):
            cwd = args[i + 1]
            i += 2
        elif args[i] == "--resume" and i + 1 < len(args):
            resume_id = args[i + 1]
            i += 2
        else:
            rest_args.append(args[i])
            i += 1

    # Load or create session
    if resume_id:
        session = Session.load(resume_id)
        ui.print_success(f"Resumed session: {session.session_id}")
    else:
        session = Session()

    runtime = ConversationRuntime(cwd=cwd, session=session)

    # One-shot mode: remaining args are a prompt
    if rest_args:
        prompt = " ".join(rest_args)
        ui.console.print(f"[bold blue]❯[/] {prompt}\n")
        runtime.run_turn(prompt)
        _print_usage(session)
        return

    # Interactive mode
    from .gpu import detect_gpus, format_gpu_info
    from .indexer import build_project_index
    gpus = detect_gpus()

    ui.print_banner(cwd, format_gpu_info(gpus))

    # 프로젝트 인덱스 생성
    index = build_project_index(cwd)
    file_count = len([l for l in index.split("\n") if l.strip() and not l.strip().startswith("...")])
    ui.print_info(f"프로젝트 인덱스: {file_count}개 파일 감지됨")

    while True:
        user_input = ui.get_user_input()

        if user_input is None:
            ui.console.print("\n[dim]Bye![/]")
            break

        if not user_input:
            continue

        # Shell escape: !command
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if cmd:
                ui.console.print(f"[dim]$ {cmd}[/]")
                try:
                    subprocess.run(cmd, shell=True, cwd=cwd)
                except Exception as e:
                    ui.print_error(str(e))
            continue

        # Slash commands
        if user_input.startswith("/"):
            response = handle_slash_command(user_input, session, cwd)
            if response == "__EXIT__":
                ui.console.print("[dim]Bye![/]")
                break
            if response is not None:
                # Commands now render their own output via rich
                if response:  # Non-empty means legacy text to display
                    ui.print_command_response(response)
                continue
            ui.print_warning(f"Unknown command: {user_input.split()[0]}")
            continue

        # Regular prompt → run turn
        result = runtime.run_turn(user_input)
        _print_usage(session)

        # Auto-save periodically
        if len(session.messages) % 10 == 0:
            session.save()


def _print_usage(session: Session):
    """Print token usage after each turn."""
    total = session.total_prompt_tokens + session.total_completion_tokens
    if total > 0:
        ui.console.print(
            f"[dim]  tokens: {total:,} "
            f"(prompt {session.total_prompt_tokens:,} + "
            f"completion {session.total_completion_tokens:,})[/]"
        )


if __name__ == "__main__":
    main()
