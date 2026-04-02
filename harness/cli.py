"""Interactive REPL — the main entry point."""
from __future__ import annotations

import os
import subprocess
import sys

from .commands import handle_slash_command
from .runtime import ConversationRuntime
from .session import Session

BANNER = """\
╔══════════════════════════════════════════════════════════════╗
║  🐙 Qwopus — Local AI Coding Agent                         ║
║     Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled        ║
║     Q5_K_M · 2×RTX 5060 Ti · llama.cpp                     ║
╠══════════════════════════════════════════════════════════════╣
║  /help for commands · /quit to exit · !cmd for shell        ║
╚══════════════════════════════════════════════════════════════╝
"""


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
        print(f"📂 Resumed session: {session.session_id}")
    else:
        session = Session()

    runtime = ConversationRuntime(cwd=cwd, session=session)

    # One-shot mode: remaining args are a prompt
    if rest_args:
        prompt = " ".join(rest_args)
        print(f"📝 {prompt}\n")
        runtime.run_turn(prompt)
        return

    print(BANNER)
    print(f"📁 Working directory: {cwd}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        # Shell escape: !command
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if cmd:
                try:
                    result = subprocess.run(cmd, shell=True, cwd=cwd)
                except Exception as e:
                    print(f"Error: {e}")
            continue

        # Slash commands
        if user_input.startswith("/"):
            response = handle_slash_command(user_input, session, cwd)
            if response == "__EXIT__":
                print("👋 Bye!")
                break
            if response is not None:
                print(response)
                continue
            # Unknown slash command — treat as regular prompt
            print(f"Unknown command. Sending to LLM...")

        # Regular prompt → run turn
        print()
        result = runtime.run_turn(user_input)
        print()

        # Auto-save periodically
        if len(session.messages) % 10 == 0:
            session.save()


if __name__ == "__main__":
    main()
