"""Hook system — automatic actions before/after tool execution."""
from __future__ import annotations

import logging
import subprocess
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)


class HookRunner:
    """Runs the hooks defined in settings."""

    def __init__(self, settings: Settings, cwd: str):
        self.settings = settings
        self.cwd = cwd

    def run_pre_tool(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Pre-tool hook. Returns output if any, or 'BLOCK' to block execution."""
        hooks = self.settings.get("hooks.pre_tool", [])
        return self._run_hooks(hooks, tool_name=tool_name, arguments=arguments)

    def run_post_tool(self, tool_name: str, output: str, success: bool) -> str | None:
        """Post-tool hook. Returns output if any."""
        hooks = self.settings.get("hooks.post_tool", [])
        return self._run_hooks(hooks, tool_name=tool_name, output=output[:500], success=success)

    def run_pre_turn(self, user_input: str) -> str | None:
        """Hook that runs before a turn starts."""
        hooks = self.settings.get("hooks.pre_turn", [])
        return self._run_hooks(hooks, user_input=user_input)

    def run_post_turn(self, response: str) -> str | None:
        """Hook that runs after a turn finishes."""
        hooks = self.settings.get("hooks.post_turn", [])
        return self._run_hooks(hooks, response=response[:500])

    def _run_hooks(self, hooks: list[str], **context) -> str | None:
        """Run a list of hook commands."""
        if not hooks:
            return None

        outputs = []
        for hook_cmd in hooks:
            try:
                # Substitute context variables into the command
                cmd = hook_cmd.format(**context)
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    cwd=self.cwd, timeout=10,
                )
                if result.stdout.strip():
                    outputs.append(result.stdout.strip())
                # Returning BLOCK can block tool execution
                if result.stdout.strip() == "BLOCK":
                    return "BLOCK"
            except subprocess.TimeoutExpired:
                logger.warning("Hook timed out: %s", hook_cmd)
            except (KeyError, ValueError) as e:
                # format failed — ignore when a variable is missing
                logger.debug("Hook variable substitution failed: %s — %s", hook_cmd, e)
            except Exception as e:
                logger.warning("Hook execution failed: %s — %s", hook_cmd, e)

        return "\n".join(outputs) if outputs else None
