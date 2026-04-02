"""훅 시스템 — 도구 실행 전/후 자동 동작."""
from __future__ import annotations

import logging
import subprocess
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)


class HookRunner:
    """설정에 정의된 훅을 실행한다."""

    def __init__(self, settings: Settings, cwd: str):
        self.settings = settings
        self.cwd = cwd

    def run_pre_tool(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """도구 실행 전 훅. 출력이 있으면 반환, 차단 시 'BLOCK' 반환."""
        hooks = self.settings.get("hooks.pre_tool", [])
        return self._run_hooks(hooks, tool_name=tool_name, arguments=arguments)

    def run_post_tool(self, tool_name: str, output: str, success: bool) -> str | None:
        """도구 실행 후 훅. 출력이 있으면 반환."""
        hooks = self.settings.get("hooks.post_tool", [])
        return self._run_hooks(hooks, tool_name=tool_name, output=output[:500], success=success)

    def run_pre_turn(self, user_input: str) -> str | None:
        """턴 시작 전 훅."""
        hooks = self.settings.get("hooks.pre_turn", [])
        return self._run_hooks(hooks, user_input=user_input)

    def run_post_turn(self, response: str) -> str | None:
        """턴 완료 후 훅."""
        hooks = self.settings.get("hooks.post_turn", [])
        return self._run_hooks(hooks, response=response[:500])

    def _run_hooks(self, hooks: list[str], **context) -> str | None:
        """훅 명령어 목록을 실행한다."""
        if not hooks:
            return None

        outputs = []
        for hook_cmd in hooks:
            try:
                # 컨텍스트 변수를 명령어에 치환
                cmd = hook_cmd.format(**context)
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    cwd=self.cwd, timeout=10,
                )
                if result.stdout.strip():
                    outputs.append(result.stdout.strip())
                # BLOCK 반환으로 도구 실행 차단 가능
                if result.stdout.strip() == "BLOCK":
                    return "BLOCK"
            except subprocess.TimeoutExpired:
                logger.warning("훅 타임아웃: %s", hook_cmd)
            except (KeyError, ValueError) as e:
                # format 실패 — 변수가 없는 경우 무시
                logger.debug("훅 변수 치환 실패: %s — %s", hook_cmd, e)
            except Exception as e:
                logger.warning("훅 실행 실패: %s — %s", hook_cmd, e)

        return "\n".join(outputs) if outputs else None
