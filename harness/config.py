"""설정 관리 — ~/.qwopus/settings.json 로드 및 기본값."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".qwopus"
SETTINGS_PATH = CONFIG_DIR / "settings.json"

# 기본 설정
DEFAULTS: dict[str, Any] = {
    # 모델 설정
    "model": {
        "repo": "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
        "file": "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    # 도구 설정
    "tools": {
        "max_tool_rounds": 8,
        "tool_output_limit": 1500,
        "bash_timeout": 120,
    },
    # 세션 설정
    "session": {
        "auto_save_interval": 10,
        "max_history_messages": 60,
    },
    # 권한 설정
    "permissions": {
        "auto_allow": [],          # 확인 없이 허용할 도구 (예: ["Glob", "Grep"])
        "deny": [],                # 차단할 도구
        "confirm_dangerous": True,  # 위험 명령 확인 여부
    },
    # 훅 설정
    "hooks": {
        "pre_tool": [],   # 도구 실행 전 셸 명령어 (예: ["echo 'running {tool}'"])
        "post_tool": [],  # 도구 실행 후 셸 명령어
        "pre_turn": [],   # 턴 시작 전
        "post_turn": [],  # 턴 완료 후
    },
    # UI 설정
    "ui": {
        "show_thinking": True,
        "show_token_usage": True,
        "streaming": True,
    },
}


@dataclass
class Settings:
    """사용자 설정을 관리한다."""
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, dotpath: str, default=None):
        """점(.) 구분 경로로 설정값을 가져온다. 예: 'model.temperature'"""
        keys = dotpath.split(".")
        obj = self.data
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    def set(self, dotpath: str, value):
        """점 구분 경로로 설정값을 저장한다."""
        keys = dotpath.split(".")
        obj = self.data
        for key in keys[:-1]:
            if key not in obj or not isinstance(obj[key], dict):
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value

    def save(self):
        """설정을 파일에 저장한다."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls) -> Settings:
        """설정 파일을 로드한다. 없으면 기본값으로 생성."""
        data = _deep_copy(DEFAULTS)

        if SETTINGS_PATH.exists():
            try:
                user_data = json.loads(SETTINGS_PATH.read_text())
                _deep_merge(data, user_data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("설정 파일 로드 실패: %s — 기본값 사용", e)
        else:
            # 기본 설정 파일 생성
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        return cls(data=data)


def _deep_merge(base: dict, override: dict):
    """override의 값을 base에 깊은 병합한다."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))
