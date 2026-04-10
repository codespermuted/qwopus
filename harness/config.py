"""Settings management — loads ~/.qwopus/settings.json with defaults."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".qwopus"
SETTINGS_PATH = CONFIG_DIR / "settings.json"

# Default settings
DEFAULTS: dict[str, Any] = {
    # Model settings
    "model": {
        "repo": "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
        "file": "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    # Tool settings
    "tools": {
        "max_tool_rounds": 8,
        "tool_output_limit": 1500,
        "bash_timeout": 120,
    },
    # Session settings
    "session": {
        "auto_save_interval": 10,
        "max_history_messages": 60,
    },
    # Permission settings
    "permissions": {
        "auto_allow": [],          # Tools auto-allowed without confirmation (e.g. ["Glob", "Grep"])
        "deny": [],                # Tools to block
        "confirm_dangerous": True,  # Whether to confirm dangerous commands
    },
    # Hook settings
    "hooks": {
        "pre_tool": [],   # Shell commands run before a tool (e.g. ["echo 'running {tool}'"])
        "post_tool": [],  # Shell commands run after a tool
        "pre_turn": [],   # Before a turn starts
        "post_turn": [],  # After a turn finishes
    },
    # UI settings
    "ui": {
        "show_thinking": True,
        "show_token_usage": True,
        "streaming": True,
    },
}


@dataclass
class Settings:
    """Manages user settings."""
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, dotpath: str, default=None):
        """Get a setting using a dot-delimited path, e.g. 'model.temperature'."""
        keys = dotpath.split(".")
        obj = self.data
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    def set(self, dotpath: str, value):
        """Set a setting using a dot-delimited path."""
        keys = dotpath.split(".")
        obj = self.data
        for key in keys[:-1]:
            if key not in obj or not isinstance(obj[key], dict):
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value

    def save(self):
        """Save settings to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls) -> Settings:
        """Load the settings file. Creates defaults if none exists."""
        data = _deep_copy(DEFAULTS)

        if SETTINGS_PATH.exists():
            try:
                user_data = json.loads(SETTINGS_PATH.read_text())
                _deep_merge(data, user_data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load settings file: %s — using defaults", e)
        else:
            # Create default settings file
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        return cls(data=data)


def _deep_merge(base: dict, override: dict):
    """Deep-merge the values of override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))
