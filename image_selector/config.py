"""Settings loaded from the environment and an optional `.env` file."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; plain environment variables still work.
    load_dotenv = None

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# `deepseek-v4-flash-vision-exp` is a retired alias that DeepSeek now routes to `deepseek-flash`.
DEFAULT_VISION_MODEL = "deepseek-flash"
DEFAULT_DECISION_MODEL = "jev-latest"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Anchored to the project so the same memory is used whichever directory the app is launched from.
DEFAULT_MEMORY_FILE = str(PROJECT_ROOT / "image_selector_memory.jsonl")


class ConfigError(RuntimeError):
    """A required setting is missing."""


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    typesafe_api_key: str
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    vision_model: str = DEFAULT_VISION_MODEL
    decision_model: str = DEFAULT_DECISION_MODEL
    memory_file: str = DEFAULT_MEMORY_FILE

    @classmethod
    def load(cls) -> "Settings":
        """Read settings from the environment, loading `.env` first when python-dotenv is installed."""
        if load_dotenv is not None:
            load_dotenv()

        missing = [name for name in ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY") if not _env(name)]
        if missing:
            raise ConfigError(f"Missing required setting(s): {', '.join(missing)}. Set them in the environment or in .env.")

        return cls(
            deepseek_api_key=_env("DEEPSEEK_API_KEY"),
            typesafe_api_key=_env("TYPESAFE_API_KEY"),
            deepseek_base_url=_env("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
            vision_model=_env("DEEPSEEK_VISION_MODEL") or DEFAULT_VISION_MODEL,
            decision_model=_env("TYPESAFE_DECISION_MODEL") or DEFAULT_DECISION_MODEL,
            memory_file=str(PROJECT_ROOT / _env("IMAGE_SELECTOR_MEMORY_FILE")) if _env("IMAGE_SELECTOR_MEMORY_FILE") else DEFAULT_MEMORY_FILE,
        )


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()
