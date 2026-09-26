from pathlib import Path

import pytest

from image_selector import config
from image_selector.config import ConfigError, Settings

ENV_NAMES = ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_VISION_MODEL", "TYPESAFE_DECISION_MODEL", "IMAGE_SELECTOR_MEMORY_FILE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", None)  # keep a developer's real .env out of tests
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_load_reads_keys_and_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", " ts-key ")

    assert Settings.load() == Settings(deepseek_api_key="ds-key", typesafe_api_key="ts-key")


def test_memory_file_defaults_to_project_root_and_resolves_relative_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    assert Path(Settings.load().memory_file) == config.PROJECT_ROOT / "image_selector_memory.jsonl"

    monkeypatch.setenv("IMAGE_SELECTOR_MEMORY_FILE", "data/memory.jsonl")
    assert Path(Settings.load().memory_file) == config.PROJECT_ROOT / "data" / "memory.jsonl"


def test_load_applies_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    monkeypatch.setenv("DEEPSEEK_VISION_MODEL", "other-vision")
    monkeypatch.setenv("TYPESAFE_DECISION_MODEL", "jev-1")

    settings = Settings.load()
    assert (settings.vision_model, settings.decision_model) == ("other-vision", "jev-1")


def test_load_names_missing_keys(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    with pytest.raises(ConfigError, match="TYPESAFE_API_KEY"):
        Settings.load()
