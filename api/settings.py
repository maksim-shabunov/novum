"""API configuration, read from the environment (and .env if present).

Secrets never live in code or in a committed file. `.env` is gitignored;
`.env.example` documents every key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core import paths


def _load_dotenv() -> None:
    """Load .env if python-dotenv is installed. Absence is not an error."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        return
    env_file = paths.PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    default_artifact: str = "rad750.npz"
    allowed_origins: list[str] = field(default_factory=list)
    api_key: str | None = None

    @property
    def artifacts_dir(self) -> Path:
        return paths.artifacts_dir()

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        return cls(
            host=os.environ.get("NOVUM_API_HOST", "127.0.0.1"),
            port=int(os.environ.get("NOVUM_API_PORT", "8000")),
            log_level=os.environ.get("NOVUM_LOG_LEVEL", "INFO"),
            default_artifact=os.environ.get("NOVUM_DEFAULT_ARTIFACT", "rad750.npz"),
            allowed_origins=_split_csv(os.environ.get("NOVUM_ALLOWED_ORIGINS", "")),
            api_key=(os.environ.get("NOVUM_API_KEY") or "").strip() or None,
        )


def get_settings() -> Settings:
    return Settings.from_env()
