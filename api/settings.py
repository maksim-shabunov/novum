"""API configuration, read from the environment (and .env if present).

Secrets never live in code or in a committed file. `.env` is gitignored;
`.env.example` documents every key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core import paths
from core.env import load_env


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


#: The console runs on its own port and calls this API cross-origin, so without
#: this the briefing panel is dead on arrival in every default setup -- local
#: dev, `docker compose up`, and a judge's laptop alike.
#:
#: ANY loopback port, not a fixed list. A hardcoded :3000 works until someone
#: runs the console on another port, and then the brief fails with a CORS error
#: in a console nobody has open. A loopback origin is by definition already on
#: this machine, so allowing it grants nothing the host did not already have.
#: Anything public still has to be named explicitly in NOVUM_ALLOWED_ORIGINS.
LOCAL_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"


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

    @property
    def cors_origin_regex(self) -> str | None:
        """Loopback origins, unless this is a real deployment.

        With NOVUM_API_KEY set the thing is not a laptop any more, so the
        convenience default drops away and every origin must be named
        explicitly in NOVUM_ALLOWED_ORIGINS.
        """
        return None if self.auth_enabled else LOCAL_ORIGIN_REGEX

    @classmethod
    def from_env(cls) -> Settings:
        load_env()   # no-op after the first call; see core.env
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
