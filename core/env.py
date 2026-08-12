"""Process-entry environment loading. One call, one place, done once.

`.env` is optional by design: every key it can carry has a safe default baked
into the code, so a missing file is never an error. What it must not do is load
*inconsistently* -- a bare ``load_dotenv()`` resolves its path relative to the
calling module's file, so whether a secret is visible depends on which file
happened to make the call. That is why this module pins the path to the project
root and every entry point calls the same function.

REAL ENVIRONMENT ALWAYS WINS. `.env` is a convenience for a developer laptop;
an exported variable, a Docker `-e` flag, or a CI secret is a deliberate act and
overriding it from a file on disk would be a surprise. `override=False` is the
whole rule.

Call it at process entry -- the top of a script's ``main()``, or module import
for the ASGI app, which is the same thing under uvicorn. Library modules read
``os.environ`` and nothing else: a function that quietly loads a file the caller
did not ask for cannot be tested in isolation.
"""

from __future__ import annotations

import os
from pathlib import Path

from core import paths

#: Where `.env` lives, unless NOVUM_ENV_FILE says otherwise.
DEFAULT_ENV_FILE = paths.PROJECT_ROOT / ".env"

_loaded_path: Path | None = None
_have_loaded = False


def env_file() -> Path:
    """The `.env` this process would load. NOVUM_ENV_FILE overrides the default."""
    raw = os.environ.get("NOVUM_ENV_FILE", "").strip()
    if not raw:
        return DEFAULT_ENV_FILE
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (paths.PROJECT_ROOT / p)


def load_env(path: Path | None = None, *, force: bool = False) -> Path | None:
    """Load `.env` into ``os.environ`` without clobbering real variables.

    Returns the file that was loaded, or None when there was nothing to load
    (no file, or python-dotenv not installed). Idempotent: repeated calls are
    a no-op unless ``force`` is set, so an entry point may call it defensively
    without worrying about who called it first.
    """
    global _loaded_path, _have_loaded

    if _have_loaded and not force:
        return _loaded_path

    _have_loaded = True
    _loaded_path = None

    target = Path(path) if path is not None else env_file()
    if not target.is_file():
        return None

    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        # Declared in pyproject, but a serving-only box may have skipped it.
        # A missing convenience is not a reason to refuse to start.
        return None

    load_dotenv(target, override=False)
    _loaded_path = target
    return target


def reset_for_tests() -> None:
    """Forget that anything was loaded. Tests only."""
    global _loaded_path, _have_loaded
    _loaded_path = None
    _have_loaded = False
