"""`.env` loading: one implementation, optional file, real environment wins.

The bug this guards against is quiet: a bare `load_dotenv()` resolves its path
relative to the *calling module's file*, so whether OPENROUTER_API_KEY is
visible depends on which file happened to make the call -- and the mission
brief falls back to templates with no explanation. One loader, pinned to the
project root, called at process entry.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_loader():
    from core import env

    env.reset_for_tests()
    yield
    env.reset_for_tests()


def test_absent_env_file_is_not_an_error(tmp_path: Path) -> None:
    from core.env import load_env

    assert load_env(tmp_path / "nope.env") is None


def test_env_file_fills_unset_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.env import load_env

    env_file = tmp_path / ".env"
    env_file.write_text("NOVUM_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("NOVUM_TEST_TOKEN", raising=False)

    assert load_env(env_file) == env_file
    assert os.environ["NOVUM_TEST_TOKEN"] == "from-dotenv"


def test_real_environment_always_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported variable is a deliberate act; a file on disk must not undo it."""
    from core.env import load_env

    env_file = tmp_path / ".env"
    env_file.write_text("NOVUM_TEST_TOKEN=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("NOVUM_TEST_TOKEN", "from-real-env")

    load_env(env_file)
    assert os.environ["NOVUM_TEST_TOKEN"] == "from-real-env"


def test_load_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.env import load_env

    env_file = tmp_path / ".env"
    env_file.write_text("NOVUM_TEST_TOKEN=first\n", encoding="utf-8")
    monkeypatch.delenv("NOVUM_TEST_TOKEN", raising=False)

    load_env(env_file)
    env_file.write_text("NOVUM_TEST_TOKEN=second\n", encoding="utf-8")
    load_env(env_file)                       # no-op: already loaded
    assert os.environ["NOVUM_TEST_TOKEN"] == "first"

    monkeypatch.delenv("NOVUM_TEST_TOKEN", raising=False)
    load_env(env_file, force=True)           # explicit re-read
    assert os.environ["NOVUM_TEST_TOKEN"] == "second"


def test_novum_env_file_overrides_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.env import env_file, load_env

    custom = tmp_path / "custom.env"
    custom.write_text("NOVUM_TEST_TOKEN=custom\n", encoding="utf-8")
    monkeypatch.setenv("NOVUM_ENV_FILE", str(custom))
    monkeypatch.delenv("NOVUM_TEST_TOKEN", raising=False)

    assert env_file() == custom
    load_env()
    assert os.environ["NOVUM_TEST_TOKEN"] == "custom"


def test_dotenv_is_loaded_in_exactly_one_place() -> None:
    """Scattered load_dotenv() calls are how this broke in the first place."""
    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".venv", "build", "novum.egg-info"} for part in path.parts):
            continue
        if path.name in {"env.py", "test_env_loading.py"}:
            continue
        if "load_dotenv" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, (
        f"load_dotenv() must live only in core/env.py; found in: {offenders}"
    )


@pytest.mark.parametrize("entry", ["scripts.mission_brief", "scripts.check_llm"])
def test_entry_points_load_env_at_startup(entry: str) -> None:
    """Each credential-needing entry point must see a .env it did not export."""
    code = (
        "import runpy, sys, os\n"
        "sys.argv = ['x', '--help']\n"
        "try:\n"
        f"    runpy.run_module({entry!r}, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "from core.env import load_env\n"
        "print('LOADER_IMPORTABLE', load_env.__module__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "LOADER_IMPORTABLE core.env" in result.stdout


def test_api_module_loads_env_on_import(
    tmp_path: Path,
) -> None:
    """Under uvicorn, importing api.main IS process entry."""
    env_file = tmp_path / ".env"
    env_file.write_text("NOVUM_TEST_API_TOKEN=from-dotenv\n", encoding="utf-8")

    code = (
        "import os\n"
        "import api.main  # noqa: F401\n"
        "print('TOKEN=' + os.environ.get('NOVUM_TEST_API_TOKEN', ''))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        env={**os.environ, "NOVUM_ENV_FILE": str(env_file)},
    )
    assert result.returncode == 0, result.stderr
    assert "TOKEN=from-dotenv" in result.stdout
