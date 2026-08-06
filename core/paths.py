"""Filesystem layout for NOVUM.

Single source of truth for every path the project reads or writes. Each getter
consults an environment variable first so a remote box can point data/ at a
large scratch volume without editing code. Relative overrides resolve against
the project root, not the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_dir(var: str, default: Path) -> Path:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def data_dir() -> Path:
    """Root of all downloaded and derived data. Gitignored."""
    return _env_dir("NOVUM_DATA_DIR", PROJECT_ROOT / "data")


def raw_dir() -> Path:
    """Downloaded zips and their extracted trees."""
    return data_dir() / "raw"


def processed_dir() -> Path:
    """Memory-mapped float32 arrays plus the manifest."""
    return data_dir() / "processed"


def artifacts_dir() -> Path:
    """Trained weights and published metrics. Committed to git."""
    return _env_dir("NOVUM_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")


def runs_dir() -> Path:
    """Logs, sweep output and per-run metrics. Gitignored."""
    return _env_dir("NOVUM_RUNS_DIR", PROJECT_ROOT / "runs")


def metrics_dir() -> Path:
    return runs_dir() / "metrics"


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


def processed_meta_path() -> Path:
    return processed_dir() / "meta.json"


def manifest_csv_path() -> Path:
    return processed_dir() / "manifest.csv"


def manifest_parquet_path() -> Path:
    return processed_dir() / "manifest.parquet"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def rel(p: Path) -> str:
    """Path relative to the project root when possible, for tidy log lines."""
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
