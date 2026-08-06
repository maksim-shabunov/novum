"""Run provenance: what was trained, from what, on which commit, at what cost.

Every artifact gets a sidecar JSON written from here. The point is that six
months from now you can look at artifacts/rad750.npz and know exactly which
config and which commit produced it, and whether it would still fit in the
flight compute budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    """sha256 of the resolved config. Same config => same hash, always."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_commit() -> str | None:
    """Current commit SHA, or None outside a repo (e.g. inside a built image)."""
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes. None if unknown."""
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def git_branch() -> str | None:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ru_maxrss is kilobytes on Linux but bytes on macOS/BSD. Getting this wrong
    reports a 1 GB training run as 1 MB, so the unit is handled explicitly.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def environment_info() -> dict:
    import numpy as np  # local import keeps this module importable bare

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        # NOVUM is CPU-only by contract; record it so a stray CUDA run is obvious.
        "device": "cpu",
    }


class Timer:
    """Wall-clock timer. `with Timer() as t: ...` then `t.elapsed`."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start


@dataclass
class RunProvenance:
    """The sidecar record written next to every artifact."""

    name: str
    tier: str
    config_hash: str
    config: dict
    git_commit: str | None
    git_dirty: bool | None
    git_branch: str | None
    wall_clock_seconds: float
    param_count: int
    flops_per_inference: int
    peak_rss_bytes: int
    n_train_samples: int
    seed: int
    created_utc: str
    environment: dict = field(default_factory=environment_info)
    extra: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        name: str,
        tier: str,
        config: dict,
        wall_clock_seconds: float,
        param_count: int,
        flops_per_inference: int,
        n_train_samples: int,
        seed: int,
        extra: dict | None = None,
    ) -> RunProvenance:
        return cls(
            name=name,
            tier=tier,
            config_hash=config_hash(config),
            config=config,
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            git_branch=git_branch(),
            wall_clock_seconds=float(wall_clock_seconds),
            param_count=int(param_count),
            flops_per_inference=int(flops_per_inference),
            peak_rss_bytes=peak_rss_bytes(),
            n_train_samples=int(n_train_samples),
            seed=int(seed),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            extra=extra or {},
        )

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_branch": self.git_branch,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "param_count": self.param_count,
            "flops_per_inference": self.flops_per_inference,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": round(self.peak_rss_bytes / (1024 * 1024), 1),
            "n_train_samples": self.n_train_samples,
            "seed": self.seed,
            "created_utc": self.created_utc,
            "environment": self.environment,
            "config": self.config,
            **({"extra": self.extra} if self.extra else {}),
        }

    def write(self, path: Path) -> Path:
        """Write the sidecar atomically next to the artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path


def sidecar_path_for(artifact_path: Path) -> Path:
    """artifacts/rad750.npz -> artifacts/rad750.json"""
    return Path(artifact_path).with_suffix(".json")
