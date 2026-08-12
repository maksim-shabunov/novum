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


def arrays_content_sha256(arrays: dict) -> str:
    """Hash of the model's numerical content, not its container.

    Two .npz files holding identical arrays can differ byte-for-byte (zip
    timestamps, compression level, entry order), so file_sha256 of the artifact
    cannot answer "are these the same weights?". This can: it hashes each
    array's name, dtype, shape and raw bytes in sorted-key order, and nothing
    else. Same weights => same hash, on any platform, in any container.
    """
    import numpy as np  # local import keeps this module importable bare

    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype.str).encode("utf-8"))  # dtype.str includes endianness
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _git_output(*args: str) -> tuple[bool, str]:
    """(the command succeeded, its trimmed stdout).

    Separate from `_git` because EMPTY OUTPUT IS AN ANSWER. `git status
    --porcelain` prints nothing for a clean tree, and collapsing that to None
    made "clean" indistinguishable from "not a repository" -- so `git_dirty`
    could report true or unknown but never false, and no artifact could ever be
    tied to a commit.
    """
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
        return False, ""
    if out.returncode != 0:
        return False, ""
    return True, out.stdout.strip()


def _git(*args: str) -> str | None:
    """Trimmed stdout, or None when the command failed or said nothing."""
    ok, text = _git_output(*args)
    return text or None if ok else None


def git_commit() -> str | None:
    """Current commit SHA, or None outside a repo (e.g. inside a built image)."""
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes.

    False for a clean tree, None only when git could not answer at all -- inside
    a built image, say. The three cases are genuinely different and the sidecar
    records which one applied.
    """
    ok, status = _git_output("status", "--porcelain")
    if not ok:
        return None
    return bool(status)


#: Git state as it was when the process started, captured before anything is
#: written. See `snapshot_git_state`.
_GIT_AT_START: dict[str, object] | None = None


def snapshot_git_state() -> dict[str, object]:
    """Freeze the repository's state at process entry, before any output.

    WHY THIS EXISTS. `git_dirty()` asks the working tree a question, and writing
    an artifact changes the answer. `scripts/train.py` writes `rad750.npz` and
    then records provenance, so the tree it reports on is one the artifact has
    already dirtied -- and because the artifact is tracked, EVERY sidecar records
    `git_dirty: true` no matter how clean the tree was when the run began. The
    field was structurally incapable of ever being false, which is worse than
    useless: it reads like a real warning.

    Capturing at entry answers the question the field is actually asking -- was
    this built from a known commit -- rather than the tautology that a build
    writes files. Idempotent, so any entry point may call it.
    """
    global _GIT_AT_START
    if _GIT_AT_START is None:
        _GIT_AT_START = {
            "git_commit": git_commit(),
            "git_branch": git_branch(),
            "git_dirty": git_dirty(),
        }
    return dict(_GIT_AT_START)


def git_state() -> dict[str, object]:
    """The snapshot if one was taken, otherwise the tree as it is right now."""
    if _GIT_AT_START is not None:
        return dict(_GIT_AT_START)
    return {
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
    }


def reset_git_snapshot() -> None:
    """Drop the snapshot. Tests only."""
    global _GIT_AT_START
    _GIT_AT_START = None


def git_branch() -> str | None:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ru_maxrss is kilobytes on Linux but bytes on macOS/BSD. Getting this wrong
    reports a 1 GB training run as 1 MB, so the unit is handled explicitly.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def _blas_info() -> dict:
    """BLAS backend and thread count actually in use.

    Recorded because they are the two knobs that make "same code, same data,
    different last digits" happen: Accelerate vs OpenBLAS vs MKL sum in
    different orders, and thread count changes reduction trees. threadpoolctl
    inspects the loaded shared libraries, so this reports what is really
    linked, not what the wheel metadata claims.
    """
    try:
        from threadpoolctl import threadpool_info  # noqa: PLC0415

        pools = [p for p in threadpool_info() if p.get("user_api") == "blas"]
        if pools:
            return {
                "blas_backend": ",".join(
                    sorted({str(p.get("internal_api", "?")) for p in pools})
                ),
                "blas_threads": max(int(p.get("num_threads", 0)) for p in pools),
            }
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass

    # Fallback: numpy's build metadata (what it was linked against, which on
    # macOS wheels is Accelerate) and the strongest thread hint available.
    backend = "unknown"
    try:
        import numpy as np  # noqa: PLC0415

        cfg = np.show_config(mode="dicts")  # numpy >= 1.25
        backend = str(cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"))
    except Exception:  # noqa: BLE001
        pass
    threads = os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    return {
        "blas_backend": backend,
        "blas_threads": int(threads) if threads and threads.isdigit() else os.cpu_count(),
    }


def environment_info() -> dict:
    import numpy as np  # local import keeps this module importable bare

    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        # NOVUM is CPU-only by contract; record it so a stray CUDA run is obvious.
        "device": "cpu",
        **_blas_info(),
    }
    # Record torch only when this run actually used it. sys.modules, not an
    # import: a PCA run must not pull torch in just to write a sidecar.
    torch = sys.modules.get("torch")
    if torch is not None:
        info["torch"] = str(torch.__version__)
        info["torch_threads"] = int(torch.get_num_threads())
    return info


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


#: Both sidecars (artifacts/<name>.json and artifacts/metrics/<name>.json) carry
#: this schema marker and share the identity block below, so tooling can join
#: them on (config_hash, content_sha256) without guessing which fields exist.
SIDECAR_SCHEMA_VERSION = 2


def identity_block(
    *,
    name: str,
    tier: str | None,
    model_type: str | None,
    config_hash_: str | None,
    content_sha256: str | None,
    artifact: str | None,
) -> dict:
    """The fields every sidecar shares. One builder so they cannot drift."""
    return {
        "sidecar_schema": SIDECAR_SCHEMA_VERSION,
        "name": name,
        "tier": tier,
        "model_type": model_type,
        "config_hash": config_hash_,
        "content_sha256": content_sha256,
        "artifact": artifact,
        # The state at process entry, not at write time -- writing the artifact
        # dirties the tree it would otherwise be reporting on.
        **git_state(),
    }


def compute_budget_block(
    *,
    flops_per_inference: int,
    cycles_per_flop: float,
    budget_cycles_per_frame: float | None,
) -> dict:
    """The compute-budget verdict. This is the point of the project.

    A model exceeding its tier budget is NOT an error -- "the snapdragon model
    needs 7x the RAD750 cycle budget" is a publishable result, not a failure.
    So this reports plainly and never raises.
    """
    cycles = float(flops_per_inference) * float(cycles_per_flop)
    block = {
        "cycles_per_flop": float(cycles_per_flop),
        "cycles_per_inference": cycles,
        "budget_cycles_per_frame": budget_cycles_per_frame,
    }
    if budget_cycles_per_frame and budget_cycles_per_frame > 0:
        utilisation = cycles / float(budget_cycles_per_frame)
        block["fits_compute_budget"] = bool(cycles <= float(budget_cycles_per_frame))
        block["budget_utilisation"] = utilisation
    else:
        block["fits_compute_budget"] = None
        block["budget_utilisation"] = None
    return block


@dataclass
class RunProvenance:
    """The training sidecar written next to every artifact."""

    name: str
    tier: str
    model_type: str
    config_hash: str
    content_sha256: str | None
    artifact: str | None
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
    compute_budget: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    environment: dict = field(default_factory=environment_info)
    extra: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        name: str,
        tier: str,
        model_type: str,
        config: dict,
        content_sha256: str | None,
        artifact: str | None,
        wall_clock_seconds: float,
        param_count: int,
        flops_per_inference: int,
        n_train_samples: int,
        seed: int,
        training: dict | None = None,
        extra: dict | None = None,
    ) -> RunProvenance:
        compute_cfg = config.get("compute", {}) or {}
        return cls(
            name=name,
            tier=tier,
            model_type=model_type,
            config_hash=config_hash(config),
            content_sha256=content_sha256,
            artifact=artifact,
            config=config,
            **git_state(),
            wall_clock_seconds=float(wall_clock_seconds),
            param_count=int(param_count),
            flops_per_inference=int(flops_per_inference),
            peak_rss_bytes=peak_rss_bytes(),
            n_train_samples=int(n_train_samples),
            seed=int(seed),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            compute_budget=compute_budget_block(
                flops_per_inference=int(flops_per_inference),
                cycles_per_flop=float(compute_cfg.get("cycles_per_flop", 1.0)),
                budget_cycles_per_frame=compute_cfg.get("budget_cycles_per_frame"),
            ),
            training=training or {},
            extra=extra or {},
        )

    def to_json(self) -> dict:
        return {
            **identity_block(
                name=self.name,
                tier=self.tier,
                model_type=self.model_type,
                config_hash_=self.config_hash,
                content_sha256=self.content_sha256,
                artifact=self.artifact,
            ),
            "kind": "training",
            "created_utc": self.created_utc,
            "wall_clock_seconds": round(self.wall_clock_seconds, 3),
            "param_count": self.param_count,
            "flops_per_inference": self.flops_per_inference,
            # The two keys the project contract names explicitly, surfaced at
            # the top level exactly as specified, duplicating compute_budget.
            "fits_compute_budget": self.compute_budget.get("fits_compute_budget"),
            "budget_utilisation": self.compute_budget.get("budget_utilisation"),
            "compute_budget": self.compute_budget,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": round(self.peak_rss_bytes / (1024 * 1024), 1),
            "n_train_samples": self.n_train_samples,
            "seed": self.seed,
            **({"training": self.training} if self.training else {}),
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
