"""Shared fixtures.

The synthetic dataset fixture builds a real processed tree -- arrays, manifest
and meta.json written by the same code the pipeline uses -- inside a tmp dir,
with NOVUM_DATA_DIR pointed at it. Tests therefore exercise the actual loaders
rather than a mock of them, and never touch the 2.4 GB real dataset.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import (  # noqa: E402
    FRAME_SHAPE,
    PROCESSED_DTYPE,
    PROCESSED_FORMAT_VERSION,
    SPLIT_NOVEL_BYCLASS,
    SPLIT_NOVEL_CANONICAL,
    SPLIT_TEST_TYPICAL,
    SPLIT_TRAIN,
)
from core.manifest import ManifestRow, write_manifest  # noqa: E402

N_TYPICAL_BASIS = 4


def make_typical(rng: np.random.Generator, n: int) -> np.ndarray:
    """Low-rank terrain: a few shared spatial patterns plus mild noise.

    Amplitude is kept near 120 +/- 25 DN so the clip to [0, 255] almost never
    fires. An earlier version used +/- 40, and the clipping quietly destroyed
    the low-rank structure the whole fixture exists to provide.
    """
    h, w, c = FRAME_SHAPE
    dim = h * w * c

    basis = rng.normal(size=(N_TYPICAL_BASIS, dim)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    weights = rng.normal(size=(n, N_TYPICAL_BASIS)).astype(np.float32)

    # Rescale so each pixel has unit variance before we map into DN units.
    flat = (weights @ basis) * (np.sqrt(dim) / np.sqrt(N_TYPICAL_BASIS))
    flat += rng.normal(scale=0.1, size=flat.shape).astype(np.float32)
    return (120.0 + 25.0 * flat.reshape(n, h, w, c)).clip(0, 255).astype(np.float32)


def make_novel(rng: np.random.Generator, n: int) -> np.ndarray:
    """Off-manifold terrain: full-rank noise the 4-dim basis cannot explain."""
    h, w, c = FRAME_SHAPE
    frames = rng.normal(loc=120.0, scale=25.0, size=(n, h, w, c)).astype(np.float32)
    frames[:, ::3, ::2, :] += 45.0  # high-frequency structure, off the manifold
    return frames.clip(0, 255).astype(np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260806)


@pytest.fixture
def synthetic_processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rng) -> Path:
    """Build a small processed dataset and point NOVUM_DATA_DIR at it."""
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)

    monkeypatch.setenv("NOVUM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("NOVUM_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("NOVUM_RUNS_DIR", str(tmp_path / "runs"))

    plan = {
        SPLIT_TRAIN: (make_typical(rng, 120), "typical"),
        SPLIT_TEST_TYPICAL: (make_typical(rng, 40), "typical"),
        SPLIT_NOVEL_CANONICAL: (make_novel(rng, 30), "veins"),
        SPLIT_NOVEL_BYCLASS: (make_novel(rng, 34), "veins"),
    }

    rows: list[ManifestRow] = []
    splits_meta: dict[str, dict] = {}

    # Each split gets its own filename block: real splits hold distinct frames,
    # and a fixture that reuses names would trip the double-count guard.
    for offset, (split_name, (frames, label)) in enumerate(plan.items()):
        base = 1000 * (offset + 1)
        path = processed / f"{split_name}.npy"
        np.save(path, frames)
        for i in range(len(frames)):
            seq = base + i
            sol = 10 + (i % 17)
            rows.append(
                ManifestRow(
                    index=i,
                    split=split_name,
                    class_=label,
                    sol=sol,
                    source_filename=f"mcam{seq:05d}_R0_sol{sol:04d}_{i}.npy",
                )
            )
        splits_meta[split_name] = {
            "path": path.name,
            "count": len(frames),
            "shape": [len(frames), *FRAME_SHAPE],
            "fingerprint": f"sha256:test-{split_name}",
            "n_unique_files": len(frames),
            "n_unparsed_sol": 0,
        }

    write_manifest(rows, processed / "manifest.csv", processed / "manifest.parquet")
    (processed / "meta.json").write_text(
        json.dumps(
            {
                "format_version": PROCESSED_FORMAT_VERSION,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "synthetic:test",
                "frame_shape": list(FRAME_SHAPE),
                "dtype": PROCESSED_DTYPE,
                "splits": splits_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return processed
