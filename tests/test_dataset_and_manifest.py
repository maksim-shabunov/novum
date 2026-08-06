"""Manifest IO, split loading, chunking, and the sim window scaffolding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.dataset import (
    FRAME_SHAPE,
    SPLIT_NOVEL_CANONICAL,
    SPLIT_TEST_TYPICAL,
    SPLIT_TRAIN,
    ChunkedArray,
    ProcessedMeta,
    iter_chunks,
    load_array,
    load_split,
)
from core.manifest import (
    MANIFEST_COLUMNS,
    ManifestRow,
    class_counts,
    group_by_split,
    read_manifest,
    write_manifest,
)
from sim.window import chronological_order, plan_windows, replay


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def test_manifest_columns_are_exactly_the_contract() -> None:
    assert MANIFEST_COLUMNS == ("index", "split", "class", "sol", "source_filename")


def test_manifest_round_trip(tmp_path: Path) -> None:
    rows = [
        ManifestRow(0, "train_typical", "typical", 69, "mcam00487_R0_sol0069_7.npy"),
        ManifestRow(1, "train_typical", "typical", None, "unparseable.npy"),
    ]
    write_manifest(rows, tmp_path / "manifest.csv")
    back = read_manifest(tmp_path / "manifest.csv")
    assert back == rows


def test_a_missing_sol_survives_the_csv_round_trip_as_none(tmp_path: Path) -> None:
    write_manifest([ManifestRow(0, "s", "c", None, "f.npy")], tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv")[0].sol is None


def test_parquet_is_written_when_pyarrow_is_available(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    written = write_manifest(
        [ManifestRow(0, "s", "c", 1, "f.npy")], tmp_path / "m.csv", tmp_path / "m.parquet"
    )
    assert len(written) == 2
    assert (tmp_path / "m.parquet").exists()


def test_reading_a_missing_manifest_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        read_manifest(tmp_path / "nope.csv")


def test_group_by_split_and_class_counts() -> None:
    rows = [
        ManifestRow(0, "a", "typical", 1, "x.npy"),
        ManifestRow(1, "a", "typical", 2, "y.npy"),
        ManifestRow(0, "b", "veins", 3, "z.npy"),
    ]
    grouped = group_by_split(rows)
    assert set(grouped) == {"a", "b"} and len(grouped["a"]) == 2
    assert class_counts(rows) == {"typical": 2, "veins": 1}


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------
def test_loads_the_synthetic_splits(synthetic_processed: Path) -> None:
    meta = ProcessedMeta.load()
    assert set(meta.splits) >= {SPLIT_TRAIN, SPLIT_TEST_TYPICAL, SPLIT_NOVEL_CANONICAL}

    split = load_split(SPLIT_TRAIN)
    assert len(split) == 120
    assert split.array.shape[1:] == FRAME_SHAPE
    assert len(split.rows) == 120


def test_arrays_are_memory_mapped_not_loaded(synthetic_processed: Path) -> None:
    arr = load_array(SPLIT_TRAIN)
    assert isinstance(arr, np.memmap)


def test_an_unknown_split_names_the_available_ones(synthetic_processed: Path) -> None:
    with pytest.raises(KeyError, match="Available"):
        load_array("not_a_split")


def test_missing_processed_data_says_what_to_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NOVUM_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="make data"):
        ProcessedMeta.load()


def test_iter_chunks_covers_every_frame_exactly_once() -> None:
    arr = np.arange(7 * 4 * 4 * 2, dtype=np.float32).reshape(7, 4, 4, 2)
    chunks = list(iter_chunks(arr, chunk_size=3))
    assert [len(c) for c in chunks] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate(chunks), arr)


def test_iter_chunks_yields_writable_copies_of_a_memmap(synthetic_processed: Path) -> None:
    """Models centre their batches in place; a read-only view would explode."""
    arr = load_array(SPLIT_TRAIN)
    chunk = next(iter_chunks(arr, 8))
    chunk -= 1.0  # must not raise
    assert chunk.dtype == np.float32


def test_chunked_array_is_reiterable(synthetic_processed: Path) -> None:
    """PCA makes several passes; a one-shot generator would silently see zero."""
    chunks = ChunkedArray(load_array(SPLIT_TRAIN), chunk_size=16)
    first = sum(len(c) for c in chunks)
    second = sum(len(c) for c in chunks)
    assert first == second == 120 == chunks.n_frames


def test_chunked_array_respects_a_limit(synthetic_processed: Path) -> None:
    chunks = ChunkedArray(load_array(SPLIT_TRAIN), chunk_size=16, limit=40)
    assert sum(len(c) for c in chunks) == 40


def test_chronological_index_puts_unknown_sols_last(synthetic_processed: Path) -> None:
    split = load_split(SPLIT_TRAIN)
    order = split.chronological_index()
    sols = split.sols[order]
    assert (np.diff(sols) >= 0).all()


# ---------------------------------------------------------------------------
# Simulator scaffolding (replay itself is a stub)
# ---------------------------------------------------------------------------
def test_chronological_order_sorts_by_sol_then_index() -> None:
    rows = [
        ManifestRow(0, "s", "c", 100, "a.npy"),
        ManifestRow(1, "s", "c", None, "b.npy"),
        ManifestRow(2, "s", "c", 5, "c.npy"),
    ]
    assert list(chronological_order(rows)) == [2, 0, 1]


def test_plan_windows_groups_by_sol_span() -> None:
    rows = [ManifestRow(i, "s", "c", sol, f"f{i}.npy") for i, sol in enumerate([1, 2, 3, 40, 41])]
    windows = plan_windows(rows, sols_per_window=10)
    assert len(windows) == 2
    assert windows[0].n_candidates == 3
    assert windows[1].first_sol == 40
    assert windows[0].budget.bits > 0


def test_plan_windows_keeps_undated_frames_in_a_final_window() -> None:
    rows = [
        ManifestRow(0, "s", "c", 1, "a.npy"),
        ManifestRow(1, "s", "c", None, "b.npy"),
    ]
    windows = plan_windows(rows, sols_per_window=5)
    assert sum(w.n_candidates for w in windows) == 2
    assert windows[-1].first_sol == -1


def test_plan_windows_on_no_rows() -> None:
    assert plan_windows([]) == []


def test_replay_is_a_marked_stub() -> None:
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        replay(np.zeros((1, 64, 64, 6), dtype=np.float32), [], model=None)
