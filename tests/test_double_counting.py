"""The test_novel double-counting guard.

These are the assertions that stop the evaluation set silently doubling. The
numbers in the docstrings were verified against the real Zenodo archive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.dataset import (
    SPLIT_NOVEL_BYCLASS,
    SPLIT_NOVEL_CANONICAL,
    SPLIT_TEST_TYPICAL,
    DoubleCountError,
    assert_no_double_count,
    assert_splits_combinable,
    concat_splits,
)
from scripts.preprocess import (
    Sample,
    SplitPlan,
    _assert_novel_not_double_counted,
    plan_novel_splits,
)


# ---------------------------------------------------------------------------
# Primitive guards
# ---------------------------------------------------------------------------
def test_assert_no_double_count_passes_on_disjoint_groups() -> None:
    assert_no_double_count({"a": ["x.npy", "y.npy"], "b": ["z.npy"]})


def test_assert_no_double_count_flags_a_shared_filename() -> None:
    with pytest.raises(DoubleCountError, match="Double counting"):
        assert_no_double_count({"a": ["x.npy"], "b": ["x.npy"]})


def test_combining_the_two_novel_splits_is_refused() -> None:
    with pytest.raises(DoubleCountError, match="same underlying frames"):
        assert_splits_combinable([SPLIT_NOVEL_CANONICAL, SPLIT_NOVEL_BYCLASS])


def test_combining_unrelated_splits_is_allowed() -> None:
    assert_splits_combinable([SPLIT_TEST_TYPICAL, SPLIT_NOVEL_CANONICAL])


def test_concat_splits_refuses_the_overlapping_pair(synthetic_processed: Path) -> None:
    with pytest.raises(DoubleCountError):
        concat_splits([SPLIT_NOVEL_CANONICAL, SPLIT_NOVEL_BYCLASS])


def test_concat_splits_works_for_disjoint_splits(synthetic_processed: Path) -> None:
    combined = concat_splits([SPLIT_TEST_TYPICAL, SPLIT_NOVEL_CANONICAL])
    assert len(combined) == 40 + 30
    assert len(combined.rows) == 70
    assert [r.index for r in combined.rows] == list(range(70))


# ---------------------------------------------------------------------------
# The preprocessing-level assertion
# ---------------------------------------------------------------------------
def _plan(name: str, entries: list[tuple[str, str]]) -> SplitPlan:
    return SplitPlan(name, [Sample(Path(p), label) for p, label in entries])


def test_all_folder_leaking_into_the_per_class_walk_is_fatal() -> None:
    """This is the exact failure a recursive glob over test_novel/ produces."""
    canonical = _plan(SPLIT_NOVEL_CANONICAL, [("test_novel/all/a.npy", "veins")])
    leaked = _plan(
        SPLIT_NOVEL_BYCLASS,
        [("test_novel/veins/a.npy", "veins"), ("test_novel/all/a.npy", "veins")],
    )
    with pytest.raises(DoubleCountError, match="leaked into the per-class split"):
        _assert_novel_not_double_counted(canonical, leaked)


def test_duplicate_filenames_in_the_canonical_set_are_fatal() -> None:
    canonical = _plan(
        SPLIT_NOVEL_CANONICAL,
        [("test_novel/all/a.npy", "veins"), ("test_novel/all/nested/a.npy", "veins")],
    )
    with pytest.raises(DoubleCountError, match="duplicate filenames"):
        _assert_novel_not_double_counted(canonical, None)


def test_multi_label_frames_are_allowed_in_the_per_class_split() -> None:
    """Five real frames sit in two class folders each. That is data, not a bug."""
    canonical = _plan(SPLIT_NOVEL_CANONICAL, [("test_novel/all/a.npy", "drill-hole|dump-pile")])
    byclass = _plan(
        SPLIT_NOVEL_BYCLASS,
        [("test_novel/drill-hole/a.npy", "drill-hole"), ("test_novel/dump-pile/a.npy", "dump-pile")],
    )
    _assert_novel_not_double_counted(canonical, byclass)  # must not raise


def test_repeated_path_in_the_per_class_split_is_fatal() -> None:
    byclass = _plan(
        SPLIT_NOVEL_BYCLASS,
        [("test_novel/veins/a.npy", "veins"), ("test_novel/veins/a.npy", "veins")],
    )
    with pytest.raises(DoubleCountError, match="same path more than once"):
        _assert_novel_not_double_counted(None, byclass)


# ---------------------------------------------------------------------------
# End-to-end over a fake raw tree with the real directory shape
# ---------------------------------------------------------------------------
def test_plan_novel_splits_separates_all_from_the_class_folders(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "test_novel"
    frame = np.zeros((64, 64, 6), dtype=np.float64)

    for folder, names in {
        "all": ["f1.npy", "f2.npy", "f3.npy"],
        "veins": ["f1.npy", "f2.npy"],
        "meteorite": ["f3.npy"],
        # f2 carries two labels, exactly like the five real multi-label frames.
        "drt": ["f2.npy", "f4.npy"],
    }.items():
        (raw / folder).mkdir(parents=True)
        for name in names:
            np.save(raw / folder / name, frame)

    canonical, byclass, labels = plan_novel_splits(tmp_path / "raw")
    assert canonical is not None and byclass is not None

    # Canonical is exactly all/, once each.
    assert canonical.n == 3
    assert sorted(canonical.basenames) == ["f1.npy", "f2.npy", "f3.npy"]

    # Per-class keeps one row per (frame, label) and never touches all/.
    assert byclass.n == 5
    assert all("all" not in Path(s.path).parts[:-1] for s in byclass.samples)

    # A multi-label frame gets a joined label in the canonical set.
    joined = {s.path.name: s.class_label for s in canonical.samples}
    assert joined["f2.npy"] == "drt|veins"
    assert labels["f2.npy"] == ["drt", "veins"] or set(labels["f2.npy"]) == {"drt", "veins"}

    # f4 exists only in a class folder, so it stays out of the canonical set.
    assert "f4.npy" not in canonical.basenames

    _assert_novel_not_double_counted(canonical, byclass)


def test_a_recursive_glob_would_have_double_counted(tmp_path: Path) -> None:
    """Demonstrates the bug the guard exists to prevent."""
    raw = tmp_path / "raw" / "test_novel"
    frame = np.zeros((64, 64, 6), dtype=np.float64)
    for folder, names in {"all": ["a.npy", "b.npy"], "veins": ["a.npy", "b.npy"]}.items():
        (raw / folder).mkdir(parents=True)
        for name in names:
            np.save(raw / folder / name, frame)

    naive = sorted((tmp_path / "raw" / "test_novel").rglob("*.npy"))
    canonical, byclass, _ = plan_novel_splits(tmp_path / "raw")

    assert len(naive) == 4          # what a recursive glob returns
    assert canonical.n == 2         # what the evaluation set actually is
    assert byclass.n == 2
