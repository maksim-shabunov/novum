"""The science-vs-rover decomposition, and the taxonomy that defines it."""

from __future__ import annotations

import numpy as np
import pytest

from core import taxonomy
from core.manifest import ManifestRow
from scripts.evaluate import decompose_by_group


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
def test_the_groups_partition_the_known_classes() -> None:
    assert taxonomy.NATURAL_CLASSES == {"veins", "broken-rock", "float", "bedrock", "meteorite"}
    assert taxonomy.ROVER_CLASSES == {"drt", "dump-pile", "drill-hole", "scuff"}
    assert taxonomy.EXCLUDED_CLASSES == {"other", "edge_cases"}
    assert not (taxonomy.NATURAL_CLASSES & taxonomy.ROVER_CLASSES)
    assert not (taxonomy.NATURAL_CLASSES & taxonomy.EXCLUDED_CLASSES)
    assert not (taxonomy.ROVER_CLASSES & taxonomy.EXCLUDED_CLASSES)


def test_split_labels_handles_the_multilabel_join() -> None:
    assert taxonomy.split_labels("drill-hole|dump-pile") == ["drill-hole", "dump-pile"]
    assert taxonomy.split_labels("veins") == ["veins"]
    assert taxonomy.split_labels("") == []
    assert taxonomy.split_labels(None) == []


def test_group_membership_rules() -> None:
    assert taxonomy.groups_for_labels(["veins"]) == {"natural"}
    assert taxonomy.groups_for_labels(["drt"]) == {"rover"}
    assert taxonomy.groups_for_labels(["other"]) == {"excluded"}
    # A frame with two rover labels is one rover frame, not two.
    assert taxonomy.groups_for_labels(["drill-hole", "dump-pile"]) == {"rover"}
    # A hypothetical straddler counts in both groups by rule.
    assert taxonomy.groups_for_labels(["veins", "drt"]) == {"natural", "rover"}
    # Unknown labels from a future dataset revision go to excluded, loudly not silently.
    assert taxonomy.groups_for_labels(["lander-debris"]) == {"excluded"}
    assert taxonomy.groups_for_labels([]) == {"excluded"}


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def _rows(labels: list[str]) -> list[ManifestRow]:
    return [
        ManifestRow(i, "test_novel_all", label, 100 + i, f"f{i}.npy")
        for i, label in enumerate(labels)
    ]


def test_decomposition_separates_the_groups() -> None:
    typical = np.array([0.0, 0.1, 0.2, 0.3])
    # natural frames score high (found), rover frames low (missed).
    novel = np.array([2.0, 2.1, 0.05, 0.06])
    rows = _rows(["veins", "meteorite", "drt", "dump-pile"])

    out = decompose_by_group(typical, novel, rows)

    assert out["natural"]["n"] == 2
    assert out["rover"]["n"] == 2
    assert out["excluded"]["n"] == 0
    assert out["natural"]["roc_auc"] == pytest.approx(1.0)
    assert out["rover"]["roc_auc"] < 0.5


def test_multilabel_frame_counts_once_per_group() -> None:
    typical = np.array([0.0, 0.1])
    novel = np.array([1.0, 2.0])
    rows = _rows(["drill-hole|dump-pile", "veins"])

    out = decompose_by_group(typical, novel, rows)
    assert out["rover"]["n"] == 1      # one frame, two rover labels
    assert out["natural"]["n"] == 1


def test_excluded_classes_are_reported_not_folded_in() -> None:
    typical = np.array([0.0, 0.1])
    novel = np.array([5.0, 5.0, 5.0])
    rows = _rows(["other", "edge_cases", "veins"])

    out = decompose_by_group(typical, novel, rows)
    assert out["excluded"]["n"] == 2
    assert out["excluded"]["label_counts"] == {"other": 1, "edge_cases": 1}
    # And they are NOT inside natural or rover.
    assert out["natural"]["n"] == 1
    assert out["rover"]["n"] == 0
    assert out["rover"]["roc_auc"] is None  # empty group -> no number, not 0.5


def test_group_ns_cover_the_canonical_set_when_no_straddlers() -> None:
    labels = ["veins"] * 3 + ["drt"] * 4 + ["other"]
    out = decompose_by_group(
        np.zeros(5), np.ones(len(labels)), _rows(labels)
    )
    assert out["natural"]["n"] + out["rover"]["n"] + out["excluded"]["n"] == len(labels)


def test_classes_block_documents_the_taxonomy() -> None:
    out = decompose_by_group(np.zeros(2), np.ones(1), _rows(["veins"]))
    assert out["classes"]["natural"] == sorted(taxonomy.NATURAL_CLASSES)
    assert out["classes"]["rover"] == sorted(taxonomy.ROVER_CLASSES)
