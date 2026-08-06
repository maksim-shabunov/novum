"""Ranking metrics. These are hand-checkable, so check them by hand."""

from __future__ import annotations

import numpy as np
import pytest

from core.scoring import (
    average_precision,
    evaluate_scores,
    precision_at_k,
    recall_at_k,
    roc_auc,
)


def test_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(y, s) == pytest.approx(1.0)


def test_inverted_separation() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(y, s) == pytest.approx(0.0)


def test_all_ties_is_exactly_one_half() -> None:
    """Every score identical means no information, and mid-ranks must say so."""
    y = np.array([0, 1, 0, 1])
    s = np.array([5.0, 5.0, 5.0, 5.0])
    assert roc_auc(y, s) == pytest.approx(0.5)


def test_known_value_by_hand() -> None:
    # positives at scores 0.4 and 0.8; negatives at 0.1, 0.5, 0.9.
    # Pairs (pos, neg): 0.4 beats 0.1 only -> 1. 0.8 beats 0.1 and 0.5 -> 2.
    # 3 of 6 pairs -> 0.5
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.4, 0.8, 0.1, 0.5, 0.9])
    assert roc_auc(y, s) == pytest.approx(3 / 6)


def test_partial_ties_count_as_half() -> None:
    y = np.array([1, 0])
    s = np.array([1.0, 1.0])
    assert roc_auc(y, s) == pytest.approx(0.5)


def test_roc_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="one class"):
        roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))


def test_roc_auc_rejects_nan_scores() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        roc_auc(np.array([0, 1]), np.array([0.5, np.nan]))


def test_precision_at_k() -> None:
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert precision_at_k(y, s, 2) == pytest.approx(1.0)
    assert precision_at_k(y, s, 4) == pytest.approx(0.5)
    assert precision_at_k(y, s, 100) == pytest.approx(3 / 5)  # k clamps to n


def test_precision_at_k_breaks_ties_pessimistically() -> None:
    """A tie at the boundary must not be resolved in the model's favour."""
    y = np.array([1, 0])
    s = np.array([1.0, 1.0])
    assert precision_at_k(y, s, 1) == pytest.approx(0.0)


def test_recall_at_k() -> None:
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    assert recall_at_k(y, s, 2) == pytest.approx(2 / 3)
    assert recall_at_k(y, s, 5) == pytest.approx(1.0)


def test_average_precision_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert average_precision(y, s) == pytest.approx(1.0)


def test_evaluate_scores_assembles_everything() -> None:
    typical = np.linspace(0.0, 1.0, 50)
    novel = np.linspace(1.5, 2.5, 10)
    result = evaluate_scores(typical, novel, k_values=(5, 10))
    assert result.n_typical == 50
    assert result.n_novel == 10
    assert result.roc_auc == pytest.approx(1.0)
    assert result.precision_at_k[5] == pytest.approx(1.0)
    assert result.novel_rate == pytest.approx(10 / 60)
    assert set(result.to_json()["precision_at_k"]) == {"5", "10"}


def test_matches_sklearn_when_available() -> None:
    """If sklearn happens to be installed, agree with it to floating point."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    s = rng.normal(size=500) + y * 0.6
    assert roc_auc(y, s) == pytest.approx(sklearn_metrics.roc_auc_score(y, s), abs=1e-12)
    assert average_precision(y, s) == pytest.approx(
        sklearn_metrics.average_precision_score(y, s), abs=1e-12
    )
