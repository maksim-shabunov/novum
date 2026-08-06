"""Ranking metrics, implemented in numpy so the serving image can use them.

scikit-learn would give us roc_auc_score for free, but sklearn is a training
dependency and `core/` must stay importable from the API. These are exact
implementations, not approximations: `roc_auc` uses the Mann-Whitney U identity
with proper mid-rank tie handling and agrees with sklearn to floating point.

Convention throughout NOVUM: **higher score means more novel**. For a
reconstruction-error model the score is the error itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


def _validate(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true and scores must be the same length, got {y.shape} and {s.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isfinite(s).all():
        n_bad = int((~np.isfinite(s)).sum())
        raise ValueError(f"{n_bad} non-finite score(s); the model produced NaN or inf")
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(f"y_true must be binary 0/1, found values {uniq[:5]}")
    return y.astype(np.int8), s


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve. Positive class (1) = novel.

    Equivalent to the probability that a randomly drawn novel frame scores
    above a randomly drawn typical frame, counting ties as half.
    """
    y, s = _validate(y_true, scores)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"ROC AUC is undefined with one class only (positives={n_pos}, negatives={n_neg})"
        )

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    # Mid-ranks: average rank within each group of tied scores.
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1

    rank_of = np.empty(s.size, dtype=np.float64)
    rank_of[order] = ranks
    sum_pos_ranks = float(rank_of[y == 1].sum())
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scoring frames that are truly novel.

    This is the operationally meaningful metric: k is how many frames actually
    fit in the downlink window, so precision@k is the fraction of transmitted
    bits that carried something worth looking at.

    Ties at the k-th boundary are broken by ascending score then index, which
    is the pessimistic choice (a tied typical frame is preferred over a tied
    novel one), so the number never flatters the model.
    """
    y, s = _validate(y_true, scores)
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, y.size)
    order = np.lexsort((np.arange(y.size), y, -s))
    return float(y[order[:k]].sum() / k)


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of all novel frames captured in the top k."""
    y, s = _validate(y_true, scores)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("recall@k is undefined with no positives")
    k = min(max(1, int(k)), y.size)
    order = np.lexsort((np.arange(y.size), y, -s))
    return float(y[order[:k]].sum() / n_pos)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation)."""
    y, s = _validate(y_true, scores)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("average precision is undefined with no positives")
    order = np.lexsort((np.arange(y.size), y, -s))
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y_sorted).sum() / n_pos)


@dataclass
class EvalResult:
    """Everything evaluate.py reports for one artifact."""

    n_typical: int
    n_novel: int
    roc_auc: float
    average_precision: float
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    novel_rate: float = 0.0
    per_class_roc_auc: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["precision_at_k"] = {str(k): v for k, v in self.precision_at_k.items()}
        d["recall_at_k"] = {str(k): v for k, v in self.recall_at_k.items()}
        return d


def evaluate_scores(
    typical_scores: np.ndarray,
    novel_scores: np.ndarray,
    k_values: list[int] | tuple[int, ...] = (10, 25, 50, 100),
) -> EvalResult:
    """Compute the full metric set from two score vectors."""
    typical_scores = np.asarray(typical_scores, dtype=np.float64).ravel()
    novel_scores = np.asarray(novel_scores, dtype=np.float64).ravel()

    y = np.concatenate(
        [np.zeros(typical_scores.size, dtype=np.int8), np.ones(novel_scores.size, dtype=np.int8)]
    )
    s = np.concatenate([typical_scores, novel_scores])

    ks = sorted({int(k) for k in k_values if int(k) > 0})
    return EvalResult(
        n_typical=int(typical_scores.size),
        n_novel=int(novel_scores.size),
        roc_auc=roc_auc(y, s),
        average_precision=average_precision(y, s),
        precision_at_k={k: precision_at_k(y, s, k) for k in ks},
        recall_at_k={k: recall_at_k(y, s, k) for k in ks},
        novel_rate=float(y.mean()),
    )
