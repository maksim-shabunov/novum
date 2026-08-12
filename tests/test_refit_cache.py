"""The refit cache must be an optimisation and nothing else.

Building the console grid replays the same mission at six downlink budgets. The
refit pool is every frame captured so far, which depends on the arrival
schedule and not on the budget, so those six runs repeat an identical sequence
of model fits -- seven of them per run on the autoencoder tiers, and by far the
largest cost in the build.

Caching that is only worth doing if it is invisible in the results, so this
asserts equality against the uncached path rather than just measuring a speedup.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.window import SimConfig, refit_chain_key, replay
from tests.test_simulator import _mission


class _LearningModel:
    """A model whose fit genuinely changes how it scores.

    This matters. The obvious fake has a no-op `fit`, and against one of those
    a cache that returned the WRONG model would still produce identical
    results -- the test would pass and prove nothing. This one scores by
    distance from the mean of whatever it was last fitted on, so handing back a
    model fitted on the wrong pool changes the ranking, the selection, and the
    science yield.
    """

    def __init__(self) -> None:
        self.centre = 0.0
        self.n_fits = 0

    def score(self, frames: np.ndarray) -> np.ndarray:
        flat = np.asarray(frames, dtype=np.float64).reshape(len(frames), -1)
        return np.abs(flat.mean(axis=1) - self.centre)

    def flops_per_inference(self) -> int:
        return 866_432

    def fit(self, chunks, *, n_samples: int, seed: int = 0):
        self.n_fits += 1
        total, count = 0.0, 0
        for chunk in chunks:
            block = np.asarray(chunk, dtype=np.float64).reshape(len(chunk), -1)
            total += float(block.sum())
            count += block.size
        self.centre = total / max(count, 1)
        return self


@pytest.fixture
def mission():
    return _mission()


@pytest.fixture
def tiny_model_factory():
    return _LearningModel


# ---------------------------------------------------------------------------
# The chain key
# ---------------------------------------------------------------------------


def test_chain_key_is_order_insensitive_within_a_pool() -> None:
    """The pool is a set of frames; the order they were appended is not data."""
    assert refit_chain_key("m", [3, 1, 2], 0) == refit_chain_key("m", [1, 2, 3], 0)


def test_chain_key_separates_history_seed_and_contents() -> None:
    base = refit_chain_key("m", [1, 2, 3], 0)
    assert base != refit_chain_key("m", [1, 2, 4], 0), "pool contents must matter"
    assert base != refit_chain_key("m", [1, 2, 3], 1), "seed must matter"
    assert base != refit_chain_key("other", [1, 2, 3], 0), "history must matter"


def test_chain_key_depends_on_the_whole_sequence() -> None:
    """Two different refit histories must never collide, even ending alike."""
    a = refit_chain_key(refit_chain_key("m", [1, 2], 0), [3, 4], 0)
    b = refit_chain_key(refit_chain_key("m", [9, 9], 0), [3, 4], 0)
    assert a != b


# ---------------------------------------------------------------------------
# Equality against the uncached path
# ---------------------------------------------------------------------------


def _run(mission, model_factory, budget: float, cache):
    return replay(
        mission,
        model_factory(),
        method="score_first",
        config=SimConfig(
            downlink_fraction=budget,
            adaptation="online",
            bootstrap_sols=0,
            refit_every_windows=2,
            sols_per_window=30,
        ).validate(),
        artifact="test.npz",
        tier="test",
        refit_cache=cache,
    )


def test_cached_and_uncached_runs_agree(mission, tiny_model_factory) -> None:
    budgets = (0.2, 0.35)

    uncached = [_run(mission, tiny_model_factory, b, None) for b in budgets]
    shared: dict = {}
    cached = [_run(mission, tiny_model_factory, b, shared) for b in budgets]

    assert shared, "the cache never stored a refit; the test proves nothing"

    for plain, memo, budget in zip(uncached, cached, budgets, strict=True):
        assert plain.science_yield == memo.science_yield, budget
        assert plain.n_sent == memo.n_sent, budget
        assert plain.n_expired == memo.n_expired, budget
        assert plain.n_refits == memo.n_refits, budget
        assert [w.selected_indices for w in plain.windows] == [
            w.selected_indices for w in memo.windows
        ], f"different frames were transmitted at budget {budget}"


def test_cache_is_reused_across_budgets(mission, tiny_model_factory) -> None:
    """The point of the exercise: the second budget must fit nothing new."""
    shared: dict = {}
    _run(mission, tiny_model_factory, 0.2, shared)
    after_first = len(shared)
    assert after_first > 0

    _run(mission, tiny_model_factory, 0.35, shared)
    assert len(shared) == after_first, (
        "the second budget added cache entries, so the refit sequences differ "
        "and the optimisation does not hold"
    )


def test_a_different_seed_does_not_share_entries(mission, tiny_model_factory) -> None:
    shared: dict = {}
    _run(mission, tiny_model_factory, 0.2, shared)
    n = len(shared)
    replay(
        mission,
        tiny_model_factory(),
        method="score_first",
        config=SimConfig(
            downlink_fraction=0.2,
            adaptation="online",
            bootstrap_sols=0,
            refit_every_windows=2,
            sols_per_window=30,
            seed=7,
        ).validate(),
        artifact="test.npz",
        tier="test",
        refit_cache=shared,
    )
    assert len(shared) > n, "a different seed reused another seed's fitted model"


def test_returned_models_are_copies_not_aliases(mission, tiny_model_factory) -> None:
    """A cached model handed out by reference would be mutated by the next refit."""
    from sim.window import _refit_model

    cache: dict = {}
    first = _refit_model(
        tiny_model_factory(), mission, list(range(40)), 0,
        cache=cache, cache_key="k",
    )
    second = _refit_model(
        tiny_model_factory(), mission, list(range(40)), 0,
        cache=cache, cache_key="k",
    )
    assert first is not second
    assert second is not cache["k"]
    probe = mission.batch(range(4))
    assert np.allclose(first.score(probe), second.score(probe))
