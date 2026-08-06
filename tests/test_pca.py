"""The RAD750 tier: fit, score, persist, and the cost accounting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.dataset import ChunkedArray
from core.models.pca import PCANoveltyModel
from core.models.registry import load_model
from core.scoring import roc_auc
from core.transforms import FrameTransform
from tests.conftest import make_novel, make_typical


def _fit(frames: np.ndarray, *, n_components: int = 8, downsample: int = 4, **kwargs):
    transform = FrameTransform(downsample=downsample, standardize="per_band")
    model = PCANoveltyModel(transform, n_components=n_components, **kwargs)
    chunks = ChunkedArray(frames, chunk_size=32)
    return model.fit(chunks, n_samples=len(frames), seed=0)


def test_fit_then_separate_typical_from_novel(rng) -> None:
    typical = make_typical(rng, 200)
    model = _fit(typical)

    scores_typical = model.score(make_typical(rng, 60))
    scores_novel = model.score(make_novel(rng, 60))

    y = np.concatenate([np.zeros(60, dtype=np.int8), np.ones(60, dtype=np.int8)])
    s = np.concatenate([scores_typical, scores_novel])
    # Not ~1.0: this fixture fits at downsample=4, and block-mean pooling
    # smooths away part of the high-frequency structure that marks the novel
    # frames. That trade-off is real, and the tier configs use downsample=2.
    assert roc_auc(y, s) > 0.90


def test_components_are_orthonormal(rng) -> None:
    """The residual score uses ||x||^2 - ||proj||^2, which needs orthonormality."""
    model = _fit(make_typical(rng, 120), n_components=6)
    gram = model.components_ @ model.components_.T
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-4)


def test_residual_matches_an_explicit_reconstruction(rng) -> None:
    """Cross-check the fast identity against the literal reconstruct-and-diff."""
    typical = make_typical(rng, 120)
    model = _fit(typical, n_components=5)

    probe = make_novel(rng, 7)
    fast = model.score(probe)

    z = model.transform.apply(probe) - model.mean_
    reconstruction = (z @ model.components_.T) @ model.components_
    slow = np.linalg.norm(z - reconstruction, axis=1)

    np.testing.assert_allclose(fast, slow, rtol=1e-3, atol=1e-3)


def test_scores_are_finite_and_non_negative(rng) -> None:
    model = _fit(make_typical(rng, 80), n_components=4)
    scores = model.score(make_typical(rng, 20))
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_more_components_never_increase_the_residual(rng) -> None:
    typical = make_typical(rng, 150)
    probe = make_typical(rng, 30)
    small = _fit(typical, n_components=3).score(probe)
    large = _fit(typical, n_components=12).score(probe)
    assert large.mean() <= small.mean() + 1e-6


@pytest.mark.parametrize("mode", ["recon_l2", "recon_mse", "mahalanobis"])
def test_energy_score_modes_rank_novel_above_typical(rng, mode: str) -> None:
    """These three are monotone in distance from the subspace, so direction is fixed."""
    typical = make_typical(rng, 150)
    model = _fit(typical, n_components=6, score_mode=mode)
    scores_typical = model.score(make_typical(rng, 40))
    scores_novel = model.score(make_novel(rng, 40))

    assert np.isfinite(scores_typical).all() and np.isfinite(scores_novel).all()
    assert scores_novel.mean() > scores_typical.mean()


def test_recon_normalized_is_a_bounded_ratio(rng) -> None:
    """The relative residual trades absolute sensitivity for illumination robustness.

    It divides the residual by the frame's own energy, so unlike the energy
    modes it is NOT guaranteed to rank novel frames higher -- a bright, busy
    novel frame can carry a small *relative* residual. Assert only what the
    definition actually promises: a finite ratio in [0, 1].
    """
    model = _fit(make_typical(rng, 150), n_components=6, score_mode="recon_normalized")
    scores = np.concatenate([model.score(make_typical(rng, 30)), model.score(make_novel(rng, 30))])
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0 + 1e-6)).all()


def test_rejects_an_unknown_score_mode() -> None:
    with pytest.raises(ValueError, match="score_mode"):
        PCANoveltyModel(FrameTransform(), score_mode="vibes")


def test_n_components_is_clamped_to_the_available_rank(rng) -> None:
    model = _fit(make_typical(rng, 12), n_components=500, downsample=8)
    assert model.n_components <= 12
    assert model.components_.shape[0] == model.n_components


def test_save_load_round_trip(rng, tmp_path: Path) -> None:
    typical = make_typical(rng, 120)
    model = _fit(typical, n_components=7)
    probe = make_novel(rng, 15)
    expected = model.score(probe)

    path = model.save(tmp_path / "rad750.npz")
    assert path.exists()

    reloaded = load_model(path)
    np.testing.assert_allclose(reloaded.score(probe), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(reloaded.components_, model.components_)
    assert reloaded.n_components == model.n_components
    assert reloaded.score_mode == model.score_mode
    assert reloaded.transform.standardize == model.transform.standardize


def test_cost_accounting_is_reported(rng) -> None:
    model = _fit(make_typical(rng, 100), n_components=8, downsample=4)
    d = model.transform.output_dim  # 16*16*6 = 1536

    # 8 components + the mean + per-band standardisation stats.
    assert model.param_count() == 8 * d + d + 6 + 6

    # The estimate must account for preprocessing as well as the projection.
    # At downsample=4 the transform reads all 64*64*6 raw pixels and actually
    # dominates the 2*k*D projection, which is worth seeing rather than hiding.
    transform_flops = model.transform.flops_per_frame()
    projection_flops = 2 * 8 * d
    total = model.flops_per_inference()

    assert total > transform_flops + projection_flops
    assert total < transform_flops + projection_flops + 10 * d
    assert transform_flops > projection_flops  # preprocessing is the bottleneck here


def test_streaming_and_cached_paths_agree(rng) -> None:
    """Forcing the streaming path must not change the arithmetic."""
    typical = make_typical(rng, 100)
    cached = _fit(typical, n_components=5, memory_budget_bytes=10**9)
    streamed = _fit(typical, n_components=5, memory_budget_bytes=0)

    probe = make_novel(rng, 20)
    np.testing.assert_allclose(cached.score(probe), streamed.score(probe), rtol=1e-4, atol=1e-4)


def test_scoring_before_fitting_raises() -> None:
    model = PCANoveltyModel(FrameTransform())
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.score(np.zeros((1, 64, 64, 6), dtype=np.float32))


def test_explained_variance_is_a_sane_fraction(rng) -> None:
    model = _fit(make_typical(rng, 200), n_components=10)
    assert 0.0 < model.explained_variance_ratio_sum <= 1.0


def test_component_signs_are_canonical(rng) -> None:
    """svd_flip: the largest-|.| element of every component is positive.

    An SVD is defined only up to per-vector sign, and different BLAS backends
    pick different signs. Scores are sign-invariant, but artifact bytes are
    not -- without canonicalisation, content_sha256 could never match across
    macOS/Accelerate and Linux/OpenBLAS even for numerically identical models.
    """
    model = _fit(make_typical(rng, 150), n_components=8)
    pivots = np.argmax(np.abs(model.components_), axis=1)
    pivot_values = model.components_[np.arange(len(pivots)), pivots]
    assert (pivot_values > 0).all()


def test_flipping_a_component_does_not_change_scores(rng) -> None:
    """The invariance svd_flip relies on, checked for every score mode."""
    typical = make_typical(rng, 150)
    probe = make_novel(rng, 20)
    for mode in ("recon_l2", "recon_mse", "recon_normalized", "mahalanobis"):
        model = _fit(typical, n_components=5, score_mode=mode)
        expected = model.score(probe)
        model.components_ = model.components_.copy()
        model.components_[2] *= -1.0  # un-canonicalise one component
        np.testing.assert_allclose(model.score(probe), expected, rtol=1e-6, atol=1e-9)


def test_content_sha256_tracks_the_weights(rng, tmp_path) -> None:
    model = _fit(make_typical(rng, 100), n_components=4)
    original = model.content_sha256()

    # Stable across save/load.
    reloaded = load_model(model.save(tmp_path / "a.npz"))
    assert reloaded.content_sha256() == original

    # And sensitive to the weights actually changing.
    reloaded.components_ = reloaded.components_.copy()
    reloaded.components_[0, 0] += 1e-3
    assert reloaded.content_sha256() != original


def test_content_sha256_is_container_independent(rng, tmp_path) -> None:
    """Two saves of the same model may differ as files; the content hash may not."""
    import time as _time

    model = _fit(make_typical(rng, 80), n_components=3)
    p1 = model.save(tmp_path / "one.npz")
    _time.sleep(1.1)  # zip stores mtimes at 2 s resolution; force them apart
    p2 = model.save(tmp_path / "two.npz")

    from core.models.registry import load_model as _load

    assert _load(p1).content_sha256() == _load(p2).content_sha256()
