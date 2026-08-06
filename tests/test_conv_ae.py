"""The autoencoder tiers: fit, determinism, persistence, cost accounting.

Every test uses a deliberately tiny network (base 4-8 channels, few epochs) so
the whole file runs in seconds. Architecture-correctness does not need a big
model; the full-size tiers are exercised by the real sweep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="train extras not installed")

from core.dataset import ChunkedArray  # noqa: E402
from core.models.conv_ae import (  # noqa: E402
    MyriadConvAutoencoder,
    SnapdragonConvAutoencoder,
    _ConvAutoencoderBase,
)
from core.models.registry import load_model  # noqa: E402
from core.scoring import roc_auc  # noqa: E402
from core.transforms import FrameTransform  # noqa: E402
from tests.conftest import make_novel, make_typical  # noqa: E402


def _tiny(frames: np.ndarray, seed: int = 0, **kwargs) -> MyriadConvAutoencoder:
    defaults = dict(
        channels=8,
        depth=2,
        latent_dim=16,
        epochs=4,
        batch_size=64,
        early_stopping_patience=3,
        val_fraction=0.15,
    )
    defaults.update(kwargs)
    transform = FrameTransform(downsample=1, standardize="per_band")
    model = MyriadConvAutoencoder(transform, **defaults)
    model.fit(ChunkedArray(frames, 64), n_samples=len(frames), seed=seed)
    return model


@pytest.fixture
def trained(rng, monkeypatch, tmp_path):
    # Isolate from the real dataset: _resolve_validation would otherwise find
    # the real validation_typical split and validate synthetic terrain against
    # actual Mastcam frames.
    monkeypatch.setenv("NOVUM_DATA_DIR", str(tmp_path / "nodata"))
    return _tiny(make_typical(rng, 240))


def test_fit_separates_typical_from_novel(trained, rng) -> None:
    scores_typical = trained.score(make_typical(rng, 40))
    scores_novel = trained.score(make_novel(rng, 40))
    y = np.concatenate([np.zeros(40, dtype=np.int8), np.ones(40, dtype=np.int8)])
    assert roc_auc(y, np.concatenate([scores_typical, scores_novel])) > 0.8


def test_scores_are_finite_and_positive(trained, rng) -> None:
    scores = trained.score(make_typical(rng, 16))
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
    assert scores.dtype == np.float64


def test_training_is_deterministic_for_a_fixed_seed(rng, monkeypatch, tmp_path) -> None:
    """Same seed => identical weights, byte for byte. The contract is seeded
    torch, seeded numpy, and seeded batch ordering; this is where it is held."""
    monkeypatch.setenv("NOVUM_DATA_DIR", str(tmp_path / "nodata"))
    frames = make_typical(rng, 200)
    a = _tiny(frames, seed=7)
    b = _tiny(frames, seed=7)
    assert a.content_sha256() == b.content_sha256()

    probe = make_novel(np.random.default_rng(1), 12)
    np.testing.assert_array_equal(a.score(probe), b.score(probe))


def test_different_seeds_give_different_weights(rng, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOVUM_DATA_DIR", str(tmp_path / "nodata"))
    frames = make_typical(rng, 200)
    assert _tiny(frames, seed=0).content_sha256() != _tiny(frames, seed=1).content_sha256()


def test_save_load_round_trip(trained, rng, tmp_path: Path) -> None:
    probe = make_novel(rng, 10)
    expected = trained.score(probe)

    path = trained.save(tmp_path / "ae.npz")
    reloaded = load_model(path)

    assert type(reloaded) is MyriadConvAutoencoder
    np.testing.assert_allclose(reloaded.score(probe), expected, rtol=1e-6, atol=1e-7)
    assert reloaded.content_sha256() == trained.content_sha256()
    assert reloaded.fit_info_.get("epochs_run") == trained.fit_info_.get("epochs_run")


def test_artifact_meta_is_readable_without_torch_semantics(trained, tmp_path: Path) -> None:
    """The npz stays a plain numpy container: listing it must not need torch."""
    from core.models.registry import read_artifact_meta

    path = trained.save(tmp_path / "ae.npz")
    meta = read_artifact_meta(path)
    assert meta["type"] == "conv_ae_myriad"
    assert meta["transform"]["standardize"] == "per_band"


def test_param_count_matches_torch(trained) -> None:
    reported = trained.param_count()
    torch_params = sum(int(p.numel()) for p in trained.net.state_dict().values())
    assert reported == torch_params + 12  # + per-band transform mean/std (6+6)


def test_flops_scale_with_architecture(rng) -> None:
    """FLOPs are computed analytically; bigger nets must cost more, and the
    known myriad/snapdragon sizes must land where the sidecars claim."""
    t = FrameTransform(downsample=1, standardize="per_band")
    small = MyriadConvAutoencoder(t, channels=16, depth=2, latent_dim=32)
    large = SnapdragonConvAutoencoder(t, channels=32, depth=3, latent_dim=128)

    f_small = small.flops_per_inference()
    f_large = large.flops_per_inference()
    assert f_large > 4 * f_small
    assert 8_000_000 < f_small < 12_000_000       # ~9.5 MFLOP by hand
    assert 45_000_000 < f_large < 55_000_000      # ~49 MFLOP by hand


def test_fit_requires_random_access(rng) -> None:
    t = FrameTransform(downsample=1, standardize="per_band")
    model = MyriadConvAutoencoder(t, channels=4, depth=1, latent_dim=4, epochs=1)
    with pytest.raises(TypeError, match="random access"):
        model.fit(iter([make_typical(rng, 8)]), n_samples=8)


def test_depth_must_divide_the_frame(rng) -> None:
    t = FrameTransform(downsample=1, standardize="per_band")
    with pytest.raises(ValueError, match="does not divide"):
        MyriadConvAutoencoder(t, depth=7)


def test_early_stopping_restores_the_best_epoch(rng, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NOVUM_DATA_DIR", str(tmp_path / "nodata"))
    model = _tiny(make_typical(rng, 200), epochs=6, early_stopping_patience=2)
    info = model.fit_info_
    assert info["epochs_run"] <= 6
    assert 1 <= info["best_epoch"] <= info["epochs_run"]
    assert info["val_source"] == "train_holdout"
    assert np.isfinite(info["best_val_loss"])


def test_scoring_before_fitting_raises() -> None:
    t = FrameTransform(downsample=1, standardize="per_band")
    model = MyriadConvAutoencoder(t)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.score(np.zeros((1, 64, 64, 6), dtype=np.float32))


def test_rejects_an_unknown_score_mode() -> None:
    t = FrameTransform(downsample=1, standardize="per_band")
    with pytest.raises(ValueError, match="score_mode"):
        MyriadConvAutoencoder(t, score_mode="vibes")


def test_tier_subclasses_declare_their_sizes() -> None:
    assert MyriadConvAutoencoder.type_name == "conv_ae_myriad"
    assert SnapdragonConvAutoencoder.type_name == "conv_ae_snapdragon"
    assert issubclass(MyriadConvAutoencoder, _ConvAutoencoderBase)
    assert SnapdragonConvAutoencoder.channels > MyriadConvAutoencoder.channels
