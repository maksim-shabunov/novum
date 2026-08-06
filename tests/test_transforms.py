"""Preprocessing must be identical at train time and serve time."""

from __future__ import annotations

import numpy as np
import pytest

from core.transforms import FrameTransform, block_downsample, build_transform


def test_block_downsample_averages_rather_than_strides() -> None:
    x = np.arange(2 * 4 * 4 * 1, dtype=np.float32).reshape(2, 4, 4, 1)
    out = block_downsample(x, 2)
    assert out.shape == (2, 2, 2, 1)
    # Top-left 2x2 block of frame 0 is [[0,1],[4,5]] -> mean 2.5
    assert out[0, 0, 0, 0] == pytest.approx(2.5)


def test_block_downsample_is_a_noop_at_factor_one() -> None:
    x = np.random.default_rng(0).normal(size=(3, 8, 8, 2)).astype(np.float32)
    np.testing.assert_array_equal(block_downsample(x, 1), x)


def test_block_downsample_rejects_an_indivisible_factor() -> None:
    x = np.zeros((1, 5, 5, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="does not divide"):
        block_downsample(x, 2)


def test_output_geometry() -> None:
    t = FrameTransform(downsample=2)
    assert t.output_shape == (32, 32, 6)
    assert t.output_dim == 32 * 32 * 6

    t4 = FrameTransform(downsample=4)
    assert t4.output_dim == 16 * 16 * 6


def test_per_band_standardisation_normalises_each_band(rng) -> None:
    frames = rng.normal(loc=100.0, scale=30.0, size=(200, 64, 64, 6)).astype(np.float32)
    frames[..., 3] *= 5.0  # one band on a wildly different scale

    t = FrameTransform(downsample=2, standardize="per_band")
    t.fit([frames])
    out = t.apply(frames).reshape(200, 32, 32, 6)

    per_band_std = out.std(axis=(0, 1, 2))
    assert np.allclose(per_band_std, per_band_std[0], rtol=0.3)


def test_a_constant_band_does_not_divide_by_zero() -> None:
    frames = np.ones((10, 64, 64, 6), dtype=np.float32) * 50.0
    t = FrameTransform(downsample=2, standardize="per_band")
    t.fit([frames])
    out = t.apply(frames)
    assert np.isfinite(out).all()


def test_apply_before_fit_raises() -> None:
    t = FrameTransform()
    with pytest.raises(RuntimeError, match="before fit"):
        t.apply(np.zeros((1, 64, 64, 6), dtype=np.float32))


def test_standardize_none_needs_no_statistics() -> None:
    t = FrameTransform(downsample=2, standardize="none")
    t.fit([])
    out = t.apply(np.full((2, 64, 64, 6), 255.0, dtype=np.float32))
    assert np.allclose(out, 1.0)


def test_accepts_a_single_unbatched_frame() -> None:
    t = FrameTransform(downsample=2, standardize="none")
    t.fit([])
    assert t.apply(np.zeros((64, 64, 6), dtype=np.float32)).shape == (1, t.output_dim)


def test_rejects_the_wrong_frame_shape() -> None:
    t = FrameTransform(standardize="none")
    t.fit([])
    with pytest.raises(ValueError, match="expected frames of shape"):
        t.apply(np.zeros((2, 32, 32, 6), dtype=np.float32))


def test_serialisation_round_trip(rng) -> None:
    frames = rng.normal(120.0, 25.0, size=(50, 64, 64, 6)).astype(np.float32)
    t = FrameTransform(downsample=2, standardize="per_band")
    t.fit([frames])
    expected = t.apply(frames)

    restored = FrameTransform.from_dict(t.to_dict(), mean=t.mean_, std=t.std_)
    np.testing.assert_allclose(restored.apply(frames), expected, rtol=1e-6)


def test_from_dict_rejects_a_future_format_version() -> None:
    with pytest.raises(ValueError, match="format_version"):
        FrameTransform.from_dict({"format_version": 99})


def test_rejects_an_unknown_standardize_mode() -> None:
    with pytest.raises(ValueError, match="standardize must be"):
        FrameTransform(standardize="fancy")


def test_build_transform_from_a_config_block() -> None:
    t = build_transform({"downsample": 4, "standardize": "none", "scale": 1.0})
    assert t.downsample == 4 and t.standardize == "none" and t.scale == 1.0


def test_flops_estimate_grows_with_output_dimension() -> None:
    assert FrameTransform(downsample=1).flops_per_frame() > FrameTransform(
        downsample=4
    ).flops_per_frame()
