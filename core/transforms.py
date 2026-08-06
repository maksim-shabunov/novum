"""Frame preprocessing shared by training and serving.

This lives in `core/` on purpose. The transform is fitted during training and
serialised into the artifact, so the serving path applies the *identical*
pipeline without importing anything from the training stack. Any divergence
between train-time and serve-time preprocessing is a silent accuracy bug, so
there is exactly one implementation.

Pipeline, in order:
    (N, 64, 64, 6) float32
      -> divide by `scale`          (raw Mastcam DN are integers in 0..255)
      -> block-mean downsample      (64x64 -> 32x32 at factor 2)
      -> per-band standardisation   (equalises the six filter bands)
      -> flatten to (N, D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

StandardizeMode = Literal["none", "per_band", "global"]
_VALID_MODES = ("none", "per_band", "global")

TRANSFORM_FORMAT_VERSION = 1


def block_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool (N, H, W, C) by an integer factor over H and W.

    Block mean, not striding: striding throws away 3/4 of the photons and makes
    the novelty score noticeably noisier on fine-grained texture like veins.
    """
    if factor <= 1:
        return x
    if x.ndim != 4:
        raise ValueError(f"expected (N, H, W, C), got shape {x.shape}")
    n, h, w, c = x.shape
    if h % factor or w % factor:
        raise ValueError(f"downsample factor {factor} does not divide frame size {h}x{w}")
    return x.reshape(n, h // factor, factor, w // factor, factor, c).mean(axis=(2, 4))


@dataclass
class FrameTransform:
    """Fitted, serialisable preprocessing. numpy only."""

    scale: float = 255.0
    downsample: int = 2
    standardize: StandardizeMode = "per_band"
    frame_shape: tuple[int, int, int] = (64, 64, 6)
    epsilon: float = 1e-6

    # Fitted state (None until fit() runs, or when standardize == "none").
    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)
    fitted_: bool = False

    def __post_init__(self) -> None:
        if self.standardize not in _VALID_MODES:
            raise ValueError(
                f"standardize must be one of {_VALID_MODES}, got {self.standardize!r}"
            )
        if int(self.downsample) < 1:
            raise ValueError(f"downsample must be >= 1, got {self.downsample}")
        self.downsample = int(self.downsample)
        self.scale = float(self.scale)
        if self.scale == 0:
            raise ValueError("scale must be non-zero")

    # -- geometry -----------------------------------------------------------
    @property
    def output_shape(self) -> tuple[int, int, int]:
        h, w, c = self.frame_shape
        return (h // self.downsample, w // self.downsample, c)

    @property
    def output_dim(self) -> int:
        h, w, c = self.output_shape
        return h * w * c

    @property
    def n_bands(self) -> int:
        return self.frame_shape[2]

    # -- fit ----------------------------------------------------------------
    def fit(self, chunks: Iterable[np.ndarray]) -> FrameTransform:
        """Fit standardisation statistics in a single streaming pass."""
        if self.standardize == "none":
            self.mean_ = None
            self.std_ = None
            self.fitted_ = True
            return self

        count = 0
        total = np.zeros(self.n_bands, dtype=np.float64)
        total_sq = np.zeros(self.n_bands, dtype=np.float64)

        for chunk in chunks:
            reduced = self._geometry(chunk)  # (n, h, w, c)
            flat = reduced.reshape(-1, self.n_bands).astype(np.float64, copy=False)
            count += flat.shape[0]
            total += flat.sum(axis=0)
            total_sq += np.einsum("ij,ij->j", flat, flat)

        if count == 0:
            raise ValueError("FrameTransform.fit received no data")

        mean = total / count
        var = np.maximum(total_sq / count - mean**2, 0.0)
        std = np.sqrt(var)

        if self.standardize == "global":
            mean = np.full(self.n_bands, float(mean.mean()))
            std = np.full(self.n_bands, float(np.sqrt(var.mean())))

        # A constant band would divide by zero; leave it untouched instead.
        std = np.where(std < self.epsilon, 1.0, std)

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self.fitted_ = True
        return self

    # -- apply --------------------------------------------------------------
    def _geometry(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, ...]
        if x.shape[1:] != tuple(self.frame_shape):
            raise ValueError(
                f"expected frames of shape {tuple(self.frame_shape)}, got {x.shape[1:]}"
            )
        x = x / np.float32(self.scale)
        return block_downsample(x, self.downsample)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Transform a batch of frames into a (N, D) float32 design matrix."""
        if not self.fitted_:
            raise RuntimeError("FrameTransform.apply called before fit()")
        reduced = self._geometry(x)
        if self.standardize != "none":
            assert self.mean_ is not None and self.std_ is not None
            reduced = (reduced - self.mean_) / self.std_
        return np.ascontiguousarray(reduced.reshape(len(reduced), -1), dtype=np.float32)

    __call__ = apply

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": TRANSFORM_FORMAT_VERSION,
            "scale": self.scale,
            "downsample": self.downsample,
            "standardize": self.standardize,
            "frame_shape": list(self.frame_shape),
            "epsilon": self.epsilon,
            "fitted": self.fitted_,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> FrameTransform:
        version = int(d.get("format_version", TRANSFORM_FORMAT_VERSION))
        if version != TRANSFORM_FORMAT_VERSION:
            raise ValueError(
                f"transform format_version {version} is not supported by this build "
                f"(expected {TRANSFORM_FORMAT_VERSION}); retrain the artifact"
            )
        t = cls(
            scale=float(d.get("scale", 255.0)),
            downsample=int(d.get("downsample", 1)),
            standardize=d.get("standardize", "per_band"),
            frame_shape=tuple(d.get("frame_shape", (64, 64, 6))),
            epsilon=float(d.get("epsilon", 1e-6)),
        )
        t.mean_ = None if mean is None else np.asarray(mean, dtype=np.float32)
        t.std_ = None if std is None else np.asarray(std, dtype=np.float32)
        t.fitted_ = bool(d.get("fitted", True))
        return t

    def flops_per_frame(self) -> int:
        """Rough multiply-add count for one frame through the transform."""
        h, w, c = self.frame_shape
        n_in = h * w * c
        cost = n_in  # scaling
        if self.downsample > 1:
            cost += n_in  # one add per input pixel for the block mean
        if self.standardize != "none":
            cost += 2 * self.output_dim  # subtract + divide
        return int(cost)


def build_transform(cfg: dict) -> FrameTransform:
    """Build an unfitted transform from a config `transform:` block."""
    cfg = cfg or {}
    return FrameTransform(
        scale=float(cfg.get("scale", 255.0)),
        downsample=int(cfg.get("downsample", 1)),
        standardize=cfg.get("standardize", "per_band"),
        frame_shape=tuple(cfg.get("frame_shape", (64, 64, 6))),
    )
