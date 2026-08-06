"""Myriad X and Snapdragon tiers: convolutional autoencoders. NOT IMPLEMENTED.

=============================================================================
STUB. Every training and inference entry point below raises NotImplementedError
with an actionable message. The tier configs (configs/tier_myriad.yaml,
configs/tier_snapdragon.yaml) are present, valid, and parse correctly, so
`make sweep` walks these tiers, records them as `not_implemented`, and keeps
going. Only the RAD750 (PCA) tier is implemented end to end today.
=============================================================================

Torch is imported lazily inside `_torch()`, never at module scope. That is what
lets `core.models.registry` list these tiers, and lets the serving image import
`core.models` at all, without torch being installed. Keep it that way: a
module-level `import torch` here silently adds ~2 GB to the API image and
breaks tests/test_no_training_deps.py.

Intended design when implemented
--------------------------------
Both tiers share one architecture, scaled by width and depth:

    encoder: [Conv2d(6 -> c, 3, stride 2) + BN + ReLU] x depth  ->  latent
    decoder: mirrored ConvTranspose2d stack                     ->  6 channels
    loss:    MSE reconstruction over typical terrain only
    score:   per-frame reconstruction MSE (higher = more novel)

  Myriad X tier      c=16, depth=2, latent 32   -- ~1-2 W VPU class
  Snapdragon tier    c=32, depth=3, latent 128  -- ~5-10 W SoC class

Training must stay CPU-only (torch.device('cpu') unconditionally, no .cuda()
anywhere) and headless. The reference point from the literature for a conv
autoencoder on this dataset is ROC AUC 0.65 (Kerner et al. 2020), which is the
bar these tiers exist to clear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..transforms import FrameTransform
from .base import NoveltyModel

_STUB_HINT = (
    "Only the RAD750 tier (model.type: pca, configs/tier_rad750.yaml) is implemented. "
    "Train it with:  make train TIER=rad750"
)


def _torch():
    """Import torch on demand, with a useful error if it is absent.

    Never call this at module scope. See the module docstring.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "torch is required for the autoencoder tiers but is not installed. "
            "Install the training extras:  make setup EXTRAS=train,serve,dev\n"
            "The API image intentionally does not ship torch."
        ) from exc
    return torch


class _ConvAutoencoderBase(NoveltyModel):
    """Shared stub body for the two autoencoder tiers."""

    type_name = "conv_ae"
    channels: int = 16
    depth: int = 2
    latent_dim: int = 32

    def __init__(
        self,
        transform: FrameTransform,
        *,
        channels: int | None = None,
        depth: int | None = None,
        latent_dim: int | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(transform, config)
        if channels is not None:
            self.channels = int(channels)
        if depth is not None:
            self.depth = int(depth)
        if latent_dim is not None:
            self.latent_dim = int(latent_dim)

    @classmethod
    def from_config(cls, cfg: dict, transform: FrameTransform) -> _ConvAutoencoderBase:
        model_cfg = cfg.get("model", {})
        return cls(
            transform,
            channels=model_cfg.get("channels"),
            depth=model_cfg.get("depth"),
            latent_dim=model_cfg.get("latent_dim"),
            config=cfg,
        )

    def _not_implemented(self, what: str) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__}.{what} is not implemented yet "
            f"(tier '{self.type_name}'). {_STUB_HINT}"
        )

    # -- every entry point is a marked stub ---------------------------------
    def fit(self, chunks: Iterable[np.ndarray], *, n_samples: int, seed: int = 0):
        raise self._not_implemented("fit")

    def score(self, frames: np.ndarray) -> np.ndarray:
        raise self._not_implemented("score")

    def param_count(self) -> int:
        raise self._not_implemented("param_count")

    def flops_per_inference(self) -> int:
        raise self._not_implemented("flops_per_inference")

    def save(self, path: str | Path) -> Path:
        raise self._not_implemented("save")

    @classmethod
    def load(cls, path: str | Path) -> NoveltyModel:
        raise NotImplementedError(
            f"{cls.__name__}.load is not implemented yet. {_STUB_HINT}"
        )


class MyriadConvAutoencoder(_ConvAutoencoderBase):
    """Small conv autoencoder sized for an Intel Myriad X class VPU. STUB."""

    type_name = "conv_ae_myriad"
    channels = 16
    depth = 2
    latent_dim = 32


class SnapdragonConvAutoencoder(_ConvAutoencoderBase):
    """Larger conv autoencoder sized for a Snapdragon class SoC. STUB."""

    type_name = "conv_ae_snapdragon"
    channels = 32
    depth = 3
    latent_dim = 128
