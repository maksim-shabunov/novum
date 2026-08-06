"""The interface every NOVUM novelty model implements.

The contract is deliberately small, because it has to be satisfiable by both a
PCA fitted with a numpy SVD and a conv autoencoder trained in torch, and
because `score` has to run inside the serving image with numpy alone.

Score convention: **higher means more novel**, always. A model whose natural
output is a similarity must negate it before returning.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..transforms import FrameTransform

ARTIFACT_FORMAT_VERSION = 1


class NoveltyModel(abc.ABC):
    """Base class for all tiers."""

    #: Registry key. Must match the `model.type` used in configs.
    type_name: str = "base"

    def __init__(self, transform: FrameTransform, config: dict | None = None) -> None:
        self.transform = transform
        self.config = config or {}
        self.fitted_ = False

    # -- training -----------------------------------------------------------
    @abc.abstractmethod
    def fit(self, chunks: Iterable[np.ndarray], *, n_samples: int, seed: int = 0) -> NoveltyModel:
        """Fit on an iterable of (n, 64, 64, 6) float32 chunks of typical terrain.

        `chunks` may be re-iterated: implementations that need several passes
        receive a callable-backed iterable from the training script.
        """

    # -- inference ----------------------------------------------------------
    @abc.abstractmethod
    def score(self, frames: np.ndarray) -> np.ndarray:
        """Return a float64 novelty score per frame. Higher = more novel."""

    def score_chunks(self, chunks: Iterable[np.ndarray]) -> np.ndarray:
        """Score a stream of chunks and concatenate the results."""
        parts = [self.score(chunk) for chunk in chunks]
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)

    # -- cost accounting ----------------------------------------------------
    @abc.abstractmethod
    def param_count(self) -> int:
        """Number of stored scalar parameters."""

    @abc.abstractmethod
    def flops_per_inference(self) -> int:
        """Estimated multiply-add count to score one frame, transform included."""

    # -- persistence --------------------------------------------------------
    @abc.abstractmethod
    def save(self, path: str | Path) -> Path:
        """Write weights to a .npz. Must round-trip through `load`."""

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> NoveltyModel:
        """Load weights written by `save`."""

    # -- helpers shared by implementations ----------------------------------
    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted or loaded; "
                "call fit() or use the load() classmethod"
            )

    @staticmethod
    def _pack_meta(meta: dict) -> np.ndarray:
        """npz stores arrays only, so metadata rides along as a JSON scalar."""
        return np.asarray(json.dumps(meta), dtype=object)

    @staticmethod
    def _unpack_meta(arr: np.ndarray) -> dict:
        return json.loads(arr.item() if hasattr(arr, "item") else str(arr))

    @staticmethod
    def _check_format(meta: dict, expected_type: str) -> None:
        version = int(meta.get("format_version", -1))
        if version != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {version} is not supported by this build "
                f"(expected {ARTIFACT_FORMAT_VERSION}); retrain with `make train`"
            )
        got = meta.get("type")
        if got != expected_type:
            raise ValueError(f"artifact holds a {got!r} model, not {expected_type!r}")
