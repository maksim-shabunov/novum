"""The cheap prefilter: deciding what is even worth scoring.

When the cycle budget cannot cover a full novelty score for every buffered
frame, something has to choose which frames get looked at properly. That
choice cannot itself cost a full score, so it uses two statistics computable
in a couple of passes over the raw pixels:

    spatial   per-frame variance -- texture energy. Veins and broken rock are
              high-variance; flat drift is not.
    spectral  variance across the six band means -- how unusual the colour
              balance is, independent of texture.

Both are z-scored against running mission statistics (Welford, updated as
frames arrive) so they combine on a common scale without a second pass and
without peeking at the future.

COST. ~5 ops per input pixel = ~123k FLOPs per frame, against 866k for a PCA
score and 9.4M for the myriad autoencoder: 14% and 1.3% respectively. Cheap
enough to run on everything, informative enough that triage beats coin-flipping
-- and `prefilter_recall` in the per-window records reports whether it actually
did, rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Ops per input pixel: one pass for the mean, one for the variance, plus the
#: per-band reduction and the two z-scores.
PREFILTER_OPS_PER_PIXEL = 5


def prefilter_flops(frame_shape: tuple[int, ...] = (64, 64, 6)) -> int:
    return int(np.prod(frame_shape)) * PREFILTER_OPS_PER_PIXEL


def frame_statistics(frames: np.ndarray) -> np.ndarray:
    """(N, 2) array of [spatial variance, spectral variance] per frame."""
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    flat = x.reshape(len(x), -1)
    spatial = flat.var(axis=1)
    band_means = x.mean(axis=(1, 2))          # (N, C)
    spectral = band_means.var(axis=1)
    return np.stack([spatial, spectral], axis=1).astype(np.float64)


@dataclass
class RunningStats:
    """Welford accumulator, so z-scoring never needs a second pass.

    Deliberately causal: statistics reflect only frames already captured. Using
    whole-mission statistics would leak the future into a decision the rover
    makes at sol 300.
    """

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if self.mean is None:
            self.mean = np.zeros(values.shape[1])
            self.m2 = np.zeros(values.shape[1])
        for row in values:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    @property
    def std(self) -> np.ndarray:
        if self.mean is None or self.count < 2:
            return np.ones(2 if self.mean is None else len(self.mean))
        return np.sqrt(np.maximum(self.m2 / (self.count - 1), 1e-12))

    def z(self, values: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if self.mean is None or self.count < 2:
            return np.zeros(len(values))
        z = (values - self.mean) / self.std
        # Sum of absolute deviations: unusual in EITHER axis is worth a look.
        return np.abs(z).sum(axis=1)


def prefilter_rank(statistics: np.ndarray, stats: RunningStats) -> np.ndarray:
    """Triage score per frame. Higher = more worth spending a full score on."""
    if len(statistics) == 0:
        return np.zeros(0)
    return stats.z(statistics)
