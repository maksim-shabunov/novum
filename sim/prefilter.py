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


# ---------------------------------------------------------------------------
# Pluggable prefilters
# ---------------------------------------------------------------------------
@dataclass
class Prefilter:
    """A triage statistic plus its honest FLOP cost.

    `statistics` returns a (N, k) feature block that RunningStats z-scores; the
    ranking is the summed absolute z-score, so a frame unusual in ANY feature
    is worth a full look.
    """

    name: str
    flops: int
    _fn: object

    def statistics(self, frames: np.ndarray) -> np.ndarray:
        return self._fn(frames)  # type: ignore[operator]


def _variance_prefilter(frame_shape) -> Prefilter:
    return Prefilter("variance", prefilter_flops(frame_shape), frame_statistics)


def _lowrank_prefilter(model, frame_shape, n_components: int) -> Prefilter | None:
    """Rank by the residual of a SEVERELY truncated principal subspace.

    The idea being tested: the variance statistic is cheap but only loosely
    related to what the model actually scores. Reusing the model's own top few
    components should rank frames much closer to the way the full model would,
    at a comparable cost -- a k=4 projection costs 2*4*D against 2*64*D for the
    full score, so about a sixteenth of it.

    Returns None when the model exposes no components (the autoencoder tiers),
    and the caller falls back to variance rather than pretending otherwise.
    """
    components = getattr(model, "components_", None)
    transform = getattr(model, "transform", None)
    mean = getattr(model, "mean_", None)
    if components is None or transform is None or mean is None:
        return None

    k = max(1, min(int(n_components), len(components)))
    basis = np.ascontiguousarray(components[:k])
    dim = basis.shape[1]

    def statistics(frames: np.ndarray) -> np.ndarray:
        z = transform.apply(frames)
        centred = z - mean
        projection = centred @ basis.T
        sq_norm = np.einsum("ij,ij->i", centred, centred, dtype=np.float64)
        sq_proj = np.einsum("ij,ij->i", projection, projection, dtype=np.float64)
        residual = np.sqrt(np.maximum(sq_norm - sq_proj, 0.0))
        # Second feature keeps the interface identical to the variance filter
        # and adds the in-subspace distance, which is cheap once projected.
        return np.stack([residual, np.abs(projection).sum(axis=1)], axis=1)

    flops = int(transform.flops_per_frame() + dim + 2 * k * dim + 2 * dim + 2 * k)
    return Prefilter(f"lowrank(k={k})", flops, statistics)


def build_prefilter(mode: str, model, frame_shape, *, n_components: int = 4) -> Prefilter:
    """Resolve a prefilter by name, falling back loudly rather than silently."""
    if mode == "variance":
        return _variance_prefilter(frame_shape)
    if mode == "lowrank":
        built = _lowrank_prefilter(model, frame_shape, n_components)
        if built is not None:
            return built
        # No components to borrow (autoencoder tiers): say so via the name so
        # the report cannot claim a lowrank prefilter that never ran.
        fallback = _variance_prefilter(frame_shape)
        return Prefilter("variance (lowrank unavailable)", fallback.flops, fallback._fn)
    raise ValueError(f"unknown prefilter mode {mode!r}; expected variance|lowrank")
