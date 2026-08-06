"""RAD750 tier: PCA novelty detection via reconstruction error.

Why PCA is the right floor tier
-------------------------------
A RAD750 runs at ~200 MHz with no vector unit. Scoring a frame must cost a
handful of dot products, not a convolution stack. PCA gives exactly that: the
model is a k x D matrix, inference is one matrix-vector product, and the
novelty score is the energy the subspace fails to explain. If terrain looks
like the terrain the rover has already driven over, the top-k principal
subspace reconstructs it well and the residual is small.

Implementation notes
--------------------
Fitting uses a randomized SVD (Halko, Martinsson & Tropp 2011) driven entirely
through chunked passes over the memory-mapped split, so peak memory is
O(n*l + D*l) rather than O(n*D). At full 64x64x6 resolution the centred design
matrix for train_typical is ~900 MB; the range finder needs ~8 MB. When the
design matrix comfortably fits in RAM the same code path caches it and skips
the repeated disk reads -- identical arithmetic, fewer passes.

There is no scikit-learn here. Not for purity: `core/` is imported by the API,
and the API image has no training dependencies (see tests/test_no_training_deps.py).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger, human_bytes
from ..transforms import FrameTransform
from .base import ARTIFACT_FORMAT_VERSION, NoveltyModel

log = get_logger("novum.models.pca")

SCORE_MODES = ("recon_l2", "recon_mse", "recon_normalized", "mahalanobis")

#: Cache the transformed design matrix in RAM below this size. Above it, stream.
DEFAULT_MEMORY_BUDGET_BYTES = 2 * 1024**3


class _DesignMatrix:
    """Centred design matrix Xc = T(X) - mean, as a set of streaming ops.

    Presents the three products the randomized SVD needs. Whether the data is
    cached in RAM or re-read from the memmap each pass is an implementation
    detail behind this boundary, so the linear algebra below is written once.
    """

    def __init__(
        self,
        chunks: Iterable[np.ndarray],
        transform: FrameTransform,
        *,
        dim: int,
        memory_budget: int = DEFAULT_MEMORY_BUDGET_BYTES,
        n_hint: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._transform = transform
        self.dim = int(dim)
        self.mean: np.ndarray | None = None
        self._cache: np.ndarray | None = None
        self.n = 0

        projected_bytes = (n_hint or 0) * self.dim * 4
        self._want_cache = bool(n_hint) and projected_bytes <= memory_budget
        if self._want_cache:
            log.info(
                "caching design matrix in RAM (%s for %d x %d float32)",
                human_bytes(projected_bytes),
                n_hint,
                self.dim,
            )
        elif n_hint:
            log.info(
                "streaming design matrix from disk (%s exceeds the %s cache budget)",
                human_bytes(projected_bytes),
                human_bytes(memory_budget),
            )

    # -- iteration ----------------------------------------------------------
    def _iter_transformed(self) -> Iterable[np.ndarray]:
        """Yield (n_i, D) float32 blocks of the *uncentred* transformed data."""
        if self._cache is not None:
            step = max(1, 1 << 20 // max(1, self.dim))
            for start in range(0, len(self._cache), step):
                yield self._cache[start : start + step]
            return

        collected: list[np.ndarray] | None = [] if self._want_cache else None
        for raw in self._chunks:
            block = self._transform.apply(raw)
            if collected is not None:
                collected.append(block)
            yield block
        if collected:
            self._cache = np.concatenate(collected, axis=0)

    def _iter_centred(self) -> Iterable[np.ndarray]:
        if self.mean is None:
            raise RuntimeError("mean_pass() must run before centred iteration")
        for block in self._iter_transformed():
            yield block - self.mean

    # -- passes -------------------------------------------------------------
    def mean_pass(self) -> tuple[np.ndarray, float, int]:
        """One pass: feature means, total squared norm, sample count."""
        total = np.zeros(self.dim, dtype=np.float64)
        total_sq = 0.0
        n = 0
        for block in self._iter_transformed():
            b64 = block.astype(np.float64, copy=False)
            total += b64.sum(axis=0)
            total_sq += float(np.einsum("ij,ij->", b64, b64))
            n += len(block)
        if n == 0:
            raise ValueError("training split is empty")
        self.mean = (total / n).astype(np.float32)
        self.n = n
        return self.mean, total_sq, n

    def matmul(self, m: np.ndarray) -> np.ndarray:
        """Xc @ m, for m of shape (D, l). Returns (n, l)."""
        out = np.empty((self.n, m.shape[1]), dtype=np.float64)
        row = 0
        for block in self._iter_centred():
            out[row : row + len(block)] = block @ m
            row += len(block)
        if row != self.n:
            raise RuntimeError(f"chunk source yielded {row} rows, expected {self.n}")
        return out

    def rmatmul(self, q: np.ndarray) -> np.ndarray:
        """Xc.T @ q, for q of shape (n, l). Returns (D, l)."""
        out = np.zeros((self.dim, q.shape[1]), dtype=np.float64)
        row = 0
        for block in self._iter_centred():
            out += block.T.astype(np.float64, copy=False) @ q[row : row + len(block)]
            row += len(block)
        if row != self.n:
            raise RuntimeError(f"chunk source yielded {row} rows, expected {self.n}")
        return out


class PCANoveltyModel(NoveltyModel):
    """Principal-subspace reconstruction-error novelty detector."""

    type_name = "pca"

    def __init__(
        self,
        transform: FrameTransform,
        *,
        n_components: int = 64,
        score_mode: str = "recon_l2",
        oversamples: int = 16,
        power_iterations: int = 2,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
        epsilon: float = 1e-8,
        config: dict | None = None,
    ) -> None:
        super().__init__(transform, config)
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}, got {score_mode!r}")
        if int(n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = int(n_components)
        self.score_mode = score_mode
        self.oversamples = max(0, int(oversamples))
        self.power_iterations = max(0, int(power_iterations))
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.epsilon = float(epsilon)

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.total_variance_: float = 0.0
        self.n_samples_seen_: int = 0

    # -- construction from config ------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, transform: FrameTransform) -> PCANoveltyModel:
        model_cfg = cfg.get("model", {})
        return cls(
            transform,
            n_components=int(model_cfg.get("n_components", 64)),
            score_mode=str(model_cfg.get("score", "recon_l2")),
            oversamples=int(model_cfg.get("oversamples", 16)),
            power_iterations=int(model_cfg.get("power_iterations", 2)),
            memory_budget_bytes=int(
                model_cfg.get("memory_budget_bytes", DEFAULT_MEMORY_BUDGET_BYTES)
            ),
            config=cfg,
        )

    # -- training -----------------------------------------------------------
    def fit(self, chunks: Iterable[np.ndarray], *, n_samples: int, seed: int = 0) -> PCANoveltyModel:
        dim = self.transform.output_dim

        if not self.transform.fitted_:
            log.info("fitting transform statistics (%s)", self.transform.standardize)
            self.transform.fit(chunks)

        source = _DesignMatrix(
            chunks,
            self.transform,
            dim=dim,
            memory_budget=self.memory_budget_bytes,
            n_hint=n_samples,
        )

        log.info("pass 1/%d: computing mean over %d frames", 2 + self.power_iterations, n_samples)
        mean, total_sq, n = source.mean_pass()

        max_rank = min(n, dim)
        k = min(self.n_components, max_rank)
        if k < self.n_components:
            log.warning(
                "n_components=%d exceeds max rank %d (n=%d, D=%d); using %d",
                self.n_components,
                max_rank,
                n,
                dim,
                k,
            )
        sketch_width = min(k + self.oversamples, max_rank)

        rng = np.random.default_rng(seed)
        log.info(
            "randomized SVD: k=%d sketch=%d power_iterations=%d D=%d",
            k,
            sketch_width,
            self.power_iterations,
            dim,
        )

        omega = rng.standard_normal((dim, sketch_width))
        q, _ = np.linalg.qr(source.matmul(omega))

        for i in range(self.power_iterations):
            log.info("power iteration %d/%d", i + 1, self.power_iterations)
            q2, _ = np.linalg.qr(source.rmatmul(q))
            q, _ = np.linalg.qr(source.matmul(q2))

        # B = Q^T Xc, obtained as the transpose of Xc^T Q.
        b = source.rmatmul(q).T
        _, singular_values, vt = np.linalg.svd(b, full_matrices=False)

        self.mean_ = mean.astype(np.float32)
        self.components_ = np.ascontiguousarray(vt[:k], dtype=np.float32)
        self.singular_values_ = singular_values[:k].astype(np.float64)
        denom = max(1, n - 1)
        self.explained_variance_ = (self.singular_values_**2) / denom
        centred_sq = max(0.0, total_sq - n * float(np.dot(mean.astype(np.float64), mean.astype(np.float64))))
        self.total_variance_ = centred_sq / denom
        self.n_samples_seen_ = n
        self.n_components = k
        self.fitted_ = True

        log.info(
            "fitted: %d components capture %.1f%% of training variance",
            k,
            100.0 * self.explained_variance_ratio_sum,
        )
        return self

    @property
    def explained_variance_ratio_sum(self) -> float:
        if self.explained_variance_ is None or self.total_variance_ <= 0:
            return 0.0
        return float(min(1.0, self.explained_variance_.sum() / self.total_variance_))

    # -- inference ----------------------------------------------------------
    def score(self, frames: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.components_ is not None and self.mean_ is not None

        z = self.transform.apply(frames)
        centred = z - self.mean_
        projection = centred @ self.components_.T

        sq_norm = np.einsum("ij,ij->i", centred, centred, dtype=np.float64)
        sq_projection = np.einsum("ij,ij->i", projection, projection, dtype=np.float64)
        # Orthonormal components => residual energy is a subtraction, not a
        # reconstruct-and-diff. Clipped because catastrophic cancellation can
        # push an essentially-zero residual a hair below zero.
        residual = np.maximum(sq_norm - sq_projection, 0.0)

        if self.score_mode == "recon_l2":
            return np.sqrt(residual)
        if self.score_mode == "recon_mse":
            return residual / self.transform.output_dim
        if self.score_mode == "recon_normalized":
            return np.sqrt(residual / (sq_norm + self.epsilon))
        if self.score_mode == "mahalanobis":
            assert self.explained_variance_ is not None
            variance = np.maximum(self.explained_variance_, self.epsilon)
            return np.einsum("ij,ij->i", projection, projection / variance, dtype=np.float64)
        raise ValueError(f"unhandled score_mode {self.score_mode!r}")  # pragma: no cover

    # -- cost accounting ----------------------------------------------------
    def param_count(self) -> int:
        self._require_fitted()
        assert self.components_ is not None and self.mean_ is not None
        n = self.components_.size + self.mean_.size
        if self.transform.mean_ is not None:
            n += self.transform.mean_.size
        if self.transform.std_ is not None:
            n += self.transform.std_.size
        return int(n)

    def flops_per_inference(self) -> int:
        self._require_fitted()
        d = self.transform.output_dim
        k = self.n_components
        cost = self.transform.flops_per_frame()  # preprocessing
        cost += d                                # centring
        cost += 2 * k * d                        # projection onto the subspace
        cost += 2 * d + 2 * k                    # the two squared norms
        return int(cost)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        assert self.components_ is not None and self.mean_ is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "type": self.type_name,
            "n_components": int(self.n_components),
            "score_mode": self.score_mode,
            "epsilon": self.epsilon,
            "total_variance": self.total_variance_,
            "n_samples_seen": self.n_samples_seen_,
            "explained_variance_ratio_sum": self.explained_variance_ratio_sum,
            "transform": self.transform.to_dict(),
        }
        arrays: dict[str, np.ndarray] = {
            "meta": self._pack_meta(meta),
            "mean": self.mean_,
            "components": self.components_,
            "singular_values": self.singular_values_,
            "explained_variance": self.explained_variance_,
        }
        if self.transform.mean_ is not None:
            arrays["transform_mean"] = self.transform.mean_
        if self.transform.std_ is not None:
            arrays["transform_std"] = self.transform.std_

        # np.savez_compressed appends ".npz" to any path that does not already
        # end in it, so the temp name must keep that suffix or the rename below
        # would chase a file numpy never wrote.
        tmp = path.with_name(f".{path.name}.tmp.npz")
        try:
            np.savez_compressed(tmp, **arrays)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        log.info("wrote %s (%s)", path, human_bytes(path.stat().st_size))
        return path

    @classmethod
    def load(cls, path: str | Path) -> PCANoveltyModel:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {path}")
        with np.load(path, allow_pickle=True) as data:
            meta = cls._unpack_meta(data["meta"])
            cls._check_format(meta, cls.type_name)
            transform = FrameTransform.from_dict(
                meta["transform"],
                mean=data["transform_mean"] if "transform_mean" in data else None,
                std=data["transform_std"] if "transform_std" in data else None,
            )
            model = cls(
                transform,
                n_components=int(meta["n_components"]),
                score_mode=str(meta.get("score_mode", "recon_l2")),
            )
            model.mean_ = np.asarray(data["mean"], dtype=np.float32)
            model.components_ = np.asarray(data["components"], dtype=np.float32)
            model.singular_values_ = np.asarray(data["singular_values"], dtype=np.float64)
            model.explained_variance_ = np.asarray(data["explained_variance"], dtype=np.float64)

        model.epsilon = float(meta.get("epsilon", 1e-8))
        model.total_variance_ = float(meta.get("total_variance", 0.0))
        model.n_samples_seen_ = int(meta.get("n_samples_seen", 0))
        model.fitted_ = True

        expected = model.transform.output_dim
        if model.components_.shape[1] != expected:
            raise ValueError(
                f"artifact components have dimension {model.components_.shape[1]} but its "
                f"transform produces {expected}; the artifact is corrupt or mismatched"
            )
        return model
