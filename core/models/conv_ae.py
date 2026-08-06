"""Myriad and Snapdragon tiers: convolutional autoencoders, CPU-only.

Both tiers share one architecture, scaled by width and depth:

    encoder   [Conv2d(k=3, stride 2) + BatchNorm + ReLU] x depth
    latent    dense bottleneck (the compression that makes novelty visible)
    decoder   mirrored ConvTranspose2d stack, plain final layer
    loss      MSE reconstruction over typical terrain only
    score     per-frame reconstruction MSE (higher = more novel)

The score is reconstruction error in the same standardized space the PCA tier
scores in, so the cross-tier comparison is like-for-like: every tier asks "how
badly does a model of typical terrain explain this frame?" and differs only in
how expressive that model is.

  Myriad tier      base 16, depth 2, latent 32   (~544k params, ~9.5 MFLOP)
                   sized for the Intel Movidius Myriad 2 VPU that ESA flew on
                   Phi-Sat-1 -- the first AI accelerator on an ESA mission
  Snapdragon tier  base 32, depth 3, latent 128  (~2.3M params, ~49 MFLOP)
                   deliberately overprovisioned relative to real flight
                   hardware; exists to answer whether more compute buys accuracy

CPU-only by contract: the device is torch.device("cpu") unconditionally and
nothing here touches CUDA. Training must be deterministic given a seed: torch
is seeded, numpy is seeded, and batch order comes from a seeded numpy Generator
(there is no torch DataLoader precisely so that ordering is owned here).

Torch is imported lazily inside `_torch()`, never at module scope. That is what
lets `core.models.registry` list these tiers, and lets the serving image import
`core.models`, without torch installed. A module-level `import torch` here
silently adds ~2 GB to the API image and breaks tests/test_no_training_deps.py.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger, human_bytes, human_duration
from ..transforms import FrameTransform
from .base import ARTIFACT_FORMAT_VERSION, NoveltyModel

log = get_logger("novum.models.conv_ae")

SCORE_MODES = ("recon_mse", "recon_l2")

#: Fallback when no validation split exists (synthetic test fixtures): hold
#: this fraction of the training frames out for early stopping.
DEFAULT_VAL_FRACTION = 0.1


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


def _build_net(torch, *, in_channels: int, base: int, depth: int, latent_dim: int, spatial: int):
    """Construct the encoder/decoder pair. Pure function of the config."""
    nn = torch.nn
    if spatial % (2**depth):
        raise ValueError(f"depth {depth} does not divide frame size {spatial}")
    chans = [base * (2**i) for i in range(depth)]
    feat_hw = spatial // (2**depth)
    feat = chans[-1] * feat_hw * feat_hw

    encoder: list = []
    prev = in_channels
    for c in chans:
        encoder += [
            nn.Conv2d(prev, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        ]
        prev = c

    decoder: list = []
    reversed_chans = list(reversed(chans))
    for i, c in enumerate(reversed_chans):
        last = i == len(reversed_chans) - 1
        out_c = in_channels if last else reversed_chans[i + 1]
        decoder.append(
            nn.ConvTranspose2d(c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        if not last:
            decoder += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(*encoder)
            self.enc_fc = nn.Linear(feat, latent_dim)
            self.dec_fc = nn.Linear(latent_dim, feat)
            self.decoder = nn.Sequential(*decoder)
            self.act = nn.ReLU(inplace=True)
            self.feat_shape = (chans[-1], feat_hw, feat_hw)

        def forward(self, x):
            h = self.encoder(x)
            z = self.enc_fc(h.flatten(1))
            h2 = self.act(self.dec_fc(z)).reshape(-1, *self.feat_shape)
            return self.decoder(h2)

    return Net(), chans, feat_hw


class _ConvAutoencoderBase(NoveltyModel):
    """Shared implementation; the two tiers differ only in size defaults."""

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
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 5,
        score_mode: str = "recon_mse",
        val_split: str = "validation_typical",
        val_fraction: float = DEFAULT_VAL_FRACTION,
        config: dict | None = None,
    ) -> None:
        super().__init__(transform, config)
        if channels is not None:
            self.channels = int(channels)
        if depth is not None:
            self.depth = int(depth)
        if latent_dim is not None:
            self.latent_dim = int(latent_dim)
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}, got {score_mode!r}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        h, w, _ = transform.output_shape
        if h != w:
            raise ValueError(f"conv tiers expect square frames, got {h}x{w}")
        if h % (2**self.depth):
            raise ValueError(f"depth {self.depth} does not divide frame size {h}")

        self.epochs = int(epochs)
        self.batch_size = max(1, int(batch_size))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = max(1, int(early_stopping_patience))
        self.score_mode = score_mode
        self.val_split = val_split
        self.val_fraction = float(val_fraction)

        self.net = None  # torch module once fitted/loaded
        self.fit_info_: dict = {}

    # -- construction from config ------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, transform: FrameTransform) -> _ConvAutoencoderBase:
        m = cfg.get("model", {})
        return cls(
            transform,
            channels=m.get("channels"),
            depth=m.get("depth"),
            latent_dim=m.get("latent_dim"),
            epochs=int(m.get("epochs", 30)),
            batch_size=int(m.get("batch_size", 128)),
            learning_rate=float(m.get("learning_rate", 1e-3)),
            weight_decay=float(m.get("weight_decay", 0.0)),
            early_stopping_patience=int(m.get("early_stopping_patience", 5)),
            score_mode=str(m.get("score", "recon_mse")),
            val_split=str(m.get("val_split", "validation_typical")),
            val_fraction=float(m.get("val_fraction", DEFAULT_VAL_FRACTION)),
            config=cfg,
        )

    # -- tensors ------------------------------------------------------------
    def _to_nchw(self, torch, raw_frames: np.ndarray):
        """Raw (n, 64, 64, 6) frames -> standardized torch (n, c, h, w)."""
        flat = self.transform.apply(raw_frames)
        h, w, c = self.transform.output_shape
        spatial = flat.reshape(len(flat), h, w, c)
        return torch.from_numpy(np.ascontiguousarray(spatial.transpose(0, 3, 1, 2)))

    def _batch_from(self, torch, array: np.ndarray, indices: np.ndarray):
        # Sorted within the batch purely for memmap read locality; membership
        # is unchanged, so the gradient (a sum over the batch) is unchanged.
        return self._to_nchw(torch, np.asarray(array[np.sort(indices)], dtype=np.float32))

    # -- training -----------------------------------------------------------
    def _resolve_validation(self, n_train: int, rng: np.random.Generator):
        """Return (val_array_or_none, holdout_indices_or_none, source label).

        Prefers the real validation_typical split (different sols than train,
        which is what makes its loss an honest generalisation signal). Falls
        back to a deterministic held-out slice of train when that split does
        not exist, e.g. under the synthetic test fixture.
        """
        try:
            from ..dataset import load_array  # noqa: PLC0415 - avoids cycle at import

            return load_array(self.val_split), None, self.val_split
        except (FileNotFoundError, KeyError) as exc:
            n_hold = max(1, int(n_train * self.val_fraction))
            holdout = rng.permutation(n_train)[:n_hold]
            log.warning(
                "no %s split (%s); holding out %d of %d training frames for early stopping",
                self.val_split,
                exc,
                n_hold,
                n_train,
            )
            return None, np.sort(holdout), "train_holdout"

    def _val_loss(self, torch, net, val_array, batch: int = 256) -> float:
        net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(val_array), batch):
                x = self._to_nchw(
                    torch, np.asarray(val_array[start : start + batch], dtype=np.float32)
                )
                recon = net(x)
                total += float(((recon - x) ** 2).sum())
                count += x.numel()
        return total / max(1, count)

    def fit(
        self, chunks: Iterable[np.ndarray], *, n_samples: int, seed: int = 0
    ) -> _ConvAutoencoderBase:
        torch = _torch()

        source = getattr(chunks, "array", None)
        if source is None:
            raise TypeError(
                "conv autoencoder tiers need random access for shuffled minibatches; "
                "pass a core.dataset.ChunkedArray (scripts/train.py does)"
            )
        n = min(int(n_samples), len(source))

        if not self.transform.fitted_:
            log.info("fitting transform statistics (%s)", self.transform.standardize)
            self.transform.fit(chunks)

        # Determinism contract: torch seeded, numpy seeded, and batch order
        # drawn from this Generator -- there is no DataLoader worker pool to
        # introduce nondeterministic interleaving.
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        h, w, c = self.transform.output_shape
        net, chans, feat_hw = _build_net(
            torch,
            in_channels=c,
            base=self.channels,
            depth=self.depth,
            latent_dim=self.latent_dim,
            spatial=h,
        )
        net = net.to(torch.device("cpu"))  # CPU-only by contract
        optimiser = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        val_array, holdout, val_source = self._resolve_validation(n, rng)
        train_indices = np.arange(n)
        if holdout is not None:
            train_indices = np.setdiff1d(train_indices, holdout)

        log.info(
            "training %s: %d params, %d train frames, val=%s, batch %d, "
            "lr %g, <=%d epochs (patience %d), torch %s on %d threads",
            self.type_name,
            sum(p.numel() for p in net.parameters()),
            len(train_indices),
            val_source,
            self.batch_size,
            self.learning_rate,
            self.epochs,
            self.patience,
            torch.__version__,
            torch.get_num_threads(),
        )

        best_val = float("inf")
        best_state = None
        best_epoch = -1
        stale = 0
        final_train = float("nan")
        started = time.perf_counter()

        for epoch in range(self.epochs):
            epoch_started = time.perf_counter()
            net.train()
            perm = rng.permutation(train_indices)
            total = 0.0
            n_elems = 0
            for start in range(0, len(perm), self.batch_size):
                x = self._batch_from(torch, source, perm[start : start + self.batch_size])
                recon = net(x)
                loss = torch.nn.functional.mse_loss(recon, x)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * x.numel()
                n_elems += x.numel()
            final_train = total / max(1, n_elems)

            if val_array is not None:
                val = self._val_loss(torch, net, val_array)
            else:
                val = self._val_loss(torch, net, source[holdout])

            improved = val < best_val - 1e-7
            if improved:
                best_val = val
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                stale = 0
            else:
                stale += 1

            log.info(
                "epoch %2d/%d  train %.6f  val %.6f%s  (%s)",
                epoch + 1,
                self.epochs,
                final_train,
                val,
                "  *" if improved else f"  [stale {stale}/{self.patience}]",
                human_duration(time.perf_counter() - epoch_started),
            )
            if stale >= self.patience:
                log.info("early stopping: no val improvement for %d epochs", self.patience)
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.fit_info_ = {
            "epochs_run": epoch + 1,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val,
            "final_train_loss": final_train,
            "early_stopped": stale >= self.patience,
            "val_source": val_source,
            "n_train_used": int(len(train_indices)),
            "train_seconds": round(time.perf_counter() - started, 2),
        }
        self.fitted_ = True
        return self

    # -- inference ----------------------------------------------------------
    def score(self, frames: np.ndarray) -> np.ndarray:
        self._require_fitted()
        torch = _torch()
        assert self.net is not None
        self.net.eval()

        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 3:
            frames = frames[None, ...]

        out = np.empty(len(frames), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(frames), 256):
                x = self._to_nchw(torch, frames[start : start + 256])
                recon = self.net(x)
                per_frame = ((recon - x) ** 2).mean(dim=(1, 2, 3)).numpy().astype(np.float64)
                if self.score_mode == "recon_l2":
                    per_frame = np.sqrt(per_frame * x[0].numel())
                out[start : start + len(per_frame)] = per_frame
        return out

    # -- cost accounting ----------------------------------------------------
    def _layer_costs(self) -> tuple[int, int]:
        """(MACs, elementwise ops) for one frame through encoder+decoder.

        BatchNorm is counted as zero: at inference its affine transform folds
        into the adjacent convolution, which is how any deployment would ship
        it. ReLU counts one op per activation.
        """
        h, w, c_in = self.transform.output_shape
        chans = [self.channels * (2**i) for i in range(self.depth)]
        macs = 0
        elementwise = 0

        # Encoder convs: output h,w halve each layer.
        prev_c, cur = c_in, h
        for c in chans:
            cur //= 2
            macs += cur * cur * prev_c * c * 9
            elementwise += cur * cur * c  # ReLU
            prev_c = c

        feat_hw = h // (2**self.depth)
        feat = chans[-1] * feat_hw * feat_hw
        macs += feat * self.latent_dim          # enc_fc
        macs += self.latent_dim * feat          # dec_fc
        elementwise += feat                     # ReLU after dec_fc

        # Decoder mirrors the encoder cost layer for layer.
        reversed_chans = list(reversed(chans))
        cur = feat_hw
        for i, c in enumerate(reversed_chans):
            out_c = c_in if i == len(reversed_chans) - 1 else reversed_chans[i + 1]
            macs += cur * cur * c * out_c * 9
            cur *= 2
            if i != len(reversed_chans) - 1:
                elementwise += cur * cur * out_c  # ReLU

        return macs, elementwise

    def param_count(self) -> int:
        self._require_fitted()
        assert self.net is not None
        n = sum(int(p.numel()) for p in self.net.state_dict().values())
        if self.transform.mean_ is not None:
            n += self.transform.mean_.size
        if self.transform.std_ is not None:
            n += self.transform.std_.size
        return int(n)

    def flops_per_inference(self) -> int:
        macs, elementwise = self._layer_costs()
        return int(2 * macs + elementwise + self.transform.flops_per_frame())

    # -- persistence --------------------------------------------------------
    @staticmethod
    def _state_key(name: str) -> str:
        return "w__" + name.replace(".", "__")

    @staticmethod
    def _state_name(key: str) -> str:
        return key[len("w__"):].replace("__", ".")

    def _persistable_arrays(self) -> dict[str, np.ndarray]:
        self._require_fitted()
        assert self.net is not None
        arrays = {
            self._state_key(name): np.ascontiguousarray(tensor.detach().cpu().numpy())
            for name, tensor in self.net.state_dict().items()
        }
        if self.transform.mean_ is not None:
            arrays["transform_mean"] = self.transform.mean_
        if self.transform.std_ is not None:
            arrays["transform_std"] = self.transform.std_
        return arrays

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "type": self.type_name,
            "channels": self.channels,
            "depth": self.depth,
            "latent_dim": self.latent_dim,
            "score_mode": self.score_mode,
            "fit_info": self.fit_info_,
            "transform": self.transform.to_dict(),
        }
        arrays = {"meta": self._pack_meta(meta), **self._persistable_arrays()}

        tmp = path.with_name(f".{path.name}.tmp.npz")
        try:
            np.savez_compressed(tmp, **arrays)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        log.info("wrote %s (%s)", path, human_bytes(path.stat().st_size))
        return path

    @classmethod
    def load(cls, path: str | Path) -> _ConvAutoencoderBase:
        torch = _torch()
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
            weights = {
                cls._state_name(key): np.array(data[key])
                for key in data.files
                if key.startswith("w__")
            }

        model = cls(
            transform,
            channels=int(meta["channels"]),
            depth=int(meta["depth"]),
            latent_dim=int(meta["latent_dim"]),
            score_mode=str(meta.get("score_mode", "recon_mse")),
        )
        h, _, c = transform.output_shape
        net, _, _ = _build_net(
            torch,
            in_channels=c,
            base=model.channels,
            depth=model.depth,
            latent_dim=model.latent_dim,
            spatial=h,
        )
        state = {name: torch.from_numpy(np.ascontiguousarray(arr)) for name, arr in weights.items()}
        net.load_state_dict(state)
        net.eval()
        model.net = net
        model.fit_info_ = dict(meta.get("fit_info", {}))
        model.fitted_ = True
        return model


class MyriadConvAutoencoder(_ConvAutoencoderBase):
    """Small conv autoencoder sized for the Myriad 2 VPU (Phi-Sat-1 class)."""

    type_name = "conv_ae_myriad"
    channels = 16
    depth = 2
    latent_dim = 32


class SnapdragonConvAutoencoder(_ConvAutoencoderBase):
    """Larger conv autoencoder for a Snapdragon-class SoC. Overprovisioned
    on purpose: it exists to measure what extra compute buys."""

    type_name = "conv_ae_snapdragon"
    channels = 32
    depth = 3
    latent_dim = 128
