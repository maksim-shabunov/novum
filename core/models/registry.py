"""Lazy model registry.

Imports are deferred by design. `core.models.conv_ae` imports torch inside its
methods, but even so, eagerly importing every implementation here would make
the module graph of the API depend on the shape of the training code. Resolving
(module, class) only when a model is actually requested keeps that from
happening, and keeps `import core.models` cheap.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .base import NoveltyModel

_REGISTRY: dict[str, tuple[str, str]] = {
    # RAD750 tier: classical, numpy-only, needs no training dependency at all.
    "pca": ("core.models.pca", "PCANoveltyModel"),
    # Myriad X tier: small conv autoencoder. Stub.
    "conv_ae_myriad": ("core.models.conv_ae", "MyriadConvAutoencoder"),
    # Snapdragon tier: larger conv autoencoder. Stub.
    "conv_ae_snapdragon": ("core.models.conv_ae", "SnapdragonConvAutoencoder"),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model_class(type_name: str) -> type[NoveltyModel]:
    """Resolve a registry key to its class, importing the module on demand."""
    if type_name not in _REGISTRY:
        raise KeyError(f"unknown model type {type_name!r}; available: {available_models()}")
    module_name, class_name = _REGISTRY[type_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_model(cfg: dict) -> NoveltyModel:
    """Instantiate the model a resolved config asks for, with its transform."""
    from ..transforms import build_transform  # local: keeps this module cheap

    model_cls = get_model_class(cfg["model"]["type"])
    transform = build_transform(cfg.get("transform", {}))
    factory = getattr(model_cls, "from_config", None)
    if factory is not None:
        return factory(cfg, transform)
    return model_cls(transform, config=cfg)  # type: ignore[call-arg]


def read_artifact_meta(path: str | Path) -> dict:
    """Read the metadata block from a .npz artifact without building a model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "meta" not in data:
            raise ValueError(f"{path} is not a NOVUM artifact (no 'meta' entry)")
        raw = data["meta"]
    return json.loads(raw.item() if hasattr(raw, "item") else str(raw))


def load_model(path: str | Path) -> NoveltyModel:
    """Load any NOVUM artifact, dispatching on the model type it records."""
    meta = read_artifact_meta(path)
    model_type = meta.get("type")
    if not model_type:
        raise ValueError(f"{path} does not record a model type")
    return get_model_class(model_type).load(path)
