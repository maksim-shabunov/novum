"""Config loading, defaulting and validation.

Configs are YAML because a human edits them and a sweep reads them. Every key
has a default here, so a tier file only states what makes that tier different.
The *resolved* config -- defaults merged with the file merged with CLI
overrides -- is what gets hashed into the artifact sidecar, so the hash
identifies the run exactly, not just the file that started it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from .provenance import config_hash

DEFAULT_CONFIG: dict[str, Any] = {
    "tier": None,  # required
    "description": "",
    "seed": 0,
    "model": {
        "type": None,  # required; must be a key in core.models.registry
    },
    "transform": {
        "scale": 255.0,          # raw Mastcam DN are integers in 0..255
        "downsample": 1,         # block-mean factor; 2 turns 64x64 into 32x32
        "standardize": "per_band",
        "frame_shape": [64, 64, 6],
    },
    "data": {
        "train_split": "train_typical",
        "max_train_samples": None,  # null = use everything
        "chunk_size": 512,          # frames per streaming batch
    },
    "eval": {
        "typical_split": "test_typical",
        "novel_split": "test_novel_all",   # NEVER test_novel_byclass: see core/dataset.py
        "k_values": [10, 25, 50, 100],
        "per_class": True,
    },
    "compute": {
        "reference_processor": "unspecified",
        "cycles_per_flop": 3.0,
        "budget_cycles_per_frame": None,
    },
    "downlink": {
        "bits_per_sample": 8,
        "compression_ratio": 4.0,
        "budget_bits_per_window": None,
    },
}

REQUIRED_KEYS: tuple[tuple[str, ...], ...] = (("tier",), ("model", "type"))


class ConfigError(ValueError):
    """Raised for a malformed or incomplete config."""


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get_path(cfg: dict, path: Iterable[str]) -> Any:
    node: Any = cfg
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> dict:
    """Set `a.b.c=value` in place. Used by --set on the CLI and by sweep.py."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
    return cfg


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML rules (so 3, true, null work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def parse_overrides(items: Iterable[str] | None) -> dict:
    """Turn ['seed=3', 'model.n_components=32'] into a nested dict."""
    out: dict = {}
    for item in items or ():
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        set_dotted(out, key.strip(), _coerce_scalar(raw.strip()))
    return out


def validate_config(cfg: dict) -> dict:
    """Validate a resolved config. Raises ConfigError with an actionable message."""
    for path in REQUIRED_KEYS:
        if _get_path(cfg, path) in (None, ""):
            raise ConfigError(f"config is missing required key {'.'.join(path)!r}")

    from .models.registry import available_models  # local: avoids an import cycle

    model_type = cfg["model"]["type"]
    if model_type not in available_models():
        raise ConfigError(
            f"unknown model type {model_type!r}; available: {sorted(available_models())}"
        )

    transform = cfg["transform"]
    shape = tuple(transform.get("frame_shape", (64, 64, 6)))
    if len(shape) != 3:
        raise ConfigError(f"transform.frame_shape must have 3 entries, got {shape}")
    factor = int(transform.get("downsample", 1))
    if factor < 1:
        raise ConfigError(f"transform.downsample must be >= 1, got {factor}")
    if shape[0] % factor or shape[1] % factor:
        raise ConfigError(
            f"transform.downsample={factor} does not divide frame size {shape[0]}x{shape[1]}"
        )
    if transform.get("standardize") not in ("none", "per_band", "global"):
        raise ConfigError(
            f"transform.standardize must be none|per_band|global, got {transform.get('standardize')!r}"
        )

    if cfg["eval"]["novel_split"] == "test_novel_byclass":
        raise ConfigError(
            "eval.novel_split must not be 'test_novel_byclass': that split contains one row "
            "per (frame, class) label and double counts multi-label frames. "
            "Use 'test_novel_all', the canonical 430-frame evaluation set."
        )

    if int(cfg.get("seed", 0)) < 0:
        raise ConfigError("seed must be non-negative")

    ks = cfg["eval"].get("k_values") or []
    if not isinstance(ks, list) or not all(isinstance(k, int) and k > 0 for k in ks):
        raise ConfigError(f"eval.k_values must be a list of positive ints, got {ks!r}")

    return cfg


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a tier YAML, merge defaults and overrides, and validate."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    cfg = deep_merge(DEFAULT_CONFIG, raw)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_source"] = str(path)
    return validate_config(cfg)


def hashable_config(cfg: dict) -> dict:
    """Strip non-semantic keys so the hash tracks meaning, not provenance."""
    out = copy.deepcopy(cfg)
    out.pop("_source", None)
    out.pop("description", None)
    return out


def resolved_config_hash(cfg: dict) -> str:
    return config_hash(hashable_config(cfg))
