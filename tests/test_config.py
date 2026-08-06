"""Config loading, validation, and the hash that identifies a run."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core import paths
from core.config import (
    ConfigError,
    deep_merge,
    load_config,
    parse_overrides,
    resolved_config_hash,
    set_dotted,
)

SHIPPED_CONFIGS = ["tier_rad750.yaml", "tier_myriad.yaml", "tier_snapdragon.yaml"]


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_every_shipped_config_is_valid(name: str) -> None:
    """The stub tiers must parse cleanly, or sweep.py cannot walk them."""
    cfg = load_config(paths.configs_dir() / name)
    assert cfg["tier"]
    assert cfg["model"]["type"]
    assert cfg["eval"]["novel_split"] == "test_novel_all"


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_no_shipped_config_asks_for_cuda(name: str) -> None:
    """NOVUM is CPU-only by contract."""
    text = (paths.configs_dir() / name).read_text(encoding="utf-8")
    assert "cuda" not in text.lower().replace("do not set to cuda", "")


def test_rad750_is_the_pca_tier() -> None:
    cfg = load_config(paths.configs_dir() / "tier_rad750.yaml")
    assert cfg["model"]["type"] == "pca"
    assert cfg["transform"]["downsample"] == 2


def test_defaults_fill_in_unspecified_keys(tmp_path: Path) -> None:
    path = tmp_path / "minimal.yaml"
    path.write_text("tier: tiny\nmodel:\n  type: pca\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg["seed"] == 0
    assert cfg["data"]["train_split"] == "train_typical"
    assert cfg["eval"]["k_values"] == [10, 25, 50, 100, 162]


def test_overrides_win_over_the_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("tier: t\nseed: 1\nmodel:\n  type: pca\n  n_components: 8\n", encoding="utf-8")
    cfg = load_config(path, parse_overrides(["seed=9", "model.n_components=3"]))
    assert cfg["seed"] == 9
    assert cfg["model"]["n_components"] == 3


def test_override_values_are_parsed_as_yaml_scalars() -> None:
    out = parse_overrides(["a.b=3", "a.c=true", "a.d=null", "a.e=text"])
    assert out == {"a": {"b": 3, "c": True, "d": None, "e": "text"}}


def test_override_without_equals_is_rejected() -> None:
    with pytest.raises(ConfigError, match="key=value"):
        parse_overrides(["justakey"])


def test_missing_required_keys_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  type: pca\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="tier"):
        load_config(path)


def test_unknown_model_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("tier: t\nmodel:\n  type: quantum_vibes\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown model type"):
        load_config(path)


def test_downsample_that_does_not_divide_the_frame_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "tier: t\nmodel:\n  type: pca\ntransform:\n  downsample: 7\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="does not divide"):
        load_config(path)


def test_evaluating_against_the_byclass_split_is_rejected(tmp_path: Path) -> None:
    """The double-count guard reaches into config validation too."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "tier: t\nmodel:\n  type: pca\neval:\n  novel_split: test_novel_byclass\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="double counts"):
        load_config(path)


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_is_reported_clearly(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("tier: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_set_dotted_creates_intermediate_levels() -> None:
    cfg: dict = {}
    set_dotted(cfg, "x.y.z", 5)
    assert cfg == {"x": {"y": {"z": 5}}}


def test_config_hash_ignores_provenance_and_prose(tmp_path: Path) -> None:
    """Same meaning, different file path or description -> same hash."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    body = {"tier": "t", "model": {"type": "pca", "n_components": 4}}
    a.write_text(yaml.safe_dump(body), encoding="utf-8")
    b.write_text(yaml.safe_dump({**body, "description": "a different note"}), encoding="utf-8")
    assert resolved_config_hash(load_config(a)) == resolved_config_hash(load_config(b))


def test_config_hash_changes_with_meaning(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("tier: t\nmodel:\n  type: pca\n  n_components: 4\n", encoding="utf-8")
    base = resolved_config_hash(load_config(path))
    changed = resolved_config_hash(load_config(path, {"model": {"n_components": 8}}))
    assert base != changed
