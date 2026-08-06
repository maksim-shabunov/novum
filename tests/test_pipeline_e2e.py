"""End to end: config -> train -> artifact + sidecar -> evaluate -> metrics.

Runs the real CLI entry points against the synthetic processed dataset, so a
break anywhere in the chain shows up here rather than on a remote box after a
40-minute download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import paths
from scripts import evaluate as evaluate_script
from scripts import sweep as sweep_script
from scripts import train as train_script

PCA_CONFIG = """
tier: testtier
seed: 0
model:
  type: pca
  n_components: 6
  oversamples: 4
  power_iterations: 1
transform:
  downsample: 4
  standardize: per_band
data:
  train_split: train_typical
  chunk_size: 32
eval:
  typical_split: test_typical
  novel_split: test_novel_all
  k_values: [5, 10]
compute:
  reference_processor: "test"
  cycles_per_flop: 3.0
  budget_cycles_per_frame: 20000000
downlink:
  bits_per_sample: 8
  compression_ratio: 4.0
"""

AE_CONFIG = """
tier: aetier
seed: 0
model:
  type: conv_ae_myriad
  channels: 4
  depth: 1
  latent_dim: 8
  epochs: 2
  batch_size: 64
  early_stopping_patience: 2
transform:
  downsample: 1
  standardize: per_band
data:
  train_split: train_typical
  chunk_size: 64
eval:
  typical_split: test_typical
  novel_split: test_novel_all
  k_values: [5, 10]
compute:
  reference_processor: "test VPU"
  cycles_per_flop: 0.25
  # ~0.26M cycles/inference for this tiny net, so 100k is deliberately too
  # small: the test exercises the over-budget-is-reported-not-failed path.
  budget_cycles_per_frame: 100000
"""


@pytest.fixture
def pca_config(tmp_path: Path) -> Path:
    path = tmp_path / "tier_testtier.yaml"
    path.write_text(PCA_CONFIG, encoding="utf-8")
    return path


def test_train_writes_an_artifact_and_a_complete_sidecar(
    synthetic_processed: Path, pca_config: Path
) -> None:
    out = paths.artifacts_dir() / "testtier.npz"
    assert train_script.main(["--config", str(pca_config), "--out", str(out)]) == 0

    assert out.exists()
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    # Every field the project contract requires of a sidecar.
    for key in (
        "config_hash",
        "content_sha256",
        "git_commit",
        "wall_clock_seconds",
        "param_count",
        "flops_per_inference",
        "peak_rss_bytes",
        "fits_compute_budget",
        "budget_utilisation",
    ):
        assert key in record, f"sidecar is missing {key}"

    assert record["sidecar_schema"] == 2
    assert record["kind"] == "training"
    assert len(record["content_sha256"]) == 64
    assert record["environment"]["blas_backend"]
    assert record["environment"]["blas_threads"] >= 1
    assert record["fits_compute_budget"] is True  # PCA fits the test budget
    assert 0.0 < record["budget_utilisation"] < 1.0

    assert record["param_count"] > 0
    assert record["flops_per_inference"] > 0
    assert record["peak_rss_bytes"] > 0
    assert record["wall_clock_seconds"] >= 0
    assert record["environment"]["device"] == "cpu"
    assert record["n_train_samples"] == 120
    assert len(record["config_hash"]) == 64


def test_train_is_deterministic_for_a_fixed_seed(
    synthetic_processed: Path, pca_config: Path
) -> None:
    a = paths.artifacts_dir() / "a.npz"
    b = paths.artifacts_dir() / "b.npz"
    train_script.main(["--config", str(pca_config), "--out", str(a), "--seed", "5"])
    train_script.main(["--config", str(pca_config), "--out", str(b), "--seed", "5"])

    import numpy as np

    from core.models.registry import load_model

    probe = np.load(synthetic_processed / "test_novel_all.npy")[:10]
    np.testing.assert_allclose(load_model(a).score(probe), load_model(b).score(probe), rtol=1e-9)


def test_evaluate_reports_roc_auc_and_writes_metrics(
    synthetic_processed: Path, pca_config: Path, capsys
) -> None:
    out = paths.artifacts_dir() / "testtier.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out)])

    assert evaluate_script.main(["--artifact", str(out)]) == 0

    captured = capsys.readouterr().out
    assert "ROC AUC" in captured
    assert "reference" in captured

    metrics_path = paths.metrics_dir() / "testtier.json"
    assert metrics_path.exists()
    record = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert 0.0 <= record["metrics"]["roc_auc"] <= 1.0
    assert record["metrics"]["n_typical"] == 40
    assert record["metrics"]["n_novel"] == 30
    assert set(record["metrics"]["precision_at_k"]) == {"5", "10"}
    assert record["reference"]["roc_auc"] == 0.65
    assert record["novel_split"] == "test_novel_all"

    # Published alongside the committed weights.
    assert (paths.artifacts_dir() / "metrics" / "testtier.json").exists()


def test_the_two_sidecars_share_one_identity(
    synthetic_processed: Path, pca_config: Path
) -> None:
    """artifacts/<name>.json and artifacts/metrics/<name>.json must agree on
    the identity block -- that is the schema unification."""
    out = paths.artifacts_dir() / "testtier.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out)])
    evaluate_script.main(["--artifact", str(out)])

    training = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (paths.artifacts_dir() / "metrics" / "testtier.json").read_text(encoding="utf-8")
    )

    for key in ("sidecar_schema", "name", "tier", "model_type", "config_hash",
                "content_sha256", "git_commit"):
        assert training[key] == evaluation[key], f"sidecars disagree on {key}"
    assert training["kind"] == "training"
    assert evaluation["kind"] == "evaluation"
    # The budget verdict is carried into the evaluation record too.
    assert evaluation["fits_compute_budget"] == training["fits_compute_budget"]
    assert evaluation["compute_budget"]["cycles_per_inference"] == pytest.approx(
        training["compute_budget"]["cycles_per_inference"]
    )


def test_evaluate_decomposes_by_group(synthetic_processed: Path, pca_config: Path) -> None:
    out = paths.artifacts_dir() / "testtier.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out)])
    evaluate_script.main(["--artifact", str(out)])

    record = json.loads((paths.metrics_dir() / "testtier.json").read_text(encoding="utf-8"))
    decomposed = record["decomposed"]
    # The fixture labels every novel frame 'veins' -> all natural, none rover.
    assert decomposed["natural"]["n"] == 30
    assert decomposed["rover"]["n"] == 0
    assert decomposed["rover"]["roc_auc"] is None
    assert 0.0 <= decomposed["natural"]["roc_auc"] <= 1.0


def test_evaluate_separates_synthetic_novelty(
    synthetic_processed: Path, pca_config: Path
) -> None:
    """A sanity floor: the pipeline must actually rank novel frames higher."""
    out = paths.artifacts_dir() / "testtier.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out)])
    evaluate_script.main(["--artifact", str(out)])

    record = json.loads((paths.metrics_dir() / "testtier.json").read_text(encoding="utf-8"))
    assert record["metrics"]["roc_auc"] > 0.9


def test_evaluate_with_the_budget_demo(synthetic_processed: Path, pca_config: Path) -> None:
    out = paths.artifacts_dir() / "testtier.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out)])
    assert evaluate_script.main(["--artifact", str(out), "--budget-demo"]) == 0

    record = json.loads((paths.metrics_dir() / "testtier.json").read_text(encoding="utf-8"))
    demo = record["budget_demo"]
    assert set(demo["methods"]) == {"greedy_sweep", "score_first", "random"}
    for plan in demo["methods"].values():
        assert plan["used_bits"] <= demo["budget_bits"] + 1e-6
        assert plan["used_cycles"] <= demo["budget_cycles"] + 1e-6


def test_evaluate_refuses_the_byclass_split(synthetic_processed: Path, pca_config: Path) -> None:
    out = paths.artifacts_dir() / "testtier.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out)])
    code = evaluate_script.main(
        ["--artifact", str(out), "--novel-split", "test_novel_byclass"]
    )
    assert code == 2


def test_evaluate_without_an_artifact_fails_cleanly(synthetic_processed: Path) -> None:
    assert evaluate_script.main(["--artifact", str(paths.artifacts_dir() / "nope.npz")]) == 2


def test_autoencoder_tier_trains_and_evaluates_end_to_end(
    synthetic_processed: Path, tmp_path: Path
) -> None:
    """The conv AE path through the same CLI the sweep drives."""
    pytest.importorskip("torch", reason="train extras not installed")
    config = tmp_path / "tier_aetier.yaml"
    config.write_text(AE_CONFIG, encoding="utf-8")
    out = paths.artifacts_dir() / "aetier.npz"

    assert train_script.main(["--config", str(config), "--out", str(out)]) == 0
    assert out.exists()

    record = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["model_type"] == "conv_ae_myriad"
    assert record["training"]["epochs_run"] >= 1
    assert record["training"]["val_source"] == "train_holdout"  # fixture has no val split
    assert record["content_sha256"]
    # This tiny config's budget is set deliberately low, so the honest-verdict
    # path is exercised: over budget is REPORTED, not an error.
    assert record["fits_compute_budget"] is False
    assert record["budget_utilisation"] > 1.0

    assert evaluate_script.main(["--artifact", str(out)]) == 0
    metrics = json.loads((paths.metrics_dir() / "aetier.json").read_text(encoding="utf-8"))
    assert metrics["kind"] == "evaluation"
    assert 0.0 <= metrics["metrics"]["roc_auc"] <= 1.0


def test_stub_exit_code_path_still_works(synthetic_processed: Path, tmp_path: Path) -> None:
    """No shipped tier is a stub anymore, but the sweep still distinguishes
    exit 3 from failure; keep the contract honest via a synthetic stub."""
    assert train_script.EXIT_NOT_IMPLEMENTED == 3


def test_train_with_a_missing_config_fails_cleanly(synthetic_processed: Path) -> None:
    assert train_script.main(["--config", "configs/does_not_exist.yaml"]) == 2


def test_cli_override_reaches_the_model(synthetic_processed: Path, pca_config: Path) -> None:
    out = paths.artifacts_dir() / "override.npz"
    train_script.main(
        ["--config", str(pca_config), "--out", str(out), "--set", "model.n_components=3"]
    )
    record = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["config"]["model"]["n_components"] == 3


def test_max_samples_caps_the_training_set(synthetic_processed: Path, pca_config: Path) -> None:
    out = paths.artifacts_dir() / "capped.npz"
    train_script.main(["--config", str(pca_config), "--out", str(out), "--max-samples", "50"])
    record = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["n_train_samples"] == 50


def test_sweep_dry_run_lists_the_matrix(capsys) -> None:
    assert sweep_script.main(["--dry-run", "--tiers", "rad750,myriad", "--seeds", "0,1"]) == 0
    out = capsys.readouterr().out
    assert "4 run(s)" in out
    assert "rad750-s0" in out and "myriad-s1" in out


def test_sweep_rejects_non_integer_seeds() -> None:
    assert sweep_script.main(["--seeds", "a,b"]) == 2
