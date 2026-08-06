"""make report: rebuild RESULTS.md from stored metrics, never retrain.

The fixture fabricates a sweep metrics directory, so these tests run in
milliseconds and prove the generator against known numbers -- including the
task-3 inversion: a model that wins at k=10 and loses at k=window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.budgets import frames_per_window
from scripts.report import ReportError, build_markdown, load_runs, write_report

WINDOW = 162  # what the shipped configs derive: 8 Mbit / 49,152 bits


def _record(tier: str, seed: int, *, p_at: dict[int, float], aggregate: float,
            natural: float, rover: float, per_class: dict[str, tuple[int, float]]) -> dict:
    return {
        "sidecar_schema": 2,
        "kind": "evaluation",
        "name": tier,
        "tier": tier,
        "seed": seed,
        "artifact": f"artifacts/{tier}.npz",
        "metrics": {
            "roc_auc": aggregate,
            "precision_at_k": {str(k): v for k, v in p_at.items()},
            "recall_at_k": {str(k): 0.1 for k in p_at},
        },
        "decomposed": {
            "natural": {"n": 169, "roc_auc": natural},
            "rover": {"n": 261, "roc_auc": rover},
            "excluded": {"n": 0, "roc_auc": None},
        },
        "per_class_roc_auc": {
            label: {"n": n, "roc_auc": auc} for label, (n, auc) in per_class.items()
        },
        "cost": {
            "param_count": 1000 * (1 if tier == "rad750" else 10),
            "flops_per_inference": 866_432 if tier == "rad750" else 49_209_344,
            "cycles_per_inference": 2.6e6,
            "wall_clock_seconds": 15.0,
        },
        "compute_budget": {
            "cycles_per_inference": 2.6e6,
            "budget_cycles_per_frame": 2e7,
            "budget_utilisation": 0.13,
            "fits_compute_budget": True,
        },
    }


@pytest.fixture
def fake_sweep(tmp_path: Path) -> Path:
    """Two tiers, two seeds. 'snapdragon' wins at k=10 and loses at k=window."""
    metrics = tmp_path / "sweep" / "metrics"
    metrics.mkdir(parents=True)
    per_class = {"veins": (30, 0.94), "drt": (111, 0.42), "other": (1, 0.27)}
    for seed in (0, 1):
        for tier, p10, pw in (("rad750", 0.90, 0.75), ("snapdragon", 1.00, 0.70)):
            record = _record(
                tier, seed,
                p_at={10: p10, 25: 0.8, WINDOW: pw + seed * 0.01},
                aggregate=0.65, natural=0.88, rover=0.50,
                per_class=per_class,
            )
            (metrics / f"{tier}-s{seed}.json").write_text(json.dumps(record))
    return tmp_path / "sweep"


def test_frames_per_window_shows_its_arithmetic() -> None:
    frames, bits, derivation = frames_per_window(8_000_000, (64, 64, 6))
    assert frames == 162
    assert bits == pytest.approx(49_152.0)
    assert "8,000,000" in derivation and "49,152" in derivation and "= 162 frames" in derivation


def test_frames_per_window_rejects_a_window_too_small_for_one_frame() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        frames_per_window(1_000, (64, 64, 6))


def test_report_leads_with_precision_at_window(fake_sweep: Path, tmp_path: Path) -> None:
    out = write_report(fake_sweep, tmp_path / "RESULTS.md", strict=True)
    text = out.read_text()

    # The headline table column is p@window; p@10 appears only in the k-curve.
    headline = text.split("## The precision@k curve")[0]
    assert "p@window" in headline
    assert "| p@10" not in headline

    curve = text.split("## The precision@k curve")[1]
    assert f"p@{WINDOW} (window)" in curve
    assert "p@10" in curve
    assert "diagnostic" in curve  # labelled, not just present

    # The inversion is representable: mean p@window rad750 0.755 > snapdragon 0.705.
    assert "0.755" in curve and "0.705" in curve


def test_report_cannot_be_misread_as_snapdragon_cheap(fake_sweep: Path, tmp_path: Path) -> None:
    """Own-tier % must be labelled with its denominator AND sit beside the
    RAD750-relative cost in the same table."""
    text = write_report(fake_sweep, tmp_path / "R.md", strict=True).read_text()
    headline = text.split("## The precision@k curve")[0]
    assert "% of own tier budget" in headline
    assert "% of RAD750 budget" in headline
    assert "CANNOT be compared between rows" in text
    # snapdragon's 49.2 MFLOP at 3 cycles/flop = 147.6M cycles = 738% of 20M.
    assert "**738%**" in headline


def test_report_includes_per_class_with_sd_and_group_separation(
    fake_sweep: Path, tmp_path: Path
) -> None:
    text = write_report(fake_sweep, tmp_path / "R.md", strict=True).read_text()
    per_class = text.split("## Per-class ROC AUC")[1]
    assert "natural — Mars made it" in per_class
    assert "rover — the rover made it" in per_class
    assert "excluded — too few frames to rate" in per_class
    assert "±" in per_class            # sd is reported per class
    assert "| veins | 30 |" in per_class
    assert "| drt | 111 |" in per_class


def test_report_names_missing_fields_and_runs_instead_of_retraining(
    fake_sweep: Path, tmp_path: Path
) -> None:
    # Strip per-class from one run.
    victim = fake_sweep / "metrics" / "snapdragon-s1.json"
    record = json.loads(victim.read_text())
    del record["per_class_roc_auc"]
    victim.write_text(json.dumps(record))

    with pytest.raises(ReportError) as excinfo:
        write_report(fake_sweep, tmp_path / "R.md", strict=True)
    message = str(excinfo.value)
    assert "snapdragon-s1" in message
    assert "per_class_roc_auc" in message
    assert "re-evaluate" in message.lower()
    assert "scripts.evaluate" in message      # the exact command to run
    assert "retrain" in message.lower()       # ...and the instruction not to


def test_report_missing_sweep_dir_says_run_make_sweep(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="make sweep"):
        load_runs(tmp_path / "nowhere")


def test_non_strict_mode_still_writes_what_it_can(fake_sweep: Path, tmp_path: Path) -> None:
    """Mid-sweep the store is incomplete by definition; the sweep must not die."""
    victim = fake_sweep / "metrics" / "snapdragon-s1.json"
    record = json.loads(victim.read_text())
    del record["per_class_roc_auc"]
    victim.write_text(json.dumps(record))

    out = write_report(fake_sweep, tmp_path / "R.md", strict=False)
    assert out.exists()
    assert "p@window" in out.read_text()


def test_build_markdown_orders_tiers_by_hardware_not_alphabet(fake_sweep: Path) -> None:
    runs = load_runs(fake_sweep)
    text = build_markdown(runs)
    headline = text.split("## The precision@k curve")[0]
    assert headline.index("| rad750 |") < headline.index("| snapdragon |")
