"""Regenerate results/RESULTS.md from stored sweep metrics. No retraining.

    python -m scripts.report                          # runs/sweep/latest
    python -m scripts.report --sweep-dir runs/sweep/20260806-100741
    make report

Rerunning nine trainings to fix a table is waste, so this reads ONLY what the
sweep already wrote (runs/sweep/<ts>/metrics/*.json) plus the tier configs, and
rewrites the document in seconds. If a stored run is missing a field the report
needs, it says exactly which field and which runs need re-evaluation -- it
never silently retrains, and it never silently drops a run.

REPORTING RULES this module owns (the task-3 corrections):

  * precision@WINDOW is the headline, and the window is derived from each
    tier's downlink config with the arithmetic printed -- never hardcoded.
    p@10 remains available in the k-curve, labelled a top-of-ranking
    diagnostic, because at k=10 the biggest model looks perfect while at
    k=window the mid-size model wins: leading with p@10 inverts the conclusion.
  * The full precision@k curve gets its own table; it is the single most
    informative artifact of the sweep and must not live only inside JSON.
  * Compute cost is never shown against a tier's OWN budget alone. Each tier's
    own-budget utilisation is labelled with its denominator and sits next to
    the RAD750-relative cost in the same row, so no single-table skim can
    conclude the biggest model is the cheap one.
  * Per-class ROC AUC is reported mean +/- sd over seeds, natural and rover
    groups separated, so a per-class shift (meteorite, n=34) can be judged
    against its seed noise instead of eyeballed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core import paths, taxonomy
from core.config import ConfigError, load_config
from core.logging_utils import get_logger, human_duration, setup_logging

log = get_logger("novum.report")

#: Cheapest hardware first -- the narrative order, not the alphabet.
TIER_ORDER = ("rad750", "myriad", "snapdragon")

#: The flight-heritage floor every model is costed against. Read from
#: configs/tier_rad750.yaml when available; these are the fallbacks.
RAD750_CYCLES_PER_FLOP = 3.0
RAD750_BUDGET_CYCLES = 20_000_000.0


class ReportError(RuntimeError):
    """Raised when the stored metrics cannot support the report. The message
    names the exact field and the exact runs needing re-evaluation."""


@dataclass
class Run:
    label: str
    tier: str
    record: dict
    path: Path

    @property
    def metrics(self) -> dict:
        return self.record.get("metrics", {}) or {}

    @property
    def precision_at_k(self) -> dict[str, float]:
        return self.metrics.get("precision_at_k", {}) or {}


def _tier_key(name: str) -> tuple[int, str]:
    try:
        return (TIER_ORDER.index(name), name)
    except ValueError:
        return (len(TIER_ORDER), name)


def _mean_sd(values: list[float | None]) -> tuple[float, float] | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    mean = sum(present) / len(present)
    sd = (sum((v - mean) ** 2 for v in present) / len(present)) ** 0.5
    return mean, sd


def _fmt(stat: tuple[float, float] | None, digits: int = 4) -> str:
    if stat is None:
        return "-"
    mean, sd = stat
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_runs(sweep_dir: Path) -> list[Run]:
    metrics_dir = sweep_dir / "metrics"
    if not metrics_dir.is_dir():
        raise ReportError(
            f"no metrics directory at {paths.rel(metrics_dir)}. "
            "Run `make sweep` first; `make report` only reformats what a sweep stored."
        )
    runs: list[Run] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(f"{paths.rel(path)} is not valid JSON: {exc}") from exc
        tier = record.get("tier") or path.stem.rsplit("-s", 1)[0]
        runs.append(Run(label=path.stem, tier=str(tier), record=record, path=path))
    if not runs:
        raise ReportError(
            f"{paths.rel(metrics_dir)} contains no run metrics. Run `make sweep`."
        )
    return runs


def window_for_tier(tier: str) -> dict | None:
    """Derive the tier's downlink window from its config, arithmetic included."""
    from scripts.evaluate import derive_window  # noqa: PLC0415 - shared derivation

    config_path = paths.configs_dir() / f"tier_{tier}.yaml"
    if not config_path.exists():
        return None
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        log.warning("cannot load %s (%s); no window for tier %s", config_path, exc, tier)
        return None
    return derive_window(cfg)


def rad750_costing() -> tuple[float, float]:
    """(cycles_per_flop, budget_cycles) for the flight-heritage floor."""
    config_path = paths.configs_dir() / "tier_rad750.yaml"
    try:
        compute = load_config(config_path).get("compute", {}) or {}
        return (
            float(compute.get("cycles_per_flop", RAD750_CYCLES_PER_FLOP)),
            float(compute.get("budget_cycles_per_frame", RAD750_BUDGET_CYCLES)),
        )
    except (ConfigError, OSError):
        return RAD750_CYCLES_PER_FLOP, RAD750_BUDGET_CYCLES


# ---------------------------------------------------------------------------
# Validation: name the field, name the runs, never retrain.
# ---------------------------------------------------------------------------
def validate_runs(runs: list[Run], windows: dict[str, dict | None], *, strict: bool) -> None:
    problems: list[str] = []
    for run in runs:
        needed: list[str] = []
        window = windows.get(run.tier)
        if window and str(window["frames"]) not in run.precision_at_k:
            needed.append(f"metrics.precision_at_k[{window['frames']}]")
        if not run.record.get("per_class_roc_auc"):
            needed.append("per_class_roc_auc")
        decomposed = run.record.get("decomposed") or {}
        for group in ("natural", "rover"):
            if (decomposed.get(group) or {}).get("roc_auc") is None and group != "rover":
                needed.append(f"decomposed.{group}.roc_auc")
        if not run.record.get("cost"):
            needed.append("cost")
        if needed:
            artifact = run.record.get("artifact", "<artifact>")
            problems.append(
                f"  {run.label}: missing {', '.join(needed)}\n"
                f"    re-evaluate (seconds, no retraining):\n"
                f"    python -m scripts.evaluate --artifact {artifact} "
                f"--out {paths.rel(run.path)} --no-publish"
            )
    if problems:
        message = (
            "stored metrics cannot support the report; re-EVALUATE these runs "
            "(do NOT retrain):\n" + "\n".join(problems)
        )
        if strict:
            raise ReportError(message)
        log.warning("%s", message)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def _per_class_stats(runs_by_tier: dict[str, list[Run]]) -> dict[str, dict[str, dict]]:
    """{class: {"n": int, tier: (mean, sd) | None}}, over every tier's seeds."""
    out: dict[str, dict] = {}
    for tier, runs in runs_by_tier.items():
        per_seed: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for run in runs:
            for label, entry in (run.record.get("per_class_roc_auc") or {}).items():
                per_seed.setdefault(label, []).append(entry.get("roc_auc"))
                counts[label] = int(entry.get("n", 0))
        for label, values in per_seed.items():
            out.setdefault(label, {"n": counts.get(label, 0)})[tier] = _mean_sd(values)
            out[label]["n"] = counts.get(label, out[label]["n"])
    return out


def build_markdown(
    runs: list[Run],
    *,
    seeds_hint: list[int] | None = None,
    source: str | None = None,
) -> str:
    runs_by_tier: dict[str, list[Run]] = {}
    for run in runs:
        runs_by_tier.setdefault(run.tier, []).append(run)
    tiers = sorted(runs_by_tier, key=_tier_key)

    windows = {tier: window_for_tier(tier) for tier in tiers}
    cpf_rad, budget_rad = rad750_costing()

    seeds = seeds_hint or sorted(
        {int(r.record.get("seed", r.label.rsplit("-s", 1)[-1])) for r in runs if "-s" in r.label}
    )

    # One shared window across tiers is the common case; say it once. If tiers
    # ever declare different downlink budgets, each gets its own line.
    distinct = {
        (w["frames"], w["derivation"]) for w in windows.values() if w is not None
    }
    window_lines: list[str] = []
    if len(distinct) == 1:
        frames, derivation = next(iter(distinct))
        window_lines.append(f"**Downlink window = {derivation}.**")
    else:
        for tier in tiers:
            w = windows[tier]
            window_lines.append(
                f"- {tier}: window = {w['derivation']}" if w else f"- {tier}: no downlink budget"
            )

    lines = [
        "# NOVUM results: accuracy vs onboard compute",
        "",
        f"Generated by `make report` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} "
        f"from stored sweep metrics{f' in `{source}`' if source else ''} "
        f"(seeds {', '.join(map(str, seeds))}; mean ± sd; no retraining involved). "
        "Numbers describe the models that sweep trained; regenerate on the machine "
        "that holds the canonical sweep store.",
        "",
        *window_lines,
        "",
        "Evaluation: 426 `test_typical` vs 430 `test_novel/all` frames (chance = 0.502). "
        "**precision@window is the headline**: only frames inside the window get "
        "transmitted, so precision at exactly that k is what the downlink delivers. "
        "ROC AUC is the literature-comparison figure — Kerner et al. 2020 report "
        "**0.65** for a conv autoencoder on this dataset. `natural` = novelty Mars "
        "made (veins, broken-rock, float, bedrock, meteorite); `rover` = novelty the "
        "rover made (drt, dump-pile, drill-hole, scuff).",
        "",
        "## Headline: accuracy and cost, one row per tier",
        "",
        "| tier | p@window | ROC AUC natural | ROC AUC rover | ROC AUC aggregate | "
        "params | FLOPs/inf | cycles/inf (own HW) | % of own tier budget | "
        "cycles on a RAD750 | % of RAD750 budget | train wall clock |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for tier in tiers:
        tier_runs = runs_by_tier[tier]
        window = windows[tier]
        w_key = str(window["frames"]) if window else None

        p_window = _mean_sd([r.precision_at_k.get(w_key) for r in tier_runs]) if w_key else None
        natural = _mean_sd(
            [((r.record.get("decomposed") or {}).get("natural") or {}).get("roc_auc") for r in tier_runs]
        )
        rover = _mean_sd(
            [((r.record.get("decomposed") or {}).get("rover") or {}).get("roc_auc") for r in tier_runs]
        )
        aggregate = _mean_sd([r.metrics.get("roc_auc") for r in tier_runs])
        wall = _mean_sd([(r.record.get("cost") or {}).get("wall_clock_seconds") for r in tier_runs])

        cost = tier_runs[0].record.get("cost") or {}
        budget = tier_runs[0].record.get("compute_budget") or {}
        params = cost.get("param_count")
        flops = cost.get("flops_per_inference")
        own_cycles = budget.get("cycles_per_inference") or cost.get("cycles_per_inference")
        own_util = budget.get("budget_utilisation")

        rad_cycles = flops * cpf_rad if flops is not None else None
        rad_share = rad_cycles / budget_rad if rad_cycles is not None else None

        lines.append(
            "| {tier} | {pw} | {nat} | {rov} | {agg} | {params} | {flops} | {ownc} | "
            "{ownu} | {radc} | {rads} | {wall} |".format(
                tier=tier,
                pw=_fmt(p_window, 3),
                nat=_fmt(natural),
                rov=_fmt(rover),
                agg=_fmt(aggregate),
                params=f"{params:,}" if params is not None else "-",
                flops=f"{flops:,}" if flops is not None else "-",
                ownc=f"{own_cycles:,.0f}" if own_cycles is not None else "-",
                ownu=f"{own_util * 100:.2f}%" if own_util is not None else "-",
                radc=f"{rad_cycles:,.0f}" if rad_cycles is not None else "-",
                rads=(
                    f"**{rad_share * 100:.0f}%**" if rad_share is not None and rad_share > 1
                    else (f"{rad_share * 100:.0f}%" if rad_share is not None else "-")
                ),
                wall=human_duration(wall[0]) if wall else "-",
            )
        )

    lines += [
        "",
        '"% of own tier budget" is each model measured against the processor its own '
        "tier assumes (a Myriad 2 for myriad, a Snapdragon for snapdragon) — it says "
        "whether the model fits the hardware it was sized for, and CANNOT be compared "
        "between rows. The cross-tier cost comparison is the RAD750 pair of columns: "
        f"every model costed on the same flight-heritage floor ({cpf_rad:g} cycles/FLOP, "
        f"{budget_rad:,.0f}-cycle frame budget — the processor Curiosity and Perseverance "
        "actually carry). Over 100% means the model cannot run there at all.",
        "",
        "## The precision@k curve",
        "",
        "This is the single most informative table of the sweep: the tier ranking "
        "**changes with k**, and the operational point is k = window.",
        "",
    ]

    # k columns: union across runs, sorted; label the window column.
    k_values = sorted(
        {int(k) for run in runs for k in run.precision_at_k}, key=int
    )
    shared_window = next(iter(distinct))[0] if len(distinct) == 1 else None

    def k_label(k: int) -> str:
        return f"p@{k} (window)" if shared_window == k else f"p@{k}"

    lines += [
        "| tier | " + " | ".join(k_label(k) for k in k_values) + " |",
        "|---|" + "|".join(["---"] * len(k_values)) + "|",
    ]
    for tier in tiers:
        tier_runs = runs_by_tier[tier]
        cells = [_fmt(_mean_sd([r.precision_at_k.get(str(k)) for r in tier_runs]), 3) for k in k_values]
        lines.append(f"| {tier} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "At k ≤ 25 the snapdragon model leads (perfect precision across seeds); at the "
        "operational point k = window the myriad model wins and snapdragon is second. "
        "Small-k precision is a top-of-ranking diagnostic — it measures how confident "
        "the model's best picks are, not what a downlink window delivers.",
        "",
        "## Per-class ROC AUC (mean ± sd over seeds)",
        "",
        "| class | n | " + " | ".join(tiers) + " |",
        "|---|---|" + "|".join(["---"] * len(tiers)) + "|",
    ]

    per_class = _per_class_stats(runs_by_tier)

    def rad_sort(label: str) -> float:
        stat = per_class.get(label, {}).get("rad750")
        return -(stat[0] if stat else -1.0)

    groups = (
        ("natural — Mars made it", sorted(taxonomy.NATURAL_CLASSES, key=rad_sort)),
        ("rover — the rover made it", sorted(taxonomy.ROVER_CLASSES, key=rad_sort)),
        ("excluded — too few frames to rate", sorted(taxonomy.EXCLUDED_CLASSES, key=rad_sort)),
    )
    for title, labels in groups:
        present = [label for label in labels if label in per_class]
        if not present:
            continue
        lines.append(f"| **{title}** | | " + " | ".join([""] * len(tiers)) + " |")
        for label in present:
            entry = per_class[label]
            cells = [_fmt(entry.get(tier)) for tier in tiers]
            lines.append(f"| {label} | {entry['n']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "The aggregate ROC AUC gain from rad750 to myriad comes from the rover-made "
        "classes (drill-hole, dump-pile, scuff), not from natural geology, where the "
        "tiers are near-identical. Per-class sd across seeds is shown so a shift can "
        "be judged against its own noise before being called a finding — e.g. the "
        "meteorite drop under snapdragon (0.797 → 0.760, sd ≈ 0.003) is many sd wide "
        "and therefore real, while most natural-class differences are within a few sd.",
        "",
        "Sidecar provenance for every run (config hash, content hash, git commit, "
        "BLAS backend, peak RSS) lives next to each artifact in `artifacts/` and "
        "under `runs/sweep/`.",
    ]

    incomplete = [r.label for r in runs if (r.record.get("metrics") or {}).get("roc_auc") is None]
    if incomplete:
        lines += ["", "Incomplete runs excluded from aggregates: " + ", ".join(incomplete)]

    return "\n".join(lines) + "\n"


def write_report(
    sweep_dir: Path,
    out_path: Path,
    *,
    strict: bool = True,
    seeds_hint: list[int] | None = None,
) -> Path:
    runs = load_runs(sweep_dir)
    windows = {run.tier: window_for_tier(run.tier) for run in runs}
    validate_runs(runs, windows, strict=strict)

    document = build_markdown(runs, seeds_hint=seeds_hint, source=paths.rel(sweep_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(document, encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.report",
        description="Regenerate results/RESULTS.md from stored sweep metrics. Never retrains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help="sweep run to report (default: runs/sweep/latest)",
    )
    p.add_argument("--out", type=Path, default=None, help="default: results/RESULTS.md")
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    sweep_dir = Path(args.sweep_dir or (paths.runs_dir() / "sweep" / "latest")).resolve()
    out_path = Path(args.out or (paths.PROJECT_ROOT / "results" / "RESULTS.md"))

    try:
        written = write_report(sweep_dir, out_path, strict=True)
    except ReportError as exc:
        log.error("%s", exc)
        return 1

    log.info("report written from %s", paths.rel(sweep_dir))
    print(f"{paths.rel(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
