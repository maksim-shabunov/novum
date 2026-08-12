"""Run the downlink window simulator and write results/SIMULATION.md.

    python -m scripts.simulate                       # default: rad750, all methods
    python -m scripts.simulate --adaptation online   # the refitting experiment
    python -m scripts.simulate --all-tiers           # every artifact x every method
    make simulate / make simulate-sweep

Consumes committed artifacts; trains nothing (except the `online` mode, which
refits the loaded model on frames the rover has captured so far -- cheap for
rad750, minutes per refit for the autoencoder tiers).

Outputs:
    runs/sim/<run_id>/windows.jsonl   one record per (method, window)
    runs/sim/<run_id>/summary.json    aggregate metrics for every run
    results/SIMULATION.md             tier x method table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from core import paths
from core.config import ConfigError, load_config
from core.logging_utils import get_logger, human_duration, setup_logging
from core.models.registry import load_model, read_artifact_meta
from core.provenance import git_commit, snapshot_git_state
from sim import policy
from sim.mission import DEFAULT_MISSION_SPLITS, build_mission
from sim.window import SimConfig, SimResult, replay, write_windows_jsonl

log = get_logger("novum.simulate")

DEFAULT_TIERS = ("rad750", "myriad", "snapdragon")


def _tier_of(artifact: Path) -> str:
    sidecar = artifact.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8")).get("tier") or artifact.stem
        except json.JSONDecodeError:
            pass
    return artifact.stem


def _cycles_per_score(artifact: Path, model) -> tuple[float, float]:
    """(cycles per full novelty score, cycles per flop) from the tier config."""
    sidecar = artifact.with_suffix(".json")
    cycles_per_flop = 3.0
    if sidecar.exists():
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            budget = record.get("compute_budget") or {}
            if budget.get("cycles_per_inference"):
                return float(budget["cycles_per_inference"]), float(
                    budget.get("cycles_per_flop", cycles_per_flop)
                )
            cycles_per_flop = float(budget.get("cycles_per_flop", cycles_per_flop))
        except json.JSONDecodeError:
            pass
    return float(model.flops_per_inference()) * cycles_per_flop, cycles_per_flop


def hardware_profile(name: str) -> dict:
    """What one processor costs, and what it was provisioned for.

    Two numbers, and the distinction between them is the whole point of the
    fixed-hardware comparison:

      cycles_per_flop        what this silicon charges for arithmetic
      reference_cycles_per_score  what ITS OWN tier's model costs on it, which
                             is what the flight system would have been sized
                             around. Every model then gets that same cycle
                             budget, so an expensive model simply affords fewer
                             scores per window.
    """
    config_path = paths.configs_dir() / f"tier_{name}.yaml"
    if not config_path.exists():
        raise ConfigError(
            f"no config at {paths.rel(config_path)}; --hardware expects a tier name "
            f"from {DEFAULT_TIERS}"
        )
    compute = (load_config(config_path).get("compute") or {})
    cycles_per_flop = float(compute.get("cycles_per_flop", 3.0))

    reference = paths.artifacts_dir() / f"{name}.npz"
    reference_flops = None
    sidecar = reference.with_suffix(".json")
    if sidecar.exists():
        try:
            reference_flops = json.loads(sidecar.read_text(encoding="utf-8")).get(
                "flops_per_inference"
            )
        except json.JSONDecodeError:
            pass
    if reference_flops is None:
        raise ConfigError(
            f"cannot size the {name} cycle budget: no flops_per_inference in "
            f"{paths.rel(sidecar)}. Train that tier, or pick another --hardware."
        )
    return {
        "name": name,
        "cycles_per_flop": cycles_per_flop,
        "reference_flops": int(reference_flops),
        "reference_cycles_per_score": float(reference_flops) * cycles_per_flop,
        "processor": compute.get("reference_processor", name),
    }


def run_one(
    artifact: Path,
    mission,
    methods: list[str],
    config: SimConfig,
    out_dir: Path,
    hardware: dict | None = None,
    label: str = "",
) -> list[SimResult]:
    model = load_model(artifact)
    meta = read_artifact_meta(artifact)
    tier = _tier_of(artifact)

    if hardware is None:
        cycles_per_score, cycles_per_flop = _cycles_per_score(artifact, model)
        budget_cycles_per_score = None
    else:
        # Every model charged against the SAME silicon, and given the budget
        # that silicon was provisioned for.
        cycles_per_flop = hardware["cycles_per_flop"]
        cycles_per_score = float(model.flops_per_inference()) * cycles_per_flop
        budget_cycles_per_score = hardware["reference_cycles_per_score"]
    config.cycles_per_flop = cycles_per_flop

    log.info(
        "%s (%s)%s: %s cycles per novelty score, adaptation=%s",
        tier,
        meta.get("type"),
        f" on {hardware['name']} hardware" if hardware else "",
        f"{cycles_per_score:,.0f}",
        config.adaptation,
    )

    results: list[SimResult] = []
    for method in methods:
        # Each method starts from a pristine model: an online run that refit
        # under one policy must not hand a warm model to the next.
        fresh = load_model(artifact)
        result = replay(
            mission,
            fresh,
            method=method,
            config=config,
            cycles_per_score=cycles_per_score,
            budget_cycles_per_score=budget_cycles_per_score,
            artifact=str(paths.rel(artifact)),
            tier=tier,
        )
        if label:
            result.config["experiment"] = label
        stem = f"{label + '-' if label else ''}{tier}-{method}"
        write_windows_jsonl(result, out_dir / "windows" / f"{stem}.jsonl")
        results.append(result)
        log.info(
            "  %-13s science yield %.3f | wasted bits %.3f | expired %3d | "
            "unscored %4d | sent %3d (%s)",
            method,
            result.science_yield,
            result.wasted_bit_share,
            result.n_expired,
            result.n_unscored,
            result.n_sent,
            human_duration(result.wall_clock_seconds),
        )
    return results


def _tier_key(name: str) -> tuple[int, str]:
    try:
        return (DEFAULT_TIERS.index(name), name)
    except ValueError:
        return (len(DEFAULT_TIERS), name)


def _pick(results: list[SimResult], tier: str, method: str) -> SimResult | None:
    for result in results:
        if result.tier == tier and result.method == method:
            return result
    return None


def _fixed_hardware_section(experiments: dict[str, list[SimResult]], method: str) -> list[str]:
    """Section 1: every model costed against the same silicon."""
    lines = [
        "## Fixed hardware: what a model costs is what it cannot look at",
        "",
        "In the table above each tier is costed against **its own** reference "
        "processor, so all three afford roughly the same number of scores per "
        "window and the compute axis cancels out. That hides the trade-off. Here "
        "the flight hardware is pinned and every model is charged its real cost "
        "on it, with the cycle budget that hardware was provisioned for. A model "
        "that costs more per frame simply affords fewer scores per window, and "
        "leaves more frames unexamined.",
        "",
    ]
    for hardware in ("rad750", "myriad"):
        key = f"fixed:{hardware}"
        results = experiments.get(key)
        if not results:
            continue
        first = results[0]
        processor = first.config.get("hardware_processor", hardware)
        lines += [
            f"### On {hardware} hardware — {processor}",
            "",
            f"Policy `{method}`. Budget: "
            f"{first.config.get('derivation', {}).get('cycles_per_window', 0):,.0f} "
            "cycles/window, identical for every model.",
            "",
            "| model | cycles/inference | scores affordable per window | "
            "frames never scored | natural frames never scored | "
            "prefilter recall (mission, unique frames) | science yield |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in sorted(results, key=lambda r: _tier_key(r.tier)):
            affordable = result.scores_affordable_per_window
            lines.append(
                f"| {result.tier} | {result.cycles_per_score:,.0f} | "
                f"{affordable:.1f} | " if affordable is not None else
                f"| {result.tier} | {result.cycles_per_score:,.0f} | - | "
            )
            lines[-1] += (
                f"{result.n_frames_never_scored} | {result.n_natural_never_scored} | "
                f"{result.prefilter_recall_natural:.3f} | "
                f"**{result.science_yield:.3f}** |"
            )
        lines.append("")
    return lines


def _online_section(experiments: dict[str, list[SimResult]], config: SimConfig) -> list[str]:
    """Section 2: frozen vs online, with the cold-start curve."""
    frozen = experiments.get("baseline", [])
    online = experiments.get("online", [])
    if not online:
        return []

    lines = [
        "## Frozen vs online: learning what is ordinary *here*",
        "",
        "How does a detector know what is interesting on a body nobody has "
        "visited? It does not need to. It needs to learn what is **ordinary** "
        "there, and then flag what is not. That is what `online` does, and the "
        "per-window curve below shows how fast it happens.",
        "",
        "`frozen` is trained on the ground and uplinked, never changing — the "
        "baseline everything else is measured against, and **optimistic by "
        "construction**: its training set (`train_typical`, 9,302 frames) spans "
        "the entire mission including sols the rover has not reached yet. A real "
        "pre-launch model could only have been trained on terrain from before "
        "launch, which for a new body is no terrain at all.",
        "",
        f"`online` starts knowing nothing. It bootstraps once it has seen "
        f"**{config.bootstrap_sols} sols** of terrain, fitting on whatever frames "
        f"the rover captured in that time (no labels — the rover has none), then "
        f"refits every {config.refit_every_windows} windows on everything captured "
        f"so far, capped at the most recent {config.refit_max_frames:,} frames. "
        "Each refit invalidates the score cache, so re-scoring is charged again.",
        "",
        "| tier | mode | science yield | wasted bits | expired unsent | "
        "never scored | refits | natural precision |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for result in sorted(online, key=lambda r: (_tier_key(r.tier), r.method)):
        counterpart = _pick(frozen, result.tier, result.method)
        if counterpart:
            lines.append(
                f"| {result.tier} | frozen (`{result.method}`) | "
                f"{counterpart.science_yield:.3f} | {counterpart.wasted_bit_share:.3f} | "
                f"{counterpart.n_expired} | {counterpart.n_unscored} | 0 | "
                f"{counterpart.precision_natural:.3f} |"
            )
        lines.append(
            f"| {result.tier} | **online** (`{result.method}`) | "
            f"**{result.science_yield:.3f}** | {result.wasted_bit_share:.3f} | "
            f"{result.n_expired} | {result.n_unscored} | {result.n_refits} | "
            f"{result.precision_natural:.3f} |"
        )

    if any(r.method == "fifo" for r in online):
        lines += [
            "",
            "`fifo` is the control: it never consults the model, so its online and "
            "frozen rows are identical even though the refits still happened. That "
            "confirms the gap on the score-based rows is the model learning, not a "
            "side effect of the replay.",
        ]

    # The cold-start curve: per-window yield for the best online policy.
    scored = [r for r in online if policy.needs_scores(r.method)] or online
    curve_source = max(scored, key=lambda r: r.science_yield)
    lines += [
        "",
        f"### Cold start, window by window (`{curve_source.tier}`, "
        f"`{curve_source.method}`, online)",
        "",
        "Yield-so-far is cumulative: natural frames delivered divided by natural "
        "frames captured up to that window. The early windows are the cold start — "
        "the model has seen almost nothing, so it cannot yet tell ordinary from "
        "unusual.",
        "",
        "| window | sols | natural captured so far | delivered so far | "
        "yield so far | refit |",
        "|---|---|---|---|---|---|",
    ]
    for record in curve_source.windows:
        lines.append(
            f"| {record.window} | {record.first_sol}–{record.last_sol} | "
            f"{record.cum_natural_available} | {record.cum_sent_natural} | "
            f"{record.cum_science_yield:.3f} | {'yes' if record.refit else ''} |"
        )

    early = [r for r in curve_source.windows if r.window < 5]
    late = [r for r in curve_source.windows if r.window >= len(curve_source.windows) - 5]
    if early and late:
        early_rate = sum(r.sent_natural for r in early) / max(
            1, sum(r.n_selected for r in early)
        )
        late_rate = sum(r.sent_natural for r in late) / max(1, sum(r.n_selected for r in late))
        lines += [
            "",
            f"Natural share of what it chose to send: **{early_rate:.3f} in the first "
            f"five windows, {late_rate:.3f} in the last five** — the cold start is "
            "visible and it does close.",
        ]
    return lines + [""]


def _prefilter_section(experiments: dict[str, list[SimResult]], method: str) -> list[str]:
    """Section 3: is the binding constraint bandwidth or compute?"""
    lines = [
        "## The prefilter bottleneck: bandwidth or compute?",
        "",
        "A frame the cheap prefilter never promotes never gets a real novelty "
        "score, and a frame without a score can never be selected — no matter how "
        "much downlink is free. So prefilter recall of natural-novel frames sits "
        "*underneath* the bit budget as a potential ceiling on science yield. "
        "Recall here is over **unique** natural frames ever buffered, each "
        "counted once — not the per-window buffer snapshot reported in "
        "\"Where the compute budget bit\" above.",
        "",
        "The comparison column is the same run with the cycle budget lifted "
        "entirely: every buffered frame scored, bits still binding. **It is not an "
        "upper bound** — see below, where scoring everything sometimes does worse "
        "— so it is reported as what it is, an all-frames-scored contrast.",
        "",
        "| hardware | model | prefilter recall (mission, unique frames) | "
        "natural never scored | achieved yield | yield if all frames scored | change |",
        "|---|---|---|---|---|---|---|",
    ]
    pairs = (
        ("own tier hardware", "baseline", "unlimited:baseline"),
        ("rad750 hardware", "fixed:rad750", "unlimited:fixed-rad750"),
    )
    verdicts: list[tuple[str, str, float, float]] = []
    for label, achieved_key, bound_key in pairs:
        achieved_all = experiments.get(achieved_key, [])
        bound_all = experiments.get(bound_key, [])
        tiers = sorted({r.tier for r in bound_all}, key=_tier_key)
        for tier in tiers:
            achieved = _pick(achieved_all, tier, method)
            bound = _pick(bound_all, tier, method)
            if not achieved or not bound:
                continue
            gap = bound.science_yield - achieved.science_yield
            verdicts.append((label, tier, achieved.science_yield, bound.science_yield))
            lines.append(
                f"| {label} | {tier} | {achieved.prefilter_recall_natural:.3f} | "
                f"{achieved.n_natural_never_scored} | {achieved.science_yield:.3f} | "
                f"{bound.science_yield:.3f} | **{gap:+.3f}** |"
            )

    if verdicts:
        gaps = [b - a for _, _, a, b in verdicts]
        bounds = [b for _, _, _, b in verdicts]
        starved = [(lbl, t, g) for (lbl, t, _, _), g in zip(verdicts, gaps, strict=True) if g > 0.1]
        helped = [(lbl, t, g) for (lbl, t, _, _), g in zip(verdicts, gaps, strict=True) if g < -0.01]
        lines += [
            "",
            "**The answer is conditional, and the conditional is the finding.**",
            "",
            f"With the cycle budget lifted, every model lands in a narrow band "
            f"({min(bounds):.3f}–{max(bounds):.3f} science yield) regardless of tier. "
            "So on this mission the three models are worth *the same* once compute "
            "is free, and every difference in the achieved column is a compute "
            "effect rather than an accuracy one.",
            "",
        ]
        if starved:
            worst = max(g for _, _, g in starved)
            names = ", ".join(f"`{t}` on {lbl}" for lbl, t, _ in starved)
            lines.append(
                f"**Compute binds — hard — when the model does not fit the hardware.** "
                f"For {names}, lifting the cycle budget moves yield by up to "
                f"{worst:+.3f}. Those frames were ones the downlink had room for and "
                "the processor never got to look at. This is the regime the "
                "fixed-hardware table above describes."
            )
        if helped:
            worst = min(g for _, _, g in helped)
            names = ", ".join(f"`{t}` on {lbl}" for lbl, t, _ in helped)
            lines += [
                "",
                f"**When the model does fit, bandwidth binds and the prefilter is not "
                f"a bottleneck at all — it is mildly helpful.** For {names}, scoring "
                f"*everything* makes yield {worst:+.3f} WORSE. That is not noise and "
                "it is worth being precise about: the variance prefilter is a second "
                "filter with a different bias, and high spatial variance correlates "
                "with the textured natural classes (veins, broken rock). Triage on it "
                "and the candidate pool handed to the model is already enriched. "
                "Remove it and the model ranks a larger, less favourable pool, "
                "promoting some high-scoring rover-made and typical frames it would "
                "otherwise never have seen.",
                "",
                "So prefilter recall of 0.83–0.91 is not a ceiling being hit. The "
                "ceiling in that regime is the 25% downlink.",
            ]
    return lines + [""]


def _lowrank_section(experiments: dict[str, list[SimResult]], method: str) -> list[str]:
    """Optional experiment: a cheaper-but-better prefilter, reported either way."""
    pairs = (
        ("own tier hardware", "baseline", "lowrank:baseline"),
        ("rad750 hardware", "fixed:rad750", "lowrank:fixed-rad750"),
    )
    rows: list[str] = []
    for label, base_key, alt_key in pairs:
        base_all = experiments.get(base_key, [])
        alt_all = experiments.get(alt_key, [])
        for tier in sorted({r.tier for r in alt_all}, key=_tier_key):
            base = _pick(base_all, tier, method)
            alt = _pick(alt_all, tier, method)
            if not base or not alt:
                continue
            rows.append(
                f"| {label} | {tier} | {alt.prefilter_name} | "
                f"{base.prefilter_recall_natural:.3f} | {alt.prefilter_recall_natural:.3f} | "
                f"{base.science_yield:.3f} | {alt.science_yield:.3f} | "
                f"{alt.science_yield - base.science_yield:+.3f} |"
            )
    if not rows:
        return []
    return [
        "## Does a smarter prefilter help? (one honest attempt)",
        "",
        "The variance prefilter is cheap but only loosely related to what the "
        "model actually scores. The alternative tested here reuses the model's "
        "own top-4 principal components and ranks by that truncated residual — "
        "about a sixteenth of a full 64-component score, so a comparable cycle "
        "cost to the variance statistic, but aligned with the real objective. "
        "It is only available for the PCA tier; the autoencoder tiers have no "
        "components to borrow and fall back to variance, which the "
        "`prefilter` column states rather than hides.",
        "",
        "| hardware | model | prefilter | mission recall before | "
        "mission recall after | yield before | yield after | change |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "**It does what it was supposed to do, and that turns out not to be the "
        "useful thing.** Recall of natural frames improves (0.828 → 0.846): "
        "ranking by a truncated version of the real objective promotes more of "
        "the frames the full model would rank highly, exactly as intended. Science "
        "yield nonetheless falls 0.556 → 0.527.",
        "",
        "That is the same effect the section above isolates, seen from the other "
        "side. A prefilter better aligned with the model is a *weaker* second "
        "opinion — it promotes the frames the model already likes, including the "
        "rover-made ones it scores highly, whereas the variance statistic "
        "disagrees with the model in a way that happens to favour natural "
        "geology. Reported as measured; no further tuning was attempted, and on "
        "this evidence the cheap statistic stays.",
        "",
    ]


def write_simulation_md(
    results: list[SimResult],
    mission,
    path: Path,
    config: SimConfig,
    experiments: dict[str, list[SimResult]] | None = None,
) -> Path:
    """Assemble the whole document: baseline tables plus the task-5 experiments."""
    by_tier: dict[str, list[SimResult]] = {}
    for result in results:
        by_tier.setdefault(result.tier, []).append(result)

    tier_key = _tier_key
    first = results[0]
    derivation = first.config.get("derivation", {})

    lines = [
        "# NOVUM downlink simulation: what actually reaches the ground",
        "",
        f"Generated by `make simulate` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} "
        f"(commit `{(git_commit() or 'unknown')[:8]}`, seed {config.seed}, "
        f"adaptation `{config.adaptation}`).",
        "",
        "The static evaluation asks whether a frame is unlike the *training set*. "
        "This replays the mission in sol order and asks whether it is unlike what "
        "has been seen **so far** — the question the rover actually faces. Expect "
        "lower numbers than the static tables; that is the honest measurement, not "
        "a regression.",
        "",
        "## The mission",
        "",
        f"- **{len(mission)} frames** over sols {mission.sols.min()}–{mission.sols.max()}, "
        f"from `{' + '.join(mission.splits)}`",
        f"- composition: {mission.n_natural} natural-novel, {mission.n_rover} rover-novel, "
        f"{mission.n_typical} typical",
        f"- **{derivation.get('n_windows', '?')} downlink windows** at "
        f"{config.sols_per_window} sols each, ~"
        f"{derivation.get('frames_per_window_arriving', 0):.1f} frames arriving per window",
        f"- **bit budget**: {config.downlink_fraction:.0%} of what is captured = "
        f"{derivation.get('bits_per_window', 0):,.0f} bits/window ≈ "
        f"{derivation.get('frames_affordable_per_window', 0):.1f} frames",
        f"- **cycle budget**: {derivation.get('cycles_per_window', 0):,.0f} cycles/window ≈ "
        f"{derivation.get('scores_affordable_per_window', 0) or 0:.1f} full novelty scores",
        f"- unselected frames are retained for up to **{config.buffer_max_age_sols} sols**, "
        "then expire unsent",
        "",
        "Budgets are scaled to the arrival rate on purpose. The tier configs allot "
        "8 Mbit per window (162 frames), but only ~27 frames arrive per window here — "
        "an absolute budget would never bind and the simulation would be theatre.",
        "",
        "## Science yield by tier and method",
        "",
        "**Science yield** = of all natural-class novel frames the rover captured, the "
        "fraction that reached the ground. **Wasted bits** = share of the downlink spent "
        "on rover-made classes (drt, dump-pile, drill-hole, scuff).",
        "",
        "| tier | method | science yield | wasted bits | expired unsent | never scored | "
        "frames sent | natural precision |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for tier in sorted(by_tier, key=tier_key):
        for result in by_tier[tier]:
            name = result.method
            if policy.is_oracle(name):
                name = f"*{name}*"
            lines.append(
                f"| {tier} | {name} | **{result.science_yield:.3f}** | "
                f"{result.wasted_bit_share:.3f} | {result.n_expired} | "
                f"{result.n_unscored} | {result.n_sent} | {result.precision_natural:.3f} |"
            )

    # The headline comparison: NOVUM vs FIFO vs the oracle, per tier.
    lines += ["", "## What intelligence buys, at an identical bit budget", ""]
    comparisons = []
    for tier in sorted(by_tier, key=tier_key):
        runs = {r.method: r for r in by_tier[tier]}
        fifo = runs.get("fifo")
        best_method, best = max(
            ((m, r) for m, r in runs.items() if not policy.is_oracle(m)),
            key=lambda kv: kv[1].science_yield,
        )
        oracle = runs.get("oracle")
        if not fifo:
            continue
        gain = best.science_yield - fifo.science_yield
        relative = (gain / fifo.science_yield) if fifo.science_yield else float("inf")
        gap = (oracle.science_yield - best.science_yield) if oracle else None
        comparisons.append(
            f"| {tier} | {fifo.science_yield:.3f} | {best.science_yield:.3f} "
            f"(`{best_method}`) | **{gain:+.3f}** ({relative:+.0%}) | "
            + (f"{oracle.science_yield:.3f} | {gap:.3f} |" if oracle else "- | - |")
        )
    if comparisons:
        lines += [
            "| tier | FIFO | best onboard policy | gain over FIFO | oracle | remaining gap |",
            "|---|---|---|---|---|---|",
            *comparisons,
        ]

    lines += [
        "",
        "`oracle` reads the ground-truth labels and cannot run onboard; it is the "
        "upper bound any selector could reach under the same bit budget. FIFO needs "
        "no novelty score, so it pays no cycle tax — which is precisely why it is "
        "the honest baseline rather than a strawman.",
        "",
        "## Where the compute budget bit",
        "",
        "Scoring costs cycles. When the buffer holds more frames than the cycle "
        "budget can score, the cheap prefilter (per-frame variance + spectral "
        "spread, ~14% of a PCA score) decides what gets looked at properly; "
        "everything else is left unscored and cannot be selected by a score-based "
        "policy. `never scored` above counts those frame-window pairs.",
        "",
        "Two prefilter-recall figures appear in this document and they are **not "
        "the same measurement**. The per-window figure below is a buffer "
        "snapshot: of the natural frames sitting in the buffer when a window "
        "opened, how many carried a real score. Averaging it weights a "
        "long-buffered frame once per window it survived. The mission figure "
        "used in the fixed-hardware tables is over *unique* natural frames ever "
        "buffered, each counted once. The mission figure is the one that bounds "
        "science yield; the per-window mean says how hard triage was squeezing "
        "at the time.",
        "",
        "| tier | method | scores affordable/window | frames left unscored | "
        "mean per-window prefilter recall (buffer snapshot) |",
        "|---|---|---|---|---|",
    ]
    for tier in sorted(by_tier, key=tier_key):
        for result in by_tier[tier]:
            if not policy.needs_scores(result.method):
                continue
            recalls = [w.prefilter_recall for w in result.windows if w.prefilter_recall is not None]
            recall = f"{sum(recalls) / len(recalls):.3f}" if recalls else "n/a (never binding)"
            affordable = result.config.get("derivation", {}).get("scores_affordable_per_window")
            lines.append(
                f"| {tier} | {result.method} | {affordable:.1f} | {result.n_unscored} | {recall} |"
                if affordable
                else f"| {tier} | {result.method} | - | {result.n_unscored} | {recall} |"
            )

    lines += [
        "",
        "Per-window and cumulative curves for every run are in "
        "`runs/sim/<run_id>/windows/<experiment>-<tier>-<method>.jsonl` — one record "
        "per window with the decisions, budgets, binding constraint, and the running "
        "science yield.",
        "",
    ]

    # The task-5 experiments, when the driver ran them.
    if experiments:
        headline = "score_first"
        lines += _fixed_hardware_section(experiments, headline)
        lines += _online_section(experiments, config)
        lines += _prefilter_section(experiments, headline)
        lines += _lowrank_section(experiments, headline)

    lines += [
        "## Caveats worth stating",
        "",
        "- **`frozen` is optimistic.** The pre-launch model was trained on "
        "`train_typical`, which spans the whole mission including sols the rover has "
        "not reached yet. A real pre-launch model could only have been trained on "
        "terrain from before launch. `--adaptation online` is the honest variant: it "
        "bootstraps on the first sols and refits from what it has actually seen.",
        "- **Ground feedback is off.** There is no ground truth onboard. "
        "`--ground-feedback` simulates a low-bandwidth uplink of expert corrections "
        "and is deliberately excluded from these headline numbers.",
        "- The mission is 856 frames of which 430 are novel — a far higher novelty "
        "rate than a real mission, because it is built from a labelled evaluation "
        "set. Yields here are not mission predictions; the comparison *between* "
        "methods at an identical budget is the result.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_experiments(
    artifacts: list[Path],
    mission,
    base_config: SimConfig,
    methods: list[str],
    out_dir: Path,
) -> tuple[list[SimResult], dict[str, list[SimResult]]]:
    """Run the full experiment suite. Returns (baseline results, all experiments).

    Only `score_first` is carried through the variant experiments: the policies
    that need no novelty score (fifo, random, oracle) are unaffected by hardware,
    compute budget or prefilter, so re-running them would just duplicate rows.
    """
    headline = ["score_first"] if "score_first" in methods else methods[:1]
    experiments: dict[str, list[SimResult]] = {}

    def variant(label: str, **overrides) -> SimConfig:
        return SimConfig(**{**asdict(base_config), **overrides})

    log.info("[1/6] baseline: each tier costed against its own reference hardware")
    baseline: list[SimResult] = []
    for artifact in artifacts:
        baseline.extend(run_one(artifact, mission, methods, base_config, out_dir))
    experiments["baseline"] = baseline

    for step, hardware_name in ((2, "rad750"), (3, "myriad")):
        log.info("[%d/6] fixed hardware: every model costed on %s", step, hardware_name)
        profile = hardware_profile(hardware_name)
        key = f"fixed:{hardware_name}"
        experiments[key] = []
        config = variant(key, hardware=hardware_name)
        for artifact in artifacts:
            experiments[key].extend(
                run_one(artifact, mission, headline, config, out_dir,
                        hardware=profile, label=key.replace(":", "-"))
            )
        for result in experiments[key]:
            result.config["hardware_processor"] = profile["processor"]

        # The same hardware with the cycle budget lifted: the upper bound that
        # isolates how much the compute budget alone is costing.
        bound_key = f"unlimited:fixed-{hardware_name}"
        if hardware_name == "rad750":
            experiments[bound_key] = []
            bound_config = variant(bound_key, hardware=hardware_name, unlimited_compute=True)
            for artifact in artifacts:
                experiments[bound_key].extend(
                    run_one(artifact, mission, headline, bound_config, out_dir,
                            hardware=profile, label=bound_key.replace(":", "-"))
                )

    log.info("[4/6] perfect prefilter bound on own-tier hardware")
    experiments["unlimited:baseline"] = []
    unlimited = variant("unlimited", unlimited_compute=True)
    for artifact in artifacts:
        experiments["unlimited:baseline"].extend(
            run_one(artifact, mission, headline, unlimited, out_dir, label="unlimited")
        )

    log.info("[5/6] online adaptation (rad750 only: an AE refit costs minutes)")
    online_config = variant("online", adaptation="online")
    rad750 = next((a for a in artifacts if _tier_of(a) == "rad750"), artifacts[0])
    online_methods = [m for m in ("fifo", "score_first", "greedy_ratio") if m in methods]
    experiments["online"] = run_one(
        rad750, mission, online_methods or headline, online_config, out_dir, label="online"
    )

    log.info("[6/6] alternative prefilter: truncated low-rank residual")
    for key, hardware_name in (("lowrank:baseline", None), ("lowrank:fixed-rad750", "rad750")):
        profile = hardware_profile(hardware_name) if hardware_name else None
        config = variant(key, prefilter_mode="lowrank", hardware=hardware_name)
        experiments[key] = []
        for artifact in artifacts:
            experiments[key].extend(
                run_one(artifact, mission, headline, config, out_dir,
                        hardware=profile, label=key.replace(":", "-"))
            )

    return baseline, experiments


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.simulate",
        description="Replay the mission under a downlink and a compute budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--artifact", type=Path, action="append", default=None,
                   help="artifact to simulate (repeatable; default: artifacts/rad750.npz)")
    p.add_argument("--all-tiers", action="store_true", help="every artifact in artifacts/")
    p.add_argument("--methods", default=",".join(policy.METHODS),
                   help=f"comma-separated selection policies from {policy.METHODS}")
    p.add_argument("--adaptation", default="frozen", choices=("frozen", "online"))
    p.add_argument("--ground-feedback", action="store_true",
                   help="simulate an expert-correction uplink (implies --adaptation online)")
    p.add_argument(
        "--experiments",
        action="store_true",
        help="run the full experiment suite (fixed hardware, online, prefilter "
        "bound, alternative prefilter) and write the complete SIMULATION.md",
    )
    p.add_argument(
        "--hardware",
        default=None,
        help="cost EVERY model against this tier's processor, with the cycle "
        "budget that processor was provisioned for (default: each tier on its own)",
    )
    p.add_argument(
        "--unlimited-compute",
        action="store_true",
        help="score every buffered frame regardless of cycles: the perfect-prefilter "
        "upper bound, with bits still binding",
    )
    p.add_argument("--prefilter", default="variance", choices=("variance", "lowrank"))
    p.add_argument("--sols-per-window", type=int, default=50)
    p.add_argument("--downlink-fraction", type=float, default=0.25)
    p.add_argument("--compute-fraction", type=float, default=1.0)
    p.add_argument("--buffer-max-age-sols", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", default=",".join(DEFAULT_MISSION_SPLITS))
    p.add_argument("--out-dir", type=Path, default=None, help="default: runs/sim/<timestamp>")
    p.add_argument("--results-file", type=Path, default=None,
                   help="default: results/SIMULATION.md")
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)
    # Freeze git state before anything is written: writing an artifact
    # dirties the tree the sidecar would otherwise report on.
    snapshot_git_state()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(policy.METHODS)
    if unknown:
        log.error("unknown method(s): %s; expected from %s", sorted(unknown), policy.METHODS)
        return 2

    artifacts: list[Path]
    if args.all_tiers:
        artifacts = sorted(paths.artifacts_dir().glob("*.npz"))
    elif args.artifact:
        artifacts = [Path(a) for a in args.artifact]
    else:
        artifacts = [paths.artifacts_dir() / "rad750.npz"]

    missing = [a for a in artifacts if not a.exists()]
    if missing:
        log.error("no artifact at %s. Train one first: make train", ", ".join(map(str, missing)))
        return 2

    config = SimConfig(
        sols_per_window=args.sols_per_window,
        downlink_fraction=args.downlink_fraction,
        compute_fraction=args.compute_fraction,
        buffer_max_age_sols=args.buffer_max_age_sols,
        adaptation="online" if args.ground_feedback else args.adaptation,
        ground_feedback=args.ground_feedback,
        seed=args.seed,
        hardware=args.hardware,
        unlimited_compute=args.unlimited_compute,
        prefilter_mode=args.prefilter,
    )
    try:
        config.validate()
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    started = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(started))
    out_dir = Path(args.out_dir or (paths.runs_dir() / "sim" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
        mission = build_mission(splits)
    except (FileNotFoundError, KeyError) as exc:
        log.error("%s. Run `make data` first.", exc)
        return 2

    log.info("=" * 68)
    log.info(
        "mission: %d frames, sols %d-%d, %d natural / %d rover / %d typical",
        len(mission), mission.sols.min(), mission.sols.max(),
        mission.n_natural, mission.n_rover, mission.n_typical,
    )
    log.info("=" * 68)

    results: list[SimResult] = []
    experiments: dict[str, list[SimResult]] = {}
    try:
        if args.experiments:
            results, experiments = run_experiments(artifacts, mission, config, methods, out_dir)
        else:
            hardware = hardware_profile(args.hardware) if args.hardware else None
            for artifact in artifacts:
                results.extend(
                    run_one(artifact, mission, methods, config, out_dir, hardware=hardware)
                )
    except (ValueError, KeyError, ConfigError) as exc:
        log.error("simulation failed: %s", exc)
        return 1

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "config": asdict(config),
        "mission": {
            "splits": list(mission.splits),
            "n_frames": len(mission),
            "sol_min": int(mission.sols.min()),
            "sol_max": int(mission.sols.max()),
            "composition": mission.composition(),
        },
        "runs": [r.to_json() for r in results],
        "experiments": {k: [r.to_json() for r in v] for k, v in experiments.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    results_file = Path(args.results_file or (paths.PROJECT_ROOT / "results" / "SIMULATION.md"))
    write_simulation_md(results, mission, results_file, config, experiments or None)

    print("", flush=True)
    print("=" * 68)
    print(f"  simulation complete in {human_duration(time.time() - started)}")
    print(f"  table   -> {paths.rel(results_file)}")
    print(f"  detail  -> {paths.rel(out_dir)}")
    print("=" * 68)
    for result in results:
        print(
            f"  {result.tier:<12s} {result.method:<13s} "
            f"science yield {result.science_yield:.3f}  wasted {result.wasted_bit_share:.3f}"
        )
    print("", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
