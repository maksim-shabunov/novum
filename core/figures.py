"""Every published figure, read from the artifacts that produced it.

THE PROBLEM. The same numbers appear in README.md, results/RESULTS.md,
results/SIMULATION.md and the mission brief. Four documents, one truth, and
nothing stopping them from drifting apart -- a rerun changes SIMULATION.md and
leaves the README quoting last week's yield, with no error anywhere. A reader
who spots the disagreement cannot tell which one is wrong, so both become
worthless.

So no document is allowed to hold a headline figure of its own. This module
reads them out of `runs/sim/<run>/summary.json` and `artifacts/metrics/*.json`,
and `tests/test_published_figures.py` fails when a figure written in prose does
not match what the artifacts say. Regenerate, or the tests go red.

Adding a figure to the prose means adding it here first. That is deliberate
friction: a number nobody can source is a number nobody should print.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core import paths

#: The tiers, in the order they are always presented.
TIERS = ("rad750", "myriad", "snapdragon")

#: The policy whose results are the headline. `score_first` is NOVUM proper.
HEADLINE_POLICY = "score_first"
BASELINE_POLICY = "fifo"
CEILING_POLICY = "oracle"


class FiguresUnavailable(RuntimeError):
    """No simulation run to read. The caller decides whether that is fatal."""


def _latest_sim_run() -> Path:
    """The newest simulation run directory carrying a summary.json.

    Newest rather than pinned: the published documents are regenerated from
    whatever `make simulate-sweep` last produced, and pinning a run id here
    would let the docs and the artifacts drift in the other direction.
    """
    sim = paths.runs_dir() / "sim"
    if not sim.is_dir():
        raise FiguresUnavailable(f"no simulation runs under {sim}; run `make simulate-sweep`")
    candidates = [
        d for d in sim.iterdir() if d.is_dir() and (d / "summary.json").is_file()
    ]
    if not candidates:
        raise FiguresUnavailable(
            f"no summary.json under {sim}; run `make simulate-sweep`"
        )
    # Prefer a full sweep: the single-tier quick runs carry fewer experiments
    # and would silently narrow the published tables.
    def rank(d: Path) -> tuple[int, float]:
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        return (len(summary.get("experiments") or {}), d.stat().st_mtime)

    return max(candidates, key=rank)


@lru_cache(maxsize=4)
def _summary(run_dir: str | None = None) -> dict[str, Any]:
    path = Path(run_dir) if run_dir else _latest_sim_run()
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _run(summary: dict, *, tier: str, method: str, experiment: str = "baseline") -> dict:
    """One replay out of the sweep, by (experiment, tier, policy)."""
    runs = summary.get("experiments", {}).get(experiment) or summary.get("runs", [])
    for entry in runs:
        if entry.get("tier") == tier and entry.get("method") == method:
            return entry
    raise FiguresUnavailable(
        f"no run for tier={tier} method={method} experiment={experiment}"
    )


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Figure:
    """One published quantity: where it came from, and how it must be written.

    `text` is the exact string the documents must contain. The checker compares
    prose against this rather than re-deriving a format, so a rounding change
    here is a documentation failure rather than a silent inconsistency.
    """

    key: str
    value: float
    text: str
    source: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "text": self.text,
            "source": self.source,
            "note": self.note,
        }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _yield_figures(summary: dict, src: str) -> list[Figure]:
    out: list[Figure] = []
    fifo = _run(summary, tier="rad750", method=BASELINE_POLICY)
    novum = _run(summary, tier="rad750", method=HEADLINE_POLICY)
    oracle = _run(summary, tier="rad750", method=CEILING_POLICY)

    out.append(
        Figure(
            "FIFO_YIELD", fifo["science_yield"], _pct(fifo["science_yield"]), src,
            "oldest-first, pays no cycle tax -- the honest baseline",
        )
    )
    out.append(
        Figure(
            "NOVUM_YIELD", novum["science_yield"], _pct(novum["science_yield"]), src,
            "novelty ranking at an identical bit budget",
        )
    )
    out.append(
        Figure(
            "ORACLE_YIELD", oracle["science_yield"], _pct(oracle["science_yield"]), src,
            "reads ground-truth labels; cannot run onboard",
        )
    )
    # Decimal forms as well as percentages: the results tables quote 0.556 and
    # the prose quotes 55.6%, and both need to be checkable against the source.
    for key, entry in (("FIFO", fifo), ("NOVUM", novum), ("ORACLE", oracle)):
        out.append(
            Figure(
                f"{key}_YIELD_DECIMAL", entry["science_yield"],
                f"{entry['science_yield']:.3f}", src,
            )
        )

    ratio = novum["science_yield"] / fifo["science_yield"] if fifo["science_yield"] else 0
    out.append(Figure("NOVUM_VS_FIFO_RATIO", ratio, f"{ratio:.1f}×", src))
    delta = (novum["science_yield"] - fifo["science_yield"]) * 100
    out.append(Figure("NOVUM_VS_FIFO_PP", delta, f"{delta:+.1f} pp", src))
    out.append(
        Figure(
            "NATURAL_TOTAL", novum["n_natural_total"], str(novum["n_natural_total"]), src,
            "natural-science frames in the mission -- the yield denominator",
        )
    )
    out.append(
        Figure("NOVUM_NATURAL_SENT", novum["n_sent_natural"], str(novum["n_sent_natural"]), src)
    )
    out.append(
        Figure("FIFO_NATURAL_SENT", fifo["n_sent_natural"], str(fifo["n_sent_natural"]), src)
    )
    return out


def _compute_figures(summary: dict, src: str) -> list[Figure]:
    """The findings that come from pinning the silicon and lifting the budget."""
    out: list[Figure] = []

    # Compute free: every tier converges, so accuracy is not what separated them.
    unlimited = summary.get("experiments", {}).get("unlimited:baseline")
    if unlimited:
        yields = [
            r["science_yield"] for r in unlimited if r["method"] == HEADLINE_POLICY
        ]
        if yields:
            lo, hi = min(yields), max(yields)
            out.append(Figure("UNLIMITED_YIELD_LOW", lo, f"{lo:.3f}", src))
            out.append(Figure("UNLIMITED_YIELD_HIGH", hi, f"{hi:.3f}", src))
            out.append(
                Figure(
                    "UNLIMITED_YIELD_BAND", hi - lo, f"{lo:.3f}-{hi:.3f}", src,
                    "all tiers, cycle budget lifted: capacity buys nothing here",
                )
            )

    # The same model on flight silicon it does not fit.
    fixed = summary.get("experiments", {}).get("fixed:rad750")
    if fixed:
        for tier in TIERS:
            entry = next(
                (r for r in fixed if r["tier"] == tier and r["method"] == HEADLINE_POLICY),
                None,
            )
            if entry is None:
                continue
            scores = entry.get("scores_affordable_per_window")
            out.append(
                Figure(
                    f"RAD750HW_{tier.upper()}_YIELD",
                    entry["science_yield"], f"{entry['science_yield']:.3f}", src,
                    f"{tier} charged its real cost on a RAD750",
                )
            )
            if scores is not None:
                out.append(
                    Figure(
                        f"RAD750HW_{tier.upper()}_SCORES",
                        scores, f"{scores:.1f}", src,
                        "novelty scores affordable per window on that processor",
                    )
                )
                # The console shows two decimals and the demo script reads that
                # figure aloud; both forms need to be checkable.
                out.append(
                    Figure(
                        f"RAD750HW_{tier.upper()}_SCORES_2DP",
                        scores, f"{scores:.2f}", src,
                    )
                )
            if entry.get("cycles_per_score"):
                out.append(
                    Figure(
                        f"RAD750HW_{tier.upper()}_CYCLES",
                        entry["cycles_per_score"],
                        f"{entry['cycles_per_score']:,.0f}", src,
                        f"cycles one {tier} novelty score costs on a RAD750",
                    )
                )

    # Scoring everything makes the PCA tier worse, which is the uncomfortable one.
    base = _run(summary, tier="rad750", method=HEADLINE_POLICY)
    unl = summary.get("experiments", {}).get("unlimited:baseline") or []
    unl_rad = next(
        (r for r in unl if r["tier"] == "rad750" and r["method"] == HEADLINE_POLICY), None
    )
    if unl_rad:
        out.append(
            Figure(
                "RAD750_ALL_SCORED_YIELD",
                unl_rad["science_yield"], f"{unl_rad['science_yield']:.3f}", src,
                "rad750 with every frame scored -- LOWER than triaged",
            )
        )
        drop = unl_rad["science_yield"] - base["science_yield"]
        out.append(Figure("RAD750_ALL_SCORED_DELTA", drop, f"{drop:+.3f}", src))
        out.append(
            Figure(
                "RAD750_PREFILTER_RECALL",
                base["prefilter_recall_natural"],
                f"{base['prefilter_recall_natural']:.3f}", src,
                "unique natural frames ever scored, of those ever buffered",
            )
        )

    # Online adaptation: no prior knowledge of the terrain.
    online = summary.get("experiments", {}).get("online")
    if online:
        entry = next(
            (r for r in online if r["tier"] == "rad750" and r["method"] == HEADLINE_POLICY),
            None,
        )
        if entry:
            out.append(
                Figure(
                    "ONLINE_YIELD", entry["science_yield"],
                    f"{entry['science_yield']:.3f}", src,
                    "bootstrapped in flight, no ground training on this terrain",
                )
            )
    return out


def _static_figures(src_dir: Path) -> list[Figure]:
    """Published evaluation metrics, straight from artifacts/metrics/."""
    out: list[Figure] = []
    for tier in TIERS:
        path = src_dir / f"{tier}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        src = str(paths.rel(path))
        if "roc_auc" in metrics:
            out.append(
                Figure(
                    f"{tier.upper()}_ROC_AUC", metrics["roc_auc"],
                    f"{metrics['roc_auc']:.3f}", src,
                )
            )
        if "precision_at_window" in metrics:
            out.append(
                Figure(
                    f"{tier.upper()}_PRECISION_AT_WINDOW",
                    metrics["precision_at_window"],
                    f"{metrics['precision_at_window']:.3f}", src,
                )
            )
        if data.get("flops_per_inference"):
            out.append(
                Figure(
                    f"{tier.upper()}_FLOPS", data["flops_per_inference"],
                    f"{data['flops_per_inference']:,}", src,
                )
            )
    return out


def published_figures(run_dir: str | None = None) -> dict[str, Figure]:
    """Every figure the documents are allowed to quote, keyed by name."""
    path = Path(run_dir) if run_dir else _latest_sim_run()
    summary = _summary(str(path))
    src = str(paths.rel(path / "summary.json"))

    figures: list[Figure] = []
    figures += _yield_figures(summary, src)
    figures += _compute_figures(summary, src)
    figures += _static_figures(paths.artifacts_dir() / "metrics")

    mission = summary.get("mission", {})
    if mission:
        figures.append(
            Figure("MISSION_FRAMES", mission.get("n_frames", 0),
                   str(mission.get("n_frames", 0)), src)
        )
        figures.append(
            Figure("SOL_MIN", mission.get("sol_min", 0), str(mission.get("sol_min", 0)), src)
        )
        figures.append(
            Figure("SOL_MAX", mission.get("sol_max", 0), str(mission.get("sol_max", 0)), src)
        )

    figures.append(
        Figure("SIM_RUN_ID", 0, path.name, src, "the run that produced these figures")
    )
    figures.append(
        Figure(
            "SIM_GIT_COMMIT", 0, str(summary.get("git_commit", "unknown"))[:8], src,
            "the commit the simulation was run from",
        )
    )
    return {f.key: f for f in figures}


def provenance(run_dir: str | None = None) -> dict[str, Any]:
    """Enough to tie the published numbers to an exact tree."""
    path = Path(run_dir) if run_dir else _latest_sim_run()
    summary = _summary(str(path))
    sidecars = {}
    for tier in TIERS:
        sidecar = paths.artifacts_dir() / f"{tier}.json"
        if sidecar.is_file():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecars[tier] = {
                "git_commit": data.get("git_commit"),
                "git_dirty": data.get("git_dirty"),
                "content_sha256": (data.get("content_sha256") or "")[:12],
            }
    return {
        "run_id": path.name,
        "run_created_utc": summary.get("created_utc"),
        "run_git_commit": summary.get("git_commit"),
        "artifacts": sidecars,
        "clean": all(not s.get("git_dirty") for s in sidecars.values()) if sidecars else False,
    }
