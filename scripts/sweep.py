"""Run a (tier x seed) matrix sequentially and write a combined results table.

    python -m scripts.sweep                              # every tier, seeds 0,1,2
    python -m scripts.sweep --tiers rad750 --seeds 0,1,2,3,4
    python -m scripts.sweep --dry-run

Built to be started under tmux on a remote box and left alone:

  * **Sequential** -- one run at a time, so peak RSS in the sidecar means
    something and a 4-core VPS is not thrashed.
  * **Fault tolerant** -- a failing or unimplemented tier is recorded and the
    sweep continues. Stub tiers land as `not_implemented`, not `failed`.
  * **Crash safe** -- results.csv and results.md are rewritten after every run,
    so a sweep killed at hour six still has everything up to hour six.
  * **Interruptible** -- Ctrl-C (or SIGTERM) finishes the current write and
    exits cleanly rather than leaving a half-written table.

Output lands in runs/sweep/<timestamp>/ with `runs/sweep/latest` pointing at it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core import paths
from core.logging_utils import get_logger, human_duration, setup_logging

log = get_logger("novum.sweep")

DEFAULT_TIERS = ("rad750", "myriad", "snapdragon")
DEFAULT_SEEDS = (0, 1, 2)

STATUS_OK = "ok"
STATUS_NOT_IMPLEMENTED = "not_implemented"
STATUS_TRAIN_FAILED = "train_failed"
STATUS_EVAL_FAILED = "eval_failed"

EXIT_NOT_IMPLEMENTED = 3

_INTERRUPTED = False


def _handle_signal(signum, _frame) -> None:  # pragma: no cover - signal path
    global _INTERRUPTED
    _INTERRUPTED = True
    log.warning("received signal %s; finishing the current run then stopping", signum)


@dataclass
class RunRecord:
    tier: str
    seed: int
    status: str
    config: str = ""
    artifact: str = ""
    log_file: str = ""
    roc_auc: float | None = None
    roc_auc_natural: float | None = None
    roc_auc_rover: float | None = None
    average_precision: float | None = None
    precision_at_k: dict[str, float] = field(default_factory=dict)
    param_count: int | None = None
    flops_per_inference: int | None = None
    cycles_per_inference: float | None = None
    budget_utilisation: float | None = None
    fits_compute_budget: bool | None = None
    wall_clock_seconds: float | None = None
    peak_rss_mib: float | None = None
    config_hash: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.tier}-s{self.seed}"


def _k_columns(records: list[RunRecord]) -> list[str]:
    keys: set[str] = set()
    for record in records:
        keys.update(record.precision_at_k)
    return sorted(keys, key=lambda k: int(k))


def _flat_row(record: RunRecord, k_cols: list[str]) -> dict:
    row = {
        "tier": record.tier,
        "seed": record.seed,
        "status": record.status,
        "roc_auc": "" if record.roc_auc is None else f"{record.roc_auc:.4f}",
        "average_precision": (
            "" if record.average_precision is None else f"{record.average_precision:.4f}"
        ),
    }
    for k in k_cols:
        value = record.precision_at_k.get(k)
        row[f"p@{k}"] = "" if value is None else f"{value:.4f}"
    row.update(
        {
            "param_count": record.param_count if record.param_count is not None else "",
            "flops_per_inference": (
                record.flops_per_inference if record.flops_per_inference is not None else ""
            ),
            "wall_clock_s": (
                "" if record.wall_clock_seconds is None else f"{record.wall_clock_seconds:.1f}"
            ),
            "peak_rss_mib": "" if record.peak_rss_mib is None else f"{record.peak_rss_mib:.1f}",
            "config_hash": record.config_hash[:12],
            "artifact": record.artifact,
            "log": record.log_file,
            "note": record.note,
        }
    )
    return row


def write_results(records: list[RunRecord], out_dir: Path, *, started: float) -> tuple[Path, Path]:
    """Rewrite results.csv, results.md and results.json. Called after every run."""
    k_cols = _k_columns(records)
    rows = [_flat_row(r, k_cols) for r in records]
    columns = list(rows[0]) if rows else ["tier", "seed", "status"]

    csv_path = out_dir / "results.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)

    # -- markdown -----------------------------------------------------------
    md_columns = ["tier", "seed", "status", "roc_auc", *[f"p@{k}" for k in k_cols], "wall_clock_s", "peak_rss_mib"]
    ok = [r for r in records if r.status == STATUS_OK]
    lines = [
        "# NOVUM sweep results",
        "",
        f"- started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(started))}",
        f"- elapsed: {human_duration(time.time() - started)}",
        f"- runs: {len(records)} ({len(ok)} ok, "
        f"{sum(1 for r in records if r.status == STATUS_NOT_IMPLEMENTED)} not implemented, "
        f"{sum(1 for r in records if r.status.endswith('failed'))} failed)",
        "- reference: ROC AUC 0.65 (Kerner et al. 2020, conv autoencoder)",
        "",
        "| " + " | ".join(md_columns) + " |",
        "|" + "|".join(["---"] * len(md_columns)) + "|",
    ]
    def _cell(value) -> str:
        # Not `value or "-"`: seed 0 is falsy and would render as a missing cell.
        return "-" if value is None or value == "" else str(value)

    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(c, "")) for c in md_columns) + " |")

    if ok:
        lines += ["", "## Per-tier summary", "", "| tier | runs | mean ROC AUC | sd | best |", "|---|---|---|---|---|"]
        by_tier: dict[str, list[float]] = {}
        for record in ok:
            if record.roc_auc is not None:
                by_tier.setdefault(record.tier, []).append(record.roc_auc)
        for tier, values in sorted(by_tier.items()):
            mean = sum(values) / len(values)
            sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            lines.append(
                f"| {tier} | {len(values)} | {mean:.4f} | {sd:.4f} | {max(values):.4f} |"
            )

    failures = [r for r in records if r.status not in (STATUS_OK, STATUS_NOT_IMPLEMENTED)]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r.label}` ({r.status}) -- see `{r.log_file}`" for r in failures]

    md_path = out_dir / "results.md"
    md_tmp = md_path.with_suffix(".md.tmp")
    md_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(md_tmp, md_path)

    json_path = out_dir / "results.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps([asdict(r) for r in records], indent=2) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)

    return csv_path, md_path


def _mean_sd(values: list[float]) -> tuple[float, float] | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return mean, sd


def _fmt_mean_sd(stat: tuple[float, float] | None, digits: int = 4) -> str:
    if stat is None:
        return "-"
    mean, sd = stat
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def write_results_summary(records: list[RunRecord], path: Path, *, seeds: list[int]) -> Path:
    """Write results/RESULTS.md: one row per tier, aggregated over seeds.

    This table is the central evidence of the project -- each tier directly
    comparable on accuracy AND compute, so the reader can see what extra
    compute did or did not buy. Rewritten after every run, so an interrupted
    sweep still leaves a truthful partial table.
    """
    ok = [r for r in records if r.status == STATUS_OK]
    by_tier: dict[str, list[RunRecord]] = {}
    for record in ok:
        by_tier.setdefault(record.tier, []).append(record)

    def tier_order(name: str) -> tuple[int, str]:
        """Cheapest hardware first -- the narrative order, not the alphabet."""
        try:
            return (DEFAULT_TIERS.index(name), name)
        except ValueError:
            return (len(DEFAULT_TIERS), name)

    lines = [
        "# NOVUM results: accuracy vs onboard compute",
        "",
        f"Generated by `make sweep` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
        f"Seeds {', '.join(map(str, seeds))}; metrics are mean ± sd over seeds.",
        "",
        "Evaluation: 426 `test_typical` vs 430 `test_novel/all` frames (chance = 0.502). "
        "**precision@10 is the headline**: a downlink window carries ~162 frames, so only "
        "the top of the ranking is operational. ROC AUC is the literature-comparison "
        "figure -- Kerner et al. 2020 report **0.65** for a conv autoencoder on this dataset. "
        "`natural` = novelty Mars made (veins, broken-rock, float, bedrock, meteorite); "
        "`rover` = novelty the rover made (drt, dump-pile, drill-hole, scuff).",
        "",
        "| tier | p@10 | ROC AUC natural | ROC AUC rover | ROC AUC aggregate | params | "
        "FLOPs/inf | cycles/inf | budget used | fits | train wall clock |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for tier in sorted(by_tier, key=tier_order):
        runs = by_tier[tier]
        p10 = _mean_sd([float(r.precision_at_k.get("10")) for r in runs if r.precision_at_k.get("10") is not None])
        natural = _mean_sd([r.roc_auc_natural for r in runs])
        rover = _mean_sd([r.roc_auc_rover for r in runs])
        aggregate = _mean_sd([r.roc_auc for r in runs])
        wall = _mean_sd([r.wall_clock_seconds for r in runs])

        # Params/FLOPs/cycles are architecture properties: identical across
        # seeds, so a single value, not a mean.
        params = runs[0].param_count
        flops = runs[0].flops_per_inference
        cycles = runs[0].cycles_per_inference
        utilisation = runs[0].budget_utilisation
        fits = runs[0].fits_compute_budget

        lines.append(
            "| {tier} | {p10} | {nat} | {rov} | {agg} | {params} | {flops} | {cycles} | "
            "{util} | {fits} | {wall} |".format(
                tier=tier,
                p10=_fmt_mean_sd(p10, 3),
                nat=_fmt_mean_sd(natural),
                rov=_fmt_mean_sd(rover),
                agg=_fmt_mean_sd(aggregate),
                params=f"{params:,}" if params is not None else "-",
                flops=f"{flops:,}" if flops is not None else "-",
                cycles=f"{cycles:,.0f}" if cycles is not None else "-",
                util=f"{utilisation * 100:.2f}%" if utilisation is not None else "-",
                fits="yes" if fits else ("no" if fits is not None else "-"),
                wall=f"{human_duration(wall[0])}" if wall else "-",
            )
        )

    incomplete = [r for r in records if r.status != STATUS_OK]
    if incomplete:
        lines += ["", "Incomplete runs: " + ", ".join(f"`{r.label}` ({r.status})" for r in incomplete)]

    # The cross-tier punchline: which of these models could run on the
    # processor the rover actually flies? Same FLOPs, RAD750 cost model.
    rad750_rows = []
    for tier in sorted(by_tier, key=tier_order):
        flops = by_tier[tier][0].flops_per_inference
        if flops is None:
            continue
        rad750_cycles = flops * 3.0            # RAD750 cycles/flop
        ratio = rad750_cycles / 20_000_000.0   # RAD750 budget_cycles_per_frame
        rad750_rows.append(
            f"| {tier} | {rad750_cycles:,.0f} | {ratio * 100:.0f}% | "
            f"{'yes' if ratio <= 1.0 else f'no - needs {ratio:.1f}x the budget'} |"
        )
    if rad750_rows:
        lines += [
            "",
            "## Could it fly on the processor the rover actually has?",
            "",
            "Same models costed on the RAD750 (3 cycles/FLOP, 20M-cycle frame budget) -- "
            "the flight-heritage floor both Curiosity and Perseverance carry:",
            "",
            "| model | cycles on a RAD750 | share of RAD750 budget | fits |",
            "|---|---|---|---|",
            *rad750_rows,
        ]

    lines += [
        "",
        "Sidecar provenance for every run (config hash, content hash, git commit, "
        "BLAS backend, peak RSS) lives next to each artifact in `artifacts/` and "
        "under `runs/sweep/`.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _run(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess, appending its output to log_path. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            cwd=paths.PROJECT_ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def _tail(path: Path, n: int = 12) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(f"    | {line}" for line in lines[-n:])


def run_one(
    tier: str,
    seed: int,
    out_dir: Path,
    extra_train_args: list[str],
    *,
    publish: bool = False,
) -> RunRecord:
    """Train and evaluate one (tier, seed) cell.

    With publish=True the run trains DIRECTLY into artifacts/<tier>.npz and its
    evaluation is published to artifacts/metrics/<tier>.json -- so the sweep's
    designated seed doubles as the committed canonical artifact instead of
    being trained twice. The sweep's own metrics copy still lands in the sweep
    directory either way, which is what the results table reads.
    """
    config = paths.configs_dir() / f"tier_{tier}.yaml"
    label = f"{tier}-s{seed}"
    artifact = (
        paths.artifacts_dir() / f"{tier}.npz" if publish else out_dir / "artifacts" / f"{label}.npz"
    )
    metrics = out_dir / "metrics" / f"{label}.json"
    log_path = out_dir / "logs" / f"{label}.log"

    record = RunRecord(
        tier=tier,
        seed=seed,
        status=STATUS_TRAIN_FAILED,
        config=str(paths.rel(config)),
        artifact=str(paths.rel(artifact) if publish else artifact.relative_to(out_dir)),
        log_file=str(log_path.relative_to(out_dir)),
    )

    if not config.exists():
        record.status = STATUS_TRAIN_FAILED
        record.note = f"no config at {paths.rel(config)}"
        log.error("%s: %s", label, record.note)
        return record

    # -- train --------------------------------------------------------------
    code = _run(
        [
            sys.executable, "-m", "scripts.train",
            "--config", str(config),
            "--out", str(artifact),
            "--seed", str(seed),
            *extra_train_args,
        ],
        log_path,
    )
    if code == EXIT_NOT_IMPLEMENTED:
        record.status = STATUS_NOT_IMPLEMENTED
        record.note = "tier is a stub (NotImplementedError)"
        log.info("%s: not implemented, skipping", label)
        return record
    if code != 0:
        record.note = f"train exited {code}"
        log.error("%s: training failed (exit %d)\n%s", label, code, _tail(log_path))
        return record

    sidecar = artifact.with_suffix(".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            record.param_count = data.get("param_count")
            record.flops_per_inference = data.get("flops_per_inference")
            record.wall_clock_seconds = data.get("wall_clock_seconds")
            record.peak_rss_mib = data.get("peak_rss_mib")
            record.config_hash = data.get("config_hash", "")
            budget = data.get("compute_budget", {}) or {}
            record.cycles_per_inference = budget.get("cycles_per_inference")
            record.budget_utilisation = budget.get("budget_utilisation")
            record.fits_compute_budget = budget.get("fits_compute_budget")
        except json.JSONDecodeError:
            log.warning("%s: sidecar is not valid JSON", label)

    # -- evaluate -----------------------------------------------------------
    record.status = STATUS_EVAL_FAILED
    code = _run(
        [
            sys.executable, "-m", "scripts.evaluate",
            "--artifact", str(artifact),
            "--out", str(metrics),
            *([] if publish else ["--no-publish"]),
        ],
        log_path,
    )
    if code != 0:
        record.note = f"evaluate exited {code}"
        log.error("%s: evaluation failed (exit %d)\n%s", label, code, _tail(log_path))
        return record

    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
        metric_block = data.get("metrics", {})
        record.roc_auc = metric_block.get("roc_auc")
        record.average_precision = metric_block.get("average_precision")
        record.precision_at_k = dict(metric_block.get("precision_at_k", {}))
        decomposed = data.get("decomposed", {}) or {}
        record.roc_auc_natural = (decomposed.get("natural") or {}).get("roc_auc")
        record.roc_auc_rover = (decomposed.get("rover") or {}).get("roc_auc")
        record.status = STATUS_OK
    except (json.JSONDecodeError, OSError) as exc:
        record.note = f"could not read metrics: {exc}"
        log.error("%s: %s", label, record.note)
        return record

    log.info(
        "%s: ROC AUC %.4f  (%s, peak RSS %.0f MiB)",
        label,
        record.roc_auc or 0.0,
        human_duration(record.wall_clock_seconds or 0.0),
        record.peak_rss_mib or 0.0,
    )
    return record


def _link_latest(out_dir: Path) -> None:
    latest = out_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.name)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        log.debug("could not update the 'latest' symlink: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.sweep",
        description="Run a (tier x seed) matrix and write a combined results table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tiers", default=",".join(DEFAULT_TIERS), help="comma-separated tier names")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)), help="comma-separated seeds")
    p.add_argument("--out-dir", type=Path, default=None, help="default: runs/sweep/<timestamp>")
    p.add_argument("--max-samples", type=int, default=None, help="cap training frames per run")
    p.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="aggregated per-tier table (default: results/RESULTS.md)",
    )
    p.add_argument(
        "--publish-seed",
        type=int,
        default=0,
        help="the seed whose artifact becomes the committed artifacts/<tier>.npz "
        "(-1 to disable publishing)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        log.error("--seeds must be comma-separated integers, got %r", args.seeds)
        return 2
    if not tiers or not seeds:
        log.error("empty matrix: tiers=%s seeds=%s", tiers, seeds)
        return 2

    matrix = [(t, s) for t in tiers for s in seeds]
    if args.dry_run:
        print(f"{len(matrix)} run(s):")
        for tier, seed in matrix:
            print(f"  {tier}-s{seed}  ({paths.rel(paths.configs_dir() / f'tier_{tier}.yaml')})")
        return 0

    started = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(started))
    out_dir = Path(args.out_dir or (paths.runs_dir() / "sweep" / stamp))
    for sub in ("artifacts", "metrics", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    _link_latest(out_dir)

    extra_train_args: list[str] = []
    if args.max_samples is not None:
        extra_train_args += ["--max-samples", str(args.max_samples)]

    log.info("=" * 68)
    log.info("NOVUM sweep: %d run(s) over tiers=%s seeds=%s", len(matrix), tiers, seeds)
    log.info("output: %s", paths.rel(out_dir))
    log.info("=" * 68)

    results_file = Path(args.results_file or (paths.PROJECT_ROOT / "results" / "RESULTS.md"))

    records: list[RunRecord] = []
    for i, (tier, seed) in enumerate(matrix, start=1):
        if _INTERRUPTED:
            log.warning("interrupted; %d run(s) not started", len(matrix) - i + 1)
            break
        publish = seed == args.publish_seed
        log.info("[%d/%d] %s-s%d%s", i, len(matrix), tier, seed, " (publishing)" if publish else "")
        records.append(run_one(tier, seed, out_dir, extra_train_args, publish=publish))
        write_results(records, out_dir, started=started)
        write_results_summary(records, results_file, seeds=seeds)

    csv_path, md_path = write_results(records, out_dir, started=started)
    summary_path = write_results_summary(records, results_file, seeds=seeds)

    n_ok = sum(1 for r in records if r.status == STATUS_OK)
    n_stub = sum(1 for r in records if r.status == STATUS_NOT_IMPLEMENTED)
    n_fail = sum(1 for r in records if r.status.endswith("failed"))

    print("", flush=True)
    print("=" * 68)
    print(f"  sweep complete in {human_duration(time.time() - started)}")
    print(f"  {n_ok} ok | {n_stub} not implemented | {n_fail} failed")
    print(f"  table    -> {paths.rel(summary_path)}")
    print(f"  detail   -> {paths.rel(csv_path)}")
    print(f"  detail   -> {paths.rel(md_path)}")
    print("=" * 68)
    for record in records:
        auc = f"{record.roc_auc:.4f}" if record.roc_auc is not None else "-"
        print(f"  {record.label:<20s} {record.status:<16s} ROC AUC {auc}")
    print("", flush=True)

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
