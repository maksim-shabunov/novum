"""Ground-side mission brief generator.

Reads the decision log written by the simulator and produces a two-level
operator briefing:
  - per-window notes written back into the .jsonl records
  - a mission summary written to results/MISSION_BRIEF.md

Usage:
    python -m scripts.mission_brief RUN=<run_id>
    make report-mission RUN=<run_id>
    make report-mission RUN=<run_id> VARIANT=rad750-score_first

Flags:
    --offline     Force the template path; never calls the LLM.
    --run-id      The run directory name under runs/sim/ (or "latest").
    --variant     The JSONL file stem (default: first score_first file found).
    --out         Override output path (default: results/MISSION_BRIEF.md).
    --model       Override NOVUM_REPORT_MODEL for this invocation.
    --no-write-back  Skip writing notes back into the .jsonl file.

Every token count and cost estimate is written to
    runs/sim/<run_id>/report_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from core import paths
from core.env import load_env
from core.ground.llm_provider import REASON_HELP, Usage, get_model
from core.ground.report_gen import (
    Generation,
    mission_summary,
    validate_numbers,
    window_note,
)
from core.logging_utils import get_logger, setup_logging

log = get_logger("novum.mission_brief")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_run_dir(run_id: str) -> Path:
    sim_dir = paths.runs_dir() / "sim"
    if run_id in ("latest", "online-latest"):
        # Accept symlink or directory
        candidate = sim_dir / run_id
        if candidate.exists():
            return candidate.resolve()
        # Fall back to newest timestamped dir
        dirs = sorted(
            (d for d in sim_dir.iterdir() if d.is_dir() and d.name[0].isdigit()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not dirs:
            raise FileNotFoundError(f"No sim runs found in {sim_dir}")
        return dirs[0].resolve()

    candidate = sim_dir / run_id
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Run directory not found: {candidate}\n"
            f"Available: {[d.name for d in sim_dir.iterdir() if d.is_dir()]}"
        )
    return candidate.resolve()


def _pick_variant(windows_dir: Path, variant: str | None) -> Path:
    """Return the .jsonl path for the chosen variant."""
    if variant:
        # Allow stem or full filename
        if not variant.endswith(".jsonl"):
            variant = variant + ".jsonl"
        p = windows_dir / variant
        if not p.exists():
            raise FileNotFoundError(f"Variant not found: {p}")
        return p

    # Default: prefer score_first on rad750, then first score_first found
    for stem in ("rad750-score_first", "myriad-score_first", "snapdragon-score_first"):
        p = windows_dir / f"{stem}.jsonl"
        if p.exists():
            return p

    # Any score_first
    candidates = sorted(windows_dir.glob("*-score_first.jsonl"))
    if candidates:
        return candidates[0]

    # Anything at all
    candidates = sorted(windows_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No .jsonl files in {windows_dir}")
    return candidates[0]


def _load_windows(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {jsonl_path}")
    return records


def _find_run_meta(summary: dict, jsonl_stem: str) -> dict | None:
    """Find the run entry in summary.json that matches the jsonl stem."""
    # stem is like "rad750-score_first" -> tier=rad750, method=score_first
    parts = jsonl_stem.split("-", 1)
    if len(parts) != 2:
        return None
    tier, method = parts
    # method may use underscores
    for run in summary.get("runs", []):
        if run.get("tier") == tier and run.get("method") == method:
            return run
    return None


def _find_fifo_run(summary: dict, tier: str) -> dict | None:
    for run in summary.get("runs", []):
        if run.get("tier") == tier and run.get("method") == "fifo":
            return run
    return None


def _accumulate_usage(totals: Usage, u: Usage | None) -> None:
    if u is None:
        return
    totals.prompt_tokens += u.prompt_tokens
    totals.completion_tokens += u.completion_tokens
    totals.cost_usd += u.cost_usd


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _generate_window_notes(
    windows: list[dict],
    *,
    offline: bool,
    n_natural_total: int | None = None,
) -> tuple[list[tuple[dict, Generation]], Usage]:
    """Generate one note per window. Returns (results, total_usage)."""
    results: list[tuple[dict, Generation]] = []
    totals = Usage()
    for rec in windows:
        gen = window_note(rec, offline=offline, n_natural_total=n_natural_total)
        _accumulate_usage(totals, gen.usage)
        results.append((rec, gen))
    return results, totals


def _skip_summary(generations: list[Generation]) -> dict[str, int]:
    """How many generations were skipped, by reason. Empty when all used the LLM."""
    counts: dict[str, int] = {}
    for gen in generations:
        if not gen.used_llm and gen.skip_reason:
            counts[gen.skip_reason] = counts.get(gen.skip_reason, 0) + 1
    return counts


def _write_notes_back(jsonl_path: Path, noted_windows: list[tuple[dict, Generation]]) -> None:
    """Append operator_note field to each record and overwrite the jsonl."""
    lines: list[str] = []
    for rec, gen in noted_windows:
        updated = dict(rec)
        updated["operator_note"] = gen.text
        updated["operator_note_llm"] = gen.used_llm
        lines.append(json.dumps(updated))
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, jsonl_path)
    log.info("wrote notes back to %s", paths.rel(jsonl_path))


def _write_report_meta(run_dir: Path, meta: dict) -> None:
    path = run_dir / "report_meta.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    log.info("report metadata -> %s", paths.rel(path))


def generate_brief(
    run_dir: Path,
    jsonl_path: Path,
    *,
    offline: bool = False,
    write_back: bool = True,
    model_override: str | None = None,
) -> tuple[str, dict]:
    """Top-level function: produce the brief and return (markdown, meta_dict).

    meta_dict contains token usage, cost, and validation results.
    """
    if model_override:
        os.environ["NOVUM_REPORT_MODEL"] = model_override

    # Load data
    windows = _load_windows(jsonl_path)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    stem = jsonl_path.stem
    run_meta = _find_run_meta(summary, stem)
    if run_meta is None:
        log.warning("could not find matching run entry for %s in summary.json; using empty", stem)
        run_meta = {"method": "unknown", "tier": "unknown"}

    fifo_meta = _find_fifo_run(summary, run_meta.get("tier", ""))
    mission = summary.get("mission", {})

    t0 = time.monotonic()
    total_usage = Usage()

    # --- Per-window notes ---
    # The mission total is what makes a per-window denominator judgeable: 31
    # natural frames is a thin basis for a ratio when the mission holds 169.
    noted_windows, window_totals = _generate_window_notes(
        windows, offline=offline, n_natural_total=run_meta.get("n_natural_total")
    )
    _accumulate_usage(total_usage, window_totals)

    # Write notes back into jsonl
    if write_back:
        _write_notes_back(jsonl_path, noted_windows)

    # --- Mission summary ---
    summary = mission_summary(
        windows=windows,
        run=run_meta,
        fifo=fifo_meta,
        mission=mission,
        offline=offline,
    )
    _accumulate_usage(total_usage, summary.usage)

    # --- Validation ---
    untraced = validate_numbers(summary.text, windows, run_meta, mission, fifo_meta)
    validation_note = ""
    if untraced:
        log.warning(
            "validation: %d number(s) in the summary could not be traced to source: %s",
            len(untraced),
            untraced[:10],
        )
        validation_note = (
            "\n\n---\n⚠ **Validation flag**: the following numbers in the summary "
            "could not be traced to the source decision log: "
            + ", ".join(untraced[:20])
            + ". Treat these figures with caution.\n"
        )

    # --- Assemble document ---
    all_generations = [gen for _, gen in noted_windows] + [summary]
    n_llm = sum(1 for _, gen in noted_windows if gen.used_llm)
    n_template = len(noted_windows) - n_llm
    skips = _skip_summary(all_generations)

    # Never "LLM unavailable": name the cause and what to do about it.
    if summary.used_llm and not skips:
        mode_tag = "🤖 LLM-generated"
    elif summary.used_llm:
        mode_tag = f"🤖 LLM-generated, {n_template} window note(s) fell back"
    else:
        mode_tag = f"📋 OFFLINE — template-rendered ({summary.reason_line()})"

    header_lines = [
        f"<!-- generated by scripts.mission_brief on "
        f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} -->",
        f"<!-- mode: {mode_tag} | model: {get_model()} -->",
        "",
    ]

    # Per-window section
    window_lines = ["## Per-Window Operator Notes", ""]
    for rec, gen in noted_windows:
        tag = "" if gen.used_llm else f" *(template — {gen.skip_reason})*"
        window_lines.append(f"### Window {rec['window']}{tag}")
        window_lines.append("")
        window_lines.append(gen.text)
        window_lines.append("")

    # One line per distinct cause, so 27 identical failures read as one fact.
    skip_note = ""
    if skips:
        skip_note = "\n\n---\n**Provider skipped** — " + "; ".join(
            f"`{reason}` × {count}: {REASON_HELP.get(reason, 'no detail')}"
            for reason, count in sorted(skips.items())
        ) + "\n"

    full_doc = (
        "\n".join(header_lines)
        + summary.text
        + "\n"
        + "\n".join(window_lines)
        + skip_note
        + validation_note
    )

    elapsed = time.monotonic() - t0
    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        "run_id": run_dir.name,
        "jsonl": str(paths.rel(jsonl_path)),
        "mode": "llm" if summary.used_llm else "offline",
        "model": get_model(),
        "skip_reason": summary.skip_reason,
        "skip_detail": summary.skip_detail,
        "skips_by_reason": skips,
        "unknown_placeholders": sorted(
            {k for gen in all_generations for k in gen.unknown_placeholders}
        ),
        "window_notes": {
            "n_llm": n_llm,
            "n_template": n_template,
        },
        "usage": total_usage.as_dict(),
        "validation_untraced": untraced,
        "elapsed_seconds": round(elapsed, 2),
    }

    return full_doc, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.mission_brief",
        description=(
            "Generate a ground-side mission briefing from the simulator decision log. "
            "Uses OpenRouter by default; falls back to deterministic templates if the "
            "API is unavailable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-id",
        default="online-latest",
        help="Run directory under runs/sim/ (or 'latest').",
    )
    p.add_argument(
        "--variant",
        default=None,
        help="JSONL file stem, e.g. rad750-score_first. Default: first score_first found.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Default: results/MISSION_BRIEF.md.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Force the template path; never call the LLM.",
    )
    p.add_argument(
        "--no-write-back",
        action="store_true",
        help="Do not write operator notes back into the .jsonl file.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the LLM model for this run (any OpenRouter model id).",
    )
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    # Process entry: credentials become visible here and nowhere else.
    loaded = load_env()
    log.debug("env file: %s", paths.rel(loaded) if loaded else "none (using os.environ only)")

    try:
        run_dir = _resolve_run_dir(args.run_id)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    try:
        windows_dir = run_dir / "windows"
        jsonl_path = _pick_variant(windows_dir, args.variant)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    log.info(
        "generating brief: run=%s variant=%s offline=%s",
        run_dir.name,
        paths.rel(jsonl_path),
        args.offline,
    )

    try:
        doc, meta = generate_brief(
            run_dir,
            jsonl_path,
            offline=args.offline,
            write_back=not args.no_write_back,
            model_override=args.model,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    # Write the brief
    out_path = Path(args.out or (paths.PROJECT_ROOT / "results" / "MISSION_BRIEF.md"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".md.tmp")
    tmp.write_text(doc, encoding="utf-8")
    os.replace(tmp, out_path)
    log.info("mission brief -> %s", paths.rel(out_path))

    # Write run metadata
    _write_report_meta(run_dir, meta)

    # Print summary
    u = meta["usage"]
    mode = meta["mode"]
    print(f"{paths.rel(out_path)}")
    if mode == "llm":
        print(
            f"  model:  {meta['model']}\n"
            f"  tokens: {u['prompt_tokens']} in + {u['completion_tokens']} out "
            f"= {u['total_tokens']} total\n"
            f"  cost:   ${u['cost_usd']:.6f}"
        )
    else:
        print(f"  mode:   offline (template) — {meta['skip_reason']}")
        print(f"          {REASON_HELP.get(meta['skip_reason'], '')}")
        if meta["skip_detail"] and meta["skip_reason"] != "offline_requested":
            print(f"          {meta['skip_detail'][:200]}")
    for reason, count in sorted(meta["skips_by_reason"].items()):
        print(f"  skipped: {count} generation(s) — {reason}")
    if meta["validation_untraced"]:
        print(f"  validation: {len(meta['validation_untraced'])} untraced numbers flagged")

    return 0


if __name__ == "__main__":
    sys.exit(main())
