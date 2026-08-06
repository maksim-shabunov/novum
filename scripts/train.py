"""Train one tier and write a weight artifact plus its provenance sidecar.

    python -m scripts.train --config configs/tier_rad750.yaml --out artifacts/rad750.npz
    python -m scripts.train --config configs/tier_rad750.yaml --seed 3 --set model.n_components=32

Writes two files:

    artifacts/<name>.npz    weights (committed to git -- kilobytes to megabytes)
    artifacts/<name>.json   config hash, git commit, wall clock, parameter count,
                            estimated FLOPs per inference, peak RSS

Exit codes:
    0  trained and written
    2  bad arguments, missing config, or missing processed data
    3  the requested tier is a stub (NotImplementedError) -- sweep.py relies on
       this being distinguishable from a genuine failure
    1  anything else
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core import paths
from core.config import ConfigError, load_config, parse_overrides, resolved_config_hash
from core.dataset import ChunkedArray, load_array
from core.logging_utils import get_logger, human_bytes, human_duration, setup_logging
from core.models.registry import build_model
from core.provenance import RunProvenance, Timer, sidecar_path_for

log = get_logger("novum.train")

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_NOT_IMPLEMENTED = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.train",
        description="Train a NOVUM novelty model from a tier config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path, help="path to a configs/tier_*.yaml")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .npz (default: artifacts/<tier>.npz)",
    )
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set model.n_components=32 (repeatable)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap training frames (smoke tests); overrides data.max_train_samples",
    )
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    # -- config -------------------------------------------------------------
    try:
        overrides = parse_overrides(args.set)
        if args.seed is not None:
            overrides.setdefault("seed", args.seed)
        if args.max_samples is not None:
            overrides.setdefault("data", {})["max_train_samples"] = args.max_samples
        cfg = load_config(args.config, overrides)
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_BAD_ARGS

    tier = cfg["tier"]
    out_path = args.out or (paths.artifacts_dir() / f"{tier}.npz")
    out_path = Path(out_path)
    seed = int(cfg["seed"])

    log.info("=" * 68)
    log.info("tier            %s (%s)", tier, cfg["model"]["type"])
    log.info("config          %s", paths.rel(Path(cfg["_source"])))
    log.info("config hash     %s", resolved_config_hash(cfg)[:16])
    log.info("seed            %d", seed)
    log.info("output          %s", paths.rel(out_path))
    log.info("=" * 68)

    # -- data ---------------------------------------------------------------
    train_split = cfg["data"]["train_split"]
    try:
        array = load_array(train_split)
    except (FileNotFoundError, KeyError) as exc:
        log.error("%s", exc)
        return EXIT_BAD_ARGS

    limit = cfg["data"].get("max_train_samples")
    chunks = ChunkedArray(array, cfg["data"].get("chunk_size", 512), limit=limit)
    n_samples = chunks.n_frames
    if n_samples == 0:
        log.error("training split %s is empty", train_split)
        return EXIT_BAD_ARGS
    log.info(
        "training on %d frames from %s%s",
        n_samples,
        train_split,
        f" (capped from {len(array)})" if limit else "",
    )

    # -- fit ----------------------------------------------------------------
    try:
        model = build_model(cfg)
    except (KeyError, ValueError) as exc:
        log.error("could not build model: %s", exc)
        return EXIT_BAD_ARGS

    try:
        with Timer() as timer:
            model.fit(chunks, n_samples=n_samples, seed=seed)
    except NotImplementedError as exc:
        log.error("%s", exc)
        log.error("tier %r is a stub; nothing was written", tier)
        return EXIT_NOT_IMPLEMENTED

    log.info("fit completed in %s", human_duration(timer.elapsed))

    # -- persist ------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)

    provenance = RunProvenance.build(
        name=out_path.stem,
        tier=tier,
        config=cfg,
        wall_clock_seconds=timer.elapsed,
        param_count=model.param_count(),
        flops_per_inference=model.flops_per_inference(),
        n_train_samples=n_samples,
        seed=seed,
        extra={"train_split": train_split, "artifact_bytes": out_path.stat().st_size},
    )
    sidecar = provenance.write(sidecar_path_for(out_path))

    # -- report -------------------------------------------------------------
    record = provenance.to_json()
    budget_cycles = cfg["compute"].get("budget_cycles_per_frame")
    cycles_per_flop = float(cfg["compute"].get("cycles_per_flop", 3.0))
    est_cycles = record["flops_per_inference"] * cycles_per_flop

    log.info("-" * 68)
    log.info("  parameters        %s", f"{record['param_count']:,}")
    log.info("  FLOPs/inference   %s", f"{record['flops_per_inference']:,}")
    log.info(
        "  est. cycles/frame %s on %s",
        f"{est_cycles:,.0f}",
        cfg["compute"].get("reference_processor", "unspecified"),
    )
    if budget_cycles:
        pct = 100.0 * est_cycles / float(budget_cycles)
        verdict = "within budget" if pct <= 100 else "OVER BUDGET"
        log.info("  compute budget    %.1f%% of %s cycles (%s)", pct, f"{budget_cycles:,}", verdict)
        if pct > 100:
            log.warning("this model would not fit the %s compute budget onboard", tier)
    log.info("  peak RSS          %s", human_bytes(record["peak_rss_bytes"]))
    log.info("  wall clock        %s", human_duration(record["wall_clock_seconds"]))
    log.info("  artifact          %s (%s)", paths.rel(out_path), human_bytes(out_path.stat().st_size))
    log.info("  sidecar           %s", paths.rel(sidecar))
    log.info("-" * 68)
    log.info("next: python -m scripts.evaluate --artifact %s", paths.rel(out_path))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
