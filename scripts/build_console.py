"""Precompute everything the mission-control console reads.

    make console

Produces, under `web/public/data/`:

    atlas.png       every mission frame as one sprite sheet
    mission.json    the frame stream: sol, class, group, bits, atlas position
    grid.json       the full cross product of runs, with per-window detail

WHY PRECOMPUTE. A judge dragging the downlink slider must get an answer in the
same frame, on a free-tier host, with no GPU and possibly no spare CPU. Replaying
the mission live takes a fifth of a second on a warm laptop and rather longer on
a shared vCPU, and it needs the 3.7 GB processed dataset and the trained
artifacts. So the grid is computed once, here, and committed: after `git clone`
the console runs with no data download, no training step, and no model load.

The cross product is the finding, so every axis is independent -- flight
hardware, model tier, downlink budget, adaptation and policy all vary freely.
Choosing snapdragon-on-RAD750 has to be reachable, because watching the
expensive model fail to afford a single score per window is the demo.

SIZE. Per-window frame lists are the bulk. `arrived` is identical for every run
(the mission stream does not depend on the policy) so it lives once in
mission.json; each run stores only what it selected and what it lost.
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
from core.logging_utils import get_logger, setup_logging

log = get_logger("novum.console")

#: Model tiers, and the flight processor each is costed against. Every
#: (tier, hardware) pair is computed, including the ones that fail badly --
#: especially those.
TIERS = ("rad750", "myriad", "snapdragon")
HARDWARE = ("rad750", "myriad", "snapdragon")

#: Policies the console offers. `oracle` reads ground-truth labels and cannot
#: run onboard; it is the ceiling, labelled as such in the UI.
POLICIES = ("fifo", "score_first", "oracle")

#: Downlink budget ladder, as a fraction of the bits captured per window. A
#: slider needs discrete stops to index a precomputed grid; these span "almost
#: nothing gets through" to "bandwidth barely binds".
BUDGETS = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60)

ADAPTATIONS = ("frozen", "online")

#: The default cell on first load. It has to make the point with no interaction:
#: the real flight processor, the model that fits on it, a scarce budget.
DEFAULT_CELL = {
    "hardware": "rad750",
    "tier": "rad750",
    "budget": 0.25,
    "adaptation": "frozen",
    "policy": "score_first",
}


def cell_key(hardware: str, tier: str, budget: float, adaptation: str, policy: str) -> str:
    return f"{hardware}|{tier}|{budget:g}|{adaptation}|{policy}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _json_bytes(obj: object) -> bytes:
    # separators: this file is read by a browser, not a human.
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
def build_mission_payload(mission, windows) -> dict:
    """The frame stream and window plan, shared by every cell in the grid."""
    from core.thumbnails import atlas_geometry

    columns, rows, width, height = atlas_geometry(len(mission))
    frames = []
    for f in mission.frames:
        r, c = divmod(f.index, columns)
        frames.append(
            {
                "i": f.index,
                "sol": f.sol,
                "group": f.group,
                "cls": f.class_,
                "bits": round(f.bits, 1),
                "ax": c,
                "ay": r,
            }
        )
    return {
        "n_frames": len(mission),
        "composition": mission.composition(),
        "sol_min": min(f.sol for f in mission.frames),
        "sol_max": max(f.sol for f in mission.frames),
        "atlas": {"columns": columns, "rows": rows, "width": width, "height": height},
        "frames": frames,
        "windows": [
            {
                "w": win.index,
                "first_sol": win.first_sol,
                "last_sol": win.last_sol,
                "arrived": list(win.frame_indices),
            }
            for win in windows
        ],
    }


def _run_payload(result) -> dict:
    """One cell: headline metrics plus per-window detail the UI draws."""
    return {
        "science_yield": round(result.science_yield, 4),
        "n_sent": result.n_sent,
        "n_sent_natural": result.n_sent_natural,
        "n_natural_total": result.n_natural_total,
        "n_expired": result.n_expired,
        "n_expired_natural": result.n_expired_natural,
        "wasted_bit_share": round(result.wasted_bit_share, 4),
        "precision_natural": round(result.precision_natural, 4),
        "bits_used": round(result.bits_used, 1),
        "bits_available": round(result.bits_available, 1),
        "prefilter_recall_natural": round(result.prefilter_recall_natural, 4),
        "n_natural_never_scored": result.n_natural_never_scored,
        "n_unscored": result.n_unscored,
        "n_refits": result.n_refits,
        "cycles_per_score": round(result.cycles_per_score, 1),
        "scores_affordable_per_window": (
            round(result.scores_affordable_per_window, 2)
            if result.scores_affordable_per_window
            else None
        ),
        "windows": [
            {
                "w": r.window,
                "sent": r.n_selected,
                "arrived": r.n_arrived,
                "buffered": r.n_buffered,
                "scored": r.n_scored,
                "unscored": r.n_unscored,
                "expired": r.n_expired,
                "evicted": r.n_evicted,
                "bound": r.binding_constraint,
                "bits_used": round(r.bits_used, 1),
                "bits_budget": round(r.bits_budget, 1),
                "cycles_used": round(r.cycles_used, 1),
                "cycles_budget": round(r.cycles_budget, 1),
                "nat": r.sent_natural,
                "rov": r.sent_rover,
                "typ": r.sent_typical,
                "cum_nat": r.cum_sent_natural,
                "cum_avail": r.cum_natural_available,
                "cum_yield": round(r.cum_science_yield, 4),
                "recall": (
                    round(r.prefilter_recall, 4) if r.prefilter_recall is not None else None
                ),
                "refit": r.refit,
                # Which frames, not how many. The console's whole argument is
                # about *which* images were worth their bits; the buffer at any
                # window is derived client-side as everything that has arrived
                # minus everything sent or lost.
                "sel": list(r.selected_indices),
                "lost": list(r.lost_indices),
            }
            for r in result.windows
        ],
    }


#: Set once per worker process. The mission array is 84 MB; rebuilding it per
#: cell would cost more than the replays do.
_WORKER_MISSION = None


def _worker_init() -> None:
    global _WORKER_MISSION

    # One thread per worker. BLAS and torch each default to filling the machine,
    # so N processes times N threads oversubscribes every core and the whole
    # grid runs slower than it would on four workers. Parallelism belongs at the
    # cell level here -- the cells are independent and the inner ops are small.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    from sim.mission import build_mission

    setup_logging("WARNING", force=True)   # 300 replays of INFO is not progress
    _WORKER_MISSION = build_mission()


def _replay_lane(lane: tuple) -> list[tuple[str, dict]]:
    """Replay every downlink budget for one (hardware, tier, adaptation, policy).

    A LANE, not a cell, and that grouping is the whole optimisation. The refit
    pool is every frame captured so far, which does not depend on the downlink
    budget -- so the six budgets in a lane repeat an identical sequence of model
    fits. Running them in one process lets them share `refit_cache` and pay for
    that sequence once instead of six times.
    """
    hw, tier, adaptation, policy, budgets, profile = lane
    from core.models.registry import load_model
    from sim.window import SimConfig, replay

    artifact = paths.artifacts_dir() / f"{tier}.npz"
    refit_cache: dict = {}
    out: list[tuple[str, dict]] = []

    for budget in budgets:
        # Always a pristine model: an online run refits in place, and a cell
        # that inherited another cell's learning would not be the run it claims
        # to be. The cache returns copies, so this stays true.
        model = load_model(artifact)
        config = SimConfig(
            downlink_fraction=budget,
            adaptation=adaptation,
            hardware=hw,
            cycles_per_flop=profile["cycles_per_flop"],
        ).validate()
        result = replay(
            _WORKER_MISSION,
            model,
            method=policy,
            config=config,
            cycles_per_score=(
                float(model.flops_per_inference()) * profile["cycles_per_flop"]
            ),
            budget_cycles_per_score=profile["reference_cycles_per_score"],
            artifact=f"{tier}.npz",
            tier=tier,
            refit_cache=refit_cache,
        )
        out.append(
            (cell_key(hw, tier, budget, adaptation, policy), _run_payload(result))
        )
    return out


def build_grid(*, budgets, adaptations, quick: bool, workers: int) -> dict:
    """Replay every (hardware, tier, budget, adaptation, policy) combination.

    Costing follows `scripts.simulate.hardware_profile`: a model is charged its
    real cost on the chosen silicon, against the cycle budget that silicon was
    provisioned for by its OWN reference model. That asymmetry is the point --
    it is why snapdragon on a RAD750 cannot afford a single score per window.

    Cells are independent, so they fan out across processes. Online adaptation
    on an autoencoder tier refits seven times per run at a couple of seconds an
    epoch; serially the full grid is hours, and the whole reason this is
    precomputed is that nobody should wait for it twice.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from scripts.simulate import hardware_profile

    tiers = TIERS[:1] if quick else TIERS
    hardware = HARDWARE[:1] if quick else HARDWARE

    available: list[str] = []
    for tier in tiers:
        if (paths.artifacts_dir() / f"{tier}.npz").exists():
            available.append(tier)
        else:
            log.warning("no artifact for tier %s; skipping", tier)
    if not available:
        raise FileNotFoundError(
            "no trained artifacts found; run `make train` or fetch the committed ones"
        )

    profiles = {hw: hardware_profile(hw) for hw in hardware}
    lanes = [
        (hw, tier, adaptation, policy, list(budgets), profiles[hw])
        for hw in hardware
        for tier in available
        for adaptation in adaptations
        for policy in POLICIES
    ]
    # Slowest first: an online autoencoder lane runs seven model fits and a
    # frozen PCA lane runs none, so scheduling the long ones first keeps every
    # worker busy to the end instead of leaving one straggler running alone.
    lanes.sort(key=lambda lane: (lane[2] != "online", lane[1] == "rad750"))

    cells: dict[str, dict] = {}
    n_cells = sum(len(lane[4]) for lane in lanes)
    started = time.monotonic()
    log.info(
        "replaying %d cells in %d lanes across %d workers",
        n_cells, len(lanes), workers,
    )

    # as_completed, not map: map yields in submission order, and since the
    # slowest lanes are submitted first that reports nothing at all until the
    # longest run finishes. A grid build that looks hung for twenty minutes is
    # one somebody kills.
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = [pool.submit(_replay_lane, lane) for lane in lanes]
        for done, future in enumerate(as_completed(futures), start=1):
            for key, payload in future.result():
                cells[key] = payload
            elapsed = time.monotonic() - started
            log.info(
                "  %d/%d lanes, %d cells (%.0fs elapsed, ~%.0fs left)",
                done, len(lanes), len(cells), elapsed,
                elapsed / done * (len(lanes) - done),
            )

    return {
        "axes": {
            "hardware": list(hardware),
            "tier": list(available),
            "budget": list(budgets),
            "adaptation": list(adaptations),
            "policy": list(POLICIES),
        },
        "processors": {
            hw: {
                "processor": p["processor"],
                "cycles_per_flop": p["cycles_per_flop"],
                "reference_cycles_per_score": round(p["reference_cycles_per_score"], 1),
            }
            for hw, p in profiles.items()
        },
        "default": DEFAULT_CELL,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.build_console",
        description="Precompute the atlas, mission stream and run grid for web/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: web/public/data.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="One tier, one processor: a smoke test, not a shippable grid.",
    )
    p.add_argument(
        "--skip-atlas", action="store_true", help="Leave an existing atlas.png alone."
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Parallel replay processes. Each holds an 84 MB copy of the mission.",
    )
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)
    load_env()

    out = args.out or (paths.PROJECT_ROOT / "web" / "public" / "data")
    out.mkdir(parents=True, exist_ok=True)

    from sim.mission import build_mission
    from sim.window import SimConfig, plan_windows

    log.info("building mission stream")
    mission = build_mission()
    # The window plan must match what `replay` derives, or the UI misaligns the
    # frame grid against the decisions it is meant to explain.
    windows = plan_windows(mission.rows, sols_per_window=SimConfig().sols_per_window)
    log.info(
        "mission: %d frames, %d windows, composition %s",
        len(mission), len(windows), mission.composition(),
    )

    # --- atlas ----------------------------------------------------------
    atlas_path = out / "atlas.png"
    if args.skip_atlas and atlas_path.exists():
        log.info("keeping existing %s", paths.rel(atlas_path))
    else:
        from core.thumbnails import build_atlas

        log.info("rendering %d frames to a sprite atlas", len(mission))
        png, atlas_meta = build_atlas(mission.array)
        _atomic_write(atlas_path, png)
        _atomic_write(out / "atlas.json", _json_bytes(atlas_meta))
        log.info(
            "atlas -> %s (%.1f MB, %dx%d)",
            paths.rel(atlas_path), len(png) / 1e6,
            atlas_meta["width"], atlas_meta["height"],
        )

    # --- mission stream --------------------------------------------------
    payload = build_mission_payload(mission, windows)
    _atomic_write(out / "mission.json", _json_bytes(payload))
    log.info("mission stream -> %s", paths.rel(out / "mission.json"))

    # --- grid ------------------------------------------------------------
    budgets = BUDGETS[:2] if args.quick else BUDGETS
    adaptations = ADAPTATIONS[:1] if args.quick else ADAPTATIONS
    log.info("replaying the run grid")
    grid = build_grid(
        budgets=budgets,
        adaptations=adaptations,
        quick=args.quick,
        workers=args.workers,
    )
    grid_bytes = _json_bytes(grid)
    _atomic_write(out / "grid.json", grid_bytes)
    log.info(
        "grid -> %s (%d cells, %.1f MB)",
        paths.rel(out / "grid.json"), len(grid["cells"]), len(grid_bytes) / 1e6,
    )

    print(f"console data -> {paths.rel(out)}")
    print(f"  atlas.png    {atlas_path.stat().st_size / 1e6:.1f} MB")
    print(f"  mission.json {(out / 'mission.json').stat().st_size / 1e6:.1f} MB")
    print(f"  grid.json    {len(grid_bytes) / 1e6:.1f} MB  ({len(grid['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
