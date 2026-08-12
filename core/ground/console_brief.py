"""Turn a precomputed console grid cell back into a mission briefing.

The console ships `grid.json`: every (hardware, tier, budget, adaptation,
policy) run, replayed offline, with short keys because a browser downloads it.
`report_gen` speaks the simulator's own record shape. This module is the
adapter between them, so the briefing a judge reads is generated from exactly
the numbers the panels beside it are drawing -- not from a separate run that
happened to be recorded on a different day.

Deliberately numpy-free and torch-free: it is imported by the API, and the
architectural rule is that nothing in the serving path drags in a trainer.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from core import paths

from .llm_provider import REASON_HELP, get_model
from .report_gen import mission_summary


def console_dir() -> Path:
    """Where `scripts.build_console` wrote its output."""
    import os

    raw = os.environ.get("NOVUM_CONSOLE_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (paths.PROJECT_ROOT / p)
    return paths.PROJECT_ROOT / "web" / "public" / "data"


@lru_cache(maxsize=1)
def load_grid() -> dict[str, Any]:
    path = console_dir() / "grid.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no console grid at {path}. Run `make console` to build it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_mission() -> dict[str, Any]:
    path = console_dir() / "mission.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no mission stream at {path}. Run `make console` to build it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cell_key(hardware: str, tier: str, budget: float, adaptation: str, policy: str) -> str:
    """Must match `cell_key` in scripts/build_console.py."""
    return f"{hardware}|{tier}|{budget:g}|{adaptation}|{policy}"


# ---------------------------------------------------------------------------
# Shape adapters
# ---------------------------------------------------------------------------
def _window_records(cell: dict, mission: dict, policy: str) -> list[dict]:
    """Console window rows -> the record shape `report_gen` reads."""
    sols = {w["w"]: (w["first_sol"], w["last_sol"]) for w in mission.get("windows", [])}
    out: list[dict] = []
    for r in cell.get("windows", []):
        first, last = sols.get(r["w"], (-1, -1))
        out.append(
            {
                "window": r["w"],
                "first_sol": first,
                "last_sol": last,
                "method": policy,
                "n_arrived": r["arrived"],
                "n_buffered": r["buffered"],
                "n_scored": r["scored"],
                "n_unscored": r["unscored"],
                "n_selected": r["sent"],
                "n_expired": r["expired"],
                "n_evicted": r["evicted"],
                "bits_budget": r["bits_budget"],
                "bits_used": r["bits_used"],
                "cycles_budget": r["cycles_budget"],
                "cycles_used": r["cycles_used"],
                "binding_constraint": r["bound"],
                "sent_natural": r["nat"],
                "sent_rover": r["rov"],
                "sent_typical": r["typ"],
                "cum_sent_natural": r["cum_nat"],
                "cum_natural_available": r["cum_avail"],
                "cum_science_yield": r["cum_yield"],
                "prefilter_recall": r.get("recall"),
                "refit": r.get("refit", False),
            }
        )
    return out


def _run_meta(cell: dict, hardware: str, tier: str, adaptation: str, policy: str) -> dict:
    return {
        "method": policy,
        "tier": tier,
        "hardware": hardware,
        "adaptation": adaptation,
        "windows": len(cell.get("windows", [])),
        "science_yield": cell["science_yield"],
        "wasted_bit_share": cell["wasted_bit_share"],
        "n_sent": cell["n_sent"],
        "n_sent_natural": cell["n_sent_natural"],
        "n_natural_total": cell["n_natural_total"],
        "bits_used": cell["bits_used"],
        "bits_available": cell["bits_available"],
        "n_expired": cell["n_expired"],
        "n_expired_natural": cell["n_expired_natural"],
        "n_refits": cell["n_refits"],
        "n_unscored": cell["n_unscored"],
        "prefilter_recall_natural": cell["prefilter_recall_natural"],
        "n_natural_never_scored": cell["n_natural_never_scored"],
        "scores_affordable_per_window": cell.get("scores_affordable_per_window"),
    }


def brief_for_cell(
    hardware: str,
    tier: str,
    budget: float,
    adaptation: str,
    policy: str,
    *,
    offline: bool = True,
) -> dict[str, Any]:
    """The mission briefing for one console cell.

    `offline=True` by default and that is not a limitation: the deterministic
    template states the same figures the model would have been handed, in the
    same order, in under a millisecond. A console that only reads well with an
    API key is a console that fails in front of the person it was built for.
    """
    grid = load_grid()
    mission = load_mission()

    key = cell_key(hardware, tier, budget, adaptation, policy)
    cell = grid["cells"].get(key)
    if cell is None:
        raise KeyError(key)

    windows = _window_records(cell, mission, policy)
    run = _run_meta(cell, hardware, tier, adaptation, policy)
    fifo_cell = grid["cells"].get(cell_key(hardware, tier, budget, adaptation, "fifo"))
    fifo = (
        _run_meta(fifo_cell, hardware, tier, adaptation, "fifo") if fifo_cell else None
    )
    mission_meta = {
        "n_frames": mission.get("n_frames"),
        "sol_min": mission.get("sol_min"),
        "sol_max": mission.get("sol_max"),
        "composition": mission.get("composition", {}),
    }

    generation = mission_summary(windows, run, fifo, mission_meta, offline=offline)
    return {
        "text": generation.text,
        "mode": "llm" if generation.used_llm else "offline",
        "skip_reason": generation.skip_reason,
        "skip_help": REASON_HELP.get(generation.skip_reason or "", None),
        "model": get_model() if generation.used_llm else None,
        "usage": generation.usage.as_dict() if generation.usage else None,
        "cell": key,
    }
