"""The console's briefing endpoint reads the same runs the panels draw.

Two things matter here. The adapter must map the console's short keys onto the
record shape `report_gen` expects without silently dropping a field, and the
whole path must work with no API key at all -- a console that only reads well
when a key is present fails in front of the person it was built for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CELL = {
    "science_yield": 0.5562,
    "n_sent": 180,
    "n_sent_natural": 94,
    "n_natural_total": 169,
    "n_expired": 604,
    "n_expired_natural": 70,
    "wasted_bit_share": 0.1601,
    "precision_natural": 0.522,
    "bits_used": 10359770.3,
    "bits_available": 10688541.9,
    "prefilter_recall_natural": 0.828,
    "n_natural_never_scored": 29,
    "n_unscored": 1019,
    "n_refits": 0,
    "cycles_per_score": 2599296.0,
    "scores_affordable_per_window": 31.7,
    "windows": [
        {
            "w": w,
            "sent": 6,
            "arrived": 39,
            "buffered": 130,
            "scored": 30,
            "unscored": 68,
            "expired": 40,
            "evicted": 0,
            "bound": "bits",
            "bits_used": 381065.5,
            "bits_budget": 395871.9,
            "cycles_used": 80244096.0,
            "cycles_budget": 82407310.0,
            "nat": 4,
            "rov": 0,
            "typ": 2,
            "cum_nat": 8 * (w + 1),
            "cum_avail": 23 * (w + 1),
            "cum_yield": 8 / 23,
            "recall": 0.21,
            "refit": False,
            "sel": [1, 2, 3, 4, 5, 6],
            "lost": [7, 8],
        }
        for w in range(3)
    ],
}

MISSION = {
    "n_frames": 856,
    "composition": {"natural": 169, "rover": 261, "typical": 426},
    "sol_min": 13,
    "sol_max": 1666,
    "atlas": {"columns": 32, "rows": 27, "width": 1024, "height": 864},
    "frames": [],
    "windows": [
        {"w": w, "first_sol": 13 + 50 * w, "last_sol": 62 + 50 * w, "arrived": []}
        for w in range(3)
    ],
}


@pytest.fixture()
def console_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    grid = {
        "axes": {
            "hardware": ["rad750"],
            "tier": ["rad750"],
            "budget": [0.25],
            "adaptation": ["frozen"],
            "policy": ["fifo", "score_first", "oracle"],
        },
        "processors": {"rad750": {"processor": "BAE RAD750", "cycles_per_flop": 3.0,
                                  "reference_cycles_per_score": 2599296.0}},
        "default": {
            "hardware": "rad750", "tier": "rad750", "budget": 0.25,
            "adaptation": "frozen", "policy": "score_first",
        },
        "cells": {
            "rad750|rad750|0.25|frozen|score_first": CELL,
            "rad750|rad750|0.25|frozen|fifo": dict(CELL, n_sent_natural=26,
                                                   science_yield=0.1538),
        },
    }
    (tmp_path / "grid.json").write_text(json.dumps(grid), encoding="utf-8")
    (tmp_path / "mission.json").write_text(json.dumps(MISSION), encoding="utf-8")

    monkeypatch.setenv("NOVUM_CONSOLE_DIR", str(tmp_path))
    from core.ground import console_brief

    console_brief.load_grid.cache_clear()
    console_brief.load_mission.cache_clear()
    yield tmp_path
    console_brief.load_grid.cache_clear()
    console_brief.load_mission.cache_clear()


def test_cell_key_matches_the_builder() -> None:
    """One typo here and every briefing 404s while the panels keep working."""
    from core.ground.console_brief import cell_key
    from scripts.build_console import cell_key as builder_key

    for budget in (0.05, 0.1, 0.25, 0.6):
        assert cell_key("rad750", "myriad", budget, "online", "fifo") == builder_key(
            "rad750", "myriad", budget, "online", "fifo"
        )


def test_brief_works_with_no_api_key(
    console_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from core.ground.console_brief import brief_for_cell

    brief = brief_for_cell("rad750", "rad750", 0.25, "frozen", "score_first")
    assert brief["mode"] == "offline"
    assert brief["skip_reason"] == "offline_requested"
    assert "Mission Briefing" in brief["text"]
    # The figures the panels show must be the figures the prose states.
    assert "55.6%" in brief["text"]
    assert "94 of 169" in brief["text"]


def test_brief_carries_the_fifo_comparison(console_dir: Path) -> None:
    from core.ground.console_brief import brief_for_cell

    text = brief_for_cell("rad750", "rad750", 0.25, "frozen", "score_first")["text"]
    assert "FIFO baseline yield" in text


def test_unknown_cell_raises_keyerror(console_dir: Path) -> None:
    from core.ground.console_brief import brief_for_cell

    with pytest.raises(KeyError):
        brief_for_cell("myriad", "snapdragon", 0.6, "online", "oracle")


def test_missing_grid_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOVUM_CONSOLE_DIR", str(tmp_path / "nothing-here"))
    from core.ground import console_brief

    console_brief.load_grid.cache_clear()
    with pytest.raises(FileNotFoundError, match="make console"):
        console_brief.load_grid()
    console_brief.load_grid.cache_clear()


def test_window_records_keep_every_field_report_gen_reads(console_dir: Path) -> None:
    """The adapter is where a renamed key would silently become a missing figure."""
    from core.ground.console_brief import _window_records, load_mission

    records = _window_records(CELL, load_mission(), "score_first")
    required = {
        "window", "first_sol", "last_sol", "n_arrived", "n_buffered", "n_scored",
        "n_unscored", "n_selected", "n_expired", "n_evicted", "bits_budget",
        "bits_used", "cycles_budget", "cycles_used", "binding_constraint",
        "sent_natural", "sent_rover", "sent_typical", "cum_sent_natural",
        "cum_natural_available", "cum_science_yield", "prefilter_recall", "refit",
    }
    assert required <= set(records[0])
    assert records[0]["first_sol"] == 13


def test_api_brief_endpoint_serves_offline(
    console_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Called as the router calls it — no HTTP client dependency needed."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from api.main import mission_brief

    body = mission_brief(
        hardware="rad750", tier="rad750", budget=0.25,
        adaptation="frozen", policy="score_first",
    )
    assert body["mode"] == "offline"
    assert body["text"]

    with pytest.raises(HTTPException) as exc:
        mission_brief(tier="snapdragon", budget=0.6)
    assert exc.value.status_code == 404


def test_api_axes_endpoint_reports_the_grid(console_dir: Path) -> None:
    pytest.importorskip("fastapi")
    from api.main import console_axes

    body = console_axes()
    assert body["n_cells"] == 2
    assert body["default"]["policy"] == "score_first"
    assert "rad750" in body["processors"]
