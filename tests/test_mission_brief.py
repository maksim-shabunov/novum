"""Tests for the ground-side mission report (task 6).

Covers:
  - LLM provider: absent key -> ProviderError
  - window_note template path (no network, always works)
  - mission_summary template path (no network, always works)
  - number validation: real numbers pass, invented numbers fail
  - CLI: --offline flag produces a report and exits 0
  - CLI: absent key -> offline fallback, exits 0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Minimal window record matching the real JSONL schema
WINDOW_0 = {
    "window": 0,
    "first_sol": 13,
    "last_sol": 52,
    "method": "score_first",
    "n_arrived": 39,
    "n_buffered": 39,
    "n_prefiltered": 39,
    "n_scored": 25,
    "n_unscored": 14,
    "n_selected": 6,
    "n_expired": 0,
    "n_evicted": 0,
    "bits_budget": 395871.92,
    "bits_used": 388117.96,
    "cycles_budget": 82407310.0,
    "cycles_used": 80244096.0,
    "cycles_scoring": 64982400.0,
    "cycles_prefilter": 14376960.0,
    "binding_constraint": "bits",
    "sent_natural": 0,
    "sent_rover": 0,
    "sent_typical": 6,
    "bits_natural": 0.0,
    "bits_rover": 0.0,
    "bits_typical": 388117.96,
    "cum_sent_natural": 0,
    "cum_natural_available": 0,
    "cum_science_yield": 0.0,
    "cum_wasted_bit_share": 0.0,
    "refit": False,
    "prefilter_recall": None,
}

RUN_META = {
    "method": "score_first",
    "tier": "rad750",
    "adaptation": "frozen",
    "windows": 1,
    "science_yield": 0.55,
    "wasted_bit_share": 0.06,
    "n_sent": 6,
    "n_sent_natural": 0,
    "n_sent_rover": 0,
    "n_sent_typical": 6,
    "n_natural_total": 169,
    "bits_used": 388117.96,
    "bits_available": 395871.92,
    "n_expired": 0,
    "n_refits": 0,
}

FIFO_META = {
    "method": "fifo",
    "tier": "rad750",
    "adaptation": "frozen",
    "science_yield": 0.15,
    "wasted_bit_share": 0.32,
}

MISSION = {
    "n_frames": 856,
    "sol_min": 13,
    "sol_max": 1666,
    "composition": {"natural": 169, "rover": 261, "typical": 426},
}


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


def test_provider_raises_when_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from core.ground.llm_provider import ProviderError, complete

    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        complete("system", "user")


# NOTE: these deliberately do NOT reload the module. `get_model()` reads the
# environment on every call, so a reload buys nothing -- and it re-binds
# ProviderError to a new class object while `report_gen`'s `except` clause still
# holds the old one, so every fallback in the suite stops being caught. That
# failure only appears when the tests run in a particular order, which is the
# worst kind.
def test_provider_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVUM_REPORT_MODEL", "meta-llama/llama-3-8b-instruct")
    from core.ground import llm_provider

    assert llm_provider.get_model() == "meta-llama/llama-3-8b-instruct"


def test_provider_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOVUM_REPORT_MODEL", raising=False)
    from core.ground import llm_provider

    assert llm_provider.get_model() == llm_provider.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# window_note template path
# ---------------------------------------------------------------------------


def test_window_note_template_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from core.ground.report_gen import window_note

    gen = window_note(WINDOW_0, offline=True)
    assert isinstance(gen.text, str)
    assert len(gen.text) > 20
    assert gen.usage is None
    assert gen.used_llm is False
    assert gen.skip_reason == "offline_requested"


def test_window_note_template_contains_key_facts() -> None:
    from core.ground.report_gen import window_note_template

    note = window_note_template(WINDOW_0)
    assert "Window 0" in note
    assert "39" in note   # n_arrived
    assert "bits" in note.lower()


def test_window_note_falls_back_when_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from core.ground.report_gen import window_note

    gen = window_note(WINDOW_0, offline=False)
    # Should have fallen back to template, and said why.
    assert isinstance(gen.text, str)
    assert gen.usage is None
    assert gen.used_llm is False
    assert gen.skip_reason == "key_missing"
    assert "OPENROUTER_API_KEY" in gen.reason_line()


# ---------------------------------------------------------------------------
# mission_summary template path
# ---------------------------------------------------------------------------


def test_mission_summary_template_runs() -> None:
    from core.ground.report_gen import mission_summary_template

    text = mission_summary_template([WINDOW_0], RUN_META, FIFO_META, MISSION)
    assert "# Mission Briefing" in text
    assert "score_first" in text
    assert "OFFLINE" in text  # marker present in template output


def test_mission_summary_offline_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from core.ground.report_gen import mission_summary

    gen = mission_summary([WINDOW_0], RUN_META, FIFO_META, MISSION, offline=True)
    assert isinstance(gen.text, str)
    assert gen.usage is None
    assert gen.used_llm is False
    assert gen.skip_reason == "offline_requested"
    assert "offline_requested" in gen.text


def test_mission_summary_falls_back_when_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from core.ground.report_gen import mission_summary

    gen = mission_summary([WINDOW_0], RUN_META, FIFO_META, MISSION, offline=False)
    assert isinstance(gen.text, str)
    assert gen.used_llm is False
    # The report must name the cause, not shrug with "LLM unavailable".
    assert gen.skip_reason == "key_missing"
    assert "key_missing" in gen.text
    assert "OPENROUTER_API_KEY" in gen.text


# ---------------------------------------------------------------------------
# Reporting defects: window enumeration, yield denominator, recall labelling
# ---------------------------------------------------------------------------


def test_full_run_of_windows_collapses_to_all() -> None:
    from core.ground.report_gen import format_window_set

    every = list(range(27))
    assert format_window_set(every, every) == "all 27 windows"


def test_partial_window_sets_use_ranges_not_enumeration() -> None:
    from core.ground.report_gen import format_window_set

    universe = list(range(27))
    text = format_window_set([0, 1, 2, 3, 4, 9, 21, 22, 23, 24, 25, 26], universe)
    assert "0–4" in text
    assert "9" in text
    assert "21–26" in text
    # The whole point: not one index per comma.
    assert "1, 2, 3" not in text


def test_two_window_group_is_not_a_range() -> None:
    from core.ground.report_gen import format_window_set

    assert "3, 4" in format_window_set([3, 4], list(range(10)))


def test_mission_summary_does_not_enumerate_every_window() -> None:
    from core.ground.report_gen import mission_summary_template

    windows = [dict(WINDOW_0, window=i, binding_constraint="bits") for i in range(27)]
    text = mission_summary_template(windows, RUN_META, FIFO_META, MISSION)
    assert "all 27 windows" in text
    assert "0, 1, 2, 3" not in text


def test_early_window_yield_is_flagged_as_provisional() -> None:
    """Window 6 read 77.4% and window 8 read 81.8% on a mission that finished
    at 55.6%, purely because only 31 and 33 of 169 natural frames were in hand.
    Both must carry their counts and be marked provisional."""
    from core.ground.report_gen import window_note_template

    window_6 = dict(
        WINDOW_0, window=6,
        cum_sent_natural=24, cum_natural_available=31, cum_science_yield=24 / 31,
    )
    window_8 = dict(
        WINDOW_0, window=8,
        cum_sent_natural=27, cum_natural_available=33, cum_science_yield=27 / 33,
    )
    late = dict(
        WINDOW_0, window=26,
        cum_sent_natural=94, cum_natural_available=169, cum_science_yield=94 / 169,
    )

    note_6 = window_note_template(window_6, n_natural_total=169)
    assert "24 of 31 natural frames captured so far" in note_6
    assert "77.4%" in note_6
    assert "provisional" in note_6

    note_8 = window_note_template(window_8, n_natural_total=169)
    assert "27 of 33 natural frames captured so far" in note_8
    assert "provisional" in note_8

    # Once the denominator is the whole mission, the ratio stands on its own.
    note_26 = window_note_template(late, n_natural_total=169)
    assert "94 of 169 natural frames captured so far" in note_26
    assert "provisional" not in note_26
    assert "not of the mission total" in note_26


def test_yield_flag_falls_back_to_an_absolute_floor() -> None:
    """Without the mission total, a handful of frames is still a handful."""
    from core.ground.report_gen import window_note_template

    thin = dict(
        WINDOW_0, window=2,
        cum_sent_natural=4, cum_natural_available=13, cum_science_yield=4 / 13,
    )
    note = window_note_template(thin)
    assert "4 of 13 natural frames captured so far" in note
    assert "provisional" in note


def test_per_window_and_mission_recall_are_labelled_apart() -> None:
    from core.ground.report_gen import mission_summary_template, window_note_template

    rec = dict(
        WINDOW_0,
        prefilter_recall=10 / 19,
        prefilter_scored_natural=10,
        prefilter_buffered_natural=19,
    )
    note = window_note_template(rec)
    assert "Prefilter recall this window" in note
    assert "10 of 19 buffered natural frames" in note

    run = dict(RUN_META, prefilter_recall_natural=0.828, n_natural_never_scored=29)
    summary = mission_summary_template([rec], run, FIFO_META, MISSION)
    assert "mission, unique frames" in summary
    assert "82.8%" in summary
    # And the per-window mean must not masquerade as the mission figure.
    assert "Different denominator from the mission figure" in summary


def test_notes_written_back_are_not_fed_to_the_model_next_run() -> None:
    from core.ground.report_gen import _strip_derived

    polluted = dict(WINDOW_0, operator_note="a long note from last time",
                    operator_note_llm=True)
    cleaned = _strip_derived([polluted])[0]
    assert "operator_note" not in cleaned
    assert "operator_note_llm" not in cleaned
    assert cleaned["n_arrived"] == WINDOW_0["n_arrived"]


# ---------------------------------------------------------------------------
# Number validation
# ---------------------------------------------------------------------------


def test_validation_passes_for_source_numbers() -> None:
    from core.ground.report_gen import validate_numbers, window_note_template

    note = window_note_template(WINDOW_0)
    untraced = validate_numbers(note, [WINDOW_0], RUN_META, MISSION)
    # Template only emits numbers from the source; should be minimal
    # We allow up to 5 formatting artefacts (rounding, percentages computed)
    assert len(untraced) < 6, f"too many untraced numbers: {untraced}"


def test_validation_flags_invented_numbers() -> None:
    from core.ground.report_gen import validate_numbers

    fake_text = "The rover sent 9999 frames and achieved 87.654% yield."
    untraced = validate_numbers(fake_text, [WINDOW_0], RUN_META, MISSION)
    # 9999 should not appear in source
    assert "9999" in untraced


def test_validation_reads_thousands_separators_as_one_number() -> None:
    """A live run wrote "103,597,670.33" where the log said 10,359,770.33.

    The old tokenizer split on commas, flagged the fragments "597" and
    "670.33", and never tested the number the model actually claimed -- so an
    order-of-magnitude error was reported as three harmless-looking artefacts.
    """
    from core.ground.report_gen import _extract_numbers, validate_numbers

    assert _extract_numbers("used 103,597,670.33 bits") == {"103597670.33"}

    honest = "Cycle spend was 64,982,400 on scoring."
    assert validate_numbers(honest, [WINDOW_0], RUN_META, MISSION) == []

    inflated = "Cycle spend was 649,824,000 on scoring."
    assert "649824000" in validate_numbers(inflated, [WINDOW_0], RUN_META, MISSION)


def test_validation_accepts_source_values_at_six_decimals() -> None:
    """0.5562130177514792 quoted as "0.556213" is faithful, not invented."""
    from core.ground.report_gen import validate_numbers

    run = dict(RUN_META, science_yield=0.5562130177514792)
    text = "Science yield reached 0.556213 across the mission."
    assert validate_numbers(text, [WINDOW_0], run, MISSION) == []


def test_validation_catches_a_misrounded_percentage() -> None:
    """0.556213 is 55.6%, not 56.6%. The cross-record composite set used to
    trace almost any two-digit number by coincidence and waved this through."""
    from core.ground.report_gen import validate_numbers

    windows = [dict(WINDOW_0, window=i) for i in range(27)]
    run = dict(RUN_META, science_yield=0.5562130177514792)

    assert validate_numbers(
        "Science yield was 55.6% of natural frames.", windows, run, MISSION
    ) == []
    assert "56.6" in validate_numbers(
        "Science yield was 56.6% of natural frames.", windows, run, MISSION
    )


def test_validation_passes_common_small_ints() -> None:
    from core.ground.report_gen import validate_numbers

    text = "1 window processed, 100% budget used in 0 expired frames."
    untraced = validate_numbers(text, [WINDOW_0], RUN_META, MISSION)
    # Small formatting integers must not be flagged
    assert "1" not in untraced
    assert "100" not in untraced
    assert "0" not in untraced


# ---------------------------------------------------------------------------
# Validation, end to end: an inventing model must not produce a clean report
# ---------------------------------------------------------------------------


class _StubResponse:
    """Stands in for a model that returned prose we did not sanction."""

    def __init__(self, text: str) -> None:
        from core.ground.llm_provider import Usage

        self.text = text
        self.usage = Usage(prompt_tokens=10, completion_tokens=5, model="stub")


def _stub_provider(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    from core.ground import report_gen

    monkeypatch.setattr(
        report_gen, "complete", lambda *a, **k: _StubResponse(text), raising=True
    )


def test_model_written_figures_are_discarded_not_published(
    run_dir_with_jsonl: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A figure the model wrote itself never reaches the document at all.

    This is the constraint doing its job: the generation is thrown away and the
    deterministic template is published, so there is nothing left to validate.
    """
    from scripts.mission_brief import generate_brief

    run_dir, jsonl_path = run_dir_with_jsonl
    _stub_provider(
        monkeypatch,
        "## Science Yield\n\nThe rover transmitted 9999 frames and reached "
        "87.654% science yield, up 4321 frames on the baseline.\n",
    )

    doc, meta = generate_brief(run_dir, jsonl_path, offline=False, write_back=False)

    assert meta["mode"] == "offline"
    assert meta["skip_reason"] == "unsanctioned_figures"
    assert "9999" not in doc
    assert "87.654" not in doc


_COMPLIANT_SUMMARY = """\
## Science Yield
- {{SCIENCE_YIELD}}
- {{FIFO_SCIENCE_YIELD}}
- {{N_EXPIRED_TOTAL}}

Selection quality, not spare bandwidth, produced the gap.
"""


def test_placeholders_render_to_their_labelled_facts(
    run_dir_with_jsonl: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Figure lines are accepted, and each token brings its own label."""
    from scripts.mission_brief import generate_brief

    run_dir, jsonl_path = run_dir_with_jsonl
    _stub_provider(monkeypatch, _COMPLIANT_SUMMARY)

    doc, meta = generate_brief(run_dir, jsonl_path, offline=False, write_back=False)

    assert meta["mode"] == "llm"
    assert "{{" not in doc, "a placeholder survived into the published document"
    assert "natural frames delivered" in doc
    assert "Selection quality, not spare bandwidth" in doc
    assert meta["validation_untraced"] == []


def test_the_mislabelling_bug_is_now_unrepresentable(
    run_dir_with_jsonl: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original defect: "604 natural frames expired" when 604 is the total.

    Two independent defences. Attaching a noun to the token is a mixed line and
    the generation is thrown out; and had it survived, the fact's own wording
    names the denominator anyway.
    """
    from core.ground.facts import build_mission_facts, misplaced_placeholders
    from scripts.mission_brief import generate_brief

    assert misplaced_placeholders("{{N_EXPIRED_TOTAL}} natural frames expired.")

    run_dir, jsonl_path = run_dir_with_jsonl
    _stub_provider(monkeypatch, "{{N_EXPIRED_TOTAL}} natural frames expired.\n")
    _doc, meta = generate_brief(run_dir, jsonl_path, offline=False, write_back=False)
    assert meta["skip_reason"] == "unsanctioned_figures"

    sheet = build_mission_facts([WINDOW_0], RUN_META, FIFO_META, MISSION)
    phrase = sheet.by_key()["N_EXPIRED_TOTAL"].render()
    assert "all classes" in phrase and "not only natural" in phrase


def test_unknown_placeholders_are_dropped(
    run_dir_with_jsonl: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A citation of a figure the fact layer never issued resolves to nothing."""
    from scripts.mission_brief import generate_brief

    run_dir, jsonl_path = run_dir_with_jsonl
    _stub_provider(monkeypatch, "- {{TOTALLY_MADE_UP_METRIC}}\n\nNo basis for that.\n")

    doc, meta = generate_brief(run_dir, jsonl_path, offline=False, write_back=False)
    assert "TOTALLY_MADE_UP_METRIC" not in doc
    assert "TOTALLY_MADE_UP_METRIC" in meta["unknown_placeholders"]


def test_catalogue_shows_no_values(
    run_dir_with_jsonl: tuple[Path, Path],
) -> None:
    """The prompt must contain no digits, or the model copies them."""
    from core.ground.facts import build_mission_facts, build_window_facts

    for sheet in (
        build_mission_facts([WINDOW_0], RUN_META, FIFO_META, MISSION),
        build_window_facts(WINDOW_0, 169),
    ):
        catalogue = sheet.catalogue()
        assert not any(ch.isdigit() for ch in catalogue), catalogue


def test_corrupted_record_still_flags_via_the_backstop() -> None:
    """The numeric validator remains, for prose that did not come from facts.

    Feed a deliberately corrupted record: a figure that traced to the honest
    log must stop tracing.
    """
    from core.ground.report_gen import validate_numbers

    text = "Cycle spend was 64,982,400 on scoring across the window."
    assert validate_numbers(text, [WINDOW_0], RUN_META, MISSION) == []

    corrupted = dict(WINDOW_0, cycles_scoring=11111111.0)
    assert "64982400" in validate_numbers(text, [corrupted], RUN_META, MISSION)


# ---------------------------------------------------------------------------
# CLI acceptance
# ---------------------------------------------------------------------------


def _run_cli(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)  # default: no key
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "scripts.mission_brief"] + list(args),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )


@pytest.fixture()
def run_dir_with_jsonl(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal sim run dir with a summary.json and one .jsonl file."""
    run_dir = tmp_path / "sim" / "test-run"
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True)

    # Write summary.json
    summary = {
        "created_utc": "2026-01-01T00:00:00Z",
        "git_commit": "abc123",
        "config": {},
        "mission": MISSION,
        "runs": [
            dict(RUN_META, config={}, windows=1),
            dict(FIFO_META, n_sent=5, n_sent_natural=1, n_sent_rover=1,
                 n_sent_typical=3, n_natural_total=169, bits_used=388000.0,
                 bits_available=395871.92, n_expired=0, n_refits=0,
                 windows=1, config={}, artifact="", tier="rad750",
                 adaptation="frozen"),
        ],
        "experiments": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # Write windows JSONL
    jsonl_path = windows_dir / "rad750-score_first.jsonl"
    jsonl_path.write_text(json.dumps(WINDOW_0) + "\n", encoding="utf-8")

    return run_dir, jsonl_path


def test_cli_offline_produces_report_exits_0(
    run_dir_with_jsonl: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, jsonl_path = run_dir_with_jsonl
    out_path = tmp_path / "MISSION_BRIEF.md"

    # Patch NOVUM_RUNS_DIR to point at our tmp sim structure
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.mission_brief",
            "--run-id", "test-run",
            "--offline",
            "--out", str(out_path),
            "--no-write-back",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            **os.environ,
            "OPENROUTER_API_KEY": "",
            "NOVUM_RUNS_DIR": str(run_dir.parent.parent),
        },
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    content = out_path.read_text()
    assert "Mission Briefing" in content or "Window 0" in content


def test_cli_no_key_falls_back_offline(
    run_dir_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, _ = run_dir_with_jsonl
    out_path = tmp_path / "BRIEF.md"

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.mission_brief",
            "--run-id", "test-run",
            "--out", str(out_path),
            "--no-write-back",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            **{k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"},
            "NOVUM_RUNS_DIR": str(run_dir.parent.parent),
        },
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    content = out_path.read_text()
    assert len(content) > 100


def test_cli_write_back_adds_operator_note(
    run_dir_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, jsonl_path = run_dir_with_jsonl
    out_path = tmp_path / "BRIEF.md"

    subprocess.run(
        [
            sys.executable, "-m", "scripts.mission_brief",
            "--run-id", "test-run",
            "--offline",
            "--out", str(out_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            **os.environ,
            "OPENROUTER_API_KEY": "",
            "NOVUM_RUNS_DIR": str(run_dir.parent.parent),
        },
    )
    # The jsonl should now have operator_note field
    lines = [json.loads(ln) for ln in jsonl_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "operator_note" in lines[0]
    assert len(lines[0]["operator_note"]) > 10


def test_cli_report_meta_written(
    run_dir_with_jsonl: tuple[Path, Path], tmp_path: Path
) -> None:
    run_dir, _ = run_dir_with_jsonl
    out_path = tmp_path / "BRIEF.md"

    subprocess.run(
        [
            sys.executable, "-m", "scripts.mission_brief",
            "--run-id", "test-run",
            "--offline",
            "--out", str(out_path),
            "--no-write-back",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={
            **os.environ,
            "OPENROUTER_API_KEY": "",
            "NOVUM_RUNS_DIR": str(run_dir.parent.parent),
        },
    )
    meta_path = run_dir / "report_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "usage" in meta
    assert "generated_utc" in meta
    assert meta["mode"] == "offline"
