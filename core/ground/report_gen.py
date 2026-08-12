"""Per-window operator notes and mission summary.

Two levels of output, both with LLM + deterministic-template fallback:

  window_note(record)          -> short operator note for one window
  mission_summary(windows, run_meta, fifo_meta)  -> mission briefing

THE MODEL NEVER WRITES A FIGURE. `core.ground.facts` turns the decision log
into labelled Facts; the model receives them as `{{PLACEHOLDER}}` tokens and may
only arrange prose around them; the values are substituted back in afterwards.
A reply containing any digit of the model's own is discarded and the
deterministic template is published instead. That removes both hallucinated
numbers and correct-number-wrong-label sentences by construction, rather than
trying to detect them after the fact.

`validate_numbers` remains at the bottom as a backstop for the prose that does
survive -- it catches a figure with no basis in the source, which is what the
placeholder scheme already prevents, and costs nothing to keep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .facts import (
    FactSheet,
    build_mission_facts,
    build_window_facts,
    misplaced_placeholders,
    unsanctioned_numerals,
)
from .llm_provider import (
    DETAIL_NOT_FOR_PUBLICATION,
    REASON_HELP,
    REASON_OFFLINE_REQUESTED,
    REASON_UNSANCTIONED_FIGURES,
    ProviderError,
    Usage,
    complete,
)

#: The only user message either generation sends: a menu of placeholders.
_FACTS_USER_TMPL = """\
FIGURES you may cite, by placeholder:
{catalogue}

Write it now, citing figures only as {{{{PLACEHOLDER}}}} tokens.
"""

#: Cumulative science yield is over natural frames captured SO FAR, so its
#: denominator grows all mission. While it is small the ratio says almost
#: nothing: window 6 read 77.4% and window 8 read 81.8% on a mission that
#: finished at 55.6%, purely because only 31 and 33 natural frames had been
#: captured by then. Counts are always printed; the ratio is marked provisional
#: until at least this fraction of the mission's natural frames are in hand.
YIELD_DENOMINATOR_FRACTION = 0.5

#: Fallback when the mission total is not available to the caller. Below this
#: many frames a single frame moves the ratio by more than five points.
MIN_YIELD_DENOMINATOR = 20


def _yield_is_provisional(available: int, n_natural_total: int | None) -> bool:
    if n_natural_total:
        return available < YIELD_DENOMINATOR_FRACTION * n_natural_total
    return available < MIN_YIELD_DENOMINATOR


#: Keys the brief writes back into the JSONL. They must never be fed to the
#: model on a later run -- a note written last time is not evidence this time,
#: and it grows the prompt every regeneration.
_DERIVED_KEYS = ("operator_note", "operator_note_llm")


@dataclass(frozen=True)
class Generation:
    """One piece of generated prose, and an honest account of where it came from.

    `skip_reason` is None exactly when `used_llm` is True. Otherwise it is one
    of the llm_provider REASON_* codes, so the report can say *why* it fell
    back rather than shrugging with "LLM unavailable".
    """

    text: str
    usage: Usage | None = None
    used_llm: bool = False
    skip_reason: str | None = None
    skip_detail: str | None = None
    #: Placeholders the model invented that the fact layer never issued. They
    #: are dropped from the output; a non-empty tuple means the model tried to
    #: cite a figure that does not exist.
    unknown_placeholders: tuple[str, ...] = ()

    def reason_line(self) -> str:
        """One actionable line naming the cause, or '' when the LLM was used."""
        if self.used_llm or not self.skip_reason:
            return ""
        help_text = REASON_HELP.get(self.skip_reason, "the provider was not used")
        if self.skip_detail and self.skip_reason not in DETAIL_NOT_FOR_PUBLICATION:
            return f"{self.skip_reason}: {help_text} ({self.skip_detail})"
        return f"{self.skip_reason}: {help_text}"


def _strip_derived(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records as the simulator wrote them, minus anything the brief added."""
    return [{k: v for k, v in r.items() if k not in _DERIVED_KEYS} for r in records]


def _prepare_for_prompt(
    records: list[dict[str, Any]], n_natural_total: int | None = None
) -> list[dict[str, Any]]:
    """What the model sees: no stale notes, and thin denominators marked.

    A rule in the system prompt ("do not report an early cumulative yield as a
    peak") is something a small model has to remember. A field in the record it
    is reading is something it can simply quote. Give it the fact.
    """
    out: list[dict[str, Any]] = []
    for rec in _strip_derived(records):
        available = rec.get("cum_natural_available")
        if available is not None:
            rec["cum_yield_provisional"] = _yield_is_provisional(
                available, n_natural_total
            )
            rec["cum_yield_note"] = (
                f"ratio is over the {available} natural frames captured so far, "
                f"NOT the mission total of {n_natural_total or 'unknown'}"
            )
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
#: The one rule that makes mislabelling impossible: the model never writes a
#: figure, so it can never attach the wrong noun to one.
_NO_NUMERALS_RULE = """\
YOU MAY NOT WRITE NUMBERS. Not digits, not spelled-out quantities, not dates,
not percentages, not window indices. Every figure you need already exists as a
placeholder in the FIGURES list below.

You are not shown the values, on purpose. Each placeholder expands after you
reply into a complete phrase that states both the number and what it measures
-- {{EXPIRED_TOTAL}} becomes "604 frames expired unsent across the mission (all
classes)". So a placeholder is a whole statement, not a noun to build a
sentence around, and you cannot know a figure well enough to restate it.

WRITE EVERY LINE AS EXACTLY ONE OF THESE TWO KINDS:

  1. A FIGURE LINE: a bullet, one placeholder, nothing else.
         - {{EXPIRED_TOTAL}}
     Never introduce it, never label it, never add a clause to it.

  2. A PROSE LINE: your own interpretation, containing NO placeholder and NO
     digit. This is where you explain what the figures mean.
         Buffer pressure, not bandwidth, was the dominant loss channel.

Rejected, and why:
  BAD:  Expiry was heavy: {{EXPIRED_TOTAL}}.    <- figure line with prose on it
  BAD:  - {{EXPIRED_TOTAL}} natural frames      <- relabels the figure
  BAD:  604 frames expired.                     <- writes a figure
  BAD:  Roughly six hundred frames expired.     <- writes a figure

Every section looks exactly like this -- figure lines, then one prose
paragraph, and the paragraph never mentions a figure token:

  ## Downlink Efficiency
  - {{BITS_UTILISATION}}
  - {{WASTED_BIT_SHARE}}

  Nearly the whole budget was spent, so the remaining gain has to come from
  choosing better frames rather than from finding spare bandwidth.

If a figure you want is not in the list, make the point qualitatively or leave
it out. A reply that breaks these rules is discarded and the deterministic
template is published in its place.
"""

_WINDOW_SYSTEM = """\
You are an uplink/downlink operations engineer writing a concise operator note
for one Mars rover downlink window.

Output exactly this shape: the figure lines that matter for this window, then
one short prose sentence saying what an operator should take from them.

  - {{ARRIVED}}
  - {{SELECTED}}
  - {{BINDING_CONSTRAINT}}
  - {{CUM_YIELD}}

  Downlink, not triage, is what held this window back.

Pick four to seven figure lines from the list -- always the binding constraint
and the cumulative yield, plus whatever is unusual this window (heavy expiry,
eviction, a refit, a prefilter-recall drop). Keep the closing sentence under
twenty-five words. No headings: a window note is bullets and one sentence.

""" + _NO_NUMERALS_RULE

_WINDOW_USER_TMPL = """\
FIGURES you may cite, by placeholder:
{catalogue}

Write the operator note now, citing figures only as {{PLACEHOLDER}} tokens.
"""

_SUMMARY_SYSTEM = """\
You are an uplink/downlink operations engineer writing a mission briefing for
a Mars rover science-data triage run.  You are given a list of FIGURES, each a
placeholder token bound to a measured quantity and its label.

Write a structured briefing with these sections:
  ## Science Yield
  ## Downlink Efficiency
  ## Compute vs Bandwidth
  ## Adaptation (if applicable)
  ## Recommendations

Rules:
  - Total length: 300-500 words.
  - Do not compare two figures unless the FIGURES list says they share a
    denominator. Where a placeholder's phrase names its denominator, that
    denominator is part of the claim -- do not paraphrase it away.
  - ## Recommendations is PROSE ONLY. No figure lines, no placeholders, no
    bullets containing a token. An action is something to do, not a number;
    the evidence for it is already listed in the section above it.

""" + _NO_NUMERALS_RULE

_SUMMARY_USER_TMPL = """\
FIGURES you may cite, by placeholder:
{catalogue}

Write the briefing now, citing figures only as {{PLACEHOLDER}} tokens.
"""

# ---------------------------------------------------------------------------
# Template (offline) implementations
# ---------------------------------------------------------------------------


def _pct(num: float, denom: float) -> str:
    if denom == 0:
        return "0%"
    return f"{num / denom * 100:.1f}%"


def _kb(bits: float) -> str:
    return f"{bits / 8192:.1f} KiB"


def format_window_set(indices: list[int], universe: list[int], *, max_groups: int = 8) -> str:
    """Describe a set of window indices without enumerating all of them.

    "all 27 windows" when it is every window, "0–4, 9, 21–26" when it is not.
    A comma-separated wall of every index is unreadable on the page and pure
    noise in a prompt, and it hides the one thing worth seeing: whether the
    constraint bound *everywhere* or only in patches.
    """
    picked = sorted(set(indices))
    if not picked:
        return "none"
    total = len(set(universe)) or len(picked)
    if len(picked) == total:
        return f"all {total} windows"

    groups: list[tuple[int, int]] = []
    start = prev = picked[0]
    for i in picked[1:]:
        if i == prev + 1:
            prev = i
            continue
        groups.append((start, prev))
        start = prev = i
    groups.append((start, prev))

    def render(lo: int, hi: int) -> str:
        if lo == hi:
            return str(lo)
        if hi == lo + 1:
            return f"{lo}, {hi}"
        return f"{lo}–{hi}"

    shown = [render(lo, hi) for lo, hi in groups[:max_groups]]
    text = ", ".join(shown)
    if len(groups) > max_groups:
        text += f", … (+{len(groups) - max_groups} more runs)"
    return f"{len(picked)} of {total} windows: {text}"


def _cumulative_yield_phrase(
    rec: dict[str, Any], n_natural_total: int | None = None
) -> str:
    """Cumulative science yield with its denominator visible.

    The per-window ratio is over natural frames CAPTURED SO FAR, not over the
    mission total, so window 6 can read 77.4% on a mission that ends at 55.6%.
    Printing the counts makes the two impossible to confuse; flagging a thin
    denominator stops the ratio being quoted as an achievement.
    """
    sent = rec.get("cum_sent_natural")
    available = rec.get("cum_natural_available")
    ratio = rec.get("cum_science_yield")

    if sent is None or available is None:
        # Pre-fix logs carry only the ratio; say what it is and no more.
        return f"Cumulative science yield (of natural frames captured so far): {ratio * 100:.1f}%."
    if not available:
        return "Cumulative science yield: no natural frames captured yet."

    pct = f"{ratio * 100:.1f}%" if ratio is not None else _pct(sent, available)
    stem = f"Cumulative science yield: {sent} of {available} natural frames captured so far"
    if _yield_is_provisional(available, n_natural_total):
        of_total = f" of the mission's {n_natural_total}" if n_natural_total else ""
        return (
            f"{stem} ({pct} — provisional: only {available}{of_total} natural "
            f"frames captured so far, so this ratio is not comparable to the "
            f"mission figure)."
        )
    return f"{stem} ({pct} of what has been captured, not of the mission total)."


def window_note_template(
    rec: dict[str, Any], n_natural_total: int | None = None
) -> str:
    """Deterministic offline note for one window record."""
    w = rec["window"]
    arrived = rec["n_arrived"]
    sent = rec["n_selected"]
    budget_bits = rec["bits_budget"]
    used_bits = rec["bits_used"]
    budget_cyc = rec["cycles_budget"]
    used_cyc = rec["cycles_used"]
    constraint = rec["binding_constraint"]
    expired = rec["n_expired"]
    evicted = rec.get("n_evicted", 0)
    scored = rec["n_scored"]
    unscored = rec["n_unscored"]
    buffered = rec.get("n_buffered") or 0
    refit = rec.get("refit", False)
    prefilter = rec.get("prefilter_recall")
    sent_natural = rec.get("sent_natural", 0)
    sent_rover = rec.get("sent_rover", 0)
    sent_typical = rec.get("sent_typical", 0)

    parts: list[str] = [
        f"Window {w}: {arrived} frames arrived, {sent} transmitted "
        f"({_pct(sent, buffered)} of buffer). "
        if buffered > arrived
        else f"Window {w}: {arrived} frames arrived, {sent} transmitted. "
    ]
    parts.append(
        f"Binding constraint: {constraint} "
        f"({_pct(used_bits, budget_bits)} of bits budget, "
        f"{_pct(used_cyc, budget_cyc)} of cycles budget used). "
    )
    parts.append(
        f"Sent {sent_natural} natural / {sent_rover} rover / {sent_typical} typical. "
    )
    parts.append(f"Scored {scored} frames, {unscored} unscored. ")
    if prefilter is not None:
        # NOT the mission-level prefilter recall: this is one window's buffer
        # snapshot, that one is unique frames across the whole mission. Naming
        # the denominator is the only thing that keeps them apart.
        scored_nat = rec.get("prefilter_scored_natural")
        buffered_nat = rec.get("prefilter_buffered_natural")
        if scored_nat is not None and buffered_nat:
            parts.append(
                f"Prefilter recall this window: {scored_nat} of {buffered_nat} "
                f"buffered natural frames had a real score ({prefilter * 100:.1f}%). "
            )
        else:
            parts.append(
                f"Prefilter recall this window (buffered natural frames with a "
                f"real score): {prefilter * 100:.1f}%. "
            )
    if expired:
        parts.append(f"Expired this window: {expired}. ")
    if evicted:
        parts.append(f"Evicted: {evicted}. ")
    if refit:
        parts.append("Model refit performed this window. ")
    parts.append(_cumulative_yield_phrase(rec, n_natural_total))
    return "".join(parts)


def mission_summary_template(
    windows: list[dict[str, Any]],
    run: dict[str, Any],
    fifo: dict[str, Any] | None,
    mission: dict[str, Any],
    *,
    skip_reason: str | None = None,
    skip_detail: str | None = None,
) -> str:
    """Deterministic offline mission briefing."""
    method = run.get("method", "unknown")
    tier = run.get("tier", "unknown")
    adaptation = run.get("adaptation", "frozen")
    n_windows = run.get("windows", len(windows))
    science_yield = run.get("science_yield", 0.0)
    wasted = run.get("wasted_bit_share", 0.0)
    n_sent = run.get("n_sent", 0)
    n_natural = run.get("n_sent_natural", 0)
    n_natural_total = run.get("n_natural_total", 0)
    bits_used = run.get("bits_used", 0.0)
    bits_avail = run.get("bits_available", 0.0)
    n_expired = run.get("n_expired", 0)
    n_refits = run.get("n_refits", 0)

    fifo_yield = (fifo or {}).get("science_yield", None)
    fifo_wasted = (fifo or {}).get("wasted_bit_share", None)

    # Find binding-constraint windows
    bits_windows = [w["window"] for w in windows if w.get("binding_constraint") == "bits"]
    cycles_windows = [w["window"] for w in windows if w.get("binding_constraint") == "cycles"]

    # Cold-start: first window with cumulative yield > 0
    warm_window: int | None = None
    for w in windows:
        if (w.get("cum_science_yield") or 0.0) > 0:
            warm_window = w["window"]
            break

    # High-expiry windows (top 3)
    sorted_expiry = sorted(windows, key=lambda w: w.get("n_expired", 0), reverse=True)
    top_expiry = sorted_expiry[:3]

    lines: list[str] = [
        f"# Mission Briefing — {method} on {tier}",
        f"*Run: {n_windows} windows, sols "
        f"{mission.get('sol_min', '?')}–{mission.get('sol_max', '?')}*",
        "",
        "## Science Yield",
        "",
        f"Science yield (fraction of natural-scene frames transmitted): "
        f"**{science_yield * 100:.1f}%** "
        f"({n_natural} of {n_natural_total} natural frames sent).",
    ]
    if fifo_yield is not None:
        delta = science_yield - fifo_yield
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"FIFO baseline yield: {fifo_yield * 100:.1f}%. "
            f"Delta vs FIFO: {sign}{delta * 100:.1f} pp."
        )
    lines.append(
        "Note the two denominators: this headline is over all "
        f"{n_natural_total} natural frames in the mission, while the "
        "per-window `cum_science_yield` below is over natural frames captured "
        "*so far*. Early windows therefore read high on a handful of frames "
        "and are not comparable to this figure."
    )
    lines += [
        "",
        "## Downlink Efficiency",
        "",
        f"Total bits used: {_kb(bits_used)} of {_kb(bits_avail)} available "
        f"({_pct(bits_used, bits_avail)} utilisation). "
        f"Wasted bit share: {wasted * 100:.1f}%.",
    ]
    if fifo_wasted is not None:
        lines.append(
            f"FIFO wasted bit share: {fifo_wasted * 100:.1f}%."
        )
    lines.append(f"Total frames transmitted: {n_sent}. Expired frames: {n_expired}.")

    lines += [
        "",
        "## Compute vs Bandwidth",
        "",
    ]
    all_windows = [w["window"] for w in windows]
    if bits_windows:
        lines.append(
            f"Bandwidth-limited: {format_window_set(bits_windows, all_windows)}."
        )
    if cycles_windows:
        lines.append(
            f"Compute-limited: {format_window_set(cycles_windows, all_windows)}."
        )
    if not bits_windows and not cycles_windows:
        lines.append("No binding-constraint data recorded.")

    # Two prefilter-recall figures exist and they are NOT the same measurement.
    # Mission recall is unique natural frames that ever earned a score, over
    # unique natural frames ever buffered. The per-window figure is one window's
    # buffer snapshot, and averaging it counts a long-buffered frame once per
    # window it survived. Report both, named, or a reader will read one as the
    # other and conclude the numbers disagree.
    mission_recall = run.get("prefilter_recall_natural")
    if mission_recall is not None:
        never_scored = run.get("n_natural_never_scored")
        tail = f" ({never_scored} natural frames never scored at all)" if never_scored else ""
        lines.append(
            f"Prefilter recall — mission, unique frames: {mission_recall * 100:.1f}%"
            f"{tail}. This is the ceiling on science yield that no amount of "
            "downlink can lift."
        )
    per_window = [
        w["prefilter_recall"] for w in windows if w.get("prefilter_recall") is not None
    ]
    if per_window:
        mean_recall = sum(per_window) / len(per_window)
        lines.append(
            f"Prefilter recall — mean over the {len(per_window)} window(s) where "
            f"triage actually bound: {mean_recall * 100:.1f}%. Different "
            "denominator from the mission figure above (a buffer snapshot per "
            "window, not unique frames); the two are not interchangeable."
        )

    if top_expiry and top_expiry[0].get("n_expired", 0) > 0:
        lines.append(
            "High-expiry windows: "
            + ", ".join(
                f"window {w['window']} ({w.get('n_expired', 0)} expired)"
                for w in top_expiry
                if w.get("n_expired", 0) > 0
            )
            + "."
        )

    lines += [
        "",
        "## Adaptation",
        "",
    ]
    if adaptation == "frozen":
        lines.append("Adaptation was frozen (no in-mission refits).")
    else:
        lines.append(f"Adaptation mode: {adaptation}. Total refits: {n_refits}.")
        if warm_window is not None:
            lines.append(
                f"First window with non-zero cumulative yield: window {warm_window} "
                f"(cold-start ends here or earlier)."
            )

    lines += [
        "",
        "## Recommendations",
        "",
    ]
    recs: list[str] = []
    if wasted > 0.10:
        recs.append(
            f"Wasted bit share of {wasted * 100:.1f}% is above 10%: review buffer eviction "
            "policy or increase downlink fraction."
        )
    if cycles_windows:
        recs.append(
            f"{len(cycles_windows)} window(s) were compute-limited: consider a lighter "
            "prefilter or reduced scoring budget."
        )
    if n_expired > 50:
        recs.append(
            f"{n_expired} frames expired across the mission: buffer age or capacity "
            "settings may need tightening."
        )
    if not recs:
        recs.append("No significant operational issues detected in this run.")
    for r in recs:
        lines.append(f"- {r}")

    lines.append("")
    why = REASON_HELP.get(skip_reason or "", "")
    if skip_reason and why:
        detail = (
            f" Detail: {skip_detail}"
            if skip_detail and skip_reason not in DETAIL_NOT_FOR_PUBLICATION
            else ""
        )
        lines.append(
            f"_⚠ OFFLINE REPORT — `{skip_reason}`: {why}.{detail} "
            "Figures are template-rendered directly from the decision log._"
        )
    else:
        lines.append(
            "_⚠ OFFLINE REPORT — the language model was not used; "
            "figures are template-rendered directly from the decision log._"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Number validation
# ---------------------------------------------------------------------------
# Thousands separators are part of the number: without the comma group a model
# writing "103,597,670.33" is torn into "103", "597" and "670.33", and the last
# two get flagged as inventions even though the figure is exact.
#
# The lookbehind rejects \w, not just digits, so the "750" inside `rad750` is
# a tier name rather than a claim about the mission.
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")

#: Decimal places a model might round a source value to. The old set stopped at
#: four, so a faithfully quoted 0.5562130177514792 written as "0.556213" was
#: reported as untraceable.
_PRECISIONS = (0, 1, 2, 3, 4, 5, 6)


def _extract_numbers(text: str) -> set[str]:
    """Extract all decimal and integer tokens from a string, commas removed."""
    return {tok.replace(",", "") for tok in _NUMBER_RE.findall(text)}


def _forms(v: float) -> set[str]:
    """Every string a faithful writer might use for one source value."""
    out = {str(int(v)) if v == int(v) else str(v)}
    for p in _PRECISIONS:
        out.add(f"{v:.{p}f}")
        out.add(f"{v * 100:.{p}f}")      # percentage form
    if abs(v) > 1000:
        out.add(f"{v / 8192:.1f}")       # KiB form
        out.add(f"{v / 8192:.2f}")
    return out


def _source_numbers(
    windows: list[dict],
    run: dict,
    mission: dict,
    fifo: dict | None = None,
) -> set[str]:
    """Collect every numeric value from the source records.

    Also derives the same arithmetic forms that the template emits so that
    computed values (percentages, deltas, KiB conversions) are traced back
    to their inputs and don't trigger false-positive validation flags.

    Composites are derived WITHIN a record, not across the whole mission. The
    cross product of every scalar in every window is ~450k pairs, and a set
    that large traces almost any two- or three-digit number by coincidence --
    which is how a summary claiming "56.6%" for a yield of 0.556 passed
    unflagged. Grouping by record keeps the arithmetic the template actually
    performs and drops the accidental matches.
    """
    nums: set[str] = set()
    groups: list[list[float]] = []

    def _collect(d: dict, into: list[float]) -> None:
        for v in d.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                into.append(float(v))
                nums.update(_forms(float(v)))
            elif isinstance(v, dict):
                _collect(v, into)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _collect(item, into)

    for w in windows:
        scalars: list[float] = []
        _collect(w, scalars)
        groups.append(scalars)

    # The aggregates share a group: the template compares run against fifo and
    # against the mission composition, so those pairings are legitimate.
    aggregate: list[float] = []
    _collect(run, aggregate)
    _collect(mission, aggregate)
    if fifo:
        _collect(fifo, aggregate)
    groups.append(aggregate)

    # Two-operand composites the template produces -- ratio as a percentage,
    # and difference -- derived only between scalars of the same record.
    for scalars in groups:
        for i, a in enumerate(scalars):
            for b in scalars[i + 1 :]:
                if b:
                    nums.add(f"{a / b * 100:.1f}")
                    nums.add(f"{a / b * 100:.0f}")
                if a:
                    nums.add(f"{b / a * 100:.1f}")
                    nums.add(f"{b / a * 100:.0f}")
                diff = abs(a - b)
                nums.add(f"{diff * 100:.1f}")
                nums.add(f"{diff:.1f}")

    return nums


def validate_numbers(
    text: str,
    windows: list[dict],
    run: dict,
    mission: dict,
    fifo: dict | None = None,
) -> list[str]:
    """Return list of numbers appearing in text that cannot be traced to the source.

    Accepts an optional fifo dict so FIFO-derived figures in the template
    are also validated.  A few small integers and common formatting constants
    are not flagged; the caller decides what to do with the result.
    """
    # Numbers too small or common to be meaningful science figures
    ALWAYS_OK = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100",
                 "0.0", "0.00", "0.000", "1.0", "100.0"}
    source = _source_numbers(windows, run, mission, fifo)
    found = _extract_numbers(text)
    untraced = [n for n in sorted(found) if n not in source and n not in ALWAYS_OK]
    return untraced


# ---------------------------------------------------------------------------
# LLM-driven implementations with fallback
# ---------------------------------------------------------------------------


def _generate_from_facts(
    system: str,
    sheet: FactSheet,
    *,
    max_tokens: int,
    attempts: int = 3,
) -> tuple[str, Usage, list[str]]:
    """Ask for prose made of placeholders, and refuse prose that is not.

    Returns (rendered_text, usage, unknown_placeholder_keys). Raises
    ProviderError -- including a synthetic one when the model keeps writing
    figures of its own, because a generation we cannot trust is worth exactly
    as much as no generation at all.
    """
    user = _FACTS_USER_TMPL.format(catalogue=sheet.catalogue())
    totals = Usage()
    last_offenders: list[str] = []

    for attempt in range(attempts):
        prompt = user
        if attempt:
            prompt = (
                f"{user}\n\nYour previous reply contained figures you wrote "
                f"yourself: {', '.join(last_offenders[:8])}. Replace every one "
                "with the matching {{PLACEHOLDER}} token, or drop the claim. "
                "The reply must contain no digits at all."
            )
        resp = complete(system, prompt, max_tokens=max_tokens, temperature=0.2)
        totals.prompt_tokens += resp.usage.prompt_tokens
        totals.completion_tokens += resp.usage.completion_tokens
        totals.cost_usd += resp.usage.cost_usd
        totals.model = resp.usage.model

        text = resp.text.strip()
        last_offenders = unsanctioned_numerals(text) + [
            f"line mixes prose with a figure: {line[:70]}"
            for line in misplaced_placeholders(text)
        ]
        if last_offenders:
            continue

        rendered, unknown = sheet.render(text)
        return rendered.strip(), totals, unknown

    raise ProviderError(
        "model wrote figures instead of placeholders after "
        f"{attempts} attempts: {'; '.join(last_offenders[:6])}",
        reason=REASON_UNSANCTIONED_FIGURES,
    )


def window_note(
    rec: dict[str, Any],
    *,
    offline: bool = False,
    n_natural_total: int | None = None,
) -> Generation:
    """Generate a per-window operator note.

    When offline=True or the LLM call fails, returns the template note plus the
    reason the provider was skipped.
    """
    if offline:
        return Generation(
            text=window_note_template(rec, n_natural_total),
            skip_reason=REASON_OFFLINE_REQUESTED,
        )

    try:
        text, usage, unknown = _generate_from_facts(
            _WINDOW_SYSTEM,
            build_window_facts(rec, n_natural_total),
            max_tokens=384,
        )
        return Generation(
            text=text, usage=usage, used_llm=True, unknown_placeholders=tuple(unknown)
        )
    except ProviderError as exc:
        return Generation(
            text=window_note_template(rec, n_natural_total),
            skip_reason=exc.reason,
            skip_detail=exc.detail,
        )


def mission_summary(
    windows: list[dict[str, Any]],
    run: dict[str, Any],
    fifo: dict[str, Any] | None,
    mission: dict[str, Any],
    *,
    offline: bool = False,
) -> Generation:
    """Generate the mission briefing.

    When offline=True or the LLM call fails, returns the template briefing plus
    the reason the provider was skipped.
    """
    if offline:
        return Generation(
            text=mission_summary_template(
                windows, run, fifo, mission, skip_reason=REASON_OFFLINE_REQUESTED
            ),
            skip_reason=REASON_OFFLINE_REQUESTED,
        )

    try:
        text, usage, unknown = _generate_from_facts(
            _SUMMARY_SYSTEM,
            build_mission_facts(windows, run, fifo, mission),
            max_tokens=1400,
        )
        return Generation(
            text=text, usage=usage, used_llm=True, unknown_placeholders=tuple(unknown)
        )
    except ProviderError as exc:
        return Generation(
            text=mission_summary_template(
                windows, run, fifo, mission,
                skip_reason=exc.reason,
                skip_detail=exc.detail,
            ),
            skip_reason=exc.reason,
            skip_detail=exc.detail,
        )
