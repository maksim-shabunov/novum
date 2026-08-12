"""Facts the model may cite, and the placeholders it must cite them by.

THE PROBLEM THIS SOLVES. The numeric validator catches a number that is not in
the source. It cannot catch a number that IS in the source wearing the wrong
label -- "604 natural frames expired" when 604 is the total across all classes.
Both figures are real; only the sentence is false. No amount of checking the
output fixes that, because the output is where the damage already happened.

So the model is not allowed to write figures at all. The deterministic layer
builds a Fact for every quantity worth stating; each Fact knows its own label
and renders as a complete, self-describing noun phrase:

    Fact("EXPIRED_TOTAL", value=604, ...).render()
        -> "604 frames expired unsent across the mission (all classes)"

The model receives `{{EXPIRED_TOTAL}}` and writes prose around it. It cannot
attach "natural" to that number, because the words "all classes" travel with
the number and are substituted in after generation. A figure the fact layer did
not produce has no placeholder, so it cannot be cited; a bare numeral in the
output means the model went off-script and the generation is rejected.

This is a constraint on generation, not a check on it. The numeric validator
stays as a backstop for the prose the model does write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: `{{KEY}}` -- deliberately unlike anything in ordinary prose or markdown.
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")

#: Any run of digits. Used to prove the model wrote no figures of its own.
_DIGIT_RE = re.compile(r"\d")

_PLACEHOLDER_TOKEN = "\x00PLACEHOLDER\x00"

#: A markdown ordered-list marker at the start of a line is structure, the same
#: as `##` or `-`. It states nothing about the mission, so it is not a figure.
#: Anchored to line start so a numeral in prose can never hide behind it.
_LIST_MARKER_RE = re.compile(r"^(\s*)\d+[.)](\s)", re.MULTILINE)


@dataclass(frozen=True)
class Fact:
    """One quantity, its label, and the phrase that states both.

    `phrase` is what lands in the document. It always names what the number
    measures, so the surrounding prose cannot relabel it.
    """

    key: str
    label: str
    value: Any
    unit: str = ""
    phrase: str = ""

    def render(self) -> str:
        return self.phrase or f"{self.value}{self.unit} {self.label}".strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "phrase": self.render(),
        }


@dataclass
class FactSheet:
    """The complete set of figures a generation is allowed to cite."""

    facts: list[Fact] = field(default_factory=list)

    def add(
        self,
        key: str,
        label: str,
        value: Any,
        unit: str = "",
        phrase: str = "",
    ) -> None:
        self.facts.append(Fact(key, label, value, unit, phrase))

    def __len__(self) -> int:
        return len(self.facts)

    def __contains__(self, key: object) -> bool:
        return any(f.key == key for f in self.facts)

    def by_key(self) -> dict[str, Fact]:
        return {f.key: f for f in self.facts}

    def catalogue(self) -> str:
        """The menu handed to the model: placeholders and what they measure.

        VALUES ARE DELIBERATELY ABSENT. Showing "55.6%" next to a token invites
        the model to type the number instead of the token, which is exactly the
        failure being designed out -- and once it has seen a figure it can also
        put a wrong noun next to it. A prompt containing no digits cannot leak
        one into the output. The model chooses which figures belong where and
        writes the interpretation; the fact layer supplies every quantity.
        """
        return "\n".join(f"  {{{{{f.key}}}}} — {f.label}" for f in self.facts)

    def render(self, text: str) -> tuple[str, list[str]]:
        """Substitute every known placeholder. Returns (text, unknown_keys)."""
        known = self.by_key()
        unknown: list[str] = []

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            fact = known.get(key)
            if fact is None:
                unknown.append(key)
                return ""
            return fact.render()

        return PLACEHOLDER_RE.sub(_sub, text), unknown


def misplaced_placeholders(text: str) -> list[str]:
    """Lines that mix a figure with prose of the model's own.

    Each fact renders as a complete, self-labelling phrase, so a model that
    writes "a science yield of {{SCIENCE_YIELD}}" produces "a science yield of
    55.6% science yield (94 of 169 natural frames)" -- correct, and unreadable.
    The fix is structural: a figure gets its own line and prose lines carry no
    figures. Commentary and evidence stop fighting for the same sentence.
    """
    bad: list[str] = []
    for line in text.splitlines():
        if not PLACEHOLDER_RE.search(line):
            continue
        remainder = PLACEHOLDER_RE.sub("", line).strip()
        # Bullets, emphasis and terminal punctuation are decoration, not claims.
        remainder = remainder.strip("-*_•.;:, \t")
        if remainder:
            bad.append(line.strip())
    return bad


def unsanctioned_numerals(text: str) -> list[str]:
    """Digits the model wrote itself, ignoring the placeholders it was given.

    Placeholders are blanked before the scan, so `{{SCIENCE_YIELD}}` is fine and
    a literal `55.6` is not -- the whole point being that the model states no
    figure that did not come from the fact layer.
    """
    masked = PLACEHOLDER_RE.sub(_PLACEHOLDER_TOKEN, text)
    masked = _LIST_MARKER_RE.sub(r"\1-\2", masked)
    offenders: list[str] = []
    for line in masked.splitlines():
        for token in re.findall(r"[\w.,%+-]*\d[\w.,%+-]*", line):
            if _DIGIT_RE.search(token):
                offenders.append(token.strip(".,"))
    return offenders


# ---------------------------------------------------------------------------
# Formatting helpers. Every phrase names its own denominator.
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _kib(bits: float) -> str:
    return f"{bits / 8192:,.1f} KiB"


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _window_set_phrase(indices: list[int], universe: list[int]) -> str:
    from .report_gen import format_window_set

    return format_window_set(indices, universe)


# ---------------------------------------------------------------------------
# Mission-level facts
# ---------------------------------------------------------------------------
def build_mission_facts(
    windows: list[dict[str, Any]],
    run: dict[str, Any],
    fifo: dict[str, Any] | None,
    mission: dict[str, Any],
) -> FactSheet:
    """Every figure the mission briefing is permitted to state."""
    sheet = FactSheet()
    all_idx = [w["window"] for w in windows]

    method = run.get("method", "unknown")
    tier = run.get("tier", "unknown")
    adaptation = run.get("adaptation", "frozen")
    n_natural_total = run.get("n_natural_total", 0)
    n_natural_sent = run.get("n_sent_natural", 0)
    science_yield = run.get("science_yield", 0.0)

    sheet.add("POLICY", "the scheduling policy", method, phrase=f"Policy: `{method}`")
    sheet.add("TIER", "the model tier", tier, phrase=f"Model tier: {tier}")
    sheet.add(
        "ADAPTATION", "the adaptation mode", adaptation,
        phrase=f"Adaptation: {adaptation}",
    )
    sheet.add(
        "N_WINDOWS", "downlink windows in the run", len(all_idx),
        phrase=f"Downlink windows: {len(all_idx)}",
    )
    sheet.add(
        "SOL_RANGE", "the sol range of the mission",
        f"{mission.get('sol_min', '?')}-{mission.get('sol_max', '?')}",
        phrase=f"Sol range: {mission.get('sol_min', '?')}–{mission.get('sol_max', '?')}",
    )

    # --- science yield ---------------------------------------------------
    sheet.add(
        "SCIENCE_YIELD", "science yield over the whole mission", science_yield, "%",
        phrase=(
            f"Mission science yield: {_pct(science_yield)} — {n_natural_sent} of "
            f"the mission's {n_natural_total} natural frames delivered"
        ),
    )
    if fifo:
        fifo_yield = fifo.get("science_yield", 0.0)
        fifo_natural = fifo.get("n_sent_natural", 0)
        sheet.add(
            "FIFO_SCIENCE_YIELD", "FIFO baseline science yield", fifo_yield, "%",
            phrase=(
                f"FIFO baseline yield: {_pct(fifo_yield)} — {fifo_natural} of the "
                f"same {n_natural_total} natural frames"
            ),
        )
        delta = science_yield - fifo_yield
        sheet.add(
            "YIELD_DELTA_VS_FIFO", "yield gain over FIFO, percentage points", delta, "pp",
            phrase=f"Gain over FIFO: {delta * 100:+.1f} percentage points",
        )
        if fifo_yield > 0:
            ratio = science_yield / fifo_yield
            sheet.add(
                "YIELD_RATIO_VS_FIFO", "yield as a multiple of FIFO", ratio, "x",
                phrase=f"Science delivered versus FIFO: {ratio:.1f}× at an identical bit budget",
            )

    # --- downlink --------------------------------------------------------
    bits_used = run.get("bits_used", 0.0)
    bits_avail = run.get("bits_available", 0.0)
    sheet.add(
        "BITS_USED", "bits spent on downlink", bits_used, "bits",
        phrase=f"Downlink spent: {_kib(bits_used)}",
    )
    sheet.add(
        "BITS_AVAILABLE", "bits available across the mission", bits_avail, "bits",
        phrase=f"Downlink available: {_kib(bits_avail)}",
    )
    if bits_avail:
        sheet.add(
            "BITS_UTILISATION", "share of the bit budget used",
            bits_used / bits_avail, "%",
            phrase=f"Bit budget used: {_pct(bits_used / bits_avail)}",
        )
    wasted = run.get("wasted_bit_share", 0.0)
    sheet.add(
        "WASTED_BIT_SHARE", "share of transmitted bits spent on rover hardware",
        wasted, "%",
        phrase=f"Bits spent on rover-hardware frames: {_pct(wasted)} of those transmitted",
    )
    if fifo:
        sheet.add(
            "FIFO_WASTED_BIT_SHARE", "FIFO share of bits spent on rover hardware",
            fifo.get("wasted_bit_share", 0.0), "%",
            phrase=(
                f"Bits spent on rover-hardware frames by FIFO: "
                f"{_pct(fifo.get('wasted_bit_share', 0.0))}, same measure"
            ),
        )
    sheet.add(
        "N_SENT", "frames transmitted, all classes", run.get("n_sent", 0), "frames",
        phrase=f"Frames transmitted: {_n(run.get('n_sent', 0))} in total, all classes",
    )
    sheet.add(
        "N_EXPIRED_TOTAL", "frames that expired unsent, all classes",
        run.get("n_expired", 0), "frames",
        phrase=(
            f"Frames expired unsent: {_n(run.get('n_expired', 0))} across the "
            "mission, all classes — not only natural"
        ),
    )
    if run.get("n_expired_natural") is not None:
        sheet.add(
            "N_EXPIRED_NATURAL", "natural frames that expired unsent",
            run["n_expired_natural"], "frames",
            phrase=(
                f"Natural-science frames expired unsent: "
                f"{_n(run['n_expired_natural'])}"
            ),
        )

    # --- compute vs bandwidth --------------------------------------------
    bits_w = [w["window"] for w in windows if w.get("binding_constraint") == "bits"]
    cyc_w = [w["window"] for w in windows if w.get("binding_constraint") == "cycles"]
    sheet.add(
        "BANDWIDTH_LIMITED", "windows where bits bound the selection", len(bits_w),
        phrase=(
            f"Bandwidth-limited: {_window_set_phrase(bits_w, all_idx)}"
            if bits_w
            else "Bandwidth-limited: no window"
        ),
    )
    sheet.add(
        "COMPUTE_LIMITED", "windows where cycles bound the selection", len(cyc_w),
        phrase=(
            f"Compute-limited: {_window_set_phrase(cyc_w, all_idx)}"
            if cyc_w
            else "Compute-limited: no window"
        ),
    )
    if run.get("prefilter_recall_natural") is not None:
        sheet.add(
            "PREFILTER_RECALL_MISSION",
            "prefilter recall over unique natural frames ever buffered",
            run["prefilter_recall_natural"], "%",
            phrase=(
                f"Prefilter recall (mission, unique frames): "
                f"{_pct(run['prefilter_recall_natural'])} of natural frames ever "
                "buffered earned a real score"
            ),
        )
    per_window = [w["prefilter_recall"] for w in windows
                  if w.get("prefilter_recall") is not None]
    if per_window:
        sheet.add(
            "PREFILTER_RECALL_WINDOW_MEAN",
            "mean per-window prefilter recall (buffer snapshot)",
            sum(per_window) / len(per_window), "%",
            phrase=(
                f"Prefilter recall (mean per-window buffer snapshot): "
                f"{_pct(sum(per_window) / len(per_window))} across the "
                f"{len(per_window)} windows where triage bound — a different "
                "denominator from the mission figure"
            ),
        )
    if run.get("n_unscored") is not None:
        sheet.add(
            "N_UNSCORED", "frame-window pairs left unscored", run["n_unscored"],
            phrase=(
                f"Left unscored by the cycle budget: {_n(run['n_unscored'])} "
                "frame-window pairs"
            ),
        )
    if run.get("scores_affordable_per_window"):
        sheet.add(
            "SCORES_PER_WINDOW", "novelty scores affordable per window",
            run["scores_affordable_per_window"],
            phrase=(
                f"Novelty scores affordable per window: "
                f"{run['scores_affordable_per_window']:.1f}"
            ),
        )

    # --- adaptation ------------------------------------------------------
    sheet.add(
        "N_REFITS", "in-mission model refits", run.get("n_refits", 0),
        phrase=(
            f"In-mission refits: {run.get('n_refits', 0)}"
            if run.get("n_refits")
            else "In-mission refits: none, the model was frozen"
        ),
    )

    # --- notable windows -------------------------------------------------
    by_expiry = sorted(windows, key=lambda w: w.get("n_expired", 0), reverse=True)[:3]
    hot = [w for w in by_expiry if w.get("n_expired", 0) > 0]
    if hot:
        sheet.add(
            "TOP_EXPIRY_WINDOWS", "the windows that lost the most frames to expiry",
            [w["window"] for w in hot],
            phrase=(
                "Heaviest expiry: "
                + ", ".join(
                    f"window {w['window']} ({w.get('n_expired', 0)} frames)"
                    for w in hot
                )
            ),
        )
    return sheet


# ---------------------------------------------------------------------------
# Window-level facts
# ---------------------------------------------------------------------------
def build_window_facts(
    rec: dict[str, Any], n_natural_total: int | None = None
) -> FactSheet:
    """Every figure one window's operator note is permitted to state."""
    from .report_gen import _yield_is_provisional

    sheet = FactSheet()
    w = rec["window"]
    sheet.add("WINDOW", "the window index", w, phrase=f"Window: {w}")
    sheet.add(
        "SOLS", "the sols this window covers", f"{rec['first_sol']}-{rec['last_sol']}",
        phrase=f"Sols covered: {rec['first_sol']}–{rec['last_sol']}",
    )
    sheet.add(
        "ARRIVED", "frames captured during this window", rec["n_arrived"], "frames",
        phrase=f"Frames captured this window: {rec['n_arrived']}",
    )
    sheet.add(
        "BUFFERED", "frames in the buffer when the window opened",
        rec.get("n_buffered", 0), "frames",
        phrase=f"Buffer awaiting a decision: {rec.get('n_buffered', 0)} frames",
    )
    sheet.add(
        "SELECTED", "frames transmitted this window", rec["n_selected"], "frames",
        phrase=f"Frames transmitted: {rec['n_selected']}",
    )
    sheet.add(
        "SENT_BREAKDOWN", "what was transmitted, by class",
        [rec.get("sent_natural", 0), rec.get("sent_rover", 0), rec.get("sent_typical", 0)],
        phrase=(
            f"Transmitted by class: {rec.get('sent_natural', 0)} natural-science, "
            f"{rec.get('sent_rover', 0)} rover-hardware, "
            f"{rec.get('sent_typical', 0)} typical"
        ),
    )
    constraint = rec["binding_constraint"]
    sheet.add(
        "BINDING_CONSTRAINT", "which budget bound the selection", constraint,
        phrase=(
            f"Binding constraint: {constraint} — "
            f"{_pct(rec['bits_used'] / rec['bits_budget']) if rec['bits_budget'] else 'n/a'} "
            "of the bit budget and "
            f"{_pct(rec['cycles_used'] / rec['cycles_budget']) if rec['cycles_budget'] else 'n/a'} "
            "of the cycle budget spent"
        ),
    )
    sheet.add(
        "SCORED", "frames given a full novelty score", rec["n_scored"], "frames",
        phrase=f"Earned a full novelty score: {rec['n_scored']} frames",
    )
    sheet.add(
        "UNSCORED", "frames the cycle budget could not score", rec["n_unscored"],
        "frames",
        phrase=f"Unaffordable to score this window: {rec['n_unscored']} frames",
    )
    if rec.get("n_expired"):
        sheet.add(
            "EXPIRED", "frames that expired this window, all classes",
            rec["n_expired"], "frames",
            phrase=(
                f"Expired unsent this window: {rec['n_expired']} frames, all classes"
            ),
        )
    if rec.get("n_evicted"):
        sheet.add(
            "EVICTED", "frames evicted for storage capacity", rec["n_evicted"], "frames",
            phrase=f"Evicted to free onboard storage: {rec['n_evicted']} frames",
        )
    if rec.get("prefilter_recall") is not None:
        scored_nat = rec.get("prefilter_scored_natural")
        buffered_nat = rec.get("prefilter_buffered_natural")
        counts = (
            f" ({scored_nat} of {buffered_nat} buffered natural frames)"
            if scored_nat is not None and buffered_nat
            else ""
        )
        sheet.add(
            "PREFILTER_RECALL", "prefilter recall in this window's buffer",
            rec["prefilter_recall"], "%",
            phrase=(
                f"Prefilter recall this window: {_pct(rec['prefilter_recall'])}"
                f"{counts} — a buffer snapshot, not the mission-level recall"
            ),
        )
    sent = rec.get("cum_sent_natural")
    available = rec.get("cum_natural_available")
    if sent is not None and available:
        ratio = rec.get("cum_science_yield") or (sent / available)
        provisional = _yield_is_provisional(available, n_natural_total)
        total_clause = f" out of the mission's {n_natural_total}" if n_natural_total else ""
        sheet.add(
            "CUM_YIELD", "cumulative science yield over frames captured so far",
            ratio, "%",
            phrase=(
                f"Cumulative science yield: {_pct(ratio)} — {sent} of the "
                f"{available} natural frames captured so far{total_clause}"
                + (
                    ", provisional while the denominator is this small and not "
                    "comparable to the mission figure"
                    if provisional
                    else ", measured over frames captured so far rather than the "
                    "mission total"
                )
            ),
        )
    elif sent is not None:
        sheet.add(
            "CUM_YIELD", "cumulative science yield", 0.0, "%",
            phrase="Cumulative science yield: none yet, no natural frames captured",
        )
    if rec.get("refit"):
        sheet.add(
            "REFIT", "whether the model refit this window", True,
            phrase="Model refit: performed this window, on frames captured so far",
        )
    return sheet
