"""The fact layer: the model may arrange figures, never write them.

The defect this replaces was a correct number under a wrong label -- "604
natural frames expired" when 604 counted every class. Validation cannot catch
that, because both halves of the sentence are individually true. So the model
is given placeholders instead of numbers, and each placeholder carries its own
label into the document.
"""

from __future__ import annotations

from core.ground.facts import (
    Fact,
    FactSheet,
    build_mission_facts,
    build_window_facts,
    misplaced_placeholders,
    unsanctioned_numerals,
)

WINDOW = {
    "window": 3,
    "first_sol": 130,
    "last_sol": 180,
    "n_arrived": 4,
    "n_buffered": 130,
    "n_scored": 30,
    "n_unscored": 68,
    "n_selected": 6,
    "n_expired": 40,
    "n_evicted": 0,
    "bits_budget": 395871.92,
    "bits_used": 381065.53,
    "cycles_budget": 82407310.0,
    "cycles_used": 80244096.0,
    "binding_constraint": "bits",
    "sent_natural": 4,
    "sent_rover": 0,
    "sent_typical": 2,
    "cum_sent_natural": 8,
    "cum_natural_available": 23,
    "cum_science_yield": 8 / 23,
    "prefilter_recall": 4 / 19,
    "prefilter_scored_natural": 4,
    "prefilter_buffered_natural": 19,
    "refit": False,
}

RUN = {
    "method": "score_first",
    "tier": "rad750",
    "adaptation": "frozen",
    "science_yield": 0.5562130177514792,
    "wasted_bit_share": 0.1601,
    "n_sent": 180,
    "n_sent_natural": 94,
    "n_natural_total": 169,
    "n_expired": 604,
    "n_expired_natural": 70,
    "bits_used": 10359770.3,
    "bits_available": 10688541.9,
    "n_refits": 0,
    "n_unscored": 1019,
    "prefilter_recall_natural": 0.828,
    "n_natural_never_scored": 29,
    "scores_affordable_per_window": 31.7,
}

FIFO = {"science_yield": 0.1538, "n_sent_natural": 26, "wasted_bit_share": 0.325}

MISSION = {
    "n_frames": 856,
    "sol_min": 13,
    "sol_max": 1666,
    "composition": {"natural": 169, "rover": 261, "typical": 426},
}


# ---------------------------------------------------------------------------
# Placeholders carry their own label
# ---------------------------------------------------------------------------


def test_every_fact_renders_a_self_labelling_phrase() -> None:
    """A bare number in a phrase is a number the prose can relabel."""
    sheet = build_mission_facts([WINDOW], RUN, FIFO, MISSION)
    assert len(sheet) > 10
    for fact in sheet.facts:
        phrase = fact.render()
        assert phrase.strip(), fact.key
        # Anything numeric must arrive with words attached.
        if any(ch.isdigit() for ch in phrase):
            assert len(phrase.split()) >= 3, f"{fact.key}: {phrase!r}"


def test_the_total_expiry_fact_names_its_denominator() -> None:
    sheet = build_mission_facts([WINDOW], RUN, FIFO, MISSION)
    phrase = sheet.by_key()["N_EXPIRED_TOTAL"].render()
    assert "all classes" in phrase
    assert "not only natural" in phrase
    # And the natural-only figure is a separate, separately-labelled fact.
    natural = sheet.by_key()["N_EXPIRED_NATURAL"].render()
    assert "atural-science" in natural
    assert natural != phrase


def test_the_two_prefilter_recalls_are_distinguishable_in_prose() -> None:
    mission = build_mission_facts([WINDOW], RUN, FIFO, MISSION).by_key()
    window = build_window_facts(WINDOW, 169).by_key()
    assert "unique" in mission["PREFILTER_RECALL_MISSION"].render()
    assert "buffer" in window["PREFILTER_RECALL"].render()
    assert "not the mission-level recall" in window["PREFILTER_RECALL"].render()


def test_cumulative_yield_fact_is_marked_provisional_early() -> None:
    early = build_window_facts(WINDOW, 169).by_key()["CUM_YIELD"].render()
    assert "captured so far" in early
    assert "provisional" in early

    late = dict(WINDOW, cum_sent_natural=94, cum_natural_available=169)
    phrase = build_window_facts(late, 169).by_key()["CUM_YIELD"].render()
    assert "provisional" not in phrase


# ---------------------------------------------------------------------------
# The catalogue given to the model
# ---------------------------------------------------------------------------


def test_catalogue_contains_no_digits() -> None:
    """A prompt with no numbers in it cannot leak one into the output."""
    for sheet in (
        build_mission_facts([WINDOW], RUN, FIFO, MISSION),
        build_window_facts(WINDOW, 169),
    ):
        assert not any(ch.isdigit() for ch in sheet.catalogue())


def test_catalogue_lists_every_key() -> None:
    sheet = build_mission_facts([WINDOW], RUN, FIFO, MISSION)
    catalogue = sheet.catalogue()
    for fact in sheet.facts:
        assert f"{{{{{fact.key}}}}}" in catalogue


# ---------------------------------------------------------------------------
# Rendering and rejection
# ---------------------------------------------------------------------------


def test_render_substitutes_known_and_drops_unknown() -> None:
    sheet = FactSheet()
    sheet.add("A", "the a", 1, phrase="one apple")
    text, unknown = sheet.render("- {{A}}\n- {{NOPE}}\n")
    assert "one apple" in text
    assert "NOPE" not in text
    assert unknown == ["NOPE"]


def test_bare_numerals_are_caught() -> None:
    assert unsanctioned_numerals("Science yield was 55.6% overall.")
    assert unsanctioned_numerals("It sent 94 frames.")
    assert not unsanctioned_numerals("- {{SCIENCE_YIELD}}")
    assert not unsanctioned_numerals("Bandwidth, not compute, was the constraint.")


def test_ordered_list_markers_are_structure_not_figures() -> None:
    """A numbered recommendation states nothing about the mission."""
    assert not unsanctioned_numerals("1. Keep the current policy.\n2. Watch expiry.")
    # ...but a numeral in the body of that line still is a figure.
    assert unsanctioned_numerals("1. Keep the policy; it delivered 94 frames.")


def test_mixed_lines_are_rejected() -> None:
    """A figure line carries one placeholder and nothing else."""
    assert misplaced_placeholders("Expiry was heavy: {{N_EXPIRED_TOTAL}}.")
    assert misplaced_placeholders("- {{N_EXPIRED_TOTAL}} natural frames")
    assert not misplaced_placeholders("- {{N_EXPIRED_TOTAL}}")
    assert not misplaced_placeholders("- **{{N_EXPIRED_TOTAL}}**")
    assert not misplaced_placeholders("Buffer pressure dominated the losses.")


def test_fact_without_a_phrase_still_labels_itself() -> None:
    assert "frames expired" in Fact("K", "frames expired", 604).render()
