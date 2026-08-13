"""Published prose must agree with the artifacts it claims to quote.

Four documents carry the same headline numbers. Nothing but a check stops a
rerun from moving a figure in `results/` and leaving the README quoting last
week's yield -- and a reader who spots the disagreement cannot tell which one is
wrong, so both become worthless.

These tests are the enforcement. They fail loudly on drift rather than warning
somewhere nobody looks.
"""

from __future__ import annotations

import pytest

from core import paths
from core.figures import FiguresUnavailable, published_figures
from scripts.check_figures import DEFAULT_DOCS, check_document, unmarked_suspects

pytestmark = pytest.mark.skipif(
    not (paths.runs_dir() / "sim").is_dir(),
    reason="no simulation runs on this machine; nothing to check figures against",
)


@pytest.fixture(scope="module")
def figures():
    try:
        return published_figures()
    except FiguresUnavailable as exc:
        pytest.skip(str(exc))


# ---------------------------------------------------------------------------
# The documents
# ---------------------------------------------------------------------------


def test_every_marked_figure_matches_its_source(figures) -> None:
    problems: list[str] = []
    for name in DEFAULT_DOCS:
        problems += check_document(paths.PROJECT_ROOT / name, figures)
    assert not problems, "published prose disagrees with its artifacts:\n" + "\n".join(
        f"  {p}" for p in problems
    )


def test_no_headline_figure_is_quoted_without_a_marker(figures) -> None:
    """An unmarked figure is one the checker cannot protect."""
    suspects: list[str] = []
    for name in DEFAULT_DOCS:
        suspects += unmarked_suspects(paths.PROJECT_ROOT / name, figures)
    assert not suspects, "figures quoted without a source marker:\n" + "\n".join(
        f"  {s}" for s in suspects
    )


def test_the_readme_states_the_headline_comparison(figures) -> None:
    """A reader who stops after ten lines must know what this achieved."""
    readme = (paths.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    head = "\n".join(readme.splitlines()[:40])
    for key in ("FIFO_YIELD", "NOVUM_YIELD", "ORACLE_YIELD"):
        assert figures[key].text in head, f"{key} missing from the opening of the README"


def test_readme_has_every_mandated_section() -> None:
    """The challenge mandates these headings, in this order."""
    readme = (paths.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "## Problem statement",
        "## Solution description",
        "## AI approach and architecture",
        "## Selected challenge theme",
        "## How IBM Bob was used",
    ]
    positions = []
    for heading in required:
        assert heading in readme, f"README is missing the required section {heading!r}"
        positions.append(readme.index(heading))
    assert positions == sorted(positions), (
        "the mandated sections are present but out of order: "
        f"{[h for _, h in sorted(zip(positions, required, strict=True))]}"
    )


def test_licence_and_dataset_attribution_are_present() -> None:
    """CC-BY-4.0 requires the attribution; its absence would be noticed."""
    root = paths.PROJECT_ROOT
    assert (root / "LICENSE").is_file(), "no LICENSE for the code"

    dataset = (root / "DATASET_LICENSE").read_text(encoding="utf-8")
    for required in ("CC-BY-4.0", "10.5281/zenodo.3732485", "Kerner", "NASA"):
        assert required in dataset, f"DATASET_LICENSE does not mention {required!r}"

    readme = (root / "README.md").read_text(encoding="utf-8")
    for required in ("CC-BY-4.0", "zenodo.3732485", "Kerner", "NASA/JPL-Caltech/MSSS"):
        assert required in readme, f"README attribution does not mention {required!r}"


# ---------------------------------------------------------------------------
# The checker itself
# ---------------------------------------------------------------------------


def test_drift_is_actually_caught(figures, tmp_path) -> None:
    """The point of the exercise: a wrong figure must fail, not warn."""
    doc = tmp_path / "drifted.md"
    doc.write_text(
        "NOVUM delivers **99.9%**<!--@NOVUM_YIELD--> of the frames.\n", encoding="utf-8"
    )
    problems = check_document(doc, figures)
    assert problems, "a figure disagreeing with its source passed the check"
    assert "99.9%" in problems[0]
    assert figures["NOVUM_YIELD"].text in problems[0]


def test_a_correct_figure_passes(figures, tmp_path) -> None:
    """And the checker must not cry wolf, or it will be switched off."""
    doc = tmp_path / "honest.md"
    doc.write_text(
        f"NOVUM delivers **{figures['NOVUM_YIELD'].text}**<!--@NOVUM_YIELD--> "
        f"against {figures['FIFO_YIELD'].text}<!--@FIFO_YIELD--> for FIFO.\n",
        encoding="utf-8",
    )
    assert check_document(doc, figures) == []


def test_markers_survive_markdown_emphasis_and_punctuation(figures, tmp_path) -> None:
    """Figures appear inside tables, brackets and bold; all must still match."""
    value = figures["RAD750HW_RAD750_YIELD"].text
    doc = tmp_path / "decorated.md"
    doc.write_text(
        f"| **{value}**<!--@RAD750HW_RAD750_YIELD--> |\n"
        f"yield ({value}<!--@RAD750HW_RAD750_YIELD-->) held.\n"
        f"fell to *{value}*<!--@RAD750HW_RAD750_YIELD-->, then rose.\n",
        encoding="utf-8",
    )
    assert check_document(doc, figures) == []


def test_a_negative_figure_keeps_its_sign(figures, tmp_path) -> None:
    """Stripping punctuation must not eat the minus off -0.047."""
    fig = figures.get("RAD750_ALL_SCORED_DELTA")
    if fig is None:
        pytest.skip("no all-scored delta in this run")
    assert fig.text.startswith("-")
    doc = tmp_path / "negative.md"
    doc.write_text(f"a change of **{fig.text}**<!--@RAD750_ALL_SCORED_DELTA-->\n",
                   encoding="utf-8")
    assert check_document(doc, figures) == []


def test_an_unknown_key_is_an_error_not_a_silent_pass(figures, tmp_path) -> None:
    doc = tmp_path / "typo.md"
    doc.write_text("yield was 55.6%<!--@NOVUM_YEILD-->\n", encoding="utf-8")
    problems = check_document(doc, figures)
    assert problems and "unknown figure" in problems[0]


def test_every_figure_names_where_it_came_from(figures) -> None:
    """A figure whose source cannot be opened is a figure nobody can check.

    The source is a file for a single-run figure and a directory for one
    aggregated across seeds; either way it has to exist on disk.
    """
    for key, fig in figures.items():
        assert fig.source, f"{key} has no source"
        path = paths.PROJECT_ROOT / fig.source
        assert path.exists(), f"{key} cites {fig.source}, which does not exist"


def test_a_multi_word_figure_is_compared_whole(figures, tmp_path) -> None:
    """`0.720 ± 0.004` is one value, not a bare standard deviation.

    The marker sits after the whole phrase, so the checker must read back as
    many words as the figure has -- otherwise it silently compared "0.004"
    against "0.720 ± 0.004" and reported a disagreement that was its own.
    """
    fig = figures.get("RAD750_ROC_AUC_SPREAD")
    if fig is None:
        pytest.skip("no sweep spread in this run")
    assert " " in fig.text

    good = tmp_path / "good.md"
    good.write_text(f"| {fig.text}<!--@RAD750_ROC_AUC_SPREAD--> |\n", encoding="utf-8")
    assert check_document(good, figures) == []

    bad = tmp_path / "bad.md"
    bad.write_text("| 0.111 ± 0.999<!--@RAD750_ROC_AUC_SPREAD--> |\n", encoding="utf-8")
    assert check_document(bad, figures)
