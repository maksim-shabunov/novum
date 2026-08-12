"""Fail if a figure written in prose disagrees with the artifact it came from.

    make check-figures

Documents mark their headline numbers with an invisible marker:

    NOVUM delivers **55.6%**<!--@NOVUM_YIELD--> of the natural-science frames

The marker names a key in `core.figures`; this script reads the value straight
out of `runs/sim/<run>/summary.json` and `artifacts/metrics/*.json` and compares.
A regenerated simulation that moves a number turns every document quoting it
red, which is the only way four documents stay honest about one truth.

Markers render as nothing in Markdown, so a reader sees ordinary prose. Being
invisible is the point: a number that needs a footnote to be trusted is a number
that will be quoted without it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from core import paths
from core.figures import FiguresUnavailable, published_figures

#: `text<!--@KEY-->`. The captured group is the rendered figure immediately
#: before the marker, emphasis and all.
MARKER_RE = re.compile(r"(?P<text>[^\s<>]+)[ \t]*<!--@(?P<key>[A-Z0-9_]+)-->")

#: Markdown decoration that is presentation, not value.
_DECORATION = "*_`~ \t"

#: Punctuation that can sit against a figure without being part of it.
#: NOTE the ASCII hyphen is absent on purpose -- a negative figure like -0.047
#: begins with one, and stripping it would silently compare the wrong value.
#: En and em dashes never occur inside a figure, so those are safe.
_LEFT_PUNCT = "([{<«–—"
_RIGHT_PUNCT = ")]}>».,;:–—"

#: Documents that are allowed to carry markers. Anything else is generated and
#: has no business hand-quoting a figure.
DEFAULT_DOCS = (
    "README.md",
    "docs/bob-usage.md",
    "docs/demo-script.md",
    "web/README.md",
)


def _strip(text: str) -> str:
    """The bare figure, with markdown emphasis and adjacent punctuation removed."""
    for _ in range(3):   # e.g.  **(0.556**  -> emphasis, bracket, emphasis
        text = text.strip(_DECORATION)
        text = text.lstrip(_LEFT_PUNCT).rstrip(_RIGHT_PUNCT)
    return text.strip(_DECORATION)


def check_document(path: Path, figures: dict) -> list[str]:
    """Return one message per disagreement. Empty means the document is honest."""
    problems: list[str] = []
    if not path.is_file():
        return problems

    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in MARKER_RE.finditer(line):
            key = match.group("key")
            written = _strip(match.group("text"))
            figure = figures.get(key)
            if figure is None:
                problems.append(
                    f"{paths.rel(path)}:{lineno}: unknown figure {key!r} -- "
                    "add it to core.figures or remove the marker"
                )
                continue
            if written != figure.text:
                problems.append(
                    f"{paths.rel(path)}:{lineno}: {key} says {written!r} but "
                    f"{figure.source} says {figure.text!r}"
                )
    return problems


def unmarked_suspects(path: Path, figures: dict) -> list[str]:
    """Figures that appear in prose without a marker tying them to a source.

    Advisory rather than fatal: prose legitimately contains numbers that are not
    published figures. But a bare "55.6%" sitting next to a marked one is
    exactly how the two drift apart, so it is worth naming.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    marked = {m.group(0) for m in MARKER_RE.finditer(text)}
    stripped = text
    for m in marked:
        stripped = stripped.replace(m, "")

    suspects: list[str] = []
    distinctive = {
        f.text: key
        for key, f in figures.items()
        # Short figures ("26", "94") occur too often in ordinary prose to be
        # evidence of anything, and a zero is not distinctive at any length --
        # "0.000" is equally the snapdragon yield and the standard deviation of
        # a precision score, so flagging it only teaches people to ignore
        # warnings.
        if len(f.text) >= 5 and any(ch.isdigit() for ch in f.text) and f.value != 0
    }
    for value, key in distinctive.items():
        # Not when it is a fragment of a longer number: "$0.000002" contains
        # "0.000" and is a pricing example, not the science yield.
        pattern = re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d])")
        if pattern.search(stripped):
            suspects.append(
                f"{paths.rel(path)}: {value!r} appears unmarked; "
                f"mark it <!--@{key}--> so it is checked"
            )
    return suspects


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.check_figures",
        description="Verify every marked figure against the artifact it came from.",
    )
    p.add_argument("docs", nargs="*", default=None, help="Documents to check.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on unmarked figures that look like published values.",
    )
    p.add_argument("--list", action="store_true", help="Print every figure and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        figures = published_figures()
    except FiguresUnavailable as exc:
        print(f"cannot check figures: {exc}", file=sys.stderr)
        return 1

    if args.list:
        for key in sorted(figures):
            f = figures[key]
            print(f"  {key:32} {f.text:>16}   {f.source}")
            if f.note:
                print(f"  {'':32} {'':>16}   {f.note}")
        return 0

    docs = [Path(d) for d in (args.docs or DEFAULT_DOCS)]
    docs = [d if d.is_absolute() else paths.PROJECT_ROOT / d for d in docs]

    problems: list[str] = []
    warnings: list[str] = []
    checked = 0
    for doc in docs:
        if not doc.is_file():
            continue
        checked += 1
        problems += check_document(doc, figures)
        warnings += unmarked_suspects(doc, figures)

    for w in warnings:
        print(f"  warn  {w}")
    for p in problems:
        print(f"  FAIL  {p}", file=sys.stderr)

    if problems or (args.strict and warnings):
        detail = f"{len(problems)} disagreement(s)"
        if args.strict and warnings:
            detail += f" and {len(warnings)} unmarked figure(s)"
        print(
            f"\n{detail} across {checked} document(s). "
            "Regenerate the documents, or rerun the simulation.",
            file=sys.stderr,
        )
        return 1

    marked = sum(
        len(MARKER_RE.findall(d.read_text(encoding="utf-8"))) for d in docs if d.is_file()
    )
    print(f"{marked} marked figure(s) across {checked} document(s) agree with their sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
