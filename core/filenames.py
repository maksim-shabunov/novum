"""Tolerant parsing of Mastcam sample filenames.

Filenames encode the sol (Martian day) and the camera, but the format is not
consistent across the release. Both of these are real names from the archive:

    mcam00487_R0_sol0069_7.npy      -> sol 69,  camera R
    mcam00117_MR_0_sol0024_39.npy   -> sol 24,  camera MR

The sol matters because the downlink simulator replays frames in chronological
sol order, which is the only ordering that makes "novelty relative to terrain
seen so far" meaningful. Parsing is therefore tolerant by contract: an
unrecognised name yields None and a warning, never an exception. A single odd
filename must not be able to abort a 9,000-file preprocessing run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches sol0069, sol_69, SOL-1326. Deliberately anchored to nothing.
_SOL_RE = re.compile(r"sol[_\-]?(\d{1,6})", re.IGNORECASE)

# Camera token: the alphabetic group right after the mcam sequence id.
# mcam00487_R0_...  -> R      mcam00117_MR_0_... -> MR
_CAM_AFTER_ID_RE = re.compile(r"mcam\d+[_\-]([A-Za-z]{1,3})", re.IGNORECASE)
# Fallback: any Mastcam-like camera token sitting immediately before the sol.
_CAM_BEFORE_SOL_RE = re.compile(r"[_\-]([MLR]{1,2})[_\-]?\d*[_\-]sol", re.IGNORECASE)

_SEQ_RE = re.compile(r"mcam(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class FrameMeta:
    """Everything we can recover from a sample filename. Fields may be None."""

    source_filename: str
    sol: int | None = None
    camera: str | None = None
    sequence: int | None = None

    @property
    def parsed(self) -> bool:
        return self.sol is not None


def parse_sol(filename: str) -> int | None:
    """Return the sol as an int, or None if the name does not encode one."""
    m = _SOL_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:  # pragma: no cover - regex already guarantees digits
        return None


def parse_camera(filename: str) -> str | None:
    """Return the camera token (e.g. 'R', 'MR', 'ML') uppercased, or None."""
    for rx in (_CAM_AFTER_ID_RE, _CAM_BEFORE_SOL_RE):
        m = rx.search(filename)
        if not m:
            continue
        token = m.group(1).upper()
        # Guard against swallowing the literal "SOL" as a camera token.
        if token and token != "SOL":
            return token
    return None


def parse_sequence(filename: str) -> int | None:
    m = _SEQ_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:  # pragma: no cover
        return None


def parse_frame_meta(filename: str) -> FrameMeta:
    """Parse a filename into FrameMeta. Never raises."""
    name = str(filename)
    return FrameMeta(
        source_filename=name,
        sol=parse_sol(name),
        camera=parse_camera(name),
        sequence=parse_sequence(name),
    )


def sort_key_chronological(meta: FrameMeta) -> tuple[int, int, str]:
    """Stable chronological key. Unparseable sols sort last, not first.

    Sorting them first would silently place unknown frames at the start of a
    replay and corrupt the "terrain seen so far" baseline.
    """
    return (
        meta.sol if meta.sol is not None else 10**9,
        meta.sequence if meta.sequence is not None else 10**9,
        meta.source_filename,
    )
