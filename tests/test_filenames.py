"""Filename parsing must be tolerant: return None, never raise."""

from __future__ import annotations

import pytest

from core.filenames import (
    FrameMeta,
    parse_camera,
    parse_frame_meta,
    parse_sequence,
    parse_sol,
    sort_key_chronological,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Both formats are real names from the Zenodo archive.
        ("mcam00487_R0_sol0069_7.npy", 69),
        ("mcam00117_MR_0_sol0024_39.npy", 24),
        ("mcam07566_R0_sol1496_4.npy", 1496),
        ("mcam02946_MR_0_sol0696_1.npy", 696),
        ("SOL_1234_whatever.npy", 1234),
        ("sol-7.npy", 7),
    ],
)
def test_parse_sol(filename: str, expected: int) -> None:
    assert parse_sol(filename) == expected


@pytest.mark.parametrize(
    "filename",
    ["no_day_here.npy", "", "mcam00487_R0_.npy", "solaris.npy", "12345.npy"],
)
def test_parse_sol_returns_none_instead_of_raising(filename: str) -> None:
    assert parse_sol(filename) is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("mcam00487_R0_sol0069_7.npy", "R"),
        ("mcam00117_MR_0_sol0024_39.npy", "MR"),
        ("mcam00012_ML_0_sol0013_0.npy", "ML"),
    ],
)
def test_parse_camera(filename: str, expected: str) -> None:
    assert parse_camera(filename) == expected


def test_parse_camera_never_returns_the_literal_sol_token() -> None:
    assert parse_camera("mcam_sol0069.npy") != "SOL"


def test_parse_sequence() -> None:
    assert parse_sequence("mcam00487_R0_sol0069_7.npy") == 487
    assert parse_sequence("nothing.npy") is None


def test_parse_frame_meta_round_trip() -> None:
    meta = parse_frame_meta("mcam00117_MR_0_sol0024_39.npy")
    assert meta == FrameMeta(
        source_filename="mcam00117_MR_0_sol0024_39.npy", sol=24, camera="MR", sequence=117
    )
    assert meta.parsed


def test_unparseable_names_sort_last_not_first() -> None:
    """An unknown sol must never seed the front of a chronological replay."""
    names = ["mcam1_R0_sol0100_0.npy", "garbage.npy", "mcam2_R0_sol0005_0.npy"]
    ordered = sorted((parse_frame_meta(n) for n in names), key=sort_key_chronological)
    assert [m.source_filename for m in ordered] == [
        "mcam2_R0_sol0005_0.npy",
        "mcam1_R0_sol0100_0.npy",
        "garbage.npy",
    ]
