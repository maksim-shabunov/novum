"""Thumbnail rendering: a browser has to be able to show a six-band float array.

The PNG encoder is hand-written to keep Pillow out of the serving image, so it
is worth proving it produces files a decoder actually accepts, and that the
band mapping is a fixed constant rather than something that drifts per frame.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from core.thumbnails import (
    CHANNEL_GAINS,
    RGB_BANDS,
    SOURCE,
    TILE,
    atlas_geometry,
    build_atlas,
    encode_png,
    frame_to_rgb,
)


def _frames(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Roughly the archive's dynamic range, with the Mars band structure:
    # dark blue band, bright red and NIR.
    base = rng.normal(0, 12, size=(n, SOURCE, SOURCE, 6)).astype(np.float32)
    return base + np.array([100.0, 40.0, 131.0, 150.0, 149.0, 149.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# PNG encoder
# ---------------------------------------------------------------------------


def _decode_png(data: bytes) -> np.ndarray:
    """A minimal reference decoder, so the test does not trust the encoder."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, width, height, idat = 8, 0, 0, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", payload[:10])
            assert (depth, colour) == (8, 2), "expected 8-bit RGB"
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length

    raw = zlib.decompress(idat)
    stride = width * 3
    out = np.zeros((height, stride), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.uint8)
    at = 0
    for y in range(height):
        ftype = raw[at]
        line = np.frombuffer(raw[at + 1 : at + 1 + stride], dtype=np.uint8).astype(np.int32)
        at += 1 + stride
        recon = np.zeros(stride, dtype=np.int32)
        for x in range(stride):
            a = recon[x - 3] if x >= 3 else 0
            b = int(prior[x])
            c = int(prior[x - 3]) if x >= 3 else 0
            if ftype == 0:
                pred = 0
            elif ftype == 1:
                pred = a
            elif ftype == 2:
                pred = b
            elif ftype == 3:
                pred = (a + b) // 2
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            recon[x] = (line[x] + pred) & 0xFF
        out[y] = recon.astype(np.uint8)
        prior = out[y]
    return out.reshape(height, width, 3)


def test_encoded_png_round_trips_exactly() -> None:
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(9, 7, 3), dtype=np.uint8)
    assert np.array_equal(_decode_png(encode_png(rgb)), rgb)


def test_encoder_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Band mapping and normalisation
# ---------------------------------------------------------------------------


def test_mapping_is_fixed_not_per_frame() -> None:
    """Two frames must stay comparable by eye; that is what the grid is for."""
    assert RGB_BANDS == (2, 0, 1)
    assert len(CHANNEL_GAINS) == 3


def test_tile_divides_the_source_frame() -> None:
    assert SOURCE % TILE == 0, "the downsample must be an exact box average"


def test_frame_renders_to_the_tile_size() -> None:
    tile = frame_to_rgb(_frames(1)[0])
    assert tile.shape == (TILE, TILE, 3)
    assert tile.dtype == np.uint8


def test_mars_renders_red_not_green() -> None:
    """The raw band ratio makes Mars sulphur-yellow; the gains exist to fix it."""
    tile = frame_to_rgb(_frames(1)[0]).astype(float)
    r, g, b = tile[..., 0].mean(), tile[..., 1].mean(), tile[..., 2].mean()
    assert r > g > b, f"expected a red-dominant tile, got R={r:.0f} G={g:.0f} B={b:.0f}"


def test_flat_frame_does_not_divide_by_zero() -> None:
    flat = np.full((SOURCE, SOURCE, 6), 42.0, dtype=np.float32)
    tile = frame_to_rgb(flat)
    assert tile.shape == (TILE, TILE, 3)
    assert np.all(tile == 128)


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------


def test_atlas_geometry_covers_every_frame() -> None:
    columns, rows, width, height = atlas_geometry(856, columns=32)
    assert columns * rows >= 856
    assert (width, height) == (columns * TILE, rows * TILE)


def test_atlas_places_frames_at_their_advertised_position() -> None:
    """The UI slices tiles by (column, row); if that mapping is off, every
    thumbnail in the console is the wrong frame."""
    frames = _frames(5, seed=7)
    png, meta = build_atlas(frames, columns=3)
    sheet = _decode_png(png)
    assert (sheet.shape[1], sheet.shape[0]) == (meta["width"], meta["height"])

    for i in range(len(frames)):
        r, c = divmod(i, meta["columns"])
        placed = sheet[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
        assert np.array_equal(placed, frame_to_rgb(frames[i])), f"frame {i} misplaced"


def test_atlas_metadata_documents_the_mapping() -> None:
    _png, meta = build_atlas(_frames(2), columns=2)
    assert meta["rgb_bands"] == list(RGB_BANDS)
    assert "band_meaning" in meta
    assert "per frame" in meta["normalisation"]
    assert "not a calibrated" in meta["note"]
