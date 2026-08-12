"""Turn (64, 64, 6) multispectral frames into something a browser can show.

A judge looking at this console has to see the frames. The whole argument is
about which images were worth their bits, and that argument cannot be made with
bar charts. But a browser cannot display a six-band float32 array, and the UI
must never touch a .npy at request time, so the conversion happens once, here,
at build time.

BAND MAPPING. The archive stores Mastcam's six filter bands in L1-L6 order, not
wavelength order. The band statistics say so plainly: band 1 averages 43.8 DN
against 107.7 for band 0 and ~150 for bands 3-5, and bands 2-5 correlate at
0.96-0.99 with each other. That is the Mars spectrum -- almost no blue, a bright
red and near-infrared plateau -- and it pins the ordering:

    band 0 = L1  527 nm  green
    band 1 = L2  445 nm  blue
    band 2 = L3  676 nm  red
    band 3 = L4  751 nm  near-infrared
    band 4 = L5  867 nm  near-infrared
    band 5 = L6 1012 nm  near-infrared

So approximate true colour is R<-band 2, G<-band 0, B<-band 1. Fixed, stated,
and identical for every frame: a thumbnail that changed its own mapping would
make two frames incomparable by eye, which is the one thing the grid is for.

NORMALISATION is per frame, across the three chosen bands JOINTLY. Per-channel
normalisation would white-balance every tile independently and throw away the
red cast that makes Mars look like Mars -- and worse, would make a rust-coloured
vein and a grey drill hole render the same. Joint scaling keeps colour a real
signal. The 2nd-98th percentile window keeps one hot pixel from crushing the
rest of the tile.

This is a VISUALISATION, not a calibrated radiometric product. No science is
done on these PNGs; they exist so a human can see what the rover saw.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

#: (red, green, blue) source band indices. See the module docstring.
RGB_BANDS: tuple[int, int, int] = (2, 0, 1)

#: Fixed per-channel gains, applied before the stretch and identical for every
#: frame. The archive stores raw DN, not cross-calibrated radiance, so the raw
#: band ratio (1.00 : 0.76 : 0.31) renders Mars as sulphur yellow-green. The
#: surface is nearer 1.00 : 0.72 : 0.50, and almost all of the error is in the
#: blue band. These gains correct that and nothing else. They are constants, not
#: a per-frame white balance: two tiles must stay comparable by eye, which is
#: the entire purpose of showing them side by side.
CHANNEL_GAINS: tuple[float, float, float] = (1.08, 0.88, 1.45)

#: Slight lift of the midtones. Regolith occupies a narrow bright band and the
#: features worth seeing -- drill holes, vein fill, broken faces -- sit below it.
GAMMA = 0.9

#: Percentile window for per-frame contrast stretch.
CLIP_PERCENTILES: tuple[float, float] = (2.0, 98.0)

#: Tile size in the atlas. Frames are 64x64, and 32 is the one reduction that
#: is an exact 2x2 box average -- no resampling kernel, no ringing, no invented
#: detail. It also halves the sheet to under 2 MB, which is the difference
#: between a console that paints immediately on a free-tier host and one that
#: visibly waits. The grid renders tiles around 44 px, so the browser upscales
#: slightly; on 64x64 texture patches that is invisible, and it is a better
#: trade than a 7 MB download.
TILE = 32

#: Source frames are this size. TILE must divide it exactly.
SOURCE = 64


def _box_downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Average factor x factor blocks. Done in DN space, before any stretch."""
    if factor == 1:
        return x
    h, w, c = x.shape
    return x.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3))


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """One (H, W, 6) frame -> (TILE, TILE, 3) uint8, ready to encode."""
    x = np.asarray(frame, dtype=np.float32)
    if x.shape[0] % TILE or x.shape[1] % TILE:
        raise ValueError(f"TILE={TILE} does not divide frame shape {x.shape[:2]}")
    x = _box_downsample(x, x.shape[0] // TILE)
    rgb = np.stack([x[..., b] for b in RGB_BANDS], axis=-1)
    rgb = rgb * np.asarray(CHANNEL_GAINS, dtype=np.float32)

    # Degeneracy is about SPATIAL detail, not the post-gain range. A tile that
    # is constant across its pixels still has a gap between its channels once
    # the gains are applied, and stretching that gap to full range would paint
    # a featureless patch in saturated colour -- inventing a signal out of a
    # constant. Ask whether any channel varies across the tile at all.
    flat = rgb.reshape(-1, 3)
    if float(np.ptp(flat, axis=0).max()) <= 1e-6:
        return np.full(rgb.shape, 128, dtype=np.uint8)

    lo, hi = np.percentile(rgb, CLIP_PERCENTILES)
    if hi <= lo:
        return np.full(rgb.shape, 128, dtype=np.uint8)

    unit = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) ** GAMMA
    return (unit * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# A minimal PNG encoder.
#
# Pillow would do this in one line, but it is a dependency the serving image
# does not otherwise need, and PNG's RGB8 case is genuinely small: a header, one
# deflate stream of filtered scanlines, a terminator. Filtering is vectorised --
# only *decoding* has to run sequentially, because only the decoder needs the
# reconstructed neighbours.
# ---------------------------------------------------------------------------
def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _filtered_scanlines(rgb: np.ndarray) -> bytes:
    """Per-row adaptive filtering, minimum-sum-of-absolute-differences."""
    height, width, channels = rgb.shape
    raw = rgb.reshape(height, width * channels).astype(np.int16)
    bpp = channels

    out = bytearray()
    prior = np.zeros(width * channels, dtype=np.int16)
    for row in raw:
        left = np.concatenate([np.zeros(bpp, dtype=np.int16), row[:-bpp]])
        up_left = np.concatenate([np.zeros(bpp, dtype=np.int16), prior[:-bpp]])

        # Paeth predictor, vectorised over the row.
        p = left + prior - up_left
        pa, pb, pc = np.abs(p - left), np.abs(p - prior), np.abs(p - up_left)
        paeth = np.where(
            (pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prior, up_left)
        )

        candidates = {
            0: row,
            1: row - left,
            2: row - prior,
            3: row - ((left + prior) // 2),
            4: row - paeth,
        }
        best = min(candidates, key=lambda k: int(np.abs(candidates[k]).sum()))
        out.append(best)
        out.extend((candidates[best] & 0xFF).astype(np.uint8).tobytes())
        prior = row
    return bytes(out)


def encode_png(rgb: np.ndarray, *, level: int = 9) -> bytes:
    """Encode an (H, W, 3) uint8 array as a PNG byte string."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {rgb.shape}")
    height, width = rgb.shape[:2]

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    body = zlib.compress(_filtered_scanlines(rgb), level)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", body)
        + _chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------
def atlas_geometry(n_frames: int, columns: int = 32) -> tuple[int, int, int, int]:
    """(columns, rows, width_px, height_px) for an n-frame atlas."""
    columns = max(1, columns)
    rows = (n_frames + columns - 1) // columns
    return columns, rows, columns * TILE, rows * TILE


def build_atlas(frames: np.ndarray, columns: int = 32) -> tuple[bytes, dict]:
    """Pack every frame into one sprite sheet.

    ONE image, not 856. The grid shows a hundred tiles at a time and a free-tier
    host serving that many separate requests is the difference between a console
    that feels instant and one that visibly loads. The UI slices tiles out with
    background-position, which costs the browser nothing.
    """
    n = len(frames)
    columns, rows, width, height = atlas_geometry(n, columns)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(n):
        r, c = divmod(i, columns)
        tile = frame_to_rgb(frames[i])
        sheet[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile

    meta = {
        "n_frames": n,
        "tile": TILE,
        "columns": columns,
        "rows": rows,
        "width": width,
        "height": height,
        "rgb_bands": list(RGB_BANDS),
        "channel_gains": list(CHANNEL_GAINS),
        "gamma": GAMMA,
        "band_meaning": {
            "0": "L1 527 nm green",
            "1": "L2 445 nm blue",
            "2": "L3 676 nm red",
            "3": "L4 751 nm NIR",
            "4": "L5 867 nm NIR",
            "5": "L6 1012 nm NIR",
        },
        "normalisation": (
            f"per frame, joint across the three mapped bands, "
            f"{CLIP_PERCENTILES[0]}-{CLIP_PERCENTILES[1]} percentile stretch"
        ),
        "note": "visualisation only; not a calibrated radiometric product",
    }
    return encode_png(sheet), meta
