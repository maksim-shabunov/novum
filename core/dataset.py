"""Processed-split access, and the guardrails against double counting.

THE test_novel GOTCHA
---------------------
`test_novel/` in the Zenodo archive contains eleven per-class subfolders AND an
`all/` folder. Verified against the real archive:

    all/                       430 files
    class folders combined     451 files  (446 unique basenames)
    all/ basenames             a strict SUBSET of the class-folder basenames
    file contents              byte-identical between all/ and its class copy
    5 basenames                appear in TWO class folders (genuinely multi-label)

So a naive `glob('test_novel/**/*.npy')` returns 881 paths for at most 446
distinct frames, inflating the evaluation set by ~2x and silently corrupting
every metric. NOVUM therefore materialises two separate splits that must never
be concatenated:

    test_novel_all      -> canonical evaluation set (430 frames, one row each)
    test_novel_byclass  -> per-class breakdown (451 rows; a frame that carries
                           two class labels appears once per label)

`concat_splits` refuses to combine them, and `assert_no_double_count` is the
hard assertion behind that refusal.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths
from .manifest import ManifestRow, read_manifest, rows_for_split

SPLIT_TRAIN = "train_typical"
SPLIT_VAL = "validation_typical"
SPLIT_TEST_TYPICAL = "test_typical"
SPLIT_NOVEL_CANONICAL = "test_novel_all"
SPLIT_NOVEL_BYCLASS = "test_novel_byclass"

ALL_SPLITS: tuple[str, ...] = (
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST_TYPICAL,
    SPLIT_NOVEL_CANONICAL,
    SPLIT_NOVEL_BYCLASS,
)

#: Splits that describe the same underlying frames at different granularity.
#: Combining any two of these double counts. This is enforced, not documented.
OVERLAPPING_SPLIT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({SPLIT_NOVEL_CANONICAL, SPLIT_NOVEL_BYCLASS}),
)

FRAME_SHAPE: tuple[int, int, int] = (64, 64, 6)
PROCESSED_DTYPE = "float32"
PROCESSED_FORMAT_VERSION = 1


class DoubleCountError(AssertionError):
    """Raised when an operation would count the same frame more than once."""


@dataclass(frozen=True)
class SplitInfo:
    name: str
    path: Path
    count: int
    shape: tuple[int, ...]
    fingerprint: str
    n_unique_files: int
    n_unparsed_sol: int

    @property
    def has_intra_split_duplicates(self) -> bool:
        return self.n_unique_files != self.count


@dataclass
class ProcessedMeta:
    format_version: int
    created_utc: str
    frame_shape: tuple[int, ...]
    dtype: str
    splits: dict[str, SplitInfo]
    source: str = "zenodo:3732485"

    @classmethod
    def load(cls, path: Path | None = None) -> ProcessedMeta:
        path = Path(path or paths.processed_meta_path())
        if not path.exists():
            raise FileNotFoundError(
                f"No processed dataset at {path}. Run `make data` first "
                "(python -m scripts.fetch_data && python -m scripts.preprocess)."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format_version") != PROCESSED_FORMAT_VERSION:
            raise ValueError(
                f"{path} has format_version={raw.get('format_version')}, expected "
                f"{PROCESSED_FORMAT_VERSION}. Re-run preprocessing with --force."
            )
        base = path.parent
        splits = {
            name: SplitInfo(
                name=name,
                path=base / d["path"],
                count=int(d["count"]),
                shape=tuple(d["shape"]),
                fingerprint=str(d["fingerprint"]),
                n_unique_files=int(d.get("n_unique_files", d["count"])),
                n_unparsed_sol=int(d.get("n_unparsed_sol", 0)),
            )
            for name, d in raw["splits"].items()
        }
        return cls(
            format_version=int(raw["format_version"]),
            created_utc=str(raw["created_utc"]),
            frame_shape=tuple(raw["frame_shape"]),
            dtype=str(raw["dtype"]),
            splits=splits,
            source=str(raw.get("source", "zenodo:3732485")),
        )

    def to_json(self) -> dict:
        return {
            "format_version": self.format_version,
            "created_utc": self.created_utc,
            "source": self.source,
            "frame_shape": list(self.frame_shape),
            "dtype": self.dtype,
            "splits": {
                name: {
                    "path": info.path.name,
                    "count": info.count,
                    "shape": list(info.shape),
                    "fingerprint": info.fingerprint,
                    "n_unique_files": info.n_unique_files,
                    "n_unparsed_sol": info.n_unparsed_sol,
                }
                for name, info in self.splits.items()
            },
        }


@dataclass
class SplitData:
    """A processed split: its memory-mapped array plus its manifest rows."""

    name: str
    array: np.ndarray  # memmap, shape (N, 64, 64, 6), float32
    rows: list[ManifestRow]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.array):
            raise ValueError(
                f"Split {self.name}: manifest has {len(self.rows)} rows but array has "
                f"{len(self.array)} frames. The processed data is inconsistent; "
                "re-run `python -m scripts.preprocess --force`."
            )

    def __len__(self) -> int:
        return len(self.array)

    @property
    def basenames(self) -> list[str]:
        return [r.source_filename for r in self.rows]

    @property
    def sols(self) -> np.ndarray:
        """Sols as int64, with -1 for frames whose filename did not parse."""
        return np.array([-1 if r.sol is None else r.sol for r in self.rows], dtype=np.int64)

    def chronological_index(self) -> np.ndarray:
        """Row indices ordered by sol; unparseable sols go last (never first)."""
        sols = self.sols
        keys = np.where(sols < 0, np.iinfo(np.int64).max, sols)
        return np.lexsort((np.arange(len(keys)), keys))


def load_manifest_rows(split: str | None = None) -> list[ManifestRow]:
    rows = read_manifest(paths.manifest_csv_path())
    return rows_for_split(rows, split) if split else rows


def load_array(split: str, *, mmap_mode: str | None = "r") -> np.ndarray:
    """Memory-map a split's float32 array. Never loads it fully into RAM."""
    meta = ProcessedMeta.load()
    if split not in meta.splits:
        raise KeyError(f"Unknown split {split!r}. Available: {sorted(meta.splits)}")
    info = meta.splits[split]
    if not info.path.exists():
        raise FileNotFoundError(
            f"Split {split!r} is registered in meta.json but {info.path} is missing. "
            "Re-run `python -m scripts.preprocess --force`."
        )
    arr = np.load(info.path, mmap_mode=mmap_mode)
    if arr.shape[1:] != FRAME_SHAPE:
        raise ValueError(f"Split {split!r} has frame shape {arr.shape[1:]}, expected {FRAME_SHAPE}")
    return arr


def load_split(split: str, *, mmap_mode: str | None = "r") -> SplitData:
    return SplitData(
        name=split,
        array=load_array(split, mmap_mode=mmap_mode),
        rows=load_manifest_rows(split),
    )


def iter_chunks(
    array: np.ndarray, chunk_size: int = 512, *, limit: int | None = None
) -> Iterator[np.ndarray]:
    """Yield contiguous chunks copied out of a memmap, as float32.

    Copying is intentional: downstream code mutates (centres, scales) the batch,
    and mutating a memmap opened 'r' would raise, while 'r+' would corrupt the
    processed data on disk.
    """
    n = len(array) if limit is None else min(limit, len(array))
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # np.array(..., copy=True), not np.asarray: asarray on a memmap slice of
        # matching dtype hands back the read-only view itself, and the first
        # in-place op downstream raises "output array is read-only".
        yield np.array(array[start:stop], dtype=np.float32, copy=True)


class ChunkedArray:
    """A re-iterable, memory-bounded view over a split's frames.

    Fitting needs several passes over the data (mean, then a randomized range
    finder, then power iterations). A plain generator is exhausted after one
    pass, so the training path hands models this instead: every `iter()` starts
    a fresh sweep over the memmap.
    """

    def __init__(self, array: np.ndarray, chunk_size: int = 512, limit: int | None = None) -> None:
        self.array = array
        self.chunk_size = max(1, int(chunk_size))
        self.limit = None if limit is None else max(0, int(limit))

    @property
    def n_frames(self) -> int:
        return len(self.array) if self.limit is None else min(self.limit, len(self.array))

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter_chunks(self.array, self.chunk_size, limit=self.limit)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ChunkedArray(n_frames={self.n_frames}, chunk_size={self.chunk_size})"


def assert_no_double_count(
    groups: dict[str, Sequence[str]],
    *,
    context: str = "",
) -> None:
    """Assert that no source filename appears in more than one named group.

    `groups` maps a label (usually a split name) to that group's source
    filenames. Raises DoubleCountError listing the offending names.
    """
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for label, names in groups.items():
        for name in names:
            prior = seen.get(name)
            if prior is not None and prior != label:
                collisions.append((name, prior, label))
            else:
                seen[name] = label

    if collisions:
        sample = "; ".join(f"{n} in both {a} and {b}" for n, a, b in collisions[:5])
        raise DoubleCountError(
            f"Double counting detected{' in ' + context if context else ''}: "
            f"{len(collisions)} filename(s) appear in more than one group. {sample}"
            + ("..." if len(collisions) > 5 else "")
        )


def assert_splits_combinable(split_names: Sequence[str]) -> None:
    """Reject combinations of splits that describe the same frames."""
    requested = set(split_names)
    for group in OVERLAPPING_SPLIT_GROUPS:
        overlap = requested & group
        if len(overlap) > 1:
            raise DoubleCountError(
                f"Refusing to combine {sorted(overlap)}: these splits describe the same "
                "underlying frames at different granularity. "
                f"Use {SPLIT_NOVEL_CANONICAL!r} as the evaluation set and "
                f"{SPLIT_NOVEL_BYCLASS!r} only for the per-class breakdown."
            )


def concat_splits(split_names: Sequence[str], *, mmap_mode: str | None = "r") -> SplitData:
    """Concatenate splits after proving the combination cannot double count."""
    assert_splits_combinable(split_names)

    parts = [load_split(name, mmap_mode=mmap_mode) for name in split_names]
    assert_no_double_count(
        {p.name: p.basenames for p in parts},
        context=f"concat_splits({list(split_names)})",
    )

    array = np.concatenate([np.asarray(p.array) for p in parts], axis=0)
    rows: list[ManifestRow] = []
    for part in parts:
        offset = len(rows)
        rows.extend(
            ManifestRow(
                index=offset + r.index,
                split=r.split,
                class_=r.class_,
                sol=r.sol,
                source_filename=r.source_filename,
            )
            for r in part.rows
        )
    return SplitData(name="+".join(split_names), array=array, rows=rows)
