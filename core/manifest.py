"""The frame manifest: one row per sample, in every processed split.

Columns are exactly the five the project contract specifies:

    index, split, class, sol, source_filename

`index` is the row offset **within that split's array**, so the primary key is
the pair (split, index). CSV is always written because it is dependency-free
and readable by the serving image, which has no pandas or pyarrow. Parquet is
written additionally when pyarrow is available.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

MANIFEST_COLUMNS: tuple[str, ...] = ("index", "split", "class", "sol", "source_filename")

# `class` is a Python keyword, so the dataclass field is `class_` and we map it
# to/from the wire name in one place.
_FIELD_TO_COLUMN = {
    "index": "index",
    "split": "split",
    "class_": "class",
    "sol": "sol",
    "source_filename": "source_filename",
}
_COLUMN_TO_FIELD = {v: k for k, v in _FIELD_TO_COLUMN.items()}


@dataclass(frozen=True)
class ManifestRow:
    index: int
    split: str
    class_: str
    sol: int | None
    source_filename: str

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "split": self.split,
            "class": self.class_,
            "sol": "" if self.sol is None else self.sol,
            "source_filename": self.source_filename,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ManifestRow:
        sol_raw = d.get("sol", "")
        if sol_raw is None or (isinstance(sol_raw, str) and sol_raw.strip() == ""):
            sol: int | None = None
        else:
            try:
                sol = int(sol_raw)
            except (TypeError, ValueError):
                sol = None
        return cls(
            index=int(d["index"]),
            split=str(d["split"]),
            class_=str(d.get("class", "") or ""),
            sol=sol,
            source_filename=str(d["source_filename"]),
        )


def _atomic_write_csv(rows: Sequence[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_manifest(
    rows: Sequence[ManifestRow],
    csv_path: Path,
    parquet_path: Path | None = None,
) -> list[Path]:
    """Write the manifest atomically. Returns the paths actually written."""
    written = [csv_path]
    _atomic_write_csv(rows, csv_path)

    if parquet_path is not None:
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError:
            return written

        table = pa.table(
            {
                "index": pa.array([r.index for r in rows], type=pa.int64()),
                "split": pa.array([r.split for r in rows], type=pa.string()),
                "class": pa.array([r.class_ for r in rows], type=pa.string()),
                "sol": pa.array([r.sol for r in rows], type=pa.int64()),
                "source_filename": pa.array(
                    [r.source_filename for r in rows], type=pa.string()
                ),
            }
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, parquet_path)
        written.append(parquet_path)

    return written


def read_manifest(csv_path: Path) -> list[ManifestRow]:
    """Read the manifest using only the standard library."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"No manifest at {csv_path}. Run `make data` (or python -m scripts.preprocess)."
        )
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest {csv_path} is missing columns: {sorted(missing)}")
        return [ManifestRow.from_dict(d) for d in reader]


def group_by_split(rows: Iterable[ManifestRow]) -> dict[str, list[ManifestRow]]:
    out: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        out[row.split].append(row)
    for split_rows in out.values():
        split_rows.sort(key=lambda r: r.index)
    return dict(out)


def rows_for_split(rows: Iterable[ManifestRow], split: str) -> list[ManifestRow]:
    return sorted((r for r in rows if r.split == split), key=lambda r: r.index)


def class_counts(rows: Iterable[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.class_] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


assert set(_FIELD_TO_COLUMN) == {f.name for f in fields(ManifestRow)}
assert set(_COLUMN_TO_FIELD) == set(MANIFEST_COLUMNS)
