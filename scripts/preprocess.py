"""Convert the extracted .npy tree into memory-mapped float32 arrays + a manifest.

    python -m scripts.preprocess              # idempotent; skips finished splits
    python -m scripts.preprocess --force      # rebuild everything
    python -m scripts.preprocess --only test_typical --limit 64

Each split becomes one contiguous `.npy` array of shape (N, 64, 64, 6) written
with `np.lib.format.open_memmap`, so training streams it without ever loading
it whole, and one row per frame in `data/processed/manifest.csv`.

Source frames are float64 (196,736 bytes each). Nothing in this pipeline needs
more than float32 -- the values are 8-bit DN to begin with -- so the conversion
halves the on-disk footprint and doubles the number of frames per page of RAM.

THE test_novel GOTCHA
---------------------
`test_novel/` holds eleven per-class folders AND an `all/` folder whose files
are byte-identical copies of files in those folders. Globbing the tree gives
881 paths for 446 distinct frames. This script keeps them in two separate
splits and asserts that `all/` never leaks into the per-class walk:

    test_novel_all      430 rows -- the canonical evaluation set
    test_novel_byclass  451 rows -- the per-class breakdown (5 frames carry
                                    two labels and appear once per label)

Both facts were verified against the real archive; see `_assert_novel_not_double_counted`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import paths
from core.dataset import (
    FRAME_SHAPE,
    PROCESSED_DTYPE,
    PROCESSED_FORMAT_VERSION,
    SPLIT_NOVEL_BYCLASS,
    SPLIT_NOVEL_CANONICAL,
    SPLIT_TEST_TYPICAL,
    SPLIT_TRAIN,
    SPLIT_VAL,
    DoubleCountError,
)
from core.filenames import parse_sol
from core.logging_utils import Progress, get_logger, human_bytes, setup_logging
from core.manifest import ManifestRow, write_manifest
from core.provenance import peak_rss_bytes

log = get_logger("novum.preprocess")

#: Preprocessing is O(1) in dataset size. If peak RSS ever crosses this, the
#: streaming contract has regressed -- see StreamingArrayWriter.
RSS_BUDGET_BYTES = 500 * 1024 * 1024

#: Raw directory name -> the split(s) it produces.
TYPICAL_SPLITS: dict[str, str] = {
    "train_typical": SPLIT_TRAIN,
    "validation_typical": SPLIT_VAL,
    "test_typical": SPLIT_TEST_TYPICAL,
}
NOVEL_RAW_DIR = "test_novel"
CANONICAL_SUBDIR = "all"
TYPICAL_CLASS_LABEL = "typical"
UNKNOWN_CLASS_LABEL = "unknown"

ALL_OUTPUT_SPLITS = (
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST_TYPICAL,
    SPLIT_NOVEL_CANONICAL,
    SPLIT_NOVEL_BYCLASS,
)


@dataclass
class Sample:
    """One row to be written: where it comes from and how it is labelled."""

    path: Path
    class_label: str


@dataclass
class SplitPlan:
    name: str
    samples: list[Sample]

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def basenames(self) -> list[str]:
        return [s.path.name for s in self.samples]

    def fingerprint(self) -> str:
        """Identity of the input file set: names and sizes, order-independent."""
        h = hashlib.sha256()
        for sample in sorted(self.samples, key=lambda s: (s.class_label, s.path.name)):
            try:
                size = sample.path.stat().st_size
            except OSError:
                size = -1
            h.update(f"{sample.class_label}|{sample.path.name}|{size}\n".encode())
        return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_split_dir(raw_dir: Path, name: str) -> Path | None:
    """Locate an extracted split directory, tolerating one level of nesting."""
    direct = raw_dir / name
    if direct.is_dir():
        return direct
    for child in sorted(raw_dir.iterdir()) if raw_dir.is_dir() else []:
        candidate = child / name
        if candidate.is_dir():
            return candidate
    return None


def _npy_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.npy"), key=lambda p: p.name)


def plan_typical_split(raw_dir: Path, raw_name: str, split_name: str) -> SplitPlan | None:
    directory = find_split_dir(raw_dir, raw_name)
    if directory is None:
        return None
    files = _npy_files(directory)
    if not files:
        log.warning("%s exists but contains no .npy files", paths.rel(directory))
        return None
    return SplitPlan(split_name, [Sample(p, TYPICAL_CLASS_LABEL) for p in files])


def plan_novel_splits(raw_dir: Path) -> tuple[SplitPlan | None, SplitPlan | None, dict[str, list[str]]]:
    """Build the canonical and per-class novel splits, keeping them separate."""
    directory = find_split_dir(raw_dir, NOVEL_RAW_DIR)
    if directory is None:
        return None, None, {}

    canonical_dir = directory / CANONICAL_SUBDIR
    class_dirs = sorted(
        d for d in directory.iterdir() if d.is_dir() and d.name != CANONICAL_SUBDIR
    )

    # Per-class walk: explicitly enumerates class folders and never touches all/.
    byclass_samples: list[Sample] = []
    labels_by_basename: dict[str, list[str]] = defaultdict(list)
    for class_dir in class_dirs:
        for path in sorted(class_dir.glob("*.npy"), key=lambda p: p.name):
            byclass_samples.append(Sample(path, class_dir.name))
            labels_by_basename[path.name].append(class_dir.name)

    canonical_samples: list[Sample] = []
    if canonical_dir.is_dir():
        for path in sorted(canonical_dir.glob("*.npy"), key=lambda p: p.name):
            labels = sorted(set(labels_by_basename.get(path.name, ())))
            canonical_samples.append(
                Sample(path, "|".join(labels) if labels else UNKNOWN_CLASS_LABEL)
            )
    else:
        log.warning(
            "%s has no all/ folder; falling back to the per-class union as the "
            "canonical set (deduplicated by filename)",
            paths.rel(directory),
        )
        seen: set[str] = set()
        for sample in byclass_samples:
            if sample.path.name not in seen:
                seen.add(sample.path.name)
                canonical_samples.append(
                    Sample(sample.path, "|".join(sorted(set(labels_by_basename[sample.path.name]))))
                )

    canonical = SplitPlan(SPLIT_NOVEL_CANONICAL, canonical_samples) if canonical_samples else None
    byclass = SplitPlan(SPLIT_NOVEL_BYCLASS, byclass_samples) if byclass_samples else None
    return canonical, byclass, dict(labels_by_basename)


# ---------------------------------------------------------------------------
# The double-counting guard
# ---------------------------------------------------------------------------
def _assert_novel_not_double_counted(
    canonical: SplitPlan | None,
    byclass: SplitPlan | None,
) -> None:
    """Hard assertion: neither novel split may count a frame twice.

    Specifically:
      1. No path from `all/` may appear in the per-class split. This is the
         failure a recursive glob produces, and it inflates the novel set from
         446 distinct frames to 881 rows.
      2. The canonical set must have unique filenames -- it is the denominator
         of every reported metric.
      3. The per-class split must have unique *paths*. Repeated basenames are
         legitimate there (a frame with two labels), repeated paths are not.
    """
    if byclass is not None:
        leaked = [
            s.path for s in byclass.samples if CANONICAL_SUBDIR in s.path.parts[:-1]
        ]
        if leaked:
            raise DoubleCountError(
                f"{len(leaked)} file(s) from test_novel/{CANONICAL_SUBDIR}/ leaked into the "
                f"per-class split (e.g. {leaked[0]}). all/ duplicates the class folders; "
                "counting both inflates the novel set roughly 2x."
            )

        paths_seen = [str(s.path) for s in byclass.samples]
        if len(set(paths_seen)) != len(paths_seen):
            dupes = [p for p, c in Counter(paths_seen).items() if c > 1][:3]
            raise DoubleCountError(
                f"per-class split lists the same path more than once (e.g. {dupes})"
            )

    if canonical is not None:
        names = canonical.basenames
        if len(set(names)) != len(names):
            dupes = [n for n, c in Counter(names).items() if c > 1][:3]
            raise DoubleCountError(
                f"canonical novel split contains duplicate filenames (e.g. {dupes}); "
                "it is the evaluation denominator and must hold each frame exactly once."
            )

    if canonical is not None and byclass is not None:
        canonical_names = set(canonical.basenames)
        byclass_names = set(byclass.basenames)
        multi = sum(1 for _, c in Counter(byclass.basenames).items() if c > 1)
        log.info(
            "novel splits: canonical=%d frames, per-class=%d rows over %d unique frames "
            "(%d multi-label)",
            len(canonical_names),
            byclass.n,
            len(byclass_names),
            multi,
        )
        orphans = canonical_names - byclass_names
        if orphans:
            log.warning(
                "%d canonical frame(s) have no class folder entry; labelled %r",
                len(orphans),
                UNKNOWN_CLASS_LABEL,
            )
        extra = byclass_names - canonical_names
        if extra:
            log.info(
                "%d class-folder frame(s) are not in all/; they stay out of the "
                "canonical evaluation set by design",
                len(extra),
            )


def _warn_on_cross_split_overlap(plans: list[SplitPlan]) -> dict[str, int]:
    """Typical splits should be disjoint. Report, but do not abort, if not."""
    typical = [p for p in plans if p.name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST_TYPICAL)]
    overlaps: dict[str, int] = {}
    for i, a in enumerate(typical):
        for b in typical[i + 1 :]:
            shared = set(a.basenames) & set(b.basenames)
            if shared:
                key = f"{a.name}&{b.name}"
                overlaps[key] = len(shared)
                log.warning(
                    "%d filename(s) appear in both %s and %s (e.g. %s). Train/test "
                    "separation may be compromised; metrics will be optimistic.",
                    len(shared),
                    a.name,
                    b.name,
                    sorted(shared)[0],
                )
    return overlaps


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------
#: Frames copied per block when trimming a short split. 64 frames = ~6 MiB.
_TRIM_BLOCK_FRAMES = 64


class StreamingArrayWriter:
    """Append frames to a preallocated .npy in constant memory.

    Writing through `np.lib.format.open_memmap` is the obvious implementation
    and the wrong one here: every assignment dirties a page, and the kernel
    keeps those pages resident until writeback. Building train_typical that way
    peaked at 914 MiB RSS for an 872 MiB array -- the whole output, held in RAM.
    On the 2 GB server this project targets, that is the difference between
    working and being OOM-killed.

    So the file is created (header + preallocation) with open_memmap, the
    mapping is dropped immediately, and frames are then written sequentially
    through an ordinary buffered file handle. Peak RSS becomes O(1) in the
    number of frames: one 192 KB source frame and a 1 MiB write buffer.
    """

    def __init__(
        self,
        path: Path,
        n_frames: int,
        frame_shape: tuple[int, ...] = FRAME_SHAPE,
        dtype: type = np.float32,
    ) -> None:
        self.path = Path(path)
        self.n_frames = int(n_frames)
        self.frame_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)
        self.written = 0
        self._fh = None

        # Create the header and preallocate, then drop the mapping at once.
        # Mapping a file does not make its pages resident; only touching them
        # does, and we never touch one.
        mm = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=self.dtype, shape=(self.n_frames, *self.frame_shape)
        )
        self.data_offset = int(mm.offset)
        del mm

    @property
    def frame_nbytes(self) -> int:
        return int(np.prod(self.frame_shape)) * self.dtype.itemsize

    def __enter__(self) -> StreamingArrayWriter:
        self._fh = open(self.path, "r+b", buffering=1024 * 1024)
        self._fh.seek(self.data_offset)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def append(self, frame: np.ndarray) -> None:
        if self._fh is None:
            raise RuntimeError("StreamingArrayWriter used outside its context manager")
        if self.written >= self.n_frames:
            raise RuntimeError(
                f"tried to write frame {self.written} into an array sized {self.n_frames}"
            )
        # tofile writes straight through the handle, with no intermediate bytes
        # object. ascontiguousarray is a no-op when the source is already C-order
        # float32, and a single-frame copy otherwise.
        np.ascontiguousarray(frame, dtype=self.dtype).tofile(self._fh)
        self.written += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def finalize(self) -> Path:
        """Close the file, shrinking it if fewer frames arrived than planned."""
        self.close()
        if self.written == self.n_frames:
            return self.path
        return self._trim_to(self.written)

    def _trim_to(self, count: int) -> Path:
        """Rewrite the array with a smaller leading dimension, block by block.

        The .npy header encodes the shape, and a shorter shape can encode to a
        different header length, so the data cannot simply be truncated in
        place. Copying is done with plain file IO -- no mmap -- to keep the
        memory guarantee intact even on this rare path.
        """
        trimmed = self.path.with_suffix(self.path.suffix + ".trim")
        trimmed.unlink(missing_ok=True)

        mm = np.lib.format.open_memmap(
            trimmed, mode="w+", dtype=self.dtype, shape=(count, *self.frame_shape)
        )
        dest_offset = int(mm.offset)
        del mm

        block = _TRIM_BLOCK_FRAMES * self.frame_nbytes
        remaining = count * self.frame_nbytes
        with open(self.path, "rb") as src, open(trimmed, "r+b") as dst:
            src.seek(self.data_offset)
            dst.seek(dest_offset)
            while remaining > 0:
                chunk = src.read(min(block, remaining))
                if not chunk:
                    raise OSError(f"unexpected end of {self.path} while trimming")
                dst.write(chunk)
                remaining -= len(chunk)
            dst.flush()
            os.fsync(dst.fileno())

        self.path.unlink(missing_ok=True)
        trimmed.replace(self.path)
        return self.path


def build_split(
    plan: SplitPlan,
    out_dir: Path,
    *,
    skip_bad: bool = False,
) -> tuple[Path, list[ManifestRow], int]:
    """Write one split's float32 array. Returns (path, manifest rows, unparsed sols).

    Memory is O(1) in the number of frames: exactly one source frame is
    resident at a time. See StreamingArrayWriter.
    """
    final_path = out_dir / f"{plan.name}.npy"
    tmp_path = out_dir / f"{plan.name}.npy.tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path.unlink(missing_ok=True)

    rows: list[ManifestRow] = []
    unparsed: list[str] = []
    written = 0

    writer = StreamingArrayWriter(tmp_path, plan.n)
    try:
        with writer, Progress(plan.n, f"building {plan.name}", unit="frames", logger=log) as bar:
            for sample in plan.samples:
                try:
                    # One frame at a time. Never the whole split, never a list
                    # of frames -- that is the entire memory contract here.
                    frame = np.load(sample.path)
                except (OSError, ValueError) as exc:
                    if skip_bad:
                        log.warning("skipping unreadable %s: %s", sample.path.name, exc)
                        bar.advance()
                        continue
                    raise RuntimeError(
                        f"could not read {sample.path}: {exc}. "
                        "Re-run `python -m scripts.fetch_data --force`, or pass --skip-bad."
                    ) from exc

                if frame.shape != FRAME_SHAPE:
                    msg = f"{sample.path.name} has shape {frame.shape}, expected {FRAME_SHAPE}"
                    if skip_bad:
                        log.warning("skipping %s", msg)
                        bar.advance()
                        continue
                    raise ValueError(msg)

                writer.append(frame)
                del frame  # do not keep a reference alive across the loop

                sol = parse_sol(sample.path.name)
                if sol is None:
                    unparsed.append(sample.path.name)
                rows.append(
                    ManifestRow(
                        index=written,
                        split=plan.name,
                        class_=sample.class_label,
                        sol=sol,
                        source_filename=sample.path.name,
                    )
                )
                written += 1
                bar.advance()

        if written != plan.n:
            log.warning("%s: wrote %d of %d planned frames", plan.name, written, plan.n)
        writer.finalize()
        tmp_path.replace(final_path)
    except BaseException:
        writer.close()
        tmp_path.unlink(missing_ok=True)
        tmp_path.with_suffix(tmp_path.suffix + ".trim").unlink(missing_ok=True)
        raise

    if unparsed:
        log.warning(
            "%s: %d filename(s) had no parseable sol (e.g. %s). They keep an empty sol "
            "and sort last in any chronological replay.",
            plan.name,
            len(unparsed),
            ", ".join(unparsed[:3]),
        )

    return final_path, rows, len(unparsed)


def load_existing_meta(out_dir: Path) -> dict:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if raw.get("format_version") == PROCESSED_FORMAT_VERSION else {}


def split_is_current(plan: SplitPlan, out_dir: Path, existing: dict) -> bool:
    """True when a previous run already produced exactly this split."""
    entry = (existing.get("splits") or {}).get(plan.name)
    if not entry:
        return False
    if entry.get("fingerprint") != plan.fingerprint():
        return False
    array_path = out_dir / entry.get("path", "")
    if not array_path.exists():
        return False
    try:
        header = np.load(array_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return tuple(header.shape) == tuple(entry.get("shape", ()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.preprocess",
        description="Convert extracted .npy frames into memmapped float32 arrays + manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=None, help="default: data/raw")
    p.add_argument("--out-dir", type=Path, default=None, help="default: data/processed")
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="SPLIT",
        help=f"restrict to one split (repeatable). One of: {', '.join(ALL_OUTPUT_SPLITS)}",
    )
    p.add_argument("--force", action="store_true", help="rebuild even if already current")
    p.add_argument("--limit", type=int, default=None, help="cap frames per split (smoke tests)")
    p.add_argument("--skip-bad", action="store_true", help="skip unreadable frames instead of failing")
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    raw_dir = args.raw_dir or paths.raw_dir()
    out_dir = args.out_dir or paths.processed_dir()

    if not raw_dir.is_dir():
        log.error("no raw data at %s. Run `python -m scripts.fetch_data` first.", paths.rel(raw_dir))
        return 2

    # -- plan ---------------------------------------------------------------
    plans: list[SplitPlan] = []
    for raw_name, split_name in TYPICAL_SPLITS.items():
        plan = plan_typical_split(raw_dir, raw_name, split_name)
        if plan is None:
            log.warning("split %s not found under %s; skipping", raw_name, paths.rel(raw_dir))
        else:
            plans.append(plan)

    canonical, byclass, _ = plan_novel_splits(raw_dir)
    _assert_novel_not_double_counted(canonical, byclass)
    plans.extend(p for p in (canonical, byclass) if p is not None)

    if not plans:
        log.error(
            "no splits found under %s. Expected directories named %s. "
            "Run `python -m scripts.fetch_data`.",
            paths.rel(raw_dir),
            ", ".join([*TYPICAL_SPLITS, NOVEL_RAW_DIR]),
        )
        return 2

    if args.only:
        wanted = set(args.only)
        unknown = wanted - set(ALL_OUTPUT_SPLITS)
        if unknown:
            log.error("unknown split(s): %s", ", ".join(sorted(unknown)))
            return 2
        plans = [p for p in plans if p.name in wanted]

    if args.limit:
        plans = [SplitPlan(p.name, p.samples[: args.limit]) for p in plans]
        log.warning("--limit %d: this is a smoke-test build, not a full dataset", args.limit)

    overlaps = _warn_on_cross_split_overlap(plans)

    # -- build --------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_meta(out_dir)
    existing_splits = dict(existing.get("splits") or {})
    manifest_rows: list[ManifestRow] = []
    built: list[str] = []
    reused: list[str] = []

    for plan in plans:
        if not args.force and split_is_current(plan, out_dir, existing):
            log.info("%s is up to date (%d frames); skipping", plan.name, plan.n)
            reused.append(plan.name)
            continue

        log.info("building %s: %d frames", plan.name, plan.n)
        array_path, rows, n_unparsed = build_split(plan, out_dir, skip_bad=args.skip_bad)
        manifest_rows.extend(rows)
        existing_splits[plan.name] = {
            "path": array_path.name,
            "count": len(rows),
            "shape": [len(rows), *FRAME_SHAPE],
            "fingerprint": plan.fingerprint(),
            "n_unique_files": len({r.source_filename for r in rows}),
            "n_unparsed_sol": n_unparsed,
        }
        built.append(plan.name)

    # Every split we did NOT rebuild keeps its previous manifest rows: the
    # reused ones AND any split excluded by --only. Carrying forward only the
    # `reused` list once truncated the manifest to a single split after
    # `--only train_typical --force`, leaving arrays with zero manifest rows.
    carried = {name for name in existing_splits if name not in built}
    if carried:
        try:
            from core.manifest import read_manifest

            previous = read_manifest(out_dir / "manifest.csv")
            carried_rows = [r for r in previous if r.split in carried]
            manifest_rows.extend(carried_rows)

            rows_by_split: dict[str, int] = {}
            for row in carried_rows:
                rows_by_split[row.split] = rows_by_split.get(row.split, 0) + 1
            for name in sorted(carried):
                expected = int(existing_splits[name].get("count", 0))
                got = rows_by_split.get(name, 0)
                if got != expected and (out_dir / existing_splits[name]["path"]).exists():
                    log.warning(
                        "%s: previous manifest holds %d rows but the array has %d frames; "
                        "rebuilding that split's rows",
                        name,
                        got,
                        expected,
                    )
                    plan = next((p for p in plans if p.name == name), None)
                    if plan is not None:
                        manifest_rows = [r for r in manifest_rows if r.split != name]
                        _, rows, _ = build_split(plan, out_dir, skip_bad=args.skip_bad)
                        manifest_rows.extend(rows)
                    else:
                        log.warning(
                            "%s is not in this run's plan (--only?); its manifest rows stay "
                            "missing until a full `python -m scripts.preprocess` run",
                            name,
                        )
        except (FileNotFoundError, ValueError) as exc:
            log.warning("could not carry manifest rows forward (%s); rebuilding reused splits", exc)
            for plan in plans:
                if plan.name in reused:
                    _, rows, _ = build_split(plan, out_dir, skip_bad=args.skip_bad)
                    manifest_rows.extend(rows)

    manifest_rows.sort(key=lambda r: (ALL_OUTPUT_SPLITS.index(r.split), r.index))

    # -- write metadata -----------------------------------------------------
    written = write_manifest(
        manifest_rows,
        out_dir / "manifest.csv",
        out_dir / "manifest.parquet",
    )
    if len(written) == 1:
        log.info("wrote manifest.csv (pyarrow not installed, so no parquet)")

    meta = {
        "format_version": PROCESSED_FORMAT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "zenodo:3732485",
        "frame_shape": list(FRAME_SHAPE),
        "dtype": PROCESSED_DTYPE,
        "splits": {k: existing_splits[k] for k in ALL_OUTPUT_SPLITS if k in existing_splits},
    }
    if overlaps:
        meta["cross_split_filename_overlaps"] = overlaps

    meta_tmp = out_dir / "meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta_tmp.replace(out_dir / "meta.json")

    # -- report -------------------------------------------------------------
    total_bytes = sum(
        (out_dir / info["path"]).stat().st_size
        for info in meta["splits"].values()
        if (out_dir / info["path"]).exists()
    )
    log.info("-" * 68)
    for name, info in meta["splits"].items():
        size = (out_dir / info["path"]).stat().st_size if (out_dir / info["path"]).exists() else 0
        note = ""
        if info["n_unique_files"] != info["count"]:
            note = f"  ({info['n_unique_files']} unique frames, multi-label rows)"
        log.info("  %-20s %6d frames  %10s%s", name, info["count"], human_bytes(size), note)
    log.info("  %-20s %6d frames  %10s", "TOTAL", sum(i["count"] for i in meta["splits"].values()), human_bytes(total_bytes))
    log.info("-" * 68)
    log.info("built: %s | reused: %s", ", ".join(built) or "none", ", ".join(reused) or "none")
    log.info("manifest: %s", paths.rel(out_dir / "manifest.csv"))

    free = shutil.disk_usage(out_dir).free
    if free < total_bytes:
        log.warning("only %s free on this volume", human_bytes(free))

    # Preprocessing is O(1) in dataset size by construction. Reporting the
    # number every run makes a regression visible immediately rather than at
    # 3am on a 2 GB box, and `built` guards against reporting a no-op run.
    peak = peak_rss_bytes()
    if built:
        log.info("peak RSS %s (budget %s)", human_bytes(peak), human_bytes(RSS_BUDGET_BYTES))
        if peak > RSS_BUDGET_BYTES:
            log.warning(
                "peak RSS %s exceeded the %s streaming budget. Preprocessing is supposed to "
                "hold one frame at a time; this suggests the streaming write path regressed.",
                human_bytes(peak),
                human_bytes(RSS_BUDGET_BYTES),
            )

    log.info("next: make train TIER=rad750")
    return 0


if __name__ == "__main__":
    sys.exit(main())
