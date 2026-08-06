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

log = get_logger("novum.preprocess")

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
def build_split(
    plan: SplitPlan,
    out_dir: Path,
    *,
    skip_bad: bool = False,
) -> tuple[Path, list[ManifestRow], int]:
    """Write one split's float32 array. Returns (path, manifest rows, unparsed sols)."""
    final_path = out_dir / f"{plan.name}.npy"
    tmp_path = out_dir / f"{plan.name}.npy.tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path.unlink(missing_ok=True)

    rows: list[ManifestRow] = []
    unparsed: list[str] = []
    written = 0

    array = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(plan.n, *FRAME_SHAPE),
    )
    try:
        with Progress(plan.n, f"building {plan.name}", unit="frames", logger=log) as bar:
            for sample in plan.samples:
                try:
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

                array[written] = frame.astype(np.float32, copy=False)

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
            # Some frames were skipped; shrink the array to what we actually wrote.
            log.warning("%s: wrote %d of %d planned frames", plan.name, written, plan.n)
            array.flush()
            del array
            trimmed = np.load(tmp_path, mmap_mode="r")[:written]
            resized = out_dir / f"{plan.name}.npy.tmp2"
            np.save(resized, np.asarray(trimmed))
            tmp_path.unlink(missing_ok=True)
            resized.replace(tmp_path)
        else:
            array.flush()
            del array

        tmp_path.replace(final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        (out_dir / f"{plan.name}.npy.tmp2").unlink(missing_ok=True)
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

    # Splits we skipped still need their manifest rows carried forward.
    if reused:
        try:
            from core.manifest import read_manifest

            previous = read_manifest(out_dir / "manifest.csv")
            manifest_rows.extend(r for r in previous if r.split in reused)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("could not reuse manifest rows for %s (%s); rebuilding them", reused, exc)
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

    log.info("next: make train TIER=rad750")
    return 0


if __name__ == "__main__":
    sys.exit(main())
