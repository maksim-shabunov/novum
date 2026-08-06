"""Download and extract the Mastcam novelty dataset from Zenodo record 3732485.

    python -m scripts.fetch_data                 # all four splits
    python -m scripts.fetch_data --only test_novel.zip
    python -m scripts.fetch_data --force         # re-download everything

Designed for a flaky link on a remote box:

  * **Resume** -- partial downloads land in `<name>.zip.part` and continue with
    an HTTP Range request. Zenodo returns 206, verified against the live API.
  * **Checksum caching** -- md5 comes from the Zenodo API (with the published
    values embedded as an offline fallback). A verified file is recorded in
    `.fetch_cache.json`, so re-runs skip it without re-hashing 250 MB.
  * **Idempotent** -- running this twice costs a few stat calls and nothing else.
  * **Non-interactive** -- no prompts, ever. Exit code 0 means everything is on
    disk and verified.

The four archives total ~332 MB compressed and ~2.4 GB extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from core import paths
from core.logging_utils import Progress, get_logger, human_bytes, setup_logging

log = get_logger("novum.fetch")

ZENODO_RECORD_ID = os.environ.get("NOVUM_ZENODO_RECORD_ID", "3732485")
ZENODO_API = "https://zenodo.org/api/records/{record}"
FILE_URL = "https://zenodo.org/api/records/{record}/files/{name}/content"
USER_AGENT = "novum-fetch/0.1 (+https://doi.org/10.5281/zenodo.3732485)"

CACHE_FILENAME = ".fetch_cache.json"
CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class RemoteFile:
    name: str
    md5: str | None
    size: int | None

    @property
    def url(self) -> str:
        return FILE_URL.format(record=ZENODO_RECORD_ID, name=self.name)

    @property
    def split_dir(self) -> str:
        return self.name[:-4] if self.name.endswith(".zip") else self.name


#: Published values for record 3732485, captured from the Zenodo API. Used when
#: the API is unreachable so an offline mirror or a proxied box still verifies.
FALLBACK_FILES: dict[str, tuple[str, int]] = {
    "train_typical.zip": ("c00a056386eafc8597e0295890a79b40", 257603661),
    "validation_typical.zip": ("6062a278279c10bf7ead7438846940b7", 36673763),
    "test_typical.zip": ("1393469a7348e233f2154d501864d7b6", 12100634),
    "test_novel.zip": ("b423a12d6539eac4bf2c02acb729ba8e", 25924517),
}

#: Frame counts verified against the real archives. Used only for warnings --
#: a mismatch is worth surfacing but must never abort a download.
EXPECTED_FILE_COUNTS: dict[str, int] = {
    "train_typical": 9302,
    "validation_typical": 1386,
    "test_typical": 426,
    "test_novel": 881,  # 430 in all/ + 451 across the class folders
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_path(dest: Path) -> Path:
    return dest / CACHE_FILENAME


def load_cache(dest: Path) -> dict:
    path = _cache_path(dest)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("ignoring unreadable cache at %s", path)
        return {}


def save_cache(dest: Path, cache: dict) -> None:
    path = _cache_path(dest)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Remote metadata
# ---------------------------------------------------------------------------
def fetch_remote_files(record_id: str, timeout: float) -> list[RemoteFile]:
    """Ask Zenodo for the file list and checksums; fall back to the embedded table."""
    url = ZENODO_API.format(record=record_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        files = []
        for entry in payload.get("files", []):
            checksum = str(entry.get("checksum", ""))
            md5 = checksum.split("md5:", 1)[1] if checksum.startswith("md5:") else None
            files.append(RemoteFile(name=entry["key"], md5=md5, size=entry.get("size")))
        if files:
            log.info("resolved %d files from the Zenodo API (record %s)", len(files), record_id)
            return sorted(files, key=lambda f: f.size or 0)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as exc:
        log.warning("could not reach the Zenodo API (%s); using embedded checksums", exc)

    return sorted(
        (RemoteFile(name=n, md5=md5, size=size) for n, (md5, size) in FALLBACK_FILES.items()),
        key=lambda f: f.size or 0,
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def md5_of(path: Path, label: str = "") -> str:
    h = hashlib.md5()  # noqa: S324 - matching Zenodo's published digest, not security
    size = path.stat().st_size
    with Progress(size, f"verifying {label or path.name}", unit="bytes", logger=log) as bar:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                h.update(block)
                bar.advance(len(block))
    return h.hexdigest()


def _open_ranged(url: str, offset: int, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset > 0:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def download_file(remote: RemoteFile, dest: Path, *, retries: int, timeout: float) -> Path:
    """Download one archive with resume. Returns the final path."""
    target = dest / remote.name
    part = dest / f"{remote.name}.part"
    dest.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if remote.size and offset > remote.size:
            log.warning("partial file is larger than expected; restarting %s", remote.name)
            part.unlink()
            offset = 0

        try:
            with _open_ranged(remote.url, offset, timeout) as resp:
                status = getattr(resp, "status", resp.getcode())

                if offset and status == 200:
                    # Server ignored the Range header; start over rather than
                    # append a second copy of the file onto the first.
                    log.warning("server ignored Range for %s; restarting", remote.name)
                    offset = 0
                    mode = "wb"
                elif offset and status == 206:
                    log.info("resuming %s at %s", remote.name, human_bytes(offset))
                    mode = "ab"
                else:
                    mode = "wb"

                declared = resp.headers.get("Content-Length")
                total = remote.size or ((int(declared) + offset) if declared else None)

                label = f"{remote.name}"
                with Progress(total, label, unit="bytes", logger=log) as bar:
                    bar.set(offset)
                    with open(part, mode) as fh:
                        while True:
                            block = resp.read(CHUNK)
                            if not block:
                                break
                            fh.write(block)
                            bar.advance(len(block))
                        fh.flush()
                        os.fsync(fh.fileno())

            actual = part.stat().st_size
            if remote.size and actual != remote.size:
                raise OSError(
                    f"size mismatch after download: got {actual}, expected {remote.size}"
                )

            os.replace(part, target)
            return target

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                # Range beyond EOF: the .part is already complete (or corrupt).
                log.warning("range not satisfiable for %s; verifying what we have", remote.name)
                if part.exists() and remote.size and part.stat().st_size == remote.size:
                    os.replace(part, target)
                    return target
                if part.exists():
                    part.unlink()
                continue
            backoff = min(60.0, 2.0**attempt)
            log.warning(
                "attempt %d/%d for %s failed: %s (retrying in %.0fs)",
                attempt,
                retries,
                remote.name,
                exc,
                backoff,
            )
            if attempt < retries:
                time.sleep(backoff)

    raise RuntimeError(f"failed to download {remote.name} after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _safe_members(zf: zipfile.ZipFile, dest: Path) -> list[zipfile.ZipInfo]:
    """Reject absolute paths, parent traversal and symlinks before extracting."""
    dest_resolved = dest.resolve()
    safe: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"refusing to extract unsafe path {name!r} from the archive")
        target = (dest_resolved / name).resolve()
        if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
            raise ValueError(f"refusing to extract {name!r} outside {dest_resolved}")
        # Mode bits 0xA000 mark a symlink; the archive should contain none.
        if (info.external_attr >> 16) & 0xF000 == 0xA000:
            raise ValueError(f"refusing to extract symlink {name!r} from the archive")
        safe.append(info)
    return safe


def extract_archive(archive: Path, dest: Path, *, force: bool = False) -> tuple[Path, int]:
    """Extract an archive into `dest`. Returns (split directory, .npy count)."""
    split_dir = dest / archive.stem
    with zipfile.ZipFile(archive) as zf:
        members = _safe_members(zf, dest)
        n_expected = sum(1 for m in members if m.filename.endswith(".npy"))

        if split_dir.exists() and not force:
            existing = sum(1 for _ in split_dir.rglob("*.npy"))
            if existing == n_expected:
                log.info("%s already extracted (%d frames)", archive.stem, existing)
                return split_dir, existing
            log.warning(
                "%s has %d frames on disk but the archive holds %d; re-extracting",
                archive.stem,
                existing,
                n_expected,
            )

        with Progress(len(members), f"extracting {archive.name}", logger=log) as bar:
            for info in members:
                zf.extract(info, dest)
                bar.advance()

    count = sum(1 for _ in split_dir.rglob("*.npy"))
    expected = EXPECTED_FILE_COUNTS.get(archive.stem)
    if expected is not None and count != expected:
        log.warning(
            "%s: extracted %d .npy files, expected %d. Preprocessing will still run, "
            "but the dataset may differ from the verified release.",
            archive.stem,
            count,
            expected,
        )
    return split_dir, count


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def process_one(
    remote: RemoteFile,
    dest: Path,
    cache: dict,
    *,
    force: bool,
    extract: bool,
    retries: int,
    timeout: float,
) -> dict:
    target = dest / remote.name
    entry = dict(cache.get(remote.name, {}))
    expected_md5 = remote.md5 or (FALLBACK_FILES.get(remote.name) or (None, None))[0]

    # -- acquire ------------------------------------------------------------
    need_download = True
    if target.exists() and not force:
        cached_ok = (
            entry.get("md5")
            and entry["md5"] == expected_md5
            and entry.get("size") == target.stat().st_size
        )
        if cached_ok:
            log.info("%s already downloaded and verified (cached)", remote.name)
            need_download = False
        else:
            digest = md5_of(target, remote.name)
            if expected_md5 and digest == expected_md5:
                log.info("%s already on disk and verified", remote.name)
                entry.update(md5=digest, size=target.stat().st_size)
                need_download = False
            else:
                log.warning("%s failed checksum verification; re-downloading", remote.name)

    if need_download:
        log.info("downloading %s (%s)", remote.name, human_bytes(remote.size or 0))
        target = download_file(remote, dest, retries=retries, timeout=timeout)
        digest = md5_of(target, remote.name)
        if expected_md5 and digest != expected_md5:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {remote.name}: got {digest}, expected {expected_md5}. "
                "The download is corrupt; re-run to try again."
            )
        entry.update(
            md5=digest,
            size=target.stat().st_size,
            downloaded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    # -- extract ------------------------------------------------------------
    if extract:
        split_dir, count = extract_archive(target, dest, force=force)
        entry["extracted"] = {
            "dir": split_dir.name,
            "n_npy": count,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    return entry


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.fetch_data",
        description="Download and extract the Mastcam novelty dataset (Zenodo 3732485).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dest", type=Path, default=None, help="destination (default: data/raw)")
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="restrict to one archive (repeatable), e.g. --only test_novel.zip",
    )
    p.add_argument("--force", action="store_true", help="re-download and re-extract everything")
    p.add_argument("--no-extract", action="store_true", help="download only, do not unzip")
    p.add_argument(
        "--delete-zips",
        action="store_true",
        help="remove archives after successful extraction to reclaim ~332 MB",
    )
    p.add_argument("--record-id", default=ZENODO_RECORD_ID, help="Zenodo record id")
    p.add_argument("--retries", type=int, default=int(os.environ.get("NOVUM_HTTP_RETRIES", 5)))
    p.add_argument("--timeout", type=float, default=float(os.environ.get("NOVUM_HTTP_TIMEOUT", 60)))
    p.add_argument("--list", action="store_true", help="show the remote file list and exit")
    p.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    global ZENODO_RECORD_ID
    ZENODO_RECORD_ID = args.record_id

    dest = args.dest or paths.raw_dir()
    dest.mkdir(parents=True, exist_ok=True)

    remotes = fetch_remote_files(args.record_id, args.timeout)
    if args.only:
        wanted = set(args.only)
        remotes = [r for r in remotes if r.name in wanted]
        unknown = wanted - {r.name for r in remotes}
        if unknown:
            log.error("unknown archive(s): %s", ", ".join(sorted(unknown)))
            return 2

    if args.list:
        for r in remotes:
            print(f"{r.name:26s} {human_bytes(r.size or 0):>10s}  md5:{r.md5}")
        return 0

    if not remotes:
        log.error("no archives selected")
        return 2

    total_bytes = sum(r.size or 0 for r in remotes)
    log.info(
        "fetching %d archive(s), %s total, into %s",
        len(remotes),
        human_bytes(total_bytes),
        paths.rel(dest),
    )

    cache = load_cache(dest)
    failures: list[str] = []

    for remote in remotes:
        try:
            cache[remote.name] = process_one(
                remote,
                dest,
                cache,
                force=args.force,
                extract=not args.no_extract,
                retries=args.retries,
                timeout=args.timeout,
            )
            save_cache(dest, cache)
            if args.delete_zips and not args.no_extract:
                (dest / remote.name).unlink(missing_ok=True)
                log.info("removed %s after extraction", remote.name)
        except Exception as exc:  # noqa: BLE001 - one bad archive must not sink the rest
            log.error("FAILED %s: %s", remote.name, exc)
            failures.append(remote.name)

    save_cache(dest, cache)

    if failures:
        log.error("%d archive(s) failed: %s", len(failures), ", ".join(failures))
        log.error("re-run the same command; completed downloads are skipped and partials resume")
        return 1

    log.info("all archives present and verified in %s", paths.rel(dest))
    log.info("next: python -m scripts.preprocess")
    return 0


if __name__ == "__main__":
    sys.exit(main())
