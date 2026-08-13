"""Build the console grid on Modal instead of on the laptop.

    make console-modal-upload     # once: push data + artifacts to a Volume
    make console-modal            # ~3 minutes, writes web/public/data/

WHY. The grid is 324 replays and the expensive ones are online adaptation on the
autoencoder tiers, which refit the model seven times per run. That is a few
minutes on sixteen idle cores and considerably worse on a laptop that is also
running a dev server, a test suite and a browser -- ask me how I know. Moving it
to a container with real cores makes `make console` something you run without
planning your afternoon around it.

The replay logic is NOT duplicated here. This module uploads inputs, calls the
same `build_grid` the local path calls, and writes the same files. If the two
ever disagree, that is a bug rather than a design.

The container needs the training stack (torch, for the autoencoder tiers), which
is exactly why this lives in `scripts/` and not in `api/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

APP_NAME = "novum-console"

#: Inputs live on a Volume rather than being shipped per call: 92 MB of frames
#: and weights that do not change between grid builds.
VOLUME_NAME = "novum-data"
VOLUME_MOUNT = Path("/vol")

#: What `console-modal-upload` pushes, and where the container expects it.
#:
#: `validation_typical` is NOT optional even though no mission frame comes from
#: it. An online refit is a real model fit, and the autoencoder tiers early-stop
#: against that split; without it they silently hold out a tenth of the refit
#: pool instead, and every online cell in the grid is then a different run from
#: the one `make console` produces locally. It costs 136 MB and it buys the
#: local and remote builds agreeing.
UPLOAD_PATHS = (
    # 872 MB, and only the training sweep needs it -- but without it
    # `scripts.train` fails inside the container with a split it cannot open, so
    # it is part of the volume rather than an optional extra.
    ("data/processed/train_typical.npy", "processed/train_typical.npy"),
    ("data/processed/test_typical.npy", "processed/test_typical.npy"),
    ("data/processed/test_novel_all.npy", "processed/test_novel_all.npy"),
    ("data/processed/validation_typical.npy", "processed/validation_typical.npy"),
    ("data/processed/manifest.csv", "processed/manifest.csv"),
    ("data/processed/meta.json", "processed/meta.json"),
    ("artifacts/rad750.npz", "artifacts/rad750.npz"),
    ("artifacts/rad750.json", "artifacts/rad750.json"),
    ("artifacts/myriad.npz", "artifacts/myriad.npz"),
    ("artifacts/myriad.json", "artifacts/myriad.json"),
    ("artifacts/snapdragon.npz", "artifacts/snapdragon.npz"),
    ("artifacts/snapdragon.json", "artifacts/snapdragon.json"),
)

#: The lanes are independent and CPU-bound. Sixteen is comfortably more than
#: the 54 lanes need to finish in one wave of the slow ones.
CPU_COUNT = 16

PROJECT_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    # CPU wheels only. NOVUM is CPU-by-design and a CUDA runtime would triple
    # the image for nothing.
    .pip_install(
        "numpy>=1.26",
        "pyyaml>=6.0",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "scikit-learn>=1.3",
        "torch>=2.2",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir(PROJECT_ROOT / "core", "/app/core")
    .add_local_dir(PROJECT_ROOT / "sim", "/app/sim")
    .add_local_dir(PROJECT_ROOT / "scripts", "/app/scripts")
    .add_local_dir(PROJECT_ROOT / "configs", "/app/configs")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT): volume},
    cpu=CPU_COUNT,
    memory=16384,
    timeout=60 * 30,
)
def build(quick: bool = False, skip_atlas: bool = False) -> dict[str, bytes]:
    """Run the whole console build in the container; return the files.

    Returns raw bytes rather than writing to the Volume so the caller gets the
    output in one round trip and there is no second step to forget. The payload
    is about 8 MB.
    """
    import os
    import sys as _sys

    _sys.path.insert(0, "/app")
    # core.paths resolves everything from these, so the container's layout is a
    # configuration detail rather than a fork of the path logic.
    os.environ["NOVUM_DATA_DIR"] = str(VOLUME_MOUNT)
    os.environ["NOVUM_ARTIFACTS_DIR"] = str(VOLUME_MOUNT / "artifacts")

    from core.logging_utils import setup_logging
    from scripts.build_console import (
        ADAPTATIONS,
        BUDGETS,
        _json_bytes,
        build_grid,
        build_mission_payload,
    )
    from sim.mission import build_mission
    from sim.window import SimConfig, plan_windows

    setup_logging("INFO", force=True)

    mission = build_mission()
    windows = plan_windows(mission.rows, sols_per_window=SimConfig().sols_per_window)

    out: dict[str, bytes] = {
        "mission.json": _json_bytes(build_mission_payload(mission, windows))
    }

    if not skip_atlas:
        from core.thumbnails import build_atlas

        png, atlas_meta = build_atlas(mission.array)
        out["atlas.png"] = png
        out["atlas.json"] = _json_bytes(atlas_meta)

    grid = build_grid(
        budgets=BUDGETS[:2] if quick else BUDGETS,
        adaptations=ADAPTATIONS[:1] if quick else ADAPTATIONS,
        quick=quick,
        workers=CPU_COUNT,
    )
    out["grid.json"] = _json_bytes(grid)
    return out


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(quick: bool = False, skip_atlas: bool = False, out: str = "") -> None:
    """`modal run scripts/modal_console.py` — build remotely, write locally."""
    target = Path(out) if out else PROJECT_ROOT / "web" / "public" / "data"
    target.mkdir(parents=True, exist_ok=True)

    files = build.remote(quick=quick, skip_atlas=skip_atlas)
    for name, payload in sorted(files.items()):
        path = target / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        print(f"  {_display(path)}  {len(payload) / 1e6:.1f} MB")
    print(f"console data -> {_display(target)}")


def _display(path: Path) -> str:
    """Project-relative when it can be, absolute otherwise -- `--out` may point
    anywhere, and a crash while printing a success message is a silly way to
    lose a build that already succeeded."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _remote_files() -> set[str]:
    """Everything already on the volume, as posix paths."""
    found: set[str] = set()
    for top in ("processed", "artifacts"):
        try:
            for entry in volume.listdir(top, recursive=True):
                found.add(entry.path.lstrip("/"))
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
    return found


def upload(argv: list[str] | None = None) -> int:
    """Push the inputs to the Volume. Idempotent: run it as often as you like.

    Skipping what is already there is not just a speed optimisation.
    `batch_upload` commits ATOMICALLY, so a batch containing one already-present
    file is rejected in full -- and because the per-file logging happens while
    the batch is being queued rather than when it lands, a failed upload prints
    exactly like a successful one. That is how a 136 MB file silently failed to
    arrive here and a whole grid got built against the wrong data.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.modal_console",
        description="Upload frames and artifacts to the Modal volume.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-upload files already present."
    )
    args = parser.parse_args(argv)

    missing = [src for src, _ in UPLOAD_PATHS if not (PROJECT_ROOT / src).is_file()]
    if missing:
        print("missing locally, cannot upload:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nRun `make data` and `make train` first.", file=sys.stderr)
        return 1

    present = set() if args.force else _remote_files()
    todo = [(src, dest) for src, dest in UPLOAD_PATHS if dest not in present]
    for _src, dest in UPLOAD_PATHS:
        if dest in present:
            print(f"  skip    {dest}  (already on the volume)")

    if todo:
        total = sum((PROJECT_ROOT / src).stat().st_size for src, _ in todo)
        print(f"uploading {len(todo)} file(s), {total / 1e6:.1f} MB")
        with volume.batch_upload(force=args.force) as batch:
            for src, dest in todo:
                batch.put_file(PROJECT_ROOT / src, dest)

    # Verify against the volume rather than against having called put_file.
    landed = _remote_files()
    absent = [dest for _src, dest in UPLOAD_PATHS if dest not in landed]
    if absent:
        print("\nFAILED -- these never arrived:", file=sys.stderr)
        for dest in absent:
            print(f"  {dest}", file=sys.stderr)
        return 1

    print(f"volume {VOLUME_NAME!r} has all {len(UPLOAD_PATHS)} inputs")
    return 0


if __name__ == "__main__":
    sys.exit(upload())
