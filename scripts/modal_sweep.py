"""Run the (tier x seed) training sweep on Modal and bring the results home.

    make sweep-modal

Nine training runs -- three tiers, three seeds. The snapdragon tier alone is
about nine minutes each on a laptop, so the sweep is roughly forty minutes of a
machine you would rather be using, and it competes with everything else on it.
Modal runs the tiers in parallel on cores that are doing nothing else.

The results are identical in every way that is reported. Training is
deterministic given a seed, so a rerun on the SAME machine reproduces the
weights bit for bit; across machines the last digits differ with the BLAS
implementation, and the metrics agree to more decimal places than anyone
publishes. What comes back is the same science, computed somewhere else.

PROVENANCE. The container has no `.git`, so it is told which commit it is
building from via NOVUM_GIT_COMMIT (see core.provenance). Without that every
sidecar would record `git_commit: null` -- untraceable, which is the failure the
field exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

from scripts.modal_console import (
    CPU_COUNT,
    VOLUME_MOUNT,
    VOLUME_NAME,
    image,
)

APP_NAME = "novum-sweep"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIERS = ("rad750", "myriad", "snapdragon")
SEEDS = (0, 1, 2)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _prepare_container(git: dict[str, str]) -> None:
    """Point core.paths at the volume and declare the commit being built."""
    import os
    import sys as _sys

    _sys.path.insert(0, "/app")
    os.environ["NOVUM_DATA_DIR"] = str(VOLUME_MOUNT)
    os.environ["NOVUM_ARTIFACTS_DIR"] = "/tmp/artifacts"
    os.environ["NOVUM_RUNS_DIR"] = "/tmp/runs"
    os.environ.update(git)
    # One thread per process: several tiers run concurrently and BLAS would
    # otherwise try to fill the machine from each of them at once.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(max(1, CPU_COUNT // 3)))

    # The volume is read-only in spirit; artifacts are produced into /tmp and
    # returned, so a failed run cannot leave a half-written weight file behind
    # for the next one to pick up.
    for path in ("/tmp/artifacts", "/tmp/artifacts/metrics", "/tmp/runs"):
        os.makedirs(path, exist_ok=True)
    # scripts.simulate reads tier configs from artifacts/<tier>.json, so the
    # reference artifacts must be visible under the artifacts dir.
    for name in os.listdir(VOLUME_MOUNT / "artifacts"):
        src = VOLUME_MOUNT / "artifacts" / name
        dst = Path("/tmp/artifacts") / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())


@app.function(image=image, volumes={str(VOLUME_MOUNT): volume}, cpu=6, memory=16384,
              timeout=60 * 45)
def train_tier(tier: str, seeds: list[int], git: dict[str, str]) -> dict[str, bytes]:
    """Train one tier at every seed, evaluate each, return the files.

    One tier per container so the three run concurrently: snapdragon dominates
    the wall clock and there is no reason for rad750 to wait behind it.
    """
    import subprocess

    _prepare_container(git)
    out: dict[str, bytes] = {}

    for seed in seeds:
        artifact = Path("/tmp/artifacts") / f"{tier}.npz"
        subprocess.run(
            [
                sys.executable, "-m", "scripts.train",
                "--config", f"/app/configs/tier_{tier}.yaml",
                "--out", str(artifact), "--seed", str(seed),
            ],
            cwd="/app", check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "scripts.evaluate", "--artifact", str(artifact)],
            cwd="/app", check=True,
        )
        # Seed 0 is the published artifact; the others exist for the spread.
        if seed == seeds[0]:
            out[f"artifacts/{tier}.npz"] = artifact.read_bytes()
            out[f"artifacts/{tier}.json"] = artifact.with_suffix(".json").read_bytes()
            metrics = Path("/tmp/artifacts/metrics") / f"{tier}.json"
            if metrics.is_file():
                out[f"artifacts/metrics/{tier}.json"] = metrics.read_bytes()
        # Every seed's metrics feed the +/- spread in results/RESULTS.md.
        for path in Path("/tmp/runs").rglob("*.json"):
            out[f"runs/{path.relative_to('/tmp/runs')}"] = path.read_bytes()

    return out


@app.local_entrypoint()
def main(seeds: str = "0,1,2", out: str = "") -> None:
    """`modal run scripts/modal_sweep.py` -- sweep remotely, write locally."""
    import subprocess as sp

    root = Path(out) if out else PROJECT_ROOT
    commit = sp.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                    capture_output=True, text=True, check=True).stdout.strip()
    branch = sp.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT,
                    capture_output=True, text=True, check=True).stdout.strip()
    dirty = sp.run(
        ["git", "status", "--porcelain", "--", ".",
         ":(exclude)artifacts", ":(exclude)runs", ":(exclude)results",
         ":(exclude)web/public/data"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()

    if dirty:
        print("REFUSING: the source tree is dirty, so the sweep would produce")
        print("artifacts that cannot be tied to a commit. Commit first:")
        for line in dirty.splitlines()[:10]:
            print(f"  {line}")
        raise SystemExit(1)

    git = {
        "NOVUM_GIT_COMMIT": commit,
        "NOVUM_GIT_BRANCH": branch,
        "NOVUM_GIT_DIRTY": "false",
    }
    print(f"sweeping from {commit[:8]} on {branch} (clean)")

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    written = 0
    for files in train_tier.starmap([(t, seed_list, git) for t in TIERS]):
        for name, payload in sorted(files.items()):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            written += 1
            print(f"  {name}  {len(payload) / 1e6:.2f} MB")
    print(f"{written} file(s) written under {root}")


if __name__ == "__main__":
    raise SystemExit(
        "run this with:  modal run scripts/modal_sweep.py"
    )
