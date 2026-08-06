"""Preprocessing must stream: peak RSS bounded, and independent of dataset size.

The target server has 2 GB of RAM. The obvious implementation -- assigning into
a `np.lib.format.open_memmap` array -- dirties a page per frame and the kernel
holds those pages resident until writeback, so building `train_typical` peaked
at **914 MiB for an 872 MiB array**: the entire output, in RAM. That is an
OOM kill on a 2 GB box.

`StreamingArrayWriter` preallocates the file, drops the mapping, and appends
through an ordinary buffered handle. These tests are what stop that regressing.

Note on what each test can actually catch: at the frame counts a unit test can
afford, the *absolute* budget check passes even for the broken implementation
(600 frames only dirty ~59 MiB). The scaling test is the real regression
detector -- it measures the marginal RSS cost per frame, which is ~0 when
streaming and ~96 KiB when not, and that gap is unmissable at any size.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from core.dataset import FRAME_SHAPE
from scripts.preprocess import RSS_BUDGET_BYTES, StreamingArrayWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAME_BYTES_F32 = int(np.prod(FRAME_SHAPE)) * 4  # 98,304

#: Marginal RSS allowed per frame, chosen to sit unambiguously between the two
#: regimes rather than near either:
#:
#:   streaming      ~5 KiB/frame   measured -- the per-frame Sample and
#:                                 ManifestRow metadata, which is genuinely O(n)
#:                                 and costs ~50 MiB across the full dataset
#:   not streaming  ~96 KiB/frame  a whole float32 frame held resident
#:
#: 20 KiB is 4x above the metadata cost and 5x below the failure mode, so this
#: neither flakes on allocator noise nor misses a real regression.
MAX_MARGINAL_RSS_PER_FRAME = 20 * 1024

_RUNNER = textwrap.dedent(
    """
    import sys
    from scripts.preprocess import main
    from core.provenance import peak_rss_bytes
    rc = main(sys.argv[1:])
    print("PEAK_RSS_BYTES", peak_rss_bytes())
    sys.exit(rc)
    """
)


def _write_raw_split(root: Path, split: str, n: int) -> Path:
    """Create `n` float64 frames, matching the real archive's dtype and shape."""
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 255, size=FRAME_SHAPE).round()  # float64, like the real data
    for i in range(n):
        np.save(directory / f"mcam{i:05d}_R0_sol{i % 200:04d}_{i}.npy", frame)
    return directory


def _run_preprocess(raw_dir: Path, out_dir: Path, split: str) -> tuple[int, int]:
    """Run preprocess in a clean subprocess. Returns (exit code, peak RSS bytes)."""
    result = subprocess.run(
        [
            sys.executable, "-c", _RUNNER,
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out_dir),
            "--only", split,
            "--force",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"preprocess failed ({result.returncode}):\n{result.stderr[-3000:]}")
    marker = [ln for ln in result.stdout.splitlines() if ln.startswith("PEAK_RSS_BYTES")]
    assert marker, f"no RSS marker in output:\n{result.stdout}\n{result.stderr[-2000:]}"
    return result.returncode, int(marker[0].split()[1])


# ---------------------------------------------------------------------------
# The memory contract
# ---------------------------------------------------------------------------
def test_peak_rss_stays_within_the_streaming_budget(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw_split(raw, "test_typical", 400)

    _, peak = _run_preprocess(raw, tmp_path / "processed", "test_typical")

    assert peak < RSS_BUDGET_BYTES, (
        f"preprocessing peaked at {peak / 1024**2:.0f} MiB, over the "
        f"{RSS_BUDGET_BYTES / 1024**2:.0f} MiB budget"
    )


def test_memory_does_not_grow_with_dataset_size(tmp_path: Path) -> None:
    """The load-bearing test: marginal RSS per frame must be ~0, not ~96 KiB."""
    small_n, large_n = 150, 600

    raw_small = tmp_path / "raw_small"
    _write_raw_split(raw_small, "test_typical", small_n)
    _, peak_small = _run_preprocess(raw_small, tmp_path / "out_small", "test_typical")

    raw_large = tmp_path / "raw_large"
    _write_raw_split(raw_large, "test_typical", large_n)
    _, peak_large = _run_preprocess(raw_large, tmp_path / "out_large", "test_typical")

    extra_frames = large_n - small_n
    extra_output = extra_frames * FRAME_BYTES_F32
    marginal = max(0, peak_large - peak_small) / extra_frames

    assert marginal < MAX_MARGINAL_RSS_PER_FRAME, (
        f"peak RSS grew {(peak_large - peak_small) / 1024**2:.1f} MiB while writing "
        f"{extra_output / 1024**2:.1f} MiB more output "
        f"({marginal / 1024:.1f} KiB/frame). Preprocessing is holding the output in "
        f"memory instead of streaming it; see StreamingArrayWriter."
    )

    # The whole point: extrapolated to the real 9,302-frame split, still bounded.
    implied = peak_small + marginal * 9302
    assert implied < RSS_BUDGET_BYTES, (
        f"extrapolated peak RSS for train_typical is {implied / 1024**2:.0f} MiB"
    )


# ---------------------------------------------------------------------------
# StreamingArrayWriter itself
# ---------------------------------------------------------------------------
def test_writer_output_is_identical_to_np_save(tmp_path: Path) -> None:
    """Byte-for-byte equality with the obvious implementation, or it is not a drop-in."""
    rng = np.random.default_rng(1)
    frames = rng.uniform(0, 255, size=(7, *FRAME_SHAPE)).round()  # float64 source

    streamed = tmp_path / "streamed.npy"
    writer = StreamingArrayWriter(streamed, len(frames))
    with writer:
        for frame in frames:
            writer.append(frame)
    writer.finalize()

    reference = tmp_path / "reference.npy"
    np.save(reference, frames.astype(np.float32))

    assert streamed.read_bytes() == reference.read_bytes()


def test_writer_round_trips_through_a_memmap_read(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    frames = rng.uniform(0, 255, size=(5, *FRAME_SHAPE)).astype(np.float64)

    path = tmp_path / "a.npy"
    writer = StreamingArrayWriter(path, len(frames))
    with writer:
        for frame in frames:
            writer.append(frame)
    writer.finalize()

    loaded = np.load(path, mmap_mode="r")
    assert loaded.shape == (5, *FRAME_SHAPE)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(np.asarray(loaded), frames.astype(np.float32))


def test_writer_trims_when_frames_are_skipped(tmp_path: Path) -> None:
    """The --skip-bad path: the array must shrink to what was actually written."""
    rng = np.random.default_rng(3)
    frames = rng.uniform(0, 255, size=(3, *FRAME_SHAPE)).astype(np.float64)

    path = tmp_path / "short.npy"
    writer = StreamingArrayWriter(path, 10)  # planned 10, only 3 arrive
    with writer:
        for frame in frames:
            writer.append(frame)
    writer.finalize()

    loaded = np.load(path, mmap_mode="r")
    assert loaded.shape == (3, *FRAME_SHAPE)
    np.testing.assert_allclose(np.asarray(loaded), frames.astype(np.float32))


def test_writer_refuses_to_overrun_its_allocation(tmp_path: Path) -> None:
    writer = StreamingArrayWriter(tmp_path / "b.npy", 1)
    frame = np.zeros(FRAME_SHAPE, dtype=np.float64)
    with writer:
        writer.append(frame)
        with pytest.raises(RuntimeError, match="sized 1"):
            writer.append(frame)


def test_writer_rejects_use_outside_its_context_manager(tmp_path: Path) -> None:
    writer = StreamingArrayWriter(tmp_path / "c.npy", 1)
    with pytest.raises(RuntimeError, match="outside its context manager"):
        writer.append(np.zeros(FRAME_SHAPE, dtype=np.float64))


def test_writer_never_maps_the_output_while_writing(tmp_path: Path) -> None:
    """Guard the mechanism, not just the symptom: no live mmap during append.

    A future refactor that "helpfully" reintroduces a memmap here would pass the
    round-trip tests and quietly restore the 914 MiB behaviour.
    """
    writer = StreamingArrayWriter(tmp_path / "d.npy", 2)
    with writer:
        assert not isinstance(writer._fh, np.memmap)
        writer.append(np.zeros(FRAME_SHAPE, dtype=np.float64))
        # The handle is a plain buffered binary file, not an array.
        assert hasattr(writer._fh, "write") and not hasattr(writer._fh, "flush_to_disk")
        assert writer._fh.mode == "rb+"
    writer.finalize()
