"""The mission stream: what the rover captured, in the order it captured it.

The static evaluation asks "is this frame unlike the training set". The mission
asks a strictly harder question: "is this frame unlike what we have seen *so
far*". This module builds the chronological stream that makes the second
question answerable -- every frame carries its sol, its class, and its group
(natural / rover / typical), and nothing is ever reordered except by sol.

FRAME SOURCE. By default the mission is `test_typical` + `test_novel_all`: the
same 856 frames the static evaluation scores, so simulation numbers and static
numbers are about the same data and can be compared directly. Composition:

    426 typical, 169 natural-novel, 261 rover-novel

The 169 natural frames are the denominator of science yield -- the fraction of
Mars-made novelty that actually reached the ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core import taxonomy
from core.dataset import load_split
from core.manifest import ManifestRow

#: Splits that make up the mission, in the order they are concatenated.
DEFAULT_MISSION_SPLITS = ("test_typical", "test_novel_all")

GROUP_TYPICAL = "typical"


@dataclass(frozen=True)
class MissionFrame:
    """One captured frame, with everything the simulator needs to judge it."""

    index: int          # position in the mission array
    sol: int
    split: str
    class_: str
    source_filename: str
    group: str          # natural | rover | typical | excluded
    bits: float         # downlink cost after compression

    @property
    def is_novel(self) -> bool:
        return self.split != "test_typical"

    @property
    def is_natural(self) -> bool:
        return self.group == taxonomy.GROUP_NATURAL

    @property
    def is_rover(self) -> bool:
        return self.group == taxonomy.GROUP_ROVER


@dataclass
class MissionStream:
    """Frames in sol order, plus the array they live in."""

    frames: list[MissionFrame]
    array: np.ndarray                     # (N, 64, 64, 6) in mission order
    rows: list[ManifestRow]               # manifest rows, mission order
    splits: tuple[str, ...] = DEFAULT_MISSION_SPLITS

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def sols(self) -> np.ndarray:
        return np.array([f.sol for f in self.frames], dtype=np.int64)

    @property
    def n_natural(self) -> int:
        return sum(1 for f in self.frames if f.is_natural)

    @property
    def n_rover(self) -> int:
        return sum(1 for f in self.frames if f.is_rover)

    @property
    def n_typical(self) -> int:
        return sum(1 for f in self.frames if f.group == GROUP_TYPICAL)

    def composition(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for frame in self.frames:
            out[frame.group] = out.get(frame.group, 0) + 1
        return dict(sorted(out.items()))

    def batch(self, indices) -> np.ndarray:
        """Frames by mission index, as a contiguous float32 batch."""
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, *self.array.shape[1:]), dtype=np.float32)
        # Sorted read for memmap locality; caller gets them back in its order.
        order = np.argsort(idx, kind="stable")
        gathered = np.asarray(self.array[idx[order]], dtype=np.float32)
        restore = np.empty_like(order)
        restore[order] = np.arange(len(order))
        return gathered[restore]


def build_mission(
    splits: tuple[str, ...] = DEFAULT_MISSION_SPLITS,
    *,
    bits_per_sample: int = 8,
    compression_ratio: float = 4.0,
    texture_bits: bool = True,
) -> MissionStream:
    """Load the splits, label every frame, and sort the whole thing by sol.

    Per-frame bit costs come from the texture model in core.budgets when
    `texture_bits` is on: busy scenes compress worse, which is what makes
    novelty-per-bit a different ordering from novelty alone. Turn it off for a
    flat cost model.
    """
    from core.budgets import estimate_bits_from_frames, estimate_frame_bits

    arrays: list[np.ndarray] = []
    rows: list[ManifestRow] = []
    origin: list[str] = []
    for split in splits:
        data = load_split(split)
        arrays.append(np.asarray(data.array))
        rows.extend(data.rows)
        origin.extend([split] * len(data.rows))

    array = np.concatenate(arrays, axis=0)
    del arrays

    if texture_bits:
        bits = estimate_bits_from_frames(
            array, bits_per_sample=bits_per_sample, compression_ratio=compression_ratio
        )
    else:
        flat = estimate_frame_bits(
            array.shape[1:], bits_per_sample=bits_per_sample, compression_ratio=compression_ratio
        )
        bits = np.full(len(array), flat, dtype=np.float64)

    # Sort by sol, then by original order. Frames with no parseable sol sort
    # last -- an unknown capture date must never seed the "seen so far" state.
    keys = np.array(
        [np.iinfo(np.int64).max if r.sol is None else r.sol for r in rows], dtype=np.int64
    )
    order = np.lexsort((np.arange(len(rows)), keys))

    frames: list[MissionFrame] = []
    for new_index, old_index in enumerate(order):
        row = rows[old_index]
        split = origin[old_index]
        if split == "test_typical":
            group = GROUP_TYPICAL
        else:
            groups = taxonomy.group_for_class_field(row.class_)
            # A frame counts as natural if any label is natural; the archive has
            # no natural/rover straddlers, but the rule is defined either way.
            group = (
                taxonomy.GROUP_NATURAL
                if taxonomy.GROUP_NATURAL in groups
                else (taxonomy.GROUP_ROVER if taxonomy.GROUP_ROVER in groups
                      else taxonomy.GROUP_EXCLUDED)
            )
        frames.append(
            MissionFrame(
                index=new_index,
                sol=int(row.sol) if row.sol is not None else -1,
                split=split,
                class_=row.class_,
                source_filename=row.source_filename,
                group=group,
                bits=float(bits[old_index]),
            )
        )

    return MissionStream(
        frames=frames,
        array=np.ascontiguousarray(array[order]),
        rows=[rows[i] for i in order],
        splits=tuple(splits),
    )


@dataclass
class BufferedFrame:
    """A captured frame awaiting a downlink decision."""

    frame: MissionFrame
    captured_sol: int
    score: float | None = None          # novelty, once the model has paid for it
    prefilter: float | None = None      # cheap triage statistic
    times_considered: int = 0

    def age_sols(self, current_sol: int) -> int:
        return max(0, current_sol - self.captured_sol)


@dataclass
class FrameBuffer:
    """Bounded, age-limited store of frames not yet transmitted.

    Q2 of the design questions, answered: unselected frames are RETAINED. A
    frame stays a candidate for later windows until it exceeds
    `max_age_sols`, then it expires unsent. That is the real tension the
    simulator exists to expose -- transmit now, or gamble that a later window
    has room before the frame ages out. Expiries are counted, never silent.

    `capacity` is onboard storage: when full, the OLDEST frames are dropped
    first (they are closest to expiry anyway).
    """

    max_age_sols: int = 200
    capacity: int | None = None
    items: dict[int, BufferedFrame] = field(default_factory=dict)
    n_expired: int = 0
    n_evicted: int = 0
    expired_natural: int = 0
    expired_rover: int = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, frame: MissionFrame, sol: int) -> None:
        self.items[frame.index] = BufferedFrame(frame=frame, captured_sol=sol)

    def expire(self, current_sol: int) -> list[BufferedFrame]:
        """Drop frames older than the age limit. Returns what was dropped."""
        dropped = [
            item for item in self.items.values()
            if item.age_sols(current_sol) > self.max_age_sols
        ]
        for item in dropped:
            del self.items[item.frame.index]
            self.n_expired += 1
            if item.frame.is_natural:
                self.expired_natural += 1
            elif item.frame.is_rover:
                self.expired_rover += 1
        return dropped

    def enforce_capacity(self) -> list[BufferedFrame]:
        """Evict oldest-first when onboard storage is full."""
        if self.capacity is None or len(self.items) <= self.capacity:
            return []
        ordered = sorted(self.items.values(), key=lambda i: (i.captured_sol, i.frame.index))
        evicted = ordered[: len(self.items) - self.capacity]
        for item in evicted:
            del self.items[item.frame.index]
            self.n_evicted += 1
            if item.frame.is_natural:
                self.expired_natural += 1
            elif item.frame.is_rover:
                self.expired_rover += 1
        return evicted

    def remove(self, indices) -> None:
        for index in indices:
            self.items.pop(int(index), None)

    def candidates(self) -> list[BufferedFrame]:
        """Buffered frames in capture order -- the order FIFO transmits in."""
        return sorted(self.items.values(), key=lambda i: (i.captured_sol, i.frame.index))
