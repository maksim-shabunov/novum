"""Downlink window simulator: replay the mission, sol by sol, under both budgets.

The static evaluation scores a frame against the whole training set. This
replays the mission instead: frames arrive in sol order, a relay window opens
every so often with a finite number of bits, and between windows the flight
processor has a finite number of cycles. "Novel" recovers its real meaning --
unlike the terrain seen *so far*, not unlike a training set that already
contains the whole mission.

Expect lower numbers here than in the static evaluation. That is the honest
measurement, not a regression.

THE THREE DESIGN QUESTIONS, ANSWERED
------------------------------------
Q1 Does the model refit as new terrain arrives, or stay frozen?
   Both, switchable, because the comparison is the interesting experiment.
   `frozen` (default) is trained on the ground and uplinked, never changes.
   `online` bootstraps on the first `bootstrap_sols` sols and refits on a
   cadence from frames captured so far -- with NO labels, since the rover has
   none. Note the caveat frozen carries: its training set (`train_typical`)
   spans the entire mission, including sols the rover has not reached yet, so
   frozen is optimistic by construction. `online` is the one that answers "how
   does a detector behave on terrain nobody has seen before".

Q2 Are unselected frames discarded or retained?
   Retained, in an age-limited buffer (`sim.mission.FrameBuffer`). A frame not
   selected stays a candidate until it passes `buffer_max_age_sols`, then it
   expires unsent and is counted. Transmit now, or gamble on a later window.

Q3 Does a downlinked frame's label feed back into the model?
   No, not in the baseline -- there is no ground truth onboard and assuming one
   would make the numbers dishonest. `ground_feedback: true` (off by default,
   kept out of headline results) simulates a low-bandwidth uplink of expert
   corrections after each window: confirmed-novel frames are removed from the
   pool the model refits "typical" on. It requires `adaptation: online`,
   because a frozen model has nothing to feed back into.

THE COMPUTE BUDGET ACTUALLY BINDS
---------------------------------
Scoring a frame costs cycles. When the buffer holds more frames than the cycle
budget can score, the system must decide what to even look at, using the cheap
prefilter in `sim.prefilter` (~14% of a PCA score, ~1.3% of a myriad score).
Frames that never get scored cannot be selected by a score-based policy; they
are counted in `n_unscored` and reported. This second-order scarcity is a real
part of the problem and is not hand-waved.

Scores are cached: a frame scored in window 7 and still buffered in window 8
is not re-scored, and not re-charged. That is what the flight software would
do. A refit invalidates the cache, and the re-scoring is charged again.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from core.budgets import BudgetSpec
from core.logging_utils import get_logger
from core.manifest import ManifestRow

from . import policy, prefilter
from .mission import FrameBuffer, MissionStream

log = get_logger("novum.sim")

ADAPTATION_MODES = ("frozen", "online")


@dataclass(frozen=True)
class DownlinkWindow:
    """One relay pass: a sol range and the two budgets available for it."""

    index: int
    first_sol: int
    last_sol: int
    budget: BudgetSpec
    frame_indices: tuple[int, ...] = field(default=())

    @property
    def n_candidates(self) -> int:
        return len(self.frame_indices)

    @property
    def sol_span(self) -> int:
        return self.last_sol - self.first_sol + 1

    def __repr__(self) -> str:
        return (
            f"DownlinkWindow(#{self.index}, sols {self.first_sol}-{self.last_sol}, "
            f"{self.n_candidates} candidates, {self.budget.bits:.0f} bits)"
        )


def chronological_order(rows: Sequence[ManifestRow]) -> np.ndarray:
    """Row indices in mission order: ascending sol, then manifest index.

    Frames whose filename carried no parseable sol sort **last**. Sorting them
    first would seed the "terrain seen so far" baseline with frames of unknown
    date, which is exactly the kind of quiet mistake that makes a replay look
    better than it is.
    """
    sols = np.array(
        [np.iinfo(np.int64).max if r.sol is None else r.sol for r in rows], dtype=np.int64
    )
    return np.lexsort((np.arange(len(rows)), sols))


def plan_windows(
    rows: Sequence[ManifestRow],
    *,
    sols_per_window: int = 10,
    bits_per_window: float = 8_000_000.0,
    cycles_per_window: float = 2_000_000_000.0,
) -> list[DownlinkWindow]:
    """Group frames into fixed-length sol windows, in mission order.

    Frames with no parseable sol are collected into a final window rather than
    dropped, so nothing silently disappears from the replay.
    """
    if sols_per_window < 1:
        raise ValueError(f"sols_per_window must be >= 1, got {sols_per_window}")
    if not rows:
        return []

    order = chronological_order(rows)
    budget = BudgetSpec(bits=bits_per_window, cycles=cycles_per_window)

    dated: list[tuple[int, int]] = []  # (sol, row index)
    undated: list[int] = []
    for position in order:
        row = rows[int(position)]
        if row.sol is None:
            undated.append(int(position))
        else:
            dated.append((int(row.sol), int(position)))

    windows: list[DownlinkWindow] = []
    if dated:
        first_sol = dated[0][0]
        current: list[int] = []
        current_start = first_sol
        for sol, position in dated:
            if sol - current_start >= sols_per_window and current:
                windows.append(
                    DownlinkWindow(
                        index=len(windows),
                        first_sol=current_start,
                        last_sol=rows[current[-1]].sol or current_start,
                        budget=budget,
                        frame_indices=tuple(current),
                    )
                )
                current = []
                current_start = sol
            current.append(position)
        if current:
            windows.append(
                DownlinkWindow(
                    index=len(windows),
                    first_sol=current_start,
                    last_sol=rows[current[-1]].sol or current_start,
                    budget=budget,
                    frame_indices=tuple(current),
                )
            )

    if undated:
        windows.append(
            DownlinkWindow(
                index=len(windows),
                first_sol=-1,
                last_sol=-1,
                budget=budget,
                frame_indices=tuple(undated),
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Everything that defines a run. Hashable into the summary for provenance.

    BUDGETS ARE SCALED TO THE ARRIVAL RATE, not taken as absolutes. The shipped
    tier configs allot 8 Mbit per window = 162 frames, but only ~27 frames
    arrive per window across this 856-frame mission: an absolute budget would
    never bind and the simulation would be theatre. `downlink_fraction` sets the
    per-window bit budget to that fraction of the bits arriving per window, so
    scarcity is real and stated. Absolute overrides are available for a study
    that wants them.
    """

    sols_per_window: int = 50
    downlink_fraction: float = 0.25       # transmit ~25% of what is captured
    compute_fraction: float = 1.0         # cycles provisioned for the arrival rate
    bits_per_window: float | None = None      # absolute override
    cycles_per_window: float | None = None    # absolute override

    buffer_max_age_sols: int = 200
    buffer_capacity: int | None = None

    adaptation: str = "frozen"
    bootstrap_sols: int = 200
    refit_every_windows: int = 4
    refit_max_frames: int = 4000
    ground_feedback: bool = False

    cycles_per_flop: float = 3.0
    compress_flops_per_frame: int | None = None   # default: 2 ops per raw sample
    seed: int = 0

    def validate(self) -> SimConfig:
        if self.adaptation not in ADAPTATION_MODES:
            raise ValueError(
                f"adaptation must be one of {ADAPTATION_MODES}, got {self.adaptation!r}"
            )
        if self.ground_feedback and self.adaptation != "online":
            raise ValueError(
                "ground_feedback requires adaptation: online -- a frozen model has "
                "nothing to feed corrections back into"
            )
        if not 0 < self.downlink_fraction <= 1:
            raise ValueError(f"downlink_fraction must be in (0, 1], got {self.downlink_fraction}")
        if self.compute_fraction <= 0:
            raise ValueError(f"compute_fraction must be positive, got {self.compute_fraction}")
        if self.sols_per_window < 1:
            raise ValueError(f"sols_per_window must be >= 1, got {self.sols_per_window}")
        return self


@dataclass
class WindowRecord:
    """One window's decisions. Serialised to windows.jsonl."""

    window: int
    first_sol: int
    last_sol: int
    method: str
    n_arrived: int
    n_buffered: int
    n_prefiltered: int
    n_scored: int
    n_unscored: int
    n_selected: int
    n_expired: int
    n_evicted: int
    bits_budget: float
    bits_used: float
    cycles_budget: float
    cycles_used: float
    cycles_scoring: float
    cycles_prefilter: float
    binding_constraint: str
    sent_natural: int
    sent_rover: int
    sent_typical: int
    bits_natural: float
    bits_rover: float
    bits_typical: float
    cum_sent_natural: int
    cum_natural_available: int
    cum_science_yield: float
    cum_wasted_bit_share: float
    refit: bool = False
    prefilter_recall: float | None = None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class SimResult:
    """Aggregate outcome of one (tier, method) replay."""

    method: str
    artifact: str
    tier: str
    adaptation: str
    config: dict
    windows: list[WindowRecord]
    science_yield: float
    wasted_bit_share: float
    n_expired: int
    n_expired_natural: int
    n_unscored: int
    n_sent: int
    n_sent_natural: int
    n_sent_rover: int
    n_sent_typical: int
    n_natural_total: int
    bits_used: float
    bits_available: float
    precision_natural: float
    n_refits: int
    wall_clock_seconds: float

    def to_json(self) -> dict:
        out = asdict(self)
        out["windows"] = len(self.windows)
        return out


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------
def _derive_budgets(
    mission: MissionStream,
    windows: list[DownlinkWindow],
    config: SimConfig,
    cycles_per_score: float,
) -> tuple[float, float, dict]:
    """Per-window bit and cycle budgets, scaled to what actually arrives."""
    n_windows = max(1, len(windows))
    bits_total = float(sum(f.bits for f in mission.frames))
    bits_per_window_arriving = bits_total / n_windows
    frames_per_window_arriving = len(mission) / n_windows

    bits = (
        float(config.bits_per_window)
        if config.bits_per_window
        else bits_per_window_arriving * config.downlink_fraction
    )
    cycles = (
        float(config.cycles_per_window)
        if config.cycles_per_window
        else frames_per_window_arriving * cycles_per_score * config.compute_fraction
    )
    derivation = {
        "n_windows": n_windows,
        "frames_per_window_arriving": frames_per_window_arriving,
        "bits_per_window_arriving": bits_per_window_arriving,
        "bits_per_window": bits,
        "cycles_per_window": cycles,
        "cycles_per_score": cycles_per_score,
        "downlink_fraction": config.downlink_fraction,
        "compute_fraction": config.compute_fraction,
        "frames_affordable_per_window": bits / (bits_total / len(mission)),
        "scores_affordable_per_window": cycles / cycles_per_score if cycles_per_score else None,
    }
    return bits, cycles, derivation


def _refit_model(model, mission: MissionStream, indices: Sequence[int], seed: int):
    """Refit the novelty model on frames captured so far. No labels used."""
    from core.dataset import ChunkedArray

    if len(indices) < 8:
        return model
    subset = mission.batch(sorted(indices))
    chunks = ChunkedArray(subset, chunk_size=min(256, len(subset)))
    model.fit(chunks, n_samples=len(subset), seed=seed)
    return model


def replay(
    mission: MissionStream,
    model,
    *,
    method: str = "greedy_ratio",
    config: SimConfig | None = None,
    cycles_per_score: float | None = None,
    artifact: str = "",
    tier: str = "",
    windows: Iterable[DownlinkWindow] | None = None,
) -> SimResult:
    """Replay the mission window by window under one selection policy.

    Deterministic given `config.seed`: the only stochastic element is the
    `random` policy, and it draws from a seeded Generator.
    """
    config = (config or SimConfig()).validate()
    if method not in policy.METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {policy.METHODS}")

    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)

    frame_shape = mission.array.shape[1:]
    prefilter_cycles = prefilter.prefilter_flops(frame_shape) * config.cycles_per_flop
    compress_flops = config.compress_flops_per_frame or (2 * int(np.prod(frame_shape)))
    compress_cycles = compress_flops * config.cycles_per_flop
    if cycles_per_score is None:
        cycles_per_score = float(model.flops_per_inference()) * config.cycles_per_flop
    mean_bits = float(np.mean([f.bits for f in mission.frames])) if len(mission) else 1.0

    if windows is None:
        planned = plan_windows(mission.rows, sols_per_window=config.sols_per_window)
        bits_budget, cycles_budget, derivation = _derive_budgets(
            mission, planned, config, cycles_per_score
        )
        windows = plan_windows(
            mission.rows,
            sols_per_window=config.sols_per_window,
            bits_per_window=bits_budget,
            cycles_per_window=cycles_budget,
        )
    else:
        windows = list(windows)
        bits_budget = windows[0].budget.bits if windows else 0.0
        cycles_budget = windows[0].budget.cycles if windows else 0.0
        derivation = {"bits_per_window": bits_budget, "cycles_per_window": cycles_budget}

    buffer = FrameBuffer(
        max_age_sols=config.buffer_max_age_sols, capacity=config.buffer_capacity
    )
    stats = prefilter.RunningStats()

    # Online adaptation state. `seen` is every frame captured so far -- what the
    # rover could refit on. `known_novel` is only ever populated by the optional
    # ground-feedback uplink; without it the rover has no labels at all.
    seen: list[int] = []
    known_novel: set[int] = set()
    score_cache: dict[int, float] = {}
    n_refits = 0
    bootstrapped = config.adaptation == "frozen"

    records: list[WindowRecord] = []
    sent_natural = sent_rover = sent_typical = 0
    bits_natural = bits_rover = bits_typical = 0.0
    total_bits_used = 0.0
    natural_seen = 0
    n_unscored_total = 0
    sent_indices: list[int] = []

    for window in windows:
        arriving = [mission.frames[i] for i in window.frame_indices]
        current_sol = max((f.sol for f in arriving), default=window.last_sol)

        # -- capture ----------------------------------------------------------
        for frame in arriving:
            buffer.add(frame, frame.sol)
            seen.append(frame.index)
            if frame.is_natural:
                natural_seen += 1

        n_expired_before = buffer.n_expired
        n_evicted_before = buffer.n_evicted
        buffer.expire(current_sol)
        buffer.enforce_capacity()
        n_expired = buffer.n_expired - n_expired_before
        n_evicted = buffer.n_evicted - n_evicted_before

        candidates = buffer.candidates()
        if not candidates:
            continue

        # -- online adaptation ------------------------------------------------
        did_refit = False
        if config.adaptation == "online":
            due_bootstrap = not bootstrapped and current_sol >= config.bootstrap_sols
            due_refit = (
                bootstrapped
                and config.refit_every_windows > 0
                and window.index % config.refit_every_windows == 0
            )
            if due_bootstrap or due_refit:
                pool = [i for i in seen if i not in known_novel][-config.refit_max_frames :]
                model = _refit_model(model, mission, pool, config.seed)
                score_cache.clear()   # a refit invalidates every cached score
                bootstrapped = True
                did_refit = True
                n_refits += 1

        # -- reserve cycles for transmitting -----------------------------------
        # Compressing a frame for downlink costs cycles too. Without a reserve,
        # a score-based policy spends its entire budget scoring and then cannot
        # compress anything to send -- it would leave the relay pass idle, which
        # no designed system does. So set aside what a full window's worth of
        # transmission costs, and triage with what is left. Scoring still gets
        # squeezed hard; that scarcity is the point and it is not hidden.
        transmit_reserve = float(np.ceil(window.budget.bits / mean_bits)) * compress_cycles
        transmit_reserve = min(transmit_reserve, window.budget.cycles)
        cycles_left = window.budget.cycles - transmit_reserve

        # -- prefilter (cheap, runs on everything) ----------------------------
        new_stats = [c for c in candidates if c.prefilter is None]
        cycles_prefilter = 0.0
        if new_stats:
            batch = mission.batch([c.frame.index for c in new_stats])
            statistics = prefilter.frame_statistics(batch)
            ranked = prefilter.prefilter_rank(statistics, stats)
            stats.update(statistics)
            for item, value in zip(new_stats, ranked, strict=True):
                item.prefilter = float(value)
            cycles_prefilter = len(new_stats) * prefilter_cycles
            cycles_left -= cycles_prefilter

        # -- triage: who gets a full novelty score ----------------------------
        n_scored = 0
        n_unscored = 0
        cycles_scoring = 0.0
        if policy.needs_scores(method):
            unscored = [c for c in candidates if c.frame.index not in score_cache]
            # Best prefilter rank first: the whole point of triage.
            unscored.sort(key=lambda c: (-(c.prefilter or 0.0), c.frame.index))
            affordable = int(max(0.0, cycles_left) // cycles_per_score) if cycles_per_score else len(unscored)
            to_score = unscored[:affordable]
            n_unscored = len(unscored) - len(to_score)
            n_unscored_total += n_unscored

            if to_score:
                batch = mission.batch([c.frame.index for c in to_score])
                fresh = model.score(batch)
                for item, value in zip(to_score, fresh, strict=True):
                    score_cache[item.frame.index] = float(value)
                n_scored = len(to_score)
                cycles_scoring = n_scored * cycles_per_score
                cycles_left -= cycles_scoring

            for item in candidates:
                item.score = score_cache.get(item.frame.index)
            # A frame with no score cannot be ranked by a score-based policy.
            eligible = [c for c in candidates if c.score is not None]
        else:
            eligible = candidates

        if not eligible:
            continue

        # -- selection under both budgets -------------------------------------
        scores = np.array([c.score if c.score is not None else 0.0 for c in eligible])
        bits = np.array([c.frame.bits for c in eligible])
        cycles = np.full(len(eligible), compress_cycles)
        capture_order = np.array(
            [c.captured_sol * 10_000 + c.frame.index for c in eligible], dtype=np.float64
        )
        is_natural = np.array([c.frame.is_natural for c in eligible])

        # Whatever triage did not spend rolls back into the transmit allowance.
        selection_budget = BudgetSpec(
            bits=window.budget.bits,
            cycles=max(transmit_reserve + max(cycles_left, 0.0), compress_cycles),
        )
        plan = policy.select(
            method,
            scores=scores,
            bits=bits,
            cycles=cycles,
            capture_order=capture_order,
            is_natural=is_natural,
            budget=selection_budget,
            rng=rng,
        )

        chosen = [eligible[i] for i in plan.selected]

        # -- transmit ---------------------------------------------------------
        window_natural = window_rover = window_typical = 0
        for item in chosen:
            frame = item.frame
            if frame.is_natural:
                sent_natural += 1
                window_natural += 1
                bits_natural += frame.bits
            elif frame.is_rover:
                sent_rover += 1
                window_rover += 1
                bits_rover += frame.bits
            else:
                sent_typical += 1
                window_typical += 1
                bits_typical += frame.bits
            sent_indices.append(frame.index)
        total_bits_used += plan.used_bits

        buffer.remove([c.frame.index for c in chosen])
        for item in chosen:
            score_cache.pop(item.frame.index, None)

        # -- optional ground feedback (off by default) -------------------------
        if config.ground_feedback:
            # An expert on Earth labels what arrived and uplinks corrections.
            # Only the transmitted frames -- that is the low-bandwidth part.
            for item in chosen:
                if item.frame.is_novel:
                    known_novel.add(item.frame.index)

        # -- record ------------------------------------------------------------
        wasted = bits_rover / total_bits_used if total_bits_used > 0 else 0.0
        prefilter_recall = None
        if policy.needs_scores(method) and n_unscored:
            scored_natural = sum(
                1 for c in candidates
                if c.frame.is_natural and c.frame.index in score_cache
            )
            buffered_natural = sum(1 for c in candidates if c.frame.is_natural)
            if buffered_natural:
                prefilter_recall = scored_natural / buffered_natural

        records.append(
            WindowRecord(
                window=window.index,
                first_sol=window.first_sol,
                last_sol=window.last_sol,
                method=method,
                n_arrived=len(arriving),
                n_buffered=len(candidates),
                n_prefiltered=len(new_stats),
                n_scored=n_scored,
                n_unscored=n_unscored,
                n_selected=len(chosen),
                n_expired=n_expired,
                n_evicted=n_evicted,
                bits_budget=window.budget.bits,
                bits_used=plan.used_bits,
                cycles_budget=window.budget.cycles,
                cycles_used=cycles_prefilter + cycles_scoring + plan.used_cycles,
                cycles_scoring=cycles_scoring,
                cycles_prefilter=cycles_prefilter,
                binding_constraint=plan.binding_constraint,
                sent_natural=window_natural,
                sent_rover=window_rover,
                sent_typical=window_typical,
                bits_natural=bits_natural,
                bits_rover=bits_rover,
                bits_typical=bits_typical,
                cum_sent_natural=sent_natural,
                cum_natural_available=natural_seen,
                cum_science_yield=(sent_natural / natural_seen) if natural_seen else 0.0,
                cum_wasted_bit_share=wasted,
                refit=did_refit,
                prefilter_recall=prefilter_recall,
            )
        )

    # Frames still buffered when the mission ends never reached the ground.
    n_natural_total = mission.n_natural
    n_sent = sent_natural + sent_rover + sent_typical
    bits_available = float(bits_budget * len(windows))

    return SimResult(
        method=method,
        artifact=artifact,
        tier=tier,
        adaptation=config.adaptation,
        config={**asdict(config), "derivation": derivation},
        windows=records,
        science_yield=(sent_natural / n_natural_total) if n_natural_total else 0.0,
        wasted_bit_share=(bits_rover / total_bits_used) if total_bits_used > 0 else 0.0,
        n_expired=buffer.n_expired + buffer.n_evicted,
        n_expired_natural=buffer.expired_natural,
        n_unscored=n_unscored_total,
        n_sent=n_sent,
        n_sent_natural=sent_natural,
        n_sent_rover=sent_rover,
        n_sent_typical=sent_typical,
        n_natural_total=n_natural_total,
        bits_used=total_bits_used,
        bits_available=bits_available,
        precision_natural=(sent_natural / n_sent) if n_sent else 0.0,
        n_refits=n_refits,
        wall_clock_seconds=round(time.perf_counter() - started, 3),
    )


def write_windows_jsonl(result: SimResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in result.windows:
            fh.write(json.dumps(record.to_json()) + "\n")
    return path
