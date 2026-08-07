"""The downlink simulator: budgets that bind, frames that expire, replay that reproduces.

These use a synthetic mission (built by the fixture, not the real 856 frames)
so they run in milliseconds and can assert exact counts. The properties they
guard are the ones that would silently turn the simulation into theatre:
budgets that never bind, frames that vanish instead of expiring, an oracle that
is not actually an upper bound, and a replay that does not reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import taxonomy
from core.budgets import BudgetSpec
from core.manifest import ManifestRow
from sim import policy
from sim.mission import FrameBuffer, MissionFrame, MissionStream
from sim.prefilter import RunningStats, frame_statistics, prefilter_flops
from sim.window import SimConfig, plan_windows, replay, write_windows_jsonl

FRAME_SHAPE = (64, 64, 6)


class _FakeModel:
    """Scores a frame by its mean: deterministic, and orderable by construction."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.n_scored = 0
        self.n_fits = 0

    def score(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.float32)
        self.n_scored += len(frames)
        return frames.reshape(len(frames), -1).mean(axis=1).astype(np.float64) * self.scale

    def flops_per_inference(self) -> int:
        return 866_432

    def fit(self, chunks, *, n_samples: int, seed: int = 0):
        self.n_fits += 1
        return self


def _mission(n_natural: int = 12, n_rover: int = 12, n_typical: int = 24) -> MissionStream:
    """Natural frames are bright (score high), rover and typical are dim."""
    rng = np.random.default_rng(0)
    specs: list[tuple[str, float]] = (
        [(taxonomy.GROUP_NATURAL, 200.0)] * n_natural
        + [(taxonomy.GROUP_ROVER, 100.0)] * n_rover
        + [("typical", 100.0)] * n_typical
    )
    rng.shuffle(specs)

    frames: list[MissionFrame] = []
    rows: list[ManifestRow] = []
    array = np.zeros((len(specs), *FRAME_SHAPE), dtype=np.float32)
    for i, (group, level) in enumerate(specs):
        sol = 10 + i * 5
        array[i] = level + rng.normal(0, 1, size=FRAME_SHAPE)
        split = "test_typical" if group == "typical" else "test_novel_all"
        class_ = {"typical": "typical", taxonomy.GROUP_NATURAL: "veins",
                  taxonomy.GROUP_ROVER: "drt"}[group]
        frames.append(
            MissionFrame(index=i, sol=sol, split=split, class_=class_,
                         source_filename=f"f{i}.npy", group=group, bits=1000.0)
        )
        rows.append(ManifestRow(i, split, class_, sol, f"f{i}.npy"))
    return MissionStream(frames=frames, array=array, rows=rows)


@pytest.fixture
def mission() -> MissionStream:
    return _mission()


@pytest.fixture
def tight() -> SimConfig:
    """Both budgets deliberately scarce, so both actually bind."""
    return SimConfig(
        sols_per_window=40, downlink_fraction=0.25, compute_fraction=0.5,
        buffer_max_age_sols=60, seed=0,
    )


# ---------------------------------------------------------------------------
# The budgets must bind
# ---------------------------------------------------------------------------
def test_bit_budget_binds_and_is_never_exceeded(mission, tight) -> None:
    result = replay(mission, _FakeModel(), method="score_first", config=tight)
    assert result.windows
    for record in result.windows:
        assert record.bits_used <= record.bits_budget + 1e-6
    # Scarcity is real: not everything captured gets sent.
    assert result.n_sent < len(mission)


def test_compute_budget_leaves_frames_unscored(mission, tight) -> None:
    """The second-order scarcity: too little compute to even look at everything."""
    result = replay(mission, _FakeModel(), method="score_first", config=tight)
    assert result.n_unscored > 0, "cycle budget never bound; the triage path is untested"
    assert any(r.n_unscored > 0 for r in result.windows)


def test_policies_that_need_no_score_pay_no_cycle_tax(mission, tight) -> None:
    """FIFO is cheap by nature -- that is why it is the honest baseline."""
    fifo = replay(mission, _FakeModel(), method="fifo", config=tight)
    scored = replay(mission, _FakeModel(), method="score_first", config=tight)
    assert fifo.n_unscored == 0
    assert scored.n_unscored > 0
    assert all(r.cycles_scoring == 0 for r in fifo.windows)


def test_a_model_is_only_charged_once_per_frame_until_a_refit(mission, tight) -> None:
    """Scores are cached: a frame still buffered next window is not re-scored."""
    model = _FakeModel()
    result = replay(mission, model, method="score_first", config=tight)
    total_scored = sum(r.n_scored for r in result.windows)
    assert model.n_scored == total_scored
    # Strictly fewer scorings than (buffered frames summed over windows).
    assert total_scored < sum(r.n_buffered for r in result.windows)


# ---------------------------------------------------------------------------
# Retention and expiry
# ---------------------------------------------------------------------------
def test_frames_expire_and_are_counted_not_dropped_silently(mission) -> None:
    config = SimConfig(sols_per_window=40, downlink_fraction=0.05,
                       buffer_max_age_sols=30, seed=0)
    result = replay(mission, _FakeModel(), method="score_first", config=config)
    assert result.n_expired > 0
    assert sum(r.n_expired for r in result.windows) == result.n_expired

    # Every captured frame is accounted for: sent, expired, or still buffered.
    assert result.n_sent + result.n_expired <= len(mission)


def test_a_longer_age_limit_expires_fewer_frames(mission) -> None:
    short = replay(mission, _FakeModel(), method="score_first",
                   config=SimConfig(sols_per_window=40, buffer_max_age_sols=30, seed=0))
    long = replay(mission, _FakeModel(), method="score_first",
                  config=SimConfig(sols_per_window=40, buffer_max_age_sols=400, seed=0))
    assert long.n_expired < short.n_expired


def test_buffer_expiry_records_the_group_that_was_lost() -> None:
    buffer = FrameBuffer(max_age_sols=10)
    natural = MissionFrame(0, 5, "test_novel_all", "veins", "a.npy",
                           taxonomy.GROUP_NATURAL, 1000.0)
    rover = MissionFrame(1, 5, "test_novel_all", "drt", "b.npy",
                         taxonomy.GROUP_ROVER, 1000.0)
    buffer.add(natural, 5)
    buffer.add(rover, 5)
    buffer.expire(current_sol=100)
    assert buffer.n_expired == 2
    assert buffer.expired_natural == 1
    assert buffer.expired_rover == 1
    assert len(buffer) == 0


def test_buffer_capacity_evicts_oldest_first() -> None:
    buffer = FrameBuffer(max_age_sols=10_000, capacity=2)
    for i in range(4):
        buffer.add(
            MissionFrame(i, 10 * i, "test_typical", "typical", f"{i}.npy", "typical", 1.0),
            10 * i,
        )
    buffer.enforce_capacity()
    assert len(buffer) == 2
    assert sorted(buffer.items) == [2, 3]  # the two most recent survive


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def test_novelty_ranking_beats_fifo_on_science_yield(mission, tight) -> None:
    """The whole point: at an identical bit budget, does thinking help?"""
    fifo = replay(mission, _FakeModel(), method="fifo", config=tight)
    smart = replay(mission, _FakeModel(), method="score_first", config=tight)
    assert smart.science_yield > fifo.science_yield


def test_oracle_is_an_upper_bound_on_every_other_policy(mission, tight) -> None:
    oracle = replay(mission, _FakeModel(), method="oracle", config=tight)
    for method in ("fifo", "random", "score_first", "greedy_ratio"):
        other = replay(mission, _FakeModel(), method=method, config=tight)
        assert other.science_yield <= oracle.science_yield + 1e-9, (
            f"{method} beat the oracle, so the oracle is not an upper bound"
        )


def test_fifo_transmits_in_capture_order(mission) -> None:
    config = SimConfig(sols_per_window=40, downlink_fraction=0.25, seed=0)
    result = replay(mission, _FakeModel(), method="fifo", config=config)
    # In the first window FIFO must pick the earliest-captured frames.
    first = result.windows[0]
    assert first.n_selected > 0
    assert first.sent_natural + first.sent_rover + first.sent_typical == first.n_selected


def test_every_named_method_runs(mission, tight) -> None:
    for method in policy.METHODS:
        result = replay(mission, _FakeModel(), method=method, config=tight)
        assert 0.0 <= result.science_yield <= 1.0
        assert 0.0 <= result.wasted_bit_share <= 1.0


def test_unknown_method_is_rejected(mission) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        replay(mission, _FakeModel(), method="telepathy")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_replay_is_deterministic_given_a_seed(mission, tight) -> None:
    a = replay(mission, _FakeModel(), method="random", config=tight)
    b = replay(mission, _FakeModel(), method="random", config=tight)
    assert a.science_yield == b.science_yield
    assert [r.n_selected for r in a.windows] == [r.n_selected for r in b.windows]


def test_different_seeds_change_the_random_policy(mission) -> None:
    """Needs a budget wide enough to select several frames: at one frame per
    window every seed picks the same single candidate and the test proves
    nothing."""
    def run(seed: int):
        return replay(mission, _FakeModel(), method="random",
                      config=SimConfig(sols_per_window=40, downlink_fraction=0.6, seed=seed))

    a, b = run(0), run(99)
    composition_a = [(r.sent_natural, r.sent_rover, r.sent_typical) for r in a.windows]
    composition_b = [(r.sent_natural, r.sent_rover, r.sent_typical) for r in b.windows]
    assert composition_a != composition_b, "the random policy ignored its seed"


# ---------------------------------------------------------------------------
# Adaptation modes
# ---------------------------------------------------------------------------
def test_online_mode_refits_and_frozen_does_not(mission, tight) -> None:
    frozen_model = _FakeModel()
    frozen = replay(mission, frozen_model, method="score_first", config=tight)
    assert frozen_model.n_fits == 0
    assert frozen.n_refits == 0

    online_config = SimConfig(**{**tight.__dict__, "adaptation": "online",
                                 "bootstrap_sols": 20, "refit_every_windows": 1})
    online_model = _FakeModel()
    online = replay(mission, online_model, method="score_first", config=online_config)
    assert online_model.n_fits > 0
    assert online.n_refits == online_model.n_fits
    assert any(r.refit for r in online.windows)


def test_ground_feedback_requires_online_adaptation() -> None:
    with pytest.raises(ValueError, match="requires adaptation: online"):
        SimConfig(adaptation="frozen", ground_feedback=True).validate()


def test_invalid_adaptation_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="adaptation must be"):
        SimConfig(adaptation="clairvoyant").validate()


# ---------------------------------------------------------------------------
# Metrics and outputs
# ---------------------------------------------------------------------------
def test_science_yield_denominator_is_natural_frames_captured(mission, tight) -> None:
    result = replay(mission, _FakeModel(), method="oracle", config=tight)
    assert result.n_natural_total == mission.n_natural
    assert result.science_yield == pytest.approx(result.n_sent_natural / mission.n_natural)


def test_wasted_bits_counts_only_rover_classes(mission, tight) -> None:
    result = replay(mission, _FakeModel(), method="fifo", config=tight)
    assert 0.0 <= result.wasted_bit_share <= 1.0
    final = result.windows[-1]
    total = final.bits_natural + final.bits_rover + final.bits_typical
    assert result.wasted_bit_share == pytest.approx(final.bits_rover / total, rel=1e-6)


def test_cumulative_curves_are_monotone(mission, tight) -> None:
    result = replay(mission, _FakeModel(), method="score_first", config=tight)
    sent = [r.cum_sent_natural for r in result.windows]
    available = [r.cum_natural_available for r in result.windows]
    assert sent == sorted(sent)
    assert available == sorted(available)
    for record in result.windows:
        assert record.cum_sent_natural <= record.cum_natural_available


def test_windows_jsonl_round_trips(mission, tight, tmp_path: Path) -> None:
    result = replay(mission, _FakeModel(), method="greedy_ratio", config=tight)
    path = write_windows_jsonl(result, tmp_path / "w.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == len(result.windows)
    assert records[0]["method"] == "greedy_ratio"
    assert {"n_unscored", "cum_science_yield", "binding_constraint"} <= set(records[0])


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------
def test_prefilter_is_far_cheaper_than_a_full_score() -> None:
    assert prefilter_flops(FRAME_SHAPE) < 0.2 * 866_432        # vs PCA
    assert prefilter_flops(FRAME_SHAPE) < 0.02 * 9_428_992     # vs myriad AE


def test_prefilter_statistics_separate_textured_frames() -> None:
    rng = np.random.default_rng(0)
    flat = np.full((4, *FRAME_SHAPE), 100.0, dtype=np.float32)
    busy = rng.normal(100.0, 30.0, size=(4, *FRAME_SHAPE)).astype(np.float32)
    stats = frame_statistics(np.concatenate([flat, busy]))
    assert stats[:4, 0].mean() < stats[4:, 0].mean()   # spatial variance


def test_running_stats_are_causal() -> None:
    """Z-scores must never depend on frames that have not arrived yet."""
    stats = RunningStats()
    early = np.array([[1.0, 1.0], [1.1, 0.9]])
    stats.update(early)
    before = stats.z(np.array([[5.0, 5.0]]))
    stats.update(np.array([[100.0, 100.0]]))
    after = stats.z(np.array([[5.0, 5.0]]))
    assert not np.isclose(before, after), "later frames failed to move the baseline"


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------
def test_plan_windows_carries_the_budget(mission) -> None:
    windows = plan_windows(mission.rows, sols_per_window=40,
                           bits_per_window=5000.0, cycles_per_window=1e6)
    assert windows
    assert all(isinstance(w.budget, BudgetSpec) for w in windows)
    assert windows[0].budget.bits == 5000.0
    assert sum(w.n_candidates for w in windows) == len(mission.rows)
