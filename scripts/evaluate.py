"""Score test_typical against test_novel/all and report ranking metrics.

    python -m scripts.evaluate                                  # artifacts/rad750.npz
    python -m scripts.evaluate --artifact artifacts/myriad.npz
    python -m scripts.evaluate --budget-demo

Writes `runs/metrics/<name>.json` and, unless --no-publish is passed, also
`artifacts/metrics/<name>.json` so the committed artifact directory carries the
numbers that go with the committed weights. Both records share the sidecar
identity block (core.provenance.identity_block), so they join cleanly on
config_hash and content_sha256.

METRIC ORDERING IS DELIBERATE. precision@k leads the printed output because a
downlink window carries ~162 frames (8 Mbit at 49,152 bits/frame): only the
top of the ranking has any operational meaning -- a frame ranked 400th of 856
does not get transmitted whether the model ordered the tail well or not.
ROC AUC integrates over every threshold, most of which the mission will never
operate at, so it appears below the headline as the literature-comparison
figure (Kerner et al. 2020 report 0.65 for a conv autoencoder on this dataset).

The aggregate is also decomposed (core.taxonomy): novelty because Mars made it
(natural) versus novelty because the rover did it (rover). Task-1 showed the
aggregate hiding a 0.5-point gap between those two, and the decomposition is
what makes the cross-tier comparison scientifically legible.

The novel split is always `test_novel_all`. Never `test_novel_byclass`, which
has one row per (frame, label) and would double count multi-label frames --
core.config rejects that configuration outright.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from core import paths, taxonomy
from core.budgets import (
    BudgetSpec,
    estimate_bits_from_frames,
    estimate_frame_cycles,
    select_two_budget,
)
from core.dataset import SPLIT_NOVEL_BYCLASS, ChunkedArray, load_split
from core.logging_utils import get_logger, setup_logging
from core.models.registry import load_model, read_artifact_meta
from core.provenance import file_sha256, identity_block, sidecar_path_for
from core.scoring import evaluate_scores, roc_auc

log = get_logger("novum.evaluate")

#: Kerner et al. 2020, convolutional autoencoder on the Mastcam novelty set.
REFERENCE_CONV_AE_ROC_AUC = 0.65
REFERENCE_SOURCE = "Kerner et al. 2020, conv autoencoder on Mastcam novelty (ROC AUC 0.65)"

#: One 8 Mbit downlink window at 49,152 bits/frame ~= 162 frames.
FRAMES_PER_WINDOW = 162

DEFAULT_K_VALUES = (10, 25, 50, 100, FRAMES_PER_WINDOW)


def _score_split(model, split_name: str, chunk_size: int = 512) -> tuple[np.ndarray, list]:
    split = load_split(split_name)
    scores = model.score_chunks(ChunkedArray(split.array, chunk_size))
    log.info(
        "scored %-20s %5d frames  mean=%.4f  sd=%.4f",
        split_name,
        len(scores),
        float(scores.mean()),
        float(scores.std()),
    )
    return scores, split.rows


def _group_auc(typical_scores: np.ndarray, group_scores: np.ndarray) -> float | None:
    if group_scores.size == 0:
        return None
    y = np.concatenate(
        [np.zeros(typical_scores.size, dtype=np.int8), np.ones(group_scores.size, dtype=np.int8)]
    )
    return roc_auc(y, np.concatenate([typical_scores, group_scores]))


def decompose_by_group(
    typical_scores: np.ndarray,
    novel_scores: np.ndarray,
    novel_rows: list,
) -> dict:
    """Science-vs-housekeeping decomposition over the canonical novel set.

    Each canonical frame carries its class labels ('|'-joined for the five
    multi-label frames); core.taxonomy maps labels to groups. A frame counts in
    every group any of its labels belongs to (in the real archive no frame
    straddles natural and rover, but the rule is defined). 'other' and
    'edge_cases' have 1 and 3 frames -- far too few for a rate to mean anything
    -- so they are reported under 'excluded' and never folded into a group.
    """
    masks = {
        taxonomy.GROUP_NATURAL: np.zeros(len(novel_rows), dtype=bool),
        taxonomy.GROUP_ROVER: np.zeros(len(novel_rows), dtype=bool),
        taxonomy.GROUP_EXCLUDED: np.zeros(len(novel_rows), dtype=bool),
    }
    excluded_labels: dict[str, int] = {}
    for i, row in enumerate(novel_rows):
        labels = taxonomy.split_labels(row.class_)
        for group in taxonomy.groups_for_labels(labels):
            masks[group][i] = True
        for label in labels:
            if label not in taxonomy.NATURAL_CLASSES and label not in taxonomy.ROVER_CLASSES:
                excluded_labels[label] = excluded_labels.get(label, 0) + 1

    out: dict = {"classes": {
        taxonomy.GROUP_NATURAL: sorted(taxonomy.NATURAL_CLASSES),
        taxonomy.GROUP_ROVER: sorted(taxonomy.ROVER_CLASSES),
        taxonomy.GROUP_EXCLUDED: sorted(taxonomy.EXCLUDED_CLASSES),
    }}
    for group, mask in masks.items():
        scores = novel_scores[mask]
        out[group] = {
            "n": int(mask.sum()),
            "roc_auc": _group_auc(typical_scores, scores),
        }
    out[taxonomy.GROUP_EXCLUDED]["label_counts"] = excluded_labels
    return out


def _per_class_auc(model, typical_scores: np.ndarray, chunk_size: int) -> dict[str, dict]:
    """AUC for each novelty class, scored against the same typical baseline."""
    try:
        split = load_split(SPLIT_NOVEL_BYCLASS)
    except (FileNotFoundError, KeyError) as exc:
        log.warning("per-class breakdown unavailable: %s", exc)
        return {}

    scores = model.score_chunks(ChunkedArray(split.array, chunk_size))
    by_class: dict[str, list[float]] = {}
    for row, score in zip(split.rows, scores, strict=True):
        by_class.setdefault(row.class_, []).append(float(score))

    out: dict[str, dict] = {}
    for label, values in sorted(by_class.items()):
        arr = np.asarray(values, dtype=np.float64)
        y = np.concatenate(
            [np.zeros(typical_scores.size, dtype=np.int8), np.ones(arr.size, dtype=np.int8)]
        )
        s = np.concatenate([typical_scores, arr])
        try:
            out[label] = {"n": int(arr.size), "roc_auc": roc_auc(y, s)}
        except ValueError as exc:  # pragma: no cover - needs a degenerate split
            log.warning("skipping class %s: %s", label, exc)
    return out


def _budget_demo(model, cfg: dict, typical_split: str, novel_split: str, chunk_size: int) -> dict:
    """Illustrate selection under both budgets over the combined test set.

    The two splits are distinct sets of frames, so concatenating their scores
    counts nothing twice. (Concatenating test_novel_all with test_novel_byclass
    would, which is why core.dataset.concat_splits refuses to.)
    """
    typical = load_split(typical_split)
    novel = load_split(novel_split)

    frames = np.concatenate([np.asarray(typical.array), np.asarray(novel.array)], axis=0)
    values = np.concatenate(
        [
            model.score_chunks(ChunkedArray(typical.array, chunk_size)),
            model.score_chunks(ChunkedArray(novel.array, chunk_size)),
        ]
    )
    truth = np.concatenate([np.zeros(len(typical), dtype=np.int8), np.ones(len(novel), dtype=np.int8)])

    downlink = cfg.get("downlink", {})
    compute = cfg.get("compute", {})
    bits = estimate_bits_from_frames(
        frames,
        bits_per_sample=int(downlink.get("bits_per_sample", 8)),
        compression_ratio=float(downlink.get("compression_ratio", 4.0)),
    )
    per_frame_cycles = estimate_frame_cycles(
        model.flops_per_inference(), cycles_per_flop=float(compute.get("cycles_per_flop", 3.0))
    )
    cycles = np.full(len(frames), per_frame_cycles, dtype=np.float64)

    bit_budget = downlink.get("budget_bits_per_window") or float(bits.sum() * 0.1)
    cycle_budget = (compute.get("budget_cycles_per_frame") or per_frame_cycles) * len(frames)
    budget = BudgetSpec(bits=float(bit_budget), cycles=float(cycle_budget))

    out: dict = {
        "budget_bits": budget.bits,
        "budget_cycles": budget.cycles,
        "bits_per_frame_mean": float(bits.mean()),
        "cycles_per_frame": per_frame_cycles,
        "n_candidates": int(len(frames)),
        "methods": {},
    }
    for method in ("greedy_sweep", "score_first", "random"):
        plan = select_two_budget(values, bits, cycles, budget, method=method, seed=0)
        selected_truth = truth[plan.selected] if plan.n_selected else np.zeros(0)
        record = plan.to_json()
        record["precision"] = float(selected_truth.mean()) if plan.n_selected else 0.0
        record["novel_recovered"] = int(selected_truth.sum())
        out["methods"][method] = record
        log.info(
            "  %-13s selected %4d frames, %3d novel, precision %.3f, %.0f%% of downlink used",
            method,
            plan.n_selected,
            record["novel_recovered"],
            record["precision"],
            100 * plan.bit_utilisation,
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.evaluate",
        description="Evaluate a NOVUM artifact: ROC AUC and precision@k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="path to a .npz artifact (default: artifacts/rad750.npz)",
    )
    p.add_argument("--out", type=Path, default=None, help="metrics JSON (default: runs/metrics/<name>.json)")
    p.add_argument("--k", type=int, action="append", default=None, help="precision@k value (repeatable)")
    p.add_argument("--typical-split", default=None, help="override eval.typical_split")
    p.add_argument("--novel-split", default=None, help="override eval.novel_split")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--no-per-class", action="store_true", help="skip the per-class AUC breakdown")
    p.add_argument(
        "--budget-demo",
        action="store_true",
        help="also run two-budget selection over the test set and record the plan",
    )
    p.add_argument(
        "--no-publish",
        action="store_true",
        help="do not copy metrics into artifacts/metrics/ (runs/ only)",
    )
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    artifact = Path(args.artifact or (paths.artifacts_dir() / "rad750.npz"))
    if not artifact.exists():
        log.error(
            "no artifact at %s. Train one first:  make train TIER=rad750", paths.rel(artifact)
        )
        return 2

    # -- load model and the config it was trained with ----------------------
    try:
        model = load_model(artifact)
        artifact_meta = read_artifact_meta(artifact)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        log.error("could not load %s: %s", paths.rel(artifact), exc)
        return 2
    except NotImplementedError as exc:
        log.error("%s", exc)
        return 3

    sidecar_path = sidecar_path_for(artifact)
    sidecar: dict = {}
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("sidecar %s is not valid JSON; using defaults", paths.rel(sidecar_path))
    else:
        log.warning("no sidecar at %s; falling back to default eval settings", paths.rel(sidecar_path))

    cfg = sidecar.get("config", {})
    eval_cfg = cfg.get("eval", {})
    typical_split = args.typical_split or eval_cfg.get("typical_split", "test_typical")
    novel_split = args.novel_split or eval_cfg.get("novel_split", "test_novel_all")
    k_values = args.k or eval_cfg.get("k_values") or list(DEFAULT_K_VALUES)

    if novel_split == SPLIT_NOVEL_BYCLASS:
        log.error(
            "refusing to evaluate against %s: it has one row per (frame, label) and "
            "double counts multi-label frames. Use test_novel_all.",
            SPLIT_NOVEL_BYCLASS,
        )
        return 2

    log.info("=" * 68)
    log.info("artifact   %s", paths.rel(artifact))
    log.info("model      %s (tier %s)", artifact_meta.get("type"), sidecar.get("tier", "?"))
    log.info("typical    %s", typical_split)
    log.info("novel      %s", novel_split)
    log.info("=" * 68)

    # -- score --------------------------------------------------------------
    try:
        typical_scores, _ = _score_split(model, typical_split, args.chunk_size)
        novel_scores, novel_rows = _score_split(model, novel_split, args.chunk_size)
    except (FileNotFoundError, KeyError) as exc:
        log.error("%s", exc)
        return 2

    try:
        result = evaluate_scores(typical_scores, novel_scores, k_values)
    except ValueError as exc:
        log.error("could not compute metrics: %s", exc)
        return 1

    decomposed = decompose_by_group(typical_scores, novel_scores, novel_rows)

    per_class = {}
    if not args.no_per_class and eval_cfg.get("per_class", True):
        per_class = _per_class_auc(model, typical_scores, args.chunk_size)

    budget_demo = {}
    if args.budget_demo:
        log.info("two-budget selection demo:")
        budget_demo = _budget_demo(model, cfg, typical_split, novel_split, args.chunk_size)

    # -- integrity + compute-budget verdict from the training sidecar -------
    content_sha256 = sidecar.get("content_sha256")
    try:
        recomputed = model.content_sha256()
        if content_sha256 and recomputed != content_sha256:
            log.warning(
                "content_sha256 mismatch: sidecar says %s but the loaded model hashes "
                "to %s. The sidecar does not describe these weights.",
                content_sha256[:16],
                recomputed[:16],
            )
        content_sha256 = recomputed
    except NotImplementedError:
        pass

    compute_budget = dict(sidecar.get("compute_budget") or {})

    # -- assemble -----------------------------------------------------------
    name = artifact.stem
    record = {
        **identity_block(
            name=name,
            tier=sidecar.get("tier"),
            model_type=artifact_meta.get("type"),
            config_hash_=sidecar.get("config_hash"),
            content_sha256=content_sha256,
            artifact=str(paths.rel(artifact)),
        ),
        "kind": "evaluation",
        "artifact_sha256": file_sha256(artifact),
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "typical_split": typical_split,
        "novel_split": novel_split,
        "metrics": result.to_json(),
        "decomposed": decomposed,
        "per_class_roc_auc": per_class,
        "reference": {
            "roc_auc": REFERENCE_CONV_AE_ROC_AUC,
            "source": REFERENCE_SOURCE,
        },
        "compute_budget": compute_budget,
        "fits_compute_budget": compute_budget.get("fits_compute_budget"),
        "budget_utilisation": compute_budget.get("budget_utilisation"),
        "cost": {
            "param_count": sidecar.get("param_count"),
            "flops_per_inference": sidecar.get("flops_per_inference"),
            "cycles_per_inference": compute_budget.get("cycles_per_inference"),
            "wall_clock_seconds": sidecar.get("wall_clock_seconds"),
        },
    }
    if budget_demo:
        record["budget_demo"] = budget_demo

    out_path = Path(args.out or (paths.metrics_dir() / f"{name}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    published = None
    if not args.no_publish:
        published = paths.artifacts_dir() / "metrics" / f"{name}.json"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    # -- report -------------------------------------------------------------
    # precision@k leads: a downlink window carries ~162 frames, so only the
    # top of the ranking is operational. ROC AUC follows as the figure that
    # compares against published work.
    delta = result.roc_auc - REFERENCE_CONV_AE_ROC_AUC
    print("", flush=True)
    print("=" * 68)
    print(f"  NOVUM evaluation: {name}  ({sidecar.get('tier', '?')} tier)")
    print("=" * 68)
    print(f"  {result.n_typical} typical vs {result.n_novel} novel frames"
          f"  |  chance = {result.novel_rate:.3f}")
    print()
    print("  HEADLINE - operational (what actually gets downlinked)")
    for k in sorted(result.precision_at_k):
        marker = "  <- one downlink window" if k == FRAMES_PER_WINDOW else ""
        print(
            f"    precision@{k:<4d}      {result.precision_at_k[k]:.4f}"
            f"   (recall {result.recall_at_k[k]:.4f}){marker}"
        )
    print()
    print("  DECOMPOSED - what kind of novelty the model actually sees")
    for group in ("natural", "rover"):
        entry = decomposed.get(group, {})
        auc = entry.get("roc_auc")
        print(
            f"    roc_auc_{group:<9s}  "
            f"{'n/a' if auc is None else format(auc, '.4f')}   (n={entry.get('n', 0)})"
        )
    excluded = decomposed.get("excluded", {})
    if excluded.get("n"):
        counts = ", ".join(f"{k} n={v}" for k, v in sorted(excluded.get("label_counts", {}).items()))
        auc = excluded.get("roc_auc")
        print(
            f"    excluded            {'n/a' if auc is None else format(auc, '.4f')}"
            f"   ({counts}; too few frames for a rate to mean anything)"
        )
    print()
    print("  LITERATURE COMPARISON")
    print(f"    roc_auc aggregate   {result.roc_auc:.4f}   "
          f"(reference {REFERENCE_CONV_AE_ROC_AUC:.2f} conv AE, {delta:+.4f})")
    print(f"    average precision   {result.average_precision:.4f}")
    if per_class:
        print("    per-class ROC AUC:")
        for label, entry in sorted(per_class.items(), key=lambda kv: -kv[1]["roc_auc"]):
            groups = taxonomy.groups_for_labels([label])
            tag = "natural" if taxonomy.GROUP_NATURAL in groups else (
                "rover" if taxonomy.GROUP_ROVER in groups else "excl.")
            print(f"      {label:<16s} n={entry['n']:<4d} {entry['roc_auc']:.4f}  [{tag}]")
    print()
    print("  COMPUTE BUDGET")
    cycles = compute_budget.get("cycles_per_inference")
    budget_cycles = compute_budget.get("budget_cycles_per_frame")
    if cycles is not None and budget_cycles:
        fits = compute_budget.get("fits_compute_budget")
        utilisation = compute_budget.get("budget_utilisation") or 0.0
        verdict = (
            f"FITS ({utilisation * 100:.1f}% of budget)"
            if fits
            else f"EXCEEDS budget by {utilisation:.1f}x"
        )
        print(f"    {cycles:,.0f} cycles/frame vs {budget_cycles:,.0f} budget -> {verdict}")
    else:
        print("    no training sidecar found; budget verdict unavailable")
    print("=" * 68)
    print(f"  metrics -> {paths.rel(out_path)}")
    if published:
        print(f"  published -> {paths.rel(published)}")
    print("", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
