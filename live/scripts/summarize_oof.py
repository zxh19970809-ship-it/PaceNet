#!/usr/bin/env python
"""Aggregate video OOF predictions into subject-level metrics and CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CLASS_NAMES = ("CP", "ITW", "TD")
VARIANTS = ("global", "uniform", "unconstrained", "cspt")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "classification_cv_20261012"


def to_vector(value: Any, width: int = 3) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    result = [float(x) for x in value]
    if len(result) != width:
        raise ValueError(f"Expected {width} numeric values, got {len(result)}")
    return result


def normalize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["fold"] = int(row["fold"])
    normalized["label_index"] = int(row["label_index"])
    normalized["logits"] = to_vector(row["logits"])
    normalized["probabilities"] = to_vector(row["probabilities"])
    for key in ("phase_entropy_normalized",):
        value = row.get(key)
        normalized[key] = None if value in (None, "") else float(value)
    occupancy = row.get("phase_occupancy")
    if isinstance(occupancy, str):
        normalized["phase_occupancy"] = json.loads(occupancy) if occupancy else None
    return normalized


def aggregate_subject_rows(video_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average clips within view/visit, views equally within visit, visits equally."""
    by_view: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in video_rows:
        row = normalize_prediction_row(raw)
        key = (str(row["variant"]), int(row["fold"]), str(row["subject_id"]), str(row["visit"]), str(row["view"]))
        by_view[key].append(row)

    view_means: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, clips in by_view.items():
        variant, fold, subject_id, visit, view = key
        labels = {int(row["label_index"]) for row in clips}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent labels for subject {subject_id}")
        logits = np.mean([row["logits"] for row in clips], axis=0)
        probs = np.mean([row["probabilities"] for row in clips], axis=0)
        probs = probs / max(float(probs.sum()), 1e-12)
        entropy_values = [row["phase_entropy_normalized"] for row in clips if row.get("phase_entropy_normalized") is not None]
        occupancies = [row.get("phase_occupancy") for row in clips if row.get("phase_occupancy") is not None]
        view_means[(variant, fold, subject_id, visit)].append({
            "variant": variant,
            "fold": fold,
            "subject_id": subject_id,
            "class_label": clips[0]["class_label"],
            "label_index": labels.pop(),
            "visit": visit,
            "view": view,
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "n_videos_in_view": len(clips),
            "phase_entropy_normalized": float(np.mean(entropy_values)) if entropy_values else None,
            "phase_occupancy": np.mean(occupancies, axis=0).tolist() if occupancies else None,
        })

    by_visit: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for key, views in view_means.items():
        variant, fold, subject_id, visit = key
        logits = np.mean([row["logits"] for row in views], axis=0)
        probs = np.mean([row["probabilities"] for row in views], axis=0)
        probs = probs / max(float(probs.sum()), 1e-12)
        entropies = [row["phase_entropy_normalized"] for row in views if row["phase_entropy_normalized"] is not None]
        occupancies = [row["phase_occupancy"] for row in views if row["phase_occupancy"] is not None]
        by_visit[(variant, fold, subject_id)].append({
            "variant": variant,
            "fold": fold,
            "subject_id": subject_id,
            "class_label": views[0]["class_label"],
            "label_index": int(views[0]["label_index"]),
            "visit": visit,
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "n_views_in_visit": len(views),
            "view_names": sorted(row["view"] for row in views),
            "phase_entropy_normalized": float(np.mean(entropies)) if entropies else None,
            "phase_occupancy": np.mean(occupancies, axis=0).tolist() if occupancies else None,
        })

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for key, visits in by_visit.items():
        variant, fold, subject_id = key
        grouped[key] = visits

    subjects: list[dict[str, Any]] = []
    for (variant, fold, subject_id), visits in sorted(grouped.items()):
        label_indices = {int(row["label_index"]) for row in visits}
        if len(label_indices) != 1:
            raise ValueError(f"Inconsistent true labels across visits for subject {subject_id}")
        logits = np.mean([row["logits"] for row in visits], axis=0)
        probs = np.mean([row["probabilities"] for row in visits], axis=0)
        probs = probs / max(float(probs.sum()), 1e-12)
        entropies = [row["phase_entropy_normalized"] for row in visits if row["phase_entropy_normalized"] is not None]
        occupancies = [row["phase_occupancy"] for row in visits if row["phase_occupancy"] is not None]
        subjects.append({
            "variant": variant,
            "fold": fold,
            "subject_id": subject_id,
            "class_label": visits[0]["class_label"],
            "label_index": label_indices.pop(),
            "predicted_label": CLASS_NAMES[int(np.argmax(probs))],
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "n_visits": len(visits),
            "n_views": len({v for row in visits for v in row["view_names"]}),
            "visits": sorted(row["visit"] for row in visits),
            "phase_entropy_normalized": float(np.mean(entropies)) if entropies else None,
            "phase_occupancy": np.mean(occupancies, axis=0).tolist() if occupancies else None,
        })
    return subjects


def metrics_for(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_subjects": 0,
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "macro_f1": math.nan,
            "class_recall": {name: math.nan for name in CLASS_NAMES},
            "confusion": [[0] * 3 for _ in range(3)],
        }
    confusion = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        true_index = int(row["label_index"])
        pred_probs = to_vector(row["probabilities"])
        pred_index = int(np.argmax(pred_probs))
        confusion[true_index, pred_index] += 1
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = float(confusion[index, index])
        support = float(confusion[index].sum())
        predicted_count = float(confusion[:, index].sum())
        recall = true_positive / support if support else math.nan
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[name] = recall
        f1s.append(f1)
    return {
        "n_subjects": int(confusion.sum()),
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "balanced_accuracy": float(np.nanmean(list(recalls.values()))),
        "macro_f1": float(np.mean(f1s)),
        "class_recall": recalls,
        "confusion": confusion.tolist(),
    }


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})
    temp.replace(path)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(value.tolist() if isinstance(value, np.ndarray) else value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def bootstrap_indices_by_class(rows: Sequence[dict[str, Any]], rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray([int(row["label_index"]) for row in rows], dtype=np.int64)
    draws: list[int] = []
    for class_index in range(3):
        ids = np.flatnonzero(labels == class_index)
        if len(ids):
            draws.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
    return np.asarray(draws, dtype=np.int64)


def bootstrap_macro_f1_ci(
    rows: Sequence[dict[str, Any]], repetitions: int, seed: int
) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = bootstrap_indices_by_class(rows, rng)
        values.append(metrics_for([rows[int(i)] for i in indices])["macro_f1"])
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def paired_bootstrap_cspt(
    subject_rows: Sequence[dict[str, Any]], controls: Sequence[str], repetitions: int, seed: int
) -> list[dict[str, Any]]:
    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in subject_rows:
        by_variant[str(row["variant"])][str(row["subject_id"])] = row
    cspt = by_variant.get("cspt", {})
    results = []
    for offset, control in enumerate(controls):
        control_rows = by_variant.get(control, {})
        shared_ids = sorted(set(cspt) & set(control_rows))
        if not shared_ids:
            continue
        cspt_rows = [cspt[sid] for sid in shared_ids]
        other_rows = [control_rows[sid] for sid in shared_ids]
        if any(int(a["label_index"]) != int(b["label_index"]) for a, b in zip(cspt_rows, other_rows)):
            raise ValueError(f"True labels differ across paired OOF predictions for {control}")
        observed = metrics_for(cspt_rows)["macro_f1"] - metrics_for(other_rows)["macro_f1"]
        rng = np.random.default_rng(seed + offset * 10_009)
        deltas = []
        for _ in range(repetitions):
            sampled = bootstrap_indices_by_class(cspt_rows, rng)
            a = [cspt_rows[int(i)] for i in sampled]
            b = [other_rows[int(i)] for i in sampled]
            deltas.append(metrics_for(a)["macro_f1"] - metrics_for(b)["macro_f1"])
        lo, hi = np.quantile(deltas, [0.025, 0.975])
        prob_le_zero = float(np.mean(np.asarray(deltas) <= 0))
        prob_ge_zero = float(np.mean(np.asarray(deltas) >= 0))
        results.append({
            "metric": "subject_macro_f1",
            "comparison": f"cspt - {control}",
            "control_variant": control,
            "n_paired_subjects": len(shared_ids),
            "observed_delta": float(observed),
            "bootstrap_95ci_low": float(lo),
            "bootstrap_95ci_high": float(hi),
            "bootstrap_two_sided_p": min(1.0, 2.0 * min(prob_le_zero, prob_ge_zero)),
            "bootstrap_repetitions": repetitions,
        })
    return results


def write_summary(
    output_dir: Path,
    video_rows: Sequence[dict[str, Any]],
    subject_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    expected_folds: int,
) -> None:
    normalized_subjects = [normalize_prediction_row(row) for row in subject_rows]
    for row in normalized_subjects:
        row["predicted_label"] = CLASS_NAMES[int(np.argmax(row["probabilities"]))]
        row["class_label"] = str(row["class_label"])
    repetitions = int(config["evaluation"].get("bootstrap_repetitions", 2000))
    seed = int(config["evaluation"].get("bootstrap_seed", config["seed"] + 345_019))
    by_variant_fold: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_subjects:
        by_variant_fold[(str(row["variant"]), int(row["fold"]))].append(row)
        by_variant[str(row["variant"])].append(row)

    fold_metrics: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for (variant, fold), rows in sorted(by_variant_fold.items()):
        metric = metrics_for(rows)
        for name, recall in metric["class_recall"].items():
            recall_rows.append({"scope": "fold", "variant": variant, "fold": fold, "class_label": name, "recall": recall, "support": sum(int(r["label_index"]) == CLASS_NAMES.index(name) for r in rows)})
        for i, actual in enumerate(CLASS_NAMES):
            confusion_rows.append({"scope": "fold", "variant": variant, "fold": fold, "true_label": actual, **{f"predicted_{CLASS_NAMES[j]}": metric["confusion"][i][j] for j in range(3)}})
        fold_metrics.append({
            "variant": variant, "fold": fold, "n_subjects": metric["n_subjects"],
            "accuracy": metric["accuracy"], "balanced_accuracy": metric["balanced_accuracy"],
            "macro_f1": metric["macro_f1"],
            "recall_CP": metric["class_recall"]["CP"], "recall_ITW": metric["class_recall"]["ITW"],
            "recall_TD": metric["class_recall"]["TD"],
        })

    pooled_metrics: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        metric = metrics_for(rows)
        ci_low, ci_high = bootstrap_macro_f1_ci(rows, repetitions, seed + VARIANTS.index(variant) * 1009)
        fold_values = [row["macro_f1"] for row in fold_metrics if row["variant"] == variant]
        fold_mean = float(np.mean(fold_values)) if fold_values else math.nan
        fold_std = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else (0.0 if fold_values else math.nan)
        pooled_metrics.append({
            "variant": variant,
            "n_subjects": metric["n_subjects"],
            "folds_completed": len(fold_values),
            "expected_folds": expected_folds,
            "complete_five_fold_oof": int(len(fold_values) == expected_folds),
            "pooled_accuracy": metric["accuracy"],
            "pooled_balanced_accuracy": metric["balanced_accuracy"],
            "pooled_subject_macro_f1": metric["macro_f1"],
            "subject_macro_f1_bootstrap_95ci_low": ci_low,
            "subject_macro_f1_bootstrap_95ci_high": ci_high,
            "fold_macro_f1_mean": fold_mean,
            "fold_macro_f1_std": fold_std,
            "fold_macro_f1_mean_plusminus_std": f"{fold_mean:.6f} +/- {fold_std:.6f}" if fold_values else "",
        })
        for name, recall in metric["class_recall"].items():
            recall_rows.append({"scope": "pooled", "variant": variant, "fold": "all", "class_label": name, "recall": recall, "support": sum(int(r["label_index"]) == CLASS_NAMES.index(name) for r in rows)})
        for i, actual in enumerate(CLASS_NAMES):
            confusion_rows.append({"scope": "pooled", "variant": variant, "fold": "all", "true_label": actual, **{f"predicted_{CLASS_NAMES[j]}": metric["confusion"][i][j] for j in range(3)}})

    phase_rows: list[dict[str, Any]] = []
    by_phase: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in video_rows:
        row = dict(raw)
        try:
            fold = int(row["fold"])
        except (ValueError, KeyError):
            continue
        entropy = row.get("phase_entropy_normalized")
        if isinstance(entropy, str):
            entropy = float(entropy) if entropy else None
        occupancy = row.get("phase_occupancy")
        if isinstance(occupancy, str):
            occupancy = json.loads(occupancy) if occupancy else None
        row["phase_entropy_normalized"] = entropy
        row["phase_occupancy"] = occupancy
        by_phase[(str(row.get("variant", "")), fold)].append(row)
    for (variant, fold), rows in sorted(by_phase.items()):
        entropies = [float(row["phase_entropy_normalized"]) for row in rows if row.get("phase_entropy_normalized") is not None]
        occ = [to_vector(row["phase_occupancy"], 4) for row in rows if row.get("phase_occupancy") not in (None, "")]
        mean_occ = np.mean(occ, axis=0).tolist() if occ else [None] * 4
        phase_rows.append({
            "variant": variant,
            "fold": fold,
            "n_videos": len(rows),
            "mean_normalized_assignment_entropy": float(np.mean(entropies)) if entropies else "",
            "mean_video_level_occupancy_phase0": mean_occ[0],
            "mean_video_level_occupancy_phase1": mean_occ[1],
            "mean_video_level_occupancy_phase2": mean_occ[2],
            "mean_video_level_occupancy_phase3": mean_occ[3],
            "interpretation": "mean across held-out videos; global has no phase assignments",
        })

    paired_rows = paired_bootstrap_cspt(
        normalized_subjects, ("global", "uniform", "unconstrained"), repetitions, seed + 91_007
    )
    atomic_csv(output_dir / "metrics_fold.csv", fold_metrics)
    atomic_csv(output_dir / "metrics_pooled.csv", pooled_metrics)
    atomic_csv(output_dir / "class_recall.csv", recall_rows)
    atomic_csv(output_dir / "subject_confusion.csv", confusion_rows)
    atomic_csv(output_dir / "phase_summary.csv", phase_rows)
    atomic_csv(output_dir / "paired_cspt_bootstrap.csv", paired_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-oof", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--seed", type=int, default=20261012)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = args.video_oof or args.output / "predictions" / "video_oof.csv"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video-level OOF file not found: {video_path}")
    video_rows = read_rows(video_path)
    subject_path = args.output / "predictions" / "subject_oof.csv"
    if subject_path.is_file() and args.video_oof is None:
        subject_rows = read_rows(subject_path)
    else:
        subject_rows = aggregate_subject_rows(video_rows)
    config = {
        "seed": args.seed,
        "evaluation": {"bootstrap_repetitions": args.bootstrap_repetitions or 2000, "bootstrap_seed": args.seed + 345_019},
    }
    write_summary(args.output, video_rows, subject_rows, config, expected_folds=5)
    print(f"Summary written under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
