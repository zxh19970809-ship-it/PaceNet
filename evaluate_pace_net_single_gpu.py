"""Evaluate a PACE-Net checkpoint on its configured Scoliosis1K TEST_SET."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OPENGAIT_ROOT = PROJECT_ROOT / "opengait"
CLASS_NAMES = ("negative", "neutral", "positive")


def install_single_process_dist_shim() -> None:
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    torch.distributed.broadcast = lambda tensor, src=0: tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfgs",
        type=Path,
        default=PROJECT_ROOT / "configs/pace_net/pace_net_scoliosis1k.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        help="Override data_cfg.dataset_root; repeat for multi-modal roots.",
    )
    parser.add_argument(
        "--dataset-partition",
        type=Path,
        help="Override data_cfg.dataset_partition for machine-local evaluation.",
    )
    parser.add_argument(
        "--split-label",
        default=None,
        help=(
            "Human-readable split label stored in metrics.json. By default the "
            "official JSON is labelled official TEST_SET and every other JSON "
            "is explicitly labelled as a configured development split."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PACE-Net evaluation.")

    cfg_path = args.cfgs.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(OPENGAIT_ROOT))
    install_single_process_dist_shim()

    from modeling import models
    from utils import config_loader, get_msg_mgr, init_seeds

    cfgs = config_loader(str(cfg_path))
    if args.dataset_root:
        cfgs["data_cfg"]["dataset_root"] = [
            str(path.resolve()) for path in args.dataset_root
        ]
    if args.dataset_partition:
        cfgs["data_cfg"]["dataset_partition"] = str(
            args.dataset_partition.resolve()
        )
    cfgs["data_cfg"]["num_workers"] = args.num_workers
    cfgs["evaluator_cfg"]["sampler"]["batch_size"] = 1
    # Instantiate the inference graph first. Training-only loss buffers are
    # deliberately absent from it and are handled explicitly below.
    cfgs["evaluator_cfg"]["restore_hint"] = 0

    msg_mgr = get_msg_mgr()
    msg_mgr.init_logger(str(output_dir), log_to_file=True)
    init_seeds(2026)
    torch.cuda.set_device(0)

    model_class = getattr(models, cfgs["model_cfg"]["model"])
    model = model_class(cfgs, training=False)
    checkpoint_payload = torch.load(
        checkpoint, map_location=torch.device("cuda", 0), weights_only=True
    )
    inference_state = {
        key: value for key, value in checkpoint_payload["model"].items()
        if not key.startswith("loss_aggregator.")
    }
    incompatible = model.load_state_dict(inference_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch after filtering training-only loss state: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        inference = model.inference(rank=0)
    peak_cuda_memory_gib = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

    loader = model.test_loader
    statuses = list(loader.dataset.types_list)
    subject_ids = list(loader.dataset.label_list)
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    true_ids = np.asarray([label_map[status] for status in statuses])
    logits = np.asarray(inference["embeddings"]).mean(axis=-1)
    pred_ids = logits.argmax(axis=1)
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=[0, 1, 2], zero_division=0
    )
    matrix = confusion_matrix(true_ids, pred_ids, labels=[0, 1, 2])
    phase_occupancy = inference.get("phase_occupancy")
    phase_entropy = inference.get("phase_entropy")
    partition_path = Path(cfgs["data_cfg"]["dataset_partition"])
    split_label = args.split_label
    if split_label is None:
        split_label = (
            "official TEST_SET"
            if partition_path.name == "Scoliosis1K_118.json"
            else f"configured development TEST_SET ({partition_path.stem})"
        )
    metrics = {
        "split": split_label,
        "sample_count": int(len(true_ids)),
        "accuracy": float(accuracy_score(true_ids, pred_ids)),
        "macro_precision": float(
            precision_score(true_ids, pred_ids, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(true_ids, pred_ids, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(true_ids, pred_ids, average="macro", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "positive_to_neutral_count": int(matrix[2, 1]),
        "phase": (
            None
            if phase_occupancy is None
            else {
                "mean_occupancy": np.asarray(phase_occupancy).mean(axis=0).tolist(),
                "min_occupancy": float(np.asarray(phase_occupancy).min()),
                "max_occupancy": float(np.asarray(phase_occupancy).max()),
                "mean_normalized_entropy": (
                    None
                    if phase_entropy is None
                    else float(np.asarray(phase_entropy).mean())
                ),
            }
        ),
        "peak_cuda_memory_gib": float(peak_cuda_memory_gib),
        "class_order": list(CLASS_NAMES),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "split_json": str(partition_path),
        "split_sha256": sha256(partition_path),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    final_probabilities = torch.softmax(
        torch.from_numpy(logits).float(), dim=1
    ).numpy()
    phase_occupancy_array = (
        None if phase_occupancy is None else np.asarray(phase_occupancy)
    )
    phase_entropy_array = (
        None if phase_entropy is None else np.asarray(phase_entropy).reshape(-1)
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = [
            "subject_id", "true_label", "predicted_label",
            "logit_negative", "logit_neutral", "logit_positive",
            "prob_negative", "prob_neutral", "prob_positive",
        ]
        if phase_occupancy_array is not None:
            header.extend(
                f"phase_{index + 1}_occupancy"
                for index in range(phase_occupancy_array.shape[1])
            )
        if phase_entropy_array is not None:
            header.append("phase_normalized_entropy")
        writer.writerow(header)
        for index, subject_id in enumerate(subject_ids):
            row = [
                subject_id,
                statuses[index],
                CLASS_NAMES[pred_ids[index]],
                *[float(value) for value in logits[index]],
                *[float(value) for value in final_probabilities[index]],
            ]
            if phase_occupancy_array is not None:
                row.extend(float(value) for value in phase_occupancy_array[index])
            if phase_entropy_array is not None:
                row.append(float(phase_entropy_array[index]))
            writer.writerow(row)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
