"""Run one real PACE-Net training batch through backward without an update.

The report is deliberately machine-readable so the same command can guard
future A0/A1/A2 configs before a long run.  ``optimizer.step`` and
``scheduler.step`` are never called.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
OPENGAIT_ROOT = PROJECT_ROOT / "opengait"
CLASSES = ("negative", "neutral", "positive")


def install_single_process_dist_shim() -> None:
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    torch.distributed.broadcast = lambda tensor, src=0: tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfgs",
        type=Path,
        default=PROJECT_ROOT / "configs/pace_net/pace_net_scoliosis1k.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output/phase_a1a2_train_batch_smoke/report.json",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--probe-batches", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--disable-amp", action="store_true",
        help="Run the same batch in float32 to isolate mixed-precision overflow.",
    )
    parser.add_argument(
        "--require-balanced-sampler",
        action="store_true",
        help="Fail unless every probed P x K batch contains all three classes.",
    )
    return parser.parse_args()


def as_number(value: object) -> object:
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
        return float(tensor.item()) if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def gradient_summary(module: torch.nn.Module | None) -> dict[str, object]:
    if module is None:
        return {"present": False, "tensor_count": 0, "nonzero_tensors": 0,
                "all_finite": True, "l2_norm": 0.0}
    named_gradients = [(name, parameter.grad) for name, parameter in module.named_parameters()
                       if parameter.requires_grad and parameter.grad is not None]
    gradients = [gradient for _, gradient in named_gradients]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nonzero = sum(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    squared_norm = sum(float(gradient.detach().float().square().sum())
                       for gradient in gradients)
    return {
        "present": True,
        "tensor_count": len(gradients),
        "nonzero_tensors": nonzero,
        "all_finite": finite,
        "l2_norm": math.sqrt(squared_norm),
        "nonfinite_parameters": [
            name for name, gradient in named_gradients
            if not bool(torch.isfinite(gradient).all())
        ],
    }


def parameter_probe(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    probe: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            probe[name] = parameter.detach().reshape(-1)[:16].cpu().clone()
        if len(probe) == 12:
            break
    return probe


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training-batch smoke test")
    if args.probe_batches < 1:
        raise ValueError("probe-batches must be positive")

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(OPENGAIT_ROOT))
    install_single_process_dist_shim()

    from modeling import models
    from utils import config_loader, get_msg_mgr, init_seeds

    cfg_path = args.cfgs.resolve()
    cfgs = config_loader(str(cfg_path))
    cfgs["data_cfg"]["num_workers"] = args.num_workers
    cfgs["trainer_cfg"]["restore_hint"] = 0
    cfgs["trainer_cfg"]["with_test"] = False
    if args.disable_amp:
        cfgs["trainer_cfg"]["enable_float16"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    get_msg_mgr().init_logger(str(args.output.parent), log_to_file=False)
    init_seeds(args.seed, cuda_deterministic=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(0)

    model_class = getattr(models, cfgs["model_cfg"]["model"])
    model = model_class(cfgs, training=True)
    dataset = model.train_loader.dataset
    sampler = model.train_loader.batch_sampler

    probe_iterator = iter(sampler)
    coverage_histogram: Counter[int] = Counter()
    class_count_totals: Counter[str] = Counter()
    complete_batches = 0
    for _ in range(args.probe_batches):
        indices = next(probe_iterator)
        batch_classes = [dataset.types_list[index] for index in indices]
        present = set(batch_classes)
        coverage_histogram[len(present)] += 1
        class_count_totals.update(batch_classes)
        complete_batches += int(present == set(CLASSES))

    # Make the actual loader batch independent of how many probe batches were
    # requested, which keeps the backward evidence reproducible.
    init_seeds(args.seed, cuda_deterministic=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    raw_batch = next(iter(model.train_loader))
    clinical_labels = list(raw_batch[2])
    subject_indices = list(raw_batch[1])
    subject_ids = [dataset.label_set[index] for index in subject_indices]
    expected_class_indices = [CLASSES.index(label) for label in clinical_labels]

    model.train()
    model.optimizer.zero_grad(set_to_none=True)
    optimizer_state_before = len(model.optimizer.state)
    parameters_before = parameter_probe(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    prepared = model.inputs_pretreament(raw_batch)
    with torch.amp.autocast("cuda", enabled=model.engine_cfg["enable_float16"]):
        result = model(prepared)
    training_feat = result["training_feat"]
    inference_feat = result["inference_feat"]
    loss_sum, loss_info = model.loss_aggregator(training_feat)
    if model.engine_cfg["enable_float16"]:
        model.Scaler.scale(loss_sum).backward()
        model.Scaler.unscale_(model.optimizer)
    else:
        loss_sum.backward()
    torch.cuda.synchronize()

    parameters_unchanged = all(
        torch.equal(before, dict(model.named_parameters())[name].detach()
                    .reshape(-1)[:16].cpu())
        for name, before in parameters_before.items()
    )
    softmax_labels = training_feat["softmax"]["labels"].detach().cpu().tolist()
    triplet_labels = training_feat["triplet"]["labels"].detach().cpu().tolist()
    logits = inference_feat["embeddings"]
    assignments = inference_feat.get("phase_assignments")
    occupancy = inference_feat.get("phase_occupancy")

    modules = {
        name: gradient_summary(getattr(model, name, None))
        for name in (
            "map_backbone", "sil_backbone", "motion_backbone",
            "phase_tokenizer", "phase_context", "FCs", "BNNecks",
        )
    }
    all_gradients = gradient_summary(model)
    checks = {
        "total_loss_finite": bool(torch.isfinite(loss_sum)),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "softmax_labels_match_clinical_types": softmax_labels == expected_class_indices,
        "triplet_labels_match_subject_indices": triplet_labels == subject_indices,
        "gradients_finite": bool(all_gradients["all_finite"]),
        "gradients_nonzero": int(all_gradients["nonzero_tensors"]) > 0,
        "parameters_unchanged_without_optimizer_step": parameters_unchanged,
        "optimizer_state_unchanged": len(model.optimizer.state) == optimizer_state_before,
    }
    if assignments is not None and occupancy is not None:
        valid_assignments = assignments[training_feat["phase_regularization"]["valid_mask"]]
        assignment_mode = getattr(model, "phase_assignment_mode", "soft")
        checks.update({
            "phase_assignments_finite": bool(torch.isfinite(assignments).all()),
            "valid_phase_probabilities_sum_to_one": bool(torch.allclose(
                valid_assignments.sum(dim=-1),
                torch.ones_like(valid_assignments.sum(dim=-1)),
                atol=2e-3,
            )),
            "phase_not_immediately_collapsed": (
                float(occupancy.detach().min()) > 0.05
                and float(occupancy.detach().max()) < 0.80
            ),
        })
        if assignment_mode == "uniform_bins":
            checks.update({
                "uniform_assignments_are_hard": bool(torch.all(
                    (valid_assignments == 0) | (valid_assignments == 1)
                )),
                "capacity_matcher_is_in_autograd_graph": (
                    modules["phase_tokenizer"]["tensor_count"] > 0
                ),
                "capacity_matcher_has_zero_functional_gradient": (
                    modules["phase_tokenizer"]["l2_norm"] == 0.0
                ),
            })
        else:
            checks["phase_tokenizer_has_gradient"] = (
                modules["phase_tokenizer"]["l2_norm"] > 0.0
            )
    if args.require_balanced_sampler:
        checks["all_probed_batches_cover_three_classes"] = (
            complete_batches == args.probe_batches
        )

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False,
        "optimizer_step_called": False,
        "scheduler_step_called": False,
        "config": str(cfg_path),
        "partition": cfgs["data_cfg"]["dataset_partition"],
        "sampler": {
            "type": type(sampler).__name__,
            "batch_size": cfgs["trainer_cfg"]["sampler"]["batch_size"],
            "probed_batches": args.probe_batches,
            "complete_three_class_batches": complete_batches,
            "coverage_histogram": dict(sorted(coverage_histogram.items())),
            "sampled_class_totals": dict(class_count_totals),
        },
        "real_batch": {
            "subject_ids": subject_ids,
            "subject_indices": subject_indices,
            "clinical_labels": clinical_labels,
            "clinical_class_indices": expected_class_indices,
            "input_shapes": [list(tensor.shape) for tensor in prepared[0]],
            "logits_shape": list(logits.shape),
        },
        "loss": {
            "total": float(loss_sum.detach()),
            "configured_weights": {
                entry["log_prefix"]: entry.get("loss_term_weight", 1.0)
                for entry in cfgs["loss_cfg"]
            },
            "details": {key: as_number(value) for key, value in loss_info.items()},
        },
        "phase": None if assignments is None else {
            "assignment_mode": getattr(model, "phase_assignment_mode", "soft"),
            "assignment_shape": list(assignments.shape),
            "mean_occupancy": occupancy.detach().float().mean(dim=0).cpu().tolist(),
            "min_occupancy": float(occupancy.detach().min()),
            "max_occupancy": float(occupancy.detach().max()),
            "mean_normalized_entropy": float((
                -(assignments.clamp_min(1e-8).float()
                  * assignments.clamp_min(1e-8).float().log()).sum(dim=-1)
                / math.log(assignments.size(-1))
            )[training_feat["phase_regularization"]["valid_mask"]].mean().detach()),
        },
        "gradients": {"all": all_gradients, **modules},
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "phase_tokenizer": sum(
                parameter.numel()
                for parameter in model.phase_tokenizer.parameters()
            ) if getattr(model, "phase_tokenizer", None) is not None else 0,
        },
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(0) / 1024**3,
        },
        "checks": checks,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit("Smoke checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
