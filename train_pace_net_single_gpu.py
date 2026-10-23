"""Windows single-GPU launcher for the Scoliosis1K PACE-Net experiment.

The released OpenGait entry point always initialises NCCL, which is not
available on native Windows.  This launcher preserves the OpenGait model,
sampler, optimiser, scheduler, loss and checkpoint code, while supplying the
rank/world-size operations required by their single-process path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
OPENGAIT_ROOT = PROJECT_ROOT / "opengait"


def _install_single_process_dist_shim() -> None:
    """Provide rank semantics without creating a Windows process group."""
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    torch.distributed.broadcast = lambda tensor, src=0: tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PACE-Net with the OpenGait engine on one Windows GPU."
    )
    parser.add_argument(
        "--cfgs",
        type=Path,
        default=PROJECT_ROOT / "configs/pace_net/pace_net_scoliosis1k.yaml",
    )
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--total-iter", type=int, default=None)
    parser.add_argument("--save-iter", type=int, default=None)
    parser.add_argument("--log-iter", type=int, default=None)
    parser.add_argument("--batch-p", type=int, default=None)
    parser.add_argument("--batch-k", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--dataset-partition",
        type=Path,
        default=None,
        help="Override TRAIN_SET/TEST_SET JSON (used by the final official run).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Override the aligned skeleton-map and silhouette dataset roots. "
            "pass skeleton-map root first and silhouette root second."
        ),
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable in-training TEST_SET evaluation; the final evaluator runs once.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--restore-hint", type=int, default=None)
    parser.add_argument(
        "--fast-cudnn",
        action="store_true",
        help="Enable cuDNN benchmark mode; faster but less deterministic.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_overrides(cfgs: dict, args: argparse.Namespace) -> None:
    trainer = cfgs["trainer_cfg"]
    sampler = trainer["sampler"]
    if args.save_name is not None:
        trainer["save_name"] = args.save_name
    if args.total_iter is not None:
        trainer["total_iter"] = args.total_iter
    if args.save_iter is not None:
        trainer["save_iter"] = args.save_iter
    if args.log_iter is not None:
        trainer["log_iter"] = args.log_iter
    if args.frames is not None:
        sampler["frames_num_fixed"] = args.frames
    if args.lr is not None:
        cfgs["optimizer_cfg"]["lr"] = args.lr
    if args.num_workers is not None:
        cfgs["data_cfg"]["num_workers"] = args.num_workers
    if args.dataset_partition is not None:
        cfgs["data_cfg"]["dataset_partition"] = str(
            args.dataset_partition.resolve()
        )
    if args.dataset_root is not None:
        cfgs["data_cfg"]["dataset_root"] = [
            str(path.resolve()) for path in args.dataset_root
        ]
    if args.no_validation:
        trainer["with_test"] = False
    if args.batch_p is not None or args.batch_k is not None:
        current_p, current_k = sampler["batch_size"]
        sampler["batch_size"] = [
            args.batch_p if args.batch_p is not None else current_p,
            args.batch_k if args.batch_k is not None else current_k,
        ]
    if args.restore_hint is not None:
        trainer["restore_hint"] = args.restore_hint


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; PACE-Net training requires an NVIDIA GPU.")

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(OPENGAIT_ROOT))
    _install_single_process_dist_shim()

    # Imports must follow the rank shim because the stock OpenGait utilities
    # query torch.distributed.get_rank() during initialisation.
    from modeling import models
    from utils import config_loader, get_msg_mgr, init_seeds, params_count

    cfg_path = args.cfgs.resolve()
    cfgs = config_loader(str(cfg_path))
    _apply_overrides(cfgs, args)
    trainer = cfgs["trainer_cfg"]

    output_path = (
        PROJECT_ROOT
        / "output"
        / cfgs["data_cfg"]["dataset_name"]
        / cfgs["model_cfg"]["model"]
        / trainer["save_name"]
    )
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = cfgs["model_cfg"]["model"]
    model_source = {
        "PACENet": PROJECT_ROOT / "opengait/modeling/models/pace_net.py",
    }.get(model_name)
    source_files = {
        "launcher": Path(__file__).resolve(),
        "sampler": PROJECT_ROOT / "opengait/data/sampler.py",
    }
    if model_source is not None:
        source_files["model"] = model_source
    if model_name == "PACENet":
        source_files.update(
            {
                "phase_modules": PROJECT_ROOT
                / "opengait/modeling/phase_modeling.py",
                "phase_losses": PROJECT_ROOT
                / "opengait/modeling/losses/phase.py",
            }
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "config_path": str(cfg_path),
        "config_sha256": _sha256(cfg_path),
        "split_sha256": _sha256(Path(cfgs["data_cfg"]["dataset_partition"])),
        "source_sha256": {
            name: _sha256(path) for name, path in source_files.items()
        },
        "sampler_type": trainer["sampler"].get("type"),
        "phase_ablation_stage": cfgs["model_cfg"].get("ablation_stage"),
        "warm_start": (
            trainer["restore_hint"]
            if isinstance(trainer.get("restore_hint"), str)
            else None
        ),
    }
    (output_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "effective_config.yaml").write_text(
        yaml.safe_dump(cfgs, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    msg_mgr = get_msg_mgr()
    msg_mgr.init_manager(
        str(output_path),
        log_to_file=True,
        log_iter=trainer["log_iter"],
        iteration=(trainer["restore_hint"]
                   if isinstance(trainer["restore_hint"], int) else 0),
    )
    msg_mgr.log_info(trainer)
    msg_mgr.log_info(manifest)

    init_seeds(args.seed, cuda_deterministic=not args.fast_cudnn)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    model_cfg = cfgs["model_cfg"]
    msg_mgr.log_info(model_cfg)
    model_class = getattr(models, model_cfg["model"])
    model = model_class(cfgs, training=True)
    if trainer["fix_BN"]:
        model.fix_BN()
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Single-GPU model initialization finished.")

    try:
        model_class.run_train(model)
    finally:
        peak_gib = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        msg_mgr.log_info("Peak CUDA memory allocated: %.3f GiB", peak_gib)
        msg_mgr.writer.flush()
        msg_mgr.writer.close()


if __name__ == "__main__":
    # Required by Windows spawn-based DataLoader workers.
    main()
