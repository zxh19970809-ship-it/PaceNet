# PACE-Net

Implementation accompanying **PACE-Net: Phase-Aware Constrained Encoding for Video-Based Adolescent Scoliosis Screening**.

PACE-Net uses shared, constrained temporal assignments to organize aligned skeleton-map and silhouette descriptors into learned group tokens. Corresponding modality tokens are fused and contextualized before three-class prediction. The groups have no predefined physiological gait-phase meaning.

This source package contains the primary Scoliosis1K model, the historical DMR-equipped tokenizer controls, and the separate LIVE-GaitNeuroKids experiment. It is derived from OpenGait commit `b0d55456c2eb3576a568d9f5603eb439388a1e32`. Datasets, trained weights, individual predictions, and training logs are not included.

## Contents

| Path | Purpose |
|---|---|
| `opengait/modeling/models/pace_net.py` | Primary network and historical control switches |
| `opengait/modeling/phase_modeling.py` | Shared temporal tokenization and mass balancing |
| `opengait/modeling/losses/phase.py` | Structural regularization |
| `configs/pace_net/pace_net_scoliosis1k.yaml` | Primary Scoliosis1K configuration |
| `configs/pace_net/dmr_controls/` | Historical DMR-equipped matched controls |
| `datasets/Scoliosis1K/` | Official split and data-preparation instructions |
| `live/` | Separate LIVE preprocessing, models, and subject-independent CV |
| `reproducibility/` | Existing aggregate reference results |

## Installation

Use Python 3.10 or newer. Install a compatible PyTorch/torchvision build for your GPU, then install the dependencies:

```bash
pip install -r requirements.txt
```

The primary experiment's recorded environment used PyTorch 2.11.0+cu128, CUDA 12.8, scikit-learn 1.7.2, and PyYAML 6.0.3 on an RTX 5090 D. Package checks were also run with Python 3.11, PyTorch 2.9.0+cu130, and torchvision 0.24.0. Training and full-dataset evaluation require a CUDA GPU; the unit tests use CPU.

## Scoliosis1K data

Obtain Scoliosis1K and its released pose data from the [official project](https://zhouzi180.github.io/Scoliosis1K/), following its access conditions. Prepare aligned skeleton-map and silhouette PKLs as described in [the data README](datasets/Scoliosis1K/README.md).

Set the two `data_cfg.dataset_root` paths in `configs/pace_net/pace_net_scoliosis1k.yaml`, in this order:

1. Skeleton maps: two channels, 64 x 64.
2. Silhouettes: one channel, 64 x 64.

Keep the supplied `Scoliosis1K_118.json` split. The evaluation unit is a released sequence entry: 744 training sequences and 749 test sequences. The historical `subject_id` CSV column denotes the released sequence-directory identifier; it does not establish subject-independent AIS evaluation.

## Primary training and evaluation

Run from the repository root. The single-GPU launchers also support native Windows without NCCL:

```bash
python train_pace_net_single_gpu.py --cfgs configs/pace_net/pace_net_scoliosis1k.yaml
```

Evaluate an existing compatible checkpoint:

```bash
python evaluate_pace_net_single_gpu.py --cfgs configs/pace_net/pace_net_scoliosis1k.yaml --checkpoint /path/to/checkpoint.pt --output-dir evaluation/primary
```

The evaluator writes `metrics.json` and sequence-level `predictions.csv` with mean group logits, probabilities, class predictions, and assignment summaries. The reference checkpoint's reported accuracy is **95.19%** and macro-F1 is **94.35%**, recorded in [`reference_metrics.json`](reproducibility/reference_metrics.json). These are existing experimental results, not a guarantee for a newly trained run. Weights are not bundled; evaluating this reference result requires its matching checkpoint. The original checkpoint-compatible computation is retained.

The configuration records the complete training settings, including the class-balanced `[15, 2]` sampler, AMP, and ordered 30-frame training clips.

## Historical DMR-equipped controls

These three configurations share their own DMR-equipped setting:

| Configuration | Reported accuracy |
|---|---:|
| `configs/pace_net/dmr_controls/uniform_dmr.yaml` | 84.91% |
| `configs/pace_net/dmr_controls/unconstrained_dmr.yaml` | 88.12% |
| `configs/pace_net/dmr_controls/cspt_dmr.yaml` | 91.72% |

Use the same launchers with the relevant `--cfgs` argument. Their original batch size, precision, learning rate, and other numerical settings are retained. They are not strict component ablations of the 95.19% primary configuration.

## LIVE-GaitNeuroKids

See [`live/README.md`](live/README.md) for data preparation and five-fold subject-independent evaluation. Its CP/ITW/TD task and model implementation are separate from the primary AIS experiment. Existing aggregate reference tables are in `reproducibility/live/`; participant-level records are not included.

## Checks

```bash
python tests/test_phase_modeling.py
python tests/test_clinical_sampler.py
python train_pace_net_single_gpu.py --help
python evaluate_pace_net_single_gpu.py --help
python live/scripts/run_cv.py --help
```

## Attribution and use

Retain [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md). The bundled runtime includes OpenGait-derived code with its recorded academic-use restriction. This package does not assign a new license or broaden upstream permissions. Dataset access and redistribution follow the respective dataset terms.
