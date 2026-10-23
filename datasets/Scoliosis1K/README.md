# Scoliosis1K inputs

Obtain the released silhouette PKLs and Scoliosis1K-Pose PKLs from the [official dataset project](https://zhouzi180.github.io/Scoliosis1K/), following its access agreement. This repository contains no dataset samples.

The primary model consumes two aligned roots:

```text
Scoliosis1K-skeleton-map-pkl/<sequence_id>/<class>/<view>/*.pkl
Scoliosis1K-pkl/<sequence_id>/<class>/<view>/*.pkl
```

The skeleton-map array contains aligned limb and joint heatmaps. Its model layout is `[frames, 2, 64, 64]`; silhouettes use `[frames, 64, 64]`. Keep sequence identifiers, class/view directories, and temporal ordering aligned across the two roots. Match the class naming used by the released evaluator: `negative`, `neutral`, `positive`.

For the OpenGait pose-to-map conversion, run from the repository root on a Linux CUDA/NCCL environment:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 datasets/pretreatment_heatmap.py --pose_data_path /path/to/Scoliosis1K-pose-pkl --save_root /path/to/Scoliosis1K-skeleton-map-pkl --dataset_name OUMVLP
```

`OUMVLP` is the existing converter's pose-format mode for this processing route, not a different training dataset. The conversion settings are stored in `configs/skeletongait/pretreatment_heatmap.yaml`.

Set `data_cfg.dataset_root` in the chosen configuration to the two prepared roots. Keep `Scoliosis1K_118.json` unchanged: 744 training and 749 test sequence entries. Sequence-directory identifiers in the loader must not be interpreted as proof of independent clinical participant identities.
