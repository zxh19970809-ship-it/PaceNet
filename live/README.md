# LIVE-GaitNeuroKids experiment

This is a separate CP/ITW/TD classification task with five-fold subject-independent evaluation. The reported analyzed cohort contains 55 participants. Its pooled CSPT macro-F1 is 65.41%; the three paired macro-F1 difference confidence intervals include zero. It is supplementary evidence on pediatric gait classification, not external clinical validation of AIS screening.

`temporal_models.py` uses two observed 512-D streams, a shared 1024-D assignment input, and average fusion of two modality tokens. The primary Scoliosis1K model retains its original checkpoint-compatible implementation. Do not substitute the LIVE model for that checkpoint.

## Inputs and dependencies

Obtain the dataset's VIDEO, DATASET and ANALYSIS archives and participant workbook from [LIVE-GaitNeuroKids version 1.1 on Zenodo](https://doi.org/10.5281/zenodo.21278362), as cited in the paper. This package includes no participant workbook, video, person-level manifest, fold assignments, or predictions. The scripts generate local manifests and QC records from an authorized dataset copy. Exact reference reproduction requires the same dataset release and accepted cohort.

```bash
pip install -r requirements-live.txt
```

Raw-video preparation also requires `ffprobe` on PATH and local `yolov8x-pose.pt` and `yolo11x-seg.pt` weights. Place those two weight files in `weights/`, or pass their paths using `--pose-weights` and `--seg-weights`. Do not commit them.

Run the following commands from the repository root. Replace `/path/to/LIVE-GaitNeuroKids` with the directory containing the original archives and workbook.

## 1. Source manifests

```bash
python live/scripts/inventory_archives.py --input-root /path/to/LIVE-GaitNeuroKids --output-root live
python live/scripts/inspect_subjects.py --xlsx /path/to/LIVE-GaitNeuroKids/PARTICIPANT_CHARACTERISTICS.xlsx --data-dir /path/to/LIVE-GaitNeuroKids --output-dir live --build
python live/scripts/prepare_candidate_manifest.py
```

These write local `live/manifests/videos.csv`, `subjects.csv`, and `eligible_videos.csv`. Candidate selection retains natural GAIT recordings from visits T0/T1/T2/C0 and left/right/sagittal views. All visits and views of a participant share the same subject group.

## 2. Candidate preprocessing and accepted cohort

Process the full candidate videos at 15 fps and record the existing track/validity QC decisions:

```bash
python live/scripts/prepare_video_inputs.py --input-zip /path/to/LIVE-GaitNeuroKids/VIDEO.zip --selection-csv live/manifests/eligible_videos.csv --eligible-manifest live/manifests/eligible_videos.csv --max-frames 0 --target-fps 15 --eligible-selection-finalized --report live/audit/agent4_candidate_preprocessing_progress.json --decisions-csv live/audit/agent4_candidate_decisions.csv --completion-marker live/audit/candidate_preprocessing_complete.json
python live/scripts/finalize_video_cohort.py
```

The finalizer writes `eligible_videos_final.csv` using the existing QC rules, a 720-slot cap, and at least 30 jointly valid frames. The scripts retain the original audit filenames so that their existing interfaces match.

Run the accepted-only pass to create the completion marker required by the CV runner:

```bash
python live/scripts/prepare_video_inputs.py --input-zip /path/to/LIVE-GaitNeuroKids/VIDEO.zip --selection-csv live/manifests/eligible_videos_final.csv --eligible-manifest live/manifests/eligible_videos_final.csv --max-frames 0 --target-fps 15 --eligible-selection-finalized --resume-inputs --report live/audit/agent4_final_preprocessing_progress.json --decisions-csv live/audit/agent4_final_decisions.csv
```

Prepared arrays go to `live/cache/video_inputs`. Config paths are relative to `live/experiment_config.json`; adjust `input_dir` if storing the cache elsewhere. No clip or preprocessing is generated merely by importing the model or requesting `--help`.

## 3. Evaluation protocol

Check prepared inputs before training:

```bash
python live/scripts/run_cv.py --config live/experiment_config.json --device cpu --check-only
```

Run the original four variants and five folds:

```bash
python live/scripts/run_cv.py --config live/experiment_config.json --device cuda:0
```

Use `--resume` to continue that run. Variants are `global`, `uniform`, `unconstrained`, and `cspt`. Subject groups remain disjoint across folds; checkpoint selection uses inner-validation subject macro-F1. Numerical settings, sampler logic, losses, and probability aggregation are retained from the experiment.

The runner writes its split plan, checkpoints, OOF predictions, and aggregate tables under `live/results/classification_cv_20261012`. To regenerate aggregate tables from an existing OOF file:

```bash
python live/scripts/summarize_oof.py --output live/results/classification_cv_20261012
```

Existing aggregate reference tables are in `../reproducibility/live/`. They contain no individual participant rows.
