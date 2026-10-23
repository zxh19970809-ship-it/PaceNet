#!/usr/bin/env python
"""Run subject-independent five-fold CV for the four LIVE-GaitNeuroKids models.

The runner consumes the root-finalized eligible_videos.csv and Agent 4 NPZs.
It deliberately never accesses the original video archives. Training batches,
including ordered frame subsamples, are planned from a data-only RNG so every
variant receives identical samples within each fold and epoch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from temporal_models import CLASS_TO_INDEX, build_temporal_model
from summarize_oof import aggregate_subject_rows, metrics_for, write_summary


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "experiment_config.json"
VARIANTS = ("global", "uniform", "unconstrained", "cspt")
VISITS = {"T0", "T1", "T2", "C0"}
SCHEMA = "live-video-input-v1"
RUNNER_SCHEMA = "live-gait-cv-run-v1"
RAW_INPUT_CACHE_MAX_ITEMS = 1_024
RAW_BYTES_PER_FRAME = 2 * 64 * 64 * np.dtype(np.float16).itemsize + 1 * 64 * 64 * np.dtype(np.uint8).itemsize + np.dtype(np.bool_).itemsize


@dataclass(frozen=True)
class VideoRecord:
    row_index: int
    zip_member: str
    subject_id: str
    class_label: str
    label_index: int
    view: str
    visit: str
    npz_path: Path
    frame_count: int
    temporal_slots: tuple[int, ...]
    valid_indices: tuple[int, ...]
    valid_mask: tuple[bool, ...]


@dataclass(frozen=True)
class RawInputArrays:
    skeleton_maps: np.ndarray
    silhouette: np.ndarray
    valid_frame_mask: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.skeleton_maps.nbytes + self.silhouette.nbytes + self.valid_frame_mask.nbytes)


class ReadOnlyNPZArrayCache:
    """Process-local LRU of raw-dtype arrays from immutable, validated NPZ inputs."""

    def __init__(self, max_items: int = RAW_INPUT_CACHE_MAX_ITEMS) -> None:
        self.max_items = int(max_items)
        self._items: OrderedDict[tuple[str, int, int], RawInputArrays] = OrderedDict()
        self._resident_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, path: Path) -> RawInputArrays:
        resolved = path.resolve()
        stat = resolved.stat()
        path_key = str(resolved)
        key = (path_key, int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            self.hits += 1
            return cached

        # A changed file gets a distinct key and replaces any stale in-process
        # entry for the same path rather than consuming a second LRU slot.
        stale_keys = [old_key for old_key in self._items if old_key[0] == path_key]
        for old_key in stale_keys:
            stale = self._items.pop(old_key)
            self._resident_bytes -= stale.nbytes

        with np.load(resolved, allow_pickle=False) as data:
            arrays = RawInputArrays(
                skeleton_maps=np.asarray(data["skeleton_maps"]),
                silhouette=np.asarray(data["silhouette"]),
                valid_frame_mask=np.asarray(data["valid_frame_mask"]),
            )
        if arrays.skeleton_maps.dtype != np.float16 or arrays.silhouette.dtype != np.uint8 or arrays.valid_frame_mask.dtype != np.bool_:
            raise ValueError(
                f"Raw cache input dtypes changed after validation for {resolved}: "
                f"maps={arrays.skeleton_maps.dtype}, silhouette={arrays.silhouette.dtype}, "
                f"valid={arrays.valid_frame_mask.dtype}"
            )
        for array in (arrays.skeleton_maps, arrays.silhouette, arrays.valid_frame_mask):
            array.setflags(write=False)
        self._items[key] = arrays
        self._resident_bytes += arrays.nbytes
        self.misses += 1
        while len(self._items) > self.max_items:
            _, evicted = self._items.popitem(last=False)
            self._resident_bytes -= evicted.nbytes
        return arrays

    def snapshot(self) -> dict[str, int]:
        return {
            "loaded_items": len(self._items),
            "resident_bytes": self._resident_bytes,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "max_items": self.max_items,
        }


RAW_INPUT_CACHE = ReadOnlyNPZArrayCache()


def estimate_raw_cache_bytes(records: Sequence[VideoRecord]) -> int:
    unique_frames: dict[str, int] = {}
    for record in records:
        unique_frames.setdefault(str(record.npz_path.resolve()), int(record.frame_count))
    return int(sum(frame_count * RAW_BYTES_PER_FRAME for frame_count in unique_frames.values()))


def sha_key(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
    os.replace(temp, path)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_full_preprocessing(
    eligible_csv: Path, config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    settings = config["preprocessing"]
    marker_path = resolve_config_path(settings["completion_marker"], config_path.resolve())
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Formal CV requires Agent 4's full-video completion marker: {marker_path}. "
            "A short smoke cache must not enter the experiment."
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != settings["required_schema"]:
        raise ValueError(
            f"Preprocessing marker schema must be {settings['required_schema']!r}; got {marker.get('schema')!r}"
        )
    if marker.get("complete") is not True:
        raise ValueError("Preprocessing marker does not report complete=true")
    if int(marker.get("max_frames", -1)) != int(settings["full_video_mode_max_frames"]):
        raise ValueError("Preprocessing marker is not from full-video mode (max_frames must be 0)")
    if settings.get("require_finalized_selection", True) and marker.get("eligible_selection_finalized") is not True:
        raise ValueError("Preprocessing marker must certify that eligible selection was finalized")
    expected_manifest_hash = file_sha256(eligible_csv)
    if marker.get("eligible_manifest_sha256") != expected_manifest_hash:
        raise ValueError("Preprocessing marker was made for a different eligible_videos.csv")
    expected_count = int(marker.get("eligible_video_count", marker.get("video_count", -1)))
    processed_count = int(marker.get("processed_video_count", marker.get("completed_video_count", -1)))
    manifest_count = len(read_csv(eligible_csv))
    if expected_count != manifest_count or processed_count != manifest_count:
        raise ValueError(
            f"Preprocessing marker counts ({expected_count} eligible, {processed_count} processed) "
            f"must both equal the finalized manifest row count ({manifest_count})"
        )
    marker_fps = marker.get("target_fps")
    expected_fps = settings.get("expected_target_fps")
    if expected_fps is not None and (
        marker_fps is None or not math.isclose(float(marker_fps), float(expected_fps), rel_tol=0.0, abs_tol=1e-3)
    ):
        raise ValueError(f"Preprocessing marker target_fps must be {expected_fps}")
    marker_hash = file_sha256(marker_path)
    return marker, marker_hash


def first_value(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def resolve_npz_path(row: dict[str, str], eligible_csv: Path, input_dir: Path) -> Path:
    explicit = first_value(row, ("npz_path", "input_npz", "output_npz", "cache_path"))
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = input_dir / path
        return path
    member = first_value(row, ("zip_member",))
    if not member:
        raise ValueError("eligible_videos.csv needs a nonempty zip_member column")
    return input_dir / f"{sha_key(member)}.npz"


def cap_temporal_slots(frame_count: int, limit: int) -> tuple[int, ...]:
    if frame_count <= limit:
        return tuple(range(frame_count))
    slots = np.linspace(0, frame_count - 1, num=limit).round().astype(np.int64)
    if len(np.unique(slots)) != limit:
        raise ValueError(f"Could not form {limit} unique temporal slots from {frame_count} frames")
    return tuple(int(x) for x in slots)


def read_npz_record(
    raw_row: dict[str, str],
    row_index: int,
    eligible_csv: Path,
    input_dir: Path,
    max_frames: int,
    train_frames: int,
    expected_target_fps: float = 15.0,
) -> VideoRecord:
    zip_member = first_value(raw_row, ("zip_member",))
    subject_id = first_value(raw_row, ("subject_id",))
    class_label = first_value(raw_row, ("class_label",)).upper()
    view = first_value(raw_row, ("view", "view_hint")).lower() or "unspecified"
    visit = first_value(raw_row, ("visit", "visit_hint", "phase")).upper()
    if not zip_member or not subject_id:
        raise ValueError(f"Manifest row {row_index + 2} needs zip_member and subject_id")
    if class_label not in CLASS_TO_INDEX:
        raise ValueError(f"Manifest row {row_index + 2}: unsupported class_label {class_label!r}")
    manifest_label = first_value(raw_row, ("label",))
    if manifest_label and int(manifest_label) != CLASS_TO_INDEX[class_label]:
        raise ValueError(
            f"Manifest row {row_index + 2}: label={manifest_label!r} disagrees with "
            f"the required {class_label}={CLASS_TO_INDEX[class_label]} mapping"
        )
    if visit not in VISITS:
        raise ValueError(f"Manifest row {row_index + 2}: visit must be one of {sorted(VISITS)}, got {visit!r}")
    if view not in {"left", "right", "sagittal"}:
        raise ValueError(f"Manifest row {row_index + 2}: view must be left, right, or sagittal, got {view!r}")
    if class_label == "TD" and visit != "C0":
        raise ValueError(f"TD row {zip_member!r} must use visit C0, got {visit!r}")
    if class_label in {"CP", "ITW"} and visit == "C0":
        raise ValueError(f"{class_label} row {zip_member!r} cannot use TD visit C0")
    video_kind = first_value(raw_row, ("video_kind",)).lower()
    if video_kind and video_kind not in {"walking", "gait", "natural_gait"}:
        raise ValueError(f"Manifest row {row_index + 2} is not a natural walking video: {video_kind!r}")
    qc_status = first_value(raw_row, ("qc_status",)).lower()
    if qc_status and qc_status not in {"ok", "eligible", "pass", "passed"}:
        raise ValueError(f"Manifest row {row_index + 2} did not pass preprocessing QC: {qc_status!r}")

    npz_path = resolve_npz_path(raw_row, eligible_csv, input_dir)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ input missing for {zip_member}: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "schema" not in data or str(data["schema"].item()) != SCHEMA:
            raise ValueError(f"Unexpected NPZ schema in {npz_path}; expected {SCHEMA}")
        required = {
            "video_id", "skeleton_maps", "silhouette", "frame_indices", "valid_frame_mask",
            "pose_valid_mask", "mask_valid_mask", "subject_id", "class_label", "view",
            "visit", "phase", "zip_member", "is_smoke", "max_frames", "target_fps",
            "sampling_mode", "source_fps", "source_frame_count", "metadata_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{npz_path} missing NPZ keys: {missing}")
        skeleton_array = np.asarray(data["skeleton_maps"])
        silhouette_array = np.asarray(data["silhouette"])
        skeleton_shape = skeleton_array.shape
        silhouette_shape = silhouette_array.shape
        frame_count = int(skeleton_shape[0]) if skeleton_shape else 0
        valid_mask_raw = np.asarray(data["valid_frame_mask"])
        pose_mask_raw = np.asarray(data["pose_valid_mask"])
        silhouette_mask_raw = np.asarray(data["mask_valid_mask"])
        if any(mask.dtype != np.bool_ for mask in (valid_mask_raw, pose_mask_raw, silhouette_mask_raw)):
            raise ValueError(f"{npz_path} validity masks must have bool dtype")
        valid_mask_array = valid_mask_raw.reshape(-1)
        pose_mask = pose_mask_raw.reshape(-1)
        silhouette_mask = silhouette_mask_raw.reshape(-1)
        frame_indices = np.asarray(data["frame_indices"])
        metadata_raw = data["metadata_json"].item()
        metadata = json.loads(str(metadata_raw)) if metadata_raw else {}
        if len(skeleton_shape) != 4 or skeleton_shape[1:] != (2, 64, 64):
            raise ValueError(f"{npz_path} skeleton_maps must be [T,2,64,64], got {skeleton_shape}")
        if len(silhouette_shape) != 4 or silhouette_shape != (frame_count, 1, 64, 64):
            raise ValueError(f"{npz_path} silhouette must be [T,1,64,64], got {silhouette_shape}")
        if skeleton_array.dtype != np.float16:
            raise ValueError(f"{npz_path} skeleton_maps must be float16, got {skeleton_array.dtype}")
        if silhouette_array.dtype != np.uint8:
            raise ValueError(f"{npz_path} silhouette must be uint8 [0,255], got {silhouette_array.dtype}")
        if not np.isfinite(skeleton_array).all() or skeleton_array.min(initial=0) < 0 or skeleton_array.max(initial=0) > 1:
            raise ValueError(f"{npz_path} skeleton_maps must be finite float16 in [0,1]")
        if silhouette_array.size and not np.isin(np.unique(silhouette_array), (0, 255)).all():
            raise ValueError(f"{npz_path} silhouette must contain only 0 and 255")
        if any(array.size != frame_count for array in (valid_mask_array, pose_mask, silhouette_mask)):
            raise ValueError(f"{npz_path} validity mask lengths do not match its video arrays")
        if not np.array_equal(valid_mask_array, pose_mask & silhouette_mask):
            raise ValueError(f"{npz_path} valid_frame_mask must equal pose_valid_mask AND mask_valid_mask")
        if frame_indices.shape != (frame_count,) or frame_indices.dtype != np.int32:
            raise ValueError(f"{npz_path} frame_indices must be int32[T]")
        if frame_count == 0 or np.any(np.diff(frame_indices.astype(np.int64)) <= 0):
            raise ValueError(f"{npz_path} frame_indices must be a nonempty, strictly increasing sequence")
        if data["is_smoke"].dtype != np.bool_ or data["max_frames"].dtype != np.int32:
            raise ValueError(f"{npz_path} is_smoke/max_frames fields have invalid dtypes")
        if data["target_fps"].dtype != np.float32 or data["source_fps"].dtype != np.float32:
            raise ValueError(f"{npz_path} target_fps/source_fps fields must be float32")
        if data["source_frame_count"].dtype != np.int32:
            raise ValueError(f"{npz_path} source_frame_count must be int32")
        if bool(data["is_smoke"].item()) or int(data["max_frames"].item()) != 0:
            raise ValueError(f"{npz_path} is a smoke or capped artifact and cannot enter formal CV")
        target_fps = float(data["target_fps"].item())
        expected_fps = float(expected_target_fps)
        if not math.isclose(target_fps, expected_fps, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"{npz_path} target_fps must be {expected_fps:g}")
        expected_sampling = f"uniform_{expected_fps:g}fps"
        if str(data["sampling_mode"].item()) != expected_sampling:
            raise ValueError(f"{npz_path} sampling_mode must be {expected_sampling!r}")
        source_count = int(data["source_frame_count"].item())
        source_fps = float(data["source_fps"].item())
        video_metadata = metadata.get("video")
        if not isinstance(video_metadata, dict):
            raise ValueError(f"{npz_path} metadata_json.video must be an object with original video timing fields")
        try:
            # ffprobe's `fps` is authoritative when available. Agent 4's
            # extraction path can fall back to OpenCV's original-precision
            # rate, so accept that metadata field only when ffprobe has none.
            metadata_fps_value = video_metadata.get("fps") or video_metadata.get("opencv_fps")
            source_fps_exact = float(metadata_fps_value)
            metadata_source_count = int(video_metadata["opencv_frame_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{npz_path} metadata_json.video must contain numeric fps (or opencv_fps fallback) "
                "and opencv_frame_count"
            ) from exc
        if not math.isfinite(source_fps_exact) or source_fps_exact <= 0:
            raise ValueError(f"{npz_path} metadata_json.video.fps must be finite and positive")
        source_fps_tolerance = float(np.finfo(np.float32).eps) * max(abs(source_fps_exact), 1.0)
        if abs(source_fps - source_fps_exact) > source_fps_tolerance:
            raise ValueError(
                f"{npz_path} source_fps={source_fps!r} disagrees with original "
                f"metadata_json.video.fps={source_fps_exact!r} beyond float32 storage tolerance"
            )
        if metadata_source_count != source_count:
            raise ValueError(
                f"{npz_path} source_frame_count={source_count} disagrees with "
                f"metadata_json.video.opencv_frame_count={metadata_source_count}"
            )
        if source_count <= 0 or source_fps <= 0 or int(frame_indices[-1]) >= source_count:
            raise ValueError(f"{npz_path} has invalid source frame metadata")
        # Rebuild the sequence with Agent 4's original precision. source_fps is
        # intentionally stored as float32 in the NPZ; using that rounded value
        # can shift a small number of half-frame rounding boundaries.
        expected_indices = np.arange(0.0, source_count / source_fps_exact, 1.0 / target_fps, dtype=np.float64)
        expected_indices = np.unique(np.rint(expected_indices * source_fps_exact).astype(np.int64))
        expected_indices = expected_indices[(expected_indices >= 0) & (expected_indices < source_count)]
        if not np.array_equal(frame_indices, expected_indices.astype(np.int32)):
            raise ValueError(f"{npz_path} does not contain the complete uniform-{target_fps:g}fps frame sequence")
        expected_fields = {
            "video_id": first_value(raw_row, ("video_id",)) or sha_key(zip_member),
            "subject_id": subject_id,
            "class_label": class_label,
            "view": view,
            "visit": visit,
            "phase": first_value(raw_row, ("phase", "visit_hint", "visit")),
            "zip_member": zip_member,
        }
        for key, expected in expected_fields.items():
            observed_raw = str(data[key].item())
            observed = observed_raw.upper() if key == "class_label" else observed_raw.lower() if key == "view" else observed_raw
            wanted = expected.upper() if key == "class_label" else expected.lower() if key == "view" else expected
            if observed != wanted:
                raise ValueError(f"{npz_path} scalar {key}={observed_raw!r} disagrees with eligible manifest {expected!r}")
            metadata_observed = str(metadata.get(key, ""))
            metadata_observed = metadata_observed.upper() if key == "class_label" else metadata_observed.lower() if key == "view" else metadata_observed
            if metadata_observed != wanted:
                raise ValueError(f"{npz_path} metadata {key}={metadata_observed!r} disagrees with eligible manifest {expected!r}")
        if metadata.get("is_smoke") is not False or int(metadata.get("max_frames", -1)) != 0:
            raise ValueError(f"{npz_path} metadata does not certify a full, non-smoke input")
        if metadata.get("sampling_mode") != expected_sampling:
            raise ValueError(f"{npz_path} metadata sampling_mode must be {expected_sampling!r}")
        if not math.isclose(float(metadata.get("target_fps_requested") or 0.0), target_fps, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"{npz_path} metadata target_fps_requested disagrees with the scalar field")

    temporal_slots = cap_temporal_slots(frame_count, max_frames)
    capped_mask = valid_mask_array[np.asarray(temporal_slots, dtype=np.int64)]
    valid_indices = tuple(int(x) for x in np.flatnonzero(capped_mask))
    if not valid_indices:
        raise ValueError(f"{npz_path} has no valid pose-and-mask frames")
    if len(valid_indices) < train_frames:
        raise ValueError(
            f"{npz_path} has {len(valid_indices)} valid frames after the {max_frames}-frame cap; "
            f"the training sampler requires at least {train_frames}"
        )
    return VideoRecord(
        row_index=row_index,
        zip_member=zip_member,
        subject_id=subject_id,
        class_label=class_label,
        label_index=CLASS_TO_INDEX[class_label],
        view=view,
        visit=visit,
        npz_path=npz_path,
        frame_count=frame_count,
        temporal_slots=temporal_slots,
        valid_indices=valid_indices,
        valid_mask=tuple(bool(x) for x in capped_mask),
    )


def load_records(eligible_csv: Path, input_dir: Path, config: dict[str, Any]) -> list[VideoRecord]:
    if not eligible_csv.is_file():
        raise FileNotFoundError(
            f"Finalized eligibility manifest not found: {eligible_csv}. "
            "The runner requires the root-approved, preprocessing-QC-filtered eligible_videos.csv."
        )
    raw_rows = read_csv(eligible_csv)
    if not raw_rows:
        raise ValueError(f"Eligibility manifest has no rows: {eligible_csv}")
    temporal = config["temporal"]
    records = [
        read_npz_record(
            row, i, eligible_csv, input_dir,
            int(temporal["max_frames_per_video"]),
            int(temporal["train_frames_per_clip"]),
            float(config["preprocessing"].get("expected_target_fps", 15.0)),
        )
        for i, row in enumerate(raw_rows)
    ]
    seen_members: set[str] = set()
    seen_video_ids: set[str] = set()
    subject_labels: dict[str, str] = {}
    for record in records:
        if record.zip_member in seen_members:
            raise ValueError(f"Duplicate zip_member in eligible manifest: {record.zip_member}")
        seen_members.add(record.zip_member)
        video_id = first_value(raw_rows[record.row_index], ("video_id",))
        if video_id:
            if video_id in seen_video_ids:
                raise ValueError(f"Duplicate video_id in eligible manifest: {video_id}")
            seen_video_ids.add(video_id)
        prior = subject_labels.setdefault(record.subject_id, record.class_label)
        if prior != record.class_label:
            raise ValueError(f"Subject {record.subject_id} has inconsistent labels {prior} and {record.class_label}")
    return sorted(records, key=lambda x: (x.subject_id, x.visit, x.view, x.zip_member))


def records_signature(records: Sequence[VideoRecord]) -> str:
    payload = [
        [r.zip_member, r.subject_id, r.class_label, r.visit, r.view, str(r.npz_path.resolve())]
        for r in records
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def make_split_plan(records: Sequence[VideoRecord], config: dict[str, Any], signature: str) -> dict[str, Any]:
    subject_labels: dict[str, str] = {}
    for record in records:
        subject_labels[record.subject_id] = record.class_label
    by_class: dict[str, list[str]] = {name: [] for name in CLASS_TO_INDEX}
    for subject_id, label in subject_labels.items():
        by_class[label].append(subject_id)
    n_folds = int(config["evaluation"]["num_folds"])
    insufficient = {label: len(ids) for label, ids in by_class.items() if len(ids) < n_folds}
    if insufficient:
        raise ValueError(f"Every class needs at least {n_folds} eligible subjects for stratified CV: {insufficient}")

    seed = int(config["seed"])
    fold_test: list[list[str]] = [[] for _ in range(n_folds)]
    for class_index, label in enumerate(CLASS_TO_INDEX):
        ids = sorted(by_class[label])
        rng = np.random.default_rng(seed + 101 * (class_index + 1))
        rng.shuffle(ids)
        # Round-robin assignment keeps each class within one subject per fold.
        for position, subject_id in enumerate(ids):
            fold_test[position % n_folds].append(subject_id)

    outer_folds = []
    all_subjects = set(subject_labels)
    val_fraction = float(config["evaluation"]["validation_fraction"])
    for fold_index in range(n_folds):
        test_ids = sorted(fold_test[fold_index])
        trainval_ids = sorted(all_subjects - set(test_ids))
        val_ids: list[str] = []
        for class_index, label in enumerate(CLASS_TO_INDEX):
            class_trainval = [sid for sid in trainval_ids if subject_labels[sid] == label]
            rng = np.random.default_rng(seed + 100_003 * (fold_index + 1) + 997 * (class_index + 1))
            rng.shuffle(class_trainval)
            n_val = max(1, int(round(len(class_trainval) * val_fraction)))
            if n_val >= len(class_trainval):
                raise ValueError(f"Fold {fold_index + 1} has too few {label} training subjects for inner validation")
            val_ids.extend(class_trainval[:n_val])
        train_ids = sorted(set(trainval_ids) - set(val_ids))
        train_subject_counts = {
            label: sum(subject_labels[sid] == label for sid in train_ids)
            for label in CLASS_TO_INDEX
        }
        batch_minimum = int(config["sampling"]["subjects_per_class_per_batch"])
        if any(count < batch_minimum for count in train_subject_counts.values()):
            raise ValueError(
                f"Five-fold inner-train cohort is too small for balanced batches in fold {fold_index + 1}: "
                f"{train_subject_counts}; need at least {batch_minimum} training subjects per class "
                f"after reserving outer test and inner validation subjects."
            )
        if set(test_ids) & (set(train_ids) | set(val_ids)) or set(train_ids) & set(val_ids):
            raise AssertionError("Fold subject sets overlap")
        outer_folds.append({
            "fold": fold_index + 1,
            "train_subjects": train_ids,
            "validation_subjects": sorted(val_ids),
            "test_subjects": test_ids,
            "class_counts": {
                partition: {
                    label: sum(subject_labels[sid] == label for sid in ids)
                    for label in CLASS_TO_INDEX
                }
                for partition, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))
            },
        })
    test_coverage = [sid for fold in outer_folds for sid in fold["test_subjects"]]
    if len(test_coverage) != len(set(test_coverage)) or set(test_coverage) != all_subjects:
        raise AssertionError("Outer folds do not hold out every eligible subject exactly once")
    return {
        "schema": "live-gait-subject-folds-v1",
        "manifest_signature_sha256": signature,
        "seed": seed,
        "num_folds": n_folds,
        "subject_count": len(subject_labels),
        "class_subject_counts": {label: len(ids) for label, ids in by_class.items()},
        "validation_fraction_within_outer_train": val_fraction,
        "validation_split_policy": "per-class deterministic shuffle; max(1, round(20% of outer-train class subjects))",
        "folds": outer_folds,
    }


def load_or_create_split_plan(
    path: Path,
    records: Sequence[VideoRecord],
    config: dict[str, Any],
    signature: str,
    source_path: Path | None = None,
) -> dict[str, Any]:
    if source_path is not None:
        if not source_path.is_file():
            raise FileNotFoundError(f"Source split plan not found: {source_path}")
        source_plan = json.loads(source_path.read_text(encoding="utf-8"))
        if source_plan.get("manifest_signature_sha256") != signature:
            raise ValueError("Source split plan was made from a different eligible manifest")
        plan = dict(source_plan)
        plan.pop("shared_protocol", None)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("manifest_signature_sha256") != signature or existing.get("folds") != plan.get("folds"):
                raise ValueError("Local split plan differs from the requested source split plan")
            if "shared_protocol" in existing:
                plan = existing
        else:
            atomic_json(path, plan)
        return plan
    if path.is_file():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("manifest_signature_sha256") != signature:
            raise ValueError(
                f"{path} belongs to a different eligible manifest. Use a new --output path to freeze new folds."
            )
        return plan
    plan = make_split_plan(records, config, signature)
    atomic_json(path, plan)
    return plan


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_epoch_plan(
    records: Sequence[VideoRecord],
    train_subjects: Sequence[str],
    subject_labels: dict[str, str],
    records_by_subject: dict[str, list[int]],
    epoch: int,
    fold_seed: int,
    config: dict[str, Any],
) -> list[list[tuple[int, tuple[int, ...]]]]:
    rng = np.random.default_rng(fold_seed + 800_011 * (epoch + 1))
    per_batch = int(config["sampling"]["subjects_per_class_per_batch"])
    clips_per_subject = int(config["sampling"]["clips_per_subject_per_batch"])
    by_class: dict[str, list[str]] = {
        label: [sid for sid in train_subjects if subject_labels[sid] == label]
        for label in CLASS_TO_INDEX
    }
    if any(len(ids) < per_batch for ids in by_class.values()):
        raise ValueError(f"Training split cannot form balanced batches: {[len(by_class[k]) for k in CLASS_TO_INDEX]}")
    train_video_count = sum(len(records_by_subject[subject_id]) for subject_id in train_subjects)
    video_batch_size = int(config["sampling"]["videos_per_batch"])
    steps_for_subject_coverage = math.ceil(max(len(ids) for ids in by_class.values()) / per_batch)
    steps_for_video_coverage = math.ceil(train_video_count / video_batch_size)
    steps = max(steps_for_subject_coverage, steps_for_video_coverage)
    class_orders: dict[str, list[str]] = {}
    for label, ids in by_class.items():
        order = list(ids)
        rng.shuffle(order)
        class_orders[label] = order

    batches: list[list[tuple[int, tuple[int, ...]]]] = []
    for step in range(steps):
        batch_subjects: list[str] = []
        for label in CLASS_TO_INDEX:
            order = class_orders[label]
            selected = [order[(step * per_batch + i) % len(order)] for i in range(per_batch)]
            batch_subjects.extend(selected)
        samples: list[tuple[int, tuple[int, ...]]] = []
        for subject_id in batch_subjects:
            choices = records_by_subject[subject_id]
            if len(choices) >= clips_per_subject:
                chosen_records = rng.choice(choices, size=clips_per_subject, replace=False).tolist()
            else:
                chosen_records = rng.choice(choices, size=clips_per_subject, replace=True).tolist()
            for record_idx in chosen_records:
                valid = records[record_idx].valid_indices
                frame_indices = tuple(sorted(int(x) for x in rng.choice(
                    valid, size=int(config["temporal"]["train_frames_per_clip"]), replace=False
                )))
                samples.append((record_idx, frame_indices))
        order = rng.permutation(len(samples))
        batches.append([samples[int(i)] for i in order])
    return batches


def build_shared_protocol(
    records: Sequence[VideoRecord],
    split_plan: dict[str, Any],
    config: dict[str, Any],
    eligible_csv: Path,
    marker_path: Path,
    marker_sha256: str,
    view_protocol: dict[str, Any],
) -> dict[str, Any]:
    subject_labels = {record.subject_id: record.class_label for record in records}
    records_by_subject: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        records_by_subject.setdefault(record.subject_id, []).append(index)
    sampler_hashes: dict[str, Any] = {}
    for fold_spec in split_plan["folds"]:
        fold = int(fold_spec["fold"])
        fold_seed = int(config["seed"]) + fold * 10_007
        epoch_digests: list[str] = []
        steps_by_epoch: list[int] = []
        for epoch in range(int(config["training"]["epochs"])):
            plan = make_epoch_plan(
                records, fold_spec["train_subjects"], subject_labels,
                records_by_subject, epoch, fold_seed, config,
            )
            steps_by_epoch.append(len(plan))
            canonical = [
                [
                    [records[record_index].zip_member,
                     [records[record_index].temporal_slots[x] for x in frame_indices]]
                    for record_index, frame_indices in batch
                ]
                for batch in plan
            ]
            epoch_digests.append(hashlib.sha256(
                json.dumps(canonical, separators=(",", ":")).encode("utf-8")
            ).hexdigest())
        combined = hashlib.sha256("".join(epoch_digests).encode("ascii")).hexdigest()
        sampler_hashes[str(fold)] = {
            "steps_per_epoch": sorted(set(steps_by_epoch)),
            "epochs_planned": len(epoch_digests),
            "sampler_plan_sha256": combined,
            "epoch_plan_sha256": epoch_digests,
        }
    preprocessing = config["preprocessing"]
    protocol = {
        "schema": "live-gait-shared-training-protocol-v1",
        "seed": int(config["seed"]),
        "seed_policy": {
            "outer_split": "base seed plus 101 times one-based class index",
            "inner_validation": "base seed plus 100003 times one-based fold plus 997 times one-based class index",
            "model_initialization_and_dropout": "base seed plus 10007 times one-based fold; reset to the same fold seed per variant",
            "batch_plan": "fold seed plus 800011 times one-based epoch; independent NumPy generator",
        },
        "outer_folds": int(config["evaluation"]["num_folds"]),
        "inner_validation_fraction": float(config["evaluation"]["validation_fraction"]),
        "validation_selection_metric": config["evaluation"]["validation_selection_metric"],
        "validation_tie_breaker": config["evaluation"]["validation_tie_breaker"],
        "epoch_budget": int(config["training"]["epochs"]),
        "early_stopping": bool(config["training"].get("early_stopping", False)),
        "amp": bool(config["training"]["amp"]),
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "sampling": config["sampling"],
        "temporal": config["temporal"],
        "loss": config["loss"],
        "model_variants": list(config["model_variants"]),
        "class_to_index": config["class_to_index"],
        "view_protocol": view_protocol,
        "batch_plan_policy": "fixed deterministic plan for each fold/epoch, reused unchanged across all variants",
        "sampler_plan_hash_by_fold": sampler_hashes,
        "eligible_manifest_sha256": file_sha256(eligible_csv),
        "preprocessing_marker": str(marker_path.resolve()),
        "preprocessing_marker_sha256": marker_sha256,
        "preprocessing_max_frames_mode": int(preprocessing["full_video_mode_max_frames"]),
        "per_video_temporal_cap": int(preprocessing["max_frames_per_video"]),
    }
    return protocol


def load_selected_video(record: VideoRecord, selected_capped_indices: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slots = np.asarray(record.temporal_slots, dtype=np.int64)
    if selected_capped_indices is not None:
        slots = slots[np.asarray(selected_capped_indices, dtype=np.int64)]
    raw_arrays = RAW_INPUT_CACHE.get(record.npz_path)
    # Preserve the pre-cache order of operations: select the identical frames
    # in their original dtype, then convert only the selected batch to float32.
    skeleton = np.asarray(raw_arrays.skeleton_maps[slots], dtype=np.float32)
    silhouette_raw = np.asarray(raw_arrays.silhouette[slots])
    valid = np.asarray(raw_arrays.valid_frame_mask[slots], dtype=np.bool_)
    if float(silhouette_raw.max(initial=0)) > 1.0:
        silhouette = silhouette_raw.astype(np.float32) / 255.0
    else:
        silhouette = silhouette_raw.astype(np.float32)
    if not np.isfinite(skeleton).all() or not np.isfinite(silhouette).all():
        raise ValueError(f"Non-finite input values in {record.npz_path}")
    if skeleton.size and (skeleton.min() < -1e-4 or skeleton.max() > 1.001):
        raise ValueError(f"skeleton_maps must be scaled to [0,1] in {record.npz_path}")
    if silhouette.size and (silhouette.min() < -1e-4 or silhouette.max() > 1.001):
        raise ValueError(f"silhouette must be scaled to [0,1] in {record.npz_path}")
    return skeleton, silhouette, valid


def triplet_loss(embeddings: torch.Tensor, subject_ids: torch.Tensor, margin: float) -> torch.Tensor:
    """OpenGait TripletLoss with its all-positive/all-negative pair reducer."""
    values = embeddings.permute(2, 0, 1).contiguous().float()  # [P,N,C]
    n_parts, batch, _ = values.shape
    squared = (values * values).sum(dim=-1)
    distances = torch.sqrt(F.relu(
        squared.unsqueeze(2) + squared.unsqueeze(1)
        - 2.0 * torch.matmul(values, values.transpose(1, 2))
    ))
    same = subject_ids.reshape(-1, 1).eq(subject_ids.reshape(1, -1))
    different = ~same
    positives = distances[:, same].reshape(n_parts, batch, -1, 1)
    negatives = distances[:, different].reshape(n_parts, batch, 1, -1)
    losses = F.relu(positives - negatives + float(margin)).flatten(1)
    nonzero = losses.ne(0)
    mean_per_part = losses.sum(dim=1) / nonzero.sum(dim=1).clamp_min(1)
    return mean_per_part.mean()


def phase_regularization(assignments: torch.Tensor, valid_mask: torch.Tensor, settings: dict[str, Any]) -> torch.Tensor:
    """Match OpenGait PhaseRegularizationLoss for the CSPT/Uniform variants."""
    probabilities = assignments.float().clamp_min(0.0)
    mask = valid_mask.to(device=probabilities.device, dtype=torch.bool)
    batch, frames, phases = probabilities.shape
    valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
    occupancy = probabilities.sum(dim=1) / valid_count.to(probabilities.dtype)
    occupancy = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    balance = (occupancy * (occupancy * phases).clamp_min(1e-8).log()).sum(dim=-1).mean()
    entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1) / math.log(phases)
    sharpness = (entropy * mask.to(entropy.dtype)).sum() / mask.sum().clamp_min(1)
    if frames > 1:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        smooth_pairs = (probabilities[:, 1:] - probabilities[:, :-1]).square().sum(dim=-1)
        smooth = (smooth_pairs * pair_mask.to(smooth_pairs.dtype)).sum() / pair_mask.sum().clamp_min(1)
        angles = torch.arange(phases, dtype=probabilities.dtype, device=probabilities.device) * (2.0 * math.pi / phases)
        phase_x = probabilities @ torch.cos(angles)
        phase_y = probabilities @ torch.sin(angles)
        vectors = F.normalize(torch.stack((phase_x, phase_y), dim=-1), dim=-1, eps=1e-6)
        cross = vectors[:, :-1, 0] * vectors[:, 1:, 1] - vectors[:, :-1, 1] * vectors[:, 1:, 0]
        direction = torch.where(
            (cross * pair_mask.to(cross.dtype)).sum(dim=1) >= 0,
            torch.ones(batch, device=cross.device, dtype=cross.dtype),
            -torch.ones(batch, device=cross.device, dtype=cross.dtype),
        ).detach()
        cycle_values = F.relu(-direction[:, None] * cross)
        cycle = (cycle_values * pair_mask.to(cycle_values.dtype)).sum() / pair_mask.sum().clamp_min(1)
    else:
        smooth = probabilities.new_zeros(())
        cycle = probabilities.new_zeros(())
    return (
        float(settings["smooth_weight"]) * smooth
        + float(settings["balance_weight"]) * balance
        + float(settings["sharpness_weight"]) * sharpness
        + float(settings["cycle_weight"]) * cycle
    )


def train_batch(
    model: torch.nn.Module,
    batch: Sequence[tuple[int, tuple[int, ...]]],
    records: Sequence[VideoRecord],
    subject_index: dict[str, int],
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    maps, silhouettes, valid_masks, classes, subjects = [], [], [], [], []
    for record_index, frame_indices in batch:
        record = records[record_index]
        skeleton, silhouette, _ = load_selected_video(record, frame_indices)
        maps.append(skeleton)
        silhouettes.append(silhouette)
        valid_masks.append(np.ones(len(frame_indices), dtype=np.bool_))
        classes.append(record.label_index)
        subjects.append(subject_index[record.subject_id])
    map_tensor = torch.from_numpy(np.stack(maps)).to(device=device, non_blocking=True)
    sil_tensor = torch.from_numpy(np.stack(silhouettes)).to(device=device, non_blocking=True)
    mask_tensor = torch.from_numpy(np.stack(valid_masks)).to(device=device, non_blocking=True)
    class_tensor = torch.as_tensor(classes, dtype=torch.long, device=device)
    subject_tensor = torch.as_tensor(subjects, dtype=torch.long, device=device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
    with amp_ctx:
        output = model(
            map_tensor,
            sil_tensor,
            mask_tensor,
            subject_ids=subject_tensor,
            class_labels=class_tensor,
        )
    train_feat = output["training_feat"]
    logits = train_feat["softmax"]["logits"].float()
    targets = train_feat["softmax"]["labels"].reshape(-1, 1).expand(-1, logits.size(-1))
    loss_cfg = config["loss"]
    ce = F.cross_entropy(
        logits * float(loss_cfg["ce_scale"]),
        targets,
        label_smoothing=float(loss_cfg["label_smoothing"]),
    )
    triplet = triplet_loss(
        train_feat["triplet"]["embeddings"],
        train_feat["triplet"]["labels"],
        float(loss_cfg["triplet_margin"]),
    )
    phase = logits.new_zeros(())
    if "phase_regularization" in train_feat:
        phase_inputs = train_feat["phase_regularization"]
        phase = phase_regularization(
            phase_inputs["assignments"], phase_inputs["valid_mask"], loss_cfg["phase_regularization"]
        )
    total = (
        float(loss_cfg.get("triplet_weight", 1.0)) * triplet
        + float(loss_cfg.get("ce_weight", 1.0)) * ce
        + float(loss_cfg["phase_weight"]) * phase
    )
    return total, {
        "total": float(total.detach().cpu()),
        "triplet": float(triplet.detach().cpu()),
        "cross_entropy": float(ce.detach().cpu()),
        "phase_regularization": float(phase.detach().cpu()),
    }


def forward_full_sequence(
    model: torch.nn.Module,
    skeleton_map: torch.Tensor,
    silhouette: torch.Tensor,
    valid_mask: torch.Tensor,
    frame_chunk: int,
) -> dict[str, Any]:
    """Encode frame chunks then tokenize the one complete, masked sequence."""
    map_features: list[torch.Tensor] = []
    sil_features: list[torch.Tensor] = []
    for start in range(0, skeleton_map.size(1), frame_chunk):
        stop = min(start + frame_chunk, skeleton_map.size(1))
        current_mask = valid_mask[:, start:stop]
        map_features.append(model._encode_frames(model.map_backbone, skeleton_map[:, start:stop], current_mask))
        sil_features.append(model._encode_frames(model.sil_backbone, silhouette[:, start:stop], current_mask))
    descriptors = torch.stack((torch.cat(map_features, dim=1), torch.cat(sil_features, dim=1)), dim=2)
    phase = None
    if model.variant == "global":
        denominator = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(descriptors.dtype)
        temporal_mean = descriptors.sum(dim=1) / denominator.unsqueeze(-1)
        parts = temporal_mean.mean(dim=1, keepdim=True).transpose(1, 2).contiguous()
    else:
        phase = model.phase_tokenizer(descriptors, valid_mask)
        tokens = phase["tokens"].mean(dim=1)
        parts = model.phase_context(tokens).transpose(1, 2).contiguous()
    projected = model.projection(parts)
    embeddings, logits = model.classifier(projected)
    return {
        "embeddings": embeddings,
        "logits": logits,
        "prediction_logits": logits.mean(dim=-1),
        "phase": phase,
    }


def infer_record(
    model: torch.nn.Module,
    record: VideoRecord,
    fold: int,
    variant: str,
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    skeleton, silhouette, valid = load_selected_video(record)
    if not valid.any():
        raise ValueError(f"No valid frames after temporal cap for {record.zip_member}")
    maps = torch.from_numpy(skeleton[None]).to(device=device)
    silhouettes = torch.from_numpy(silhouette[None]).to(device=device)
    mask = torch.from_numpy(valid[None]).to(device=device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
    with torch.inference_mode(), amp_ctx:
        output = forward_full_sequence(
            model, maps, silhouettes, mask, int(config["temporal"]["inference_frame_chunk"])
        )
    # Training CE applies this cosine-logit scale. Keep validation CE,
    # probabilities and exported logits on that same classification scale.
    logits = (
        output["prediction_logits"][0].float() * float(config["loss"]["ce_scale"])
    ).cpu().numpy()
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    phase_entropy = None
    occupancy: list[float] | None = None
    phase_output = output["phase"]
    if phase_output is not None and phase_output.get("assignments") is not None:
        assignments = phase_output["assignments"][0].float()
        phase_mask = phase_output.get("valid_mask", mask)[0].bool()
        probabilities = assignments[phase_mask].clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1) / math.log(probabilities.size(-1))
        phase_entropy = float(entropy.mean().cpu())
        occupancy = [float(x) for x in probabilities.mean(dim=0).cpu().tolist()]
    return {
        "variant": variant,
        "fold": fold,
        "zip_member": record.zip_member,
        "subject_id": record.subject_id,
        "class_label": record.class_label,
        "label_index": record.label_index,
        "visit": record.visit,
        "view": record.view,
        "n_valid_frames": int(valid.sum()),
        "frame_count_after_cap": len(valid),
        "logits": logits.tolist(),
        "probabilities": probs.tolist(),
        "phase_entropy_normalized": phase_entropy,
        "phase_occupancy": occupancy,
    }


def evaluate_records(
    model: torch.nn.Module,
    records: Sequence[VideoRecord],
    fold: int,
    variant: str,
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model.eval()  # BatchNorm remains fixed throughout both validation and test inference.
    return [infer_record(model, record, fold, variant, device, amp_enabled, config) for record in records]


def metric_and_loss(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    subject_rows = aggregate_subject_rows(rows)
    metric = metrics_for(subject_rows)
    losses = []
    for row in subject_rows:
        p = np.asarray(row["probabilities"], dtype=np.float64)
        losses.append(-math.log(max(float(p[int(row["label_index"])]), 1e-12)))
    return metric, float(np.mean(losses))


def model_output_dir(output_dir: Path, fold: int, variant: str) -> Path:
    return output_dir / f"fold_{fold:02d}" / variant


def save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def read_history(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.is_file() else []


def update_progress(path: Path, progress: dict[str, Any]) -> None:
    if isinstance(progress.get("input_cache"), dict):
        progress["input_cache"]["runtime"] = RAW_INPUT_CACHE.snapshot()
    atomic_json(path, progress)


def configure_cuda_runtime(device: torch.device) -> dict[str, Any]:
    """Apply optional process-local allocator controls without changing the protocol."""
    raw_fraction = os.environ.get("LIVE_GAIT_CUDA_MEMORY_FRACTION")
    allocator_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    allocator_conf_env = "PYTORCH_ALLOC_CONF"
    if not allocator_conf:
        allocator_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        allocator_conf_env = "PYTORCH_CUDA_ALLOC_CONF" if allocator_conf else None
    fraction: float | None = None
    if raw_fraction not in (None, ""):
        try:
            fraction = float(raw_fraction)
        except ValueError as exc:
            raise ValueError("LIVE_GAIT_CUDA_MEMORY_FRACTION must be between 0 and 1") from exc
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("LIVE_GAIT_CUDA_MEMORY_FRACTION must be between 0 and 1")
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(fraction, device=device)

    allocator_backend = None
    if device.type == "cuda" and torch.cuda.is_available():
        allocator_backend = torch.cuda.get_allocator_backend()
    return {
        "memory_fraction_env": raw_fraction,
        "memory_fraction": fraction if device.type == "cuda" else None,
        "allocator_conf": allocator_conf,
        "allocator_conf_env": allocator_conf_env if allocator_conf else None,
        "allocator_backend": allocator_backend,
    }


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated_current_bytes": None,
            "reserved_current_bytes": None,
            "allocated_peak_bytes": None,
            "reserved_peak_bytes": None,
        }
    return {
        "allocated_current_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_current_bytes": int(torch.cuda.memory_reserved(device)),
        "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def max_optional_int(*values: int | None) -> int | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


def append_runtime_telemetry(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def history_through_epoch(history: Sequence[dict[str, Any]], checkpoint_epoch: int) -> list[dict[str, Any]]:
    """Discard rows written after the latest durable checkpoint during interruption."""
    return [row for row in history if int(row.get("epoch", 0)) <= checkpoint_epoch]


def train_one_fold_variant(
    records: Sequence[VideoRecord],
    all_test_records: Sequence[VideoRecord],
    fold_spec: dict[str, Any],
    fold: int,
    variant: str,
    fit_views: list[str] | None,
    test_view: str | None,
    output_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    subject_index: dict[str, int],
    subject_labels: dict[str, str],
    records_by_subject: dict[str, list[int]],
    amp_enabled: bool,
    resume: bool,
    progress: dict[str, Any],
    progress_path: Path,
    runtime_telemetry_path: Path,
    runtime_settings: dict[str, Any],
) -> None:
    run_dir = model_output_dir(output_dir, fold, variant)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    oof_video_path = output_dir / "predictions" / f"fold_{fold:02d}_{variant}_video_oof.csv"
    oof_subject_path = output_dir / "predictions" / f"fold_{fold:02d}_{variant}_subject_oof.csv"
    if oof_video_path.exists() and oof_subject_path.exists():
        if resume:
            key = f"fold_{fold:02d}/{variant}"
            existing = dict(progress["combinations"].get(key, {}))
            existing.update({"status": "complete", "reused_existing_oof": True})
            progress["combinations"][key] = existing
            return
        raise FileExistsError(f"Existing outputs found for fold {fold} {variant}; pass --resume to continue")
    if latest_path.exists() and not resume:
        raise FileExistsError(f"Checkpoint exists at {latest_path}; pass --resume to continue or choose a new --output")

    fold_seed = int(config["seed"]) + fold * 10_007
    seed_everything(fold_seed)
    model = build_temporal_model(variant).to(device)
    optimizer_cfg = config["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        momentum=float(optimizer_cfg["momentum"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    scheduler_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(scheduler_cfg["milestones"]), gamma=float(scheduler_cfg["gamma"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    best_macro_f1 = -math.inf
    best_val_loss = math.inf
    best_epoch = 0
    history = read_history(history_path) if resume else []
    if latest_path.is_file() and resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_macro_f1 = float(checkpoint["best_val_macro_f1"])
        best_val_loss = float(checkpoint["best_val_subject_cross_entropy"])
        best_epoch = int(checkpoint["best_epoch"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        del checkpoint
    elif start_epoch == 0 and history_path.exists() and not resume:
        raise FileExistsError(f"History exists at {history_path}; pass --resume or choose a new output directory")
    history = history_through_epoch(history, start_epoch)

    train_ids = fold_spec["train_subjects"]
    validation_ids = set(fold_spec["validation_subjects"])
    validation_records = [r for r in records if r.subject_id in validation_ids]
    epochs = int(config["training"]["epochs"])
    history_fields = [
        "epoch", "learning_rate", "train_total", "train_triplet", "train_cross_entropy",
        "train_phase_regularization", "validation_subject_macro_f1", "validation_subject_accuracy",
        "validation_subject_balanced_accuracy", "validation_subject_cross_entropy", "is_best",
    ]
    if start_epoch == 0:
        history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        synchronize_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        plans = make_epoch_plan(records, train_ids, subject_labels, records_by_subject, epoch, fold_seed, config)
        epoch_sums = {"total": 0.0, "triplet": 0.0, "cross_entropy": 0.0, "phase_regularization": 0.0}
        first_train_batch_telemetry: dict[str, Any] | None = None
        for batch_index, batch in enumerate(plans):
            optimizer.zero_grad(set_to_none=True)
            if epoch == start_epoch and batch_index == 0:
                synchronize_device(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                forward_started = time.perf_counter()
                loss, breakdown = train_batch(model, batch, records, subject_index, device, amp_enabled, config)
                synchronize_device(device)
                forward_seconds = time.perf_counter() - forward_started
                forward_memory = cuda_memory_snapshot(device)

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                backward_started = time.perf_counter()
                scaler.scale(loss).backward()
                synchronize_device(device)
                backward_seconds = time.perf_counter() - backward_started
                backward_memory = cuda_memory_snapshot(device)
                first_train_batch_telemetry = {
                    "forward_and_loss": {"seconds": forward_seconds, **forward_memory},
                    "backward": {"seconds": backward_seconds, **backward_memory},
                }
            else:
                loss, breakdown = train_batch(model, batch, records, subject_index, device, amp_enabled, config)
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for key in epoch_sums:
                epoch_sums[key] += breakdown[key]
            del loss, breakdown
        scheduler.step()
        batch_count = max(len(plans), 1)
        epoch_sums = {key: value / batch_count for key, value in epoch_sums.items()}
        del plans

        synchronize_device(device)
        train_seconds = time.perf_counter() - train_started
        train_memory_before_release = cuda_memory_snapshot(device)
        train_peak_allocated = train_memory_before_release["allocated_peak_bytes"]
        train_peak_reserved = train_memory_before_release["reserved_peak_bytes"]
        if first_train_batch_telemetry is not None:
            forward_memory = first_train_batch_telemetry["forward_and_loss"]
            backward_memory = first_train_batch_telemetry["backward"]
            train_peak_allocated = max_optional_int(
                train_peak_allocated,
                forward_memory["allocated_peak_bytes"],
                backward_memory["allocated_peak_bytes"],
            )
            train_peak_reserved = max_optional_int(
                train_peak_reserved,
                forward_memory["reserved_peak_bytes"],
                backward_memory["reserved_peak_bytes"],
            )

        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_memory = cuda_memory_snapshot(device)
        train_memory["allocated_peak_bytes"] = train_peak_allocated
        train_memory["reserved_peak_bytes"] = train_peak_reserved

        synchronize_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation_started = time.perf_counter()
        validation_rows = evaluate_records(
            model, validation_records, fold, variant, device, amp_enabled, config
        )
        val_metrics, val_ce = metric_and_loss(validation_rows)
        del validation_rows
        synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        validation_memory = cuda_memory_snapshot(device)
        is_best = (
            val_metrics["macro_f1"] > best_macro_f1 + 1e-12
            or (abs(val_metrics["macro_f1"] - best_macro_f1) <= 1e-12 and val_ce < best_val_loss)
        )
        if is_best:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_val_loss = val_ce
            best_epoch = epoch + 1
            best_payload = {
                "schema": "live-gait-best-checkpoint-v1",
                "variant": variant,
                "fold": fold,
                "best_epoch": best_epoch,
                "validation_subject_macro_f1": best_macro_f1,
                "validation_subject_cross_entropy": best_val_loss,
                "model_state_dict": model.state_dict(),
                "config": config,
            }
            save_torch_atomic(best_path, best_payload)
            del best_payload
        row = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate_used,
            "train_total": epoch_sums["total"],
            "train_triplet": epoch_sums["triplet"],
            "train_cross_entropy": epoch_sums["cross_entropy"],
            "train_phase_regularization": epoch_sums["phase_regularization"],
            "validation_subject_macro_f1": val_metrics["macro_f1"],
            "validation_subject_accuracy": val_metrics["accuracy"],
            "validation_subject_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_subject_cross_entropy": val_ce,
            "is_best": int(is_best),
        }
        history.append(row)
        atomic_csv(history_path, history, history_fields)
        checkpoint_payload = {
            "schema": "live-gait-latest-checkpoint-v1",
            "variant": variant,
            "fold": fold,
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "best_val_subject_cross_entropy": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "python_rng_state": random.getstate(),
            "config": config,
        }
        save_torch_atomic(latest_path, checkpoint_payload)
        del checkpoint_payload
        key = f"fold_{fold:02d}/{variant}"
        progress["combinations"][key] = {
            "status": "training",
            "epoch": epoch + 1,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_validation_subject_macro_f1": best_macro_f1,
            "last_validation_subject_macro_f1": val_metrics["macro_f1"],
            "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        update_progress(progress_path, progress)
        append_runtime_telemetry(runtime_telemetry_path, {
            "schema": "live-gait-runtime-telemetry-v1",
            "event": "epoch",
            "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fold": fold,
            "variant": variant,
            "epoch": epoch + 1,
            "timing_seconds": {
                "train": train_seconds,
                "validation": validation_seconds,
            },
            "cuda_memory": {
                "train": train_memory,
                "validation": validation_memory,
                "first_train_batch": first_train_batch_telemetry,
            },
            "runtime_settings": runtime_settings,
        })
        if device.type == "cuda":
            synchronize_device(device)
            torch.cuda.empty_cache()
        print(
            f"fold={fold} variant={variant} epoch={epoch + 1}/{epochs} "
            f"train_loss={epoch_sums['total']:.5f} val_subject_macro_f1={val_metrics['macro_f1']:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    if not best_path.is_file():
        raise RuntimeError(f"No best validation checkpoint produced for fold {fold}, {variant}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch_for_report = int(best["best_epoch"])
    best_macro_f1_for_report = float(best["validation_subject_macro_f1"])
    model.load_state_dict(best["model_state_dict"])
    del best
    gc.collect()
    model.eval()
    test_subjects = set(fold_spec["test_subjects"])
    test_records = [
        r for r in all_test_records
        if r.subject_id in test_subjects and (test_view is None or r.view == test_view)
    ]
    absent_test_subjects = test_subjects - {r.subject_id for r in test_records}
    if absent_test_subjects:
        raise ValueError(
            f"Outer test subjects lack the requested view {test_view!r}: {sorted(absent_test_subjects)}"
        )
    # This is the first use of this fold's test subjects; their records are not
    # consulted for training, validation, epoch selection, or parameter choice.
    oof_video_rows = evaluate_records(model, test_records, fold, variant, device, amp_enabled, config)
    fit_view_label = "all" if fit_views is None else "|".join(fit_views)
    test_view_label = test_view or "all"
    for row in oof_video_rows:
        row["fit_views"] = fit_view_label
        row["test_view_filter"] = test_view_label
    oof_subject_rows = aggregate_subject_rows(oof_video_rows)
    for row in oof_subject_rows:
        row["fit_views"] = fit_view_label
        row["test_view_filter"] = test_view_label
    atomic_csv(oof_video_path, oof_video_rows)
    atomic_csv(oof_subject_path, oof_subject_rows)
    progress["combinations"][f"fold_{fold:02d}/{variant}"] = {
        "status": "complete",
        "epochs_completed": epochs,
        "best_epoch": best_epoch_for_report,
        "best_validation_subject_macro_f1": best_macro_f1_for_report,
        "video_oof_rows": len(oof_video_rows),
        "subject_oof_rows": len(oof_subject_rows),
        "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    update_progress(progress_path, progress)


def collect_completed_oof(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_dir = output_dir / "predictions"
    video_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for path in sorted(pred_dir.glob("fold_*_*_video_oof.csv")):
        video_rows.extend(read_csv(path))
    for path in sorted(pred_dir.glob("fold_*_*_subject_oof.csv")):
        subject_rows.extend(read_csv(path))
    return video_rows, subject_rows


def generate_summaries(output_dir: Path, config: dict[str, Any], expected_folds: int) -> None:
    video_rows, subject_rows = collect_completed_oof(output_dir)
    if video_rows:
        atomic_csv(output_dir / "predictions" / "video_oof.csv", video_rows)
    if subject_rows:
        atomic_csv(output_dir / "predictions" / "subject_oof.csv", subject_rows)
    write_summary(output_dir, video_rows, subject_rows, config, expected_folds)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "live-gait-cv-config-v1":
        raise ValueError(f"Unsupported experiment config schema in {path}")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eligible-csv", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 for the planned run; cpu is available for small checks")
    parser.add_argument("--fold", type=int, help="Run one 1-based outer fold")
    parser.add_argument("--variant", nargs="+", choices=VARIANTS, help="Run one or more variants")
    parser.add_argument("--split-plan", type=Path, help="Reuse a previously frozen subject split plan")
    parser.add_argument("--fit-views", nargs="+", choices=("left", "right", "sagittal"), help="Restrict training/inner validation to these views")
    parser.add_argument("--test-view", choices=("left", "right", "sagittal"), help="Evaluate only this held-out view; requires --fit-views")
    parser.add_argument("--resume", action="store_true", help="Continue from latest.pt and reuse completed OOF combinations")
    parser.add_argument("--check-only", action="store_true", help="Validate manifest/NPZs without training or freezing folds")
    return parser.parse_args(argv)


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    experiment_root = args.config.resolve().parent
    eligible_csv = args.eligible_csv or resolve_config_path(config["eligible_csv"], args.config.resolve())
    input_dir = args.input_dir or resolve_config_path(config["input_dir"], args.config.resolve())
    output_dir = args.output or resolve_config_path(config["output_dir"], args.config.resolve())
    if not output_dir.is_absolute():
        output_dir = experiment_root / output_dir
    device = torch.device(args.device)
    if not args.check_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for training but is unavailable; use --check-only for a CPU input audit")
    if not args.check_only and device.type not in {"cuda", "cpu"}:
        raise ValueError("--device must be cuda:0 or cpu")
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    records = load_records(eligible_csv, input_dir, config)
    signature = records_signature(records)
    if args.check_only:
        marker_path = resolve_config_path(config["preprocessing"]["completion_marker"], args.config.resolve())
        marker_valid = False
        if marker_path.is_file():
            try:
                require_full_preprocessing(eligible_csv, args.config, config)
                marker_valid = True
            except (ValueError, KeyError, TypeError):
                marker_valid = False
        subjects = {record.subject_id: record.class_label for record in records}
        print(json.dumps({
            "manifest": str(eligible_csv),
            "input_dir": str(input_dir),
            "video_count": len(records),
            "subject_counts": {label: sum(value == label for value in subjects.values()) for label in CLASS_TO_INDEX},
            "full_preprocessing_marker_valid": marker_valid,
            "split_plan_frozen": False,
            "gpu_training_started": False,
        }, ensure_ascii=False, indent=2))
        return 0

    runtime_settings = configure_cuda_runtime(device)
    marker, marker_hash = require_full_preprocessing(eligible_csv, args.config, config)
    marker_path = resolve_config_path(config["preprocessing"]["completion_marker"], args.config.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_telemetry_path = output_dir / "runtime_telemetry.jsonl"
    append_runtime_telemetry(runtime_telemetry_path, {
        "schema": "live-gait-runtime-telemetry-v1",
        "event": "run_start",
        "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "runtime_settings": runtime_settings,
    })
    split_path = output_dir / "split_plan.json"
    split_plan = load_or_create_split_plan(split_path, records, config, signature, args.split_plan)
    if args.test_view and not args.fit_views:
        raise ValueError("--test-view requires an explicit --fit-views list")
    fit_views = sorted(set(args.fit_views)) if args.fit_views else None
    test_view = args.test_view
    if fit_views and test_view in fit_views:
        raise ValueError("The held-out test view cannot also be used for fitting")
    fit_records = [r for r in records if fit_views is None or r.view in fit_views]
    if not fit_records:
        raise ValueError("No eligible videos remain after applying --fit-views")
    subject_labels: dict[str, str] = {r.subject_id: r.class_label for r in records}
    subject_index = {sid: i for i, sid in enumerate(sorted(subject_labels))}
    records_by_subject: dict[str, list[int]] = {}
    for index, record in enumerate(fit_records):
        records_by_subject.setdefault(record.subject_id, []).append(index)
    required_fit_subjects = {
        sid for fold in split_plan["folds"]
        for sid in (fold["train_subjects"] + fold["validation_subjects"])
    }
    absent_fit_subjects = required_fit_subjects - set(records_by_subject)
    if absent_fit_subjects:
        raise ValueError(f"Subjects lack an eligible fit-view video: {sorted(absent_fit_subjects)}")
    if fit_views:
        views_by_subject: dict[str, set[str]] = {}
        for record in fit_records:
            views_by_subject.setdefault(record.subject_id, set()).add(record.view)
        incomplete_fit_views = {
            subject_id: sorted(set(fit_views) - views_by_subject.get(subject_id, set()))
            for subject_id in required_fit_subjects
            if set(fit_views) - views_by_subject.get(subject_id, set())
        }
        if incomplete_fit_views:
            raise ValueError(
                "Strict cross-view fitting requires every final subject to have each requested fit view; "
                f"missing views: {incomplete_fit_views}"
            )
    if test_view is not None:
        all_subjects = set(subject_labels)
        target_subjects = {record.subject_id for record in records if record.view == test_view}
        absent_test_view_subjects = all_subjects - target_subjects
        if absent_test_view_subjects:
            raise ValueError(
                f"Final subject cohort lacks the requested held-out view {test_view!r}: "
                f"{sorted(absent_test_view_subjects)}"
            )
    view_protocol = {
        "fit_views": fit_views or "all",
        "test_view": test_view or "all",
        "test_view_is_disjoint_from_fit": test_view not in (fit_views or []),
        "subject_folds_reused_from": str(args.split_plan.resolve()) if args.split_plan else "deterministically frozen from the eligible subject table",
    }
    protocol = build_shared_protocol(
        fit_records, split_plan, config, eligible_csv, marker_path, marker_hash, view_protocol
    )
    if "shared_protocol" in split_plan and split_plan["shared_protocol"] != protocol:
        raise ValueError("Frozen split protocol does not match this run config, preprocessing marker, or sampler plan")
    split_plan["shared_protocol"] = protocol
    atomic_json(split_path, split_plan)
    training_config_path = output_dir / "training_config.json"
    if training_config_path.is_file():
        previous_protocol = json.loads(training_config_path.read_text(encoding="utf-8"))
        if previous_protocol != protocol:
            raise ValueError("training_config.json differs from the immutable protocol; choose a new --output")
    else:
        atomic_json(training_config_path, protocol)

    progress_path = output_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("manifest_signature_sha256") != signature:
            raise ValueError("Existing progress.json belongs to a different eligible manifest")
        if not args.resume and progress.get("combinations"):
            raise FileExistsError(f"Run output already exists at {output_dir}; pass --resume or select a new --output")
    else:
        progress = {
            "schema": RUNNER_SCHEMA,
            "manifest": str(eligible_csv.resolve()),
            "manifest_signature_sha256": signature,
            "config": config,
            "model_variants": list(VARIANTS),
            "view_protocol": view_protocol,
            "fold_count": int(config["evaluation"]["num_folds"]),
            "combinations": {},
            "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        update_progress(progress_path, progress)

    estimated_cache_bytes = estimate_raw_cache_bytes(records)
    progress["input_cache"] = {
        "policy": "process-local LRU; validated inputs are immutable for the run",
        "arrays": {
            "skeleton_maps": "float16",
            "silhouette": "uint8",
            "valid_frame_mask": "bool",
        },
        "identity_key": "resolved path + file size + mtime_ns",
        "max_items": RAW_INPUT_CACHE_MAX_ITEMS,
        "estimated_full_cohort_bytes": estimated_cache_bytes,
        "estimated_full_cohort_gib": round(estimated_cache_bytes / (1024 ** 3), 3),
        "runtime": RAW_INPUT_CACHE.snapshot(),
    }
    update_progress(progress_path, progress)

    folds = split_plan["folds"]
    if args.fold is not None:
        if args.fold < 1 or args.fold > len(folds):
            raise ValueError(f"--fold must be between 1 and {len(folds)}")
        folds = [folds[args.fold - 1]]
    variants = args.variant or list(VARIANTS)
    for fold_spec in folds:
        fold = int(fold_spec["fold"])
        for variant in variants:
            train_one_fold_variant(
                fit_records, records, fold_spec, fold, variant, fit_views, test_view, output_dir, config, device,
                subject_index, subject_labels, records_by_subject, amp_enabled,
                args.resume, progress, progress_path, runtime_telemetry_path, runtime_settings,
            )
            gc.collect()
            if device.type == "cuda":
                synchronize_device(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            append_runtime_telemetry(runtime_telemetry_path, {
                "schema": "live-gait-runtime-telemetry-v1",
                "event": "variant_boundary",
                "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fold": fold,
                "variant": variant,
                "cuda_memory": cuda_memory_snapshot(device),
                "runtime_settings": runtime_settings,
            })
    progress["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    generate_summaries(output_dir, config, int(config["evaluation"]["num_folds"]))
    all_expected = {
        f"fold_{fold['fold']:02d}/{variant}"
        for fold in split_plan["folds"] for variant in VARIANTS
    }
    complete = all(progress["combinations"].get(key, {}).get("status") == "complete" for key in all_expected)
    progress["status"] = "complete" if complete else "partial"
    update_progress(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
