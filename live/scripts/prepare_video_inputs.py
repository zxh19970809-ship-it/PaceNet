#!/usr/bin/env python
"""Extract aligned pose-map and person-silhouette inputs from selected LIVE MOV clips.

This script deliberately has no import-time RNG/model side effects. It imports the
local Ultralytics checkpoints only when processing a clip and extracts exact ZIP
members into a separate derivative cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "manifests" / "videos.csv"
DEFAULT_INPUT_ZIP = Path("data/LIVE-GaitNeuroKids/VIDEO.zip")
DEFAULT_CACHE_ROOT = EXPERIMENT_ROOT / "cache"
DEFAULT_POSE_WEIGHTS = Path("weights/yolov8x-pose.pt")
DEFAULT_SEG_WEIGHTS = Path("weights/yolo11x-seg.pt")
DEFAULT_FFPROBE = Path(shutil.which("ffprobe") or "ffprobe")

COCO17_LIMBS = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
    (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
    (6, 12), (12, 14), (14, 16), (11, 12),
)
MAP_SIZE = 64
POSE_CONF_MIN = 0.15
MIN_VISIBLE_JOINTS = 8
TRACK_MAX_GAP = 4
GAUSSIAN_SIGMA = 1.0
MIN_RELIABLE_VALID_FRAMES = 30
MIN_RELIABLE_MEAN_POSE_CONFIDENCE = 0.25
SIGNIFICANT_TRACK_OVERLAP_MIN_FRAMES = 3
SIGNIFICANT_TRACK_OVERLAP_MIN_RATIO = 0.20
SELECTION_RULE_VERSION = "live-gait-track-qc-v2"


@dataclass
class Detection:
    frame_slot: int
    frame_index: int
    det_index: int
    box: np.ndarray
    keypoints: np.ndarray
    score: float
    track_id: int = -1


@dataclass
class Track:
    track_id: int
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0

    @property
    def last(self) -> Detection:
        return self.detections[-1]


def sha_key(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_allowed_member(member: str) -> tuple[bool, str]:
    normalized = member.replace("\\", "/")
    name = PurePosixPath(normalized).name
    upper = normalized.upper()
    if not normalized or normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
        return False, "unsafe_archive_path"
    if "__MACOSX" in upper or name.startswith("._"):
        return False, "macos_sidecar"
    if not name.lower().endswith(".mov"):
        return False, "not_mov"
    if any(tag in upper for tag in ("MARK", "CALIB", "CALIBRATION")):
        return False, "marking_or_calibration"
    return True, ""


def select_rows(manifest: Path, members: list[str] | None, selection_csv: Path | None) -> list[dict[str, str]]:
    rows = load_csv(manifest)
    if members and selection_csv:
        raise ValueError("Pass either --members or --selection-csv, not both.")
    if selection_csv:
        selected = load_csv(selection_csv)
        wanted = {r.get("zip_member", ""): r for r in selected}
        missing = sorted(set(wanted) - {r.get("zip_member", "") for r in rows})
        if missing:
            raise ValueError(f"Selection contains members missing from manifest: {missing[:3]}")
        by_member = {r["zip_member"]: r for r in rows}
        out = []
        for member, extra in wanted.items():
            row = dict(by_member[member])
            for k in ("video_id", "npz_path", "subject_id", "class_label", "view_hint", "visit_hint", "phase", "target_track_id", "selection_note"):
                if extra.get(k):
                    row[k] = extra[k]
            if extra.get("view") and not extra.get("view_hint"):
                row["view_hint"] = extra["view"]
            if extra.get("visit") and not extra.get("visit_hint"):
                row["visit_hint"] = extra["visit"]
            out.append(row)
    elif members:
        by_member = {r.get("zip_member", ""): r for r in rows}
        missing = sorted(set(members) - set(by_member))
        if missing:
            raise ValueError(f"Requested members missing from manifest: {missing[:3]}")
        out = [by_member[m] for m in members]
    else:
        raise ValueError("Select clips explicitly with --members or --selection-csv.")

    seen: set[str] = set()
    for row in out:
        member = row.get("zip_member", "")
        if member in seen:
            raise ValueError(f"Duplicate selection row for {member}")
        seen.add(member)
        if str(row.get("video_kind", "")).lower() != "walking":
            raise ValueError(f"Refusing non-walking manifest row: {member}")
        allowed, reason = is_allowed_member(member)
        if not allowed:
            raise ValueError(f"Refusing {member}: {reason}")
        if row.get("class_label") not in {"CP", "ITW", "TD"}:
            raise ValueError(f"Unexpected class label for {member}: {row.get('class_label')}")
    return out


def safe_extract_one(archive_path: Path, member: str, raw_root: Path) -> Path:
    """Copy one exact MOV member to a content-keyed cache without extracting paths."""
    raw_root.mkdir(parents=True, exist_ok=True)
    safe_name = PurePosixPath(member.replace("\\", "/")).name
    destination_dir = raw_root / sha_key(member)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name
    if destination.exists():
        return destination

    allowed, reason = is_allowed_member(member)
    if not allowed:
        raise ValueError(f"Refusing unsafe or excluded archive member: {reason}: {member}")

    temp = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(archive_path, "r") as zf:
        info = zf.getinfo(member)
        if info.is_dir() or info.file_size <= 0:
            raise ValueError(f"Archive member is empty or a directory: {member}")
        with zf.open(info, "r") as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        # Reading to EOF verifies the member CRC in Python's zip implementation.
        if temp.stat().st_size != info.file_size:
            temp.unlink(missing_ok=True)
            raise IOError(f"Extracted size mismatch for {member}")
    os.replace(temp, destination)
    return destination


def find_ffprobe() -> str | None:
    if DEFAULT_FFPROBE.exists():
        return str(DEFAULT_FFPROBE)
    from shutil import which
    return which("ffprobe")


def probe_video(path: Path) -> dict[str, Any]:
    ffprobe = find_ffprobe()
    data: dict[str, Any] = {}
    if ffprobe:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation",
            "-of", "json", str(path),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if p.returncode == 0:
            try:
                payload = json.loads(p.stdout)
                streams = payload.get("streams") or []
                if streams:
                    stream = streams[0]
                    rotations = [x.get("rotation") for x in stream.get("side_data_list", []) if x.get("rotation") is not None]
                    if not rotations and stream.get("tags", {}).get("rotate") is not None:
                        rotations = [stream["tags"]["rotate"]]
                    data = {
                        "width": int(stream.get("width", 0)),
                        "height": int(stream.get("height", 0)),
                        "fps": parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
                        "frame_count": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
                        "duration_seconds": float(stream["duration"]) if stream.get("duration") else None,
                        "rotation_degrees": float(rotations[0]) if rotations else 0.0,
                        "ffprobe": ffprobe,
                    }
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                data = {}
    return data


def parse_rate(text: str | None) -> float | None:
    if not text or text in {"0/0", "N/A"}:
        return None
    try:
        a, b = text.split("/", 1)
        return float(a) / float(b)
    except (ValueError, ZeroDivisionError):
        try:
            return float(text)
        except (TypeError, ValueError):
            return None


def open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    probe = probe_video(path)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"OpenCV FFmpeg could not open MOV: {path}")

    rotation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    decoder_rotation = float(cap.get(rotation_prop)) if rotation_prop is not None else None
    auto_enabled: bool | None = None
    if auto_prop is not None:
        try:
            cap.set(auto_prop, 1)
            auto_enabled = bool(round(cap.get(auto_prop)))
        except cv2.error:
            auto_enabled = None
    meta = {
        **probe,
        "opencv_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "opencv_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "opencv_raw_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "opencv_raw_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "opencv_rotation_degrees": decoder_rotation,
        "opencv_orientation_auto": auto_enabled,
        "decoder": "opencv_ffmpeg",
    }
    return cap, meta


def read_frames(
    cap: cv2.VideoCapture,
    meta: dict[str, Any],
    max_frames: int,
    target_fps: float = 0.0,
) -> tuple[list[np.ndarray], list[int], dict[str, Any]]:
    frame_count = int(meta.get("opencv_frame_count") or 0)
    total = frame_count if frame_count > 0 else None
    source_fps = float(meta.get("fps") or meta.get("opencv_fps") or 0.0)
    wanted_indices: set[int] | None = None
    if target_fps > 0:
        if total is None or source_fps <= 0:
            raise ValueError("--target-fps requires known source frame count and frame rate.")
        if target_fps > source_fps * 1.001:
            raise ValueError(f"Target rate {target_fps:g} fps exceeds source rate {source_fps:g} fps.")
        sample_times = np.arange(0.0, total / source_fps, 1.0 / target_fps, dtype=np.float64)
        target_indices = np.unique(np.rint(sample_times * source_fps).astype(np.int64))
        target_indices = target_indices[(target_indices >= 0) & (target_indices < total)]
        if max_frames > 0 and len(target_indices) > max_frames:
            target_indices = target_indices[np.linspace(0, len(target_indices) - 1, max_frames).round().astype(np.int64)]
        wanted_indices = set(target_indices.tolist())
    elif max_frames > 0 and total and min(max_frames, total) < total:
        # Pilot frames span the full walking clip, so the check covers movement
        # instead of observing only its opening seconds.
        wanted_indices = set(np.linspace(0, total - 1, min(max_frames, total), dtype=np.int64).tolist())
    frames: list[np.ndarray] = []
    indices: list[int] = []
    source_index = 0
    while total is None or source_index < total:
        ok, frame = cap.read()
        if not ok:
            break
        if wanted_indices is None or source_index in wanted_indices:
            frames.append(frame)
            indices.append(source_index)
        source_index += 1
    if frames:
        meta["decoded_width"] = int(frames[0].shape[1])
        meta["decoded_height"] = int(frames[0].shape[0])
        meta["source_frames_decoded"] = source_index
        meta["model_frames_sampled"] = len(frames)
        meta["sampled_frame_indices"] = indices
        meta["target_fps_requested"] = target_fps or None
        if len(indices) > 1 and source_fps > 0:
            duration = max((indices[-1] - indices[0]) / source_fps, 1e-8)
            meta["sampled_fps_observed"] = (len(indices) - 1) / duration
        else:
            meta["sampled_fps_observed"] = None
        meta["rotation_auto_expected_swapped"] = bool(
            abs(abs(float(meta.get("rotation_degrees") or 0.0)) % 180 - 90) < 1
        )
        probe_w, probe_h = int(meta.get("width") or 0), int(meta.get("height") or 0)
        rotated = meta["rotation_auto_expected_swapped"]
        expected = (probe_h, probe_w) if rotated else (probe_w, probe_h)
        meta["decoded_dimensions_match_probe_rotation"] = (expected == (frames[0].shape[1], frames[0].shape[0])) if probe_w and probe_h else None
    return frames, indices, meta


def predict_in_batches(
    model: Any,
    frames: list[np.ndarray],
    *,
    imgsz: int,
    confidence: float,
    batch_size: int,
    retina_masks: bool = False,
) -> list[Any]:
    """Run the unchanged local detector weights in bounded CUDA half-precision batches."""
    results: list[Any] = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        batch_results = model.predict(
            source=chunk,
            imgsz=imgsz,
            conf=confidence,
            iou=0.50,
            device=0,
            quantize=16,
            batch=len(chunk),
            verbose=False,
            max_det=20,
            **({"classes": [0], "retina_masks": True} if retina_masks else {}),
        )
        if len(batch_results) != len(chunk):
            raise RuntimeError(f"Batched inference returned {len(batch_results)} results for {len(chunk)} frames.")
        results.extend(batch_results)
    return results


def decode_pose(result: Any, slot: int, frame_index: int) -> list[Detection]:
    keypoints_obj = result.keypoints
    if keypoints_obj is None or keypoints_obj.xy is None:
        return []
    xy = keypoints_obj.xy.detach().cpu().numpy()
    if keypoints_obj.conf is not None:
        conf = keypoints_obj.conf.detach().cpu().numpy()
    else:
        conf = np.ones(xy.shape[:2], dtype=np.float32)
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    box_conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), np.float32)
    detections: list[Detection] = []
    for i in range(min(len(xy), len(xyxy))):
        kp = np.concatenate([xy[i], conf[i, :, None]], axis=1).astype(np.float32)
        n_visible = int((kp[:, 2] >= POSE_CONF_MIN).sum())
        if n_visible < MIN_VISIBLE_JOINTS:
            continue
        x1, y1, x2, y2 = xyxy[i].astype(np.float32)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(slot, frame_index, i, xyxy[i].astype(np.float32), kp, float(box_conf[i])))
    return detections


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-8)


def track_detections(detections_by_frame: list[list[Detection]]) -> list[Track]:
    tracks: list[Track] = []
    next_id = 1
    for frame_slot, detections in enumerate(detections_by_frame):
        eligible = [t for t in tracks if t.missed <= TRACK_MAX_GAP]
        assigned_dets: set[int] = set()
        assigned_tracks: set[int] = set()
        if eligible and detections:
            cost = np.full((len(eligible), len(detections)), 1e6, dtype=np.float32)
            for ti, track in enumerate(eligible):
                previous = track.last
                prev_box = previous.box
                prev_c = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
                prev_scale = max(float(max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1])), 1.0)
                for di, det in enumerate(detections):
                    c = np.array([(det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2])
                    scale = max(float(max(det.box[2] - det.box[0], det.box[3] - det.box[1])), 1.0)
                    distance = float(np.linalg.norm(c - prev_c) / max(prev_scale, scale))
                    iou = box_iou(prev_box, det.box)
                    ratio = abs(math.log(max(scale, 1.0) / max(prev_scale, 1.0)))
                    # Reject implausible jumps before assignment; tie handling is audited below.
                    if distance <= 2.25 and (iou >= 0.01 or distance <= 0.8):
                        cost[ti, di] = 0.55 * min(distance, 2.25) + 0.35 * (1 - iou) + 0.10 * min(ratio, 2.0)
            try:
                from scipy.optimize import linear_sum_assignment
                rr, cc = linear_sum_assignment(cost)
            except Exception:
                rr, cc = _greedy_assignment(cost)
            for ti, di in zip(rr.tolist(), cc.tolist()):
                if cost[ti, di] >= 1e5:
                    continue
                track, det = eligible[ti], detections[di]
                # Do not silently link on an ambiguous near-tie.
                alternatives = np.sort(cost[ti])
                if len(alternatives) > 1 and alternatives[0] < 1e5 and alternatives[1] - alternatives[0] < 0.04:
                    continue
                det.track_id = track.track_id
                track.detections.append(det)
                track.missed = 0
                assigned_dets.add(di)
                assigned_tracks.add(track.track_id)
        for track in tracks:
            if track.track_id not in assigned_tracks:
                track.missed += 1
        unassigned = [d for i, d in enumerate(detections) if i not in assigned_dets]
        if frame_slot == 0 and unassigned:
            unassigned.sort(key=lambda d: (float((d.box[0] + d.box[2]) / 2), float((d.box[1] + d.box[3]) / 2)))
        for det in unassigned:
            det.track_id = next_id
            tracks.append(Track(next_id, [det], 0))
            next_id += 1
    return [t for t in tracks if t.detections]


def _greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    work = cost.copy()
    while work.size and np.min(work) < 1e5:
        ti, di = np.unravel_index(np.argmin(work), work.shape)
        pairs.append((int(ti), int(di)))
        work[ti, :] = 1e6
        work[:, di] = 1e6
    if not pairs:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def track_stats(track: Track) -> dict[str, Any]:
    boxes = np.stack([d.box for d in track.detections])
    visible = [int((d.keypoints[:, 2] >= POSE_CONF_MIN).sum()) for d in track.detections]
    centers = np.stack(
        [(d.box[[0, 1]] + d.box[[2, 3]]) / 2 for d in track.detections]
    ).astype(np.float32)
    extents = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).astype(np.float32)
    # Median smoothing suppresses detector jitter for movement disambiguation.
    # A unique reliable pose-mask track is accepted independently of net motion,
    # since a GAIT clip may include walking away and returning to the start point.
    smooth_centers = centers.copy()
    smooth_extents = extents.copy()
    if len(centers) >= 5:
        radius = 2
        for i in range(len(centers)):
            lo, hi = max(0, i - radius), min(len(centers), i + radius + 1)
            smooth_centers[i] = np.median(centers[lo:hi], axis=0)
            smooth_extents[i] = np.median(extents[lo:hi])
    body_scale = max(float(np.median(extents)), 1.0)
    net_motion = float(np.linalg.norm(smooth_centers[-1] - smooth_centers[0]) / body_scale) if len(centers) > 1 else 0.0
    path_motion = float(np.linalg.norm(np.diff(smooth_centers, axis=0), axis=1).sum() / body_scale) if len(centers) > 1 else 0.0
    scale_change = (
        abs(float(math.log(max(smooth_extents[-1], 1.0) / max(smooth_extents[0], 1.0))))
        if len(centers) > 1 else 0.0
    )
    moving_candidate = bool(
        len(centers) >= 6
        and (net_motion >= 0.20 or (path_motion >= 0.55 and net_motion >= 0.08) or scale_change >= 0.15)
    )
    return {
        "track_id": track.track_id,
        "detections": len(track.detections),
        "first_frame": track.detections[0].frame_index,
        "last_frame": track.detections[-1].frame_index,
        "median_box_height": float(np.median(boxes[:, 3] - boxes[:, 1])),
        "median_box_width": float(np.median(boxes[:, 2] - boxes[:, 0])),
        "mean_pose_confidence": float(np.mean([d.score for d in track.detections])),
        "median_visible_joints": float(np.median(visible)),
        "net_motion_body_scales": net_motion,
        "path_motion_body_scales": path_motion,
        "scale_change_log_ratio": scale_change,
        "moving_candidate": moving_candidate,
        "candidate_eligible": bool(len(track.detections) >= 2 and np.median(visible) >= MIN_VISIBLE_JOINTS),
    }


def summarize_track_overlaps(tracks: list[Track]) -> list[dict[str, Any]]:
    """Describe positive same-frame overlap without joining track identities."""
    overlaps = []
    for i, a in enumerate(tracks):
        a_slots = {d.frame_slot for d in a.detections}
        for b in tracks[i + 1 :]:
            shared = len(a_slots.intersection(d.frame_slot for d in b.detections))
            if shared == 0:
                continue
            smaller_n = min(len(a.detections), len(b.detections))
            ratio = shared / max(smaller_n, 1)
            overlaps.append({
                "track_id_a": a.track_id,
                "track_id_b": b.track_id,
                "shared_detection_frames": shared,
                "overlap_fraction_of_shorter_track": float(ratio),
                "significant": bool(
                    shared >= SIGNIFICANT_TRACK_OVERLAP_MIN_FRAMES
                    and ratio >= SIGNIFICANT_TRACK_OVERLAP_MIN_RATIO
                ),
            })
    return overlaps


def decode_person_masks(result: Any, shape: tuple[int, int]) -> list[dict[str, Any]]:
    boxes = result.boxes
    masks_obj = result.masks
    if boxes is None or masks_obj is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), np.int32)
    masks = masks_obj.data.detach().cpu().numpy()
    h, w = shape
    names = getattr(result, "names", {}) or {}
    out = []
    for i, box in enumerate(xyxy):
        class_name = str(names.get(int(cls[i]), "")).lower() if isinstance(names, dict) else ""
        if int(cls[i]) != 0 and class_name != "person":
            continue
        if i >= len(masks):
            continue
        mask = (masks[i] > 0.5).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append({"box": box.astype(np.float32), "confidence": float(conf[i]), "mask": mask})
    return out


def associate_mask(
    pose_det: Detection,
    masks: list[dict[str, Any]],
    min_mask_confidence: float = 0.20,
) -> tuple[dict[str, Any] | None, float, bool]:
    if not masks:
        return None, 0.0, False
    scored = []
    for item in masks:
        iou = box_iou(pose_det.box, item["box"])
        pcenter = np.array([(pose_det.box[0] + pose_det.box[2]) / 2, (pose_det.box[1] + pose_det.box[3]) / 2])
        mcenter = np.array([(item["box"][0] + item["box"][2]) / 2, (item["box"][1] + item["box"][3]) / 2])
        pscale = max(float(pose_det.box[3] - pose_det.box[1]), 1.0)
        center_score = math.exp(-float(np.linalg.norm(pcenter - mcenter)) / pscale)
        scored.append((0.72 * iou + 0.28 * center_score, iou, item))
    scored.sort(key=lambda z: z[0], reverse=True)
    best, best_iou, item = scored[0]
    competing = len(scored) > 1 and (scored[0][0] - scored[1][0] < 0.035) and scored[1][1] >= 0.15
    valid = best_iou >= 0.18 and item["confidence"] >= min_mask_confidence and not competing
    return item if valid else None, float(best), bool(competing)


def crop_transform(center: tuple[float, float], side: float) -> np.ndarray:
    scale = MAP_SIZE / max(float(side), 1.0)
    cx, cy = center
    return np.array(
        [[scale, 0.0, MAP_SIZE / 2 - scale * cx],
         [0.0, scale, MAP_SIZE / 2 - scale * cy]],
        dtype=np.float32,
    )


def warp_binary_mask(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    crop = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (MAP_SIZE, MAP_SIZE),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return (crop > 0).astype(np.uint8)


def transform_keypoints(keypoints: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xy = keypoints[:, :2]
    mapped = xy @ matrix[:, :2].T + matrix[:, 2]
    return np.concatenate([mapped.astype(np.float32), keypoints[:, 2:3]], axis=1)


def _draw_gaussian_max(canvas: np.ndarray, x: float, y: float, weight: float, sigma: float) -> None:
    if weight <= 1e-3:
        return
    radius = max(1, int(math.ceil(3 * sigma)))
    cx, cy = int(round(x)), int(round(y))
    x1, x2 = max(0, cx - radius), min(MAP_SIZE, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(MAP_SIZE, cy + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return
    yy, xx = np.mgrid[y1:y2, x1:x2]
    patch = weight * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    np.maximum(canvas[y1:y2, x1:x2], patch.astype(np.float32), out=canvas[y1:y2, x1:x2])


def _draw_limb(canvas: np.ndarray, a: np.ndarray, b: np.ndarray, weight: float, sigma: float) -> None:
    if weight <= 1e-3:
        return
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    radius = max(1, int(math.ceil(3 * sigma)))
    x1 = max(0, int(math.floor(min(ax, bx) - radius)))
    x2 = min(MAP_SIZE, int(math.ceil(max(ax, bx) + radius + 1)))
    y1 = max(0, int(math.floor(min(ay, by) - radius)))
    y2 = min(MAP_SIZE, int(math.ceil(max(ay, by) + radius + 1)))
    if x1 >= x2 or y1 >= y2:
        return
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 < 1e-6:
        _draw_gaussian_max(canvas, ax, ay, weight, sigma)
        return
    t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / length2, 0.0, 1.0)
    dx, dy = xx - (ax + t * vx), yy - (ay + t * vy)
    patch = weight * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    np.maximum(canvas[y1:y2, x1:x2], patch, out=canvas[y1:y2, x1:x2])


def make_pose_maps(keypoints: np.ndarray) -> np.ndarray:
    """Return [limb, joint] maps using confidence-weighted Gaussian fields."""
    maps = np.zeros((2, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for x, y, confidence in keypoints:
        _draw_gaussian_max(maps[1], float(x), float(y), float(confidence), GAUSSIAN_SIGMA)
    for a, b in COCO17_LIMBS:
        va, vb = float(keypoints[a, 2]), float(keypoints[b, 2])
        if va >= POSE_CONF_MIN and vb >= POSE_CONF_MIN:
            _draw_limb(maps[0], keypoints[a], keypoints[b], min(va, vb), GAUSSIAN_SIGMA)
    return np.clip(maps, 0.0, 1.0)


def make_qc_png(
    path: Path,
    frame_index: int,
    maps: np.ndarray,
    silhouette: np.ndarray | None,
    track_id: int,
    person_count: int,
    mask_confidence: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = MAP_SIZE
    limb = (np.clip(maps[0], 0, 1) * 255).astype(np.uint8)
    joint = (np.clip(maps[1], 0, 1) * 255).astype(np.uint8)
    sil = np.zeros((h, h), np.uint8) if silhouette is None else ((silhouette > 0).astype(np.uint8) * 255)
    overlay = cv2.cvtColor(sil, cv2.COLOR_GRAY2BGR)
    # Distinct color accents encode pose geometry only; no source pixels are used.
    kpts = getattr(make_qc_png, "_last_keypoints", None)
    if kpts is not None:
        for a, b in COCO17_LIMBS:
            if kpts[a, 2] >= POSE_CONF_MIN and kpts[b, 2] >= POSE_CONF_MIN:
                p1 = tuple(np.round(kpts[a, :2]).astype(int))
                p2 = tuple(np.round(kpts[b, :2]).astype(int))
                cv2.line(overlay, p1, p2, (0, 220, 255), 1, cv2.LINE_AA)
        for x, y, c in kpts:
            if c >= POSE_CONF_MIN:
                cv2.circle(overlay, tuple(np.round([x, y]).astype(int)), 1, (0, 80, 255), -1, cv2.LINE_AA)
    panels = []
    for im, label in ((limb, "limb"), (joint, "joint"), (sil, "silhouette"), (cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), "pose_on_silhouette")):
        panel = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, label, (2, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(panel, (192, 192), interpolation=cv2.INTER_NEAREST))
    board = np.concatenate(panels, axis=1)
    title = np.zeros((24, board.shape[1], 3), np.uint8)
    mask_text = "none" if mask_confidence is None else f"{mask_confidence:.3f}"
    cv2.putText(title, f"track={track_id} frame={frame_index} candidates={person_count} mask_conf={mask_text}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.concatenate([title, board], axis=0))


def requested_track_id(row: dict[str, str]) -> int | None:
    value = (row.get("target_track_id") or "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as e:
        raise ValueError(f"target_track_id must be an integer: {value}") from e
    if parsed <= 0:
        raise ValueError("target_track_id must be a positive per-clip track ID.")
    return parsed


def canonical_phase(row: dict[str, str]) -> str:
    """Prefer the manifest phase and fall back to the canonical visit field."""
    return str(row.get("phase") or row.get("visit_hint") or row.get("visit") or "").strip()


def choose_target_track(
    tracks: list[Track],
    summaries: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    requested: int | None,
) -> tuple[Track | None, str, str, list[int], list[int]]:
    """Apply the frozen, label-blind track rule; never stitch separate tracks."""
    by_id = {s["track_id"]: s for s in summaries}
    reliable = [
        t for t in tracks
        if by_id[t.track_id].get("reliable_target_candidate", False)
    ]
    reliable_ids = [t.track_id for t in reliable]
    moving = [t for t in reliable if by_id[t.track_id].get("moving_candidate", False)]
    moving_ids = [t.track_id for t in moving]

    if requested is not None:
        selected = next((t for t in tracks if t.track_id == requested), None)
        if selected is None:
            return None, "explicit_track_id", "requested_track_not_found", reliable_ids, moving_ids
        if selected.track_id not in set(reliable_ids):
            return None, "explicit_track_id", "requested_track_fails_reliability_qc", reliable_ids, moving_ids
        return selected, "explicit_track_id", "requested_reliable_track", reliable_ids, moving_ids

    if len(reliable) == 1:
        return reliable[0], "single_reliable_pose_mask_track", "only_reliable_track_meets_frozen_qc", reliable_ids, moving_ids

    if len(moving) == 1:
        return moving[0], "unique_moving_reliable_track", "only_one_reliable_track_has_consistent_scene_motion", reliable_ids, moving_ids

    if len(moving) > 1:
        moving_set = set(moving_ids)
        significant = [
            p for p in overlaps
            if p["significant"] and p["track_id_a"] in moving_set and p["track_id_b"] in moving_set
        ]
        if significant:
            return None, "unresolved_overlapping_reliable_tracks", "multiple_reliable_moving_tracks_overlap_in_time", reliable_ids, moving_ids
        selected = max(
            moving,
            key=lambda t: (
                by_id[t.track_id]["valid_pose_mask_frames"],
                -by_id[t.track_id]["first_frame"],
                -t.track_id,
            ),
        )
        return selected, "longest_nonoverlapping_reliable_track", "selected_single_track_with_most_valid_pose_mask_frames", reliable_ids, moving_ids

    if not reliable:
        reason = "no_track_meets_30_valid_frames_and_mean_pose_confidence_0.25"
    else:
        reason = "multiple_reliable_tracks_without_a_unique_moving_target"
    return None, "unresolved_no_unique_reliable_target", reason, reliable_ids, moving_ids


def full_output_path(row: dict[str, str], output_root: Path, member_hash: str) -> Path:
    """Return only the selected, content-keyed full NPZ path under our cache."""
    out_path = Path(row.get("npz_path") or (output_root / f"{member_hash}.npz")).resolve()
    expected_root = output_root.resolve()
    if not out_path.is_relative_to(expected_root):
        raise ValueError(f"Selected NPZ path escapes the derivative cache: {out_path}")
    if out_path.name != f"{member_hash}.npz":
        raise ValueError(f"Selected NPZ filename does not match the ZIP member hash: {out_path.name}")
    return out_path


def validate_resume_cache(
    row: dict[str, str],
    output_root: Path,
    target_fps: float,
) -> tuple[bool, str, dict[str, Any]]:
    """Skip only a verified, complete, same-protocol artifact for this exact clip."""
    member = row["zip_member"]
    out_path = full_output_path(row, output_root, sha_key(member))
    if not out_path.is_file():
        return False, "missing", {"output_npz": str(out_path)}
    try:
        with np.load(out_path, allow_pickle=False) as z:
            required = {
                "schema", "video_id", "skeleton_maps", "silhouette", "frame_indices", "phase",
                "valid_frame_mask", "pose_valid_mask", "mask_valid_mask", "subject_id", "class_label", "view", "visit",
                "is_smoke", "max_frames", "target_fps", "sampling_mode",
                "source_fps", "source_frame_count", "selection_rule_version", "track_id", "selection_mode",
                "selection_reason", "selected_track_pose_frames", "selected_track_valid_pair_frames",
                "selected_track_frame_coverage", "selected_track_valid_clip_fraction", "selected_track_is_partial",
                "metadata_json",
            }
            missing = required - set(z.files)
            if missing:
                return False, f"missing_keys:{','.join(sorted(missing))}", {"output_npz": str(out_path)}
            metadata = json.loads(str(z["metadata_json"].item()))
            if str(z["schema"].item()) != "live-video-input-v1" or metadata.get("schema") != "live-video-input-v1":
                return False, "schema_mismatch", {"output_npz": str(out_path)}
            if str(z["selection_rule_version"].item()) != SELECTION_RULE_VERSION or metadata.get("selection_rule_version") != SELECTION_RULE_VERSION:
                return False, "selection_rule_version_mismatch", {"output_npz": str(out_path)}
            expected_values = {
                "subject_id": row.get("subject_id", ""),
                "class_label": row.get("class_label", ""),
                "view": row.get("view_hint", ""),
                "visit": row.get("visit_hint", ""),
                "phase": canonical_phase(row),
                "video_id": row.get("video_id", sha_key(member)),
            }
            for key, expected in expected_values.items():
                if str(z[key].item()) != str(expected) or str(metadata.get(key, "")) != str(expected):
                    return False, f"identity_mismatch:{key}", {"output_npz": str(out_path)}
            if str(z["zip_member"].item()) != member or metadata.get("zip_member") != member:
                return False, "zip_member_mismatch", {"output_npz": str(out_path)}
            if "requested_track_id" not in metadata:
                return False, "missing_requested_track_metadata", {"output_npz": str(out_path)}
            expected_track = requested_track_id(row)
            stored_requested_track = metadata.get("requested_track_id")
            if stored_requested_track is not None:
                try:
                    stored_requested_track = int(stored_requested_track)
                except (TypeError, ValueError):
                    return False, "invalid_requested_track_metadata", {"output_npz": str(out_path)}
            if stored_requested_track != expected_track:
                return False, "requested_track_mismatch", {"output_npz": str(out_path)}
            if expected_track is not None and int(z["track_id"].item()) != expected_track:
                return False, "selected_track_mismatch", {"output_npz": str(out_path)}
            selected_track_id = int(z["track_id"].item())
            if int(metadata.get("track_id", -1)) != selected_track_id:
                return False, "metadata_selected_track_mismatch", {"output_npz": str(out_path)}
            selected_track_summary = next(
                (t for t in metadata.get("track_candidates", []) if int(t.get("track_id", -1)) == selected_track_id),
                None,
            )
            if not selected_track_summary or not selected_track_summary.get("reliable_target_candidate"):
                return False, "selected_track_fails_reliability_qc", {"output_npz": str(out_path)}
            if float(selected_track_summary.get("mean_pose_confidence", 0.0)) < MIN_RELIABLE_MEAN_POSE_CONFIDENCE:
                return False, "selected_track_mean_confidence_below_threshold", {"output_npz": str(out_path)}
            if bool(z["is_smoke"].item()) or int(z["max_frames"].item()) != 0:
                return False, "smoke_or_capped_artifact", {"output_npz": str(out_path)}
            stored_fps = float(z["target_fps"].item())
            if not math.isclose(stored_fps, target_fps, rel_tol=0.0, abs_tol=1e-3):
                return False, "target_fps_mismatch", {"output_npz": str(out_path)}
            expected_mode = f"uniform_{target_fps:g}fps" if target_fps else "all_frames"
            if str(z["sampling_mode"].item()) != expected_mode or metadata.get("sampling_mode") != expected_mode or int(metadata.get("max_frames", -1)) != 0 or metadata.get("is_smoke") is not False:
                return False, "sampling_metadata_mismatch", {"output_npz": str(out_path)}
            if not math.isclose(float(metadata.get("target_fps_requested") or 0.0), target_fps, rel_tol=0.0, abs_tol=1e-3):
                return False, "metadata_target_fps_mismatch", {"output_npz": str(out_path)}
            frame_indices = np.asarray(z["frame_indices"], dtype=np.int64)
            source_count = int(z["source_frame_count"].item())
            stored_source_fps = float(z["source_fps"].item())
            if source_count <= 0 or frame_indices.size == 0 or np.any(np.diff(frame_indices) <= 0):
                return False, "invalid_frame_index_sequence", {"output_npz": str(out_path)}
            if target_fps == 0:
                expected_indices = np.arange(source_count, dtype=np.int64)
            else:
                video_meta = metadata.get("video", {})
                if not isinstance(video_meta, dict):
                    return False, "missing_source_video_metadata", {"output_npz": str(out_path)}
                sampling_fps = float(video_meta.get("fps") or video_meta.get("opencv_fps") or 0.0)
                metadata_frame_count = int(video_meta.get("opencv_frame_count") or 0)
                if sampling_fps <= 0 or stored_source_fps <= 0:
                    return False, "missing_source_fps", {"output_npz": str(out_path)}
                if metadata_frame_count <= 0 or metadata_frame_count != source_count:
                    return False, "source_frame_count_metadata_mismatch", {"output_npz": str(out_path)}
                if not math.isclose(
                    stored_source_fps,
                    float(np.float32(sampling_fps)),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    return False, "source_fps_metadata_mismatch", {"output_npz": str(out_path)}
                # The extractor samples using the original Python-precision
                # video fps in metadata; source_fps is a float32 scalar and
                # can round a half-frame boundary differently when replayed.
                times = np.arange(0.0, source_count / sampling_fps, 1.0 / target_fps, dtype=np.float64)
                expected_indices = np.unique(np.rint(times * sampling_fps).astype(np.int64))
                expected_indices = expected_indices[(expected_indices >= 0) & (expected_indices < source_count)]
            if not np.array_equal(frame_indices, expected_indices):
                return False, "not_full_clip_frame_sequence", {"output_npz": str(out_path)}
            valid_frame_mask = np.asarray(z["valid_frame_mask"], dtype=np.bool_)
            pose_valid_mask = np.asarray(z["pose_valid_mask"], dtype=np.bool_)
            mask_valid_mask = np.asarray(z["mask_valid_mask"], dtype=np.bool_)
            if len(valid_frame_mask) != len(frame_indices):
                return False, "valid_mask_length_mismatch", {"output_npz": str(out_path)}
            if len(pose_valid_mask) != len(frame_indices) or len(mask_valid_mask) != len(frame_indices):
                return False, "modality_mask_length_mismatch", {"output_npz": str(out_path)}
            if not np.array_equal(valid_frame_mask, pose_valid_mask & mask_valid_mask):
                return False, "joint_valid_mask_mismatch", {"output_npz": str(out_path)}
            if int(valid_frame_mask.sum()) < MIN_RELIABLE_VALID_FRAMES:
                return False, "below_minimum_valid_pose_mask_frames", {"output_npz": str(out_path)}
            if z["skeleton_maps"].shape != (len(frame_indices), 2, MAP_SIZE, MAP_SIZE):
                return False, "skeleton_shape_mismatch", {"output_npz": str(out_path)}
            if z["skeleton_maps"].dtype != np.float16:
                return False, "skeleton_dtype_mismatch", {"output_npz": str(out_path)}
            if z["silhouette"].shape != (len(frame_indices), 1, MAP_SIZE, MAP_SIZE):
                return False, "silhouette_shape_mismatch", {"output_npz": str(out_path)}
            silhouette = z["silhouette"]
            if silhouette.dtype != np.uint8 or not np.isin(silhouette, (0, 255)).all():
                return False, "silhouette_value_contract_mismatch", {"output_npz": str(out_path)}
            return True, "verified", {
                "output_npz": str(out_path),
                "video_id": row.get("video_id", ""),
                "subject_id": row.get("subject_id", ""),
                "class_label": row.get("class_label", ""),
                "view": row.get("view_hint", ""),
                "visit": row.get("visit_hint", ""),
                "phase": canonical_phase(row),
                "source_frame_count": source_count,
                "model_frames_sampled": int(len(frame_indices)),
                "valid_frames": int(valid_frame_mask.sum()),
                "is_smoke": False,
                "max_frames": 0,
                "target_fps": target_fps or None,
                "resume_cache_validation": "verified",
            }
    except Exception as exc:
        return False, f"unreadable_or_invalid:{type(exc).__name__}", {"output_npz": str(out_path)}


def persist_progress(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(report_path.suffix + ".part")
    temp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, report_path)


def print_clip_summary(item: dict[str, Any]) -> None:
    """Emit one bounded status line; candidate and frame detail stays in JSON."""
    clip = item.get("video_id") or item.get("zip_member", "unknown")
    frames = item.get("model_frames_sampled", item.get("frames_decoded", 0))
    elapsed = item.get("elapsed_seconds")
    elapsed_text = f"{float(elapsed):.2f}s" if elapsed is not None else "cache"
    print(f"{clip} status={item.get('status', 'unknown')} frames={int(frames or 0)} elapsed={elapsed_text}", flush=True)


DECISION_FIELDS = (
    "video_id", "subject_id", "class_label", "view", "visit", "phase", "zip_member",
    "decision", "decision_reason", "status", "selection_mode", "track_id",
    "model_frames_sampled", "valid_frames", "valid_fraction",
    "selected_track_pose_frames", "selected_track_valid_pair_frames",
    "selected_track_frame_coverage", "selected_track_valid_clip_fraction",
    "output_npz", "error_type", "error",
)


def write_decisions_csv(path: Path, report: dict[str, Any]) -> None:
    """Atomically keep a one-row-per-processed-candidate decision ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in report.get("clips", []):
            is_formal_full = (
                item.get("is_smoke") is False
                and int(item.get("max_frames", -1)) == 0
                and bool(item.get("output_npz"))
                and Path(item["output_npz"]).is_file()
            )
            accepted = is_accepted_full_input(item, is_formal_full)
            row = dict(item)
            row["decision"] = "accepted" if accepted else "excluded"
            if accepted:
                row["decision_reason"] = item.get("selection_reason") or "full_npz_passed_30_frame_qc"
            elif item.get("error_type"):
                row["decision_reason"] = f"preprocessing_error:{item['error_type']}"
            else:
                row["decision_reason"] = item.get("exclusion_reason") or item.get("excluded_reason") or item.get("status", "not_processed")
            writer.writerow({key: row.get(key, "") for key in DECISION_FIELDS})
    os.replace(temp_path, path)


def is_accepted_full_input(item: dict[str, Any], file_verified: bool | None = None) -> bool:
    if file_verified is None:
        file_verified = bool(item.get("output_npz")) and Path(item["output_npz"]).is_file()
    return bool(
        item.get("status") in {"ok", "skipped_resume_verified"}
        and item.get("is_smoke") is False
        and int(item.get("max_frames", -1)) == 0
        and int(item.get("valid_frames", 0)) >= MIN_RELIABLE_VALID_FRAMES
        and file_verified
    )


def refresh_candidate_summary(report: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    retention: dict[str, dict[str, int]] = {}
    for item in report.get("clips", []):
        status = item.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        key = f"{item.get('class_label') or 'unknown'}__{item.get('view') or 'unknown'}"
        group = retention.setdefault(key, {"processed": 0, "accepted": 0, "excluded": 0})
        group["processed"] += 1
        if is_accepted_full_input(item):
            group["accepted"] += 1
        else:
            group["excluded"] += 1
    report["counts"] = counts
    report["candidate_decision_rows"] = len(report.get("clips", []))
    report["retention_by_class_view"] = retention


def process_one(
    row: dict[str, str],
    archive_path: Path,
    raw_root: Path,
    output_root: Path,
    qc_root: Path,
    pose_model: Any,
    seg_model: Any,
    max_frames: int,
    target_fps: float,
    imgsz: int,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    member = row["zip_member"]
    member_hash = sha_key(member)
    is_smoke = max_frames > 0
    if is_smoke:
        artifact_tag = f"sample{max_frames}"
        out_path = output_root / f"{member_hash}_{artifact_tag}.npz"
    else:
        # Agent 1's eligible manifest is the training-input path authority.
        # Validate it stays in our derivative cache and is content-keyed before use.
        out_path = full_output_path(row, output_root, member_hash)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")
        out_path.unlink()
    started = time.perf_counter()
    video_path = safe_extract_one(archive_path, member, raw_root)
    cap, video_meta = open_video(video_path)
    try:
        frames, frame_indices, video_meta = read_frames(cap, video_meta, max_frames, target_fps)
    finally:
        cap.release()
    if not frames:
        raise IOError(f"No decodable video frames: {member}")
    if video_meta.get("decoded_dimensions_match_probe_rotation") is False:
        raise ValueError(f"Decoded dimensions do not reflect MOV rotation metadata: {member}")

    inference_started = time.perf_counter()
    pose_by_frame: list[list[Detection]] = []
    counts = []
    pose_results = predict_in_batches(
        pose_model, frames, imgsz=imgsz, confidence=0.01, batch_size=batch_size,
    )
    for slot, (result, frame_index) in enumerate(zip(pose_results, frame_indices)):
        dets = decode_pose(result, slot, frame_index)
        pose_by_frame.append(dets)
        counts.append(len(dets))
    tracks = track_detections(pose_by_frame)
    seg_results = predict_in_batches(
        seg_model, frames, imgsz=imgsz, confidence=0.001, batch_size=batch_size,
        retina_masks=True,
    )
    masks_by_slot: list[list[dict[str, Any]]] = []
    for result, frame in zip(seg_results, frames):
        masks_by_slot.append(decode_person_masks(result, frame.shape[:2]))
    inference_seconds = time.perf_counter() - inference_started

    track_mask_matches: dict[int, dict[int, tuple[dict[str, Any], float, float]]] = {}
    track_mask_qc_matches: dict[int, dict[int, tuple[dict[str, Any], float, float]]] = {}
    track_summaries = []
    for track in tracks:
        matches: dict[int, tuple[dict[str, Any], float, float]] = {}
        qc_matches: dict[int, tuple[dict[str, Any], float, float]] = {}
        previous_pose: Detection | None = None
        previous_mask_box: np.ndarray | None = None
        for det in track.detections:
            qc_item, qc_score, qc_competing = associate_mask(
                det, masks_by_slot[det.frame_slot], min_mask_confidence=0.001
            )
            if qc_item is not None and not qc_competing:
                if previous_pose is not None and previous_mask_box is not None:
                    pose_center_before = (previous_pose.box[:2] + previous_pose.box[2:]) / 2
                    pose_center_now = (det.box[:2] + det.box[2:]) / 2
                    mask_center_before = (previous_mask_box[:2] + previous_mask_box[2:]) / 2
                    mask_center_now = (qc_item["box"][:2] + qc_item["box"][2:]) / 2
                    scale = max(
                        float(previous_pose.box[3] - previous_pose.box[1]),
                        float(det.box[3] - det.box[1]),
                        1.0,
                    )
                    pose_delta = pose_center_now - pose_center_before
                    mask_delta = mask_center_now - mask_center_before
                    motion_mismatch = float(np.linalg.norm(pose_delta - mask_delta) / scale)
                    current_offset = float(np.linalg.norm(pose_center_now - mask_center_now) / scale)
                    if motion_mismatch > 0.75 or current_offset > 0.75:
                        qc_item = None
                if qc_item is not None:
                    previous_pose = det
                    previous_mask_box = qc_item["box"]
            if qc_item is not None and not qc_competing:
                overlap = box_iou(det.box, qc_item["box"])
                qc_matches[det.frame_slot] = (qc_item, qc_score, overlap)
                if qc_item["confidence"] >= 0.20:
                    matches[det.frame_slot] = (qc_item, qc_score, overlap)
        track_mask_matches[track.track_id] = matches
        track_mask_qc_matches[track.track_id] = qc_matches
        summary = track_stats(track)
        paired_frames = len(matches)
        paired_coverage = paired_frames / max(len(track.detections), 1)
        qc_confidences = [float(m[0]["confidence"]) for m in qc_matches.values()]
        qc_ious = [float(m[2]) for m in qc_matches.values()]
        summary["mask_candidate_frames"] = len(qc_matches)
        summary["mean_mask_candidate_confidence"] = float(np.mean(qc_confidences)) if qc_confidences else 0.0
        summary["best_mask_candidate_confidence"] = float(max(qc_confidences)) if qc_confidences else 0.0
        summary["mean_mask_candidate_box_iou"] = float(np.mean(qc_ious)) if qc_ious else 0.0
        summary["paired_mask_frames"] = paired_frames
        summary["paired_mask_coverage"] = float(paired_coverage)
        summary["paired_candidate_eligible"] = bool(
            summary["candidate_eligible"]
            and paired_frames >= 2
            and paired_coverage >= 0.20
        )
        track_boxes = np.stack([d.box for d in track.detections])
        track_extents = np.maximum(
            track_boxes[:, 2] - track_boxes[:, 0],
            track_boxes[:, 3] - track_boxes[:, 1],
        )
        track_crop_side = float(np.median(track_extents) * 1.25)
        valid_pose_mask_frames = 0
        for det in track.detections:
            match = matches.get(det.frame_slot)
            if match is None:
                continue
            pose_center = (
                float((det.box[0] + det.box[2]) / 2),
                float((det.box[1] + det.box[3]) / 2),
            )
            matrix = crop_transform(pose_center, track_crop_side)
            candidate_silhouette = warp_binary_mask(match[0]["mask"], matrix)
            occupancy = float(candidate_silhouette.mean())
            if 0.025 <= occupancy <= 0.95:
                valid_pose_mask_frames += 1
        summary["valid_pose_mask_frames"] = valid_pose_mask_frames
        summary["valid_pose_mask_clip_fraction"] = valid_pose_mask_frames / max(len(frames), 1)
        summary["reliable_target_candidate"] = bool(
            summary["candidate_eligible"]
            and valid_pose_mask_frames >= MIN_RELIABLE_VALID_FRAMES
            and summary["mean_pose_confidence"] >= MIN_RELIABLE_MEAN_POSE_CONFIDENCE
        )
        summary["reliability_failures"] = []
        if valid_pose_mask_frames < MIN_RELIABLE_VALID_FRAMES:
            summary["reliability_failures"].append("fewer_than_30_valid_pose_mask_frames")
        if summary["mean_pose_confidence"] < MIN_RELIABLE_MEAN_POSE_CONFIDENCE:
            summary["reliability_failures"].append("mean_pose_confidence_below_0.25")
        track_summaries.append(summary)
    summary_by_id = {s["track_id"]: s for s in track_summaries}
    pose_tracks = [t for t in tracks if summary_by_id[t.track_id]["candidate_eligible"]]
    track_overlap_pairs = summarize_track_overlaps(tracks)
    requested = requested_track_id(row)
    chosen, selection_mode, selection_reason, reliable_track_ids, moving_reliable_track_ids = choose_target_track(
        tracks, track_summaries, track_overlap_pairs, requested
    )

    if chosen is None:
        # Produce pose/mask-only candidate QC to support a later explicit track choice.
        if out_path.exists():
            out_path.unlink()
        qc_written = 0
        for track in pose_tracks:
            stats = summary_by_id[track.track_id]
            side = max(stats["median_box_height"], stats["median_box_width"]) * 1.25
            representative = np.linspace(0, len(track.detections) - 1, min(3, len(track.detections))).round().astype(int)
            for j in representative:
                det = track.detections[int(j)]
                # QC may show a low-confidence matching silhouette for review;
                # such masks never qualify frames for an NPZ training sample.
                qc_match = track_mask_qc_matches[track.track_id].get(det.frame_slot)
                mask_item = qc_match[0] if qc_match else None
                center = (float((det.box[0] + det.box[2]) / 2), float((det.box[1] + det.box[3]) / 2))
                matrix = crop_transform(center, side)
                kp = transform_keypoints(det.keypoints, matrix)
                maps = make_pose_maps(kp)
                sil = None if mask_item is None else warp_binary_mask(mask_item["mask"], matrix)
                setattr(make_qc_png, "_last_keypoints", kp)
                make_qc_png(
                    qc_root / f"{member_hash}_track{track.track_id:03d}_f{det.frame_index:06d}.png",
                    det.frame_index, maps, sil, track.track_id, max(counts, default=0),
                    None if not qc_match else float(qc_match[0]["confidence"]),
                )
                qc_written += 1
        setattr(make_qc_png, "_last_keypoints", None)
        elapsed = time.perf_counter() - started
        return {
            "status": "excluded_unresolved_target",
            "zip_member": member,
            "subject_id": row.get("subject_id", ""),
            "class_label": row.get("class_label", ""),
            "view": row.get("view_hint", ""),
            "visit": row.get("visit_hint", ""),
            "phase": canonical_phase(row),
            "frames_decoded": len(frames),
            "model_frames_sampled": len(frames),
            "source_frames_decoded": int(video_meta.get("source_frames_decoded", len(frames))),
            "frames_with_pose_candidates": int(sum(n > 0 for n in counts)),
            "max_pose_candidates_in_frame": int(max(counts, default=0)),
            "candidate_tracks": track_summaries,
            "track_overlap_pairs": track_overlap_pairs,
            "reliable_track_ids": reliable_track_ids,
            "moving_reliable_track_ids": moving_reliable_track_ids,
            "selection_mode": selection_mode,
            "selection_reason": selection_reason,
            "exclusion_reason": selection_reason,
            "qc_images_written": qc_written,
            "output_npz": None,
            "inference_seconds": round(inference_seconds, 3),
            "frontend_frames_per_second": round(len(frames) / max(inference_seconds, 1e-8), 3),
            "elapsed_seconds": round(elapsed, 3),
            "video": video_meta,
        }

    # A shared per-track crop scale is computed once; each frame center follows the
    # same tracked pose bbox. Both modalities use precisely the same affine map.
    chosen_boxes = np.stack([d.box for d in chosen.detections])
    widths = chosen_boxes[:, 2] - chosen_boxes[:, 0]
    heights = chosen_boxes[:, 3] - chosen_boxes[:, 1]
    crop_side = float(np.median(np.maximum(widths, heights)) * 1.25)
    det_by_slot = {d.frame_slot: d for d in chosen.detections}
    frame_n = len(frames)
    skeleton_maps = np.zeros((frame_n, 2, MAP_SIZE, MAP_SIZE), dtype=np.float16)
    silhouettes = np.zeros((frame_n, 1, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    pose_valid = np.zeros(frame_n, dtype=np.bool_)
    mask_valid = np.zeros(frame_n, dtype=np.bool_)
    person_count = np.asarray(counts, dtype=np.uint8)
    pose_scores = np.zeros(frame_n, dtype=np.float32)
    mask_scores = np.zeros(frame_n, dtype=np.float32)
    mask_ious = np.zeros(frame_n, dtype=np.float32)
    crop_center_scale = np.zeros((frame_n, 3), dtype=np.float32)
    chosen_pose_kps: dict[int, np.ndarray] = {}
    for slot, frame_index in enumerate(frame_indices):
        det = det_by_slot.get(slot)
        if det is None:
            continue
        center = (float((det.box[0] + det.box[2]) / 2), float((det.box[1] + det.box[3]) / 2))
        matrix = crop_transform(center, crop_side)
        kp = transform_keypoints(det.keypoints, matrix)
        maps = make_pose_maps(kp)
        skeleton_maps[slot] = maps.astype(np.float16)
        pose_valid[slot] = True
        pose_scores[slot] = det.score
        chosen_pose_kps[slot] = kp
        crop_center_scale[slot] = (center[0], center[1], crop_side)
        match = track_mask_matches[chosen.track_id].get(slot)
        mask_item, mask_score, overlap = match if match else (None, 0.0, 0.0)
        mask_scores[slot] = mask_score
        if mask_item is not None:
            sil = warp_binary_mask(mask_item["mask"], matrix)
            occupancy = float(sil.mean())
            if 0.025 <= occupancy <= 0.95:
                silhouettes[slot, 0] = sil * 255
                mask_valid[slot] = True
                mask_ious[slot] = box_iou(det.box, mask_item["box"])
    valid = pose_valid & mask_valid
    valid_frame_count = int(valid.sum())
    selected_track_pose_frames = len(chosen.detections)
    selected_track_valid_pair_frames = int(summary_by_id[chosen.track_id]["valid_pose_mask_frames"])
    selected_track_frame_coverage = selected_track_pose_frames / max(frame_n, 1)
    selected_track_valid_clip_fraction = valid_frame_count / max(frame_n, 1)
    if valid_frame_count < MIN_RELIABLE_VALID_FRAMES:
        status = "selected_track_insufficient_valid_frames"
    else:
        status = "ok"

    qc_slots = np.linspace(0, max(frame_n - 1, 0), min(5, frame_n)).round().astype(int)
    qc_written = 0
    for slot in sorted(set(qc_slots.tolist())):
        if slot not in chosen_pose_kps:
            continue
        # The QC image is derived only from two model outputs; source RGB is never copied.
        setattr(make_qc_png, "_last_keypoints", chosen_pose_kps[slot])
        qc_match = track_mask_matches[chosen.track_id].get(slot)
        make_qc_png(
            qc_root / f"{member_hash}_track{chosen.track_id:03d}_f{frame_indices[slot]:06d}.png",
            frame_indices[slot], skeleton_maps[slot].astype(np.float32),
            silhouettes[slot, 0] if mask_valid[slot] else None,
            chosen.track_id, int(person_count[slot]),
            float(qc_match[0]["confidence"]) if qc_match else None,
        )
        qc_written += 1
    setattr(make_qc_png, "_last_keypoints", None)

    if valid_frame_count < MIN_RELIABLE_VALID_FRAMES:
        if out_path.exists():
            out_path.unlink()
        return {
            "status": status,
            "zip_member": member,
            "subject_id": row.get("subject_id", ""),
            "class_label": row.get("class_label", ""),
            "view": row.get("view_hint", ""),
            "visit": row.get("visit_hint", ""),
            "phase": canonical_phase(row),
            "frames_decoded": frame_n,
            "model_frames_sampled": frame_n,
            "source_frames_decoded": int(video_meta.get("source_frames_decoded", frame_n)),
            "frames_with_pose_candidates": int(sum(n > 0 for n in counts)),
            "max_pose_candidates_in_frame": int(max(counts, default=0)),
            "candidate_tracks": track_summaries,
            "track_overlap_pairs": track_overlap_pairs,
            "selection_mode": selection_mode,
            "selection_reason": selection_reason,
            "track_id": chosen.track_id,
            "pose_valid_frames": int(pose_valid.sum()),
            "mask_valid_frames": int(mask_valid.sum()),
            "valid_frames": valid_frame_count,
            "valid_fraction": valid_frame_count / max(frame_n, 1),
            "exclusion_reason": "fewer_than_30_valid_pose_mask_frames",
            "excluded_reason": "fewer_than_30_valid_pose_mask_frames",
            "qc_images_written": qc_written,
            "output_npz": None,
            "inference_seconds": round(inference_seconds, 3),
            "frontend_frames_per_second": round(len(frames) / max(inference_seconds, 1e-8), 3),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "video": video_meta,
        }

    metadata = {
        "schema": "live-video-input-v1",
        "selection_rule_version": SELECTION_RULE_VERSION,
        "video_id": row.get("video_id", member_hash),
        "subject_id": row.get("subject_id", ""),
        "class_label": row.get("class_label", ""),
        "zip_member": member,
        "video_kind": row.get("video_kind", ""),
        "view": row.get("view_hint", ""),
        "visit": row.get("visit_hint", ""),
        "phase": canonical_phase(row),
        "selection_note": row.get("selection_note", ""),
        "track_id": chosen.track_id,
        "requested_track_id": requested_track_id(row),
        "selection_mode": selection_mode,
        "selection_reason": selection_reason,
        "track_overlap_pairs": track_overlap_pairs,
        "reliability_qc": {
            "minimum_valid_pose_mask_frames": MIN_RELIABLE_VALID_FRAMES,
            "minimum_mean_pose_person_confidence": MIN_RELIABLE_MEAN_POSE_CONFIDENCE,
            "minimum_mask_confidence": 0.20,
            "minimum_pose_mask_box_iou": 0.18,
            "minimum_visible_coco17_joints": MIN_VISIBLE_JOINTS,
            "minimum_joint_confidence": POSE_CONF_MIN,
            "significant_track_overlap_min_shared_frames": SIGNIFICANT_TRACK_OVERLAP_MIN_FRAMES,
            "significant_track_overlap_min_shorter_track_fraction": SIGNIFICANT_TRACK_OVERLAP_MIN_RATIO,
        },
        "selected_track_pose_frames": selected_track_pose_frames,
        "selected_track_valid_pair_frames": selected_track_valid_pair_frames,
        "selected_track_frame_coverage": selected_track_frame_coverage,
        "selected_track_valid_clip_fraction": selected_track_valid_clip_fraction,
        "selected_track_is_partial": selection_mode == "longest_nonoverlapping_reliable_track" or selected_track_frame_coverage < 0.999,
        "video": video_meta,
        "pose_acceptance": {
            "model_person_confidence_floor": 0.01,
            "minimum_visible_coco17_joints": MIN_VISIBLE_JOINTS,
            "joint_confidence_floor": POSE_CONF_MIN,
        },
        "frame_count_saved": frame_n,
        "is_smoke": is_smoke,
        "max_frames": max_frames,
        "target_fps_requested": target_fps or None,
        "sampling_mode": (
            (f"uniform_{target_fps:g}fps" if target_fps else "all_frames")
            if not is_smoke
            else (f"uniform_{target_fps:g}fps_capped_{max_frames}_smoke" if target_fps else f"uniform_{max_frames}_frames_smoke")
        ),
        "frame_indices": frame_indices,
        "pose_frames": int(pose_valid.sum()),
        "mask_frames": int(mask_valid.sum()),
        "valid_frames": int(valid.sum()),
        "pose_candidate_count_per_frame": counts,
        "track_candidates": track_summaries,
        "crop_side_pixels": crop_side,
        "coordinate_transform": "shared square crop centered on tracked pose bbox center; track-median person extent x1.25; same affine used for 64x64 pose maps and silhouette",
        "pose_map_channel_order": ["limb", "joint"],
        "pose_map_sigma_64px": GAUSSIAN_SIGMA,
        "silhouette_uint8_range": [0, 255],
        "skeleton_map_float_range": [0.0, 1.0],
        "qc_images_written": qc_written,
        "source_rgb_saved": False,
        "inference_precision": "cuda_fp16",
        "inference_batch_size": batch_size,
        "inference_seconds": round(inference_seconds, 3),
        "frontend_frames_per_second": round(frame_n / max(inference_seconds, 1e-8), 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temp_out = out_path.with_suffix(".npz.part")
    with temp_out.open("wb") as f:
        np.savez_compressed(
            f,
            schema=np.asarray("live-video-input-v1"),
            selection_rule_version=np.asarray(SELECTION_RULE_VERSION),
            video_id=np.asarray(row.get("video_id", member_hash)),
            skeleton_maps=skeleton_maps,
            silhouette=silhouettes,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            valid_frame_mask=valid,
            pose_valid_mask=pose_valid,
            mask_valid_mask=mask_valid,
            subject_id=np.asarray(row.get("subject_id", "")),
            class_label=np.asarray(row.get("class_label", "")),
            zip_member=np.asarray(member),
            view=np.asarray(row.get("view_hint", "")),
            visit=np.asarray(row.get("visit_hint", "")),
            phase=np.asarray(canonical_phase(row)),
            track_id=np.asarray(chosen.track_id, dtype=np.int16),
            selection_mode=np.asarray(selection_mode),
            selection_reason=np.asarray(selection_reason),
            selected_track_pose_frames=np.asarray(selected_track_pose_frames, dtype=np.int32),
            selected_track_valid_pair_frames=np.asarray(selected_track_valid_pair_frames, dtype=np.int32),
            selected_track_frame_coverage=np.asarray(selected_track_frame_coverage, dtype=np.float32),
            selected_track_valid_clip_fraction=np.asarray(selected_track_valid_clip_fraction, dtype=np.float32),
            selected_track_is_partial=np.asarray(
                selection_mode == "longest_nonoverlapping_reliable_track" or selected_track_frame_coverage < 0.999,
                dtype=np.bool_,
            ),
            person_count=person_count,
            pose_confidence=pose_scores,
            mask_match_score=mask_scores,
            mask_box_iou=mask_ious,
            crop_center_scale=crop_center_scale,
            source_fps=np.asarray(video_meta.get("fps") or video_meta.get("opencv_fps") or 0.0, dtype=np.float32),
            source_frame_count=np.asarray(video_meta.get("opencv_frame_count") or 0, dtype=np.int32),
            rotation_degrees=np.asarray(video_meta.get("rotation_degrees") or 0.0, dtype=np.float32),
            is_smoke=np.asarray(is_smoke, dtype=np.bool_),
            max_frames=np.asarray(max_frames, dtype=np.int32),
            target_fps=np.asarray(target_fps, dtype=np.float32),
            sampling_mode=np.asarray(metadata["sampling_mode"]),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    os.replace(temp_out, out_path)
    return {
        "status": status,
        "zip_member": member,
        "subject_id": row.get("subject_id", ""),
        "class_label": row.get("class_label", ""),
        "frames_decoded": frame_n,
        "model_frames_sampled": frame_n,
        "source_frames_decoded": int(video_meta.get("source_frames_decoded", frame_n)),
        "frames_with_pose_candidates": int(sum(n > 0 for n in counts)),
        "max_pose_candidates_in_frame": int(max(counts, default=0)),
        "candidate_tracks": track_summaries,
        "track_overlap_pairs": track_overlap_pairs,
        "selection_mode": selection_mode,
        "selection_reason": selection_reason,
        "track_id": chosen.track_id,
        "selected_track_pose_frames": selected_track_pose_frames,
        "selected_track_valid_pair_frames": selected_track_valid_pair_frames,
        "selected_track_frame_coverage": selected_track_frame_coverage,
        "selected_track_valid_clip_fraction": selected_track_valid_clip_fraction,
        "selected_track_is_partial": selection_mode == "longest_nonoverlapping_reliable_track" or selected_track_frame_coverage < 0.999,
        "pose_valid_frames": int(pose_valid.sum()),
        "mask_valid_frames": int(mask_valid.sum()),
        "valid_frames": int(valid.sum()),
        "pose_acceptance": {
            "model_person_confidence_floor": 0.01,
            "minimum_visible_coco17_joints": MIN_VISIBLE_JOINTS,
            "joint_confidence_floor": POSE_CONF_MIN,
        },
        "valid_fraction": float(valid.mean()),
        "qc_images_written": qc_written,
        "inference_seconds": round(inference_seconds, 3),
        "frontend_frames_per_second": round(frame_n / max(inference_seconds, 1e-8), 3),
        "output_npz": str(out_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "video": video_meta,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eligible-manifest", type=Path, default=EXPERIMENT_ROOT / "manifests" / "eligible_videos_final.csv", help="QC-passed final accepted-only CSV whose raw-byte SHA256 is stored in the full-input completion marker.")
    parser.add_argument("--completion-marker", type=Path, default=EXPERIMENT_ROOT / "audit" / "agent4_full_preprocessing_complete.json", help="Written complete=true only after every eligible clip has a verified full-input NPZ.")
    parser.add_argument("--input-zip", type=Path, default=DEFAULT_INPUT_ZIP)
    parser.add_argument("--selection-csv", type=Path)
    parser.add_argument("--members", nargs="+", help="Exact zip_member values from videos.csv.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--pose-weights", type=Path, default=DEFAULT_POSE_WEIGHTS)
    parser.add_argument("--seg-weights", type=Path, default=DEFAULT_SEG_WEIGHTS)
    parser.add_argument("--max-frames", type=int, default=16, help="Use N frames evenly spread across each clip; 0 means every frame and requires --eligible-selection-finalized.")
    parser.add_argument("--target-fps", type=float, default=0.0, help="Uniformly sample the whole clip at this frame rate (e.g. 15); 0 keeps all frames unless --max-frames caps a smoke run.")
    parser.add_argument("--batch-size", type=int, default=8, help="CUDA inference batch size for pose and segmentation models.")
    parser.add_argument("--eligible-selection-finalized", action="store_true", help="Required for max-frames=0 after root freezes this processing pass selection (candidate pool or final accepted set).")
    parser.add_argument("--resume-inputs", action="store_true", help="In full mode, skip only NPZs whose embedded metadata and full frame sequence match this exact selection and protocol; invalid or pilot files are recomputed.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--decisions-csv", type=Path, help="Optional atomically updated one-row-per-processed-clip QC decision ledger; full mode only.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be zero or positive.")
    if args.target_fps < 0:
        raise SystemExit("--target-fps must be zero or positive.")
    if args.batch_size < 1 or args.batch_size > 16:
        raise SystemExit("--batch-size must be between 1 and 16 to bound RTX 5080 memory.")
    if args.max_frames == 0 and (not args.selection_csv or not args.eligible_selection_finalized):
        raise SystemExit("Full-video mode requires --selection-csv plus --eligible-selection-finalized after the processing selection is frozen.")
    if args.resume_inputs and args.max_frames != 0:
        raise SystemExit("--resume-inputs is available only in full mode (--max-frames 0).")
    if args.resume_inputs and args.overwrite:
        raise SystemExit("Use either --resume-inputs or --overwrite, not both; resume selectively overwrites only invalid full-output caches.")
    if args.decisions_csv and args.max_frames != 0:
        raise SystemExit("--decisions-csv is a full-input QC ledger and requires --max-frames 0.")
    if not args.input_zip.is_file():
        raise SystemExit(f"Input archive not found: {args.input_zip}")
    if not args.pose_weights.is_file() or not args.seg_weights.is_file():
        raise SystemExit("Both existing local model checkpoints must be available.")
    rows = select_rows(args.manifest, args.members, args.selection_csv)
    if args.max_frames > 0 and len(rows) > 2:
        raise SystemExit("Pre-finalization smoke mode is limited to at most two clips.")
    if args.max_frames == 0:
        required_final_fields = ("video_id", "npz_path", "subject_id", "class_label", "view_hint", "visit_hint")
        missing = [
            row.get("zip_member", "") for row in rows
            if any(not row.get(field) for field in required_final_fields)
        ]
        if missing:
            raise SystemExit(
                "Full mode requires Agent 1's finalized eligible CSV with video_id, npz_path, canonical subject_id, class, view, and visit; "
                f"incomplete rows include: {missing[:3]}"
            )
        if not args.eligible_manifest.is_file():
            raise SystemExit(f"Final eligible manifest not found: {args.eligible_manifest}")
    eligible_rows = load_csv(args.eligible_manifest) if args.eligible_manifest.is_file() else []
    eligible_members = [r.get("zip_member", "") for r in eligible_rows]
    eligible_manifest_sha256 = hashlib.sha256(args.eligible_manifest.read_bytes()).hexdigest() if args.eligible_manifest.is_file() else None
    selected_members = [r.get("zip_member", "") for r in rows]
    selection_covers_eligible_manifest = (
        bool(eligible_members)
        and len(selected_members) == len(eligible_members)
        and set(selected_members) == set(eligible_members)
    )

    raw_root = args.cache_root / "raw_video"
    output_root = args.cache_root / "video_inputs"
    qc_root = EXPERIMENT_ROOT / "audit" / "preprocessing_qc" / (
        "full" if args.max_frames == 0 else f"sample{args.max_frames}"
    )
    qc_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this preprocessing check.")
    if args.device not in {"0", "cuda:0"}:
        raise SystemExit("This runbook is configured for the verified CUDA:0 runtime.")
    device_name = torch.cuda.get_device_name(0)
    pose_model = YOLO(str(args.pose_weights))
    seg_model = YOLO(str(args.seg_weights))

    report: dict[str, Any] = {
        "schema": "live-video-preprocessing-audit-v1",
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": device_name,
            "opencv": cv2.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "pose_weights": str(args.pose_weights),
            "seg_weights": str(args.seg_weights),
        },
        "archive": str(args.input_zip),
        "manifest": str(args.manifest),
        "eligible_manifest": str(args.eligible_manifest),
        "eligible_manifest_role": "final_accepted_only" if args.eligible_manifest.name == "eligible_videos_final.csv" else "candidate_or_explicit_selection_closure",
        "eligible_manifest_sha256": eligible_manifest_sha256,
        "eligible_video_count": len(eligible_rows),
        "selection_manifest_sha256": hashlib.sha256(args.selection_csv.read_bytes()).hexdigest() if args.selection_csv and args.selection_csv.is_file() else None,
        "selection_covers_entire_eligible_manifest": selection_covers_eligible_manifest,
        "completion_marker": str(args.completion_marker),
        "max_frames": args.max_frames,
        "target_fps": args.target_fps or None,
        "batch_size": args.batch_size,
        "inference_precision": "cuda_fp16",
        "resume_inputs": args.resume_inputs,
        "selection_count": len(rows),
        "decisions_csv": str(args.decisions_csv) if args.decisions_csv else None,
        "policy": {
            "exclude_sidecars": True,
            "exclude_marking_and_calibration": True,
            "target_selection": "reliable means >=30 effective same-person pose-mask frames after mask and crop QC plus mean pose-person confidence >=0.25; one reliable track is accepted regardless of net displacement; for multiple reliable moving tracks, choose the track with most effective frames only when no pair reaches both significant-overlap thresholds (>=3 shared detection frames AND >=20% of the shorter track); significant overlap stays excluded until explicit review",
            "paired_modalities": "same decoded frame, same track center, same track-wide scale, same affine crop",
            "segmentation_qc": "person-only candidates below the training mask confidence threshold may appear in review PNGs; frame validity requires mask confidence >=0.20, IoU >=0.18, 2D pose-mask center offset <=0.75 median body heights, cross-frame pose-mask motion mismatch <=0.75 median body heights, and silhouette occupancy between 2.5% and 95%",
            "pose_acceptance": "pose detector floor 0.01; each candidate must have at least 8 of 17 COCO joints at confidence >=0.15; a frame is training-valid only when its paired mask also passes",
            "frame_validity": "pose_valid_mask AND mask_valid_mask; downstream two-stream fitting must use valid_frame_mask",
            "source_rgb_saved": False,
            "full_clip_gate": "--max-frames 0 requires --selection-csv and --eligible-selection-finalized for this fixed pass",
            "smoke_outputs": "separate <member-sha256>_sampleN.npz files with is_smoke=true and max_frames=N; these are not eligible training inputs",
            "full_outputs": "use the selected manifest npz_path (<member-sha256>.npz); --resume-inputs reprocesses any same-hash cache that fails the current schema, track rule, QC, or full-frame protocol",
        },
        "clips": [],
    }
    report_path = args.report or (EXPERIMENT_ROOT / "audit" / "agent4_preprocessing_gpu_check.json")
    report["pending_clips"] = [r.get("video_id", r.get("zip_member", "")) for r in rows]
    report["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    persist_progress(report_path, report)
    if args.decisions_csv:
        write_decisions_csv(args.decisions_csv, report)
    if args.max_frames == 0:
        # A partial or interrupted run must never leave an earlier success marker trusted.
        persist_progress(args.completion_marker, {
            "schema": "live-video-full-input-complete-v1",
            "complete": False,
            "max_frames": 0,
            "eligible_selection_finalized": bool(args.eligible_selection_finalized),
            "eligible_manifest_sha256": eligible_manifest_sha256,
            "eligible_video_count": len(eligible_rows),
            "processed_video_count": 0,
            "target_fps": args.target_fps or None,
            "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "full selection is partial or processing has not finished",
        })
    for clip_index, row in enumerate(rows):
        report["current_video_id"] = row.get("video_id", "")
        report["current_zip_member"] = row.get("zip_member", "")
        report["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        persist_progress(report_path, report)
        if args.resume_inputs:
            reusable, reason, cache_details = validate_resume_cache(row, output_root, args.target_fps)
            if reusable:
                item = {
                    "status": "skipped_resume_verified",
                    "zip_member": row["zip_member"],
                    **cache_details,
                }
                report["clips"].append(item)
                report["pending_clips"] = [r.get("video_id", r.get("zip_member", "")) for r in rows[clip_index + 1 :]]
                report["current_video_id"] = None
                report["current_zip_member"] = None
                report["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
                refresh_candidate_summary(report)
                persist_progress(report_path, report)
                if args.decisions_csv:
                    write_decisions_csv(args.decisions_csv, report)
                print_clip_summary(item)
                continue
        else:
            reason = "resume_disabled"
        torch.cuda.reset_peak_memory_stats(0)
        clip_started = time.perf_counter()
        try:
            item = process_one(
                row, args.input_zip, raw_root, output_root, qc_root,
                pose_model, seg_model, args.max_frames, args.target_fps,
                args.imgsz, args.batch_size, args.overwrite or args.resume_inputs,
            )
            item["video_id"] = row.get("video_id", "")
            item["view"] = row.get("view_hint", "")
            item["visit"] = row.get("visit_hint", "")
            item["phase"] = canonical_phase(row)
            item["is_smoke"] = args.max_frames > 0
            item["max_frames"] = args.max_frames
            item["target_fps"] = args.target_fps or None
            item["resume_cache_validation"] = reason
            item["gpu_peak_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)
            item["gpu_peak_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 1)
        except Exception as exc:
            item = {
                "status": "error",
                "zip_member": row.get("zip_member", ""),
                "subject_id": row.get("subject_id", ""),
                "class_label": row.get("class_label", ""),
                "view": row.get("view_hint", ""),
                "visit": row.get("visit_hint", ""),
                "phase": canonical_phase(row),
                "video_id": row.get("video_id", ""),
                "frames_decoded": 0,
                "model_frames_sampled": 0,
                "elapsed_seconds": round(time.perf_counter() - clip_started, 3),
                "is_smoke": args.max_frames > 0,
                "max_frames": args.max_frames,
                "target_fps": args.target_fps or None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resume_cache_validation": reason,
                "gpu_peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1),
                "gpu_peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 1),
            }
        report["clips"].append(item)
        report["pending_clips"] = [r.get("video_id", r.get("zip_member", "")) for r in rows[clip_index + 1 :]]
        report["current_video_id"] = None
        report["current_zip_member"] = None
        report["updated_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        refresh_candidate_summary(report)
        persist_progress(report_path, report)
        if args.decisions_csv:
            write_decisions_csv(args.decisions_csv, report)
        print_clip_summary(item)
    report["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["pending_clips"] = []
    report["current_video_id"] = None
    report["current_zip_member"] = None
    report["updated_local"] = report["finished_local"]
    persist_progress(report_path, report)
    if args.max_frames == 0:
        processed_video_count = sum(is_accepted_full_input(item) for item in report["clips"])
        complete = bool(
            args.eligible_selection_finalized
            and selection_covers_eligible_manifest
            and processed_video_count == len(eligible_rows)
            and len(report["clips"]) == len(eligible_rows)
            and all(is_accepted_full_input(item) for item in report["clips"])
            and hashlib.sha256(args.eligible_manifest.read_bytes()).hexdigest() == eligible_manifest_sha256
        )
        persist_progress(args.completion_marker, {
            "schema": "live-video-full-input-complete-v1",
            "complete": complete,
            "max_frames": 0,
            "eligible_selection_finalized": bool(args.eligible_selection_finalized),
            "eligible_manifest_sha256": eligible_manifest_sha256,
            "eligible_video_count": len(eligible_rows),
            "processed_video_count": processed_video_count,
            "target_fps": args.target_fps or None,
            "finished_local": report["finished_local"],
            "reason": "all final eligible clips have verified full-input NPZs" if complete else "selection or processing is incomplete; do not use as the full-input training marker",
        })
    print(f"Report written: {report_path}")
    return 0 if all(x.get("status") in {"ok", "excluded_unresolved_target", "selected_track_insufficient_valid_frames", "skipped_resume_verified"} for x in report["clips"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
