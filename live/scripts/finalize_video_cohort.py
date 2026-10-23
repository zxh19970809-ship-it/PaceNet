#!/usr/bin/env python
"""Freeze the accepted-only video cohort after candidate preprocessing QC.

This CPU-only finalizer does not run models, modify video caches, infer participant
codes, or create the Agent 4 completion marker. It verifies the completed
candidate ledger against the candidate manifest, inspects accepted NPZ files,
applies the runner's exact 720-slot temporal cap, and writes an accepted-only
manifest plus a complete decision/cohort audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "live-video-input-v1"
SELECTION_RULE_VERSION = "live-gait-track-qc-v2"
TARGET_FPS = 15.0
RUNNER_TEMPORAL_CAP = 720
MIN_VALID_AFTER_CAP = 30
EXPECTED_CLASSES = ("CP", "ITW", "TD")
EXPECTED_VIEWS = ("left", "right", "sagittal")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def canonical_fields(row: dict[str, str]) -> dict[str, str]:
    phase = first_value(row, ("phase", "visit_hint", "visit"))
    return {
        "video_id": first_value(row, ("video_id",)),
        "subject_id": first_value(row, ("subject_id",)),
        "class_label": first_value(row, ("class_label", "label")).upper(),
        "zip_member": first_value(row, ("zip_member",)),
        "view": first_value(row, ("view_hint", "view")).lower(),
        "visit": first_value(row, ("visit_hint", "visit", "phase")),
        "phase": phase,
    }


def scalar(data: Any, key: str) -> Any:
    if key not in data.files:
        raise ValueError(f"missing NPZ key {key}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"NPZ field {key} must be a scalar, got {value.shape}")
    return value.item()


def runner_cap_slots(frame_count: int) -> np.ndarray:
    """Mirror run_cv.cap_temporal_slots exactly."""
    if frame_count <= RUNNER_TEMPORAL_CAP:
        return np.arange(frame_count, dtype=np.int64)
    slots = np.linspace(0, frame_count - 1, num=RUNNER_TEMPORAL_CAP).round().astype(np.int64)
    if len(np.unique(slots)) != RUNNER_TEMPORAL_CAP:
        raise ValueError(f"runner could not form {RUNNER_TEMPORAL_CAP} unique temporal slots")
    return slots


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def validate_npz(row: dict[str, str], npz_path: Path) -> dict[str, Any]:
    """Validate the complete full-mode NPZ and return post-cap QC facts."""
    expected = canonical_fields(row)
    required = {
        "schema", "selection_rule_version", "video_id", "skeleton_maps", "silhouette",
        "frame_indices", "valid_frame_mask", "pose_valid_mask", "mask_valid_mask",
        "subject_id", "class_label", "view", "visit", "phase", "zip_member",
        "is_smoke", "max_frames", "target_fps", "sampling_mode", "source_fps",
        "source_frame_count", "metadata_json", "track_id", "selection_mode",
    }
    with np.load(npz_path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing required NPZ keys: {missing}")
        if str(scalar(data, "schema")) != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA}")
        if str(scalar(data, "selection_rule_version")) != SELECTION_RULE_VERSION:
            raise ValueError(f"selection_rule_version must be {SELECTION_RULE_VERSION}")
        if bool(scalar(data, "is_smoke")) or int(scalar(data, "max_frames")) != 0:
            raise ValueError("NPZ is a smoke or capped artifact")
        target_fps = float(scalar(data, "target_fps"))
        if not math.isclose(target_fps, TARGET_FPS, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"target_fps must equal {TARGET_FPS:g}")
        expected_sampling = f"uniform_{TARGET_FPS:g}fps"
        if str(scalar(data, "sampling_mode")) != expected_sampling:
            raise ValueError(f"sampling_mode must be {expected_sampling}")

        metadata_raw = scalar(data, "metadata_json")
        metadata = json.loads(str(metadata_raw)) if metadata_raw else {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must contain a JSON object")
        if metadata.get("is_smoke") is not False or int(metadata.get("max_frames", -1)) != 0:
            raise ValueError("metadata_json does not certify a full non-smoke NPZ")
        if metadata.get("sampling_mode") != expected_sampling:
            raise ValueError(f"metadata sampling_mode must be {expected_sampling}")
        if metadata.get("selection_rule_version") != SELECTION_RULE_VERSION:
            raise ValueError(f"metadata selection_rule_version must be {SELECTION_RULE_VERSION}")
        if not math.isclose(float(metadata.get("target_fps_requested") or 0.0), TARGET_FPS, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("metadata target_fps_requested disagrees with full-mode protocol")

        for key, wanted in expected.items():
            if not wanted:
                continue
            observed = str(scalar(data, key))
            meta_observed = str(metadata.get(key, ""))
            if key == "class_label":
                observed, wanted, meta_observed = observed.upper(), wanted.upper(), meta_observed.upper()
            elif key == "view":
                observed, wanted, meta_observed = observed.lower(), wanted.lower(), meta_observed.lower()
            if observed != wanted:
                raise ValueError(f"NPZ scalar {key}={observed!r} disagrees with candidate {wanted!r}")
            if meta_observed != wanted:
                raise ValueError(f"metadata {key}={meta_observed!r} disagrees with candidate {wanted!r}")

        skeleton = np.asarray(data["skeleton_maps"])
        silhouette = np.asarray(data["silhouette"])
        frame_indices = np.asarray(data["frame_indices"])
        valid = np.asarray(data["valid_frame_mask"])
        pose_valid = np.asarray(data["pose_valid_mask"])
        mask_valid = np.asarray(data["mask_valid_mask"])
        if skeleton.ndim != 4 or skeleton.shape[1:] != (2, 64, 64) or skeleton.dtype != np.float16:
            raise ValueError(f"skeleton_maps must be float16 [T,2,64,64], got {skeleton.dtype} {skeleton.shape}")
        frame_count = int(skeleton.shape[0])
        if frame_count <= 0:
            raise ValueError("NPZ contains no sampled frames")
        if silhouette.shape != (frame_count, 1, 64, 64) or silhouette.dtype != np.uint8:
            raise ValueError(f"silhouette must be uint8 [T,1,64,64], got {silhouette.dtype} {silhouette.shape}")
        if not np.isfinite(skeleton).all() or skeleton.min(initial=0) < 0 or skeleton.max(initial=0) > 1:
            raise ValueError("skeleton_maps contains nonfinite or out-of-range values")
        if silhouette.size and not np.isin(np.unique(silhouette), (0, 255)).all():
            raise ValueError("silhouette must contain only 0 and 255")
        if any(mask.dtype != np.bool_ or mask.shape != (frame_count,) for mask in (valid, pose_valid, mask_valid)):
            raise ValueError("validity masks must be bool [T]")
        if not np.array_equal(valid, pose_valid & mask_valid):
            raise ValueError("valid_frame_mask is not pose_valid_mask AND mask_valid_mask")
        if frame_indices.dtype != np.int32 or frame_indices.shape != (frame_count,):
            raise ValueError("frame_indices must be int32 [T]")
        if np.any(np.diff(frame_indices.astype(np.int64)) <= 0):
            raise ValueError("frame_indices must be strictly increasing")

        source_count = int(scalar(data, "source_frame_count"))
        stored_source_fps = float(scalar(data, "source_fps"))
        video_meta = metadata.get("video", {})
        if not isinstance(video_meta, dict):
            raise ValueError("metadata_json.video must be an object")
        # The extractor samples with the original Python-precision rate from
        # FFprobe/OpenCV, then stores source_fps as float32. Replaying from that
        # rounded scalar can move a small number of half-frame boundaries by
        # one source frame, so use the exact sampling-rate field recorded in
        # metadata and verify its float32 projection against the NPZ scalar.
        sampling_fps = float(video_meta.get("fps") or video_meta.get("opencv_fps") or 0.0)
        metadata_source_count = int(video_meta.get("opencv_frame_count") or 0)
        if source_count <= 0 or stored_source_fps <= 0 or sampling_fps <= 0 or int(frame_indices[-1]) >= source_count:
            raise ValueError("source frame metadata is invalid")
        if metadata_source_count and metadata_source_count != source_count:
            raise ValueError("metadata video.opencv_frame_count disagrees with source_frame_count")
        if not math.isclose(stored_source_fps, float(np.float32(sampling_fps)), rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("metadata video fps does not match the float32 source_fps scalar")
        expected_indices = np.arange(0.0, source_count / sampling_fps, 1.0 / TARGET_FPS, dtype=np.float64)
        expected_indices = np.unique(np.rint(expected_indices * sampling_fps).astype(np.int64))
        expected_indices = expected_indices[(expected_indices >= 0) & (expected_indices < source_count)]
        if not np.array_equal(frame_indices, expected_indices.astype(np.int32)):
            raise ValueError("frame_indices do not represent the complete uniform-15fps source sequence")
        # Keep a diagnostic for older consumers that recompute indices from
        # the float32 scalar instead of metadata_json.video.fps. This is not a
        # sample exclusion criterion because it is serialization precision,
        # not a missing or reordered sampled frame.
        stored_fps_indices = np.arange(0.0, source_count / stored_source_fps, 1.0 / TARGET_FPS, dtype=np.float64)
        stored_fps_indices = np.unique(np.rint(stored_fps_indices * stored_source_fps).astype(np.int64))
        stored_fps_indices = stored_fps_indices[(stored_fps_indices >= 0) & (stored_fps_indices < source_count)]
        runner_float32_sequence_mismatch_count = (
            int(np.count_nonzero(frame_indices != stored_fps_indices.astype(np.int32)))
            if frame_indices.shape == stored_fps_indices.shape else abs(int(frame_indices.size) - int(stored_fps_indices.size))
        )

        slots = runner_cap_slots(frame_count)
        capped_valid_count = int(valid[slots].sum())
        if capped_valid_count < MIN_VALID_AFTER_CAP:
            raise ValueError(
                f"only {capped_valid_count} valid pose-mask frames remain after runner's {RUNNER_TEMPORAL_CAP}-slot cap"
            )
        return {
            "full_frame_count": frame_count,
            "full_valid_frame_count": int(valid.sum()),
            "runner_cap_frame_count": int(len(slots)),
            "valid_frames_after_720": capped_valid_count,
            "source_frame_count": source_count,
            "source_fps": stored_source_fps,
            "sampling_fps_metadata_precision": sampling_fps,
            "runner_float32_sequence_mismatch_count": runner_float32_sequence_mismatch_count,
            "track_id": int(scalar(data, "track_id")),
            "selection_mode": str(scalar(data, "selection_mode")),
            "selected_track_frame_coverage": float(scalar(data, "selected_track_frame_coverage")),
            "selected_track_valid_clip_fraction": float(scalar(data, "selected_track_valid_clip_fraction")),
        }


def assert_candidate_run_complete(
    candidate_rows: list[dict[str, str]],
    decisions: list[dict[str, str]],
    report: dict[str, Any],
    candidate_manifest: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    if not report.get("finished_local"):
        raise ValueError("candidate preprocessing report has no finished_local timestamp; wait for the candidate run to finish")
    if report.get("pending_clips"):
        raise ValueError(f"candidate report still has pending clips: {len(report['pending_clips'])}")
    expected_hash = sha256_file(candidate_manifest)
    if report.get("eligible_manifest_sha256") != expected_hash:
        raise ValueError("candidate report eligible_manifest_sha256 does not match the current candidate CSV")
    if int(report.get("selection_count", -1)) != len(candidate_rows):
        raise ValueError("candidate report selection_count does not match candidate CSV row count")
    if len(report.get("clips", [])) != len(candidate_rows):
        raise ValueError("candidate report must contain exactly one terminal clip record per candidate")

    def unique_by_member(items: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for item in items:
            member = str(item.get("zip_member", "")).strip()
            if not member or member in out:
                raise ValueError(f"{label} contains a missing or duplicate zip_member: {member!r}")
            out[member] = item
        return out

    candidate_by_member = unique_by_member(candidate_rows, "candidate manifest")
    decision_by_member = unique_by_member(decisions, "candidate decisions CSV")
    report_by_member = unique_by_member(report.get("clips", []), "candidate progress report")
    candidate_members = set(candidate_by_member)
    if set(decision_by_member) != candidate_members:
        missing = sorted(candidate_members - set(decision_by_member))[:5]
        extra = sorted(set(decision_by_member) - candidate_members)[:5]
        raise ValueError(f"decision ledger membership differs from candidate CSV (missing={missing}, extra={extra})")
    if set(report_by_member) != candidate_members:
        missing = sorted(candidate_members - set(report_by_member))[:5]
        extra = sorted(set(report_by_member) - candidate_members)[:5]
        raise ValueError(f"candidate progress membership differs from candidate CSV (missing={missing}, extra={extra})")
    for member, decision in decision_by_member.items():
        if str(decision.get("decision", "")).strip().lower() not in {"accepted", "excluded"}:
            raise ValueError(f"candidate decision is not terminal for {member}: {decision.get('decision')!r}")
        report_status = str(report_by_member[member].get("status", ""))
        if report_status != str(decision.get("status", "")):
            raise ValueError(f"decision/progress status mismatch for {member}")
    return decision_by_member, report_by_member


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temp, path)


def summarize_cohort(
    candidate_rows: list[dict[str, str]],
    decision_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    retention_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    subject: dict[str, dict[str, Any]] = {}
    accepted_subject_classes: dict[str, str] = {}
    exclusion_reasons: Counter[str] = Counter()
    accepted_subjects_by_class: Counter[str] = Counter()
    accepted_clips_by_class: Counter[str] = Counter()
    for candidate, decision in zip(candidate_rows, decision_rows):
        fields = canonical_fields(candidate)
        klass, view = fields["class_label"], fields["view"]
        counts = retention_counts[(klass, view)]
        counts["candidates"] += 1
        sid = fields["subject_id"]
        group = subject.setdefault(sid, {
            "subject_id": sid,
            "class_label": klass,
            "candidate_clips": 0,
            "accepted_clips": 0,
            "excluded_clips": 0,
            "candidate_views": set(),
            "accepted_views": set(),
            "candidate_left_clips": 0,
            "candidate_right_clips": 0,
            "candidate_sagittal_clips": 0,
            "accepted_left_clips": 0,
            "accepted_right_clips": 0,
            "accepted_sagittal_clips": 0,
            "visits_accepted": set(),
        })
        if group["class_label"] != klass:
            raise ValueError(f"canonical subject {sid} has multiple class labels")
        group["candidate_clips"] += 1
        if view in EXPECTED_VIEWS:
            group["candidate_views"].add(view)
            group[f"candidate_{view}_clips"] += 1
        if decision["decision"] == "accepted":
            counts["accepted"] += 1
            group["accepted_clips"] += 1
            if view in EXPECTED_VIEWS:
                group["accepted_views"].add(view)
                group[f"accepted_{view}_clips"] += 1
            group["visits_accepted"].add(fields["visit"] or fields["phase"])
            prior = accepted_subject_classes.setdefault(sid, klass)
            if prior != klass:
                raise ValueError(f"canonical subject {sid} has inconsistent accepted class")
            accepted_clips_by_class[klass] += 1
        else:
            counts["excluded"] += 1
            group["excluded_clips"] += 1
            reason = str(decision.get("decision_reason", "")).strip() or "excluded_without_reason"
            exclusion_reasons[reason] += 1

    subject_rows: list[dict[str, Any]] = []
    for sid in sorted(subject):
        item = subject[sid]
        views = item.pop("accepted_views")
        candidate_views = item.pop("candidate_views")
        visits = item.pop("visits_accepted")
        missing_views = [v for v in EXPECTED_VIEWS if v not in views]
        item.update({
            "candidate_views": ";".join(v for v in EXPECTED_VIEWS if v in candidate_views),
            "accepted_views": ";".join(v for v in EXPECTED_VIEWS if v in views),
            "missing_accepted_views": ";".join(missing_views),
            "accepted_view_coverage_count": len(views),
            "accepted_view_coverage_fraction": round(len(views) / len(EXPECTED_VIEWS), 6),
            "accepted_visits": ";".join(sorted(v for v in visits if v)),
        })
        if item["accepted_clips"] > 0:
            accepted_subjects_by_class[item["class_label"]] += 1
        subject_rows.append(item)

    retention_rows = []
    for klass in EXPECTED_CLASSES:
        for view in EXPECTED_VIEWS:
            counts = retention_counts[(klass, view)]
            candidate_n = counts["candidates"]
            accepted_n = counts["accepted"]
            excluded_n = counts["excluded"]
            retention_rows.append({
                "class_label": klass,
                "view": view,
                "candidate_clips": candidate_n,
                "accepted_clips": accepted_n,
                "excluded_clips": excluded_n,
                "retention_fraction": round(accepted_n / candidate_n, 6) if candidate_n else None,
            })

    counts = {
        "candidate_clips": len(candidate_rows),
        "accepted_clips": sum(1 for row in decision_rows if row["decision"] == "accepted"),
        "excluded_clips": sum(1 for row in decision_rows if row["decision"] == "excluded"),
        "candidate_subjects": len(subject),
        "accepted_subjects": sum(1 for row in subject_rows if row["accepted_clips"] > 0),
        "candidate_subjects_by_class": dict(sorted(Counter(row["class_label"] for row in subject_rows).items())),
        "accepted_subjects_by_class": dict(sorted(accepted_subjects_by_class.items())),
        "accepted_clips_by_class": dict(sorted(accepted_clips_by_class.items())),
        "exclusions_by_reason": dict(sorted(exclusion_reasons.items())),
    }
    return retention_rows, subject_rows, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, default=ROOT / "manifests" / "eligible_videos.csv")
    parser.add_argument("--candidate-decisions", type=Path, default=ROOT / "audit" / "agent4_candidate_decisions.csv")
    parser.add_argument("--candidate-report", type=Path, default=ROOT / "audit" / "agent4_candidate_preprocessing_progress.json")
    parser.add_argument("--final-csv", type=Path, default=ROOT / "manifests" / "eligible_videos_final.csv")
    parser.add_argument("--decisions-csv", type=Path, default=ROOT / "audit" / "agent4_final_cohort_decisions.csv")
    parser.add_argument("--cohort-report", type=Path, default=ROOT / "audit" / "agent4_final_cohort_report.json")
    parser.add_argument("--cap-frames", type=int, default=RUNNER_TEMPORAL_CAP)
    parser.add_argument("--min-valid-frames", type=int, default=MIN_VALID_AFTER_CAP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cap_frames != RUNNER_TEMPORAL_CAP or args.min_valid_frames != MIN_VALID_AFTER_CAP:
        raise SystemExit(f"Frozen cohort rule requires --cap-frames {RUNNER_TEMPORAL_CAP} and --min-valid-frames {MIN_VALID_AFTER_CAP}.")
    for path in (args.candidate_csv, args.candidate_decisions, args.candidate_report):
        if not path.is_file():
            raise SystemExit(f"Required candidate-stage artifact not found: {path}")
    candidates = csv_rows(args.candidate_csv)
    decisions = csv_rows(args.candidate_decisions)
    with args.candidate_report.open("r", encoding="utf-8") as f:
        progress = json.load(f)
    if not candidates:
        raise SystemExit("Candidate CSV is empty; refusing to freeze an empty cohort.")
    try:
        decision_by_member, _ = assert_candidate_run_complete(candidates, decisions, progress, args.candidate_csv)
    except Exception as exc:
        raise SystemExit(f"Candidate stage is not ready for finalization: {exc}") from exc

    finalized_rows: list[dict[str, Any]] = []
    decision_output: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        fields = canonical_fields(candidate)
        member = fields["zip_member"]
        source_decision = decision_by_member[member]
        staged_decision = str(source_decision.get("decision", "")).strip().lower()
        reason = str(source_decision.get("decision_reason", "")).strip()
        status = str(source_decision.get("status", "")).strip()
        record: dict[str, Any] = {
            **fields,
            "decision": "excluded",
            "decision_reason": "",
            "candidate_stage_decision": staged_decision,
            "candidate_stage_reason": reason,
            "status": status,
            "npz_path": first_value(candidate, ("npz_path",)) or first_value(source_decision, ("output_npz",)),
            "full_valid_frames": "",
            "valid_frames_after_720": "",
            "runner_cap_frame_count": "",
            "track_id": source_decision.get("track_id", ""),
            "selection_mode": source_decision.get("selection_mode", ""),
            "selected_track_frame_coverage": source_decision.get("selected_track_frame_coverage", ""),
            "selected_track_valid_clip_fraction": source_decision.get("selected_track_valid_clip_fraction", ""),
            "output_npz": first_value(source_decision, ("output_npz",)),
        }
        if staged_decision == "excluded":
            record["decision_reason"] = reason or f"excluded_at_candidate_preprocessing:{status or 'unspecified_status'}"
        else:
            row_npz = first_value(candidate, ("npz_path",))
            ledger_npz = first_value(source_decision, ("output_npz",))
            try:
                if status not in {"ok", "skipped_resume_verified"}:
                    raise ValueError(f"candidate status {status!r} is not accepted full-input status")
                if not row_npz or not ledger_npz:
                    raise ValueError("candidate row or decision ledger has no NPZ path")
                resolved_row_npz = resolve_path(row_npz)
                resolved_ledger_npz = resolve_path(ledger_npz)
                if os.path.normcase(str(resolved_row_npz)) != os.path.normcase(str(resolved_ledger_npz)):
                    raise ValueError("candidate manifest NPZ path differs from decision ledger output_npz")
                if not resolved_row_npz.is_file():
                    raise FileNotFoundError(f"NPZ is missing: {resolved_row_npz}")
                facts = validate_npz(candidate, resolved_row_npz)
                record.update(facts)
                record["decision"] = "accepted"
                record["decision_reason"] = "passed_full_npz_qc_and_30_valid_frames_after_runner_cap"
                final_row: dict[str, Any] = dict(candidate)
                if not first_value(final_row, ("phase",)):
                    final_row["phase"] = fields["phase"]
                final_row.update({
                    "qc_status": "passed",
                    "qc_decision_reason": record["decision_reason"],
                    "qc_full_valid_frames": facts["full_valid_frame_count"],
                    "qc_valid_frames_after_720": facts["valid_frames_after_720"],
                    "qc_runner_cap_frames": facts["runner_cap_frame_count"],
                    "qc_selection_mode": facts["selection_mode"],
                    "qc_track_id": facts["track_id"],
                    "qc_selected_track_frame_coverage": facts["selected_track_frame_coverage"],
                    "qc_selected_track_valid_clip_fraction": facts["selected_track_valid_clip_fraction"],
                })
                final_rows.append(final_row)
            except Exception as exc:
                record["decision_reason"] = f"postcap_npz_validation_failed:{type(exc).__name__}:{exc}"
        if record["decision"] == "excluded" and not record["decision_reason"]:
            record["decision_reason"] = f"excluded_without_candidate_reason:{status or 'unspecified_status'}"
        decision_output.append(record)

    retention_rows, subject_rows, cohort_counts = summarize_cohort(candidates, decision_output)
    if len(decision_output) != len(candidates) or len({r["zip_member"] for r in decision_output}) != len(candidates):
        raise RuntimeError("internal error: final cohort decisions do not cover every candidate exactly once")
    if len(final_rows) != cohort_counts["accepted_clips"]:
        raise RuntimeError("internal error: accepted manifest row count differs from finalized decisions")

    candidate_hash = sha256_file(args.candidate_csv)
    decisions_hash = sha256_file(args.candidate_decisions)
    report = {
        "schema": "live-video-final-cohort-qc-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_csv": str(args.candidate_csv),
        "candidate_manifest_sha256": candidate_hash,
        "candidate_decisions_csv": str(args.candidate_decisions),
        "candidate_decisions_sha256": decisions_hash,
        "candidate_progress_report": str(args.candidate_report),
        "candidate_finished_local": progress.get("finished_local"),
        "final_accepted_csv": str(args.final_csv),
        "final_accepted_count": len(final_rows),
        "final_accepted_manifest_sha256_after_write": None,
        "temporal_qc": {
            "runner_cap_frames": RUNNER_TEMPORAL_CAP,
            "cap_rule": "same as scripts/run_cv.py cap_temporal_slots: evenly spaced rounded linspace slots only when T>720",
            "minimum_joint_valid_frames_after_cap": MIN_VALID_AFTER_CAP,
            "joint_validity": "valid_frame_mask == pose_valid_mask AND mask_valid_mask",
            "target_fps": TARGET_FPS,
            "max_frames": 0,
            "smoke_inputs_accepted": False,
            "frame_index_reconstruction": "uses metadata_json.video.fps (or opencv_fps fallback), the same Python-precision rate used by the extractor; verifies float32(source_fps) consistency and reports legacy float32-replay mismatch count without excluding valid sampled inputs",
        },
        "cohort_counts": cohort_counts,
        "retention_by_class_view": retention_rows,
        "subject_view_coverage": subject_rows,
        "decisions": decision_output,
        "completion_marker_written": False,
        "note": "This finalizer writes no Agent 4 full-preprocessing completion marker. Run the accepted-only second pass after this CSV is frozen.",
    }

    final_fields = list(candidates[0].keys())
    for extra in (
        "phase", "qc_status", "qc_decision_reason", "qc_full_valid_frames",
        "qc_valid_frames_after_720", "qc_runner_cap_frames", "qc_selection_mode",
        "qc_track_id", "qc_selected_track_frame_coverage", "qc_selected_track_valid_clip_fraction",
    ):
        if extra not in final_fields:
            final_fields.append(extra)
    decisions_fields = [
        "video_id", "subject_id", "class_label", "view", "visit", "phase", "zip_member",
        "decision", "decision_reason", "candidate_stage_decision", "candidate_stage_reason",
        "status", "full_frame_count", "full_valid_frame_count", "valid_frames_after_720",
        "runner_cap_frame_count", "track_id", "selection_mode", "selected_track_frame_coverage",
        "selected_track_valid_clip_fraction", "npz_path", "output_npz",
    ]
    retention_fields = ["class_label", "view", "candidate_clips", "accepted_clips", "excluded_clips", "retention_fraction"]
    subject_fields = [
        "subject_id", "class_label", "candidate_clips", "accepted_clips", "excluded_clips",
        "candidate_views", "accepted_views", "missing_accepted_views", "accepted_view_coverage_count",
        "accepted_view_coverage_fraction", "candidate_left_clips", "candidate_right_clips",
        "candidate_sagittal_clips", "accepted_left_clips", "accepted_right_clips",
        "accepted_sagittal_clips", "accepted_visits",
    ]
    write_csv_atomic(args.final_csv, final_rows, final_fields)
    write_csv_atomic(args.decisions_csv, decision_output, decisions_fields)
    # Store class x view and subject coverage as both JSON-native report data and
    # compact tables for quick inspection by the experiment owner.
    write_csv_atomic(args.cohort_report.with_name("agent4_final_class_view_retention.csv"), retention_rows, retention_fields)
    write_csv_atomic(args.cohort_report.with_name("agent4_final_subject_view_coverage.csv"), subject_rows, subject_fields)
    report["final_accepted_manifest_sha256_after_write"] = sha256_file(args.final_csv)
    report["final_cohort_decisions_csv"] = str(args.decisions_csv)
    report["class_view_retention_csv"] = str(args.cohort_report.with_name("agent4_final_class_view_retention.csv"))
    report["subject_view_coverage_csv"] = str(args.cohort_report.with_name("agent4_final_subject_view_coverage.csv"))
    write_json_atomic(args.cohort_report, report)

    print(
        f"Final cohort written: accepted={len(final_rows)}/{len(candidates)}, "
        f"excluded={len(candidates) - len(final_rows)}, subjects={cohort_counts['accepted_subjects']}"
    )
    print(f"Accepted manifest: {args.final_csv}")
    print(f"Decision ledger: {args.decisions_csv}")
    print(f"Cohort report: {args.cohort_report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; no completion marker was written.", file=sys.stderr)
        raise SystemExit(130)
