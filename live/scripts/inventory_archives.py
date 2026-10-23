#!/usr/bin/env python3
"""Privacy-conscious, read-only inventory of LIVE-GaitNeuroKids ZIP archives.

The script reads ZIP central directories and only opens a small number of
technical documentation, image, array-header, and video samples. It never
extracts a full archive and never reads the adjacent Excel workbook.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_INPUT_ROOT = Path("data/LIVE-GaitNeuroKids")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ("DATASET.zip", "VIDEO.zip", "ANALYSIS.zip")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARRAY_EXTS = {".npy", ".npz", ".mat", ".pkl", ".pickle", ".pt", ".pth"}
DOC_EXTS = {".md", ".txt", ".rst", ".yaml", ".yml", ".json", ".xml"}
TECH_TERMS = re.compile(
    r"readme|data\s*dictionary|codebook|schema|format|recording|video|frame|"
    r"camera|view|angle|resolution|fps|joint|skeleton|pose|mask|coordinate|"
    r"annotation|label|column|file|folder|sampling|capture",
    re.I,
)
CLINICAL_TERMS = re.compile(
    r"diagnos|severity|curve|cobb|age|sex|gender|clinical|medical|treatment|"
    r"outcome|patient\s+characteristic",
    re.I,
)
ID_LIKE = re.compile(
    r"^(?:(?:participant|subject|patient|person|sub|sid|pid|id|p|s)[-_ ]*)?"
    r"(?:\d{2,}|[a-z]{0,3}\d{2,})$",
    re.I,
)
SUBJECT_TAG = re.compile(
    r"(?:participant|subject|patient|person|sub|sid|pid|id|p|s)[-_ ]*"
    r"([a-z]*\d{2,}|\d+)",
    re.I,
)
TRIAL_TERMS = {
    "walk": re.compile(r"walk|gait|walking", re.I),
    "static": re.compile(r"static|stand|standing|posture", re.I),
    "other_task": re.compile(r"turn|run|step|trial|task", re.I),
}


def safe_component(component: str) -> str:
    """Keep only known technical/category tokens in reported path samples."""
    value = component.strip()
    if not value:
        return value
    if ID_LIKE.fullmatch(value):
        return "<participant-or-index>"
    if SUBJECT_TAG.search(value):
        return "<participant-component>"
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return "<numeric-component>"
    lower = value.casefold().strip(" ._")
    safe_tokens = {
        "cp", "itw", "td", "readme", "data", "dataset", "video", "videos",
        "analysis", "metadata", "dictionary", "codebook", "schema", "image",
        "images", "frame", "frames", "pose", "poses", "keypoint", "keypoints",
        "skeleton", "mask", "masks", "segmentation", "silhouette", "front",
        "frontal", "anterior", "side", "lateral", "profile", "oblique", "back",
        "posterior", "rear", "camera1", "camera2", "cam1", "cam2", "view1",
        "view2", "walk", "walking", "gait", "static", "standing", "trial",
        "task", "labels", "label", "annotations", "coordinates", "joints",
    }
    if lower in safe_tokens:
        return value
    if Path(value).suffix:
        return "sample" + Path(value).suffix.lower()
    return "<directory>"


def safe_member_path(name: str) -> str:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return "/".join(safe_component(part) for part in parts)


def classify_view(text: str) -> str:
    # Parse complete filename/path tokens; never infer an unmarked view.
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", text.casefold()) if t]
    token_set = set(tokens)
    if "sagittal" in token_set or "sagital" in token_set:
        return "sagittal"
    if "left" in token_set and token_set.intersection({"front", "frontal"}):
        return "front_left"
    if "right" in token_set and token_set.intersection({"front", "frontal"}):
        return "front_right"
    if "left" in token_set and "right" not in token_set:
        return "left"
    if "right" in token_set and "left" not in token_set:
        return "right"
    if token_set.intersection({"oblique", "diagonal"}):
        return "oblique"
    if token_set.intersection({"back", "posterior", "rear", "coronal", "transverse"}):
        return "back"
    if token_set.intersection({"side", "lateral", "profile", "sagittal", "lat", "camera2", "cam2", "view2"}):
        return "side"
    if token_set.intersection({"front", "frontal", "anterior", "ap", "camera1", "cam1", "view1"}):
        return "front"
    return "not_named"


def class_label(text: str) -> str | None:
    for label in ("CP", "ITW", "TD"):
        if re.search(rf"(?<![A-Za-z0-9]){label}(?![A-Za-z0-9])", text, re.I):
            return label
    return None


def video_kind(text: str) -> str:
    tokens = set(t for t in re.split(r"[^A-Za-z0-9]+", text.casefold()) if t)
    if "marker" in tokens:
        return "static_calibration"
    if tokens.intersection({"calibration", "calib", "checkerboard"}) or re.search(r"camera[_ -]?setup", text, re.I):
        return "camera_calibration"
    if tokens.intersection({"sensor", "placement", "install", "installation", "setup", "attach", "attachment"}):
        return "sensor_placement_or_setup"
    if tokens.intersection({"walk", "walking", "gait"}):
        return "walking"
    if tokens.intersection({"static", "stand", "standing", "posture", "turn", "run", "running", "step", "test"}):
        return "other_named_task"
    return "unknown"


def visit_hint(text: str) -> str:
    tokens = [t.upper() for t in re.split(r"[^A-Za-z0-9]+", text) if t]
    return next((t for t in tokens if t in {"C0", "T0", "T1", "T2", "TX"}), "")


def subject_code_from_parts(parts: tuple[str, ...]) -> str:
    """Keep a directory code only when it has an explicit numeric subject-code shape."""
    label_index = next((i for i, part in enumerate(parts[:-1]) if part.upper() in {"CP", "ITW", "TD"}), None)
    if label_index is None or label_index + 1 >= len(parts) - 1:
        return ""
    candidate = parts[label_index + 1].strip()
    if re.fullmatch(r"(?i)(?:subject|participant|sub|sid|pid|p|s)?[_-]?[a-z0-9]*\d{1,6}[a-z]?", candidate):
        return candidate
    return ""


def source_subject_code_from_member(name: str) -> str:
    """Prefer public Sxx/Pxx source codes in the basename, then the subject directory."""
    basename = PurePosixPath(name.replace("\\", "/")).stem
    match = re.search(r"(?<![A-Za-z0-9])((?:S|P)\d{1,4})(?![A-Za-z0-9])", basename, re.I)
    if match:
        return match.group(1).upper()
    return subject_code_from_parts(PurePosixPath(name.replace("\\", "/")).parts)


def is_junk_member(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    base = parts[-1] if parts else name
    return (
        any(part == "__MACOSX" for part in parts)
        or base.startswith("._")
        or base.casefold() in {".ds_store", "thumbs.db", "desktop.ini"}
    )


def junk_reason(name: str) -> str | None:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    base = parts[-1] if parts else name
    if any(part == "__MACOSX" for part in parts):
        return "__MACOSX_path"
    if base.startswith("._"):
        return "AppleDouble_basename"
    if base.casefold() in {".ds_store", "thumbs.db", "desktop.ini"}:
        return "system_metadata_file"
    return None


def video_manifest_rows(zip_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    all_subjects: dict[str, set[str]] = {"CP": set(), "ITW": set(), "TD": set()}
    walking_subjects: dict[str, set[str]] = {"CP": set(), "ITW": set(), "TD": set()}
    all_view_subjects: dict[str, dict[str, set[str]]] = {k: collections.defaultdict(set) for k in all_subjects}
    walking_view_subjects: dict[str, dict[str, set[str]]] = {k: collections.defaultdict(set) for k in all_subjects}
    walking_views_by_subject: dict[str, dict[str, set[str]]] = {
        k: collections.defaultdict(set) for k in all_subjects
    }
    class_kind_view_counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or is_junk_member(info.filename) or Path(info.filename).suffix.lower() not in VIDEO_EXTS:
                continue
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            label = class_label(info.filename) or ""
            subject = source_subject_code_from_member(info.filename)
            kind = video_kind(info.filename)
            view = classify_view(info.filename)
            row = {
                "zip_member": info.filename,
                "subject_id": subject,
                "source_subject_code": subject,
                "class_label": label,
                "video_kind": kind,
                "view_hint": "" if view == "not_named" else view,
                "visit_hint": visit_hint(info.filename),
                "bytes": info.file_size,
            }
            rows.append(row)
            counts[kind] += 1
            class_kind_view_counts[(label or "unknown", kind, view if view != "not_named" else "not_named")] += 1
            if label in all_subjects and subject:
                all_subjects[label].add(subject)
                if view != "not_named":
                    all_view_subjects[label][view].add(subject)
                if kind == "walking":
                    walking_subjects[label].add(subject)
                    if view != "not_named":
                        walking_view_subjects[label][view].add(subject)
                        walking_views_by_subject[label][subject].add(view)
    required_views = {"left", "right", "sagittal"}
    coverage = {
        "video_count_by_kind": dict(counts),
        "video_count_by_class_kind_view": {
            "|".join(key): count for key, count in sorted(class_kind_view_counts.items())
        },
        "unique_subject_code_count_by_class_all_kinds": {k: len(v) for k, v in all_subjects.items()},
        "unique_subject_code_count_by_class_walking": {k: len(v) for k, v in walking_subjects.items()},
        "named_view_subject_counts_by_class_all_kinds": {
            cls: {view: len(subjects) for view, subjects in sorted(by_view.items())}
            for cls, by_view in all_view_subjects.items()
        },
        "named_view_subject_counts_by_class_walking": {
            cls: {view: len(subjects) for view, subjects in sorted(by_view.items())}
            for cls, by_view in walking_view_subjects.items()
        },
        "walking_subjects_with_multiple_named_views_by_class": {
            cls: sum(len(views) >= 2 for views in views_by_subject.values())
            for cls, views_by_subject in walking_views_by_subject.items()
        },
        "walking_subjects_with_all_three_named_views_by_class": {
            cls: sum(required_views.issubset(views) for views in views_by_subject.values())
            for cls, views_by_subject in walking_views_by_subject.items()
        },
    }
    return rows, coverage


def compare_dataset_video_subject_coverage(dataset_zip: Path, video_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_ids: dict[str, set[str]] = {"CP": set(), "ITW": set(), "TD": set()}
    video_ids: dict[str, set[str]] = {"CP": set(), "ITW": set(), "TD": set()}
    if dataset_zip.is_file():
        with zipfile.ZipFile(dataset_zip, "r") as zf:
            for info in zf.infolist():
                if info.is_dir() or is_junk_member(info.filename):
                    continue
                parts = PurePosixPath(info.filename.replace("\\", "/")).parts
                label = class_label(info.filename)
                subject = subject_code_from_parts(parts)
                if label in dataset_ids and subject:
                    dataset_ids[label].add(subject.casefold())
    for row in video_rows:
        label = row["class_label"]
        subject = row["subject_id"]
        if label in video_ids and subject:
            video_ids[label].add(subject.casefold())
    result: dict[str, Any] = {}
    for label in ("CP", "ITW", "TD"):
        both = dataset_ids[label] & video_ids[label]
        result[label] = {
            "dataset_subject_codes": len(dataset_ids[label]),
            "video_subject_codes": len(video_ids[label]),
            "shared_subject_codes": len(both),
            "dataset_subject_codes_without_video": len(dataset_ids[label] - video_ids[label]),
            "video_subject_codes_without_dataset_match": len(video_ids[label] - dataset_ids[label]),
        }
    result["comparison_note"] = (
        "Preliminary exact-code comparison only. Dataset directory codes and video basename source codes "
        "may use different schemes; Agent2 crosswalk is required before treating differences as missing data."
    )
    return result


def read_safe_doc(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    name_lower = info.filename.lower()
    kind = "readme" if "readme" in name_lower else (
        "data_dictionary" if any(x in name_lower for x in ("dictionary", "codebook", "schema"))
        else "technical_metadata"
    )
    result: dict[str, Any] = {
        "path": safe_member_path(info.filename),
        "kind": kind,
        "uncompressed_bytes": info.file_size,
        "inspected": False,
        "technical_excerpt": [],
    }
    # Do not open arbitrary JSON/XML/TXT payloads: restrict reads to clearly named docs.
    if not any(tag in name_lower for tag in ("readme", "dictionary", "codebook", "schema", "metadata")):
        return result
    try:
        with zf.open(info) as stream:
            raw = stream.read(128 * 1024)
        decoded = raw.decode("utf-8-sig", errors="replace")
        lines: list[str] = []
        for line in decoded.splitlines():
            line = line.strip()
            if not line or len(line) > 500 or not TECH_TERMS.search(line) or CLINICAL_TERMS.search(line):
                continue
            # Avoid table/data rows and suppress numeric IDs and personal-level values.
            if line.count(",") >= 2 or line.count("\t") >= 2 or SUBJECT_TAG.search(line):
                continue
            line = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", line)
            line = re.sub(r"[A-Z][a-z]+\s+[A-Z][a-z]+", "<redacted-name>", line)
            lines.append(line)
            if len(lines) >= 12:
                break
        result["inspected"] = True
        result["technical_excerpt"] = lines
        result["excerpt_was_truncated"] = len(raw) >= 128 * 1024
    except Exception as exc:  # document issue should not stop archive inventory
        result["read_error"] = type(exc).__name__
    return result


def inspect_image(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    out: dict[str, Any] = {"path": safe_member_path(info.filename), "extension": Path(info.filename).suffix.lower()}
    try:
        from PIL import Image

        with zf.open(info) as stream:
            with Image.open(stream) as image:
                out.update({"width": image.width, "height": image.height, "mode": image.mode, "format": image.format})
    except Exception as exc:
        out["image_metadata_error"] = type(exc).__name__
    return out


def inspect_array_header(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    out: dict[str, Any] = {"path": safe_member_path(info.filename), "extension": Path(info.filename).suffix.lower()}
    if Path(info.filename).suffix.lower() != ".npy":
        out["details"] = "header not read for this container format"
        return out
    try:
        with zf.open(info) as stream:
            magic = stream.read(6)
            if magic != b"\x93NUMPY":
                out["header_error"] = "invalid_npy_magic"
                return out
            version = stream.read(2)
            if version == b"\x01\x00":
                header_len = int.from_bytes(stream.read(2), "little")
            elif version in (b"\x02\x00", b"\x03\x00"):
                header_len = int.from_bytes(stream.read(4), "little")
            else:
                out["header_error"] = "unsupported_npy_version"
                return out
            header = stream.read(min(header_len, 65536)).decode("latin1", errors="replace")
        shape_match = re.search(r"['\"]shape['\"]\s*:\s*\(([^)]*)\)", header)
        dtype_match = re.search(r"['\"]descr['\"]\s*:\s*['\"]([^'\"]+)", header)
        out["shape"] = [int(n.strip()) for n in shape_match.group(1).split(",") if n.strip().isdigit()] if shape_match else None
        out["dtype"] = dtype_match.group(1) if dtype_match else None
        # Data values are never read.
    except Exception as exc:
        out["header_error"] = type(exc).__name__
    return out


def field_hint(header: str) -> str:
    text = header.casefold().strip()
    rules = (
        ("participant", re.compile(r"participant|subject|patient|person|\bsid\b|\bpid\b", re.I)),
        ("trial", re.compile(r"trial|task|condition|sequence", re.I)),
        ("view", re.compile(r"view|camera|angle|perspective", re.I)),
        ("frame_or_time", re.compile(r"frame|time|timestamp|sample|cycle", re.I)),
        ("joint_or_landmark", re.compile(r"joint|landmark|keypoint|skeleton|segment|bone", re.I)),
        ("coordinate", re.compile(r"(^|[_ .-])[xyz]($|[_ .-])|coordinate|position", re.I)),
        ("orientation_or_angle", re.compile(r"orientation|quaternion|angle|rotation", re.I)),
        ("velocity_or_acceleration", re.compile(r"velocity|acceleration|speed", re.I)),
        ("confidence", re.compile(r"confidence|score|visibility", re.I)),
        ("label", re.compile(r"label|class|group", re.I)),
    )
    for name, rule in rules:
        if rule.search(text):
            return name
    return "other_field"


def inspect_csv_header(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    """Read only the first CSV row; never include row values in the manifest."""
    out: dict[str, Any] = {"path": safe_member_path(info.filename), "uncompressed_bytes": info.file_size}
    try:
        with zf.open(info) as stream:
            raw = stream.read(64 * 1024)
        decoded = raw.decode("utf-8-sig", errors="replace")
        first_line = next((line for line in decoded.splitlines() if line.strip()), "")
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = max((",", ";", "\t", "|"), key=first_line.count)
        first = next(csv.reader([first_line], delimiter=delimiter), [])
        nonempty = [cell.strip() for cell in first if cell.strip()]
        looks_header = bool(nonempty) and any(re.search(r"[A-Za-z_]", cell) for cell in nonempty)
        out["first_row_looks_like_header"] = looks_header
        out["column_count"] = len(first)
        out["delimiter_hint"] = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(delimiter, "unknown")
        out["field_hints"] = [field_hint(cell) for cell in first] if looks_header else []
        out["data_rows_read"] = 0
    except Exception as exc:
        out["header_inspection_error"] = type(exc).__name__
    return out


def inspect_binary_signature(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    """Classify a few opaque formats without exposing payload bytes or values."""
    with zf.open(info) as stream:
        prefix = stream.read(64)
    ext = Path(info.filename).suffix.lower()
    signature = "opaque_or_vendor_specific"
    if prefix.startswith(b"\x93NUMPY"):
        signature = "npy_array"
    elif prefix.startswith(b"PK\x03\x04"):
        signature = "zip_container"
    elif prefix.lstrip().startswith((b"<?xml", b"<")):
        signature = "xml_or_markup"
    elif prefix.lstrip().startswith((b"{", b"[")):
        signature = "json_like_text"
    elif prefix.startswith(b"\x00\x00\x00"):
        signature = "binary_container"
    else:
        try:
            prefix.decode("utf-8")
            signature = "text_or_csv_like"
        except UnicodeDecodeError:
            pass
    return {
        "path": safe_member_path(info.filename),
        "extension": ext,
        "uncompressed_bytes": info.file_size,
        "prefix_classification": signature,
    }


def ffprobe_video(zf: zipfile.ZipFile, info: zipfile.ZipInfo, ffprobe: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": safe_member_path(info.filename),
        "view_hint": classify_view(info.filename),
        "uncompressed_bytes": info.file_size,
    }
    # Decode metadata for only a few videos, and never write into the source directory.
    if info.file_size > 350 * 1024 * 1024:
        out["metadata_status"] = "skipped_sample_over_350MiB"
        return out
    temp_path: str | None = None
    try:
        suffix = Path(info.filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(prefix="live_gait_sample_", suffix=suffix, delete=False) as tmp:
            temp_path = tmp.name
            with zf.open(info) as member:
                shutil.copyfileobj(member, tmp, length=4 * 1024 * 1024)
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-show_entries", "format=duration", "-of", "json", temp_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if proc.returncode != 0:
            out["metadata_status"] = "ffprobe_failed"
            out["error_excerpt"] = proc.stderr.strip()[:300]
            return out
        meta = json.loads(proc.stdout)
        stream = (meta.get("streams") or [{}])[0]
        out["metadata_status"] = "ok"
        for key in ("codec_name", "width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames", "duration"):
            if stream.get(key) not in (None, "N/A"):
                out[key] = stream[key]
        if meta.get("format", {}).get("duration") and "duration" not in out:
            out["duration"] = meta["format"]["duration"]
    except Exception as exc:
        out["metadata_status"] = "inspection_error"
        out["error_type"] = type(exc).__name__
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    return out


def archive_report(archive_path: Path, ffprobe: str | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "archive": archive_path.name,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }
    with zipfile.ZipFile(archive_path, "r") as zf:
        raw_infos = zf.infolist()
        infos = raw_infos
        raw_files = [x for x in raw_infos if not x.is_dir()]
        junk_count = sum(is_junk_member(x.filename) for x in raw_files)
        raw_ext_counts: collections.Counter[str] = collections.Counter()
        junk_reasons: collections.Counter[str] = collections.Counter()
        for info in raw_files:
            raw_path = PurePosixPath(info.filename.replace("\\", "/"))
            raw_ext_counts[raw_path.suffix.lower() or "[no extension]"] += 1
            reason = junk_reason(info.filename)
            if reason:
                junk_reasons[reason] += 1
        infos = [x for x in infos if not is_junk_member(x.filename)]
        files = [x for x in infos if not x.is_dir()]
        ext_counts: collections.Counter[str] = collections.Counter()
        top_counts: collections.Counter[str] = collections.Counter()
        named_labels: collections.Counter[str] = collections.Counter()
        class_participants: dict[str, set[str]] = {"CP": set(), "ITW": set(), "TD": set()}
        class_extensions: dict[str, collections.Counter[str]] = {
            "CP": collections.Counter(), "ITW": collections.Counter(), "TD": collections.Counter()
        }
        view_counts: collections.Counter[str] = collections.Counter()
        trial_counts: collections.Counter[str] = collections.Counter()
        id_pattern_counts: collections.Counter[str] = collections.Counter()
        unique_subject_tokens: set[str] = set()
        keypoint_names = 0
        mask_names = 0
        image_names = 0
        video_infos: list[zipfile.ZipInfo] = []
        image_infos: list[zipfile.ZipInfo] = []
        array_infos: list[zipfile.ZipInfo] = []
        csv_infos: list[zipfile.ZipInfo] = []
        opaque_infos: dict[str, list[zipfile.ZipInfo]] = collections.defaultdict(list)
        docs: list[dict[str, Any]] = []
        path_depths: collections.Counter[int] = collections.Counter()

        for info in infos:
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename.replace("\\", "/"))
            parts = path.parts
            if parts:
                top = parts[0]
                top_counts[safe_component(top)] += 1
            path_depths[len(parts)] += 1
            ext = path.suffix.lower() or "[no extension]"
            ext_counts[ext] += 1
            joined = "/".join(parts).lower()
            for label in ("CP", "ITW", "TD"):
                if re.search(rf"(?<![a-z0-9]){label.lower()}(?![a-z0-9])", joined):
                    named_labels[label] += 1
                    class_extensions[label][ext] += 1
                    parts_lower = [p.lower() for p in parts]
                    label_index = next((i for i, p in enumerate(parts_lower) if p == label.lower()), None)
                    if label_index is not None and label_index + 1 < len(parts) - 1:
                        candidate_subject = parts[label_index + 1]
                        if ID_LIKE.fullmatch(candidate_subject) or SUBJECT_TAG.search(candidate_subject):
                            class_participants[label].add(candidate_subject.casefold())
                    break
            view = classify_view(joined)
            if view != "not_named":
                view_counts[view] += 1
            for trial, regex in TRIAL_TERMS.items():
                if regex.search(joined):
                    trial_counts[trial] += 1
            for part in parts[:-1]:
                subject_match = SUBJECT_TAG.search(part)
                if subject_match:
                    unique_subject_tokens.add(subject_match.group(0).casefold())
                    id_pattern_counts["tagged_participant_component"] += 1
            if re.search(r"key.?points?|skeleton|joint|pose", joined):
                keypoint_names += 1
            if re.search(r"mask|segmentation|silhouette", joined):
                mask_names += 1
            if ext in IMAGE_EXTS:
                image_names += 1
                if len(image_infos) < 3:
                    image_infos.append(info)
            if ext in VIDEO_EXTS:
                video_infos.append(info)
            if ext in ARRAY_EXTS:
                array_infos.append(info)
            if ext == ".csv":
                csv_infos.append(info)
            if ext in {".mvn", ".mvna", ".shpf", ".hpf"} and len(opaque_infos[ext]) < 2:
                opaque_infos[ext].append(info)
            if ext in DOC_EXTS and any(tag in joined for tag in ("readme", "dictionary", "codebook", "schema", "metadata")):
                if len(docs) < 12:
                    docs.append(read_safe_doc(zf, info))

        report.update({
            "raw_member_count_including_dirs": len(raw_infos),
            "member_count_excluding_metadata_sidecars": len(infos),
            "raw_file_count_including_metadata_sidecars": len(raw_files),
            "excluded_macos_or_system_sidecar_count": junk_count,
            "raw_extension_counts_including_sidecars": dict(sorted(raw_ext_counts.items())),
            "excluded_sidecar_counts_by_reason": dict(junk_reasons),
            "file_count": len(files),
            "directory_count": sum(x.is_dir() for x in infos),
            "uncompressed_total_bytes": sum(x.file_size for x in files),
            "extension_counts": dict(sorted(ext_counts.items(), key=lambda kv: kv[0])),
            "top_level_component_counts_redacted": dict(sorted(top_counts.items())),
            "path_depth_counts": {str(k): v for k, v in sorted(path_depths.items())},
            "class_name_occurrences_by_archive_path": dict(named_labels),
            "participant_like_directory_counts_by_class_heuristic": {k: len(v) for k, v in class_participants.items()},
            "file_extension_counts_by_class": {k: dict(v) for k, v in class_extensions.items()},
            "view_name_occurrences_by_archive_path": dict(view_counts),
            "trial_keyword_occurrences_by_archive_path": dict(trial_counts),
            "tagged_participant_like_directory_count": len(unique_subject_tokens),
            "participant_id_pattern_counts_redacted": dict(id_pattern_counts),
            "keypoint_or_pose_named_member_count": keypoint_names,
            "mask_or_silhouette_named_member_count": mask_names,
            "image_member_count": image_names,
            "video_member_count": len(video_infos),
            "array_or_model_member_count": len(array_infos),
            "documentation": docs,
        })

        report["image_samples"] = [inspect_image(zf, info) for info in image_infos]
        array_samples = array_infos[:3]
        report["array_samples"] = [inspect_array_header(zf, info) for info in array_samples]
        sampled_csvs: list[zipfile.ZipInfo] = []
        seen_csv_classes: set[str] = set()
        for info in csv_infos:
            label = class_label(info.filename) or "unclassified"
            if label not in seen_csv_classes:
                sampled_csvs.append(info)
                seen_csv_classes.add(label)
            if len(sampled_csvs) >= 4:
                break
        report["csv_header_samples"] = [inspect_csv_header(zf, info) for info in sampled_csvs]
        report["opaque_format_samples"] = [
            inspect_binary_signature(zf, info)
            for ext in (".mvn", ".mvna", ".shpf", ".hpf")
            for info in opaque_infos.get(ext, [])[:1]
        ]
        if video_infos:
            sizes = sorted(x.file_size for x in video_infos)
            report["video_size_summary"] = {
                "minimum_bytes": sizes[0],
                "median_bytes": sizes[len(sizes) // 2],
                "maximum_bytes": sizes[-1],
                "under_10KiB_count": sum(size < 10 * 1024 for size in sizes),
            }

        # Pick up to one video per named view class, then fill to five total.
        if ffprobe and video_infos:
            selected: list[zipfile.ZipInfo] = []
            selected_classes: set[str] = set()
            for info in video_infos:
                label = class_label(info.filename)
                if info.file_size < 10 * 1024:
                    continue
                if video_kind(info.filename) == "walking" and label and label not in selected_classes:
                    selected.append(info)
                    selected_classes.add(label)
                if len(selected) >= 3:
                    break
            for info in video_infos:
                if len(selected) >= 3:
                    break
                if info.file_size >= 10 * 1024 and info not in selected:
                    selected.append(info)
            report["video_samples"] = [ffprobe_video(zf, info, ffprobe) for info in selected[:3]]
        else:
            report["video_samples"] = []
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or shutil.which("ffprobe.exe"))
    args = parser.parse_args()
    output_dir = args.output_root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    archives: list[dict[str, Any]] = []
    absent: list[str] = []
    for filename in ARCHIVES:
        path = args.input_root / filename
        if path.is_file():
            archives.append(archive_report(path, args.ffprobe))
        else:
            absent.append(filename)

    videos_path = args.input_root / "VIDEO.zip"
    videos_csv_path: Path | None = None
    video_coverage: dict[str, Any] = {}
    video_rows: list[dict[str, Any]] = []
    if videos_path.is_file():
        video_rows, video_coverage = video_manifest_rows(videos_path)
        videos_csv_path = output_dir / "videos.csv"
        with videos_csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["zip_member", "subject_id", "source_subject_code", "class_label", "video_kind", "view_hint", "visit_hint", "bytes"],
            )
            writer.writeheader()
            writer.writerows(video_rows)

    workbook = args.input_root / "PARTICIPANT_CHARACTERISTICS.xlsx"
    result = {
        "generated_at_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_root": str(args.input_root),
        "inspection_policy": {
            "read_zip_central_directories": True,
            "full_archive_extraction": False,
            "sample_video_temp_copy_only": True,
            "excel_workbook_opened": False,
            "clinical_row_values_read_or_emitted": False,
            "participant_identifiers_in_archive_inventory": False,
            "anonymous_subject_codes_in_video_manifest": True,
            "subject_id_in_video_manifest_is_source_code_pending_crosswalk": True,
            "real_names_or_individual_clinical_values_emitted": False,
            "sample_limit_per_archive": {"videos": 3, "images": 3, "array_headers": 3, "documentation": 12},
        },
        "adjacent_excel_workbook": {
            "filename": workbook.name,
            "exists": workbook.is_file(),
            "bytes": workbook.stat().st_size if workbook.is_file() else None,
            "contents_inspected": False,
            "owner": "Agent2",
        },
        "missing_archives": absent,
        "dataset_video_subject_coverage_by_class": compare_dataset_video_subject_coverage(
            args.input_root / "DATASET.zip", video_rows
        ),
        "video_manifest": {
            "path": str(videos_csv_path) if videos_csv_path else None,
            "non_junk_video_rows": sum(video_coverage.get("video_count_by_kind", {}).values()),
            "row_parsing_rules": {
                "video_kind": {
                    "walking": ["GAIT", "WALK", "WALKING"],
                    "static_calibration": ["MARKER"],
                    "camera_calibration": ["CALIB", "CALIBRATION", "CHECKERBOARD"],
                },
                "view_hint": ["LEFT", "RIGHT", "SAGITTAL", "SAGITAL"],
                "visit_hint": ["C0", "T0", "T1", "T2", "TX"],
                "unmatched_view_or_visit_token": "empty string; no fallback inference",
                "subject_id": "same public source code as source_subject_code pending Agent2 crosswalk",
            },
            **video_coverage,
        },
        "archives": archives,
    }
    manifest_path = output_dir / "archive_inventory.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "video_manifest": str(videos_csv_path) if videos_csv_path else None,
        "archives": [{
            "archive": r["archive"], "file_count": r["file_count"],
            "video_member_count": r["video_member_count"],
            "keypoint_or_pose_named_member_count": r["keypoint_or_pose_named_member_count"],
            "mask_or_silhouette_named_member_count": r["mask_or_silhouette_named_member_count"],
            "video_samples_with_metadata": sum(x.get("metadata_status") == "ok" for x in r["video_samples"]),
        } for r in archives],
        "missing_archives": absent,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
