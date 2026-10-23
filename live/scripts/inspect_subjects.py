from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DEFAULT_XLSX = Path("data/LIVE-GaitNeuroKids/PARTICIPANT_CHARACTERISTICS.xlsx")
DEFAULT_DATA = Path("data/LIVE-GaitNeuroKids")


def col_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref or "")
    if not letters:
        return 0
    out = 0
    for char in letters.group(0):
        out = out * 26 + ord(char) - 64
    return out - 1


def read_xlsx_sheets(path: Path) -> dict[str, list[list[str]]]:
    """Read OOXML cell text through ZIP/XML metadata; never write to the source."""
    with zipfile.ZipFile(path) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", NS):
                shared.append("".join(t.text or "" for t in item.findall(".//main:t", NS)))

        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels.findall("pkg:Relationship", NS)}
        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook.findall(".//main:sheets/main:sheet", NS):
            name = sheet.attrib["name"]
            target = targets[sheet.attrib[f"{{{NS['rel']}}}id"]]
            member = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            root = ET.fromstring(book.read(Path(member).as_posix()))
            rows: list[list[str]] = []
            for row in root.findall(".//main:sheetData/main:row", NS):
                cells: dict[int, str] = {}
                for cell in row.findall("main:c", NS):
                    idx = col_index(cell.attrib.get("r", ""))
                    kind = cell.attrib.get("t", "")
                    value = cell.find("main:v", NS)
                    if kind == "s" and value is not None and value.text:
                        text = shared[int(value.text)]
                    elif kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.findall(".//main:t", NS))
                    else:
                        text = value.text if value is not None and value.text is not None else ""
                    cells[idx] = text
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
            sheets[name] = rows
    return sheets


def key_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def get_subject_tables(path: Path):
    sheets = read_xlsx_sheets(path)
    tables = {}
    for name, rows in sheets.items():
        nonempty = [(i, row) for i, row in enumerate(rows) if any(v.strip() for v in row)]
        if not nonempty:
            continue
        header_i, header = nonempty[0]
        header_norm = [key_name(v) for v in header]
        code_idx = next((i for i, h in enumerate(header_norm) if h in {"code", "participant code", "subject code", "participant id", "subject id"}), None)
        phase_idx = next((i for i, h in enumerate(header_norm) if h == "phase"), None)
        pathology_idx = next((i for i, h in enumerate(header_norm) if h in {"pathology", "diagnosis", "diagnosis group"}), None)
        if code_idx is None:
            raise ValueError(f"No participant code column found in sheet {name!r}.")
        records = [row for i, row in nonempty if i > header_i]
        tables[name] = {
            "headers": header,
            "records": records,
            "code_idx": code_idx,
            "phase_idx": phase_idx,
            "pathology_idx": pathology_idx,
        }
    return tables


def natural_key(code: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", code)]


def inspect_archives(data_dir: Path, codes_by_class: dict[str, set[str]], phase_sets_by_class=None):
    archive_rows = []
    all_codes = {c.upper(): label for label, codes in codes_by_class.items() for c in codes}
    normalized_phase_sets = None
    if phase_sets_by_class is not None:
        normalized_phase_sets = {
            label: {str(code).upper(): {str(phase).upper() for phase in phases}
                    for code, phases in subjects.items()}
            for label, subjects in phase_sets_by_class.items()
        }
    known_codes = sorted(all_codes, key=len, reverse=True)
    code_pattern = re.compile(
        r"(?<![A-Z0-9])(" + "|".join(re.escape(code) for code in known_codes) + r")(?![A-Z0-9])",
        re.IGNORECASE,
    ) if known_codes else None
    phase_pattern = re.compile(r"(?<![A-Z0-9])(TX|T0|T1|T2|C0)(?![A-Z0-9])", re.IGNORECASE)
    task_pattern = re.compile(r"(?<![A-Z0-9])(GAIT|CALIB|MARKER)(?![A-Z0-9])", re.IGNORECASE)
    terms = {
        "natural": r"(?<![a-z0-9])natural(?![a-z0-9])",
        "normal": r"(?<![a-z0-9])normal(?![a-z0-9])",
        "self-selected": r"(?<![a-z0-9])self[-_ ]?selected(?![a-z0-9])",
        "preferred": r"(?<![a-z0-9])preferred(?![a-z0-9])",
        "comfortable": r"(?<![a-z0-9])comfortable(?![a-z0-9])",
        "walk": r"(?<![a-z0-9])walk(?![a-z0-9])",
        "walking": r"(?<![a-z0-9])walking(?![a-z0-9])",
        "gait": r"(?<![a-z0-9])gait(?![a-z0-9])",
        "overground": r"(?<![a-z0-9])overground(?![a-z0-9])",
        "baseline": r"(?<![a-z0-9])baseline(?![a-z0-9])",
        "treatment": r"(?<![a-z0-9])treatment(?![a-z0-9])",
        "therapy": r"(?<![a-z0-9])therapy(?![a-z0-9])",
        "static": r"(?<![a-z0-9])static(?![a-z0-9])",
        "calibration": r"(?<![a-z0-9])calibration(?![a-z0-9])",
        "sensor": r"(?<![a-z0-9])sensor(?![a-z0-9])",
        "marker": r"(?<![a-z0-9])marker(?![a-z0-9])",
        "placement": r"(?<![a-z0-9])placement(?![a-z0-9])",
        "treadmill": r"(?<![a-z0-9])treadmill(?![a-z0-9])",
        "fast": r"(?<![a-z0-9])fast(?![a-z0-9])",
        "slow": r"(?<![a-z0-9])slow(?![a-z0-9])",
        "barefoot": r"(?<![a-z0-9])barefoot(?![a-z0-9])",
        "shoe": r"(?<![a-z0-9])shoes?(?![a-z0-9])",
        "trial": r"(?<![a-z0-9])trial(?![a-z0-9])",
        "view": r"(?<![a-z0-9])view(?![a-z0-9])",
        "camera": r"(?<![a-z0-9])camera(?![a-z0-9])",
        "condition": r"(?<![a-z0-9])condition(?![a-z0-9])",
        "task": r"(?<![a-z0-9])task(?![a-z0-9])",
        "TX": r"(?<![a-z0-9])tx(?![a-z0-9])",
        "T0": r"(?<![a-z0-9])t0(?![a-z0-9])",
        "T1": r"(?<![a-z0-9])t1(?![a-z0-9])",
        "T2": r"(?<![a-z0-9])t2(?![a-z0-9])",
        "C0": r"(?<![a-z0-9])c0(?![a-z0-9])",
    }
    for archive in sorted(data_dir.glob("*.zip")):
        ext_counts = Counter()
        matched = defaultdict(set)
        gait_by_phase = defaultdict(set)
        gait_any = defaultdict(set)
        natural_gait_by_phase = defaultdict(set)
        natural_gait_any = defaultdict(set)
        task_counts = Counter()
        term_counts = Counter()
        member_count = 0
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                if item.is_dir():
                    continue
                member_count += 1
                suffix = Path(item.filename).suffix.lower() or "(none)"
                ext_counts[suffix] += 1
                upper = item.filename.upper()
                found_codes = {code.upper() for code in code_pattern.findall(upper)} if code_pattern else set()
                found_phases = {phase.upper() for phase in phase_pattern.findall(upper)}
                found_tasks = {task.upper() for task in task_pattern.findall(upper)}
                for code in found_codes:
                    label = all_codes[code]
                    matched[label].add(code)
                    if "GAIT" in found_tasks:
                        subject_phases = None
                        if normalized_phase_sets is not None:
                            subject_phases = normalized_phase_sets.get(label, {}).get(code, set())
                        matched_phases = found_phases if subject_phases is None else found_phases & {p.upper() for p in subject_phases}
                        for phase in found_phases:
                            if phase not in matched_phases:
                                continue
                            gait_by_phase[(label, phase)].add(code)
                            if phase != "TX":
                                natural_gait_by_phase[(label, phase)].add(code)
                        if any(phase != "TX" for phase in matched_phases):
                            natural_gait_any[label].add(code)
                        if matched_phases:
                            gait_any[label].add(code)
                for task in found_tasks:
                    task_counts[task] += 1
                lowered = item.filename.lower()
                for term, pattern in terms.items():
                    if re.search(pattern, lowered):
                        term_counts[term] += 1
        archive_rows.append({
            "archive": archive.name,
            "members": member_count,
            "extensions": ext_counts,
            "matched_subjects": {label: len(matched[label]) for label in ("CP", "ITW", "TD")},
            "gait_subjects_by_phase": {
                phase: {label: len(gait_by_phase[(label, phase)]) for label in ("CP", "ITW", "TD")}
                for phase in ("TX", "T0", "T1", "T2", "C0")
            },
            "gait_subjects_any_phase": {label: len(gait_any[label]) for label in ("CP", "ITW", "TD")},
            "natural_gait_subjects_by_phase": {
                phase: {label: len(natural_gait_by_phase[(label, phase)]) for label in ("CP", "ITW", "TD")}
                for phase in ("T0", "T1", "T2", "C0")
            },
            "natural_gait_subjects_any_phase": {
                label: len(natural_gait_any[label]) for label in ("CP", "ITW", "TD")
            },
            "task_files": dict(task_counts),
            "terms": term_counts,
        })
    return archive_rows


def build_manifest(tables, out_dir: Path):
    codes_by_class = {}
    phase_counts = {}
    phase_sets = {}
    record_counts = {}
    duplicate_visit_subjects = {}
    for raw_class, table in tables.items():
        label = raw_class.strip().upper()
        if label not in {"CP", "ITW", "TD"}:
            continue
        codes = set()
        phases = Counter()
        visits = Counter()
        phases_by_code = defaultdict(set)
        for row in table["records"]:
            code = row[table["code_idx"]].strip()
            if not code:
                continue
            codes.add(code)
            visits[code] += 1
            if table["phase_idx"] is not None and table["phase_idx"] < len(row):
                phase = row[table["phase_idx"]].strip()
                if phase:
                    phases[phase] += 1
                    phases_by_code[code].add(phase)
        codes_by_class[label] = codes
        phase_counts[label] = phases
        phase_sets[label] = phases_by_code
        record_counts[label] = len(table["records"])
        duplicate_visit_subjects[label] = sum(n > 1 for n in visits.values())

    cross_class_duplicates = set()
    for label, codes in codes_by_class.items():
        for other_label, other_codes in codes_by_class.items():
            if label < other_label:
                cross_class_duplicates |= codes & other_codes
    if cross_class_duplicates:
        raise ValueError("Source participant codes are duplicated across diagnosis sheets; resolve before writing manifest.")

    rows = []
    for label in ("CP", "ITW", "TD"):
        for n, _code in enumerate(sorted(codes_by_class.get(label, set()), key=natural_key), start=1):
            phases = sorted(phase_sets.get(label, {}).get(_code, set()))
            rows.append({
                "subject_id": f"{label}{n:03d}",
                "class_label": label,
                "source_subject_code": _code,
                "visit_phases": "|".join(phases),
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifests" / "subjects.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "class_label", "source_subject_code", "visit_phases"])
        writer.writeheader()
        writer.writerows(rows)
    return codes_by_class, phase_counts, phase_sets, record_counts, duplicate_visit_subjects, manifest


def inspect(data_dir: Path, xlsx: Path) -> None:
    tables = get_subject_tables(xlsx)
    codes_by_class = {}
    phase_sets_by_class = defaultdict(lambda: defaultdict(set))
    for sheet, table in tables.items():
        label = sheet.upper()
        if label not in {"CP", "ITW", "TD"}:
            continue
        codes = {row[table["code_idx"]].strip() for row in table["records"]
                 if table["code_idx"] < len(row) and row[table["code_idx"]].strip()}
        codes_by_class[label] = codes
        for row in table["records"]:
            if table["code_idx"] >= len(row):
                continue
            code = row[table["code_idx"]].strip().upper()
            if code and table["phase_idx"] is not None and table["phase_idx"] < len(row):
                phase = row[table["phase_idx"]].strip().upper()
                if phase:
                    phase_sets_by_class[label][code].add(phase)
        print(f"{label}: {len(codes)} unique participants; {len(table['records'])} nonempty source records")
    print(f"Workbook sheets used: {', '.join(sorted(codes_by_class))}")
    print("Workbook headers:")
    for sheet, table in tables.items():
        print(f"  {sheet}: {', '.join(v.strip() for v in table['headers'] if v.strip())}")
    dupes = set()
    for label, codes in codes_by_class.items():
        for other_label, other_codes in codes_by_class.items():
            if label < other_label:
                dupes |= codes & other_codes
    print(f"Participant code collisions across diagnosis sheets: {len(dupes)}")
    print("ZIP member inventory uses central-directory filenames only; no archive member contents are read.")
    for item in inspect_archives(data_dir, codes_by_class, phase_sets_by_class):
        print(f"  {item['archive']}: {item['members']} files; extensions={dict(item['extensions'])}")
        print(f"    filename-to-workbook ID matches by class={item['matched_subjects']}")
        print(f"    filename keyword counts={dict(item['terms'])}")


def write_audit(data_dir: Path, codes_by_class, phase_counts, phase_sets,
                record_counts, duplicate_visit_subjects, out_dir: Path):
    archive_rows = inspect_archives(data_dir, codes_by_class, phase_sets)
    class_counts = {label: len(codes_by_class.get(label, set())) for label in ("CP", "ITW", "TD")}
    minimum = min(class_counts.values()) if class_counts else 0
    feasible = all(n >= 5 for n in class_counts.values())
    lines = [
        "# LIVE-GaitNeuroKids subject and label audit",
        "",
        "The participant workbook was read in place. The three source ZIP files were inspected through archive filenames only; no ZIP member payload was opened or extracted by this metadata audit.",
        "",
        "## Label mapping and participant unit",
        "",
        "| Workbook sheet | Canonical class label | Source records | Unique participants | Participants with repeated records |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("CP", "ITW", "TD"):
        lines.append(f"| {label} | {label} | {record_counts.get(label, 0)} | {class_counts[label]} | {duplicate_visit_subjects.get(label, 0)} |")
    lines += [
        "",
        "The worksheet names provide the direct class mapping: CP → CP, ITW → ITW, and TD → TD. CODE is the source participant key. Repeated CODE values within a class represent repeated participant records/visits, so the independent unit is the unique participant, not a row or trial. The manifest adds stable, class-prefixed anonymous subject_id values and retains the public anonymous CODE as source_subject_code for joining to archive filenames.",
        "",
        "## Visit fields",
        "",
        "The workbook has a PHASE field. The dataset descriptor defines TX as the intervention, T0 as the baseline immediately before treatment, T1 as the short-term follow-up 1–2 weeks after treatment, and T2 as the follow-up at about 3 months. The TD records use C0. The manifest lists each subject's observed phase set. Retain the same anonymous subject ID across every visit and view.",
        "",
        "| Class | Aggregate source-record counts by phase |",
        "|---|---|",
    ]
    for label in ("CP", "ITW", "TD"):
        lines.append(f"| {label} | {dict(phase_counts.get(label, {}))} |")
    baseline_phase_counts = {
        "CP": sum("T0" in phases for phases in phase_sets.get("CP", {}).values()),
        "ITW": sum("T0" in phases for phases in phase_sets.get("ITW", {}).values()),
        "TD": sum("C0" in phases for phases in phase_sets.get("TD", {}).values()),
    }
    archive_by_name = {item["archive"]: item for item in archive_rows}
    dataset = archive_by_name.get("DATASET.zip", {})
    video = archive_by_name.get("VIDEO.zip", {})
    data_baseline = {
        "CP": dataset.get("gait_subjects_by_phase", {}).get("T0", {}).get("CP", 0),
        "ITW": dataset.get("gait_subjects_by_phase", {}).get("T0", {}).get("ITW", 0),
        "TD": dataset.get("gait_subjects_by_phase", {}).get("C0", {}).get("TD", 0),
    }
    video_baseline = {
        "CP": video.get("gait_subjects_by_phase", {}).get("T0", {}).get("CP", 0),
        "ITW": video.get("gait_subjects_by_phase", {}).get("T0", {}).get("ITW", 0),
        "TD": video.get("gait_subjects_by_phase", {}).get("C0", {}).get("TD", 0),
    }
    data_all_gait = dataset.get("natural_gait_subjects_any_phase", {label: 0 for label in ("CP", "ITW", "TD")})
    video_all_gait = video.get("natural_gait_subjects_any_phase", {label: 0 for label in ("CP", "ITW", "TD")})
    baseline_feasible = all(n >= 5 for n in video_baseline.values())
    all_gait_feasible = all(n >= 5 for n in data_all_gait.values())
    lines += [
        "",
        "## Candidate inclusion counts",
        "",
        "Counts are unique source CODE values. Archive counts require a matched participant code, phase token, and GAIT task token in the filename; they show naming coverage, not input completeness.",
        "",
        "| Candidate rule | CP | ITW | TD | Five-fold class minimum met? |",
        "|---|---:|---:|---:|---|",
        f"| Baseline phase only, workbook PHASE | {baseline_phase_counts['CP']} | {baseline_phase_counts['ITW']} | {baseline_phase_counts['TD']} | {'yes' if all(n >= 5 for n in baseline_phase_counts.values()) else 'no'} |",
        f"| Baseline phase plus GAIT task, DATASET.zip filenames | {data_baseline['CP']} | {data_baseline['ITW']} | {data_baseline['TD']} | {'yes' if all(n >= 5 for n in data_baseline.values()) else 'no'} |",
        f"| Baseline phase plus GAIT task, VIDEO.zip filenames | {video_baseline['CP']} | {video_baseline['ITW']} | {video_baseline['TD']} | {'yes' if baseline_feasible else 'no'} |",
        f"| All non-TX natural GAIT phases, DATASET.zip filenames | {data_all_gait['CP']} | {data_all_gait['ITW']} | {data_all_gait['TD']} | {'yes' if all_gait_feasible else 'no'} |",
        f"| All non-TX natural GAIT phases, VIDEO.zip filenames | {video_all_gait['CP']} | {video_all_gait['ITW']} | {video_all_gait['TD']} | {'yes' if all(n >= 5 for n in video_all_gait.values()) else 'no'} |",
        "",
        "## Five-fold stratification feasibility",
        "",
        f"Source-cohort counts are CP={class_counts['CP']}, ITW={class_counts['ITW']}, TD={class_counts['TD']}. Five-fold stratification is feasible by class at the source-cohort level: {'yes' if feasible else 'no'} (minimum class size {minimum}; each class needs at least five subjects to appear in all five folds). This is a feasibility check only, not a frozen split. The main model must recompute counts on subjects with complete eligible inputs before producing fold assignments.",
        "",
        "## ZIP filename crosswalk and trial scope",
        "",
        "The archive inventory below is based only on central-directory filenames. Counts of participant-code matches show naming coverage; they are not an input-completeness or eligibility result.",
        "",
        "| Archive | Files | Matched participants by class | GAIT task files | GAIT subjects by phase | File extensions | Recognized filename terms (counts) |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for item in archive_rows:
        lines.append(
            f"| {item['archive']} | {item['members']} | {item['matched_subjects']} | "
            f"{item['task_files'].get('GAIT', 0)} | {item['gait_subjects_by_phase']} | "
            f"{dict(item['extensions'])} | {dict(item['terms'])} |"
        )
    lines += [
        "",
        "The main candidate rule is all natural GAIT visits across available phases, excluding TX intervention records and static CALIB/MARKER tasks. The dataset protocol records self-selected walking along a 12 m path and back, with two trials per session; use the walking portion rather than the 5-second standing baseline. Keep every visit, trial, and camera view for one subject together in a single fold. This is a natural-gait visit classification task, not a treatment-naive baseline diagnostic validation.",
        "",
        "The baseline-only feasibility check shows four unique ITW CODE values with T0 in the workbook, corresponding to four ITW source records; VIDEO.zip has only three T0+GAIT ITW CODE matches. Thus the strict baseline-only rule cannot support five stratified folds. The selected candidate rule instead uses all non-TX natural GAIT phases, which preserves five-fold class support in the local filename inventory and includes follow-up phases. Keep phase visible for analysis, and do not frame results as treatment-naive baseline diagnosis. The main video/input join has separately confirmed 55 cross-checked candidate subjects (CP=20, ITW=12, TD=23); this is the pre-poseQC candidate set, and fold IDs should be frozen only after poseQC.",
        "",
        "The descriptor's published cohort summary reports 25 TD participants, while this local workbook has 24 distinct TD CODE values; DATASET.zip filename matching identifies all 24 and VIDEO.zip filename matching identifies 23. The main video manifest previously reported 24 TD candidates for its baseline+GAIT rule, one more than this filename-only C0+GAIT match. The later cross-checked candidate set contains 23 TD subjects, consistent with this local filename scan. This manifest follows the local workbook and does not add a synthetic 25th TD subject.",
        "",
        "Protocol references: [LIVE-GaitNeuroKids Data Descriptor, Scientific Data (2026)](https://doi.org/10.1038/s41597-026-08424-4); [author manuscript with the detailed acquisition and naming conventions](https://doi.org/10.21203/rs.3.rs-9471086/v1).",
        "",
        "The manifest contains the source-cohort participants, not the final model-eligible set. No fold IDs were assigned.",
    ]
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "agent2_labels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect anonymized LIVE-GaitNeuroKids subject metadata and write the candidate manifest.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true", help="Write the anonymous source-cohort manifest and aggregate audit.")
    args = parser.parse_args()
    tables = get_subject_tables(args.xlsx)
    if not args.build:
        inspect(args.data_dir, args.xlsx)
        return
    (codes, phases, phase_sets, record_counts, duplicate_visits, manifest) = build_manifest(tables, args.output_dir)
    write_audit(args.data_dir, codes, phases, phase_sets, record_counts, duplicate_visits, args.output_dir)
    print(f"Wrote {manifest.relative_to(args.output_dir)} with {sum(len(v) for v in codes.values())} anonymous source-cohort subjects.")
    print("Wrote aggregate label audit; final eligibility and folds remain for the main model to determine.")


if __name__ == "__main__":
    main()
