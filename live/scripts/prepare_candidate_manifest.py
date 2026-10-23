from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache'
CLASS_TO_INDEX = {'CP': 0, 'ITW': 1, 'TD': 2}

def load(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def main():
    subjects = load(ROOT / 'manifests/subjects.csv')
    videos = load(ROOT / 'manifests/videos.csv')
    crosswalk = {r['source_subject_code'].upper(): r for r in subjects}
    assert len(crosswalk) == len(subjects), 'Duplicate source subject code'
    included, excluded, resolved_aliases = [], [], Counter()
    for row in videos:
        if row['video_kind'] != 'walking':
            excluded.append({'zip_member': row['zip_member'], 'reason': row['video_kind']})
            continue
        raw_code = row['source_subject_code'].upper()
        code = raw_code
        metadata = crosswalk.get(code)
        # Resolve only a zero-padding variant corroborated by the exact folder.
        folder_code = PurePosixPath(row['zip_member']).parts[-2].upper()
        raw_match = re.fullmatch(r'([PS])0*(\d+)', raw_code)
        folder_match = re.fullmatch(r'([PS])0*(\d+)', folder_code)
        if metadata is None and folder_code in crosswalk and raw_match and folder_match:
            if (raw_match[1], int(raw_match[2])) == (folder_match[1], int(folder_match[2])):
                code = folder_code
                metadata = crosswalk[code]
                resolved_aliases[(raw_code, code)] += 1
        if metadata is None:
            excluded.append({'zip_member': row['zip_member'], 'reason': 'source code absent from participant workbook'})
            continue
        assert metadata['class_label'] == row['class_label'], 'Clinical class conflicts across sources'
        if row['visit_hint'] not in {'T0', 'T1', 'T2', 'C0'}:
            excluded.append({'zip_member': row['zip_member'], 'reason': 'non-gait visit or unrecognized phase'})
            continue
        if row['view_hint'] not in {'left', 'right', 'sagittal'}:
            excluded.append({'zip_member': row['zip_member'], 'reason': 'unrecognized camera view'})
            continue
        video_id = hashlib.sha256(row['zip_member'].encode('utf-8')).hexdigest()[:16]
        included.append({
            'video_id': video_id,
            'subject_id': metadata['subject_id'],
            'source_subject_code': code,
            'source_subject_code_original': raw_code,
            'class_label': metadata['class_label'],
            'label': CLASS_TO_INDEX[metadata['class_label']],
            'zip_member': row['zip_member'],
            'video_kind': 'walking',
            'view_hint': row['view_hint'],
            'visit_hint': row['visit_hint'],
            'bytes': row['bytes'],
            'npz_path': str(CACHE / 'video_inputs' / f'{video_id}.npz'),
            'selection_note': 'all available natural GAIT visits; all visits and views grouped by subject',
        })
    included.sort(key=lambda r: (r['class_label'], r['subject_id'], r['zip_member']))
    assert len({r['zip_member'] for r in included}) == len(included)
    assert len({r['video_id'] for r in included}) == len(included)
    subject_class = {r['subject_id']: r['class_label'] for r in included}
    counts = Counter(subject_class.values())
    assert set(counts) == set(CLASS_TO_INDEX) and min(counts.values()) >= 5
    with (ROOT / 'manifests/eligible_videos.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(included[0]))
        writer.writeheader()
        writer.writerows(included)
    views = defaultdict(set)
    for row in included:
        views[row['subject_id']].add(row['view_hint'])
    report = {
        'date': '2026-10-12',
        'scope': 'Metadata and gait-video eligibility; input/pose QC pending',
        'inclusion': 'Natural GAIT recordings from all T0/T1/T2/C0 visits; same participant retains one group across all visits and cameras',
        'baseline_only_rejected_as_primary': 'Only 3 ITW participants have T0 GAIT video; this would not support the user-specified stratified five-fold design',
        'subjects': len(subject_class),
        'subjects_by_class': dict(counts),
        'videos': len(included),
        'videos_by_class': dict(Counter(r['class_label'] for r in included)),
        'videos_by_view': dict(Counter(r['view_hint'] for r in included)),
        'all_subjects_have_all_three_named_views': all(v == {'left', 'right', 'sagittal'} for v in views.values()),
        'folder_backed_code_aliases': [{'filename_code': pair[0], 'metadata_and_folder_code': pair[1], 'videos': count} for pair, count in resolved_aliases.items()],
        'excluded_by_reason': dict(Counter(r['reason'] for r in excluded)),
        'excluded': excluded,
        'folds_frozen': False,
        'publication_metric_available': False,
    }
    (ROOT / 'manifests/eligibility_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'excluded'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
