#!/usr/bin/env python3
"""Build a reproducible BTC analysis cohort from Phase-1 QC and BIDS metadata.

Read-only with respect to source data. Does not infer tumor histology from PAT IDs,
assume that follow-up controls underwent surgery, or declare DK68 ordering validated.
Outputs minimal, non-identifying research metadata only. Review outputs before publishing.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ID = re.compile(r'^sub-(PAT|CON)(\d+)$', re.I)
# Only explicit, non-identifying fields are extracted. No names, dates, reports or free text.
ALIASES = {
    'age': ('age', 'age_years', 'age_at_scan', 'age_at_baseline'),
    'sex': ('sex', 'gender'),
    'diagnosis': ('diagnosis', 'diagnostic_group', 'tumor_type', 'tumour_type',
                  'histology', 'histological_type'),
    'interval_days': ('interval_days', 'followup_interval_days', 'time_between_scans_days',
                      'days_since_baseline', 'days_from_baseline'),
}
VALID_TRUE = {'true', '1', 'yes'}
BAD_MISSING = {'', 'n/a', 'na', 'nan', 'none', 'null', 'not available', 'unknown', 'n/a'}


def read_table(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        sample = handle.readline()
        handle.seek(0)
        delim = '\t' if path.suffix.lower() == '.tsv' else ','
        reader = csv.DictReader(handle, delimiter=delim)
        if not reader.fieldnames:
            return [], []
        return list(reader), list(reader.fieldnames)


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def truth(x):
    return str(x).strip().lower() in VALID_TRUE


def normalize_id(val):
    val = str(val or '').strip()
    if val and not val.startswith('sub-') and re.fullmatch(r'(PAT|CON)\d+', val, re.I):
        val = 'sub-' + val
    if not ID.fullmatch(val):
        return None
    group, num = ID.fullmatch(val).groups()
    return f'sub-{group.upper()}{num}'  # preserve leading zeros


def is_missing(x):
    return str(x or '').strip().lower() in BAD_MISSING


def clean_value(field, value):
    value = str(value or '').strip()
    if is_missing(value):
        return ''
    if field in ('age', 'interval_days'):
        try:
            f = float(value)
            limit = 120 if field == 'age' else 36500
            if not (0 <= f <= limit):
                return ''
            return str(f)
        except ValueError:
            return ''
    if field == 'sex':
        v = value.lower()
        return {'m': 'M', 'male': 'M', 'f': 'F', 'female': 'F',
                'other': 'other'}.get(v, '')
    if field == 'diagnosis':
        # Strictly pre-approved diagnostic categories; unknown labels retained only in local source.
        v = re.sub(r'[ _-]+', ' ', value.lower()).strip()
        return {
            'glioma': 'glioma', 'meningioma': 'meningioma',
            'healthy': 'healthy', 'healthy control': 'healthy',
            'control': 'healthy', 'glioblastoma': 'glioblastoma',
            'low grade glioma': 'low_grade_glioma',
            'high grade glioma': 'high_grade_glioma'
        }.get(v, '')
    return ''


def collect_metadata(btc_root, stage, known):
    """Find only explicit structured BIDS participant/session/phenotype TSV files."""
    paths = []
    for path in [btc_root / 'participants.tsv', btc_root / 'sessions.tsv']:
        if path.is_file():
            paths.append(path)
    phenotype = btc_root / 'phenotype'
    if phenotype.is_dir():
        paths.extend(sorted(phenotype.glob('*.tsv')))
    for sub in sorted(btc_root.glob('sub-*')):
        if normalize_id(sub.name) in known:
            paths.extend(sorted(sub.glob('*_sessions.tsv')))
    out = defaultdict(lambda: defaultdict(list))
    inventory = []
    for path in paths:
        try:
            rows, columns = read_table(path)
            idcol = next((c for c in columns if c.lower() in
                          ('participant_id', 'subject_id', 'subject', 'participant')), None)
            # Session files may omit ID; derive only for sub-XYZ/sessions.tsv.
            parent_id = normalize_id(path.parent.name)
            inventory.append({'stage': stage, 'relative_path': str(path.relative_to(btc_root)),
                              'rows': len(rows), 'columns': ';'.join(columns),
                              'subject_key': idcol or ('directory' if parent_id else ''),
                              'status': 'read'})
            for item in rows:
                sid = normalize_id(item.get(idcol)) if idcol else parent_id
                if sid not in known:
                    continue
                for variable, aliases in ALIASES.items():
                    for col in columns:
                        if col.lower() in aliases:
                            val = clean_value(variable, item.get(col))
                            if val:
                                out[sid][variable].append((val, stage, str(path.relative_to(btc_root)), col))
        except (OSError, UnicodeError, csv.Error) as exc:
            inventory.append({'stage': stage, 'relative_path': str(path.relative_to(btc_root)),
                              'rows': '', 'columns': '', 'subject_key': '',
                              'status': f'failed_to_read:{type(exc).__name__}'})
    return out, inventory


def resolve(values):
    unique = sorted({v[0] for v in values})
    if not unique:
        return '', 'missing', ''
    if len(unique) != 1:
        return '', 'conflict', ';'.join(sorted({f'{v[1]}:{v[2]}:{v[3]}' for v in values}))
    sources = ';'.join(sorted({f'{v[1]}:{v[2]}:{v[3]}' for v in values}))
    return unique[0], 'observed', sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('/media/sophie/Data2T/MRI Data'))
    ap.add_argument('--audit', type=Path, required=True, help='Phase-1 audit results folder')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preop', default='Child/openneuro_BTC_preop')
    ap.add_argument('--postop', default='Child/openneuro BTC_postop')
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    pre, post = root / args.preop, root / args.postop
    audit = args.audit.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not pre.is_dir() or not post.is_dir():
        ap.error('BTC preop or postop data directory not found')
    for source in (pre.resolve(), post.resolve()):
        if out == source or source in out.parents:
            ap.error('--out must be outside the source BTC dataset folders')
    audit_file = audit / 'cohort_manifest.csv'
    qc_file = audit / 'connectome_qc.csv'
    if not audit_file.is_file() or not qc_file.is_file():
        ap.error('Missing Phase-1 cohort_manifest.csv or connectome_qc.csv')
    subjects, _ = read_table(audit_file)
    qc, _ = read_table(qc_file)
    known = set()
    for subject in subjects:
        sid = normalize_id(subject.get('subject'))
        if not sid or sid in known:
            ap.error('Invalid or duplicate subject in Phase-1 manifest')
        known.add(sid)
    meta = defaultdict(lambda: defaultdict(list))
    inventories = []
    for stage, dataset in (('preop', pre), ('postop', post)):
        m, inv = collect_metadata(dataset, stage, known)
        inventories.extend(inv)
        for sid in m:
            for k in m[sid]:
                meta[sid][k].extend(m[sid][k])
    qc_index = defaultdict(list)
    for q in qc:
        sid = normalize_id(q.get('subject'))
        if sid not in known:
            ap.error('QC contains unknown subject')
        qc_index[(sid, q.get('session'), q.get('key'))].append(q.get('status', ''))

    def status(sid, session, key):
        entries = qc_index.get((sid, session, key), [])
        return len(entries) == 1 and entries[0] == 'pass'

    output = []
    for s in subjects:
        sid = normalize_id(s['subject'])
        group = 'patient' if 'PAT' in sid else 'control'
        # Mask flags are availability only, never considered validated volume data.
        preok = status(sid, 'preop', 'SCthrAn') and status(sid, 'preop', 'FC_cc_DK68')
        postok = status(sid, 'postop', 'SCthrAn') and status(sid, 'postop', 'FC_cc_DK68')
        deconv_pre = status(sid, 'preop', 'FC_cc_DK68_deconv')
        deconv_post = status(sid, 'postop', 'FC_cc_DK68_deconv')
        row = dict(subject=sid, group=group,
                   preop_valid_sc_fc=preok, postop_valid_sc_fc=postok,
                   preop_deconv_fc_pass=deconv_pre, postop_deconv_fc_pass=deconv_post,
                   baseline_eligible=preok, longitudinal_eligible=preok and postok,
                   tumor_mask_T1_available=truth(s.get('preop_tumor_mask_T1')),
                   tumor_mask_MNI_available=truth(s.get('preop_tumor_mask_MNI')),
                   tumor_mask_validated=False, dk68_node_order_validated=False,
                   final_paper_eligible='pending_manual_review')
        for field in ALIASES:
            val, state, source = resolve(meta[sid][field])
            row[field] = val
            row[field + '_status'] = state
            row[field + '_source'] = source
        row['baseline_exclusion_reason'] = '' if preok else 'missing_or_failed_preop_sc_fc_qc'
        row['longitudinal_exclusion_reason'] = '' if preok and postok else (
            'missing_or_failed_preop_sc_fc_qc' if not preok else 'missing_or_failed_followup_sc_fc_qc')
        row['review_flags'] = ';'.join(x for x in (
            'missing_diagnosis' if group == 'patient' and row['diagnosis_status'] != 'observed' else '',
            'diagnosis_group_mismatch' if row['diagnosis'] == 'healthy' and group == 'patient' or
                row['diagnosis'] not in ('', 'healthy') and group == 'control' else '',
            'age_metadata_conflict' if row['age_status'] == 'conflict' else '',
            'sex_metadata_conflict' if row['sex_status'] == 'conflict' else '',
            'interval_metadata_conflict' if row['interval_days_status'] == 'conflict' else '',
            'mask_not_validated' if group == 'patient' else '',
            'dk68_node_order_not_validated'
        ) if x)
        output.append(row)

    out.mkdir(parents=True, exist_ok=True)
    columns = list(output[0]) if output else []
    write_csv(out / 'analysis_cohort.csv', output, columns)
    write_csv(out / 'metadata_inventory.csv', inventories,
              ['stage', 'relative_path', 'rows', 'columns', 'subject_key', 'status'])
    summary = {
        'n_subjects': len(output),
        'baseline_eligible': dict(Counter(r['group'] for r in output if r['baseline_eligible'])),
        'longitudinal_eligible': dict(Counter(r['group'] for r in output if r['longitudinal_eligible'])),
        'metadata_availability': {field: dict(Counter(r[field + '_status'] for r in output)) for field in ALIASES},
        'diagnosis_in_paired_patients': dict(Counter(r['diagnosis'] or 'unverified' for r in output
                                                    if r['group'] == 'patient' and r['longitudinal_eligible'])),
        'interpretation': ('Eligibility refers only to Phase-1 SC/FC numeric QC. DK68 order, '
                           'mask alignment, clinical labels and MRI acquisition QC are NOT validated.'),
        'publication_warning': ('Do not publish per-subject metadata, raw data paths, dates, '
                                'or source BIDS files without privacy/licensing review.')
    }
    (out / 'cohort_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    text = [
        'BTC PAPER COHORT — PHASE 2', '=' * 48,
        f'Subjects found: {summary["n_subjects"]}',
        f'Baseline SC+FC numerically valid: {summary["baseline_eligible"]}',
        f'Longitudinal SC+FC numerically valid: {summary["longitudinal_eligible"]}',
        'Metadata availability:'
    ] + [f'  {k}: {v}' for k, v in summary['metadata_availability'].items()] + [
        f'Paired patient diagnosis: {summary["diagnosis_in_paired_patients"]}',
        '', 'NOT YET VERIFIED: DK68 node ordering, tumor mask alignment, clinical labels, imaging QC.',
        'Control follow-up is not surgery. PAT prefix does not prove histological subtype.',
        'Subject-level outputs require privacy review before public GitHub release.'
    ]
    (out / 'cohort_summary.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('\n'.join(text))
    print(f'\nResults written to: {out}')


if __name__ == '__main__':
    main()
