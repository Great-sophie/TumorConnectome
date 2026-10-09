#!/usr/bin/env python3
"""Integrate BTC pre/post clinical metadata with the Phase-2 imaging cohort.

Reads source BIDS TSVs without modifying them. No imputation, diagnosis inference,
participant-level public release, or clinical improvement claims. Outputs are local
and require privacy review before publication.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

COGNITIVE = ['RVP_A', 'RVP_probhit', 'RTI_simpleRT_mean',
             'SOC_prob_minmoves', 'SSP_spanlength']
# Sign of improvement for published instruments is not assumed.
MISSING = {'', 'n/a', 'na', 'nan', 'null', 'not available', 'unknown', 'none', '-'}

def clean(x):
    value = str(x if x is not None else '').strip()
    return '' if value.lower() in MISSING else value

def normalize_id(x):
    s = clean(x).upper()
    s = s if s.startswith('SUB-') else 'SUB-' + s
    if not re.fullmatch(r'SUB-(PAT|CON)\d+', s):
        return ''
    return 'sub-' + s[4:]

def parse_float(s):
    x = clean(s)
    if not x:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def read_csv(path, delimiter=','):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f'No header: {path}')
        rows = [{k.strip(): clean(v) for k, v in row.items() if k is not None}
                for row in reader]
    return rows

def read_index(path, delimiter, id_col='participant_id'):
    rows = read_csv(path, delimiter)
    result = {}
    for row in rows:
        sid = normalize_id(row.get(id_col))
        if not sid:
            raise ValueError(f'Invalid participant ID in {path}: {row.get(id_col)}')
        if sid in result:
            raise ValueError(f'Duplicate participant: {sid} in {path}')
        result[sid] = row
    return result

def save_csv(path, rows, columns):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def is_true(v):
    return str(v).strip().lower() in ('true', '1', 'yes')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/media/sophie/Data2T/MRI Data'))
    p.add_argument('--cohort', type=Path, required=True, help='Phase-2 analysis_cohort.csv')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preop', default='Child/openneuro_BTC_preop')
    p.add_argument('--postop', default='Child/openneuro BTC_postop')
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    cohort_path = args.cohort.expanduser().resolve()
    out = args.out.expanduser().resolve()
    pre_path = (root / args.preop / 'participants.tsv').resolve()
    post_path = (root / args.postop / 'participants.tsv').resolve()
    for f in (cohort_path, pre_path, post_path):
        if not f.is_file(): p.error(f'Required file not found: {f}')
    for src in (pre_path.parent, post_path.parent):
        if out == src or src in out.parents:
            p.error('Output folder must be outside source datasets')
    original = read_index(cohort_path, ',', 'subject')
    pre = read_index(pre_path, '\t')
    post = read_index(post_path, '\t')
    expected = {'age', 'sex', 'tumor type & grade', 'tumor location', 'fmri TR', *COGNITIVE}
    for dataset, name in [(pre, 'preop'), (post, 'postop')]:
        fields = set(next(iter(dataset.values())).keys()) if dataset else set()
        missing = expected - fields
        if missing: p.error(f'{name} TSV missing required columns: {sorted(missing)}')
    if not set(original).issubset(pre):
        p.error(f'Phase-2 IDs absent from preop participants.tsv: {sorted(set(original)-set(pre))}')
    extras = sorted(set(post)-set(original))
    if extras:
        print('WARNING: postop subjects outside imaging cohort:', extras)
    # Avoid overwriting the Phase-2 imaging cohort; export a new corrected table.
    details, paired, exclusions = [], [], []
    for sid in sorted(original):
        base = original[sid]; a = pre.get(sid, {}); b = post.get(sid, {})
        group = 'patient' if sid.startswith('sub-PAT') else 'control'
        baseline = is_true(base.get('baseline_eligible'))
        longitudinal = is_true(base.get('longitudinal_eligible'))
        a_dx, b_dx = clean(a.get('tumor type & grade')), clean(b.get('tumor type & grade'))
        dx_mismatch = bool(a_dx and b_dx and a_dx.casefold() != b_dx.casefold())
        a_sex, b_sex = clean(a.get('sex')), clean(b.get('sex'))
        a_age, b_age = parse_float(a.get('age')), parse_float(b.get('age'))
        row = {
            'subject': sid, 'group': group, 'baseline_scfc_eligible': baseline,
            'longitudinal_scfc_eligible': longitudinal,
            'clinical_preop_available': bool(a), 'clinical_postop_available': bool(b),
            'clinical_paired': bool(a and b),
            'age_pre': a_age if a_age is not None else '',
            'age_post': b_age if b_age is not None else '',
            'sex_pre': a_sex, 'sex_post': b_sex,
            'diagnosis_pre': a_dx, 'diagnosis_post': b_dx,
            'diagnosis_changed_across_tables': dx_mismatch,
            'tumor_size_pre_cm3': parse_float(a.get('tumor size (cub cm)')) if group == 'patient' else '',
            'tumor_location_pre': clean(a.get('tumor location')) if group == 'patient' else '',
            'tumor_location_post': clean(b.get('tumor location')) if group == 'patient' else '',
            'fmri_tr_pre': parse_float(a.get('fmri TR')) or '',
            'fmri_tr_post': parse_float(b.get('fmri TR')) or '',
            'interval_days': '', 'interval_status': 'not_available_in_source_tables',
            'dk68_node_order_validated': False,
            'tumor_mask_alignment_validated': False,
            'final_paper_eligible': 'pending_review',
        }
        flags = []
        if dx_mismatch: flags.append('diagnosis_discordant')
        if a_sex and b_sex and a_sex.casefold() != b_sex.casefold(): flags.append('sex_discordant')
        if a_age is not None and b_age is not None and abs(a_age - b_age) > 3: flags.append('age_difference_over_3_years_review')
        if group == 'patient' and not a_dx: flags.append('patient_diagnosis_missing')
        if group == 'control' and a_dx.lower() not in ('', 'none', 'control', 'healthy', 'health control'):
            flags.append('control_diagnosis_review')
        if not longitudinal: flags.append('not_scfc_paired')
        if not b: flags.append('clinical_followup_missing')
        for metric in COGNITIVE:
            x, y = parse_float(a.get(metric)), parse_float(b.get(metric))
            row[f'{metric}_pre'] = x if x is not None else ''
            row[f'{metric}_post'] = y if y is not None else ''
            row[f'{metric}_delta_post_minus_pre'] = (y-x) if x is not None and y is not None else ''
            row[f'{metric}_paired_observed'] = x is not None and y is not None
            row[f'{metric}_paired_scfc_eligible'] = longitudinal and x is not None and y is not None
            if x is not None and y is not None:
                paired.append({'subject': sid, 'group': group, 'metric': metric,
                               'pre': x, 'post': y, 'delta_post_minus_pre': y-x,
                               'paired_scfc_eligible': longitudinal,
                               'interpretation': 'raw_change_only_not_improvement'})
        row['review_flags'] = ';'.join(flags)
        details.append(row)
        if flags:
            exclusions.append({'subject': sid, 'group': group,
                               'flags': ';'.join(flags),
                               'imaging_longitudinal_eligible': longitudinal})
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / 'clinical_cohort.csv', details, list(details[0]))
    save_csv(out / 'paired_cognitive_outcomes.csv', paired,
             ['subject','group','metric','pre','post','delta_post_minus_pre',
              'paired_scfc_eligible','interpretation'])
    save_csv(out / 'review_flags.csv', exclusions,
             ['subject','group','flags','imaging_longitudinal_eligible'])
    overview = []
    for metric in COGNITIVE:
        for group in ('patient', 'control'):
            subset = [r for r in details if r['group'] == group]
            overview.append({'metric': metric, 'group': group, 'n_baseline_clinical': sum(r[f'{metric}_pre'] != '' for r in subset),
                             'n_clinical_paired': sum(r[f'{metric}_paired_observed'] for r in subset),
                             'n_scfc_clinical_paired': sum(r[f'{metric}_paired_scfc_eligible'] for r in subset)})
    save_csv(out / 'clinical_completeness.csv', overview,
             ['metric','group','n_baseline_clinical','n_clinical_paired','n_scfc_clinical_paired'])
    diagnosis = Counter(r['diagnosis_pre'] or 'missing' for r in details)
    save_csv(out / 'tumor_diagnosis_summary.csv',
             [{'diagnosis_pre': d, 'n': n} for d, n in sorted(diagnosis.items())], ['diagnosis_pre','n'])
    summary = {
        'imaging_cohort_subjects': len(details),
        'preop_clinical_rows': len(pre), 'postop_clinical_rows': len(post),
        'clinical_paired_in_imaging_cohort': sum(r['clinical_paired'] for r in details),
        'scfc_paired': dict(Counter(r['group'] for r in details if r['longitudinal_scfc_eligible'])),
        'cognition_completeness': overview,
        'diagnosis_discordances': sum(r['diagnosis_changed_across_tables'] for r in details),
        'notes': ['Raw cognitive deltas are not signed as improvement/deterioration.',
                  'Anatomical lesion maps and DK68 node order not yet validated.',
                  'Participants may have extra clinical records beyond paired imaging scans.',
                  'Follow-up interval remains unavailable; avoid causal claims.',
                  'Review individual-level outputs for privacy before sharing.']}
    (out / 'clinical_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    report = ["BTC CLINICAL–IMAGING INTEGRATION — PHASE 2B", "="*58,
              f"Preop clinical records: {len(pre)}",
              f"Postop clinical records: {len(post)}",
              f"Imaging cohort IDs: {len(details)}",
              f"Clinical pairs within imaging cohort: {summary['clinical_paired_in_imaging_cohort']}",
              f"SC+FC longitudinal pairs: {summary['scfc_paired']}",
              f"Diagnosis discrepancies to review: {summary['diagnosis_discordances']}",
              '', 'Cognitive completeness (paired with usable longitudinal SC+FC):']
    report += [f"  {r['metric']} / {r['group']}: {r['n_scfc_clinical_paired']}" for r in overview]
    report += ['', 'CAUTION: Changes are post-minus-pre; clinical improvement direction not assigned.',
               'CAUTION: Subject-level tables require privacy review prior to any public GitHub release.',
               'CAUTION: Follow-up interval, DK68 order and tumor atlas alignment remain unverified.']
    (out / 'clinical_summary.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('\n'.join(report))
    print(f'\nWrote 7 output files to {out}')

if __name__ == '__main__':
    main()
