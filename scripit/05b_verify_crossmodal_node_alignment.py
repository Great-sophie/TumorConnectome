#!/usr/bin/env python3
"""BTC Phase 5B: cross-modal DK68 node alignment audit (read-only).

Validates *observed* SC.zip centres order, SCthrAn vs ZIP weights,
FC ROI_ID_table FreeSurfer cortical IDs vs SC names, ROI time-series
extraction and Pearson FC reconstruction. All records get PASS/FAIL status.

Caution: matching ROI label IDs does not independently prove the correctness
of subject-to-template registration or authorize voxel-level tumor mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

# Standard FreeSurfer aparc/Desikan-Killiany cortex ID -> region name.
# 1000+code = lh; 2000+code = rh. 1004/2004 is absent in DK cortical table.
DK_CODES = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35]
DK_NAMES = [
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
    'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
    'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
    'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
    'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
    'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
    'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
    'temporalpole', 'transversetemporal', 'insula',
]
ID_TO_NAME = {1000 + code: 'lh_' + name for code, name in zip(DK_CODES, DK_NAMES)}
ID_TO_NAME.update({2000 + code: 'rh_' + name for code, name in zip(DK_CODES, DK_NAMES)})
EXPECTED_NAMES = ['lh_' + n for n in DK_NAMES] + ['rh_' + n for n in DK_NAMES]


def safe_load(path: Path, keys: list[str]):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    obj = loadmat(path)
    for key in keys:
        if key not in obj:
            raise KeyError(f'{path.name} missing variable {key}')
    return obj


def centre_names(z: ZipFile):
    return [line.split()[0] for line in z.read('centres.txt').decode('utf-8-sig').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def ordered_ids(table: np.ndarray):
    if table.ndim != 2 or table.shape[1] < 2 or table.shape[0] < 69:
        raise ValueError(f'Invalid ROI_ID_table shape {table.shape}')
    # Published compFC.m selects MATLAB 45:78,80:113 for a 113-row table.
    # For another table length, do NOT guess a new selection rule.
    if table.shape[0] != 113:
        raise ValueError(f'Unexpected ROI_ID_table rows: {table.shape[0]} (expected 113)')
    inds = np.r_[44:78, 79:113]
    ids = table[inds, 1]
    if not np.isfinite(ids).all() or not np.array_equal(ids, np.rint(ids)):
        raise ValueError('Noninteger/nonfinite FreeSurfer labels in ROI_ID_table')
    return inds, ids.astype(int)


def audit_one(zip_path: Path, cohort: str, reference: list[str] | None,
              rtol: float, atol: float):
    directory = zip_path.parent
    row = dict(cohort=cohort, subject=directory.parent.name, session=directory.name,
               scan_dir=str(directory), status='FAIL', sc_name_match=False,
               reference_name_match=False, sc_weights_match=False,
               fc_name_match=False, timeseries_match=False, fc_reconstructed=False,
               max_sc_abs_difference='', max_fc_abs_difference='',
               first_mismatch='', error='')
    try:
        with ZipFile(zip_path) as z:
            sc_names = centre_names(z)
            weights = np.loadtxt(BytesIO(z.read('weights.txt')))
        row['sc_name_match'] = len(sc_names) == 68 and len(set(sc_names)) == 68 and sc_names == EXPECTED_NAMES
        row['reference_name_match'] = reference is None or sc_names == reference
        if not row['sc_name_match']:
            row['first_mismatch'] = next((f'idx {i}: {a} != {b}' for i, (a, b) in
                                          enumerate(zip(sc_names, EXPECTED_NAMES)) if a != b),
                                         f'SC label count {len(sc_names)}')
        mat = safe_load(directory / 'SCthrAn.mat', ['SCthrAn'])['SCthrAn']
        if mat.shape != (68, 68) or weights.shape != (68, 68):
            raise ValueError(f'SC shapes: {mat.shape}, {weights.shape}')
        if not np.isfinite(mat).all() or not np.isfinite(weights).all():
            raise ValueError('SC contains nonfinite entries')
        row['max_sc_abs_difference'] = float(np.max(np.abs(mat - weights)))
        row['sc_weights_match'] = bool(np.allclose(mat, weights, rtol=rtol, atol=atol))

        data = safe_load(directory / 'FC.mat',
                         ['ROI_ID_table', 'FC_cc_DK68', 'FC_cc'])
        idx, ids = ordered_ids(data['ROI_ID_table'])
        fc_names = [ID_TO_NAME.get(int(k), f'UNKNOWN_{k}') for k in ids]
        row['fc_name_match'] = fc_names == sc_names
        if not row['fc_name_match'] and not row['first_mismatch']:
            row['first_mismatch'] = next((f'idx {i}: SC {a} / FC {b} (ID {int(k)})'
                                          for i, (a, b, k) in enumerate(zip(sc_names, fc_names, ids))
                                          if a != b), 'FC/SC label count mismatch')
        fc = data['FC_cc_DK68']
        all_fc = data['FC_cc']
        if fc.shape != (68, 68) or all_fc.shape != (113, 113):
            raise ValueError(f'FC shapes {fc.shape}, {all_fc.shape}')
        if not np.isfinite(fc).all() or not np.isfinite(all_fc).all():
            raise ValueError('FC contains nonfinite entries')
        fc_slice = all_fc[np.ix_(idx, idx)]
        row['fc_submatrix_match'] = bool(np.allclose(fc, fc_slice, rtol=rtol, atol=atol))
        # Dataset-specific ROI variable names, e.g. PAT01T1_ROIts vs PAT01T1_ROIts_DK68
        # Load the original 113-ROI series and its published 68-ROI derivative.
        full_data = loadmat(directory / 'FC.mat')
        roi_keys = [k for k in full_data if k.endswith('_ROIts') and not k.startswith('__')]
        if len(roi_keys) != 1:
            raise ValueError(f'Expected one *_ROIts time series, found {roi_keys}')
        key = roi_keys[0]
        key68 = key + '_DK68'
        if key68 not in full_data:
            raise KeyError(f'Missing time series {key68}')
        ts = full_data[key]
        ts68 = full_data[key68]
        if ts.ndim != 2 or ts.shape[1] != 113 or ts68.shape != (ts.shape[0], 68):
            raise ValueError(f'ROI TS shapes: {ts.shape}, {ts68.shape}')
        if not np.isfinite(ts).all() or not np.isfinite(ts68).all():
            raise ValueError('ROI time series contains nonfinite entries')
        row['timeseries_match'] = bool(np.allclose(ts68, ts[:, idx], rtol=rtol, atol=atol))
        rebuilt = np.corrcoef(ts68, rowvar=False)
        if not np.isfinite(rebuilt).all():
            raise ValueError('Nonfinite FC reconstruction')
        row['max_fc_abs_difference'] = float(np.max(np.abs(rebuilt - fc)))
        row['fc_reconstructed'] = bool(np.allclose(rebuilt, fc, rtol=rtol, atol=atol))
        checks = ('sc_name_match', 'reference_name_match', 'sc_weights_match',
                  'fc_name_match', 'fc_submatrix_match', 'timeseries_match', 'fc_reconstructed')
        row['status'] = 'PASS' if all(row.get(c, False) for c in checks) else 'FAIL'
        if row['status'] == 'FAIL' and not row['error']:
            row['error'] = 'Failed checks: ' + ', '.join(c for c in checks if not row.get(c, False))
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('/media/sophie/Data2T/MRI Data'))
    ap.add_argument('--out', type=Path, default=Path('../results/crossmodal_node_validation'))
    ap.add_argument('--rtol', type=float, default=1e-7)
    ap.add_argument('--atol', type=float, default=1e-10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = [('preop', args.root / 'Child/openneuro_BTC_preop/derivatives/TVB'),
             ('postop', args.root / 'Child/openneuro BTC_postop/derivatives/TVB')]
    scans = [(z, group) for group, folder in roots
             for z in sorted(folder.glob('sub-*/ses-*/SC.zip'))]
    if not scans:
        ap.error(f'No SC.zip files found under {args.root}; verify --root')
    reference = None
    for z, _ in scans:
        try:
            with ZipFile(z) as archive:
                names = centre_names(archive)
            if len(names) == 68 and len(set(names)) == 68:
                reference = names
                break
        except Exception:
            pass
    records = [audit_one(z, group, reference, args.rtol, args.atol) for z, group in scans]
    extra_fc = [(group, f) for group, folder in roots
                for f in sorted(folder.glob('sub-*/ses-*/FC.mat'))
                if not (f.parent / 'SC.zip').exists()]
    for group, f in extra_fc:
        records.append(dict(cohort=group, subject=f.parent.parent.name, session=f.parent.name,
                            scan_dir=str(f.parent), status='FAIL',
                            error='FC.mat without SC.zip; cannot cross-validate'))
    columns = ['cohort', 'subject', 'session', 'scan_dir', 'status', 'sc_name_match',
               'reference_name_match', 'sc_weights_match', 'fc_name_match',
               'fc_submatrix_match', 'timeseries_match', 'fc_reconstructed',
               'max_sc_abs_difference', 'max_fc_abs_difference', 'first_mismatch', 'error']
    with (args.out / 'crossmodal_alignment_per_scan.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    failures = [r for r in records if r['status'] != 'PASS']
    with (args.out / 'crossmodal_alignment_failures.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(failures)
    summary = {'scan_records': len(records), 'sc_zip_records': len(scans),
               'extra_fc_without_sc': len(extra_fc), 'passed': len(records)-len(failures),
               'failed': len(failures), 'reference_sc_labels': reference,
               'scope': 'SC/FC node IDs + matrix/time-series reconstruction; NOT tumor spatial registration'}
    (args.out / 'crossmodal_alignment_summary.json').write_text(json.dumps(summary, indent=2))
    report = ('BTC CROSS-MODAL DK68 VALIDATION — PHASE 5B\n'
              + '='*58 + '\n'
              + f"Scan records: {summary['scan_records']} (SC.zip: {len(scans)})\n"
              + f"PASS: {summary['passed']} | FAIL: {summary['failed']}\n"
              + f"FC without SC.zip: {len(extra_fc)}\n"
              + 'NOTE: Node-identity validation does not validate MNI tumor-mask registration.\n')
    if failures:
        report += '\nFAILURES:\n' + '\n'.join(
            f"{r['cohort']}/{r['subject']}/{r['session']}: {r.get('error','')} {r.get('first_mismatch','')}"
            for r in failures[:40]) + '\n'
    (args.out / 'crossmodal_alignment_summary.txt').write_text(report)
    print(report)
    print('Outputs:', args.out.resolve())


if __name__ == '__main__':
    main()
