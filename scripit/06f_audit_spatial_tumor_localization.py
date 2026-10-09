#!/usr/bin/env python3
"""BTC Phase 6F: descriptive tumor localization and threshold stability QC.

Outputs are NOT registration certificates, clinical localization validation,
or authorization for tumor-to-DK68 burden. Never modifies source images.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage

DEFAULT_THRESHOLDS = (0.01, 0.1, 0.2, 0.5)


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_clinical(path):
    result = {}
    if path.exists():
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                key = row.get('subject', '').strip()
                if key:
                    result[key] = row
    return result


def clinical_hemi(s):
    s = (s or '').lower().strip()
    left = bool(re.search(r'\b(left|lt|lhs)\b|左', s))
    right = bool(re.search(r'\b(right|rt|rhs)\b|右', s))
    bilat = bool(re.search(r'\b(bilateral|both sides|midline)\b|双侧|中线', s))
    if bilat or (left and right):
        return 'BILATERAL_OR_MIDLINE'
    if left:
        return 'LEFT'
    if right:
        return 'RIGHT'
    return 'NOT_SPECIFIED'


def mask_metrics(data, affine, threshold, structure):
    binary = np.isfinite(data) & (data > threshold)
    total = int(binary.sum())
    voxel_vol = abs(np.linalg.det(affine[:3, :3]))
    empty = dict(voxels=0, volume_cm3=0.0, components=0,
                 largest_component_fraction=None, center_x_mm=None,
                 center_y_mm=None, center_z_mm=None, x_min_mm=None,
                 x_max_mm=None, left_fraction=None, right_fraction=None,
                 midline_fraction=None, laterality='EMPTY', centroid_voxel=None)
    if total == 0:
        return empty
    # Transform voxel-centre coordinates into NIfTI world / RAS+ coordinates.
    coords = np.argwhere(binary)
    xyz = coords @ affine[:3, :3].T + affine[:3, 3]
    centre = xyz.mean(axis=0)
    left = int(np.count_nonzero(xyz[:, 0] < -1))
    right = int(np.count_nonzero(xyz[:, 0] > 1))
    middle = total - left - right
    if left / total >= 0.9:
        laterality = 'LEFT_DOMINANT'
    elif right / total >= 0.9:
        laterality = 'RIGHT_DOMINANT'
    elif middle / total >= 0.5:
        laterality = 'MIDLINE_DOMINANT'
    else:
        laterality = 'BILATERAL_OR_CROSS_MIDLINE'
    _, count = ndimage.label(binary, structure=structure)
    if count:
        # size of largest 3D connected component
        labels, _ = ndimage.label(binary, structure=structure)
        sizes = np.bincount(labels[binary], minlength=count+1)[1:]
        biggest = int(sizes.max())
    else:
        biggest = 0
    return dict(voxels=total, volume_cm3=total * voxel_vol / 1000.,
                components=int(count), largest_component_fraction=biggest / total,
                center_x_mm=float(centre[0]), center_y_mm=float(centre[1]),
                center_z_mm=float(centre[2]), x_min_mm=float(xyz[:, 0].min()),
                x_max_mm=float(xyz[:, 0].max()), left_fraction=left/total,
                right_fraction=right/total, midline_fraction=middle/total,
                laterality=laterality, centroid_voxel=np.mean(coords, axis=0).tolist())


def make_figure(subject, data, img, per_threshold, thresholds, outpath, reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    use = next((r for r in per_threshold if abs(r['threshold']-0.5)<1e-8 and r['voxels']), None)
    if use is None:
        use = next((r for r in reversed(per_threshold) if r['voxels']), None)
    if use:
        xyz = np.array([use['center_x_mm'], use['center_y_mm'], use['center_z_mm'], 1.0])
        ijk = np.rint(np.linalg.inv(img.affine) @ xyz)[:3].astype(int)
    else:
        ijk = np.array(img.shape)//2
    ijk = np.clip(ijk, 0, np.array(img.shape)-1)
    # Same three spatial locations at every threshold; display voxel axes, not asserted clinical radiological orientation.
    base = reference if reference is not None else data
    fig, axes = plt.subplots(len(thresholds), 3, figsize=(10, 2.7*len(thresholds)), squeeze=False)
    for row, threshold in enumerate(thresholds):
        binary = np.isfinite(data) & (data > threshold)
        for axis in range(3):
            sl = np.take(base, ijk[axis], axis=axis)
            b = np.take(binary, ijk[axis], axis=axis)
            ax = axes[row, axis]
            finite = sl[np.isfinite(sl)]
            low, high = np.percentile(finite, [2, 98]) if finite.size else (0,1)
            if high <= low:
                high = low + 1
            ax.imshow(np.rot90(sl), cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            bm = np.rot90(b)
            ax.imshow(np.ma.masked_where(~bm, bm), cmap='autumn', alpha=.65, interpolation='nearest')
            ax.set_title(f't>{threshold:g}; axis={axis}, index={ijk[axis]}', fontsize=9)
            ax.axis('off')
    label = 'ANATOMICAL reference, EXACT SAME GRID' if reference is not None else 'MASK INTENSITY BACKGROUND (NOT ANATOMY)'
    fig.suptitle(f'{subject}: fixed slices across thresholds | {label} | SPATIAL UNVERIFIED', fontsize=10)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(outpath, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='BTC Phase 6F: descriptive MNI tumor localization and threshold stability QC')
    ap.add_argument('--root', default='/media/sophie/Data2T/MRI Data')
    ap.add_argument('--out', default='../results/tumor_spatial_localization_qc')
    ap.add_argument('--clinical', default=None, help='clinical_cohort.csv; defaults to TumorConnectome/results/clinical/clinical_cohort.csv')
    ap.add_argument('--thresholds', nargs='+', type=float, default=list(DEFAULT_THRESHOLDS))
    ap.add_argument('--baseline-threshold', type=float, default=0.5, help='Reference threshold for centroid displacement ONLY, not segmentation ground truth')
    ap.add_argument('--connectivity', type=int, choices=(1,2,3), default=1)
    ap.add_argument('--reference', default=None, help='Optional anatomical T1 already EXACTLY on BTC mask grid; no auto-registration')
    ap.add_argument('--make-figures', action='store_true')
    args = ap.parse_args()
    try:
        import nibabel as nib
    except ImportError:
        ap.error('Install nibabel in mri env: conda install -c conda-forge nibabel')
    if not args.thresholds or any(not np.isfinite(v) or v < 0 for v in args.thresholds):
        ap.error('Thresholds must be finite, nonnegative numbers')
    if not np.isfinite(args.baseline_threshold) or args.baseline_threshold < 0:
        ap.error('--baseline-threshold must be finite and nonnegative')
    thresholds = sorted(set([float(x) for x in args.thresholds] + [args.baseline_threshold]))
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    files = sorted((root/'Child/openneuro_BTC_preop/derivatives/tumor_masks').glob('sub-PAT*/anat/*_space_MNI_label-tumor.nii*'))
    if not files: ap.error('No preop MNI tumor masks found under --root')
    clinical_path = Path(args.clinical) if args.clinical else root/'Child/TumorConnectome/results/clinical/clinical_cohort.csv'
    clinical = read_clinical(clinical_path)
    reference = refimg = None
    if args.reference:
        refimg = nib.load(str(Path(args.reference).expanduser()))
        reference = np.asarray(refimg.dataobj, dtype=np.float32)
    structure = ndimage.generate_binary_structure(3, args.connectivity)
    observations=[]; cases=[]; problems=[]
    fig_dir = out/'figures'
    if args.make_figures: fig_dir.mkdir(exist_ok=True)
    for file in files:
        subject=file.parent.parent.name
        try:
            img=nib.load(str(file))
            if len(img.shape)!=3: raise ValueError(f'Mask not 3D: {img.shape}')
            data=np.asarray(img.dataobj, dtype=np.float32)
            if not np.isfinite(img.affine).all(): raise ValueError('Invalid affine')
            if reference is not None and (refimg.shape!=img.shape or not np.allclose(refimg.affine,img.affine,atol=1e-5)):
                raise ValueError('Reference must be on exact mask grid; refusing misleading overlay')
            entry=clinical.get(subject,{})
            location=str(entry.get('tumor_location_pre','') or '').strip()
            hemi=clinical_hemi(location)
            rows=[]
            for t in thresholds:
                measurement=mask_metrics(data,img.affine,t,structure)
                r={k:v for k,v in measurement.items() if k!='centroid_voxel'}
                r.update(subject=subject,threshold=t,mask_path=str(file),clinical_location=location,
                         clinical_hemisphere=hemi,clinical_comparison='NOT_ASSESSED')
                rows.append(r)
            baseline=next(r for r in rows if r['threshold']==args.baseline_threshold)
            for r in rows:
                a=[r.get('center_x_mm'),r.get('center_y_mm'),r.get('center_z_mm')]
                b=[baseline.get('center_x_mm'),baseline.get('center_y_mm'),baseline.get('center_z_mm')]
                r['centroid_shift_vs_baseline_mm']=(float(np.linalg.norm(np.asarray(a)-np.asarray(b))) if all(x is not None for x in a+b) else None)
                r['volume_ratio_vs_baseline']=(r['volume_cm3']/baseline['volume_cm3'] if baseline['volume_cm3'] else None)
                # Conservative flag only: unilateral categorical notes contradicted by >90% in opposite hemisphere.
                if hemi in ('LEFT','RIGHT') and r['laterality'] in ('LEFT_DOMINANT','RIGHT_DOMINANT'):
                    r['clinical_comparison']='HEMISPHERE_CONSISTENT' if hemi in r['laterality'] else 'HEMISPHERE_REVIEW'
                elif hemi=='NOT_SPECIFIED':
                    r['clinical_comparison']='CLINICAL_HEMISPHERE_NOT_SPECIFIED'
                else:
                    r['clinical_comparison']='AMBIGUOUS_MANUAL_REVIEW'
            observations.extend(rows)
            cases.append(dict(subject=subject,clinical_location=location,clinical_hemisphere=hemi,
                baseline_threshold=args.baseline_threshold,baseline_volume_cm3=baseline['volume_cm3'],
                baseline_components=baseline['components'],baseline_laterality=baseline['laterality'],
                baseline_centroid_x_mm=baseline['center_x_mm'],baseline_centroid_y_mm=baseline['center_y_mm'],
                baseline_centroid_z_mm=baseline['center_z_mm'],
                max_centroid_shift_mm=max((r['centroid_shift_vs_baseline_mm'] for r in rows if r['centroid_shift_vs_baseline_mm'] is not None),default=None),
                spatial_review='NOT_REVIEWED',registration='UNVERIFIED'))
            if args.make_figures:
                make_figure(subject,data,img,rows,thresholds,fig_dir/f'{subject}_spatial_thresholds_UNVERIFIED.png',reference)
        except Exception as exc:
            problems.append({'subject':subject,'path':str(file),'error':f'{type(exc).__name__}: {exc}'})
    perfields=['subject','mask_path','threshold','voxels','volume_cm3','components','largest_component_fraction',
               'center_x_mm','center_y_mm','center_z_mm','x_min_mm','x_max_mm','left_fraction','right_fraction',
               'midline_fraction','laterality','centroid_shift_vs_baseline_mm','volume_ratio_vs_baseline',
               'clinical_location','clinical_hemisphere','clinical_comparison']
    casefields=['subject','clinical_location','clinical_hemisphere','baseline_threshold','baseline_volume_cm3',
                'baseline_components','baseline_laterality','baseline_centroid_x_mm','baseline_centroid_y_mm',
                'baseline_centroid_z_mm','max_centroid_shift_mm','spatial_review','registration']
    write_csv(out/'localization_metrics_per_threshold.csv',observations,perfields)
    write_csv(out/'subject_localization_review.csv',cases,casefields)
    write_csv(out/'processing_failures.csv',problems,['subject','path','error'])
    summary={'scans_found':len(files),'processed':len(cases),'failed':len(problems),'thresholds':thresholds,
        'reference_threshold_for_centroid_comparison':args.baseline_threshold,
        'clinical_metadata_found':sum(bool(clinical.get(x['subject'],{})) for x in cases),
        'figures':len(cases) if args.make_figures else 0,'anatomical_reference_used':bool(args.reference),
        'registration_validated':False,'tumor_DK68_burden_authorized':False,
        'warnings':['Coordinates are NIfTI world (RAS+), not a demonstrated MNI template identity.',
                    'Clinical tumor location text is not a ground-truth anatomical segmentation; only coarse hemisphere flags are generated.',
                    'Continuous mask threshold choice and anatomical overlay require human QC.',
                    'No atlas overlap, regional burden or registration certification is performed.']}
    (out/'localization_audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    message=(f"BTC TUMOR SPATIAL LOCALIZATION — PHASE 6F\n{'='*55}\n"
             f"Masks: {len(files)} | processed: {len(cases)} | FAIL: {len(problems)}\n"
             f"Thresholds: {thresholds} | baseline for centroid shifts: >{args.baseline_threshold:g}\n"
             f"Clinical records matched: {summary['clinical_metadata_found']}\n"
             f"Figures: {summary['figures']} | anatomical reference: {'provided' if reference is not None else 'NOT PROVIDED'}\n"
             'ANATOMICAL REGISTRATION: UNVERIFIED\nTHRESHOLD: NOT APPROVED\nTUMOR–DK68 BURDEN: NOT AUTHORIZED\n'
             'Outputs: localization_metrics_per_threshold.csv; subject_localization_review.csv; processing_failures.csv\n')
    (out/'localization_audit_summary.txt').write_text(message,encoding='utf-8')
    print(message)

if __name__=='__main__': main()
