#!/usr/bin/env python3
"""Phase 6H: read-only native T1 / tumor-mask geometry and visual QC.

This script checks geometry and creates images for human review. It does NOT
verify tumor segmentation accuracy, Native->MNI registration or atlas mapping.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/media/sophie/Data2T/MRI Data'),
                   help='MRI Data directory containing Child/openneuro_BTC_preop')
    p.add_argument('--out', type=Path, default=Path('../results/native_t1_mask_geometry_qc'))
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--subjects', nargs='+', default=['ALL'], help='ALL or sub-PAT23 ...')
    p.add_argument('--interpolation', choices=['nearest', 'linear'], default='nearest',
                   help='nearest protects binary edges; linear is a sensitivity choice')
    p.add_argument('--no-figures', action='store_true')
    return p


def resolve_subjects(base, selected):
    available = sorted(p.name for p in (base / 'derivatives/tumor_masks').glob('sub-PAT*') if p.is_dir())
    if selected == ['ALL']:
        return available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError('Missing subjects in tumor mask directory: ' + ', '.join(missing))
    return list(dict.fromkeys(selected))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def calc_mask_metrics(arr, affine, threshold):
    from scipy.ndimage import label
    binary = np.isfinite(arr) & (arr > threshold)
    xyz = np.argwhere(binary)
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    count = int(len(xyz))
    centroid = np.full(3, np.nan)
    if count:
        centroid = np.asarray(__import__('nibabel').affines.apply_affine(affine, xyz.mean(axis=0)), dtype=float)
    cc, ncc = label(binary)
    sizes = np.bincount(cc[binary], minlength=ncc + 1)[1:] if ncc else np.array([], dtype=int)
    largest = int(sizes.max()) if len(sizes) else 0
    return {
        'voxels': count,
        'volume_cm3': count * voxel_mm3 / 1000.,
        'centroid_xyz': centroid,
        'components_6n': int(ncc),
        'largest_component_fraction': largest/count if count else np.nan,
        'mask_binary': binary,
    }


def extent_world(img):
    from itertools import product
    import nibabel as nib
    corners = np.asarray(list(product(*[(0, n-1) for n in img.shape[:3]])), dtype=float)
    xyz = nib.affines.apply_affine(img.affine, corners)
    return xyz.min(axis=0), xyz.max(axis=0)


def plot_subject(subject, t1_img, tumor_img, mask_on_t1, centroid_world, out_path, threshold):
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nibabel.processing import resample_from_to

    # Reorient BOTH to RAS in independently correct physical coordinates.
    t1_ras = nib.as_closest_canonical(t1_img)
    mask_ras = nib.as_closest_canonical(mask_on_t1)
    if (t1_ras.shape[:3] != mask_ras.shape[:3] or
            not np.allclose(t1_ras.affine, mask_ras.affine, atol=1e-4)):
        mask_ras = resample_from_to(mask_ras, (t1_ras.shape[:3], t1_ras.affine), order=0)
    data = np.asarray(t1_ras.dataobj, dtype=np.float32)
    tumor = np.asarray(mask_ras.dataobj) > threshold
    finite = data[np.isfinite(data)]
    positive = finite[finite > 0]
    lo, hi = np.percentile(positive if len(positive) else finite, [1, 99]) if len(finite) else (0, 1)
    if hi <= lo: hi = lo + 1
    center = np.rint(nib.affines.apply_affine(np.linalg.inv(t1_ras.affine), centroid_world)).astype(int)
    center = np.clip(center, 0, np.asarray(t1_ras.shape[:3])-1)

    # Correct display axis convention in RAS: each plane is explicitly labeled in WORLD mm.
    views = [
        (0, 'Sagittal', 'Posterior  ←  Anterior (Y)', 'Inferior  →  Superior (Z)'),
        (1, 'Coronal', 'Left  ←  →  Right (X)', 'Inferior  →  Superior (Z)'),
        (2, 'Axial', 'Left  ←  →  Right (X)', 'Posterior  →  Anterior (Y)'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, (axis, title, xl, yl) in zip(axs, views):
        if axis == 0:
            plane = data[center[0], :, :].T
            over = tumor[center[0], :, :].T
            extent = [*sorted([nib.affines.apply_affine(t1_ras.affine, [0,0,0])[1], nib.affines.apply_affine(t1_ras.affine, [0,t1_ras.shape[1]-1,0])[1]]),
                      *sorted([nib.affines.apply_affine(t1_ras.affine, [0,0,0])[2], nib.affines.apply_affine(t1_ras.affine, [0,0,t1_ras.shape[2]-1])[2]])]
        elif axis == 1:
            plane = data[:, center[1], :].T
            over = tumor[:, center[1], :].T
            extent = [*sorted([nib.affines.apply_affine(t1_ras.affine, [0,0,0])[0], nib.affines.apply_affine(t1_ras.affine, [t1_ras.shape[0]-1,0,0])[0]]),
                      *sorted([nib.affines.apply_affine(t1_ras.affine, [0,0,0])[2], nib.affines.apply_affine(t1_ras.affine, [0,0,t1_ras.shape[2]-1])[2]])]
        else:
            plane = data[:, :, center[2]].T
            over = tumor[:, :, center[2]].T
            extent = [*sorted([nib.affines.apply_affine(t1_ras.affine, [0,0,0])[0], nib.affines.apply_affine(t1_ras.affine, [t1_ras.shape[0]-1,0,0])[0]]),
                      *sorted([nib.affines.apply_affine(t1_ras.affine, [0,0,0])[1], nib.affines.apply_affine(t1_ras.affine, [0,t1_ras.shape[1]-1,0])[1]])]
        # For oblique native T1, use voxel-index display; labels are qualitative axis directions.
        ax.imshow(plane, cmap='gray', origin='lower', vmin=lo, vmax=hi)
        if over.any() and not over.all():
            ax.contour(over.astype(float), levels=[0.5], colors='#ffe000', linewidths=1.5)
        ax.set_title(f'{title} | voxel {center[axis]}')
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{subject} | Native T1 and tumor mask >{threshold:g} | MANUAL REVIEW REQUIRED\n'
                 f'Centroid world XYZ: ({centroid_world[0]:.1f}, {centroid_world[1]:.1f}, {centroid_world[2]:.1f}) mm', fontsize=12)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parser().parse_args()
    if not np.isfinite(args.threshold):
        raise SystemExit('ERROR: --threshold must be finite')
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
        import scipy  # noqa
        if not args.no_figures:
            import matplotlib  # noqa
    except ImportError as exc:
        raise SystemExit('Missing dependency: ' + str(exc) + '\nInstall in mri env: conda install -c conda-forge nibabel scipy matplotlib')

    root = args.root.expanduser().resolve()
    base = root / 'Child/openneuro_BTC_preop'
    if not base.is_dir():
        raise SystemExit(f'Input not found: {base}')
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / 'figures'
    if not args.no_figures: figdir.mkdir(exist_ok=True)
    subjects = resolve_subjects(base, args.subjects)
    rows, failures = [], []
    for sub in subjects:
        t1_path = base / 'origin' / sub / 'ses-preop' / 'anat' / f'{sub}_ses-preop_T1w.nii.gz'
        mask_path = base / 'derivatives' / 'tumor_masks' / sub / 'anat' / f'{sub}_space_T1_label-tumor.nii'
        try:
            if not t1_path.is_file(): raise FileNotFoundError(f'T1 missing: {t1_path}')
            if not mask_path.is_file(): raise FileNotFoundError(f'Mask missing: {mask_path}')
            t1, mask = nib.load(str(t1_path)), nib.load(str(mask_path))
            if len(t1.shape) != 3 or len(mask.shape) != 3:
                raise ValueError('Expected 3D T1 and 3D mask')
            same_grid = bool(t1.shape == mask.shape and np.allclose(t1.affine, mask.affine, atol=1e-4))
            # Nearest is appropriate for candidate binary masks, but only for QC, never overwrite.
            order = 0 if args.interpolation == 'nearest' else 1
            mask_on_t1 = resample_from_to(mask, (t1.shape[:3], t1.affine), order=order)
            orig = calc_mask_metrics(np.asarray(mask.dataobj), mask.affine, args.threshold)
            resampled = calc_mask_metrics(np.asarray(mask_on_t1.dataobj), t1.affine, args.threshold)
            if not orig['voxels'] or not resampled['voxels']:
                raise ValueError(f'Empty tumor mask at threshold >{args.threshold}')
            shift = float(np.linalg.norm(orig['centroid_xyz'] - resampled['centroid_xyz']))
            ratio = resampled['volume_cm3'] / orig['volume_cm3']
            t1lo, t1hi = extent_world(t1)
            masklo, maskhi = extent_world(mask)
            bbox_overlap = np.maximum(0, np.minimum(t1hi, maskhi) - np.maximum(t1lo, masklo))
            bbox_size = np.maximum(0, maskhi - masklo)
            bbox_overlap_ratio = float(np.prod(bbox_overlap) / np.prod(bbox_size)) if np.all(bbox_size > 0) else 0.0
            row = {
                'subject': sub, 'native_t1_path': str(t1_path), 'native_mask_path': str(mask_path),
                't1_orientation': ''.join(nib.aff2axcodes(t1.affine)),
                'mask_orientation': ''.join(nib.aff2axcodes(mask.affine)),
                'same_shape_before': t1.shape == mask.shape,
                'same_affine_before': bool(np.allclose(t1.affine, mask.affine, atol=1e-4)),
                'same_grid_before': same_grid, 'same_grid_after': True,
                'native_mask_voxels': orig['voxels'], 'resampled_voxels': resampled['voxels'],
                'native_volume_cm3': round(orig['volume_cm3'], 6),
                'resampled_volume_cm3': round(resampled['volume_cm3'], 6),
                'resampled_to_native_volume_ratio': round(ratio, 6),
                'centroid_shift_mm_after_resample': round(shift, 6),
                'centroid_x_mm': round(resampled['centroid_xyz'][0], 4),
                'centroid_y_mm': round(resampled['centroid_xyz'][1], 4),
                'centroid_z_mm': round(resampled['centroid_xyz'][2], 4),
                'components_6n': resampled['components_6n'],
                'largest_component_fraction': round(resampled['largest_component_fraction'], 6),
                'world_bbox_overlap_fraction': round(bbox_overlap_ratio, 6),
                'geometry_review_flag': ('REVIEW' if shift > 2 or ratio < 0.8 or ratio > 1.2 or bbox_overlap_ratio < 0.9 else 'NO_AUTOMATIC_RED_FLAG'),
                'manual_anatomical_review': 'PENDING',
                'manual_review_notes': '',
                'native_to_mni_registration': 'UNVERIFIED',
                'tumor_dk68_burden_authorized': 'NO',
            }
            if not args.no_figures:
                path = figdir / f'{sub}_native_T1_mask_gt{args.threshold:g}_UNVERIFIED.png'
                plot_subject(sub, t1, mask, mask_on_t1, resampled['centroid_xyz'], path, args.threshold)
            rows.append(row)
            print(f'OK {sub}: native={orig["volume_cm3"]:.3f} cm3, resampled={resampled["volume_cm3"]:.3f} cm3, shift={shift:.3f} mm; {row["geometry_review_flag"]}', flush=True)
        except Exception as exc:
            failures.append({'subject': sub, 'error': str(exc)})
            print(f'FAIL {sub}: {exc}', file=sys.stderr, flush=True)

    fields = ['subject', 'native_t1_path', 'native_mask_path', 't1_orientation', 'mask_orientation',
              'same_shape_before', 'same_affine_before', 'same_grid_before', 'same_grid_after',
              'native_mask_voxels', 'resampled_voxels', 'native_volume_cm3', 'resampled_volume_cm3',
              'resampled_to_native_volume_ratio', 'centroid_shift_mm_after_resample',
              'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm', 'components_6n',
              'largest_component_fraction', 'world_bbox_overlap_fraction', 'geometry_review_flag',
              'manual_anatomical_review', 'manual_review_notes', 'native_to_mni_registration',
              'tumor_dk68_burden_authorized']
    write_csv(out / 'native_geometry_metrics.csv', rows, fields)
    write_csv(out / 'processing_failures.csv', failures, ['subject', 'error'])
    report = {
        'phase': '6H', 'subjects_requested': len(subjects), 'processed': len(rows),
        'failed': len(failures), 'threshold': args.threshold, 'interpolation': args.interpolation,
        'figures_created': 0 if args.no_figures else len(rows),
        'automatic_geometry_review_flags': sum(r['geometry_review_flag'] == 'REVIEW' for r in rows),
        'manual_review_status': 'PENDING for every subject',
        'native_to_mni_registration': 'UNVERIFIED',
        'tumor_dk68_burden_authorized': False,
    }
    (out / 'native_geometry_qc_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = (f'BTC NATIVE T1 / MASK GEOMETRY QC — PHASE 6H\n'
               f'=======================================================\n'
               f'Subjects: {len(subjects)} | processed: {len(rows)} | failed: {len(failures)}\n'
               f'Threshold: >{args.threshold:g} | interpolation: {args.interpolation}\n'
               f'Figures: {report["figures_created"]}\n'
               f'Automated geometry flags: {report["automatic_geometry_review_flags"]}\n'
               f'ANATOMICAL VISUAL REVIEW: PENDING\n'
               f'NATIVE -> MNI REGISTRATION: UNVERIFIED\n'
               f'TUMOR-DK68 BURDEN: NOT AUTHORIZED\n'
               f'Outputs: native_geometry_metrics.csv; processing_failures.csv; native_geometry_qc_summary.json\n')
    (out / 'native_geometry_qc_summary.txt').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
