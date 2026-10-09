#!/usr/bin/env python3
"""BTC Phase 6D: descriptive threshold sensitivity of continuous tumor masks.

No atlas registration, segmentation accuracy or optimal threshold is certified.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

THRESHOLDS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.50)


def measure(data, threshold, voxel_mm3, connectivity=1):
    binary = np.isfinite(data) & (data > threshold)
    structure = ndimage.generate_binary_structure(3, connectivity)
    cc, n = ndimage.label(binary, structure=structure)
    counts = np.bincount(cc[binary], minlength=n + 1)[1:]
    largest = int(counts.max()) if len(counts) else 0
    voxels = int(binary.sum())
    return dict(threshold=float(threshold), voxels=voxels,
                volume_cm3=voxels * voxel_mm3 / 1000.0,
                components=int(n), largest_component_voxels=largest,
                largest_component_fraction=(largest / voxels if voxels else None))


def safe_nifti(path):
    import nibabel as nib
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D image, got {img.shape}")
    data = np.asarray(img.dataobj, dtype=np.float32)
    return img, data


def make_preview(subject, data, out_file, reference_data=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Choose a fixed slice based on the intensity-weighted tumor centre, not separately for each threshold.
    weights = np.clip(np.nan_to_num(data, nan=0, posinf=0, neginf=0), 0, None)
    positive = weights > 0.1
    if not np.any(positive):
        positive = weights > 0.01
    if np.any(positive):
        ijk = np.rint(np.mean(np.argwhere(positive), axis=0)).astype(int)
    else:
        ijk = np.asarray(data.shape) // 2
    z = int(ijk[2])
    base = reference_data if reference_data is not None else data
    base_slice = np.rot90(base[:, :, z])
    valid = np.isfinite(base_slice)
    vmin, vmax = np.percentile(base_slice[valid], [2, 98]) if valid.any() else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), constrained_layout=True)
    for ax, threshold in zip(axes, (0.0, 0.01, 0.1, 0.5)):
        ax.imshow(base_slice, cmap='gray', vmin=vmin, vmax=vmax)
        overlay = np.rot90(np.isfinite(data[:, :, z]) & (data[:, :, z] > threshold))
        ax.imshow(np.ma.masked_where(~overlay, overlay), cmap='autumn', alpha=0.65, interpolation='nearest')
        ax.set_title(f'mask > {threshold:g}')
        ax.axis('off')
    mode = 'T1 reference (same grid)' if reference_data is not None else 'mask intensity (NOT anatomy)'
    fig.suptitle(f'{subject} | axial voxel z={z} | {mode} | UNVERIFIED', fontsize=11)
    fig.savefig(out_file, dpi=140)
    plt.close(fig)
    return z, mode


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description='BTC MNI tumor mask threshold sensitivity, not clinical segmentation validation.')
    p.add_argument('--root', default='/media/sophie/Data2T/MRI Data', help='MRI Data directory')
    p.add_argument('--out', default='../results/tumor_mask_threshold_validation')
    p.add_argument('--thresholds', nargs='+', type=float, default=list(THRESHOLDS))
    p.add_argument('--connectivity', type=int, choices=[1, 2, 3], default=1, help='3D connectivity: 1=6, 2=18, 3=26 neighbors')
    p.add_argument('--make-figures', action='store_true', help='Make one fixed-slice threshold comparison per mask')
    p.add_argument('--reference', help='Optional anatomical image already on EXACT SAME grid as all tumor masks')
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    masks_dir = root / 'Child/openneuro_BTC_preop/derivatives/tumor_masks'
    files = sorted(masks_dir.glob('sub-PAT*/anat/*_space_MNI_label-tumor.nii*'))
    if not files:
        p.error(f'No MNI tumor masks found: {masks_dir}')
    thresholds = sorted(set(args.thresholds))
    if any(not np.isfinite(t) or t < 0 for t in thresholds):
        p.error('Thresholds must be finite and nonnegative.')
    try:
        import nibabel as nib
    except ImportError:
        p.error('Missing nibabel. Install in mri environment: conda install -c conda-forge nibabel')

    reference = None
    if args.reference:
        rimg, reference = safe_nifti(Path(args.reference))
    rows, subject_rows, failures = [], [], []
    figdir = out / 'figures'
    if args.make_figures:
        figdir.mkdir(exist_ok=True)
    for path in files:
        subject = path.parent.parent.name
        try:
            img, data = safe_nifti(path)
            if reference is not None and (rimg.shape != img.shape or not np.allclose(rimg.affine, img.affine, atol=1e-5)):
                raise ValueError('Anatomical reference is not on the exact tumor-mask voxel grid; refusing misleading overlay')
            voxel_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
            if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
                raise ValueError('Invalid voxel dimensions / affine')
            stats = [measure(data, t, voxel_mm3, args.connectivity) for t in thresholds]
            for stat in stats:
                rows.append({'subject': subject, 'mask_path': str(path), 'mask_min': float(np.nanmin(data)),
                             'mask_max': float(np.nanmax(data)), 'voxel_mm3': voxel_mm3, **stat})
            positive = np.isfinite(data) & (data > 0)
            refstat = next((s for s in stats if np.isclose(s['threshold'], 0.1)), None)
            summary = dict(subject=subject, status='PASS_DESCRIPTIVE', positive_voxels=int(positive.sum()),
                           max_value=float(np.nanmax(data)), threshold_0p1_volume_cm3=(refstat or {}).get('volume_cm3'),
                           threshold_0p1_components=(refstat or {}).get('components'),
                           figure='')
            if args.make_figures:
                outfile = figdir / f'{subject}_threshold_comparison_UNVERIFIED.png'
                z, mode = make_preview(subject, data, outfile, reference)
                summary['figure'] = str(outfile)
                summary['axial_z_index'] = z
                summary['background_kind'] = mode
            # A native-space tumor mask is useful only as a *separate* quantitative QC comparison.
            t1_path = path.parent / path.name.replace('_space_MNI_', '_space_T1_')
            if t1_path.exists():
                native_img, native = safe_nifti(t1_path)
                summary['native_T1_mask_exists'] = True
                summary['native_T1_grid_shape'] = 'x'.join(map(str, native.shape))
                summary['native_T1_positive_volume_cm3'] = float(np.sum(np.isfinite(native) & (native > 0)) * abs(np.linalg.det(native_img.affine[:3, :3])) / 1000)
                summary['native_T1_mask_max'] = float(np.nanmax(native))
            else:
                summary['native_T1_mask_exists'] = False
            subject_rows.append(summary)
        except Exception as e:
            failures.append({'subject': subject, 'mask_path': str(path), 'error': f'{type(e).__name__}: {e}'})
    write_csv(out / 'threshold_metrics_per_subject.csv', rows,
              ['subject', 'mask_path', 'threshold', 'voxels', 'volume_cm3', 'components',
               'largest_component_voxels', 'largest_component_fraction', 'voxel_mm3', 'mask_min', 'mask_max'])
    write_csv(out / 'subject_qc_summary.csv', subject_rows,
              ['subject', 'status', 'positive_voxels', 'max_value', 'threshold_0p1_volume_cm3',
               'threshold_0p1_components', 'native_T1_mask_exists', 'native_T1_grid_shape',
               'native_T1_positive_volume_cm3', 'native_T1_mask_max', 'axial_z_index', 'background_kind', 'figure'])
    write_csv(out / 'processing_failures.csv', failures, ['subject', 'mask_path', 'error'])
    stats = {'masks_found': len(files), 'processed': len(subject_rows), 'failed': len(failures),
             'thresholds': thresholds, 'connectivity': args.connectivity,
             'figures_created': sum(bool(s['figure']) for s in subject_rows),
             'anatomical_registration': 'UNVERIFIED', 'threshold_approved': False,
             'tumor_to_DK68_burden_authorized': False,
             'warnings': ['Binarized volumes are threshold-dependent sensitivity measures, not verified ground truth.',
                          'Connectivity uses scipy.ndimage 3D neighborhoods (default 6-connected).',
                          'When no reference is given, preview backgrounds display MASK INTENSITY, NOT anatomy.',
                          'Native-space and MNI volumes cannot be taken as a spatial-registration check.',
                          'Anatomical registration and individual tumor topology require independent/manual QC.']}
    (out / 'threshold_validation_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    lines = ['BTC TUMOR MASK THRESHOLD SENSITIVITY — PHASE 6D', '=' * 58,
             f"Masks: {len(files)} | processed: {len(subject_rows)} | FAIL: {len(failures)}",
             f"Thresholds: {thresholds} | connectivity: {args.connectivity} (6/18/26 neighbors)",
             f"Figures: {stats['figures_created']}", 'ANATOMICAL REGISTRATION: UNVERIFIED',
             'THRESHOLD: NOT APPROVED', 'TUMOR–DK68 BURDEN: NOT AUTHORIZED']
    for bad in failures:
        lines.append(f"FAIL {bad['subject']}: {bad['error']}")
    lines += ['', 'Outputs: threshold_metrics_per_subject.csv; subject_qc_summary.csv; processing_failures.csv; threshold_validation_summary.json']
    result = '\n'.join(lines) + '\n'
    (out / 'threshold_validation_summary.txt').write_text(result, encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
