#!/usr/bin/env python3
"""Phase 6G2: RAS-oriented, world-coordinate tumor/atlas/reference visual QC.

Visualization only. It does NOT establish that BTC MNI and FreeSurfer cvs_avg35
represent the same anatomical template and does NOT authorize regional burden.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DEFAULT_ROOT = '/media/sophie/Data2T/MRI Data'
ATLAS_REL = ('Child/TumorConnectome/results/dk68_build/'
             'DK68_BTCgrid_CANDIDATE_UNVERIFIED.nii.gz')
REF_REL = ('Child/TumorConnectome/references/atlases/freesurfer_templates/'
           'usr/local/freesurfer/8.2.0/subjects/cvs_avg35_inMNI152/mri/orig.mgz')
MASK_DIR_REL = 'Child/openneuro_BTC_preop/derivatives/tumor_masks'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument('--atlas', type=Path, default=None, help='Candidate DK68 atlas; default derived from root')
    ap.add_argument('--reference', type=Path, default=None, help='FreeSurfer template T1; default derived from root')
    ap.add_argument('--mask-dir', type=Path, default=None, help='Directory containing sub-PAT*/anat/*space_MNI_label-tumor.nii*')
    ap.add_argument('--subjects', nargs='+', default=['sub-PAT23'], help='Subject IDs, or ALL')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', type=Path, default=Path('../results/ras_anatomical_overlay_qc'))
    ap.add_argument('--window', type=float, nargs=2, default=None, metavar=('LOW','HIGH'), help='Optional reference T1 intensity window')
    return ap.parse_args()


def resolved(root: Path, override: Path | None, rel: str):
    return override.expanduser().resolve() if override is not None else (root / rel).resolve()


def canonical_grid(mask_img):
    # NIfTI affine, not array index convention, defines world orientation.
    return nib.as_closest_canonical(mask_img)


def reference_to_mask(ref_img, mask_ras):
    # Resample using world coordinates, rather than swapping array axes.
    return resample_from_to(ref_img, (mask_ras.shape[:3], mask_ras.affine), order=1)


def atlas_to_mask(atlas_img, mask_ras):
    return resample_from_to(atlas_img, (mask_ras.shape[:3], mask_ras.affine), order=0)


def extract_2d(arr, axis, idx):
    # Canonical RAS axes: 0=L->R, 1=P->A, 2=I->S.
    # Display with anatomical superior at top in sagittal/coronal;
    # anterior at top in axial, with image columns L->R in world coords.
    if axis == 0:  # horizontal screen = P->A; vertical = I->S
        return arr[idx, :, :].T
    if axis == 1:  # horizontal = L->R; vertical = I->S
        return arr[:, idx, :].T
    return arr[:, :, idx].T  # horizontal = L->R; vertical = P->A


def slice_extent(affine, shape, axis):
    # Use world-space limits so axes labels are numerically meaningful.
    coords = [0, 1, 2]
    coords.remove(axis)
    horiz, vert = (1, 2) if axis == 0 else ((0, 2) if axis == 1 else (0, 1))
    def endpoint(voxel_axis, index):
        p = np.zeros(3)
        p[voxel_axis] = index
        return nib.affines.apply_affine(affine, p)[voxel_axis]
    x0, x1 = endpoint(horiz, -0.5), endpoint(horiz, shape[horiz] - 0.5)
    y0, y1 = endpoint(vert, -0.5), endpoint(vert, shape[vert] - 0.5)
    return (x0, x1, y0, y1)


def plot_subject(subject, ref, tumor, atlas, mask_img, center_world, threshold, dst, limits=None):
    center_vox = np.rint(nib.affines.apply_affine(np.linalg.inv(mask_img.affine), center_world)).astype(int)
    center_vox = np.clip(center_vox, 0, np.array(tumor.shape) - 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white', constrained_layout=True)
    views = [('Sagittal', 0, 'Posterior', 'Anterior', 'Inferior', 'Superior'),
             ('Coronal', 1, 'Left', 'Right', 'Inferior', 'Superior'),
             ('Axial', 2, 'Left', 'Right', 'Posterior', 'Anterior')]
    if limits is None:
        vals = ref[np.isfinite(ref) & (ref > 0)]
        if vals.size:
            vmin, vmax = np.percentile(vals, [1, 99.5])
        else:
            vmin, vmax = 0, 1
    else:
        vmin, vmax = limits
    for col, (name, axis, xmin, xmax, ymin, ymax) in enumerate(views):
        idx = int(center_vox[axis])
        extent = slice_extent(mask_img.affine, tumor.shape, axis)
        base = extract_2d(ref, axis, idx)
        tum = extract_2d(tumor, axis, idx)
        dk = extract_2d(atlas, axis, idx)
        for row in (0, 1):
            ax = axes[row, col]
            ax.imshow(base, origin='lower', extent=extent, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_aspect('equal')
            ax.set_xlabel(f'{xmin} → {xmax} (world mm)', fontsize=9)
            ax.set_ylabel(f'{ymin} → {ymax} (world mm)', fontsize=9)
            ax.tick_params(labelsize=8)
        axes[0, col].set_title(f'{name} | voxel {idx} | reference T1', fontsize=11)
        # Contours at the boundaries of nonzero masks; no filled atlas or tumor that obscures T1.
        axes[1, col].contour(dk > 0, levels=[0.5], origin='lower', extent=extent,
                             colors=['deepskyblue'], linewidths=0.7)
        axes[1, col].contour(tum > threshold, levels=[0.5], origin='lower', extent=extent,
                             colors=['gold'], linewidths=1.6)
        axes[1, col].set_title(f'{name} | DK68 boundary (cyan) | tumor > {threshold:g} (yellow)', fontsize=10)
        # Mark centroid only when viewing slice through centroid.
        if axis == 0:
            xy = [center_world[1], center_world[2]]
        elif axis == 1:
            xy = [center_world[0], center_world[2]]
        else:
            xy = [center_world[0], center_world[1]]
        axes[1, col].plot(xy[0], xy[1], marker='+', markersize=9, color='magenta', markeredgewidth=1.3)
    xyz = ', '.join(f'{v:+.1f}' for v in center_world)
    fig.suptitle(f'{subject} | RAS world-space display | centroid XYZ=({xyz}) mm | UNVERIFIED', fontsize=14)
    fig.text(0.5, 0.005, 'CANDIDATE visualization only. Same world-coordinate grid does NOT verify BTC/FreeSurfer template identity or registration.',
             ha='center', va='bottom', fontsize=9, color='darkred')
    fig.savefig(dst, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return center_vox


def main():
    args = parse_args()
    if not np.isfinite(args.threshold):
        raise SystemExit('ERROR: --threshold must be finite')
    root = args.root.expanduser().resolve()
    atlas_path = resolved(root, args.atlas, ATLAS_REL)
    ref_path = resolved(root, args.reference, REF_REL)
    mask_dir = resolved(root, args.mask_dir, MASK_DIR_REL)
    out = args.out.expanduser().resolve()
    for label_, path in [('atlas', atlas_path), ('reference', ref_path), ('mask directory', mask_dir)]:
        if not path.exists():
            raise SystemExit(f'ERROR: missing {label_}: {path}')
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    ref_img = nib.load(str(ref_path))
    atlas_img = nib.load(str(atlas_path))
    requested = args.subjects
    if len(requested) == 1 and requested[0].upper() == 'ALL':
        requested = sorted(x.name for x in mask_dir.glob('sub-PAT*') if x.is_dir())
    records = []
    failures = []
    for subject in requested:
        matches = sorted((mask_dir / subject / 'anat').glob(f'{subject}_space_MNI_label-tumor.nii*'))
        if not matches:
            failures.append({'subject':subject, 'reason':'Missing MNI tumor mask'})
            continue
        try:
            img = canonical_grid(nib.load(str(matches[0])))
            data = np.asarray(img.dataobj, dtype=np.float32)
            binary = np.isfinite(data) & (data > args.threshold)
            nvox = int(binary.sum())
            if nvox == 0:
                raise ValueError(f'Empty tumor mask at threshold >{args.threshold}')
            center_vox = np.argwhere(binary).mean(axis=0)
            center_world = nib.affines.apply_affine(img.affine, center_vox)
            ref_res = reference_to_mask(ref_img, img)
            atlas_res = atlas_to_mask(atlas_img, img)
            ref = np.asarray(ref_res.dataobj, dtype=np.float32)
            dk = np.rint(np.asarray(atlas_res.dataobj)).astype(np.int32)
            unique = np.unique(dk)
            unexpected = unique[(unique != 0) & ~(((unique >= 1001) & (unique <= 1035)) | ((unique >= 2001) & (unique <= 2035)))]
            if unexpected.size:
                print(f'WARNING {subject}: atlas contains non-cortical labels {unexpected.tolist()[:12]}', file=sys.stderr)
            figpath = figures / f'{subject}_RAS_anatomical_overlay_UNVERIFIED.png'
            index = plot_subject(subject, ref, data, dk, img, center_world, args.threshold, figpath, args.window)
            records.append({'subject':subject, 'threshold':args.threshold, 'tumor_voxels':nvox,
                            'tumor_volume_cm3':nvox*abs(np.linalg.det(img.affine[:3,:3]))/1000.,
                            'centroid_x_mm':float(center_world[0]), 'centroid_y_mm':float(center_world[1]),
                            'centroid_z_mm':float(center_world[2]),
                            'RAS_voxel_i':int(index[0]), 'RAS_voxel_j':int(index[1]), 'RAS_voxel_k':int(index[2]),
                            'reference_resampled_by_world_affine':True, 'btc_registration_verified':False,
                            'template_identity_verified':False, 'figure':str(figpath)})
            print(f'OK {subject}: tumor={nvox} voxels, center XYZ={np.round(center_world,2).tolist()}')
        except Exception as exc:
            failures.append({'subject':subject,'reason':str(exc)})
            print(f'FAIL {subject}: {exc}', file=sys.stderr)
    with (out / 'subject_ras_overlay_qc.csv').open('w', newline='') as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (out / 'processing_failures.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'reason']); writer.writeheader(); writer.writerows(failures)
    summary = {'requested':len(requested),'processed':len(records),'failures':len(failures),
               'threshold':args.threshold,'reference':str(ref_path),'atlas':str(atlas_path),
               'anatomical_registration':'UNVERIFIED','tumor_DK68_burden':'NOT AUTHORIZED',
               'notes':['RAS standardized by nibabel.as_closest_canonical',
                        'Reference interpolation uses affine in world coordinates',
                        'Atlas interpolation nearest-neighbor only',
                        'Atlas-reference alignment does not verify BTC registration',
                        'Visual plausibility is not evidence of template identity']}
    (out / 'ras_overlay_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    (out / 'ras_overlay_summary.txt').write_text(
        f"PHASE 6G2 RAS ANATOMICAL OVERLAY\nProcessed {len(records)}/{len(requested)}; failed {len(failures)}\n"
        f"Threshold > {args.threshold}\nANATOMICAL REGISTRATION: UNVERIFIED\nTUMOR–DK68 BURDEN: NOT AUTHORIZED\n"
        f"Figures: {figures}\n", encoding='utf8')
    print((out / 'ras_overlay_summary.txt').read_text())
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
