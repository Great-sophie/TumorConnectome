#!/usr/bin/env python3
"""Phase 6G: candidate anatomy/atlas/tumor spatial AUDIT (not registration validation).

Outputs fixed-world-coordinate three-plane overlays at each tumor centroid,
image-space QC metrics, one-row-per-subject manual review sheet, and provenance
checklist. Never assigns PASS automatically or computes regional tumor burden.

Requires: numpy, scipy, pandas, nibabel, matplotlib.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Parent of Child/openneuro_BTC_preop')
    p.add_argument('--out', type=Path, default=Path('../results/tumor_atlas_registration_audit'))
    p.add_argument('--atlas', type=Path, default=None)
    p.add_argument('--reference', type=Path, required=True, help='Candidate template T1 image; NOT native patient T1')
    p.add_argument('--reference-source', default='', help='Exact source/version for the reference')
    p.add_argument('--btc-mni-template', default='', help='Documented BTC registration target/template (not guessed)')
    p.add_argument('--btc-transform-evidence', default='', help='Path or citation to BTC native-to-MNI transform provenance')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--subjects', nargs='*', help='Optional sub-PATXX IDs; default all 25')
    p.add_argument('--allow-reference-resample', action='store_true', help='For exploratory visualization only')
    p.add_argument('--allow-atlas-resample', action='store_true', help='For exploratory visualization only')
    p.add_argument('--no-figures', action='store_true')
    return p.parse_args()


def grid_equal(a, b, tol=1e-3):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=tol, rtol=0)


def to_target(img, target, order, permit, descriptor):
    if grid_equal(img, target):
        return img, False
    if not permit:
        raise ValueError(f'{descriptor} grid/affine differs; explicit resample flag required')
    from nibabel.processing import resample_from_to
    return resample_from_to(img, (target.shape[:3], target.affine), order=order), True


def cut(arr, axis, idx):
    # fixed voxel index in target mask grid; transpose to oriented display consistently
    return np.rot90(np.take(arr, idx, axis=axis))


def robust_gray(a):
    v = a[np.isfinite(a) & (a > 0)]
    hi = float(np.percentile(v, 99.5)) if v.size else 1.0
    return max(hi, 1e-6)


def contour(ax, binary, color, width):
    if np.any(binary) and not np.all(binary):
        ax.contour(binary.astype(float), levels=[0.5], colors=[color], linewidths=width)


def make_fig(ref, atlas, tissue, centroid_ijk, subject, out, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inds = [int(np.clip(round(c), 0, n-1)) for c, n in zip(centroid_ijk, tissue.shape)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    titles = ['Axis 0', 'Axis 1', 'Axis 2']
    gray_hi = robust_gray(ref)
    for j, (axis, idx) in enumerate(zip(range(3), inds)):
        background = cut(ref, axis, idx)
        region = cut(atlas, axis, idx) > 0
        tumor = cut(tissue, axis, idx)
        for i in range(2):
            ax = axs[i, j]
            ax.imshow(background, cmap='gray', origin='lower', vmin=0, vmax=gray_hi, interpolation='nearest')
            if i == 1:
                contour(ax, region, '#27c4fc', 0.6)
                overlay = np.ma.masked_where(~tumor, tumor.astype(float))
                ax.imshow(overlay, cmap='autumn', vmin=0, vmax=1, alpha=0.5, origin='lower', interpolation='nearest')
                contour(ax, tumor, '#ffdd00', 1.4)
            ax.set_title(f'{titles[j]} voxel {idx} | '+('reference T1' if i == 0 else f'DK68 outline + tumor > {threshold:g}'))
            ax.set_axis_off()
    fig.suptitle(f'{subject} | CANDIDATE SPATIAL OVERLAY — NOT REGISTERED/VERIFIED', fontsize=14)
    fig.text(.5, .015, 'Reference, atlas and tumor shown on same resampled voxel grid; coordinate agreement NOT established by this figure.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=[0,.025,1,.955])
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    a = cli()
    import nibabel as nib
    root = a.root.expanduser().resolve()
    out = a.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    masks = root/'Child/openneuro_BTC_preop/derivatives/tumor_masks'
    atlas_path = a.atlas or root/'Child/TumorConnectome/results/dk68_build/DK68_BTCgrid_CANDIDATE_UNVERIFIED.nii.gz'
    atlas_path = atlas_path.expanduser().resolve()
    reference_path = a.reference.expanduser().resolve()
    for kind, path in [('masks root', masks), ('atlas', atlas_path), ('reference', reference_path)]:
        if not path.exists():
            raise SystemExit(f'Missing {kind}: {path}')
    if '_ses-preop_T1w' in reference_path.name or '/origin/sub-' in str(reference_path):
        raise SystemExit('ERROR: Reference appears to be an individual native T1, not a common-space template.')
    if not (0 <= a.threshold <= 1):
        raise SystemExit('Threshold must be in [0,1].')
    candidates = sorted(masks.glob('sub-PAT*/anat/*_space_MNI_label-tumor.nii*'))
    if a.subjects:
        allow = set(a.subjects)
        candidates = [p for p in candidates if p.parent.parent.name in allow]
    if not candidates:
        raise SystemExit(f'No MNI tumor masks at {masks}')
    reference_img = nib.load(str(reference_path))
    atlas_img = nib.load(str(atlas_path))
    if len(reference_img.shape) != 3 or len(atlas_img.shape) != 3:
        raise SystemExit('Reference and atlas must be 3D.')
    records, failures = [], []
    resampled_reference = resampled_atlas = False
    for p in candidates:
        subject = p.parent.parent.name
        try:
            img = nib.load(str(p))
            if len(img.shape) != 3:
                raise ValueError('Tumor mask not 3D')
            ref_img, res_ref = to_target(reference_img, img, 1, a.allow_reference_resample, 'Reference')
            atl_img, res_atlas = to_target(atlas_img, img, 0, a.allow_atlas_resample, 'Atlas')
            resampled_reference |= res_ref
            resampled_atlas |= res_atlas
            ref = np.asarray(ref_img.dataobj, dtype=np.float32)
            atlas = np.asarray(atl_img.dataobj)
            vals = np.asarray(img.dataobj, dtype=np.float32)
            tumor = np.isfinite(vals) & (vals > a.threshold)
            if not np.any(tumor):
                raise ValueError(f'Empty mask at threshold >{a.threshold}')
            if not np.all(np.isfinite(img.affine)) or abs(np.linalg.det(img.affine[:3,:3])) < 1e-10:
                raise ValueError('Invalid affine')
            coords = np.argwhere(tumor)
            mean_ijk = coords.mean(axis=0)
            centroid_xyz = nib.affines.apply_affine(img.affine, mean_ijk)
            volume_cm3 = len(coords) * abs(np.linalg.det(img.affine[:3,:3])) / 1000
            labels, n = label(tumor)
            sizes = np.bincount(labels[tumor], minlength=n+1)[1:]
            lc_fraction = float(sizes.max()/len(coords))
            mids = nib.affines.apply_affine(img.affine, coords)[:,0]
            left = int(np.count_nonzero(mids < -1.5))
            right = int(np.count_nonzero(mids > 1.5))
            medial = len(coords)-left-right
            # Tissue/support fraction is only a loose warning: tumor can be extra-axial.
            # Reference intensity >0 is not a brain mask and MUST NOT be interpreted as one.
            ref_support = np.isfinite(ref) & (ref > 0.01 * robust_gray(ref))
            support_fraction = float(np.mean(ref_support[tumor]))
            # DK68 is cortical-only, and lack of tumor overlap is not a registration failure.
            atlas_overlap_fraction = float(np.mean((atlas > 0)[tumor]))
            fig_path = figures/f'{subject}_tumor_DK68_T1_CANDIDATE_UNVERIFIED.png'
            if not a.no_figures:
                make_fig(ref, atlas, tumor, mean_ijk, subject, fig_path, a.threshold)
            records.append(dict(subject=subject, mask_path=str(p), threshold=a.threshold,
                                volume_cm3=volume_cm3, center_x_mm=centroid_xyz[0],
                                center_y_mm=centroid_xyz[1], center_z_mm=centroid_xyz[2],
                                left_fraction=left/len(coords), right_fraction=right/len(coords),
                                midline_fraction=medial/len(coords), components=n,
                                largest_component_fraction=lc_fraction,
                                reference_positive_support_fraction=support_fraction,
                                dk68_cortical_overlap_fraction=atlas_overlap_fraction,
                                reference_resampled=res_ref, atlas_resampled=res_atlas,
                                image=str(fig_path) if not a.no_figures else '',
                                manual_review='PENDING', manual_reviewer='',
                                manual_date='', manual_anatomical_location='',
                                manual_laterality='', manual_notes=''))
        except Exception as e:
            failures.append(dict(subject=subject, mask_path=str(p), error=f'{type(e).__name__}: {e}'))
    df = pd.DataFrame(records)
    df.to_csv(out/'spatial_candidate_metrics_and_manual_review.csv', index=False)
    pd.DataFrame(failures, columns=['subject','mask_path','error']).to_csv(out/'processing_failures.csv', index=False)
    provenance = dict(
        btc_mni_template=a.btc_mni_template or 'UNKNOWN / NEED PRIMARY DOCUMENTATION',
        btc_native_to_mni_transform_evidence=a.btc_transform_evidence or 'UNKNOWN / NEED PRIMARY DOCUMENTATION',
        candidate_reference_path=str(reference_path), candidate_reference_source=a.reference_source or 'NOT RECORDED',
        candidate_atlas_path=str(atlas_path), reference_resampled_for_visualization=bool(resampled_reference),
        atlas_resampled_for_visualization=bool(resampled_atlas),
        reference_matches_btc_template_independently='UNVERIFIED',
        tumor_to_btc_mni_transform_provenance='UNVERIFIED',
        atlas_to_btc_mni_transform_provenance='UNVERIFIED',
        atlas_reference_anatomical_alignment='NEEDS MANUAL REVIEW',
        clinical_region_location_agreement='NEEDS MANUAL REVIEW',
        registration_status='UNVERIFIED', regional_burden_authorized=False,
        caveats=[
            'Matching voxel grids/affines and visual overlays do not verify template identity or transformation provenance.',
            'Reference-positive fraction is NOT a brain-overlap or registration-accuracy metric.',
            'Cortical DK68 overlap fraction is NOT regional tumor burden and is NOT a registration quality score.',
            'Manual review, BTC registration-target identity, transforms, and threshold acceptance are mandatory.',
            'Prior BTC masks may have been interpolated; do not infer anatomical correctness from volume concordance.'
        ])
    (out/'registration_provenance_checklist.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    lines = [
        'BTC PHASE 6G — CANDIDATE TUMOR / ATLAS / T1 SPATIAL AUDIT',
        '='*65,
        f'Masks found: {len(candidates)} | processed: {len(records)} | failed: {len(failures)}',
        f'Tumor display threshold: >{a.threshold:g} (candidate, NOT APPROVED)',
        f'Figures generated: {len(records) if not a.no_figures else 0}',
        f'Candidate atlas: {atlas_path}',
        f'Candidate anatomical reference: {reference_path}',
        f'Reference resampled for display: {resampled_reference}',
        f'Atlas resampled for display: {resampled_atlas}',
        'BTC template identity / transforms: NOT INDEPENDENTLY VERIFIED',
        'ANATOMICAL REGISTRATION: UNVERIFIED',
        'TUMOR–DK68 BURDEN: NOT AUTHORIZED',
        'Required: original BTC space/transform records, template identity, clinician/manual QC.',
        'Files: spatial_candidate_metrics_and_manual_review.csv; registration_provenance_checklist.json; processing_failures.csv'
    ]
    txt='\n'.join(lines)+'\n'
    (out/'registration_audit_summary.txt').write_text(txt)
    print(txt)

if __name__ == '__main__':
    main()
