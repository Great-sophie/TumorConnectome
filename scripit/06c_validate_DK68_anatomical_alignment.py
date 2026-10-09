#!/usr/bin/env python3
"""Phase 6C: anatomical *review* of a candidate BTC-grid DK68 atlas.

This tool NEVER certifies anatomical registration or computes tumor burden.
It produces reproducible geometry checks and anatomical overlay images for
human review. The provided anatomical reference must be in the intended MNI
space; a native-space T1w is NOT an acceptable reference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DK_CODES = (1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
DK_IDS = {1000+x for x in DK_CODES} | {2000+x for x in DK_CODES}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('/media/sophie/Data2T/MRI Data'))
    p.add_argument('--atlas', type=Path, default=None, help='Previously generated DK68_BTCgrid_CANDIDATE_UNVERIFIED.nii.gz')
    p.add_argument('--reference', type=Path, help='Anatomical MNI T1 reference (NOT individual native T1)')
    p.add_argument('--reference-space', default='', help='Exact provenance/template identifier, if verified')
    p.add_argument('--atlas-space', default='', help='Exact atlas space identifier, if verified')
    p.add_argument('--reference-source', default='', help='Anatomical reference source/version/URL/DOI')
    p.add_argument('--allow-reference-resample', action='store_true', help='Resample reference to BTC grid for visualization only')
    p.add_argument('--out', type=Path, default=Path('../results/dk68_anatomical_qc'))
    p.add_argument('--subjects', nargs='*', help='Optional subset e.g. sub-PAT01 sub-PAT02')
    p.add_argument('--max-panels', type=int, default=25)
    return p


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            c=f.read(1024*1024)
            if not c: break
            h.update(c)
    return h.hexdigest()


def geom(img):
    import nibabel as nib
    return {'shape': list(map(int,img.shape[:3])),
            'orientation': ''.join(nib.aff2axcodes(img.affine)),
            'zooms_mm': [float(x) for x in img.header.get_zooms()[:3]],
            'affine': np.round(img.affine,7).tolist()}


def same_grid(a,b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine,b.affine,atol=1e-4,rtol=0)


def robust_norm(volume):
    z=np.asarray(volume, dtype=np.float32)
    vals=z[np.isfinite(z) & (z > 0)]
    if len(vals)<50: vals=z[np.isfinite(z)]
    if len(vals)==0: return np.zeros_like(z)
    lo,hi=np.percentile(vals,[1,99])
    if hi <= lo: return np.zeros_like(z)
    return np.clip((z-lo)/(hi-lo),0,1)


def make_panel(atlas, reference, mask, out, subject):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # Choose slice at positive tumor weighted centroid for visibility;
    # this is a DISPLAY choice, not a tumor segmentation threshold.
    mm=np.asarray(mask,dtype=np.float32)
    valid=np.isfinite(mm) & (mm>0)
    if valid.any():
        coords=np.argwhere(valid)
        center=np.rint(np.mean(coords,axis=0)).astype(int)
    else:
        center=np.array(mm.shape)//2
    fig, axes=plt.subplots(2,3,figsize=(14,9), constrained_layout=True)
    fig.suptitle(f'{subject} | DK68 + tumor + MNI anatomical reference | UNVERIFIED',fontsize=14)
    for k,label in enumerate(('Sagittal','Coronal','Axial')):
        sl=int(np.clip(center[k],0,mm.shape[k]-1))
        def cut(arr):
            return np.rot90(np.take(arr,sl,axis=k))
        back=cut(reference)
        a=cut(atlas)
        m=cut(mm)
        # top row anatomical alone, bottom overlay
        for row in (0,1):
            ax=axes[row,k]
            ax.imshow(back,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            if row==1:
                # transparency on DK tissue; low opacity to show cortical contours
                # no claim of matching anatomy from visual appearance alone
                ax.imshow(np.ma.masked_where(a==0,a),cmap='tab20',alpha=.25,interpolation='nearest')
                ax.imshow(np.ma.masked_where(~(np.isfinite(m)&(m>0)),m),cmap='autumn',alpha=.6,interpolation='nearest')
            ax.set_title(f'{label} voxel {sl}' + (' | candidate overlay' if row else ' | anatomical T1'))
            ax.set_axis_off()
    fig.text(.5,.007,'UNVERIFIED | mask >0 shown for visualization only | template provenance, registration, and mask threshold require independent review',ha='center',fontsize=9)
    fig.savefig(out,dpi=150,bbox_inches='tight')
    plt.close(fig)


def main():
    a=args_parser().parse_args()
    a.root=a.root.expanduser().resolve()
    out=a.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    atlas_path=(a.atlas if a.atlas else a.root/'Child/TumorConnectome/results/dk68_build/DK68_BTCgrid_CANDIDATE_UNVERIFIED.nii.gz').expanduser().resolve()
    mask_root=a.root/'Child/openneuro_BTC_preop/derivatives/tumor_masks'
    paths=sorted(mask_root.glob('sub-PAT*/anat/*_space_MNI_label-tumor.nii*'))
    if a.subjects: paths=[p for p in paths if p.parent.parent.name in set(a.subjects)]
    report={'phase':'06c','registration_validated':False,'tumor_burden_authorized':False,
            'atlas':str(atlas_path),'atlas_exists':atlas_path.is_file(),
            'reference':str(a.reference.expanduser().resolve()) if a.reference else None,
            'reference_space':a.reference_space or None,'atlas_space':a.atlas_space or None,
            'reference_source':a.reference_source or None,
            'masks_found':len(paths),'issues':[], 'per_scan':[]}
    if not atlas_path.exists(): report['issues'].append('Candidate atlas not found; first run previous DK68 candidate build.')
    if not paths: report['issues'].append('No matching MNI tumor masks found.')
    if not a.reference: report['issues'].append('No MNI anatomical reference supplied: geometry QC only, no overlays.')
    if a.reference and not a.reference.exists(): report['issues'].append('Specified anatomical reference missing.')
    if a.reference and not a.reference_source: report['issues'].append('Reference provenance/source not recorded.')
    if not a.atlas_space or not a.reference_space: report['issues'].append('Atlas/reference precise template identities not independently verified.')
    elif a.atlas_space != a.reference_space: report['issues'].append('Atlas/reference template identifiers differ; inspect independently.')
    if a.reference and ('origin/' in str(a.reference) or '_ses-preop_T1w' in a.reference.name):
        report['issues'].append('Native patient T1 appears to have been supplied as MNI reference; overlays will be suppressed.')
        a.reference=None
    try:
        import nibabel as nib
    except ImportError:
        print('ERROR: nibabel not installed. In mri environment: conda install -c conda-forge nibabel',file=sys.stderr)
        return 2
    atlas_img=nib.load(str(atlas_path)) if atlas_path.is_file() else None
    atlas_arr=None
    if atlas_img is not None:
        report['atlas_geometry']=geom(atlas_img)
        atlas_raw=np.asarray(atlas_img.dataobj)
        rounded=np.rint(atlas_raw)
        labels={int(x) for x in np.unique(rounded)} if np.all(np.isfinite(rounded)) else set()
        report['atlas_labels_present']=len(DK_IDS & labels)
        report['atlas_missing_dk_labels']=sorted(DK_IDS-labels)
        report['atlas_unexpected_labels']=sorted(labels-DK_IDS-{0})
        if not np.all(np.isfinite(atlas_raw)) or not np.allclose(atlas_raw,rounded,atol=1e-4):
            report['issues'].append('Atlas is not a finite integer-valued label image.')
        if report['atlas_missing_dk_labels'] or report['atlas_unexpected_labels']:
            report['issues'].append('Atlas missing DK labels or contains unexpected labels.')
        atlas_arr=rounded.astype(np.int32)
        report['atlas_sha256']=digest(atlas_path)
    ref_img=nib.load(str(a.reference)) if a.reference and a.reference.is_file() else None
    ref_data=None
    if ref_img is not None:
        report['reference_geometry']=geom(ref_img)
        report['reference_sha256']=digest(a.reference)
    panel_dir=out/'figures'
    count=0
    for mp in paths:
        subject=mp.parent.parent.name
        row={'subject':subject,'mask':str(mp),'readable':False,'same_grid_as_atlas':False,
             'reference_same_grid':False,'overlay_generated':False,'review_status':'PENDING_MANUAL_REVIEW',
             'tumor_mask_value_min':None,'tumor_mask_value_max':None,'issues':[]}
        try:
            mimg=nib.load(str(mp)); m=np.asarray(mimg.dataobj,dtype=np.float32)
            row['readable']=True;row['mask_shape']='x'.join(map(str,mimg.shape[:3]))
            row['mask_orientation']=''.join(nib.aff2axcodes(mimg.affine))
            row['tumor_mask_value_min']=float(np.nanmin(m));row['tumor_mask_value_max']=float(np.nanmax(m))
            if atlas_img is not None: row['same_grid_as_atlas']=bool(same_grid(atlas_img,mimg))
            if not row['same_grid_as_atlas']: row['issues'].append('Mask/atlas grid mismatch.')
            if ref_img is not None:
                row['reference_same_grid']=bool(same_grid(ref_img,mimg))
                if row['reference_same_grid']: rimg=ref_img
                elif a.allow_reference_resample:
                    from nibabel.processing import resample_from_to
                    rimg=resample_from_to(ref_img,(mimg.shape[:3],mimg.affine),order=1)
                    row['issues'].append('Reference resampled by affine for QC ONLY: not anatomical registration.')
                else:
                    rimg=None
                    row['issues'].append('Reference grid differs; pass --allow-reference-resample only when spatial provenance supports it.')
                if rimg is not None and row['same_grid_as_atlas'] and count<a.max_panels:
                    panel_dir.mkdir(parents=True,exist_ok=True)
                    target=panel_dir/f'{subject}_UNVERIFIED_anatomical_overlay.png'
                    make_panel(atlas_arr,robust_norm(np.asarray(rimg.dataobj)),m,target,subject)
                    row['overlay_generated']=True;count+=1
        except Exception as e:
            row['issues'].append(type(e).__name__+': '+str(e))
        report['per_scan'].append(row)
    csv_fields=('subject','mask','readable','same_grid_as_atlas','reference_same_grid',
                'overlay_generated','review_status','tumor_mask_value_min','tumor_mask_value_max',
                'mask_shape','mask_orientation','issues')
    with open(out/'anatomical_alignment_per_scan.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csv_fields);w.writeheader()
        for r in report['per_scan']:
            w.writerow({k:('; '.join(r.get(k,[])) if k=='issues' else r.get(k,'')) for k in csv_fields})
    report['overlays_generated']=count
    report['matching_mask_grids']=sum(bool(x['same_grid_as_atlas']) for x in report['per_scan'])
    report['readable_masks']=sum(bool(x['readable']) for x in report['per_scan'])
    report['issues'].append('Visual QC does not prove registration; manual review against template anatomy and source transforms mandatory.')
    report['issues'].append('Continuous tumor mask threshold policy remains unverified; no tumor burden computed.')
    (out/'anatomical_alignment_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['BTC DK68 ANATOMICAL ALIGNMENT QC — PHASE 6C','='*55,
           f'Masks: {report["readable_masks"]}/{len(paths)} readable; atlas-grid matches: {report["matching_mask_grids"]}',
           f'Atlas labels: {report.get("atlas_labels_present","N/A")}/68',
           f'Anatomical overlays generated: {count}',
           'ANATOMICAL REGISTRATION: UNVERIFIED',
           'TUMOR BURDEN: NOT AUTHORIZED','', 'Issues / manual checks:']
    lines += ['- '+x for x in report['issues']]
    (out/'anatomical_alignment_summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('\nOutputs:',out)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
