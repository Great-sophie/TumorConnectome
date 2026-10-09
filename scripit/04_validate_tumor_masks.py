#!/usr/bin/env python3
"""TumorConnectome Phase 4: read-only QC of BTC preop tumor masks.

Usage:
 python 04_validate_tumor_masks.py --root '/media/sophie/Data2T/MRI Data' \
     --out '../results/tumor_mask_validation'

Optional --atlas accepts a DK68 label NIfTI *already known* to be in MNI space.
Geometry agreement alone does not prove atlas label ordering or biological alignment.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np

THRESHOLDS = (0.0, 0.01, 0.1, 0.5)
FIELDS = [
    'subject','space','path','status','error','shape','zooms_mm','voxel_mm3',
    'affine_det','affine_finite','finite_fraction','nan_voxels','negative_voxels',
    'min','max','n_positive','positive_p25','positive_p50','positive_p75',
    'positive_p90','positive_p99','near_binary_fraction','nonzero_bbox_vox',
    'vox_gt_0','vol_ml_gt_0','vox_gt_0p01','vol_ml_gt_0p01',
    'vox_gt_0p1','vol_ml_gt_0p1','vox_gt_0p5','vol_ml_gt_0p5',
    'ratio_vol_0p01_to_0p5','qc_flags',
]

def csv_write(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def rel_str(path, root):
    try: return str(path.relative_to(root))
    except ValueError: return str(path)

def analyze_mask(path, sid, space, root, nib):
    row = dict.fromkeys(FIELDS, '')
    row.update(subject=sid, space=space, path=rel_str(path, root), status='missing')
    if not path.is_file(): return row
    try:
        img = nib.load(str(path))
        if len(img.shape) != 3: raise ValueError(f'expected 3D NIfTI; got {img.shape}')
        aff = np.asarray(img.affine, dtype=float)
        voxel_mm3 = float(abs(np.linalg.det(aff[:3,:3])))
        data = np.asarray(img.dataobj, dtype=np.float32)
        valid = np.isfinite(data)
        finite_vals = data[valid]
        if finite_vals.size == 0: raise ValueError('no finite voxels')
        pos = finite_vals[finite_vals > 0]
        flags=[]
        if not np.all(np.isfinite(aff)) or not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
            flags.append('invalid_affine')
        if valid.sum() != data.size: flags.append('nonfinite_values')
        if np.any(finite_vals < 0): flags.append('negative_values')
        if not pos.size: flags.append('empty_mask')
        if pos.size and np.any((pos > 1.001)): flags.append('values_above_one')
        if pos.size and np.any((pos < 0.99)): flags.append('continuous_positive_values')
        if pos.size and float(np.median(pos)) < 0.01: flags.append('low_intensity_positive_tail')
        bbox=''
        # bounding box on >0.5 excludes interpolation tails; not a verified segmentation.
        ijk = np.argwhere(valid & (data > 0.5))
        if ijk.size:
            lo=ijk.min(axis=0); hi=ijk.max(axis=0)
            bbox=';'.join(f'{a}:{b}' for a,b in zip(lo,hi))
        elif pos.size: flags.append('no_voxels_above_0p5')
        row.update(status='ok', shape='x'.join(map(str,img.shape)),
                   zooms_mm=';'.join(f'{x:.6g}' for x in img.header.get_zooms()[:3]),
                   voxel_mm3=voxel_mm3, affine_det=float(np.linalg.det(aff[:3,:3])),
                   affine_finite=bool(np.isfinite(aff).all()),
                   finite_fraction=float(valid.mean()), nan_voxels=int(np.isnan(data).sum()),
                   negative_voxels=int(np.count_nonzero(finite_vals < 0)),
                   min=float(finite_vals.min()),max=float(finite_vals.max()),
                   n_positive=int(pos.size),
                   near_binary_fraction=float(np.mean((np.abs(finite_vals)<1e-6)|(np.abs(finite_vals-1)<1e-6))),
                   nonzero_bbox_vox=bbox)
        if pos.size:
            for q,k in [(25,'positive_p25'),(50,'positive_p50'),(75,'positive_p75'),(90,'positive_p90'),(99,'positive_p99')]:
                row[k]=float(np.percentile(pos,q))
        keys=[('0',0.0),('0p01',0.01),('0p1',0.1),('0p5',0.5)]
        for key,t in keys:
            n=int(np.count_nonzero(valid & (data > t)))
            row['vox_gt_'+key]=n
            row['vol_ml_gt_'+key]=float(n*voxel_mm3/1000)
        a=row['vol_ml_gt_0p01']; b=row['vol_ml_gt_0p5']
        if b and b > 0:
            ratio=a/b; row['ratio_vol_0p01_to_0p5']=ratio
            if ratio > 1.5: flags.append('threshold_sensitive_volume')
        row['qc_flags']=';'.join(flags)
        return row
    except Exception as e:
        row.update(status='error',error=f'{type(e).__name__}: {e}')
        return row

def geometry_compare(mask_path, atlas_path, nib):
    out={'atlas_path':str(atlas_path),'geometry_status':'not_checked','same_shape':'','same_affine':'',
         'max_affine_abs_difference':'','atlas_unique_labels':'','note':''}
    if not atlas_path.is_file():
        out.update(geometry_status='atlas_missing'); return out
    if not mask_path.is_file():
        out.update(geometry_status='mask_missing'); return out
    try:
        m=nib.load(str(mask_path)); a=nib.load(str(atlas_path))
        out['same_shape']=bool(m.shape == a.shape)
        out['same_affine']=bool(np.allclose(m.affine,a.affine,rtol=0,atol=1e-3))
        out['max_affine_abs_difference']=float(np.max(np.abs(m.affine-a.affine)))
        if len(a.shape)==3:
            labels=np.unique(np.asarray(a.dataobj))
            out['atlas_unique_labels']=int(labels.size)
        out['geometry_status']='same_grid' if out['same_shape'] and out['same_affine'] else 'different_grid'
        out['note']='Same voxel grid is necessary, not sufficient: verify DK68 identity, node ordering, registration and QC.'
    except Exception as e:
        out.update(geometry_status='error',note=str(e))
    return out

def plot_mask(path, out, nib, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img=nib.load(str(path))
    x=np.asarray(img.dataobj, dtype=np.float32)
    ind=np.argwhere(np.isfinite(x)&(x>0.5))
    center=np.round(ind.mean(axis=0)).astype(int) if len(ind) else np.array(x.shape)//2
    slices=[x[center[0],:,:],x[:,center[1],:],x[:,:,center[2]]]
    fig,ax=plt.subplots(1,3,figsize=(10,3.7))
    for i,(a,s) in enumerate(zip(ax,slices)):
        a.imshow(np.rot90(s),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        a.contour(np.rot90(s)>0.5,levels=[0.5],colors=['cyan'],linewidths=0.7) if np.any(s>0.5) and np.any(s<=0.5) else None
        a.set_title(('Sagittal','Coronal','Axial')[i]); a.axis('off')
    fig.suptitle(title+' — mask intensities only; >0.5 contour is illustrative',fontsize=10)
    fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',required=True,type=Path,help='MRI Data root')
    ap.add_argument('--out',required=True,type=Path,help='QC outputs')
    ap.add_argument('--atlas',type=Path,default=None,help='Optional MNI-space DK68 label NIfTI')
    ap.add_argument('--plot-subject',default='sub-PAT01',help='Representative patient, or none')
    args=ap.parse_args()
    try: import nibabel as nib
    except ImportError as e: raise SystemExit('Missing nibabel: conda install -c conda-forge nibabel') from e
    root=args.root.expanduser().resolve(); out=args.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    tumor_root=root/'Child/openneuro_BTC_preop/derivatives/tumor_masks'
    tvb_root=root/'Child/openneuro_BTC_preop/derivatives/TVB'
    if not tumor_root.is_dir(): raise SystemExit(f'Not found: {tumor_root}')
    subjects=sorted({p.name for p in tumor_root.glob('sub-PAT*') if p.is_dir()} |
                    {p.name for p in tvb_root.glob('sub-PAT*') if p.is_dir()})
    if not subjects: raise SystemExit('No PAT subjects found; check --root')
    rows=[]; pairs=[]
    for sid in subjects:
        paths={s:tumor_root/sid/'anat'/f'{sid}_space_{s}_label-tumor.nii' for s in ('T1','MNI')}
        results={s:analyze_mask(p,sid,s,root,nib) for s,p in paths.items()}
        rows.extend(results.values())
        d={'subject':sid,'T1_status':results['T1']['status'],'MNI_status':results['MNI']['status'],
           'T1_max':results['T1']['max'],'MNI_max':results['MNI']['max'],
           'T1_ml_gt_0p5':results['T1']['vol_ml_gt_0p5'],
           'MNI_ml_gt_0p5':results['MNI']['vol_ml_gt_0p5'],
           'T1_flags':results['T1']['qc_flags'],'MNI_flags':results['MNI']['qc_flags'],
           'geometry_status':'not_checked', 'analysis_ready':'unverified_alignment_and_label_definition'}
        if args.atlas:
            check=geometry_compare(paths['MNI'],args.atlas.expanduser().resolve(),nib)
            d['geometry_status']=check['geometry_status']
            d['atlas_same_shape']=check['same_shape'];d['atlas_same_affine']=check['same_affine']
            # Still not certified for lesion-aware modeling: atlas labels/order remain unverified.
        pairs.append(d)
        print(f"{sid}: T1={d['T1_status']} MNI={d['MNI_status']} atlas={d['geometry_status']}")
    csv_write(out/'tumor_mask_qc.csv',rows,FIELDS)
    keys=['subject','T1_status','MNI_status','T1_max','MNI_max','T1_ml_gt_0p5','MNI_ml_gt_0p5',
          'T1_flags','MNI_flags','geometry_status','analysis_ready']
    if args.atlas: keys.extend(['atlas_same_shape','atlas_same_affine'])
    csv_write(out/'tumor_mask_pairs.csv',pairs,keys)
    counts={s:dict(Counter(r['status'] for r in rows if r['space']==s)) for s in ('T1','MNI')}
    flags=dict(Counter(flag for row in rows for flag in str(row['qc_flags']).split(';') if flag))
    summary={'n_patients_found':len(subjects),'mask_status':counts,'flags':flags,
             'atlas_supplied':bool(args.atlas),'critical_limitations':[
               'No mask is designated verified ground truth by this automated inspection.',
               'Intensity thresholds are QC/sensitivity checks, not validated tumor volumes.',
               'Matching MNI geometry does not validate DK68 parcel identity/order.',
               'Overlay on anatomical MRI, inspect affine/registration and manual masks before regional burden.',
               'Privacy and source-license review required before publishing subject-level outputs.'
             ]}
    (out/'mask_validation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    report=['BTC TUMOR MASK VALIDATION — PHASE 4','='*52,f"Patient directories: {len(subjects)}",
            f'T1 status: {counts["T1"]}',f'MNI status: {counts["MNI"]}',f'QC flags: {flags}',
            f'Atlas supplied: {bool(args.atlas)}','','LIMITATIONS:']
    report += ['- '+s for s in summary['critical_limitations']]
    (out/'mask_validation_summary.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    sid=args.plot_subject
    if sid.lower()!='none':
        figdir=out/'figures';figdir.mkdir(exist_ok=True)
        for space in ('T1','MNI'):
            p=tumor_root/sid/'anat'/f'{sid}_space_{space}_label-tumor.nii'
            if p.is_file():
                try: plot_mask(p,figdir/f'{sid}_{space}_mask_qc.png',nib,f'{sid} {space}')
                except Exception as e: print(f'Plot warning {sid} {space}: {e}')
    print(f'\nWrote QC files to: {out}')
    print((out/'mask_validation_summary.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
