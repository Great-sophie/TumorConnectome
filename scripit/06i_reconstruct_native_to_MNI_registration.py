#!/usr/bin/env python3
"""Phase 6I: exploratory native-T1 -> reference-MNI FLIRT registration.

Independent reconstruction, NOT reproduction of original BTC transforms.
Requires FSL flirt and Python nibabel, numpy, scipy, matplotlib (for figures).
Does not modify source images. Dice only if SAME target geometry and the
user explicitly attests template equivalence via --confirm-template-identity.
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', required=True, type=Path, help='Parent of Child directory, e.g. /media/sophie/Data2T/MRI Data')
    p.add_argument('--reference', required=True, type=Path, help='Explicit MNI T1 template file (NIfTI or MGZ), provenance must be documented')
    p.add_argument('--reference-id', required=True, help='Human-readable exact reference template identity/version')
    p.add_argument('--subjects', nargs='+', default=['sub-PAT23'], help='Subjects or ALL')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--dof', type=int, choices=[6, 9, 12], default=12)
    p.add_argument('--cost', choices=['corratio','mutualinfo','normmi','normcorr','leastsq','labeldiff','bbr'], default='normmi')
    p.add_argument('--searchrx', type=int, nargs=2, default=[-90,90])
    p.add_argument('--searchry', type=int, nargs=2, default=[-90,90])
    p.add_argument('--searchrz', type=int, nargs=2, default=[-90,90])
    p.add_argument('--confirm-template-identity', action='store_true', help='User attests BTC MNI target is anatomically same reference (not inferred from grid alone)')
    p.add_argument('--skip-flirt', action='store_true', help='Use prior outputs, but require existing matrix and resampled images')
    return p.parse_args()


def resolve(p: Path, cwd: Path) -> Path:
    return p.expanduser().resolve() if p.is_absolute() else (cwd / p).resolve()


def run(cmd, logpath: Path):
    with logpath.open('a') as f:
        f.write('\n$ '+ ' '.join(map(str,cmd))+'\n')
        f.flush()
        r = subprocess.run(list(map(str,cmd)), stdout=f, stderr=subprocess.STDOUT, check=False)
        if r.returncode: raise RuntimeError(f'Command failed ({r.returncode}): {cmd[0]}; see {logpath}')


def save_nifti(input_file: Path, target: Path):
    im = nib.load(str(input_file))
    if len(im.shape)!=3: raise ValueError(f'Expected 3D image: {input_file}')
    nib.save(nib.Nifti1Image(np.asarray(im.dataobj, dtype=np.float32), im.affine), str(target))
    return im


def stats(im, threshold):
    data=np.asarray(im.dataobj)
    binary=np.isfinite(data) & (data > threshold)
    n=int(binary.sum())
    v=abs(np.linalg.det(im.affine[:3,:3]))/1000*n
    if n:
        coords=np.argwhere(binary)
        cen=nib.affines.apply_affine(im.affine, coords.mean(axis=0)).tolist()
        _,cnt=label(binary)
    else: cen=[None]*3;cnt=0
    return dict(voxels=n, volume_cm3=float(v), centroid_xyz_mm=cen, components=int(cnt)),binary


def write_csv(path, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=keys);w.writeheader();w.writerows(rows)


def make_qc_figure(ref_img, reg_t1_img, registered_mask_img, btc_img, same_grid, outpath, sid, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nibabel.processing import resample_from_to
    # Reorient all data consistently; reference T1 is NOT the patient's anatomy.
    reference = nib.as_closest_canonical(ref_img)
    patient = nib.as_closest_canonical(reg_t1_img)
    region = nib.as_closest_canonical(registered_mask_img)
    if patient.shape != reference.shape or not np.allclose(patient.affine, reference.affine, atol=1e-4):
        patient=resample_from_to(patient, (reference.shape, reference.affine), order=1)
    if region.shape != reference.shape or not np.allclose(region.affine, reference.affine, atol=1e-4):
        region=resample_from_to(region, (reference.shape, reference.affine), order=1)
    if same_grid:
        btc=nib.as_closest_canonical(btc_img)
        if btc.shape != reference.shape or not np.allclose(btc.affine, reference.affine, atol=1e-4):
            btc=resample_from_to(btc,(reference.shape,reference.affine),order=1)
    else:
        btc=None
    r=np.asarray(reference.dataobj);p=np.asarray(patient.dataobj)
    mask=np.asarray(region.dataobj)>threshold
    b=np.asarray(btc.dataobj)>threshold if btc is not None else None
    coords=np.argwhere(mask)
    cuts=np.rint(coords.mean(axis=0)).astype(int) if len(coords) else np.array(reference.shape)//2
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for i,(name,ax) in enumerate(zip(['Sagittal (X)','Coronal (Y)','Axial (Z)'],[0,1,2])):
        sl=[slice(None)]*3;sl[ax]=int(cuts[ax]);sl=tuple(sl)
        base_ref=np.asarray(r[sl]).T;base_patient=np.asarray(p[sl]).T
        m=np.asarray(mask[sl]).T
        # Both rows share the exact same cut plane; top shows registration geometry,
        # bottom shows candidate tumor over the registered patient's T1.
        for j,(background,label) in enumerate([(base_ref,'Reference T1'),(base_patient,'Registered patient T1')]):
            pane=axes[j,i]
            vals=background[np.isfinite(background) & (background>0)]
            hi=np.percentile(vals,99) if vals.size else 1
            pane.imshow(background,cmap='gray',origin='lower',vmin=0,vmax=max(hi,1e-6))
            if np.any(m) and np.any(~m):pane.contour(m.astype(float),levels=[0.5],colors='yellow',linewidths=1.2)
            if b is not None:
                bm=np.asarray(b[sl]).T
                if np.any(bm) and np.any(~bm):pane.contour(bm.astype(float),levels=[0.5],colors='magenta',linewidths=.8,linestyles='dashed')
            pane.set_title(f'{name} | {label}');pane.axis('off')
    fig.suptitle(f'{sid} | Independent FLIRT -> supplied reference | yellow=new >{threshold}; magenta=BTC (only if exact grid) | UNVERIFIED')
    fig.text(.5,.015,'Same voxel grid does NOT establish template identity. Check reference/patient anatomy and lesions manually.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.025,1,.955]);fig.savefig(outpath,dpi=160);plt.close(fig)


def main():
    a=parse_args(); cwd=Path.cwd(); root=resolve(a.root,cwd); out=resolve(a.out,cwd); out.mkdir(parents=True,exist_ok=True)
    ref=resolve(a.reference,cwd)
    if not ref.is_file(): sys.exit(f'ERROR: reference missing: {ref}')
    if not (0<a.threshold<1): sys.exit('ERROR: threshold must be in (0,1)')
    if not a.skip_flirt and shutil.which('flirt') is None: sys.exit('ERROR: FSL flirt missing from PATH (use FSL environment)')
    pre=root/'Child/openneuro_BTC_preop'
    subjects=sorted(p.name for p in (pre/'derivatives/tumor_masks').glob('sub-PAT*') if p.is_dir()) if 'ALL' in a.subjects else a.subjects
    if not subjects: sys.exit('ERROR: no subjects found')
    # Reference copied to out as NIfTI so conversion and provenance explicit.
    ref_file=out/'reference_T1_EXPLICIT.nii.gz'
    ref_img=save_nifti(ref,ref_file)
    ref_aff=ref_img.affine
    summary={'phase':'6I','reference_source':str(ref),'reference_id':a.reference_id,
             'reference_shape':list(ref_img.shape),'reference_affine':ref_aff.tolist(),
             'btc_original_transform_available':False,
             'independent_registration_not_BTC_original':True,
             'template_identity_user_attested':bool(a.confirm_template_identity),
             'dice_requires_additional_anatomical_QC':True,
             'subjects':[]}
    rows=[]; issues=[]
    for sid in subjects:
        dest=out/sid;dest.mkdir(parents=True,exist_ok=True); log=dest/'fsl_commands.log'
        t1=pre/f'origin/{sid}/ses-preop/anat/{sid}_ses-preop_T1w.nii.gz'
        native=pre/f'derivatives/tumor_masks/{sid}/anat/{sid}_space_T1_label-tumor.nii'
        btc=pre/f'derivatives/tumor_masks/{sid}/anat/{sid}_space_MNI_label-tumor.nii'
        try:
            for p in [t1,native,btc]:
                if not p.is_file(): raise FileNotFoundError(str(p))
            t1i=nib.load(str(t1)); natim=nib.load(str(native)); btci=nib.load(str(btc))
            if len(t1i.shape)!=3:raise ValueError('native T1 not 3D')
            native_t1=dest/'native_t1.nii.gz';native_mask=dest/'native_mask.nii.gz'
            save_nifti(t1,native_t1);save_nifti(native,native_mask)
            aff=dest/f'native_to_reference_FLIRT_{a.dof}DOF.mat'
            reg_t1=dest/'native_T1_in_reference.nii.gz'
            reg_mask_cont=dest/'native_mask_in_reference_trilinear.nii.gz'
            reg_mask_nn=dest/'native_mask_in_reference_nearest.nii.gz'
            if not a.skip_flirt:
                run(['flirt','-in',native_t1,'-ref',ref_file,'-omat',aff,'-out',reg_t1,'-dof',a.dof,
                     '-cost',a.cost,'-searchrx',*a.searchrx,'-searchry',*a.searchry,'-searchrz',*a.searchrz],log)
                # Important: native mask has its own LAS affine but same physical world.
                # FLIRT -applyxfm expects matrix relative to T1 source FSL coordinate frame.
                # First put native mask on exact native T1 voxel grid to avoid relying on
                # FLIRT's differing input-grid conventions.
                from nibabel.processing import resample_from_to
                mask_on_t1=resample_from_to(natim,(t1i.shape,t1i.affine),order=1)
                mask_on_t1_path=dest/'native_mask_on_T1_grid.nii.gz'
                nib.save(mask_on_t1,str(mask_on_t1_path))
                for interp, op in [('trilinear',reg_mask_cont),('nearestneighbour',reg_mask_nn)]:
                    run(['flirt','-in',mask_on_t1_path,'-ref',ref_file,'-applyxfm','-init',aff,
                         '-interp',interp,'-out',op],log)
            for fp in [aff,reg_t1,reg_mask_cont,reg_mask_nn]:
                if not fp.is_file():raise FileNotFoundError(f'Missing FLIRT output: {fp}')
            out_img=nib.load(str(reg_mask_cont)); out_nn=nib.load(str(reg_mask_nn))
            if out_img.shape!=ref_img.shape or not np.allclose(out_img.affine,ref_aff,atol=1e-4):
                raise RuntimeError('resampled mask/reference geometry mismatch')
            if out_nn.shape!=ref_img.shape or not np.allclose(out_nn.affine,ref_aff,atol=1e-4):
                raise RuntimeError('nearest mask/reference geometry mismatch')
            same_grid=(btci.shape==ref_img.shape and np.allclose(btci.affine,ref_aff,atol=1e-4))
            make_qc_figure(ref_img,nib.load(str(reg_t1)),out_img,btci,same_grid, dest/'registration_overlay_UNVERIFIED.png',sid,a.threshold)
            orig, ob=stats(natim,a.threshold); registered, rb=stats(out_img,a.threshold)
            bc,bb=stats(btci,a.threshold)
            same_grid=(btci.shape==ref_img.shape and np.allclose(btci.affine,ref_aff,atol=1e-4))
            # No overlap metrics unless supplied BTC MNI template is explicitly confirmed.
            dice=None; centroid_distance=None
            if a.confirm_template_identity and same_grid:
                denominator=int(rb.sum()+bb.sum())
                dice=float(2*np.logical_and(rb,bb).sum()/denominator) if denominator else None
                if registered['centroid_xyz_mm'][0] is not None and bc['centroid_xyz_mm'][0] is not None:
                    centroid_distance=float(np.linalg.norm(np.asarray(registered['centroid_xyz_mm'])-np.asarray(bc['centroid_xyz_mm'])))
            rec={'subject':sid,'status':'PROCESSED_UNVERIFIED','native_volume_cm3':orig['volume_cm3'],
                 'registered_volume_cm3':registered['volume_cm3'],'btc_MNI_volume_cm3':bc['volume_cm3'],
                 'registered_components':registered['components'],
                 'registered_centroid_xyz_mm':registered['centroid_xyz_mm'],
                 'btc_centroid_xyz_mm':bc['centroid_xyz_mm'],
                 'reference_grid_matches_BTC':same_grid,
                 'dice_if_template_identity_attested':dice,
                 'centroid_distance_mm_if_attested':centroid_distance,
                 'template_identity_verified':'USER_ATTESTED_ONLY' if a.confirm_template_identity else 'UNVERIFIED',
                 'registration_visual_QC':'PENDING', 'native_T1_mask_visual_QC':'PENDING'}
            rows.append({k:(json.dumps(v) if isinstance(v,list) else v) for k,v in rec.items()})
            summary['subjects'].append(rec)
            print(f'OK {sid}: native {orig["volume_cm3"]:.2f}, registered {registered["volume_cm3"]:.2f}, BTC MNI {bc["volume_cm3"]:.2f} cm3; template grid same={same_grid}; Dice={dice if dice is not None else "WITHHELD"}')
        except Exception as exc:
            issues.append({'subject':sid,'error':f'{type(exc).__name__}: {exc}'})
            print(f'FAIL {sid}: {exc}',file=sys.stderr)
    summary['failures']=issues
    with (out/'registration_reconstruction_summary.json').open('w') as f:json.dump(summary,f,indent=2)
    write_csv(out/'registration_metrics.csv',rows);write_csv(out/'processing_failures.csv',issues)
    message=(f'PHASE 6I INDEPENDENT NATIVE->MNI RECONSTRUCTION\n'
             f'Processed: {len(rows)}/{len(subjects)} | failures: {len(issues)}\n'
             f'Reference: {a.reference_id} | {ref}\n'
             f'BTC original transforms: NOT AVAILABLE\n'
             f'Template identity: {"USER ATTESTED (requires independent provenance review)" if a.confirm_template_identity else "UNVERIFIED"}\n'
             f'Dice: {"computed conditionally; not registration proof" if a.confirm_template_identity else "WITHHELD"}\n'
             'Visual QC: NOT AUTOMATICALLY PASSED\n'
             'TUMOR-DK68 BURDEN: NOT AUTHORIZED\n')
    (out/'registration_reconstruction_summary.txt').write_text(message)
    print('\n'+message)
    if issues:sys.exit(1)

if __name__=='__main__':main()
