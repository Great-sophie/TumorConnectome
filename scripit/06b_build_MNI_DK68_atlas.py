#!/usr/bin/env python3
"""TumorConnectome 06B: provenance-gated DK68 atlas resampling and spatial QC.

No atlas is downloaded or manufactured. Anatomical alignment is NOT proven by
resampling, coordinate labels, an affine, or this program's success status.

Inspect:
  python 06b_build_MNI_DK68_atlas.py --root '/media/sophie/Data2T/MRI Data' --out ../results/dk68_build
Build a *candidate*, only after verifying the exact common template/provenance:
  python 06b_build_MNI_DK68_atlas.py --root '...' --out ../results/dk68_build \
    --atlas /path/to/documented_DK68_MNI.nii.gz \
    --atlas-space 'MNI152NLin2009cAsym' --btc-space 'VERIFIED_ACTUAL_TEMPLATE' \
    --source 'URL/DOI/version for the atlas' --confirm-same-template

If atlas uses non-FreeSurfer IDs, supply a CSV with node_name,label_id columns;
all 68 must match SC.zip ordered nodes. Output uses standardized node indices 1-68,
not FreeSurfer IDs, with a saved mapping table. The output remains UNVERIFIED until
manual/anatomical overlay and transform/provenance review.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

DK_CODES = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
DK_NAMES = ['bankssts','caudalanteriorcingulate','caudalmiddlefrontal','cuneus','entorhinal',
 'fusiform','inferiorparietal','inferiortemporal','isthmuscingulate','lateraloccipital',
 'lateralorbitofrontal','lingual','medialorbitofrontal','middletemporal','parahippocampal',
 'paracentral','parsopercularis','parsorbitalis','parstriangularis','pericalcarine',
 'postcentral','posteriorcingulate','precentral','precuneus','rostralanteriorcingulate',
 'rostralmiddlefrontal','superiorfrontal','superiorparietal','superiortemporal',
 'supramarginal','frontalpole','temporalpole','transversetemporal','insula']
EXPECTED = ['lh_'+n for n in DK_NAMES]+['rh_'+n for n in DK_NAMES]
FS_IDS = [1000+x for x in DK_CODES]+[2000+x for x in DK_CODES]
MASK_REL='Child/openneuro_BTC_preop/derivatives/tumor_masks'
SC_REL='Child/openneuro_BTC_preop/derivatives/TVB/sub-PAT01/ses-preop/SC.zip'

def csv_write(path, rows, cols):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''): h.update(part)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path('/media/sophie/Data2T/MRI Data'))
    p.add_argument('--out',type=Path,default=Path('../results/dk68_build'))
    p.add_argument('--atlas',type=Path,help='Documented MNI DK68 3D discrete label volume (nii, nii.gz or mgz)')
    p.add_argument('--mapping',type=Path,help='CSV node_name,label_id (required for non-FreeSurfer labeling)')
    p.add_argument('--atlas-space',help='Exact atlas template identifier/version, not just MNI')
    p.add_argument('--btc-space',help='Independently verified exact BTC target template identifier/version')
    p.add_argument('--source',help='Atlas URL/DOI and version, for provenance')
    p.add_argument('--confirm-same-template',action='store_true',help='Explicitly confirm external anatomical space compatibility')
    p.add_argument('--preview-mask',help='Subject ID for simple UNVERIFIED mask+atlas overlay, e.g. sub-PAT01')
    a=p.parse_args()
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError:
        p.error('Requires nibabel and scipy: conda install -c conda-forge nibabel scipy')
    a.out.mkdir(parents=True,exist_ok=True)
    issues=[]; info={'phase':'06B','status':'INSPECTION_ONLY','tumor_burden_authorized':False,'registration_validated':False}
    files=sorted((a.root/MASK_REL).glob('sub-*/anat/*_space_MNI_label-tumor.nii*'))
    if not files: p.error('No BTC MNI tumor masks found under '+str(a.root/MASK_REL))
    masks=[]
    for file in files:
        try:
            img=nib.load(str(file))
            if len(img.shape)!=3 or not np.isfinite(img.affine).all(): raise ValueError('Invalid image dimension/affine')
            masks.append((file,img))
        except Exception as e: issues.append(f'Mask unreadable: {file}: {e}')
    if not masks: p.error('No readable mask')
    ref=masks[0][1]
    same=[f for f,img in masks if img.shape==ref.shape and np.allclose(img.affine,ref.affine,rtol=0,atol=1e-5)]
    info.update(mask_count=len(files),readable_mask_count=len(masks),identical_grid_count=len(same),
                reference_mask=str(masks[0][0]),target_shape=list(ref.shape),
                target_affine=np.asarray(ref.affine).tolist(),target_orientation=''.join(nib.aff2axcodes(ref.affine)))
    if len(same)!=len(masks): issues.append('Not all masks have the same target grid')
    with ZipFile(a.root/SC_REL) as z:
        names=[line.split()[0] for line in z.read('centres.txt').decode().splitlines() if line.strip()]
    if names!=EXPECTED: issues.append('SC.zip labels differ from standard DK68 list/order')
    labels=FS_IDS
    if a.mapping:
        with a.mapping.open(newline='',encoding='utf-8-sig') as f:
            records=list(csv.DictReader(f))
        if len(records)!=68 or {r['node_name'] for r in records}!=set(EXPECTED):
            p.error('Mapping must have exactly 68 unique expected DK68 node_names')
        byname={r['node_name']:int(r['label_id']) for r in records}
        labels=[byname[n] for n in EXPECTED]
    if len(set(labels))!=68 or any(x<=0 for x in labels): p.error('Atlas label IDs must be 68 unique positive integers')
    node_rows=[{'node_index_0based':i,'node_index_1based':i+1,'node_name':name,'atlas_source_label_id':lab,'output_label_id':i+1}
               for i,(name,lab) in enumerate(zip(EXPECTED,labels))]
    csv_write(a.out/'DK68_node_label_mapping.csv',node_rows,list(node_rows[0]))
    if not a.atlas:
        issues.append('No source atlas provided: cannot create any atlas image')
    else:
        if not a.atlas.is_file(): p.error('Atlas file not found: '+str(a.atlas))
        info['source_atlas']=str(a.atlas.resolve());info['source_sha256']=digest(a.atlas)
        atlas=nib.load(str(a.atlas))
        if len(atlas.shape)!=3 or not np.isfinite(atlas.affine).all(): p.error('Source atlas must be a 3D image with a finite affine')
        info.update(source_shape=list(atlas.shape),source_affine=np.asarray(atlas.affine).tolist(),
                    source_orientation=''.join(nib.aff2axcodes(atlas.affine)))
        vals=np.asanyarray(atlas.dataobj)
        if not np.isfinite(vals).all() or not np.allclose(vals,np.rint(vals),atol=1e-4,rtol=0):
            p.error('Atlas must be finite integer-valued categorical labels (not probabilities)')
        source_present=set(np.unique(np.rint(vals).astype(np.int64)).tolist())
        missing=[lab for lab in labels if lab not in source_present]
        info['missing_source_labels']=missing
        if missing: issues.append(f'Source atlas lacks {len(missing)} required DK68 labels: {missing}')
        if not (a.atlas_space and a.btc_space and a.source):
            issues.append('Exact atlas/btc template IDs and atlas provenance are required for candidate generation')
        if not a.confirm_same_template: issues.append('Explicit --confirm-same-template absent')
        if a.atlas_space and a.btc_space and a.atlas_space!=a.btc_space:
            issues.append('Template IDs differ: provide a validated transform, do not resample by affine alone')
        info.update(atlas_space=a.atlas_space,btc_space=a.btc_space,source_provenance=a.source,
                    externally_confirmed_same_template=bool(a.confirm_same_template))
        if not issues:
            # Interpolation uses world-affine geometry ONLY. It does not estimate registration.
            rs=resample_from_to(atlas,(ref.shape,ref.affine),order=0,mode='constant',cval=0)
            rsdata=np.asarray(rs.dataobj)
            rsdata=np.rint(rsdata).astype(np.int32)
            source_unused=sorted(set(np.unique(rsdata))-set(labels)-{0})
            info['non_DK_labels_after_resampling']=source_unused[:50]
            output=np.zeros(ref.shape,dtype=np.uint8)
            for node_id,lab in enumerate(labels,start=1): output[rsdata==lab]=node_id
            counts={name:int(np.count_nonzero(output==i+1)) for i,name in enumerate(EXPECTED)}
            info['output_voxel_counts']=counts
            miss_out=[n for n,count in counts.items() if count==0]
            if miss_out: issues.append('Labels vanished after resampling: '+', '.join(miss_out))
            if not issues:
                header=ref.header.copy();header.set_data_dtype(np.uint8)
                outimg=nib.Nifti1Image(output,ref.affine,header=header)
                outimg.set_qform(ref.affine,code=1);outimg.set_sform(ref.affine,code=1)
                candidate=a.out/'DK68_BTCgrid_CANDIDATE_UNVERIFIED.nii.gz'
                nib.save(outimg,str(candidate))
                info.update(status='CANDIDATE_CREATED_MANUAL_QC_REQUIRED',candidate_output=str(candidate),
                            output_sha256=digest(candidate))
                if a.preview_mask:
                    chosen=[(f,img) for f,img in masks if a.preview_mask in str(f)]
                    if not chosen: issues.append('Preview subject not found: '+a.preview_mask)
                    else:
                        try:
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            tumor=np.asarray(chosen[0][1].dataobj,dtype=float)
                            loc=np.argwhere(np.isfinite(tumor)&(tumor>0.5))
                            z=int(np.median(loc[:,2])) if len(loc) else tumor.shape[2]//2
                            fig,ax=plt.subplots(figsize=(7,7))
                            ax.imshow(np.rot90(output[:,:,z]),cmap='tab20',interpolation='nearest')
                            ax.imshow(np.ma.masked_where(np.rot90(tumor[:,:,z])<=0.5,np.rot90(tumor[:,:,z])),cmap='Reds',alpha=.65)
                            ax.set_title('UNVERIFIED atlas + tumor (not registration QC)')
                            ax.axis('off');fig.savefig(a.out/'UNVERIFIED_overlay.png',dpi=140);plt.close(fig)
                        except Exception as e: issues.append('Preview not generated: '+str(e))
    info['issues']=issues
    info['tumor_burden_authorized']=False
    (a.out/'spatial_build_summary.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=[f"BTC DK68 ATLAS BUILD — PHASE 6B",'='*54,
           f"Masks: {info['readable_mask_count']}/{info['mask_count']} readable; same grid: {info['identical_grid_count']}",
           f"Target: {info['target_shape']} / {info['target_orientation']}",
           f"Status: {info['status']}","ANATOMICAL REGISTRATION: NOT VALIDATED",'TUMOR BURDEN: NOT AUTHORIZED','',
           'Issues / manual checks:']+['- '+x for x in issues]+['- Mandatory: review source provenance, template and anatomical overlay against a BTC anatomical reference (not mask alone).']
    (a.out/'spatial_build_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
