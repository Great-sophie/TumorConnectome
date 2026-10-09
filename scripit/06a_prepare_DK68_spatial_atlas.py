#!/usr/bin/env python3
"""TumorConnectome Phase 6A: conservative DK68 spatial atlas discovery and QC.

Read only: inspects candidates and BTC MNI masks; NEVER registers/resamples images,
assigns anatomical correspondence from an affine alone, or calculates tumor burden.
An explicitly supplied --atlas is a hypothesis to inspect, not an approved atlas.

Example:
 python 06a_prepare_DK68_spatial_atlas.py --root '/media/sophie/Data2T/MRI Data' \
    --out '../results/dk68_spatial_atlas'
 python 06a_prepare_DK68_spatial_atlas.py --root '...' --atlas '/path/to/atlas.nii.gz' \
    --atlas-space MNI --atlas-provenance 'DOI or documented template version'
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit('Requires nibabel: conda install -c conda-forge nibabel') from exc

DK_CODES = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
DK_NAMES = ['bankssts','caudalanteriorcingulate','caudalmiddlefrontal','cuneus','entorhinal',
            'fusiform','inferiorparietal','inferiortemporal','isthmuscingulate',
            'lateraloccipital','lateralorbitofrontal','lingual','medialorbitofrontal',
            'middletemporal','parahippocampal','paracentral','parsopercularis',
            'parsorbitalis','parstriangularis','pericalcarine','postcentral',
            'posteriorcingulate','precentral','precuneus','rostralanteriorcingulate',
            'rostralmiddlefrontal','superiorfrontal','superiorparietal',
            'superiortemporal','supramarginal','frontalpole','temporalpole',
            'transversetemporal','insula']
EXPECTED = ['lh_'+n for n in DK_NAMES] + ['rh_'+n for n in DK_NAMES]
FS_IDS = [1000+c for c in DK_CODES] + [2000+c for c in DK_CODES]
MASK_REL = 'Child/openneuro_BTC_preop/derivatives/tumor_masks'
IMG_ENDINGS = ('.nii','.nii.gz','.mgz','.mgh')


def dump_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def atlas_kind(path):
    n=path.name.lower()
    if 'dktatlas' in n or 'dkt+' in n:
        return 'DKT_NOT_DK68'
    if 'a2009s' in n:
        return 'DESTRIEUX_NOT_DK68'
    if 'aparc+aseg' in n or 'dk68' in n or 'desikan' in n or 'aparc' in n:
        return 'POSSIBLE_DK_NEEDS_VALIDATION'
    return 'OTHER_ATLAS_NEEDS_VALIDATION'


def short_path(path, root):
    try:return str(path.relative_to(root))
    except ValueError:return str(path)


def image_header(path):
    img=nib.load(str(path))
    aff=np.asarray(img.affine,dtype=float)
    if len(img.shape)!=3 or aff.shape!=(4,4) or not np.isfinite(aff).all():
        raise ValueError('Expected 3D image with finite 4x4 affine')
    orientation=''.join(nib.aff2axcodes(aff))
    return img, {'shape':list(img.shape), 'zooms_mm':list(map(float,img.header.get_zooms()[:3])),
                 'orientation':orientation,'affine':aff.round(7).tolist(),
                 'voxel_mm3':float(abs(np.linalg.det(aff[:3,:3])))}


def spatial_extent(img):
    """Eight corner world coordinates; descriptive bounding box, not registration check."""
    corners=np.array([[x,y,z] for x in (0,img.shape[0]-1)
                      for y in (0,img.shape[1]-1) for z in (0,img.shape[2]-1)],dtype=float)
    pts=nib.affines.apply_affine(img.affine,corners)
    return {'world_min_mm':pts.min(0).round(3).tolist(),
            'world_max_mm':pts.max(0).round(3).tolist()}


def inventory_masks(root):
    base=root/MASK_REL
    paths=sorted(base.glob('sub-*/anat/*_space_MNI_label-tumor.nii*'))
    rows=[]
    for path in paths:
        row={'subject':path.parent.parent.name,'path':str(path),'status':'ERROR'}
        try:
            img,meta=image_header(path)
            row.update(status='READABLE',shape='x'.join(map(str,meta['shape'])),
                       orientation=meta['orientation'],voxel_mm3=meta['voxel_mm3'],
                       affine_json=json.dumps(meta['affine']))
        except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
    return rows,paths


def find_candidates(root, additions, limit):
    roots=[root/'Child/openneuro_BTC_preop',root/'Child/openneuro BTC_postop',
           root/'Child/openneuro',*additions]
    pattern=re.compile(r'(aparc|desikan|dk68|parcellation|atlas)',re.I)
    paths=[];seen=set()
    for base in roots:
        if not base.exists():continue
        if base.is_file():
            files=[base]
        else:
            files=(p for p in base.rglob('*') if p.is_file())
        for p in files:
            if not p.name.lower().endswith(IMG_ENDINGS) or not pattern.search(p.name):continue
            key=str(p.resolve())
            if key in seen:continue
            seen.add(key);paths.append(p)
            if len(paths)>=limit:return paths,True
    return paths,False


def get_sc_names(root):
    p=root/'Child/openneuro_BTC_preop/derivatives/TVB/sub-PAT01/ses-preop/SC.zip'
    try:
        with ZipFile(p) as z:
            names=[line.split()[0] for line in z.read('centres.txt').decode('utf-8-sig').splitlines() if line.strip()]
        return names,p
    except Exception:
        return [],p


def inspect_labels(img):
    arr=np.asarray(img.dataobj)
    valid=np.isfinite(arr)
    if not valid.any():return {'status':'INVALID_NO_FINITE_VOXELS'}
    vals,counts=np.unique(arr[valid],return_counts=True)
    integer_like=bool(np.all(np.abs(vals-np.rint(vals))<1e-4))
    present=set(int(x) for x in np.rint(vals).tolist()) if integer_like else set()
    found=sorted(set(FS_IDS).intersection(present))
    return {'status':'READABLE','integer_like':integer_like,'unique_count':int(vals.size),
            'labels_preview':[float(x) for x in vals[:35]],
            'dk_freesurfer_ids_present':len(found),
            'dk_freesurfer_missing_ids':sorted(set(FS_IDS)-present),
            'fully_contains_freesurfer_dk68':len(found)==68,
            'nonzero_voxels':int(np.count_nonzero(valid & (arr!=0)))}


def make_overlay(mask_path,atlas_path,out):
    """Preview only when arrays are on exactly the same grid; not evidence of registration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=nib.load(str(mask_path));a=nib.load(str(atlas_path))
    if m.shape!=a.shape or not np.allclose(m.affine,a.affine,atol=1e-4,rtol=0):
        return 'SKIPPED_DIFFERENT_GRID'
    mask=np.asarray(m.dataobj,dtype=float)
    atlas=np.asarray(a.dataobj,dtype=float)
    idx=np.argwhere(np.isfinite(mask)&(mask>0.5))
    center=np.rint(np.median(idx,axis=0)).astype(int) if idx.size else np.array(mask.shape)//2
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for axis,ax in enumerate(axes):
        sl=[slice(None)]*3;sl[axis]=int(center[axis]);sl=tuple(sl)
        bg=np.rot90(np.isfinite(atlas[sl]).astype(float))
        lab=np.rot90(atlas[sl]);les=np.rot90(mask[sl])
        ax.imshow(bg,cmap='gray',vmin=0,vmax=1)
        ax.imshow(np.ma.masked_where(lab==0,lab),cmap='tab20',alpha=0.40,interpolation='nearest')
        ax.imshow(np.ma.masked_where(les<=0.5,les),cmap='Reds',alpha=0.6,interpolation='nearest')
        ax.set_title(f'Axis {axis}, voxel {center[axis]}');ax.axis('off')
    fig.suptitle('UNVERIFIED registration: atlas + tumor mask (threshold >0.5)')
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)
    return 'PREVIEW_ONLY_MANUAL_REVIEW_REQUIRED'


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',type=Path,default=Path('/media/sophie/Data2T/MRI Data'))
    ap.add_argument('--out',type=Path,default=Path('../results/dk68_spatial_atlas'))
    ap.add_argument('--atlas',type=Path,help='Optional candidate anatomical label volume; no automatic registration')
    ap.add_argument('--atlas-space',choices=['MNI','T1','unknown'],default='unknown',
                    help='Spatial claim from documented provenance, not inferred from filename')
    ap.add_argument('--atlas-provenance',default='',help='Optional DOI/source/template/registration provenance')
    ap.add_argument('--extra-search',type=Path,action='append',default=[],help='Additional local atlas search folder (repeatable)')
    ap.add_argument('--max-candidates',type=int,default=500)
    ap.add_argument('--sample-subject',default='sub-PAT01')
    args=ap.parse_args()
    root=args.root.expanduser().resolve();out=args.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    masks,mask_paths=inventory_masks(root)
    dump_csv(out/'mni_tumor_mask_inventory.csv',masks,
             ['subject','path','status','shape','orientation','voxel_mm3','affine_json','error'])
    candidates,truncated=find_candidates(root,[p.expanduser().resolve() for p in args.extra_search],args.max_candidates)
    candidate_rows=[]
    for p in candidates:
        kind=atlas_kind(p)
        row={'path':str(p),'relative_path':short_path(p,root),'type':kind,
             'belongs_to_btc':'BTC' if 'openneuro_BTC_preop' in str(p) or 'openneuro BTC_postop' in str(p) else 'OTHER_DATASET_OR_LOCATION',
             'status':'NOT_INSPECTED'}
        try:
            _,meta=image_header(p)
            row.update(status='HEADER_OK',shape='x'.join(map(str,meta['shape'])),
                       orientation=meta['orientation'],voxel_mm3=meta['voxel_mm3'])
        except Exception as exc:row.update(status='HEADER_ERROR',error=str(exc))
        candidate_rows.append(row)
    dump_csv(out/'atlas_candidate_inventory.csv',candidate_rows,
             ['path','relative_path','type','belongs_to_btc','status','shape','orientation','voxel_mm3','error'])
    sc_names,zip_path=get_sc_names(root)
    map_rows=[{'node_index':i,'sc_name':n,'freesurfer_aparc_id':FS_IDS[i],
               'reference_name':EXPECTED[i],'sc_matches_reference':n==EXPECTED[i]}
              for i,n in enumerate(sc_names[:68])]
    if not map_rows:
        map_rows=[{'node_index':i,'sc_name':'','freesurfer_aparc_id':FS_IDS[i],
                   'reference_name':EXPECTED[i],'sc_matches_reference':'NOT_CHECKED'} for i in range(68)]
    dump_csv(out/'dk68_reference_node_mapping.csv',map_rows,
             ['node_index','sc_name','freesurfer_aparc_id','reference_name','sc_matches_reference'])
    atlas_report={'provided':args.atlas is not None,'space_claim':args.atlas_space,
                  'provenance':args.atlas_provenance,'status':'NOT_PROVIDED'}
    grid_rows=[];preview_status='NOT_REQUESTED'
    if args.atlas:
        path=args.atlas.expanduser().resolve();atlas_report['path']=str(path)
        try:
            atlas,meta=image_header(path)
            atlas_report.update(status='INSPECTED',header=meta,extent=spatial_extent(atlas),labels=inspect_labels(atlas))
            for mrow in masks:
                if mrow['status']!='READABLE':continue
                m,mm=image_header(Path(mrow['path']))
                same_shape=m.shape==atlas.shape
                diff=float(np.max(np.abs(m.affine-atlas.affine)))
                same_affine=bool(np.allclose(m.affine,atlas.affine,atol=1e-4,rtol=0))
                grid_rows.append({'subject':mrow['subject'],'same_shape':same_shape,
                                  'same_affine':same_affine,'max_affine_difference':diff,
                                  'same_grid':bool(same_shape and same_affine),
                                  'manual_registration_qc':'NOT_DONE'})
            chosen=next((p for p in mask_paths if args.sample_subject in str(p)),None)
            if chosen:
                try:preview_status=make_overlay(chosen,path,out/'UNVERIFIED_overlay_preview.png')
                except Exception as exc:preview_status=f'PREVIEW_ERROR:{exc}'
        except Exception as exc:atlas_report.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
    dump_csv(out/'atlas_vs_mni_mask_geometry.csv',grid_rows,
             ['subject','same_shape','same_affine','max_affine_difference','same_grid','manual_registration_qc'])
    issues=[]
    if not mask_paths:issues.append('No BTC preoperative MNI tumor masks found')
    if len(sc_names)!=68 or sc_names!=EXPECTED:issues.append('Sample SC.zip DK68 names not independently verified in this run')
    if not args.atlas:issues.append('No candidate DK68 voxel atlas explicitly supplied')
    else:
        if atlas_report['status']!='INSPECTED':issues.append('Atlas unreadable or invalid geometry')
        else:
            if atlas_kind(args.atlas)=='DKT_NOT_DK68':issues.append('Supplied atlas is DKT, not Desikan-Killiany')
            if atlas_kind(args.atlas)=='DESTRIEUX_NOT_DK68':issues.append('Supplied atlas is Destrieux, not DK68')
            if not atlas_report['labels'].get('fully_contains_freesurfer_dk68'):
                issues.append('Candidate does not contain all 68 canonical FreeSurfer aparc label IDs; may use alternative labels or be incomplete')
            if not grid_rows or not all(x['same_grid'] for x in grid_rows):
                issues.append('Atlas and all available MNI masks do not share one verified voxel grid')
            if args.atlas_space!='MNI' or not args.atlas_provenance.strip():
                issues.append('MNI spatial provenance not supplied and documented')
    issues.append('Registration correctness requires source transforms and anatomical overlay review; identical grids alone are insufficient')
    issues.append('Tumor masks contain continuous values: binarization/partial-volume policy remains unvalidated')
    report={'phase':'6A','root':str(root),'sc_source':str(zip_path),
            'sample_sc_68_names_match_reference':len(sc_names)==68 and sc_names==EXPECTED,
            'mni_mask_count':len(mask_paths),'readable_mni_masks':sum(r['status']=='READABLE' for r in masks),
            'mask_shape_distribution':dict(Counter(r.get('shape','error') for r in masks)),
            'atlas_candidates':len(candidates),'candidate_scan_truncated':truncated,
            'candidate_type_distribution':dict(Counter(r['type'] for r in candidate_rows)),
            'explicit_atlas':atlas_report,'sample_overlay_status':preview_status,
            'all_masks_same_grid_as_atlas':bool(grid_rows) and all(x['same_grid'] for x in grid_rows),
            'ready_for_regional_tumor_burden':False,'blockers':issues,
            'note':'Never approve registration, assign node-to-volume identity or calculate tumor burden automatically.'}
    (out/'spatial_atlas_qc_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['BTC DK68 SPATIAL ATLAS PREPARATION — PHASE 6A','='*60,
           f'MNI tumor masks: {len(mask_paths)} | readable: {report["readable_mni_masks"]}',
           f'SC DK68 ordered labels match standard: {report["sample_sc_68_names_match_reference"]}',
           f'Atlas candidate image files: {len(candidates)} (truncated: {truncated})',
           f'Explicit candidate atlas: {atlas_report["status"]}',
           f'All mask grids match atlas: {report["all_masks_same_grid_as_atlas"]}',
           f'Overlay: {preview_status}',
           'SPATIAL TUMOR BURDEN: NOT AUTHORIZED', '', 'BLOCKERS / REVIEW ITEMS:',
           *['- '+x for x in issues], '', 'Outputs:',
           '  atlas_candidate_inventory.csv', '  mni_tumor_mask_inventory.csv',
           '  dk68_reference_node_mapping.csv', '  atlas_vs_mni_mask_geometry.csv',
           '  spatial_atlas_qc_summary.json', '  spatial_atlas_qc_summary.txt']
    summary='\n'.join(lines)+'\n'
    (out/'spatial_atlas_qc_summary.txt').write_text(summary,encoding='utf-8')
    print(summary)

if __name__=='__main__':main()
