#!/usr/bin/env python3
"""Phase 5: conservative BTC DK68 identity / atlas / space validation.

Read-only. Does NOT infer node-to-voxel labels from numerical order, does NOT
resample masks, and does NOT claim registration from an affine match alone.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    import nibabel as nib
except ImportError:
    nib = None

PRE = 'Child/openneuro_BTC_preop'
POST = 'Child/openneuro BTC_postop'

def norm_name(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def read_lines(p):
    if not p.is_file():
        return []
    return [x.strip() for x in p.read_text(errors='replace').splitlines() if x.strip() and not x.lstrip().startswith('#')]

def extract_names(text):
    lines = [x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith('#')]
    return [re.split(r'[\t, ]+', x)[0] for x in lines]

def inspect_zip(p):
    out = {'exists':p.is_file(), 'members':[], 'region_labels':[], 'labels_member':None,
           'labels_count':None, 'warnings':[]}
    if not p.is_file(): return out
    try:
        with zipfile.ZipFile(p) as z:
            out['members'] = z.namelist()
            candidates = [x for x in z.namelist() if Path(x).name.lower() in ('region_labels.txt','labels.txt','region_names.txt')]
            if candidates:
                k=candidates[0]; b=z.read(k)
                names=extract_names(b.decode('utf-8',errors='replace'))
                out.update(labels_member=k,region_labels=names,labels_count=len(names))
            else:
                out['warnings'].append('no_named_region_label_list_in_SC_zip')
    except Exception as exc:
        out['warnings'].append(f'zip_error:{type(exc).__name__}:{exc}')
    return out

def read_matrix(p, key):
    if not p.is_file():return None,'missing'
    try:
        v=loadmat(p)[key]
        if v.shape!=(68,68) or not np.isfinite(v).all(): return v,'invalid_shape_or_nonfinite'
        return v,'ok'
    except Exception as exc:return None,f'{type(exc).__name__}:{exc}'

def candidate_atlases(root):
    search_roots=[root/'Child/openneuro_BTC_preop',root/'Child/openneuro BTC_postop',root/'Child/openneuro']
    matches=[]
    pat=re.compile(r'(aparc|dk68|desikan|parcellation|atlas)',re.I)
    for base in search_roots:
        if not base.exists():continue
        for p in base.rglob('*'):
            if p.is_file() and pat.search(p.name) and (p.name.endswith(('.nii','.nii.gz','.mgz'))):
                matches.append(str(p))
    return matches

def inspect_image(p):
    out={'file':str(p),'exists':p.is_file(),'readable':False}
    if not p.is_file():return out
    if nib is None:
        out['error']='nibabel_missing';return out
    try:
        im=nib.load(str(p)); out.update(readable=True,shape=list(im.shape),voxel_mm=list(map(float,im.header.get_zooms()[:3])),
                                     affine=np.asarray(im.affine).round(6).tolist())
        # inspect only candidate atlas integer labels. No automatic anatomical identity inference.
        arr=np.asanyarray(im.dataobj)
        finite=np.isfinite(arr)
        vals=np.unique(arr[finite]) if finite.any() else np.array([])
        out.update(unique_values_count=int(len(vals)),nonzero_labels_count=int(np.count_nonzero(vals)),
                   integer_like=bool(np.allclose(vals,np.rint(vals),atol=1e-5)) if len(vals) else False,
                   labels_preview=[float(x) for x in vals[:24]])
    except Exception as exc:out['error']=f'{type(exc).__name__}:{exc}'
    return out

def load_label_map(p):
    if not p:return None,[]
    issues=[]
    try:
        with p.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        needed={'node_index','label_value','region_name'}
        if not rows or not needed.issubset(rows[0]):return None,['label_map_requires_node_index,label_value,region_name']
        rec=[(int(x['node_index']),int(x['label_value']),x['region_name'].strip()) for x in rows]
        if len(rec)!=68 or sorted(x[0] for x in rec)!=list(range(68)) or len({x[1] for x in rec})!=68:
            issues.append('label_map_requires_68_unique_nodes_zero_based_0_to_67_and_unique_label_values')
        if any(v==0 for _,v,_ in rec):issues.append('label_zero_reserved_for_background')
        return rec,issues
    except Exception as exc:return None,[f'label_map_error:{exc}']

def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path,help='MRI Data directory')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--atlas',type=Path,help='Optional HUMAN DK68 volume image; exact MNI reference must be independently proven')
    p.add_argument('--label-map',type=Path,help='Optional independently sourced CSV: node_index (0..67),label_value,region_name')
    p.add_argument('--sample-subject',default='sub-PAT01')
    a=p.parse_args();root=a.root.expanduser().resolve();out=a.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    pre=root/PRE;post=root/POST
    if not (pre/'derivatives/TVB').is_dir():p.error(f'BTC TVB root not found: {pre}')
    parcellation_path=pre/'derivatives/TVB/parcellation.txt'
    parc_lines=read_lines(parcellation_path)
    subject=a.sample_subject
    sample=pre/'derivatives/TVB'/subject/'ses-preop'
    z=inspect_zip(sample/'SC.zip')
    sc,sc_status=read_matrix(sample/'SCthrAn.mat','SCthrAn')
    fc,fc_status=read_matrix(sample/'FC.mat','FC_cc_DK68')
    # Text parcellation is not assumed to be row-wise label list. Inspect instead of asserting order.
    parc_tokens=extract_names('\n'.join(parc_lines))
    sequence_match=(len(parc_tokens)==68 and len(z['region_labels'])==68 and
                    [norm_name(x) for x in parc_tokens]==[norm_name(x) for x in z['region_labels']])
    zip_labels_unique=(len(z['region_labels'])==68 and len(set(map(norm_name,z['region_labels'])))==68)
    atlas=inspect_image(a.atlas.expanduser().resolve()) if a.atlas else None
    masks={}
    for space in ('T1','MNI'):
        mask=pre/'derivatives/tumor_masks'/subject/'anat'/f'{subject}_space_{space}_label-tumor.nii'
        if nib is not None and mask.is_file():
            try:
                im=nib.load(str(mask));masks[space]={'file':str(mask),'shape':list(im.shape),'affine':im.affine.tolist(),'voxel_mm':list(map(float,im.header.get_zooms()[:3]))}
            except Exception as exc:masks[space]={'error':str(exc)}
        else:masks[space]={'file':str(mask),'exists':mask.is_file(),'status':'nibabel_missing_or_file_missing'}
    geometry_match=None
    if atlas and atlas.get('readable') and 'shape' in masks.get('MNI',{}):
        geometry_match=bool(atlas['shape']==masks['MNI']['shape'] and
                            np.allclose(atlas['affine'],masks['MNI']['affine'],atol=1e-4,rtol=0))
    label_map,label_map_issues=load_label_map(a.label_map) if a.label_map else (None,[])
    mapping_check='not_tested'
    mapping_rows=[]
    if label_map is not None:
        if label_map_issues:
            mapping_check='invalid_label_map'
        else:
            expected=z['region_labels']
            names_match=bool(len(expected)==68 and all(norm_name(name)==norm_name(expected[idx]) for idx,_,name in label_map))
            atlas_labels_ok=bool(atlas and atlas.get('readable') and atlas.get('integer_like') and
                                 all(float(v) in set(atlas['labels_preview']) for _,v,_ in label_map)) if False else None
            # Validate full atlas membership separately: preview alone is insufficient.
            if atlas and atlas.get('readable') and nib is not None:
                values=set(np.unique(np.asanyarray(nib.load(atlas['file']).dataobj)).tolist())
                atlas_labels_ok=all(v in values for _,v,_ in label_map)
            mapping_check=('matched_names_and_labels' if names_match and atlas_labels_ok else
                           'mismatch_or_missing_atlas_or_zip_names')
            for idx,v,name in sorted(label_map):
                mapping_rows.append({'node_index':idx,'label_value':v,'region_name':name,
                    'sczip_region_name':expected[idx] if len(expected)>idx else '',
                    'name_match':norm_name(name)==norm_name(expected[idx]) if len(expected)>idx else False,
                    'label_in_atlas':atlas_labels_ok if atlas_labels_ok is not None else ''})
    rows=[]
    for kind,folder,session in [('preop',pre,'ses-preop'),('postop',post,'ses-postop')]:
        tvb=folder/'derivatives/TVB'
        if not tvb.exists():continue
        for subject_dir in sorted(tvb.glob('sub-*')):
            if not subject_dir.is_dir():continue
            d=subject_dir/session
            if not d.exists():
                subfolders=[x for x in subject_dir.iterdir() if x.is_dir()]
                if len(subfolders)==1:d=subfolders[0]
            zz=inspect_zip(d/'SC.zip')
            ss,s_status=read_matrix(d/'SCthrAn.mat','SCthrAn')
            ff,f_status=read_matrix(d/'FC.mat','FC_cc_DK68')
            labels=zz['region_labels']
            label_status=('matches_sample' if zip_labels_unique and len(labels)==68 and
                          [norm_name(x) for x in labels]==[norm_name(x) for x in z['region_labels']]
                          else ('not_available' if not labels else 'mismatch_or_unverified'))
            rows.append({'subject':subject_dir.name,'session':kind,'sc_status':s_status,'fc_status':f_status,
                         'sczip_exists':zz['exists'],'sczip_label_count':zz['labels_count'],
                         'sczip_label_status':label_status,'sc_fc_shapes_68':bool(ss is not None and ff is not None and ss.shape==ff.shape==(68,68))})
    write_csv(out/'matrix_node_inventory.csv',rows,['subject','session','sc_status','fc_status','sczip_exists','sczip_label_count','sczip_label_status','sc_fc_shapes_68'])
    write_csv(out/'node_label_comparison.csv',mapping_rows,['node_index','label_value','region_name','sczip_region_name','name_match','label_in_atlas'])
    candidates=candidate_atlases(root)
    (out/'atlas_candidates.txt').write_text('\n'.join(candidates)+'\n',encoding='utf-8')
    blockers=[]
    if not zip_labels_unique:blockers.append('SC.zip ordered 68 labels not independently available/unique')
    if not sequence_match:blockers.append('parcellation.txt is not verified as exact SC.zip 68-node order')
    if not a.atlas:blockers.append('No human DK68 voxel atlas supplied')
    elif geometry_match is not True:blockers.append('MNI mask and atlas grid/affine not equal or cannot be inspected')
    if mapping_check!='matched_names_and_labels':blockers.append('No verified node_index-to-atlas-label mapping')
    blockers.append('Even matching geometry does not establish anatomical registration correctness; overlay/manual and provenance QC needed')
    report={'root':str(root),'sample_subject':subject,'sc_status':sc_status,'fc_status':fc_status,
            'sc_zip':z,'parcellation_txt':str(parcellation_path),'parcellation_lines':len(parc_lines),
            'parcellation_first_lines':parc_lines[:10],'parcellation_exact_order_matches_sczip':sequence_match,
            'sample_zip_68_unique_names':zip_labels_unique,'atlas':atlas,'mask_geometry':masks,
            'atlas_mni_mask_same_grid':geometry_match,'label_map_validation':mapping_check,
            'label_map_issues':label_map_issues,'matrix_records':len(rows),
            'matrix_label_status_counts':{k:sum(x['sczip_label_status']==k for x in rows) for k in ('matches_sample','not_available','mismatch_or_unverified')},
            'atlas_candidates_found':len(candidates),'blockers':blockers,
            'ready_for_tumor_burden':False,
            'note':'No spatial tumor burden calculated. Matching labels/grid alone is insufficient for published spatial validity.'}
    (out/'atlas_validation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['BTC DK68 ATLAS VALIDATION — PHASE 5','='*58,
        f'Sample: {subject} | SC={sc_status} FC={fc_status}',
        f'SC.zip labels: {z["labels_count"]} | unique ordered 68: {zip_labels_unique}',
        f'Parcellation rows: {len(parc_lines)} | exact order matches SC.zip: {sequence_match}',
        f'Atlas candidate files: {len(candidates)}',
        f'Atlas provided: {bool(a.atlas)} | same grid/affine as sample MNI mask: {geometry_match}',
        f'Explicit node-to-label mapping: {mapping_check}',
        f'Matrix session records: {len(rows)}',
        'REGIONAL TUMOR BURDEN: NOT AUTHORIZED BY THIS AUTOMATED QC',
        '', 'BLOCKERS / MANUAL CHECKS:',*[f'- {x}' for x in blockers],
        '', 'Outputs: matrix_node_inventory.csv, node_label_comparison.csv, atlas_candidates.txt, atlas_validation_summary.json']
    summary='\n'.join(lines)+'\n';(out/'atlas_validation_summary.txt').write_text(summary,encoding='utf-8');print(summary)

if __name__=='__main__':main()
