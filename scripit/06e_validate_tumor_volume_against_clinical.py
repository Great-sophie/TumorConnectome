#!/usr/bin/env python3
"""BTC Phase 6E: exploratory comparison of MNI threshold volumes,
native-T1 mask volumes and recorded clinical tumor size.

No reference source is treated as ground truth; no threshold is approved.
No claim about atlas registration is made.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def numeric(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (ValueError, TypeError):
        return np.nan


def summarize_pairs(data, xcol, ycol, threshold, target):
    paired = data[[xcol, ycol]].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    x, y = paired[xcol].to_numpy(float), paired[ycol].to_numpy(float)
    n = len(paired)
    row = {'threshold':threshold, 'reference':target, 'n':n, 'spearman_rho':np.nan,
           'spearman_p':np.nan,'mean_difference_cm3':np.nan,'median_difference_cm3':np.nan,
           'bland_altman_lower_cm3':np.nan,'bland_altman_upper_cm3':np.nan,
           'median_abs_percent_error':np.nan,'median_volume_ratio':np.nan}
    if not n:
        return row
    diff = x-y
    row['mean_difference_cm3']=float(np.mean(diff))
    row['median_difference_cm3']=float(np.median(diff))
    valid = y>0
    if np.any(valid):
        row['median_abs_percent_error']=float(np.median(np.abs(diff[valid])/y[valid])*100)
        row['median_volume_ratio']=float(np.median(x[valid]/y[valid]))
    if n>=2:
        sd = float(np.std(diff, ddof=1))
        row['bland_altman_lower_cm3']=float(np.mean(diff)-1.96*sd)
        row['bland_altman_upper_cm3']=float(np.mean(diff)+1.96*sd)
        if np.std(x)>0 and np.std(y)>0:
            res=spearmanr(x,y)
            row['spearman_rho']=float(res.statistic)
            row['spearman_p']=float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
    return row


def save_figures(df, dest, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True,exist_ok=True)
    for ycol,name in [('native_t1_gt0_cm3','nativeT1_gt0'),('native_t1_gt0p5_cm3','nativeT1_gt0p5'),('clinical_cm3','clinical')]:
        paired=df.loc[np.isclose(df.threshold,threshold),['subject','mni_volume_cm3',ycol]].dropna()
        if paired.empty: continue
        x=paired[ycol].to_numpy(float); y=paired.mni_volume_cm3.to_numpy(float)
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(x,y,s=32)
        lim=max(float(np.max(x)),float(np.max(y)),0.1)*1.05
        ax.plot([0,lim],[0,lim],'--',linewidth=1)
        ax.set(xlabel=f'{name} volume (cm³)',ylabel=f'MNI mask > {threshold:g} volume (cm³)',title=f'Volume comparison, n={len(paired)} (exploratory)')
        fig.tight_layout(); fig.savefig(dest/f'scatter_{name}_t{threshold:g}.png',dpi=160); plt.close(fig)
        means=(x+y)/2; diffs=y-x
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(means,diffs,s=32)
        ax.axhline(float(np.mean(diffs)),linestyle='--')
        if len(diffs)>1:
            std=float(np.std(diffs,ddof=1)); mu=float(np.mean(diffs))
            ax.axhline(mu+1.96*std,linestyle=':'); ax.axhline(mu-1.96*std,linestyle=':')
        ax.set(xlabel='Mean of two volumes (cm³)',ylabel='MNI minus reference (cm³)',title=f'Bland–Altman descriptive, n={len(paired)}')
        fig.tight_layout(); fig.savefig(dest/f'bland_altman_{name}_t{threshold:g}.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='BTC tumor volume comparisons (descriptive, no threshold approval).')
    ap.add_argument('--root',default='/media/sophie/Data2T/MRI Data')
    ap.add_argument('--threshold-csv',default=None,help='06d threshold_metrics_per_subject.csv')
    ap.add_argument('--clinical-csv',default=None,help='02b clinical_cohort.csv')
    ap.add_argument('--out',default='../results/tumor_volume_clinical_validation')
    ap.add_argument('--figure-threshold',type=float,default=0.1)
    ap.add_argument('--no-figures',action='store_true')
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve(); out=Path(args.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    threshold_path=Path(args.threshold_csv) if args.threshold_csv else root/'Child/TumorConnectome/results/tumor_mask_threshold_validation/threshold_metrics_per_subject.csv'
    clinical_path=Path(args.clinical_csv) if args.clinical_csv else root/'Child/TumorConnectome/results/clinical/clinical_cohort.csv'
    for path in (threshold_path,clinical_path):
        if not path.is_file(): ap.error(f'Required file not found: {path}')
    thr=pd.read_csv(threshold_path)
    clinical=pd.read_csv(clinical_path)
    required_thr={'subject','threshold','volume_cm3'}
    required_clin={'subject','tumor_size_pre_cm3'}
    if not required_thr.issubset(thr.columns): ap.error(f'Threshold CSV missing: {required_thr-set(thr.columns)}')
    if not required_clin.issubset(clinical.columns): ap.error(f'Clinical CSV missing: {required_clin-set(clinical.columns)}')
    thr=thr.copy();thr['threshold']=pd.to_numeric(thr.threshold,errors='coerce')
    if thr.duplicated(['subject','threshold']).any(): ap.error('Duplicate subject+threshold pairs in threshold CSV')
    if clinical.duplicated('subject').any():ap.error('Duplicate subjects in clinical CSV')
    clinical=clinical[['subject','tumor_size_pre_cm3']].rename(columns={'tumor_size_pre_cm3':'clinical_cm3'})
    clinical['clinical_cm3']=clinical.clinical_cm3.map(numeric)
    native={}; failures=[]
    import nibabel as nib
    subjects=sorted(thr.subject.dropna().unique())
    mask_base=root/'Child/openneuro_BTC_preop/derivatives/tumor_masks'
    for subj in subjects:
        paths=sorted((mask_base/subj/'anat').glob('*_space_T1_label-tumor.nii*'))
        if len(paths)!=1:
            failures.append({'subject':subj,'issue':f'Expected exactly one native T1 mask, found {len(paths)}'})
            native[subj]={'native_t1_gt0_cm3':np.nan,'native_t1_gt0p5_cm3':np.nan,'native_t1_max':np.nan,'native_t1_source':''}
            continue
        try:
            img=nib.load(str(paths[0])); arr=np.asarray(img.dataobj,dtype=np.float32)
            if arr.ndim!=3: raise ValueError(f'non-3D mask {arr.shape}')
            voxel=abs(float(np.linalg.det(img.affine[:3,:3])))
            if voxel<=0: raise ValueError('invalid voxel volume')
            native[subj]={'native_t1_gt0_cm3':float(np.count_nonzero(np.isfinite(arr)&(arr>0))*voxel/1000),
                          'native_t1_gt0p5_cm3':float(np.count_nonzero(np.isfinite(arr)&(arr>0.5))*voxel/1000),
                          'native_t1_max':float(np.nanmax(arr)),'native_t1_source':str(paths[0])}
        except Exception as exc:
            failures.append({'subject':subj,'issue':f'{type(exc).__name__}: {exc}'})
            native[subj]={'native_t1_gt0_cm3':np.nan,'native_t1_gt0p5_cm3':np.nan,'native_t1_max':np.nan,'native_t1_source':str(paths[0])}
    native_df=pd.DataFrame.from_dict(native,orient='index').rename_axis('subject').reset_index()
    merged=thr.rename(columns={'volume_cm3':'mni_volume_cm3'}).merge(native_df,on='subject',how='left',validate='many_to_one').merge(clinical,on='subject',how='left',validate='many_to_one')
    for ref in ('native_t1_gt0_cm3','native_t1_gt0p5_cm3','clinical_cm3'):
        merged[f'delta_vs_{ref}']=merged.mni_volume_cm3-merged[ref]
        merged[f'ratio_vs_{ref}']=np.where(merged[ref]>0,merged.mni_volume_cm3/merged[ref],np.nan)
    merged.to_csv(out/'volume_comparisons_per_subject.csv',index=False)
    rows=[]
    for t in sorted(merged.threshold.dropna().unique()):
        sample=merged[np.isclose(merged.threshold,t)]
        for reference in ('native_t1_gt0_cm3','native_t1_gt0p5_cm3','clinical_cm3'):
            rows.append(summarize_pairs(sample,'mni_volume_cm3',reference,float(t),reference))
    summary=pd.DataFrame(rows); summary.to_csv(out/'agreement_statistics_by_threshold.csv',index=False)
    # Flag records with large discrepancy against either reference; a *screening flag*, not a diagnosis.
    flagged=merged[(np.isclose(merged.threshold,args.figure_threshold)) & (
        (np.isfinite(merged.ratio_vs_clinical_cm3) & ((merged.ratio_vs_clinical_cm3>2)|(merged.ratio_vs_clinical_cm3<0.5))) |
        (np.isfinite(merged.ratio_vs_native_t1_gt0p5_cm3) & ((merged.ratio_vs_native_t1_gt0p5_cm3>2)|(merged.ratio_vs_native_t1_gt0p5_cm3<0.5)))
    )]
    flagged.to_csv(out/'large_discrepancy_review.csv',index=False)
    pd.DataFrame(failures,columns=['subject','issue']).to_csv(out/'processing_issues.csv',index=False)
    if not args.no_figures:
        if not any(np.isclose(merged.threshold,args.figure_threshold)):
            ap.error(f'Figure threshold {args.figure_threshold} not present in threshold data')
        save_figures(merged,out/'figures',args.figure_threshold)
    summary_info={'subjects_in_threshold_data':len(subjects),'thresholds':sorted(merged.threshold.dropna().unique().tolist()),
                  'native_t1_masks_readable':int(native_df.native_t1_gt0_cm3.notna().sum()),
                  'clinical_volume_available':int(clinical[clinical.subject.isin(subjects)].clinical_cm3.notna().sum()),
                  'native_t1_criteria':['>0','>0.5'], 'figure_threshold':args.figure_threshold,
                  'flagged_subjects_at_figure_threshold':int(flagged.subject.nunique()),
                  'native_t1_processing_issues':len(failures),
                  'atlas_anatomical_registration':'UNVERIFIED','selected_threshold':'NONE',
                  'tumor_to_DK68_burden_authorized':False,
                  'cautions':['Clinical and segmentation volumes may represent different lesion definitions.',
                              'Native and MNI masks need not have identical volumes after registration/resampling.',
                              'Spearman measures ranking association, not agreement.',
                              'Bland–Altman limits are descriptive, unstable in small samples.',
                              'Large discrepancy thresholds (ratio outside 0.5–2) are screening heuristics only.',
                              'No clinical or native volume is treated as ground truth.']}
    (out/'volume_validation_summary.json').write_text(json.dumps(summary_info,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['BTC TUMOR VOLUME AGREEMENT — PHASE 6E','='*58,
           f"Patients: {len(subjects)} | Native T1 masks: {summary_info['native_t1_masks_readable']} | Clinical volumes: {summary_info['clinical_volume_available']}",
           f"Thresholds: {summary_info['thresholds']}",
           f"Native T1 processing issues: {len(failures)} | Large discrepancy review flags at >{args.figure_threshold:g}: {summary_info['flagged_subjects_at_figure_threshold']}",
           '', 'MNI > threshold vs clinical:']
    for r in rows:
        if r['reference']=='clinical_cm3':
            lines.append(f"  >{r['threshold']:.2f} n={r['n']} rho={r['spearman_rho']:.3f} median ratio={r['median_volume_ratio']:.3f} median abs % error={r['median_abs_percent_error']:.1f}")
    lines+=['','THRESHOLD: NOT APPROVED','ANATOMICAL REGISTRATION: UNVERIFIED','TUMOR–DK68 BURDEN: NOT AUTHORIZED',
            'Outputs: volume_comparisons_per_subject.csv; agreement_statistics_by_threshold.csv; large_discrepancy_review.csv; figures/']
    msg='\n'.join(lines)+'\n';(out/'volume_validation_summary.txt').write_text(msg,encoding='utf-8');print(msg)

if __name__=='__main__':
    main()
