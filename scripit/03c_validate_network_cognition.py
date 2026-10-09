#!/usr/bin/env python3
"""BTC phase 03c: exploratory robustness of longitudinal coupling–RVP_A association.

Reads outputs from 03, 02b. No raw MRI modification. Subject is the unit of inference.
Primary analysis previously observed on 16 patients; these are post-hoc validations,
not independent confirmation. No clinical prediction or causal inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['spearman_all', 'pearson_all', 'spearman_connected', 'spearman_deconv_all']
COGNITIVE = 'RVP_A'

def read_csv(path, needed):
    p = Path(path).expanduser().resolve()
    if not p.is_file(): raise SystemExit(f'File missing: {p}')
    df = pd.read_csv(p, dtype={'subject':'string', 'group':'string'}, low_memory=False)
    missing = set(needed)-set(df.columns)
    if missing: raise SystemExit(f'{p.name} lacks columns: {sorted(missing)}')
    return df

def rho(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if len(x)<5 or not(np.isfinite(x).all() and np.isfinite(y).all()): return np.nan
    if len(np.unique(x))<2 or len(np.unique(y))<2: return np.nan
    return float(spearmanr(x,y).statistic)

def partial_rank(x,y,z):
    """Partial Spearman: regress rank(x), rank(y) on rank covariates and correlate residuals."""
    n=len(x)
    if n<7: return np.nan
    rr=np.column_stack([np.ones(n)]+[rankdata(v) for v in z])
    if np.linalg.matrix_rank(rr)<rr.shape[1] or n <= rr.shape[1]+3: return np.nan
    rx=rankdata(x)-rr @ np.linalg.lstsq(rr,rankdata(x),rcond=None)[0]
    ry=rankdata(y)-rr @ np.linalg.lstsq(rr,rankdata(y),rcond=None)[0]
    if np.std(rx)<1e-12 or np.std(ry)<1e-12: return np.nan
    return float(np.corrcoef(rx,ry)[0,1])

def bootstrap(x,y,z,stat,n,rng):
    vals=[]
    for _ in range(n):
        ind=rng.integers(0,len(x),size=len(x))
        v=stat(x[ind],y[ind],[a[ind] for a in z])
        if np.isfinite(v): vals.append(v)
    if len(vals)<max(50,n//3): return (np.nan,np.nan,len(vals))
    return (*np.percentile(vals,[2.5,97.5]).astype(float),len(vals))

def permute_y(x,y,z,stat,n,rng):
    observed=stat(x,y,z)
    if not np.isfinite(observed): return np.nan
    count=0
    for _ in range(n):
        if abs(stat(x,rng.permutation(y),z))>=abs(observed)-1e-12: count+=1
    return (count+1)/(n+1)

def corr_stat(x,y,z): return rho(x,y) if not z else partial_rank(x,y,z)

def summarize(x,y,covars,n_perm,n_boot,rng):
    estimate=corr_stat(x,y,covars)
    if not np.isfinite(estimate): return dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,bootstrap_valid=0)
    lo,hi,k=bootstrap(x,y,covars,corr_stat,n_boot,rng)
    # Covariate-adjusted permutation via residual permutation is implemented separately below.
    if covars:
        # Residual permutation after removing rank covariates (Freedman–Lane-style exploratory test).
        n=len(x); design=np.column_stack([np.ones(n)]+[rankdata(v) for v in covars])
        yr=rankdata(y); fit=design @ np.linalg.lstsq(design,yr,rcond=None)[0]; resid=yr-fit
        xr=rankdata(x)-design @ np.linalg.lstsq(design,rankdata(x),rcond=None)[0]
        yr_resid=yr-fit
        obs=np.corrcoef(xr,yr_resid)[0,1]
        cnt=0
        for _ in range(n_perm):
            rp=rng.permutation(resid)
            stat=np.corrcoef(xr,rp)[0,1]
            if abs(stat)>=abs(obs)-1e-12: cnt+=1
        pv=(cnt+1)/(n_perm+1)
    else:
        pv=permute_y(x,y,[],corr_stat,n_perm,rng)
    return dict(estimate=estimate,ci_low=lo,ci_high=hi,p=pv,bootstrap_valid=k)

def canonical(v):
    s=str(v).strip().lower()
    if s in ('none','nan','na','n/a',''): return 'unknown'
    if 'meningioma' in s:return 'meningioma'
    return 'non_meningioma_tumor'

def summary_table(df,label,covs,n_perm,n_boot,rng):
    fields=['delta','delta_cognition']+covs
    work=df.dropna(subset=fields).copy()
    n=len(work)
    if n<8: return dict(analysis=label,n=n,estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,status='too_few_complete_cases')
    x=work.delta.to_numpy(float);y=work.delta_cognition.to_numpy(float)
    z=[work[c].to_numpy(float) for c in covs]
    res=summarize(x,y,z,n_perm,n_boot,rng)
    return dict(analysis=label,n=n,**res,status='exploratory')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--coupling',required=True,help='03 longitudinal_changes.csv')
    ap.add_argument('--clinical',required=True,help='02b paired_cognitive_outcomes.csv')
    ap.add_argument('--cohort',required=True,help='02b clinical_cohort.csv')
    ap.add_argument('--out',required=True)
    ap.add_argument('--permutations',type=int,default=10000)
    ap.add_argument('--bootstraps',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=20261008)
    a=ap.parse_args()
    if a.permutations<100 or a.bootstraps<100: ap.error('Use at least 100 resamples for each')
    rng=np.random.default_rng(a.seed)
    cc=read_csv(a.coupling,['subject','group','method','baseline','delta'])
    cognitive=read_csv(a.clinical,['subject','group','metric','pre','post','delta_post_minus_pre','paired_scfc_eligible'])
    meta=read_csv(a.cohort,['subject','group','age_pre','sex_pre','diagnosis_pre','tumor_size_pre_cm3'])
    # Explicitly enforce unique subject-method and cognitive subject-metric records.
    if cc.duplicated(['subject','method']).any():raise SystemExit('Duplicate subject/method coupling rows')
    if cognitive.duplicated(['subject','metric']).any():raise SystemExit('Duplicate subject/metric cognitive rows')
    if meta.duplicated(['subject']).any():raise SystemExit('Duplicate subject in clinical cohort')
    cognitive=cognitive[cognitive.metric==COGNITIVE].copy()
    cognitive=cognitive[cognitive.paired_scfc_eligible.astype(str).str.lower().isin(['true','1','yes'])]
    cognitive=cognitive.rename(columns={'pre':'cognition_pre','post':'cognition_post','delta_post_minus_pre':'delta_cognition'})
    merged=cc.merge(cognitive[['subject','group','cognition_pre','cognition_post','delta_cognition']],on=['subject','group'],how='inner',validate='many_to_one')
    merged=merged.merge(meta[['subject','age_pre','sex_pre','diagnosis_pre','tumor_size_pre_cm3']],on='subject',how='left',validate='many_to_one')
    for field in ['baseline','delta','cognition_pre','cognition_post','delta_cognition','age_pre','tumor_size_pre_cm3']:
        merged[field]=pd.to_numeric(merged[field],errors='coerce')
    merged['diagnosis_category']=merged.diagnosis_pre.map(canonical)
    merged=merged.replace([np.inf,-np.inf],np.nan)
    primary=merged[(merged.group=='patient') & (merged.method=='spearman_all')].dropna(subset=['baseline','delta','cognition_pre','delta_cognition']).copy()
    if len(primary)<8:raise SystemExit(f'Primary patient sample too small (n={len(primary)}); inspect subject matching.')
    results=[]
    def add(df,label,covars):results.append(summary_table(df,label,covars,a.permutations,a.bootstraps,rng))
    add(primary,'patient_unadjusted',[])
    add(primary,'patient_adjust_baseline_cognition',['cognition_pre'])
    add(primary,'patient_adjust_baseline_both',['cognition_pre','baseline'])
    add(primary,'patient_adjust_age',['age_pre'])
    add(primary,'patient_adjust_tumor_volume',['tumor_size_pre_cm3'])
    # Diagnose individual influence without selecting observations to remove.
    loo=[]
    for sid in primary.subject:
        d=primary[primary.subject!=sid]
        loo.append(dict(subject=sid,n=len(d),rho=rho(d.delta,d.delta_cognition)))
    # Predeclared method sensitivity: same endpoint, individual patients.
    sensitivity=[]
    for method in METHODS:
        dat=merged[(merged.group=='patient')&(merged.method==method)].dropna(subset=['delta','delta_cognition'])
        res=summarize(dat.delta.to_numpy(float),dat.delta_cognition.to_numpy(float),[],a.permutations,a.bootstraps,rng) if len(dat)>=8 else dict(estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan)
        sensitivity.append(dict(method=method,n=len(dat),**res))
    # Stratified descriptive check only; underpowered for subtype claims.
    subtype=[]
    for k,g in primary.groupby('diagnosis_category'):
        subtype.append(dict(diagnosis_category=k,n=len(g),rho=rho(g.delta,g.delta_cognition)))
    # Between-group difference in rho, permutation of group labels across individuals.
    comparison=merged[merged.method=='spearman_all'].dropna(subset=['delta','delta_cognition']).copy()
    pat=comparison[comparison.group=='patient'];con=comparison[comparison.group=='control']
    interaction={'n_patient':len(pat),'n_control':len(con),'rho_patient':rho(pat.delta,pat.delta_cognition),'rho_control':rho(con.delta,con.delta_cognition)}
    if len(pat)>=8 and len(con)>=8:
        obs=interaction['rho_patient']-interaction['rho_control']; labels=comparison.group.to_numpy(); x=comparison.delta.to_numpy(float);y=comparison.delta_cognition.to_numpy(float)
        hits=0
        for _ in range(a.permutations):
            perm=rng.permutation(labels);rp=rho(x[perm=='patient'],y[perm=='patient']);rc=rho(x[perm=='control'],y[perm=='control'])
            if np.isfinite(rp) and np.isfinite(rc) and abs(rp-rc)>=abs(obs)-1e-12:hits+=1
        interaction.update(difference_rho=obs,permutation_p=(hits+1)/(a.permutations+1),note='Unadjusted group-label permutation; relies on exchangeability')
    out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out/'robustness_statistics.csv',index=False)
    pd.DataFrame(loo).to_csv(out/'leave_one_out.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'method_sensitivity.csv',index=False)
    pd.DataFrame(subtype).to_csv(out/'tumor_type_descriptive.csv',index=False)
    pd.DataFrame([interaction]).to_csv(out/'group_correlation_difference.csv',index=False)
    # Minimal plot without displaying subject IDs in the public-facing graphic.
    fig1,ax=plt.subplots(figsize=(6.6,4.2));ax.axvline(0,color='gray',linestyle='--',lw=1)
    vals=[v['rho'] for v in loo if np.isfinite(v['rho'])];ax.plot(vals,range(len(vals)),'o',markersize=4)
    ax.axvline(rho(primary.delta,primary.delta_cognition),color='tab:orange',ls='--',label='Full sample')
    ax.set(xlabel='Spearman rho after removing one patient',ylabel='Leave-one-out iteration',title='RVP_A association influence analysis')
    ax.legend();fig1.tight_layout();fig1.savefig(fig/'Fig4A_leave_one_out.png',dpi=300);plt.close(fig1)
    fig2,ax=plt.subplots(figsize=(7,4.2));ys=np.arange(len(sensitivity))
    ax.axvline(0,color='gray',linestyle='--',lw=1)
    for i,r in enumerate(sensitivity):
        if np.isfinite(r['estimate']):
            ax.plot(r['estimate'],i,'o')
            if np.isfinite(r['ci_low']) and np.isfinite(r['ci_high']):ax.plot([r['ci_low'],r['ci_high']],[i,i],'-')
    ax.set_yticks(ys,[x['method'] for x in sensitivity]);ax.invert_yaxis();ax.set(xlabel='Spearman rho (95% bootstrap CI)',title='Method sensitivity (exploratory)')
    fig2.tight_layout();fig2.savefig(fig/'Fig4B_method_sensitivity.png',dpi=300);plt.close(fig2)
    summary={'primary_n':len(primary),'primary_rho':rho(primary.delta,primary.delta_cognition),'leave_one_out_min':float(np.nanmin(vals)) if vals else None,'leave_one_out_max':float(np.nanmax(vals)) if vals else None,'results':results,'interaction':interaction,'notes':['Post-hoc robustness checks; not independent validation','Baseline-adjusted associations use partial rank residualization, not a causal model','Adjusting one covariate at a time avoids high-dimensional models in n=16','Group-label permutation requires exchangeability; patient/control differ clinically','No individual-level data should be uploaded publicly without privacy review','DK68 node order checked in Phase 05b v2; interval confounding and voxelwise atlas alignment remain unresolved']}
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=lambda obj:None if pd.isna(obj) else str(obj)),encoding='utf-8')
    lines=['BTC NETWORK–COGNITION ROBUSTNESS — PHASE 3C','='*56,f'Primary: n={len(primary)}, Spearman rho={summary["primary_rho"]:.4f}',f'Leave-one-out range: [{summary["leave_one_out_min"]:.4f}, {summary["leave_one_out_max"]:.4f}]','']
    for r in results: lines.append(f'{r["analysis"]:38s} n={r["n"]:2d} estimate={r["estimate"]:+.4f} CI=[{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}] p={r["p"]:.4f}')
    lines+=['',f'Patient-control correlation difference: {interaction.get("difference_rho",np.nan):+.4f}; exploratory permutation p={interaction.get("permutation_p",np.nan):.4f}', '', 'NOTE: All results exploratory, post-hoc robustness checks, not independent confirmation.','NOTE: Follow-up intervals and clinical confounding remain unresolved; DK68 node ordering checked separately.']
    (out/'validation_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print(f'\nWrote results: {out}')

if __name__=='__main__':main()
