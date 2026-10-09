#!/usr/bin/env python3
"""Exploratory BTC longitudinal network–cognition associations.

Inputs: Phase 03 longitudinal_changes.csv (long format: subject,group,method,
 baseline,followup,delta) and Phase 02b paired_cognitive_outcomes.csv (long:
 subject,group,metric,pre,post,delta_post_minus_pre,paired_scfc_eligible).
Read-only input. This is exploratory, unadjusted inference; no causal claims.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['RVP_A','RVP_probhit','RTI_simpleRT_mean','SOC_prob_minmoves','SSP_spanlength']
PRIMARY = 'RVP_A'
METHOD = 'spearman_all'

def read_rows(p, required):
    if not p.is_file(): raise SystemExit(f'Missing input: {p}')
    with p.open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        absent=set(required)-set(reader.fieldnames or [])
        if absent: raise SystemExit(f'{p} missing columns: {sorted(absent)}')
        return list(reader)

def number(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else None
    except (ValueError,TypeError): return None

def truth(v): return str(v).strip().lower() in ('true','1','yes')

def save_csv(path, rows, columns):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def corr(x,y):
    if len(x)<5 or len(np.unique(x))<2 or len(np.unique(y))<2: return float('nan')
    return float(spearmanr(x,y).statistic)

def perm_p(x,y,observed,rng,n):
    if not np.isfinite(observed): return float('nan')
    count=0
    for _ in range(n):
        yp=rng.permutation(y)
        if abs(corr(x,yp))>=abs(observed)-1e-12: count+=1
    return (count+1)/(n+1)

def bootstrap_ci(x,y,rng,n):
    if len(x)<5: return (float('nan'), float('nan'),0)
    vals=[]
    for _ in range(n):
        ids=rng.integers(0,len(x),len(x))
        v=corr(x[ids],y[ids])
        if np.isfinite(v): vals.append(v)
    if len(vals)<max(100,int(0.5*n)): return (float('nan'),float('nan'),len(vals))
    return (*[float(z) for z in np.percentile(vals,[2.5,97.5])],len(vals))

def bh(rows):
    ids=[i for i,r in enumerate(rows) if np.isfinite(r['permutation_p'])]
    m=len(ids)
    if not m:return
    order=sorted(ids,key=lambda i:rows[i]['permutation_p'])
    q=[0.]*m; running=1.
    for rank in range(m-1,-1,-1):
        running=min(running, rows[order[rank]]['permutation_p']*m/(rank+1))
        q[rank]=running
    for i,v in zip(order,q):rows[i]['fdr_bh_q']=v

def figure_scatter(rows,path,group,metric):
    data=[r for r in rows if r['group']==group and r['metric']==metric]
    fig,ax=plt.subplots(figsize=(6,5))
    if data:
        x=np.array([r['delta_coupling'] for r in data]);y=np.array([r['delta_cognition'] for r in data])
        ax.scatter(x,y,s=34,alpha=.8)
        ax.axvline(0,color='grey',linestyle='--',lw=.8);ax.axhline(0,color='grey',linestyle='--',lw=.8)
    else: ax.text(.5,.5,'No matched observations',transform=ax.transAxes,ha='center')
    ax.set(xlabel='Δ global SC–FC coupling (post − pre)',ylabel=f'Δ {metric} (post − pre)',title=f'{group.title()}: network–cognition changes (exploratory)')
    fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)

def figure_forest(stats,path):
    group=[r for r in stats if r['group']=='patient' and r['method']==METHOD]
    fig,ax=plt.subplots(figsize=(7,4.8))
    for i,r in enumerate(group):
        z=r['rho'];lo=r['ci95_low'];hi=r['ci95_high']
        if all(np.isfinite(v) for v in [z,lo,hi]):
            ax.plot([lo,hi],[i,i],lw=2)
            ax.plot(z,i,'o')
    ax.axvline(0,color='grey',linestyle='--',lw=.8)
    ax.set_yticks(range(len(group)),[f"{r['metric']} (n={r['n']})" for r in group]); ax.invert_yaxis()
    ax.set(xlim=(-1.05,1.05),xlabel='Spearman ρ (bootstrap 95% CI)',title='Patient network–cognition associations')
    fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coupling',type=Path,default=Path('../results/global_coupling/longitudinal_changes.csv'))
    p.add_argument('--clinical',type=Path,default=Path('../results/clinical/paired_cognitive_outcomes.csv'))
    p.add_argument('--out',type=Path,default=Path('../results/network_cognition'))
    p.add_argument('--method',default=METHOD,help='Network coupling method used in Phase 03')
    p.add_argument('--permutations',type=int,default=10000)
    p.add_argument('--bootstraps',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261008)
    args=p.parse_args()
    if args.permutations<100 or args.bootstraps<100: p.error('Use >=100 permutations and bootstraps')
    cr=read_rows(args.coupling,['subject','group','method','delta'])
    nr=read_rows(args.clinical,['subject','group','metric','pre','post','delta_post_minus_pre','paired_scfc_eligible'])
    net={}
    for r in cr:
        if r['method']!=args.method:continue
        key=r['subject'].strip()
        if key in net:raise SystemExit(f'Duplicate coupling row: {key} ({args.method})')
        if r['group'] not in ('patient','control'):raise SystemExit(f'Bad group: {r["group"]}')
        net[key]=r
    if not net:raise SystemExit(f'No coupling rows for method {args.method}')
    matched=[];seen=set();excluded=[]
    for r in nr:
        metric=r['metric']; sid=r['subject'].strip()
        if metric not in METRICS:continue
        key=(sid,metric)
        if key in seen:raise SystemExit(f'Duplicate clinical row {key}')
        seen.add(key)
        if not truth(r['paired_scfc_eligible']):continue
        n=net.get(sid)
        if n is None:
            excluded.append({'subject':sid,'metric':metric,'reason':'no_longitudinal_coupling'});continue
        if n['group']!=r['group']:raise SystemExit(f'Group mismatch for {sid}')
        x=number(n['delta']); pre=number(r['pre']);post=number(r['post']);y=number(r['delta_post_minus_pre'])
        if None in (x,pre,post,y):
            excluded.append({'subject':sid,'metric':metric,'reason':'missing_or_nonfinite'});continue
        if not np.isclose(post-pre,y,atol=1e-6,rtol=1e-6):raise SystemExit(f'Cognitive delta inconsistent for {sid}, {metric}')
        matched.append({'subject':sid,'group':r['group'],'metric':metric,'method':args.method,
                        'delta_coupling':x,'cognition_pre':pre,'cognition_post':post,'delta_cognition':y})
    rng=np.random.default_rng(args.seed)
    stats=[]
    for group in ('patient','control'):
        for metric in METRICS:
            subset=[z for z in matched if z['group']==group and z['metric']==metric]
            x=np.array([z['delta_coupling'] for z in subset],dtype=float)
            y=np.array([z['delta_cognition'] for z in subset],dtype=float)
            rho=corr(x,y);pvalue=perm_p(x,y,rho,rng,args.permutations)
            low,high,bn=bootstrap_ci(x,y,rng,args.bootstraps)
            stats.append({'group':group,'metric':metric,'method':args.method,'n':len(subset),'rho':rho,
                          'ci95_low':low,'ci95_high':high,'permutation_p':pvalue,
                          'bootstrap_valid':bn,'fdr_bh_q':float('nan'),
                          'status':'ok' if np.isfinite(rho) else 'insufficient_or_constant'})
    # FDR is within each group's four predeclared secondary outcomes; primary RVP_A reported separately.
    for group in ('patient','control'):
        secondary=[r for r in stats if r['group']==group and r['metric']!=PRIMARY]
        bh(secondary)
    out=args.out.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    save_csv(out/'matched_subject_changes.csv',matched,
             ['subject','group','metric','method','delta_coupling','cognition_pre','cognition_post','delta_cognition'])
    save_csv(out/'association_statistics.csv',stats,
             ['group','metric','method','n','rho','ci95_low','ci95_high','permutation_p','bootstrap_valid','fdr_bh_q','status'])
    save_csv(out/'excluded_records.csv',excluded,['subject','metric','reason'])
    figure_scatter(matched,figdir/'Fig3A_patient_RVP_A_scatter.png','patient',PRIMARY)
    figure_scatter(matched,figdir/'Fig3B_control_RVP_A_scatter.png','control',PRIMARY)
    figure_forest(stats,figdir/'Fig3C_patient_cognitive_associations.png')
    summary={'method':args.method,'seed':args.seed,'permutations':args.permutations,'bootstraps':args.bootstraps,
             'primary_cognitive_metric':PRIMARY,'matched_records':len(matched),'results':stats,
             'notes':['Unadjusted exploratory analysis; no causal inference or predictive validation.',
                      'RVP_A is a pre-specified candidate in this workflow, not a prospectively registered endpoint.',
                      'The FDR correction applies to four secondary outcomes within each group; RVP_A is reported separately.',
                      'Separate within-group correlations do not establish a group-interaction difference.',
                      'Potential regression to the mean, test-retest effects, follow-up interval and confounding remain.',
                      'DK68 ordering and imaging registration remain unverified.',
                      'Do not upload subject-level outputs without privacy review.']}
    def sanitize(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {k:sanitize(x) for k,x in v.items()}
        if isinstance(v,list):return [sanitize(x) for x in v]
        return v
    (out/'analysis_summary.json').write_text(json.dumps(sanitize(summary),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    lines=['BTC NETWORK–COGNITION ASSOCIATION — PHASE 3B','='*57,
           f'Method: {args.method}; permutations={args.permutations}; bootstraps={args.bootstraps}',
           'Inference: within-group, unadjusted, exploratory; cognitive delta = post − pre','']
    for r in stats:
        lines.append(f"{r['group']:7s} {r['metric']:20s} n={r['n']:2d} rho={r['rho']:+.4f} CI=[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}] p={r['permutation_p']:.4f} q_secondary={r['fdr_bh_q']:.4f}")
    lines+=['','NOTE: All p-values exploratory. RVP_A designated primary candidate before inspecting associations.',
            'NOTE: No cognitive improvement direction inferred; no clinical or causal claims.',
            'NOTE: Node identity and follow-up intervals remain unverified.']
    (out/'analysis_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print(f'\nWrote outputs to {out}')

if __name__=='__main__':main()
