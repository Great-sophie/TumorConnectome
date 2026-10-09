#!/usr/bin/env python3
"""BTC paper analysis: global SC–FC coupling and longitudinal contrasts.

Read-only source inputs. Analysis eligibility reflects *numeric* QC only, not
validation of DK68 node identity, clinical metadata, or MRI quality.

Primary exploratory estimand: difference in mean subject-level change in
whole-connectome Spearman SC–FC coupling (patient minus control).
Permutation test shuffles group labels of complete-case subject-level changes;
bootstrap CI resamples subjects within groups. Neither implies causal effects.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import rankdata, pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KEYS = {'sc': 'SCthrAn', 'fc': 'FC_cc_DK68', 'fc_deconv': 'FC_cc_DK68_deconv'}
SESSIONS = ('preop', 'postop')
METHODS = ('spearman_all', 'pearson_all', 'spearman_connected', 'spearman_log1p_all', 'spearman_deconv_all')
PRIMARY = 'spearman_all'


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path, help='MRI Data root (contains Child/...)')
    p.add_argument('--cohort', type=Path, default=None, help='analysis_cohort.csv from Phase 2')
    p.add_argument('--out', type=Path, default=None, help='Output directory')
    p.add_argument('--permutations', type=int, default=10000)
    p.add_argument('--bootstraps', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261008)
    return p.parse_args()


def rows_read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def is_true(v):
    return str(v).strip().lower() in ('true', '1', 'yes')


def finite_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def matrix(path, key, kind):
    if not path.is_file():
        raise ValueError(f'Missing file: {path}')
    x = np.asarray(loadmat(path, variable_names=[key])[key], dtype=float)
    if x.shape != (68, 68) or not np.isfinite(x).all():
        raise ValueError(f'{kind}: invalid shape/nonfinite: {x.shape}')
    if not np.allclose(x, x.T, atol=1e-6):
        raise ValueError(f'{kind}: asymmetry')
    if kind == 'sc':
        if np.min(x) < -1e-8 or not np.allclose(np.diag(x), 0, atol=1e-6):
            raise ValueError('SC negative or diagonal nonzero')
    else:
        if np.min(x) < -1.000001 or np.max(x) > 1.000001 or not np.allclose(np.diag(x), 1, atol=1e-6):
            raise ValueError('FC range/diagonal invalid')
    return x


def safe_corr(a, b, how):
    if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    if how == 'spearman':
        return float(spearmanr(a, b).statistic)
    return float(pearsonr(a, b).statistic)


def coupling(sc, fc, deconv):
    tri = np.triu_indices(68, k=1)
    s, f = sc[tri], fc[tri]
    fd = deconv[tri] if deconv is not None else None
    connected = s > 0
    return {
        'spearman_all': safe_corr(s, f, 'spearman'),
        'pearson_all': safe_corr(s, f, 'pearson'),
        'spearman_connected': safe_corr(s[connected], f[connected], 'spearman'),
        # log1p is monotonic, hence Spearman invariance is expected; serves as a check.
        'spearman_log1p_all': safe_corr(np.log1p(s), f, 'spearman'),
        'spearman_deconv_all': safe_corr(s, fd, 'spearman') if fd is not None else float('nan'),
        'n_edges_all': int(len(s)), 'n_edges_connected': int(connected.sum()),
    }


def p_permutation(a, b, rng, n):
    obs = float(np.mean(a)-np.mean(b))
    pool = np.r_[a, b]
    na = len(a)
    count = 0
    for _ in range(n):
        perm = rng.permutation(len(pool))
        d = float(np.mean(pool[perm[:na]])-np.mean(pool[perm[na:]]))
        count += abs(d) >= abs(obs)-1e-14
    return (count+1)/(n+1)


def boot_ci(a, b, rng, n):
    vals = np.empty(n)
    for i in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals[i] = aa.mean()-bb.mean()
    return np.quantile(vals, [.025, .975]).tolist()


def fig_baseline(subject_rows, output):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for j, g in enumerate(('patient', 'control')):
        vals = [x[PRIMARY] for x in subject_rows if x['group'] == g and x['session']=='preop' and np.isfinite(x[PRIMARY])]
        if vals:
            ax.boxplot([vals], positions=[j], widths=.4, showfliers=False)
            offsets = np.linspace(-.13, .13, len(vals))
            ax.scatter(j+offsets, vals, s=21)
    ax.set_xticks([0,1], ['Patients (baseline)', 'Controls (baseline)'])
    ax.set_ylabel('Global SC–FC Spearman coupling')
    ax.set_title('Baseline coupling (exploratory)')
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def fig_trajectories(change_rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(9,4), sharey=True)
    for ax,g in zip(axes,('patient','control')):
        group=[x for x in change_rows if x['group']==g and x['method']==PRIMARY]
        for x in group:
            ax.plot([0,1], [x['baseline'],x['followup']], marker='o', alpha=.45, linewidth=.8)
        if group:
            ax.plot([0,1], [np.mean([x['baseline'] for x in group]), np.mean([x['followup'] for x in group])],
                    color='black', marker='s', linewidth=2.5, label='Group mean')
        ax.set_xticks([0,1], ['Baseline','Follow-up'])
        ax.set_title(f'{g.title()} (n={len(group)})')
        ax.legend(loc='best') if group else None
    axes[0].set_ylabel('Global SC–FC Spearman coupling')
    fig.tight_layout(); fig.savefig(output,dpi=300); plt.close(fig)


def fig_changes(change_rows, output):
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    for j,g in enumerate(('patient','control')):
        vals=[x['delta'] for x in change_rows if x['group']==g and x['method']==PRIMARY]
        if vals:
            ax.boxplot([vals],positions=[j],widths=.4,showfliers=False)
            ax.scatter(j+np.linspace(-.12,.12,len(vals)),vals,s=23)
    ax.axhline(0, linestyle='--', linewidth=.8, color='black')
    ax.set_xticks([0,1],['Patients','Controls'])
    ax.set_ylabel('Follow-up minus baseline coupling')
    ax.set_title('Individual longitudinal changes')
    fig.tight_layout();fig.savefig(output,dpi=300);plt.close(fig)


def fig_sensitivity(stats, output):
    names=[]; ys=[]; lo=[]; hi=[]
    for r in stats:
        if r['analysis']=='longitudinal_difference_in_changes' and r['status']=='ok':
            names.append(r['method'].replace('_',' '));ys.append(r['estimate'])
            lo.append(r['estimate']-r['ci95_low']);hi.append(r['ci95_high']-r['estimate'])
    fig, ax=plt.subplots(figsize=(7,4.8))
    if ys:
        ax.errorbar(ys, np.arange(len(ys)), xerr=[lo,hi],fmt='o',capsize=3)
        ax.set_yticks(np.arange(len(ys)),names)
    ax.axvline(0, color='black',linestyle='--',linewidth=.8)
    ax.set_xlabel('Patient minus control mean change (95% bootstrap CI)')
    ax.set_title('Method sensitivity: exploratory')
    fig.tight_layout();fig.savefig(output,dpi=300);plt.close(fig)


def main():
    args=get_args()
    if args.permutations < 100 or args.bootstraps < 100:
        raise SystemExit('Use at least 100 permutations and 100 bootstraps')
    cohort=args.cohort or args.root/'Child/TumorConnectome/results/cohort/analysis_cohort.csv'
    out=args.out or args.root/'Child/TumorConnectome/results/global_coupling'
    rows=rows_read(cohort)
    if not rows:
        raise SystemExit(f'Empty cohort: {cohort}')
    required={'subject','group','baseline_eligible','longitudinal_eligible','preop_valid_sc_fc','postop_valid_sc_fc'}
    if not required.issubset(rows[0]):
        raise SystemExit(f'Cohort lacks columns {required-set(rows[0])}')
    ids=[x['subject'] for x in rows]
    if len(ids)!=len(set(ids)):
        raise SystemExit('Duplicate subject identifiers in cohort')
    pre=args.root/'Child/openneuro_BTC_preop/derivatives/TVB'
    post=args.root/'Child/openneuro_BTC_postop/derivatives/TVB'
    if not post.is_dir():
        alt=args.root/'Child/openneuro BTC_postop/derivatives/TVB'
        if alt.is_dir(): post=alt
    if not pre.is_dir() or not post.is_dir():
        raise SystemExit('BTC TVB input directories not found; check --root')
    records=[]; exclusions=[]
    for r in rows:
        sid=r['subject'];group=r['group']
        if group not in ('patient','control') or not sid.startswith('sub-'):
            raise SystemExit(f'Unexpected cohort group/ID: {sid} {group}')
        for session,base,flag in [('preop',pre,'preop_valid_sc_fc'),('postop',post,'postop_valid_sc_fc')]:
            if not is_true(r[flag]):
                exclusions.append(dict(subject=sid, session=session, reason='not_phase2_numeric_eligible'))
                continue
            path=base/sid/f'ses-{session}'
            try:
                sc=matrix(path/'SCthrAn.mat',KEYS['sc'],'sc')
                fdata=path/'FC.mat'
                fc=matrix(fdata,KEYS['fc'],'fc')
                try:
                    deconv=matrix(fdata,KEYS['fc_deconv'],'fc')
                except (KeyError, ValueError):
                    deconv=None  # optional sensitivity outcome only
                metrics=coupling(sc,fc,deconv)
                if not np.isfinite(metrics[PRIMARY]):
                    raise ValueError('Undefined primary correlation')
                records.append(dict(subject=sid,group=group,session=session,**metrics))
            except (OSError,ValueError,KeyError,TypeError,IndexError,NotImplementedError) as e:
                exclusions.append(dict(subject=sid,session=session,reason=f'{type(e).__name__}: {str(e)[:150]}'))
                print('EXCLUDED',sid,session,str(e))
    index={(x['subject'],x['session']):x for x in records}
    changes=[]
    for r in rows:
        if not is_true(r['longitudinal_eligible']):
            continue
        sid=r['subject'];a=index.get((sid,'preop'));b=index.get((sid,'postop'))
        if not a or not b:
            exclusions.append(dict(subject=sid,session='paired',reason='one_or_both_scan_metrics_unavailable'))
            continue
        for method in METHODS:
            if np.isfinite(a[method]) and np.isfinite(b[method]):
                changes.append(dict(subject=sid,group=r['group'],method=method,
                                    baseline=a[method],followup=b[method],delta=b[method]-a[method]))
            else:
                exclusions.append(dict(subject=sid,session='paired',reason=f'undefined_{method}'))
    rng=np.random.default_rng(args.seed)
    stats=[]
    for method in METHODS:
        for typ in ('baseline_patient_minus_control','longitudinal_difference_in_changes'):
            if typ=='baseline_patient_minus_control':
                a=np.asarray([r[method] for r in records if r['group']=='patient' and r['session']=='preop' and np.isfinite(r[method])])
                b=np.asarray([r[method] for r in records if r['group']=='control' and r['session']=='preop' and np.isfinite(r[method])])
            else:
                a=np.asarray([r['delta'] for r in changes if r['group']=='patient' and r['method']==method])
                b=np.asarray([r['delta'] for r in changes if r['group']=='control' and r['method']==method])
            result=dict(analysis=typ,method=method,n_patient=len(a),n_control=len(b),
                        estimate='',ci95_low='',ci95_high='',permutation_p_two_sided='',status='insufficient_data')
            if len(a)>=2 and len(b)>=2:
                ci=boot_ci(a,b,rng,args.bootstraps)
                result.update(estimate=float(a.mean()-b.mean()),ci95_low=ci[0],ci95_high=ci[1],
                              permutation_p_two_sided=p_permutation(a,b,rng,args.permutations),status='ok')
            stats.append(result)
    out.mkdir(parents=True,exist_ok=True)
    fields=['subject','group','session',*METHODS,'n_edges_all','n_edges_connected']
    write_csv(out/'subject_coupling.csv',records,fields)
    write_csv(out/'longitudinal_changes.csv',changes,['subject','group','method','baseline','followup','delta'])
    write_csv(out/'group_statistics.csv',stats,['analysis','method','n_patient','n_control','estimate','ci95_low','ci95_high','permutation_p_two_sided','status'])
    write_csv(out/'exclusions.csv',exclusions,['subject','session','reason'])
    figs=out/'figures';figs.mkdir(exist_ok=True)
    fig_baseline(records,figs/'Fig2A_baseline_coupling.png')
    fig_trajectories(changes,figs/'Fig2B_longitudinal_trajectories.png')
    fig_changes(changes,figs/'Fig2C_group_change.png')
    fig_sensitivity(stats,figs/'Fig2D_sensitivity.png')
    summary={'cohort_file':str(cohort),'primary_method':PRIMARY,'seed':args.seed,
             'permutations':args.permutations,'bootstraps':args.bootstraps,
             'scan_records':len(records), 'primary_paired_patients':sum(x['group']=='patient' and x['method']==PRIMARY for x in changes),
             'primary_paired_controls':sum(x['group']=='control' and x['method']==PRIMARY for x in changes),
             'exclusion_records':len(exclusions),
             'notes':['Exploratory inference; no causal claims','DK68 SC/FC node ordering checked separately in Phase 05b v2; voxelwise atlas alignment remains unverified',
                      'Group-label permutation assumes exchangeability; confounding not adjusted',
                      'No edge-wise tests; edges are not independent individuals',
                      'SC zeros retained in primary; connected-only and deconvolved FC are sensitivity checks',
                      'Spearman log1p(SC) is mathematically rank-invariant and is a diagnostic, not independent validation',
                      'Subject-level CSVs require privacy review before public dissemination']}
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    primary=next((x for x in stats if x['analysis']=='longitudinal_difference_in_changes' and x['method']==PRIMARY),None)
    lines=['BTC GLOBAL SC–FC COUPLING — PHASE 3','='*53,
           f'Scan records analyzed: {len(records)}',
           f'Primary paired patients: {summary["primary_paired_patients"]}',
           f'Primary paired controls: {summary["primary_paired_controls"]}',
           f'Primary difference in mean change: {primary["estimate"] if primary else "NA"}',
           f'95% bootstrap CI: {[primary["ci95_low"],primary["ci95_high"]] if primary else "NA"}',
           f'Two-sided group-label permutation p: {primary["permutation_p_two_sided"] if primary else "NA"}',
           '',*('CAUTION: '+x for x in summary['notes'])]
    (out/'analysis_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print('\nOutput directory:',out)


if __name__=='__main__':
    main()
