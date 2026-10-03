"""Audit prespecified secondary studies and generate manuscript-ready evidence."""
import argparse
import json
import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.summarize_refinements import audit_predictions,audit_selection,average_precision,roc_auc,NAMES,OWNERS,digest


def audit_exposure(root,record):
    with np.load(root/'reconstruction_exposure.npz',allow_pickle=False) as f:
        reconstruction={k:f[k] for k in f.files}
    groups=[reconstruction[k] for k in ['auxiliary_ids','validation_ids','evaluation_ids']]
    if any(len(x)!=len(np.unique(x)) for x in groups) or any(np.intersect1d(x,y).size for x,y in itertools.combinations(groups,2)):
        raise AssertionError('Reconstruction attack splits overlap or duplicate nodes')
    truth=reconstruction['truth_normalized_features']
    for row in record['reconstruction']:
        pred=reconstruction[row['attack']]
        if pred.shape!=truth.shape or not np.isfinite(pred).all():raise AssertionError('Invalid reconstruction prediction')
        mse=np.mean((truth-pred)**2,axis=0)
        expected=row['evaluation']
        if expected['n']!=len(truth) or not np.allclose(mse,expected['per_feature_mse'],rtol=1e-5,atol=1e-6):
            raise AssertionError('Reconstruction errors differ from saved evidence')
        values=[float(mse.mean()),float(np.sqrt(mse.mean())),float(np.mean(1-mse/np.maximum(truth.var(0),1e-8)))]
        if not np.allclose(values,[expected[k] for k in ['mean_normalized_feature_mse','rmse','mean_feature_r2']],rtol=1e-5,atol=1e-6):
            raise AssertionError('Reconstruction aggregate differs from saved evidence')
    ridge=next(x for x in record['reconstruction'] if x['attack']=='ridge')
    selected=min(ridge['validation_trials'],key=lambda x:x['validation']['mean_normalized_feature_mse'])
    if selected['alpha']!=ridge['selected_alpha']:raise AssertionError('Ridge selection differs from validation trials')
    with np.load(root/'membership_exposure.npz',allow_pickle=False) as f:
        membership={k:f[k] for k in f.files}
    aux,ev=membership['auxiliary_ids'],membership['evaluation_ids']
    if len(aux)!=len(np.unique(aux)) or len(ev)!=len(np.unique(ev)) or np.intersect1d(aux,ev).size:
        raise AssertionError('Membership attack splits overlap or duplicate nodes')
    # Balance is defined before the random auxiliary/evaluation split. Verify
    # that pool, rather than incorrectly describing each split as exactly50%.
    strata=np.column_stack([np.r_[membership['auxiliary_'+k],membership['evaluation_'+k]]
        for k in ['timestep','owner','label','degree_bucket']])
    targets=np.r_[membership['auxiliary_membership'],membership['membership']]
    for stratum in np.unique(strata,axis=0):
        mask=np.all(strata==stratum,axis=1)
        if int(targets[mask].sum())*2!=int(mask.sum()):raise AssertionError('Membership strata are not balanced')
    y=membership['membership']
    for row in record['supervised_membership']:
        score=membership[row['attack']]
        if len(score)!=len(y) or not np.isfinite(score).all():raise AssertionError('Invalid membership scores')
        order=np.argsort(-score,kind='stable');s=score[order];t=y[order]
        end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
        tp=np.r_[0,np.cumsum(t)[end]];fp=np.r_[0,(end+1)-np.cumsum(t)[end]]
        tpr=float((tp/y.sum())[fp/(len(y)-y.sum())<=.01].max())
        if abs(roc_auc(y,score)-row['roc_auc'])>1e-12 or abs(tpr-row['tpr_at_fpr_001'])>1e-12:
            raise AssertionError('Membership operating evidence differs')
        for key,value in [('n_auxiliary',len(aux)),('n_evaluation',len(ev)),
            ('auxiliary_member_count',int(membership['auxiliary_membership'].sum())),('evaluation_member_count',int(y.sum()))]:
            if row[key]!=value:raise AssertionError('Membership support differs')
    return dict(status='passed',reconstruction_n=len(truth),membership_n=len(ev),
        metric_tolerance=1e-12,reconstruction_tolerance=dict(rtol=1e-5,atol=1e-6))


def audit_transport(record):
    rows=record['records']
    if len(rows)!=35 or [r['round'] for r in rows]!=list(range(1,36)) or record['warmup']!=5 or record['steady_state_observations']!=30:
        raise AssertionError('Transport observation inventory differs')
    warm=rows[5:];times=np.array([r['seconds'] for r in warm])
    volumes=[r['application_bytes_received']+r['application_bytes_sent'] for r in warm]
    if not np.isfinite(times).all() or (times<=0).any() or any(v<=0 for v in volumes):raise AssertionError('Invalid transport measurement')
    if abs(float(np.median(times))-record['median_seconds'])>1e-12 or abs(float(np.quantile(times,.95))-record['p95_seconds'])>1e-12:
        raise AssertionError('Transport timing summary differs from observations')
    error=max(r['max_abs_error'] for r in rows)
    if error!=record['max_abs_error'] or max(error,max(r['first_max_abs_error'] for r in rows))>1e-4:
        raise AssertionError('Transport frozen parity failed')
    negatives=record['rejection_checks']
    if ('replay_rejected' not in negatives or
        not any('PEER_DID_NOT_RETURN_A_CERTIFICATE' in x for x in negatives) or
        sum('Wrong identity, recipient, round, replay or layer' in x for x in negatives)<2 or
        not any('malformed tensor shape' in x for x in negatives) or not any('Tampered payload' in x for x in negatives)):
        raise AssertionError('Transport negative inventory differs')
    # The preserved Mac pilot originally summarized the first steady frame.
    # Keep that raw record intact and derive a common median/range from all
    # observations; JSON sequence/round digit length changes frame size.
    expected=int(np.median(volumes)) if 'application_bytes_statistic' in record else volumes[0]
    if record['application_bytes_per_round']!=expected:raise AssertionError('Transport byte summary differs')
    return dict(status='passed',median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
        application_bytes_median=int(np.median(volumes)),application_bytes_minimum=min(volumes),
        application_bytes_maximum=max(volumes),application_bytes_total=sum(volumes),max_abs_error=error)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--main-summary',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads(Path(a.main_summary).read_text());manifest=json.loads((root/'ablations-v2/manifest.json').read_text())
    if manifest['status']!='completed' or len(manifest['results'])!=36:raise AssertionError('Ablations incomplete')
    expected_cells={f'{s}-o{o}-{suffix}' for s,o,suffix in itertools.product(['random','louvain'],OWNERS,
        ['coverage0.0','coverage0.25','coverage0.5','stale5','current-dropout0','fresh-dropout0'])}
    files=list((root/'ablations-v2').glob('*/result.json'))
    if {f.parent.name for f in files}!=expected_cells:raise AssertionError('Ablation artifact inventory differs')
    gate=json.loads((root/'elliptic-main-gate-v2.json').read_text())
    if manifest['source_sha256']!=gate['source_sha256'] or manifest['dataset']!=gate['dataset']:raise AssertionError('Ablation source/data differs')
    with np.load(next((root/'elliptic-main-v2').glob('*/validation_predictions.npz')),allow_pickle=False) as f:
        elliptic_steps={int(i):int(s) for i,s in zip(f['ids'],f['timestep'])}
    interventions=[]
    for cell in files:
        r=json.loads(cell.read_text());_,metrics=audit_predictions(cell.parent/'test_predictions.npz',r)
        audit_selection(cell.parent,r,elliptic_steps,'elliptic')
        config=dict(gate['selected_configs']['sage_fl'])
        if cell.parent.name.endswith('dropout0'):config['dropout']=0.
        if r['config']!=config or r['seed']!=42 or len(r['history'])!=50 or r['ownership_hash']!=gate['ownership_sha256'][f'{r["strategy"]}-{r["ownership_seed"]}']:
            raise AssertionError('Ablation recipe/ownership differs')
        if cell.parent.name.endswith('coverage0.0'):
            reference=json.loads((root/f'elliptic-main-v2/{r["strategy"]}-o{r["ownership_seed"]}-s42-fedavg/result.json').read_text())
            if abs(metrics['ap']-reference['primary_test_ap'])>1e-8:raise AssertionError('Zero exchange differs from FedAvg')
        interventions.append(dict(cell=cell.parent.name,method=r['method'],strategy=r['strategy'],ownership_seed=r['ownership_seed'],**metrics))
    bank=json.loads((root/'bank-ownership-v2/manifest.json').read_text())
    if bank['status']!='completed' or len(bank['results'])!=18:raise AssertionError('Bank grouping incomplete')
    bank_files=list((root/'bank-ownership-v2').glob('*/result.json'))
    if {f.parent.name for f in bank_files}!={f's{s}-{m}' for s,m in itertools.product([42,123,456],['local','fedavg','current','fresh','fedmlp','fedgcn'])}:
        raise AssertionError('Bank artifact inventory differs')
    ibm_gate=json.loads((root/'ibm-main-gate-v2.json').read_text())
    if bank['source_sha256']!=ibm_gate['source_sha256'] or bank['dataset']!=ibm_gate['dataset']:raise AssertionError('Bank source/data differs')
    with np.load(next((root/'ibm-main-v2').glob('*/validation_predictions.npz')),allow_pickle=False) as f:
        ibm_steps={int(i):int(s) for i,s in zip(f['ids'],f['timestep'])}
    bank_rows=[]
    for cell in bank_files:
        r=json.loads(cell.read_text());_,metrics=audit_predictions(cell.parent/'test_predictions.npz',r)
        audit_selection(cell.parent,r,ibm_steps,'ibm')
        family={'fedmlp':'mlp_fl','fedgcn':'fedgcn'}.get(r['method'],'sage_fl')
        if r['config']!=ibm_gate['selected_configs'][family] or r['ownership_hash']!=bank['ownership_hash'] or len(r['history'])!=50:
            raise AssertionError('Bank recipe/ownership differs')
        bank_rows.append(dict(method=r['method'],seed=r['seed'],**metrics))
    bank_effect=[next(r['ap'] for r in bank_rows if r['method']=='fresh' and r['seed']==s)-next(r['ap'] for r in bank_rows if r['method']=='fedavg' and r['seed']==s) for s in [42,123,456]]
    exposure=json.loads((root/'exposure-v2/exposure_results.json').read_text())
    exposure_audit=audit_exposure(root/'exposure-v2',exposure)
    transports=[]
    for folder in ['transport-mac-pilot-v2','transport-v2']:
        record=json.loads((root/folder/'transport_results.json').read_text())
        transports.append(dict(folder=folder,measurement=record,audit=audit_transport(record)))
    transport=transports[-1]['measurement']
    result=dict(ablations=interventions,bank_results=bank_rows,bank_step_minus_fedavg=bank_effect,
        bank_support=bank['support'],bank_ownership_hash=bank['ownership_hash'],exposure=exposure,exposure_audit=exposure_audit,transport=transport,transports=transports)
    (out/'secondary_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    tex=['\\subsection{Costs and Mechanism Interventions}',
        '\\begin{table}[H]\\centering\\small',
        '\\caption{Mean logical training representation/model payload estimates and train-plus-selection time under random ownership. Fanout deduplicates sources per recipient; expanded per-edge state volume is separately archived. Metadata and wire framing are excluded. Pooled controls and feature halos have distinct information access. Host time is descriptive, not isolated GPU architecture.}',
        '\\begin{tabular}{llrrrr}\\toprule','Source & Control & Boundary MiB & Updates MiB & Halo MiB & Seconds \\\\ \\midrule']
    for dataset,method in itertools.product(['elliptic','ibm'],['fedavg','current','fresh','oracle','fedgcn']):
        r=next(v for v in summary['summary'] if v['dataset']==dataset and v['strategy']=='random' and v['method']==method)
        tex.append((dataset.title() if dataset == "elliptic" else "IBM")+' & '+NAMES[method]+' & '+' & '.join(f'{r[k]/1048576:.2f}' for k in ['boundary_bytes','model_update_bytes','halo_feature_bytes'])+f' & {r["seconds"]:.2f} \\\\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}',
          'All six zero-availability interventions reproduce their paired FedAvg AP within the registered numerical tolerance. The full per-cell coverage, cache-age and dropout-zero evidence is supplied; no intervention is selected based on its test result.']
    tex+=['\\begin{table}[H]\\centering\\small','\\caption{Fixed Elliptic mechanism interventions on GTX, averaged over three ownership draws at seed 42. All are exploratory.}',
          '\\begin{tabular}{lrr}\\toprule','Intervention & Random AP & Louvain AP \\\\ \\midrule']
    for suffix,label in [('coverage0.0','Zero source availability'),('coverage0.25','25\\% source availability'),('coverage0.5','50\\% source availability'),('stale5','Five-round cache'),('current-dropout0','Round exchange, dropout 0'),('fresh-dropout0','Step exchange, dropout 0')]:
        vals=[np.mean([r['ap'] for r in interventions if r['strategy']==s and r['cell'].endswith(suffix)]) for s in ['random','louvain']]
        if any(not np.isfinite(v) for v in vals):raise AssertionError('Ablation inventory differs')
        tex.append(label+' & '+' & '.join(f'{v:.4f}' for v in vals)+' \\\\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}','\\subsection{Synthetic Bank Grouping}',
        'The sender-bank grouping uses one training-volume assignment on GTX. Its three step-exchange minus FedAvg raw-AP effects are '+', '.join(f'{v:+.5f}' for v in bank_effect)+f' (mean {np.mean(bank_effect):+.5f}). These optimization repetitions do not supply ownership-level or real-bank uncertainty.',
        '\\begin{table}[H]\\centering\\small','\\caption{Secondary fixed sender-bank grouping: means of three training seeds on the synthetic IBM proxy.}',
        '\\begin{tabular}{lrrr}\\toprule','Control & Raw AP & Frozen F1 & Recall top 1\\% \\\\ \\midrule']
    for method in ['local','fedavg','current','fresh','fedmlp','fedgcn']:
        rs=[r for r in bank_rows if r['method']==method]
        tex.append(NAMES[method]+' & '+' & '.join(f'{np.mean([r[k] for r in rs]):.4f}' for k in ['ap','f1','recall_at_1pct'])+' \\\\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}','\\subsection{Recipient Exposure and Authenticated Transport}',
        '\\begin{table}[H]\\centering\\small','\\caption{Actual-representation feature reconstruction for recipient 1. Errors use train-standardized features on disjoint foreign test-period nodes; anonymity and temporal shift limit interpretation.}',
        '\\begin{tabular}{lrrr}\\toprule','Attack & Feature MSE & RMSE & Mean feature $R^2$ \\\\ \\midrule']
    for r in exposure['reconstruction']:
        ev=r['evaluation'];tex.append({'mean':'Mean predictor','ridge':'Ridge','nonlinear_mlp':'Nonlinear MLP','logistic':'Logistic','random_forest':'Random forest'}[r['attack']]+' & '+f'{ev["mean_normalized_feature_mse"]:.4f} & {ev["rmse"]:.4f} & {ev["mean_feature_r2"]:.4f} \\\\')
    tex+=['\\bottomrule\\end{tabular}\\end{table}']
    mse={r['attack']:r['evaluation']['mean_normalized_feature_mse'] for r in exposure['reconstruction']}
    tex.append(f"The nonlinear reconstruction attack reduces mean normalized-feature MSE by {100*(1-mse['nonlinear_mlp']/mse['mean']):.1f}\\% relative to the mean predictor. Mean feature $R^2$ averages errors normalized by each test-feature variance, so its ranking can differ from pooled MSE. These finite attacks demonstrate retained feature information, not semantic recovery of anonymous fields or a privacy guarantee.")
    for r in exposure['supervised_membership']:
        tex.append({'mean':'Mean predictor','ridge':'Ridge','nonlinear_mlp':'Nonlinear MLP','logistic':'Logistic','random_forest':'Random forest'}[r['attack']]+f' membership attack ROC AUC is {r["roc_auc"]:.4f}, with TPR {r["tpr_at_fpr_001"]:.4f} at FPR no greater than 1\\% on {r["n_evaluation"]} evaluation nodes ({r["evaluation_member_count"]} members). The auxiliary set has {r["n_auxiliary"]} nodes ({r["auxiliary_member_count"]} members); the combined pool is balanced within each matching stratum before the random split.')
    for row in transports:
        ev=row['audit'];label='GTX/Mac NumPy' if row['folder']=='transport-mac-pilot-v2' else 'GTX/RTX CUDA'
        tex.append(f'The {label} snapshot probe retains 30 observations after five warmup exchanges: median {ev["median_seconds"]*1000:.2f} ms and 95th percentile {ev["p95_seconds"]*1000:.2f} ms. Application frame volume has median {ev["application_bytes_median"]:,} bytes and range {ev["application_bytes_minimum"]:,}--{ev["application_bytes_maximum"]:,} bytes per observation. Maximum transported/in-process logit error is {ev["max_abs_error"]:.2e}.')
    tex.append('Peer-authentication, recipient, stale-round, malformed-shape, authenticated-client tampering and replay negative checks reject their inputs. The parity oracle is a benchmark reference, not a defense against malicious model behavior. Source states are prepared before measurement; these probes do not measure transported training convergence or full wire overhead, and concurrent host work precludes isolated hardware or WAN-performance inference.')
    (out/'refined_secondary.tex').write_text('\n'.join(tex)+'\n')
    report=['# Secondary study evidence','',f'36 fixed Elliptic interventions;18 fixed synthetic-bank runs;two placement-specific transport probes of30steady observations each.','',
        'Bank step-exchange minus FedAvg AP by training seed: '+', '.join(f'{x:+.5f}' for x in bank_effect),
        '',json.dumps(dict(reconstruction=exposure['reconstruction'],membership=exposure['supervised_membership']),indent=2),
        '',f'Transport median {transport["median_seconds"]*1000:.2f}ms;p95 {transport["p95_seconds"]*1000:.2f}ms;max parity error {transport["max_abs_error"]:.2e}.',
        'No formal privacy, post-quantum deployment or real-bank generalization is inferred.']
    (out/'SECONDARY_RESULTS.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()
