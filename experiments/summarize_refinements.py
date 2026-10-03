"""Independent prediction audit and conditional summaries of the locked study.

Run with NumPy locally; optional matplotlib plots use --figures. No training or
selection occurs here. Main-effect AP is recomputed without sklearn.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

OWNERS=[20261003,20261004,20261005]
SEEDS=[42,123,456]
METHODS=['local','fedavg','current','fresh','shuffled','oracle','centralized','mlp','fedmlp','fedgcn']
TREES=['rf_pooled','rf_local','xgb_pooled','xgb_local']
NAMES={'local':'Local SAGE','fedavg':'FedAvg','current':'Round exchange','fresh':'Step exchange',
       'shuffled':'Shuffled exchange','oracle':'Raw-feature oracle','centralized':'Pooled SAGE',
       'mlp':'Pooled MLP','fedmlp':'FedMLP','fedgcn':'FedGCN adapter',
       'rf_pooled':'Pooled RF','rf_local':'Local RF','xgb_pooled':'Pooled XGB','xgb_local':'Local XGB'}


def audit_frozen_recipe(r,gate,kind):
    family=({'centralized':'sage_central','mlp':'mlp_central','fedmlp':'mlp_fl','fedgcn':'fedgcn'}
        .get(r['method'],'sage_fl')) if kind=='main' else r['method']
    if r['config']!=gate['selected_configs'][family]:raise AssertionError('Main recipe differs from frozen selection')
    if kind=='main':
        config=gate['main_config']
        if len(r['history'])!=config['rounds'] or [h['round'] for h in r['history']]!=list(range(1,config['rounds']+1)):
            raise AssertionError('Training budget differs from frozen gate')
        if r['ownership_hash']!=gate['ownership_sha256'][f'{r["strategy"]}-{r["ownership_seed"]}']:
            raise AssertionError('Run ownership differs from frozen gate')


def average_precision(y,p):
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
    end=np.r_[np.where(p[:-1]!=p[1:])[0],len(p)-1]
    positives=np.cumsum(y)[end]
    return float(np.sum(np.diff(np.r_[0,positives])*positives/(end+1))/max(int(y.sum()),1))


def roc_auc(y,p):
    order=np.argsort(p,kind='stable');y=y[order];p=p[order]
    end=np.r_[np.where(p[:-1]!=p[1:])[0],len(p)-1]
    pos=np.diff(np.r_[0,np.cumsum(y)[end]])
    neg=np.diff(np.r_[0,np.cumsum(1-y)[end]])
    return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(y.sum()*(len(y)-y.sum())))


def f1_threshold(y,p):
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
    end=np.r_[np.where(p[:-1]!=p[1:])[0],len(p)-1];tp=np.cumsum(y)[end]
    precision=tp/(end+1);recall=tp/y.sum()
    f1=2*precision*recall/np.maximum(precision+recall,1e-15)
    return float(p[end[np.where(f1==f1.max())[0][0]]])


def audit_predictions(path,r):
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    y,p,cp,ids=a['y'],a['probability'],a['calibrated_probability'],a['ids']
    if not len(ids)==len(np.unique(ids)) or not np.all(ids[:-1]<ids[1:]):
        raise AssertionError(f'Duplicate or unordered node support: {path}')
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not np.isfinite(cp).all():
        raise AssertionError(f'Invalid labels/scores: {path}')
    ap=average_precision(y,p)
    if abs(ap-r['primary_test_ap'])>1e-12:
        raise AssertionError(f'AP evidence differs: {path}')
    threshold=r['test']['threshold'];pred=cp>=threshold
    tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum());fn=int((~pred&(y==1)).sum())
    f1=2*tp/max(2*tp+fp+fn,1);brier=float(np.mean((cp-y)**2))
    if abs(f1-r['test']['f1'])>1e-12 or abs(brier-r['test']['brier'])>1e-12 or abs(roc_auc(y,cp)-r['test']['roc_auc'])>1e-12:
        raise AssertionError(f'Operating evidence differs: {path}')
    expected=dict(n=len(y),tp=tp,fp=fp,fn=fn,tn=int((~pred&(y==0)).sum()),
        precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),prevalence=float(y.mean()))
    if any(abs(float(r['test'][key])-value)>1e-12 for key,value in expected.items()):
        raise AssertionError(f'Confusion/support evidence differs: {path}')
    clipped=np.clip(cp,1e-7,1-1e-7)
    loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped)))
    if abs(loss-r['test']['log_loss'])>1e-12:raise AssertionError(f'Log-loss evidence differs: {path}')
    for recorded in r['test']['reliability']:
        lo,hi=recorded['lo'],recorded['hi']
        mask=(cp>=lo)&((cp<=hi) if hi==1 else (cp<hi))
        if int(mask.sum())!=recorded['n']:raise AssertionError(f'Reliability support differs: {path}')
        for key,value in [('confidence',float(cp[mask].mean()) if mask.any() else None),
            ('observed',float(y[mask].mean()) if mask.any() else None)]:
            if (value is None)!=(recorded[key] is None) or (value is not None and abs(value-recorded[key])>1e-12):
                raise AssertionError(f'Reliability scores differ: {path}')
    for budget in [.01,.05]:
        k=int(np.ceil(len(y)*budget));top=np.argsort(-cp,kind='stable')[:k]
        v=r['test']['alert_budgets'][str(budget)]
        if k!=v['k'] or abs(float(y[top].sum()/max(y.sum(),1))-v['recall'])>1e-12 or abs(float(y[top].mean())-v['precision'])>1e-12:
            raise AssertionError(f'Alert evidence differs: {path}')
    return a,dict(ap=ap,f1=f1,brier=brier,precision=r['test']['precision'],recall=r['test']['recall'],
        roc_auc=r['test']['roc_auc'],recall_at_1pct=r['test']['alert_budgets']['0.01']['recall'],
        recall_at_5pct=r['test']['alert_budgets']['0.05']['recall'],n=len(y),positive=int(y.sum()))


def audit_selection(base,r,step_map,dataset):
    with np.load(base/'validation_predictions.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    ids=a['ids'];steps=np.array([step_map[int(i)] for i in ids])
    last=38 if dataset=='elliptic' else 7
    selection=steps!=last;late=steps==last
    if not late.any() or not selection.any():raise AssertionError('Selection/calibration support differs')
    if abs(average_precision(a['y'][selection],a['probability'][selection])-r['selection_ap'])>1e-12:
        raise AssertionError('Selected checkpoint/trial metric does not match validation evidence')
    operating_record=r['test'] if r.get('test') is not None else r['validation']
    if f1_threshold(a['y'][late],a['calibrated_probability'][late])!=operating_record['threshold']:
        raise AssertionError('Test operating threshold differs from later-validation optimum')
    z=np.log(np.clip(a['probability'],1e-6,1-1e-6)/(1-np.clip(a['probability'],1e-6,1-1e-6)))
    cal=r['calibration'];coef=float(np.asarray(cal['coef']).ravel()[0]);intercept=float(np.asarray(cal['intercept']).ravel()[0])
    calculated=1/(1+np.exp(-np.clip(coef*z.astype(np.float64)+intercept,-700,700)))
    # Float32 logit ufuncs differ at their final bit between the original Linux
    # NumPy1.26 build and the independent macOS NumPy2 build. AP/operating
    # metrics use exact saved scores; coefficient reconstruction has a declared
    # absolute2e-6 probability tolerance, unrelated to any utility effect.
    if np.max(np.abs(calculated-a['calibrated_probability']))>2e-6:
        raise AssertionError('Calibration coefficients differ from saved probabilities')
    if r.get('history'):
        candidates=[h for h in r['history'] if 'selection_ap' in h]
        best=max(candidates,key=lambda h:h['selection_ap'])
        if best['round']!=r['selected_round'] or best['selection_ap']!=r['selection_ap']:
            raise AssertionError('Checkpoint is not the earliest validation maximum')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--figures');args=parser.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];ledger={};metadata={};manifests={};support={};temporal=[];step_maps={};raw_maps={}
    source_root=Path(__file__).resolve().parents[1]
    for dataset in ['elliptic','ibm']:
        base=root/'gtx'/f'{dataset}-data-audit-v2'
        audit=json.loads((base/'data_audit.json').read_text())
        identity=audit['raw_identity_audit'];file=base/'node_identity_map.npz'
        if not identity['passed'] or digest(file)!=identity['map_sha256']:raise AssertionError('Original CSV identity audit missing/changed')
        with np.load(file,allow_pickle=False) as f:raw_maps[dataset]={k:f[k] for k in f.files}
        if not np.array_equal(raw_maps[dataset]['node_index'],np.arange(identity['nodes'])):raise AssertionError('Node mapping is not contiguous')
        for path in [file,base/'data_audit.json']:ledger[str(path.relative_to(root))]=digest(path)
    for gpu,dataset,kind in itertools.product(['gtx','rtx'],['elliptic','ibm'],['main','trees-main']):
        base=root/gpu/f'{dataset}-{kind}-v2';manifest=json.loads((base/'manifest.json').read_text())
        manifests[(gpu,dataset,kind)]=manifest
        if manifest['status']!='completed' or manifest.get('failures'):
            raise AssertionError(f'Incomplete inventory: {base}')
        gate_path=root/gpu/f'{dataset}-{"main" if kind=="main" else "trees"}-gate-v2.json'
        gate=json.loads(gate_path.read_text())
        if not gate['passed'] or gate['source_sha256']!=manifest['source_sha256'] or gate['dataset']!=manifest['dataset']:
            raise AssertionError('Main source/data differs from frozen gate')
        if any(digest(source_root/path)!=sha for path,sha in manifest['source_sha256'].items()):
            raise AssertionError('Current numerical source differs from frozen run')
        ledger[str(gate_path.relative_to(root))]=digest(gate_path)
        ledger[str((base/'manifest.json').relative_to(root))]=digest(base/'manifest.json')
        metadata[dataset]=manifest['dataset']
        if dataset not in step_maps:
            vp=next(base.glob('*/validation_predictions.npz'))
            with np.load(vp,allow_pickle=False) as v:
                step_maps[dataset]={int(i):int(s) for i,s in zip(v['ids'],v['timestep'])}
        expected=set()
        methods=METHODS if kind=='main' else TREES
        for method in methods:
            pooled=method in ['centralized','mlp','rf_pooled','xgb_pooled']
            for strategy,owner,seed in itertools.product(['random'] if pooled else ['random','louvain'],
                [OWNERS[0]] if pooled else OWNERS,SEEDS):expected.add((method,strategy,owner,seed))
        observed={(r['method'],r['strategy'],r['ownership_seed'],r['seed']) for r in manifest['results']}
        if observed!=expected or len(manifest['results'])!=len(expected):
            raise AssertionError(f'Missing/duplicated registered cells: {base}')
        records={(r['method'],r['strategy'],r['ownership_seed'],r['seed']):r for r in manifest['results']}
        files=sorted(base.glob('*/result.json'))
        if len(files)!=len(expected):raise AssertionError(f'Artifact count differs from manifest: {base}')
        for cell in files:
            r=json.loads(cell.read_text());path=cell.parent/'test_predictions.npz'
            identity=(r['method'],r['strategy'],r['ownership_seed'],r['seed'])
            if identity not in records or r!=records[identity]:raise AssertionError('Cell receipt differs from manifest')
            audit_frozen_recipe(r,gate,kind)
            a,metrics=audit_predictions(path,r)
            original=raw_maps[dataset]
            expected_ids=np.flatnonzero(np.isin(original['timestep'],[39,40,41,42,43,44,45,46,47,48] if dataset=='elliptic' else [8,9])&(original['y']>=0))
            if (not np.array_equal(a['ids'],expected_ids) or not np.array_equal(a['y'],original['y'][a['ids']]) or
                not np.array_equal(a['timestep'],original['timestep'][a['ids']])):
                raise AssertionError('Prediction IDs/time/outcomes differ from original CSV audit')
            audit_selection(cell.parent,r,step_maps[dataset],dataset)
            key=(dataset,)
            actual=(a['ids'],a['y'])
            if key in support and not all(np.array_equal(x,z) for x,z in zip(support[key],actual)):
                raise AssertionError(f'Support differs across methods/GPUs: {path}')
            support[key]=actual
            row=dict(gpu=gpu,dataset=dataset,method=r['method'],strategy=r['strategy'],
                ownership_seed=r['ownership_seed'],seed=r['seed'],**metrics,
                seconds=r['seconds'],boundary_bytes=r.get('boundary_bytes',0),
                model_update_bytes=r.get('model_update_bytes',0),halo_feature_bytes=r.get('halo_feature_bytes',0),
                expanded_boundary_state_bytes=sum(h.get('received_edges',0) for h in r.get('history',[]))*64*4,
                parameters=r.get('parameters'),
                peak_cuda_bytes=r.get('peak_cuda_bytes'),selected_round=r.get('selected_round'),
                ownership_hash=r['ownership_hash'],initial_state_hash=r.get('initial_state_hash'))
            rows.append(row)
            for step in np.unique(a['timestep']):
                mask=a['timestep']==step
                temporal.append(dict(gpu=gpu,dataset=dataset,method=r['method'],strategy=r['strategy'],
                    ownership_seed=r['ownership_seed'],seed=r['seed'],timestep=int(step)+1,
                    n=int(mask.sum()),positive=int(a['y'][mask].sum()),ap=average_precision(a['y'][mask],a['probability'][mask])))
            for artifact in [cell,path,cell.parent/'validation_predictions.npz',cell.parent/'checkpoint.pt']:
                if artifact.exists():ledger[str(artifact.relative_to(root))]=digest(artifact)
    for dataset,kind in itertools.product(['elliptic','ibm'],['main','trees-main']):
        left=manifests[('gtx',dataset,kind)];right=manifests[('rtx',dataset,kind)]
        if left['source_sha256']!=right['source_sha256'] or left['dataset']!=right['dataset']:
            raise AssertionError('Paired hardware source/data mismatch')
    grid={(r['gpu'],r['dataset'],r['strategy'],r['ownership_seed'],r['seed'],r['method']):r for r in rows}
    for key,left in grid.items():
        if key[0]!='gtx':continue
        right=grid[('rtx',)+key[1:]]
        if left['ownership_hash']!=right['ownership_hash'] or left['initial_state_hash']!=right['initial_state_hash']:
            raise AssertionError('Paired hardware ownership/initialization mismatch')
    effects={}
    for dataset,strategy,method in itertools.product(['elliptic','ibm'],['random','louvain'],['current','fresh','shuffled','oracle']):
        ownership_effect=[];by_hardware={}
        for gpu in ['gtx','rtx']:
            by_hardware[gpu]=[grid[(gpu,dataset,strategy,o,s,method)]['ap']-grid[(gpu,dataset,strategy,o,s,'fedavg')]['ap'] for o,s in itertools.product(OWNERS,SEEDS)]
        for o in OWNERS:
            ownership_effect.append(float(np.mean([grid[(g,dataset,strategy,o,s,method)]['ap']-grid[(g,dataset,strategy,o,s,'fedavg')]['ap'] for g,s in itertools.product(['gtx','rtx'],SEEDS)])))
        effects[f'{dataset}/{strategy}/{method}']=dict(ownership_seeds=OWNERS,ownership_effects=ownership_effect,
            mean=float(np.mean(ownership_effect)),minimum=min(ownership_effect),maximum=max(ownership_effect),
            hardware_mean={g:float(np.mean(v)) for g,v in by_hardware.items()},hardware_cells=by_hardware)
    summary=[]
    for dataset,strategy,method in itertools.product(['elliptic','ibm'],['random','louvain'],METHODS+TREES):
        pooled=method in ['centralized','mlp','rf_pooled','xgb_pooled']
        selected=[r for r in rows if r['dataset']==dataset and r['method']==method and (pooled or r['strategy']==strategy)]
        summary.append(dict(dataset=dataset,strategy=strategy,method=method,runs=len(selected),
            **{m:float(np.mean([r[m] for r in selected])) for m in ['ap','f1','brier','precision','recall','recall_at_1pct','recall_at_5pct','seconds','boundary_bytes','expanded_boundary_state_bytes','model_update_bytes','halo_feature_bytes']}))
    result=dict(status='completed',audited_runs=len(rows),primary=effects['ibm/random/fresh'],effects=effects,
        summary=summary,datasets=metadata,
        interpretation='Conditional means across training seeds and hardware, grouped in three ownership draws; no population confidence interval or t-test.')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'evidence_sha256.json').write_text(json.dumps(ledger,indent=2))
    for filename,data in [('run_grid.csv',rows),('method_summary.csv',summary),('temporal_metrics.csv',temporal)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    report=['# Refined paired study','',f'All {len(rows)} registered neural/tree test runs completed. Independent NumPy auditing verifies AP, F1, Brier, alert budgets and identical test support across methods/GPUs.','',
        'The primary effect is raw-score AP for step exchange minus FedAvg on the fixed synthetic IBM graph under random ownership. Each ownership effect averages three training seeds and both GPUs. These are conditional descriptive effects, not institution-level uncertainty.','',
        '| Dataset / strategy / exchange | Mean AP difference | Ownership range | GTX mean | RTX mean |','|---|---:|---:|---:|---:|']
    for key,v in effects.items():report.append(f'| {key} | {v["mean"]:+.5f} | [{v["minimum"]:+.5f}, {v["maximum"]:+.5f}] | {v["hardware_mean"]["gtx"]:+.5f} | {v["hardware_mean"]["rtx"]:+.5f} |')
    report+=['','| Dataset / strategy | Method | Raw AP | Frozen F1 | Recall top 1% | Calibrated Brier |','|---|---|---:|---:|---:|---:|']
    for r in summary:report.append(f'| {r["dataset"]}/{r["strategy"]} | {NAMES[r["method"]]} | {r["ap"]:.4f} | {r["f1"]:.4f} | {r["recall_at_1pct"]:.4f} | {r["brier"]:.5f} |')
    report+=['','Elliptic is exploratory because its test outcomes were previously inspected. IBM is synthetic and its graph is a fixed sampled continuity proxy. Pooled controls are reused across ownership strategies and are not additional independent runs. Hardware timing compares complete hosts, not isolated GPU architecture. Main runs are in-process simulations. Network transport and recipient attacks have separately declared scopes.']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n')
    tex=[]
    primary=result['primary']
    tex.append(f'The primary IBM random-ownership raw-AP difference (step exchange minus FedAvg) is {primary["mean"]:+.5f}. The three ownership effects are '+', '.join(f'{x:+.5f}' for x in primary['ownership_effects'])+'. Each averages three training seeds and both GPUs; no population-level interval is asserted.\n')
    for dataset in ['elliptic','ibm']:
        tex+=['\\begin{table}[H]\\centering\\small',f'\\caption{{{(dataset.title() if dataset == "elliptic" else "IBM")} test performance, averaged across the registered hardware/seed cells. Pooled controls are reused across strategies. AP uses raw scores; F1 and Brier use validation-frozen calibration.}}',
              '\\begin{tabular}{lrrrrrr}\\toprule','& \\multicolumn{3}{c}{Random ownership} & \\multicolumn{3}{c}{Louvain ownership} \\\\',
              'Method & AP & F1 & Brier & AP & F1 & Brier \\\\ \\midrule']
        for method in METHODS+TREES:
            cells=[next(r for r in summary if r['dataset']==dataset and r['strategy']==s and r['method']==method) for s in ['random','louvain']]
            tex.append(NAMES[method]+' & '+' & '.join((f'{r[m]:.6f}' if dataset=='ibm' and m=='brier' else f'{r[m]:.4f}') for r in cells for m in ['ap','f1','brier'])+' \\\\')
        tex+=['\\bottomrule\\end{tabular}\\end{table}','']
    (out/'refined_results.tex').write_text('\n'.join(tex))
    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        dest=Path(args.figures);dest.mkdir(parents=True,exist_ok=True)
        plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'savefig.bbox':'tight'})
        fig,axes=plt.subplots(1,2,figsize=(7,4.2),sharey=True)
        show=['fedavg','current','fresh','shuffled','oracle','centralized','mlp','fedmlp','fedgcn','rf_pooled','xgb_pooled']
        for ax,dataset in zip(axes,['elliptic','ibm']):
            for i,strategy in enumerate(['random','louvain']):
                vals=[next(r['ap'] for r in summary if r['dataset']==dataset and r['strategy']==strategy and r['method']==m) for m in show]
                ax.plot(vals,np.arange(len(show))+(i-.5)*.18,'o' if i==0 else 's',label=strategy,markersize=4)
            ax.set_yticks(range(len(show)),[NAMES[m] for m in show]);ax.set_xlabel('Raw-score average precision');ax.set_title((dataset.title() if dataset == "elliptic" else "IBM"));ax.legend(frameon=False);ax.grid(axis='x',alpha=.2)
            prevalence=float(support[(dataset,)][1].mean())
            ax.axvline(prevalence,color='gray',ls=':',lw=.9,label='Constant-score AP')
            ax.set_xlim(left=0);ax.legend(frameon=False,fontsize=9)
        axes[0].invert_yaxis()
        fig.tight_layout();fig.savefig(dest/'refined_performance.pdf');fig.savefig(dest/'refined_performance.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(7,3.2))
        for ax,dataset in zip(axes,['elliptic','ibm']):
            for i,strategy in enumerate(['random','louvain']):
                v=effects[f'{dataset}/{strategy}/fresh'];x=np.arange(3)+(i-.5)*.16
                ax.plot(x,v['ownership_effects'],'o',label=strategy)
            ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(3),['03','04','05']);ax.set_xlabel('Ownership seed suffix (202610xx)');ax.set_ylabel('Step exchange minus FedAvg AP');ax.set_title((dataset.title() if dataset == "elliptic" else "IBM"));ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(dest/'refined_ownership_effects.pdf');fig.savefig(dest/'refined_ownership_effects.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(audited_runs=len(rows),primary=primary),indent=2))


if __name__=='__main__':main()
