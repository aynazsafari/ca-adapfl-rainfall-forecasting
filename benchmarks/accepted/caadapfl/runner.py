"""Portable, gated CAAdapFL benchmark orchestration.

``prepare`` validates sources and serializes calendars; it never fits a model.
``run --execute`` performs the frozen development schedule before opening any
evaluation labels. Modules must be installed together under ``caadapfl``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

SEEDS = (42, 123, 2024, 7, 999)
GROUPS = ('CORE', 'CLIMATE', 'FULL', 'TRAIN_SELECTED')
LR_CANDIDATES = (.001, .003)
MU_CANDIDATES = (.001, .01, .1)
COEFFICIENTS = tuple((a/4, b/4, (4-a-b)/4) for a in range(5) for b in range(5-a)) + ((.3, .3, .4),)
MU_METHODS = ('fedprox', 'ditto', 'graph_original', 'graph_enriched', 'graph_uniform', 'caaa')
CA_FLAGS = {'NO_CLIMATE_WEIGHT': {'use_climate': False},
            'NO_PERFORMANCE_WEIGHT': {'use_performance': False},
            'NO_PROX': {'use_prox': False},
            'NO_AUX': {'use_aux': False},
            'NO_EMBEDDING': {'use_embedding': False}}
ROUTER_SUITES = {'validation_best':'validation_best_expert','equal_weight':'equal_weight_experts',
                 'convex':'validation_convex_stack','robust_convex':'robust_validation_router',
                 'temporal_OOF_Ridge_stack':'temporal_OOF_Ridge_stack'}
EVAL_MONTHS = tuple(pd.period_range('2017-01', '2024-12', freq='M').astype(str))
MODULE_NAMES = ('features', 'classical', 'evaluation', 'neural_benchmark', 'graph_profiles')
PROTOCOL_VERSION = 'revised_archive_safe_label_embargo_raw_predictions_v1'


def _mods():
    return {name: importlib.import_module('caadapfl.' + name) for name in MODULE_NAMES}


def _jsonable(value):
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return _jsonable(value.tolist())
    if isinstance(value, np.generic): return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def runtime_environment():
    """Scientific runtime identity, independent of checkout/run/install paths.

    This is deliberately limited to libraries used by the benchmark. Thread
    limits are an execution contract, not a mutable dependency fingerprint.
    """
    packages = {}
    for distribution, module_name in (('numpy','numpy'), ('pandas','pandas'),
            ('scipy','scipy'), ('scikit-learn','sklearn'), ('torch','torch'),
            ('joblib','joblib'), ('threadpoolctl','threadpoolctl')):
        module = importlib.import_module(module_name)
        packages[distribution] = dict(distribution_version=importlib.metadata.version(distribution),
                                      loaded_version=str(module.__version__))
    torch = importlib.import_module('torch')
    build = torch.__config__.show()
    return dict(schema='scientific_runtime_v1',
                python=dict(implementation=platform.python_implementation(), version=sys.version,
                            compiler=platform.python_compiler(), word_bits=64 if sys.maxsize>2**32 else 32),
                operating_system=dict(system=platform.system(), release=platform.release(),
                                      version=platform.version(), architecture=platform.machine()),
                packages=packages,
                torch_build=dict(git_version=torch.version.git_version, debug=bool(torch.version.debug),
                                 cuda=torch.version.cuda, hip=torch.version.hip,
                                 compiled_config=build, compiled_config_sha256=hashlib.sha256(build.encode()).hexdigest()))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('tmp_' + uuid.uuid4().hex[:16])
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def fit_path(root,key):
    # Short physical names avoid Windows MAX_PATH; full hashes are authenticated
    # in every complete receipt, so a prefix collision fails without overwrite.
    return Path(root)/'fits'/key[:16]


def _without_runtime_root(value):
    """Runtime directory is provenance; every other identity field is retained."""
    if isinstance(value,dict):return {k:_without_runtime_root(v) for k,v in value.items() if k!='root'}
    if isinstance(value,(list,tuple)):return [_without_runtime_root(v) for v in value]
    return value


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if _without_runtime_root(read_json(path)) != _without_runtime_root(_jsonable(value)): raise ValueError(f'Stale artifact conflicts with current protocol: {path}')
    else: save_json(path, value)


def protocol():
    comparisons = []
    for group in GROUPS:
        for comparator in ('FEDAVG_GLOBAL', 'FEDPROX_GLOBAL'):
            comparisons.append((f'CAAA_GLOBAL_{group}', f'{comparator}_{group}'))
    comparisons += [(f'CAAA_PERSONAL_FULL', f'{model}_FULL') for model in
                    ('FEDAVG_PERSONAL', 'FEDPROX_PERSONAL', 'DITTO_PERSONAL', 'LOCAL_PERSONAL')]
    comparisons += [('CAAA_GLOBAL_FULL_ANCHOR',f'CAAA_GLOBAL_FULL_{flag}') for flag in CA_FLAGS]
    comparisons += [('CAAA_GLOBAL_CORE','CAAA_GLOBAL_CORE_CLIMATE_FREE')]
    comparisons += [('GRAPH_ENRICHED_PERSONAL_FULL',f'GRAPH_{variant}_PERSONAL_FULL') for variant in ('ORIGINAL','UNIFORM')]
    comparisons += [(f'CAAA_GLOBAL_LATENCY_LAG1', 'CAAA_GLOBAL_LATENCY_REFERENCE'),
                    (f'FEDAVG_GLOBAL_LATENCY_LAG1', 'FEDAVG_GLOBAL_LATENCY_REFERENCE'),
                    (f'FEDPROX_GLOBAL_LATENCY_LAG1', 'FEDPROX_GLOBAL_LATENCY_REFERENCE')]
    for pool in ('CORE', 'FULL'):
        comparisons += [(f'ROUTER_{pool}_robust_convex', f'ROUTER_{pool}_{other}') for other in
                        ('validation_best', 'equal_weight', 'convex', 'temporal_OOF_Ridge_stack')]
    return dict(version=PROTOCOL_VERSION, seeds=SEEDS,
                neural=dict(rounds=60, local_epochs=3, batch_size=32, cpu_threads=2,
                            shared_lr_candidates=LR_CANDIDATES, mu_candidates=MU_CANDIDATES,
                            coefficient_candidates=COEFFICIENTS, primary_rainfall='raw inverse-transformed mm; no clipping',
                            checkpoint_selection='best macro outer-validation R2; all 60 rounds, no early stopping',
                            shared_lr_selector='FULL FedAvg global, seed42, baseline-favoring shared choice',
                            mu_selector='global CAAA/FedProx; personal Ditto/each named graph',
                            coefficient_selector='CAAA global; same frozen algorithm emits global and personal identities',
                            exact_ties='smaller learning rate; smaller mu; coefficient anchor (.3,.3,.4), then lexicographic'),
                feature_groups=dict(CORE=14, CLIMATE=21, FULL=43, TRAIN_SELECTED='training-only nested Ridge fixed-group selector'),
                selected_rule='three fixed groups; alpha10 Ridge; gain >=.002 vs CORE; training-only chronological folds',
                development_folds=(('1995-11','1996-01','1998-12'),('2000-11','2001-01','2003-12'),('2004-11','2005-01','2007-11')),
                role_bounds=dict(train=('1983-01','2007-11'), monitor=('2008-01','2010-11'), validation=('2011-01','2016-11'), evaluation=('2017-01','2024-12')),
                embargo='exclude December2007/2010/2016 labels from roles; OOF fit_end <= first heldout target minus2 months',
                latency=dict(reference_extra_lag=0, shifted_extra_lag=1, common_start='1983-02', train_per_client=298, main_train_per_client=299, frozen_hyperparameters='reuse main FULL choices without retuning'),
                router=dict(pools=('six pooled/local Ridge/ET/RF CORE latest-row', 'same six FULL latest-row'),
                            calibration=('2011-01','2013-11'), score=('2014-01','2016-11'), refit_end='2016-11',
                            score_interpretation='heldout score belongs to provisional calibration weights, not final refit weights',
                            robust_objective='.65 calibration overall R2 + .35 calibration last-half R2', stack='training temporal OOF standardized Ridge alpha10'),
                cpu=dict(max_processes=2, threads_per_process=2, maximum_active_threads=4),
                runtime_reproducibility='Freeze Python/OS/architecture, seven scientific-library versions and PyTorch build/git signature; reject dependency changes on fit/evaluation resume. Runtime directory relocation is allowed.',
                primary_metrics=('equal-client macro R2','equal-client macro RMSE mm'),
                bootstrap=dict(repetitions=2000, blocks=(12,6,24), seed=731, paired_seeds=True, joint_date_blocks=True, estimand='mean per-seed macro metrics'),
                comparisons=comparisons, sign_randomization='optional; block12 + Holm over this full predeclared family',
                ablations=CA_FLAGS, true_core_climate_free='CORE14; no climate embedding, weighting or adaptive prox, zero ENSO/IOD drivers and sensitivity; requires separately matched CORE reference',
                ablation_coefficient_reference=dict(parent_label='CAAA_GLOBAL_FULL_ANCHOR / CAAA_PERSONAL_FULL_ANCHOR',coefficients=(.3,.3,.4),
                    rule='All five flags are compared with the fixed nonzero anchor parent; selected primary remains separate and may have zero aggregation coefficients. Do not restrict candidate grid.'),
                supplements=dict(IMD='disabled until root supplies matched complete-case cohort', GLDAS='disabled until root supplies matched 2001-04 start cohort; no old three-fold selector'),
                interpretation='revised archival hindcast; availability scenarios, five fixed regions; no historically-vintaged prospective evaluation claim')


def schedule():
    """Symbolic dependencies are saved before any developmental score exists."""
    entries = []
    def add(stage, family, **kw): entries.append(dict(stage=stage, family=family, **kw))
    for lr in LR_CANDIDATES: add('LR', 'fedavg', group='FULL', seed=42, lr=lr, identity='global')
    for method in MU_METHODS:
        for mu in MU_CANDIDATES: add('MU', method, group='FULL', seed=42, lr='$chosen_LR', mu=mu, identity='global' if method in ('caaa','fedprox') else 'personal')
    for coeff in COEFFICIENTS: add('COEFFICIENTS', 'caaa', group='FULL', seed=42, lr='$chosen_LR', mu='$chosen_caaa_MU', coefficients=coeff, identity='global')
    for group in GROUPS:
        for seed in SEEDS:
            for method in ('fedavg','fedprox','caaa'): add('FINAL_NEURAL', method, group=group, seed=seed, identities=('global','personal'), choices='$frozen', alias_if_identical=True)
            for family in ('Ridge','ET','RF'):
                for scope in ('pooled','local'): add('FINAL_CLASSICAL', family, scope=scope, group=group, seed=seed, representation='latest', alias_if_identical=True)
    for seed in SEEDS:
        for method in ('central','local','ditto','graph_original','graph_enriched','graph_uniform'): add('FINAL_NEURAL', method, group='FULL', seed=seed, choices='$frozen')
        add('FINAL_NEURAL', 'caaa', group='FULL', seed=seed, variant='ANCHOR', coefficients=(.3,.3,.4), choices='$frozen LR/CAprox')
        for flag, overrides in CA_FLAGS.items(): add('FINAL_NEURAL', 'caaa', group='FULL', seed=seed, variant=flag, overrides=overrides, coefficients=(.3,.3,.4), choices='$frozen LR/CAprox')
        add('FINAL_NEURAL', 'caaa', group='CORE', seed=seed, variant='CLIMATE_FREE', choices='$frozen')
        for scenario in ('LATENCY_REFERENCE','LATENCY_LAG1'):
            for method in ('fedavg','fedprox','caaa'): add('FINAL_NEURAL', method, group='FULL', seed=seed, scenario=scenario, choices='$frozen')
        for scope in ('pooled','local'): add('FINAL_CLASSICAL','ET',scope=scope,group='FULL',seed=seed,representation='flat_sequence')
        add('FINAL_CLASSICAL','ET',scope='pooled',group='FULL',seed=seed,representation='latest',target_mode='client_normalized')
        for pool in ('CORE','FULL'): add('ROUTER', 'six_fixed_classical_experts', group=pool, seed=seed, OOF_folds=3, suites=5)
    add('BASELINES','three_training_only_controls',seeds=SEEDS)
    return dict(protocol=protocol(), entries=entries, counts_by_stage={stage:sum(e['stage']==stage for e in entries) for stage in sorted({e['stage'] for e in entries})},
                note='Counts are scheduled fits/bundles before explicit identical-config aliases; OOF bundles contain 18 expert fits each. No Cartesian tuning expansion.')


def _identity(root, monthly, indices, manifest):
    mods = _mods(); m = read_json(manifest)
    if m.get('source_gate', {}).get('passed') is not True: raise ValueError('Manifest must explicitly pass the authoritative source gate')
    hashes = m.get('inputs', {})
    for name,path in (('monthly',monthly),('indices',indices)):
        claimed=hashes.get(name, {}).get('sha256')
        if not isinstance(claimed,str) or file_hash(path)!=claimed: raise ValueError(f'{name}: input hash missing or changed')
    met = tuple(m.get('feature_contract', {}).get('met_columns', ()))
    if len(met)!=22 or len(set(met))!=22: raise ValueError('Primary manifest must name exactly 22 unique meteorology columns in order')
    code={name:file_hash(mods[name].__file__) for name in MODULE_NAMES}
    code['runner']=file_hash(__file__)
    source=dict(monthly_sha256=file_hash(monthly),indices_sha256=file_hash(indices),manifest_sha256=file_hash(manifest),met_columns=met)
    identity=dict(protocol_hash=digest(protocol()),schedule_hash=digest(schedule()),source=source,code=code,
                  runtime_environment=runtime_environment())
    identity['run_sha256']=digest(identity)
    return mods,m,met,identity


def _array_archive(path, raw, scaled, fs, evaluation=False):
    arrays={}; role_names=('evaluation',) if evaluation else ('train','monitor','validation')
    for c,d in raw.items():
        roles=scaled[c]['roles']; mask=np.logical_or.reduce([roles[k] for k in role_names])
        for key in ('x','climate','drivers','target_month','origin_month'):
            arrays[c+'::raw_'+key]=np.asarray(d[key],dtype=str if key.endswith('_month') else None)[mask]
        for key in ('x','climate'):
            arrays[c+'::scaled_'+key]=np.asarray(scaled[c][key])[mask]
        if not evaluation:
            for key in ('y','y_z','heavy'): arrays[c+'::'+key]=np.asarray(d[key] if key=='y' else scaled[c][key])[mask]
        arrays[c+'::target_stats']=np.array([scaled[c]['target_mean'],scaled[c]['target_std']],dtype=np.float64)
    np.savez_compressed(path,**arrays)


def prepare(monthly, indices, manifest, output):
    started=time.perf_counter()
    output=Path(output).resolve(); mods,m,met,identity=_identity(output,monthly,indices,manifest)
    output.mkdir(parents=True,exist_ok=True)
    immutable_json(output/'run_identity.json',identity)
    immutable_json(output/'protocol.json',protocol())
    immutable_json(output/'schedule.json',schedule())
    if (output/'prepared.json').exists():
        receipt=read_json(output/'prepared.json'); _check_files(output,receipt['files'])
        return receipt
    if (output/'datasets').exists(): raise ValueError('Incomplete prior preparation: use a new output directory; existing data will not be overwritten')
    f=mods['features']; panel=f.checked_panel(monthly,identity['source']['monthly_sha256'],['rainfall_mm',*met]); idx=pd.read_csv(indices)
    if not {'month',*f.INDEX_NAMES}.issubset(idx): raise ValueError('All six exact index columns and end-month keys are required')
    target_dir=output/'datasets'; target_dir.mkdir()
    scenarios={}; truth=None
    for scenario,extra,start in (('MAIN',0,'1983-01'),('LATENCY_REFERENCE',0,'1983-02'),('LATENCY_LAG1',1,'1983-02')):
        rows,fs=f.build_rows(panel,idx,met_columns=met,with_climate=True,extra_lag=extra)
        raw=f.samples(panel,rows,fs,start_target=start)
        groups=mods['classical'].default_feature_groups(fs.names,fs.climate_names)
        if {g:len(v) for g,v in groups.items()}!={'CORE':14,'CLIMATE':21,'FULL':43}: raise ValueError('Source feature dimensions conflict with 14/21/43 protocol')
        for group in ('CORE','CLIMATE','FULL'):
            chosen=groups[group]; keep=[fs.names.index(n) for n in chosen]
            climate=tuple(n for n in fs.climate_names if n in chosen)
            schema=f.FeatureSet(tuple(chosen),climate,{n:fs.support[n] for n in chosen},tuple(n for n in met if n in chosen),extra)
            selected={c:{**d,'x':d['x'][:,:,keep], 'climate':d['x'][:,-1,[fs.names.index(n) for n in climate]]} for c,d in raw.items()}
            scaled,stats=f.train_transform(selected,schema)
            expected={'train':299 if scenario=='MAIN' else 298,'monitor':35,'validation':71,'evaluation':96}
            if any(v['role_counts']!=expected for v in stats['clients'].values()): raise ValueError('Role count or embargo mismatch')
            prefix=target_dir/(scenario+'_'+group)
            _array_archive(str(prefix)+'_fit.npz',selected,scaled,schema)
            _array_archive(str(prefix)+'_evaluation_features.npz',selected,scaled,schema,evaluation=True)
            save_json(str(prefix)+'_schema.json',dict(names=schema.names,climate_names=schema.climate_names,support=schema.support,met_columns=schema.met_columns,extra_lag=extra,statistics=stats))
        # CAAA needs only client-local two-index correlations. It does not inherit
        # privileged pairwise rainfall sharing from enriched graph construction.
        tr={c:d['target_month']<='2007-11' for c,d in raw.items()}
        local_profiles={}
        for c,d in raw.items():
            y=d['y'][tr[c]];calendar=np.array([int(k[-2:]) for k in d['target_month'][tr[c]]])
            anomaly=y-np.array([y[calendar==month].mean() for month in calendar])
            flags=[]
            sensitivity=[mods['graph_profiles']._corr(anomaly,d['drivers'][tr[c],-1,i],label=c+':local_sensitivity:'+str(i),flags=flags,min_pairs=24) for i in (0,1)]
            local_profiles[c]=dict(sensitivity=sensitivity,fit_end='2007-11',n=len(y),source='client-local target calendar anomalies and lag-eligible ENSO/IOD',flags=flags)
        save_json(target_dir/(scenario+'_local_sensitivity.json'),dict(clients=local_profiles,privacy='No cross-client raw rainfall series required for these two local correlations'))
        graph_artifact=None
        if scenario=='MAIN':
            graph_data={c:mods['graph_profiles'].ClientGraphData(tuple(d['target_month'][tr[c]]),d['y'][tr[c]],
                    {n:d['x'][tr[c],-1,fs.names.index(n)] for n in f.INDEX_NAMES},
                    {n:tuple((pd.PeriodIndex(d['target_month'][tr[c]],freq='M')-2-extra).astype(str)) for n in f.INDEX_NAMES}) for c,d in raw.items()}
            graph=mods['graph_profiles'].build_graphs(graph_data,client_order=f.CLIENTS,cutoff='2007-11')
            if any(not np.allclose(graph.sensitivities[c],local_profiles[c]['sensitivity'],rtol=0,atol=1e-14) for c in raw):raise ValueError('Local sensitivity and graph descriptor computation disagree')
            graph_artifact=scenario+'_graphs.json';graph.save(target_dir/graph_artifact)
        scenarios[scenario]=dict(start_target=start,extra_lag=extra,role_counts=expected,graph_setup_artifact=graph_artifact)
        current=pd.concat([pd.DataFrame({'client':c,'target_month':d['target_month'][d['target_month']>='2017-01'],'y':d['y'][d['target_month']>='2017-01']}) for c,d in raw.items()],ignore_index=True).sort_values(['client','target_month']).reset_index(drop=True)
        if truth is None: truth=current
        elif not current.equals(truth): raise ValueError('Sensitivity evaluation truth differs')
    truth.to_csv(target_dir/'evaluation_truth.csv',index=False,float_format='%.17g')
    files={str(p.relative_to(output)):file_hash(p) for p in target_dir.iterdir() if p.is_file()}
    receipt=dict(run_sha256=identity['run_sha256'],scenarios=scenarios,files=files,model_fits=0,evaluation_predictions=0,
                 evaluation_separation='Each fit archive contains only admitted train/monitor/validation labels; evaluation feature archives contain no y/y_z/heavy. Truth is sealed until choices_frozen.json exists.',
                 historical_vintage_claim=False,preparation_wall_seconds=time.perf_counter()-started,prepared_at_unix_seconds=time.time())
    save_json(output/'prepared.json',receipt)
    return receipt


def _check_files(root,files):
    for name,sha in files.items():
        path=Path(root)/name
        if not path.is_file() or file_hash(path)!=sha: raise ValueError(f'Missing/changed artifact: {path}')


def _check_runtime_code(root):
    identity=read_json(Path(root)/'run_identity.json')
    expected=identity['code']
    actual={name:file_hash(module.__file__) for name,module in _mods().items()}
    actual['runner']=file_hash(__file__)
    if actual!=expected:raise ValueError('Runtime source code differs from the frozen run identity')
    expected_environment=identity.get('runtime_environment')
    if not isinstance(expected_environment,dict):
        raise ValueError('Missing frozen scientific runtime environment; prepare a new run directory')
    actual_environment=runtime_environment()
    if actual_environment!=expected_environment:
        raise ValueError('Scientific runtime environment differs from the frozen run identity; refusing mixed-runtime fit/evaluation resume')


def _load_dataset(root,scenario,group,evaluation=False):
    prefix=Path(root)/'datasets'/(scenario+'_'+group)
    schema=read_json(str(prefix)+'_schema.json')
    suffix='_evaluation_features.npz' if evaluation else '_fit.npz'
    result={}
    with np.load(str(prefix)+suffix,allow_pickle=False) as archive:
        for c in schema['statistics']['clients']:
            d={}
            for key in ('x','climate','drivers','target_month','origin_month'): d[key]=archive[c+'::raw_'+key].copy()
            d['scaled_x']=archive[c+'::scaled_x'].copy();d['scaled_climate']=archive[c+'::scaled_climate'].copy()
            d['target_mean'],d['target_std']=archive[c+'::target_stats'].tolist()
            if not evaluation:
                for key in ('y','y_z','heavy'): d[key]=archive[c+'::'+key].copy()
            result[c]=d
    return result,schema


def _neural_inputs(root,scenario,group,climate_free=False):
    mods=_mods(); n=mods['neural_benchmark'];f=mods['features'];data,schema=_load_dataset(root,scenario,group)
    local=read_json(Path(root)/'datasets'/(scenario+'_local_sensitivity.json'))
    clients=[]
    for c,d in data.items():
        roles=f.split_roles(d['target_month'])
        def batch(role):
            mask=roles[role]
            return n.LabeledBatch(d['scaled_x'][mask],d['scaled_climate'][mask],np.zeros_like(d['drivers'][mask]) if climate_free else d['drivers'][mask],
                                 tuple(k+'-01' for k in d['target_month'][mask]),d['y_z'][mask],d['heavy'][mask])
        sensitivity=(0.,0.) if climate_free else tuple(local['clients'][c]['sensitivity'])
        clients.append(n.ClientInput(c,batch('train'),batch('monitor'),batch('validation'),sensitivity,d['target_mean'],d['target_std']))
    graphs={}
    if scenario=='MAIN':
        graph=read_json(Path(root)/'datasets'/'MAIN_graphs.json')
        graphs={name:n.GraphInput(tuple(graph['client_ids']),np.asarray(matrix)) for name,matrix in graph['graphs'].items()}
    return clients,graphs,data,schema


def _config(method,seed,lr,mu=.01,coeff=(.3,.3,.4),group='FULL',overrides=None):
    graph=method.removeprefix('graph_') if method.startswith('graph_') else None
    actual='climate_graph' if graph else method
    if actual!='caaa':coeff=(.3,.3,.4)  # unused CAAA fields remain canonical for exact candidate/final reuse
    values=dict(method=actual,seed=seed,rounds=60,local_epochs=3,batch_size=32,cpu_threads=2,lr=lr,prox_mu=mu,
                alpha=coeff[0],beta=coeff[1],gamma=coeff[2],graph_name=graph,
                primary_identity='global' if actual in ('central','fedavg','fedprox','caaa') else 'personal',use_embedding=group!='CORE')
    values.update(overrides or {})
    return asdict(_mods()['neural_benchmark'].StrictConfig(**values))


def _job(root,kind,scenario='MAIN',group='FULL',**kw):
    identity=read_json(Path(root)/'run_identity.json')
    if kind=='neural':kw.setdefault('climate_free',False)
    payload=dict(kind=kind,scenario=scenario,group=group,**kw)
    payload['job_sha256']=digest(dict(run_sha256=identity['run_sha256'],payload=payload))
    payload['root']=str(Path(root).resolve())
    return payload


def _complete(root,job):
    _check_runtime_code(root)
    path=fit_path(root,job['job_sha256'])
    if not path.exists(): return None
    receipt=read_json(path/'complete.json')
    if _without_runtime_root(receipt['job'])!=_without_runtime_root(job): raise ValueError('Stale fit identity; refusing overwrite')
    _check_files(path,receipt['files'])
    return receipt


def _predict_neural_validation(result,data,n,identities):
    rows=[]
    for identity in identities:
        for c,d in data.items():
            mask=(d['target_month']>='2011-01')&(d['target_month']<='2016-11')
            batch=n.FeatureBatch(d['scaled_x'][mask],d['scaled_climate'][mask],d['drivers'][mask],tuple(k+'-01' for k in d['target_month'][mask]))
            p=result.predict(c,batch,identity)
            rows.extend(dict(client=c,target_month=k[:7],y=float(y),prediction=float(v),heavy_probability=float(h),identity=identity,checkpoint_round=p.checkpoint_round) for k,y,v,h in zip(p.keys,d['y'][mask],p.rainfall_mm,p.heavy_probability))
    return pd.DataFrame(rows)


def _worker(job):
    """Fit workers never load evaluation feature or truth files."""
    os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
    from threadpoolctl import threadpool_limits
    import joblib
    root=Path(job['root']);existing=_complete(root,job)
    _check_runtime_code(root)
    if existing is not None: return existing
    temporary=root/'incomplete'/(job['job_sha256'][:16]+'_'+uuid.uuid4().hex[:8]); temporary.mkdir(parents=True)
    started=time.perf_counter();cpu_started=time.process_time();started_unix=time.time();mods=_mods(); scores={}
    try:
        with threadpool_limits(limits=2):
            if job['kind']=='neural':
                n=mods['neural_benchmark'];clients,graphs,data,schema=_neural_inputs(root,job['scenario'],job['group'],job.get('climate_free',False))
                result=n.fit(clients,n.StrictConfig(**job['config']),graphs)
                result.save(temporary/'model')
                identities=[name for name,exists in (('global',result.global_state is not None),('personal',bool(result.personal_states))) if exists]
                val=_predict_neural_validation(result,data,n,identities);val.to_csv(temporary/'validation_predictions.csv',index=False,float_format='%.17g')
                scores={identity:mods['classical'].macro_r2(val.loc[val.identity==identity],'prediction') for identity in identities}
                save_json(temporary/'setup_ledger.json',dict(input_scaler=schema['statistics']['setup_communication'],
                          target_stats_export=dict(bytes_per_client=16,total_bytes=16*len(clients),wire='two float64 target mean/std for inverse transformation; disclosed plaintext summary'),
                          graph_setup_artifact=job['scenario']+'_graphs.json' if job['config']['method']=='climate_graph' else None,
                          sensitivity_summary=dict(bytes_per_client=16,total_bytes=16*len(clients),wire='two float64 client-local training correlations',alias_of='model/messages.json round0 kind=sensitivity',additional_charge_bytes=0) if job['config']['method']=='caaa' else None,
                          graph_setup_counting='MAIN graph-family preprocessing artifact is counted once as an offline shared graph setup; public_training_graph delivery in neural messages is an alias of the corresponding graph-preprocessor delivery. FedAvg/FedProx/CAAA do not require pairwise graph preprocessing.'))
            elif job['kind']=='classical':
                data,schema=_load_dataset(root,job['scenario'],'FULL')
                group_schema=read_json(root/'datasets'/(job['scenario']+'_'+job['group']+'_schema.json'))
                cfg=mods['classical'].ControlConfig(**job['config'])
                result=mods['classical'].fit_control(data,tuple(schema['names']),tuple(group_schema['names']),cfg)
                joblib.dump(result,temporary/'model.joblib');save_json(temporary/'fit_receipt.json',result.receipt)
                val=result.predict(data,'2011-01','2016-11');val.to_csv(temporary/'validation_predictions.csv',index=False,float_format='%.17g')
                scores={'global' if cfg.scope=='pooled' else 'personal':mods['classical'].macro_r2(val,'prediction')}
            elif job['kind']=='router':
                data,schema=_load_dataset(root,'MAIN','FULL');groups=mods['classical'].default_feature_groups(tuple(schema['names']),tuple(schema['climate_names']))
                configs=[mods['classical'].ControlConfig(**read_json(fit_path(root,reference)/'complete.json')['job']['config']) for reference in job['expert_jobs']]
                if [(cfg.family,cfg.scope) for cfg in configs]!=[(family,scope) for family in ('Ridge','ET','RF') for scope in ('pooled','local')]:raise ValueError('Router pool must be ordered six fixed Ridge/ET/RF pooled/local experts')
                if any(cfg.feature_group!=job['group'] or cfg.seed!=job['seed'] or cfg.representation!='latest' or cfg.target_mode!='raw' for cfg in configs):raise ValueError('Router experts must share the exact raw latest-row pool and seed')
                oof,receipts=mods['classical'].temporal_oof_controls(data,tuple(schema['names']),groups,configs)
                oof.to_csv(temporary/'training_oof_predictions.csv',index=False,float_format='%.17g');save_json(temporary/'oof_fit_receipts.json',receipts)
                wide=None
                for reference in job['expert_jobs']:
                    source=fit_path(root,reference)
                    expert=pd.read_csv(source/'validation_predictions.csv',dtype={'target_month':str})
                    label=expert.model.iloc[0];part=expert[['client','target_month','y','prediction']].rename(columns={'prediction':label})
                    wide=part if wide is None else wide.merge(part,on=['client','target_month','y'],validate='one_to_one')
                names=[cfg.label() for cfg in configs]
                result,held=mods['classical'].fit_router_suite(wide,oof,names)
                joblib.dump(result,temporary/'model.joblib');save_json(temporary/'fit_receipt.json',result.receipt)
                wide.to_csv(temporary/'validation_experts.csv',index=False,float_format='%.17g');held.to_csv(temporary/'routing_heldout_predictions.csv',index=False,float_format='%.17g')
                scores={name:mods['classical'].macro_r2(held,name) for name in ROUTER_SUITES.values()}
            elif job['kind']=='baselines':
                data,schema=_load_dataset(root,'MAIN','FULL')
                result=mods['classical'].fit_baselines(data,tuple(schema['names']),schema['support'])
                joblib.dump(result,temporary/'model.joblib');save_json(temporary/'fit_receipt.json',result.receipt)
            else: raise ValueError('Unknown fit kind')
        elapsed=time.perf_counter()-started
        files={str(p.relative_to(temporary)):file_hash(p) for p in temporary.rglob('*') if p.is_file()}
        receipt=dict(job=job,original_root=str(root),files=files,validation_macro_R2=scores,elapsed_wall_seconds=elapsed,
                     worker_process_cpu_seconds=time.process_time()-cpu_started,started_at_unix_seconds=started_unix,completed_at_unix_seconds=time.time(),
                     worker_pid=os.getpid(),cpu_process_thread_cap=2,evaluation_labels_accessed=False)
        save_json(temporary/'complete.json',receipt)
        destination=fit_path(root,job['job_sha256']);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists(): raise ValueError('Concurrent duplicate job or stale destination')
        temporary.rename(destination)
        return receipt
    except BaseException as error:
        save_json(temporary/'failure.json',dict(job=job,error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started))
        raise


def _dispatch(jobs,workers=2):
    if workers not in (1,2): raise ValueError('Maximum two processes, each two CPU threads')
    unique={job['job_sha256']:job for job in jobs};results={};pending=[]
    for key,job in unique.items():
        done=_complete(job['root'],job)
        if done: results[key]=done
        else: pending.append(job)
    if not pending:return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_worker,job):job for job in pending}
        while futures:
            done,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
            if not done: print(f'Running {len(futures)} queued/active jobs; completed {len(results)}',flush=True)
            for future in done:
                job=futures.pop(future);results[job['job_sha256']]=future.result()
                print(f"Completed {job['kind']} {job.get('group')} {job['job_sha256'][:12]} ({results[job['job_sha256']]['elapsed_wall_seconds']:.1f}s)",flush=True)
    return results


def select_candidate(candidates,identity,prefer):
    for candidate in candidates:
        score=candidate['receipt']['validation_macro_R2'].get(identity)
        if score is None or not np.isfinite(score): raise ValueError('Invalid candidate validation macro R2')
    # Exact ties, without a tolerance that silently changes the ranking.
    return min(candidates,key=lambda c:(-c['receipt']['validation_macro_R2'][identity],prefer(c)))


def _stage(root,name,jobs,identity,preference,workers):
    receipts=_dispatch(jobs,workers)
    candidates=[dict(job=job,receipt=receipts[job['job_sha256']]) for job in jobs]
    chosen=select_candidate(candidates,identity,preference)
    frozen=dict(name=name,selection_identity=identity,candidates=candidates,selected_job_sha256=chosen['job']['job_sha256'],selected_config=chosen['job']['config'],evaluation_labels_accessed=False)
    immutable_json(Path(root)/'development'/(name+'.json'),frozen)
    return chosen['job']['config']


def _model_label(method,identity,group,variant=None,scenario='MAIN'):
    base=method.upper()
    if scenario!='MAIN':return f'{base}_{identity.upper()}_{scenario}'
    return f'{base}_{identity.upper()}_{group}'+('_'+variant if variant else '')


def run_development(root,workers=2):
    root=Path(root).resolve();mods=_mods();prepared=read_json(root/'prepared.json');_check_files(root,prepared['files'])
    _check_runtime_code(root)
    data,schema=_load_dataset(root,'MAIN','FULL');names=tuple(schema['names']);groups=mods['classical'].default_feature_groups(names,tuple(schema['climate_names']))
    selected,selection=mods['classical'].select_feature_group(data,names,groups)
    immutable_json(root/'development'/'feature_selection.json',selection)
    effective={**{g:g for g in ('CORE','CLIMATE','FULL')},'TRAIN_SELECTED':selection['selected_source_group']}
    lr_jobs=[_job(root,'neural',config=_config('fedavg',42,lr)) for lr in LR_CANDIDATES]
    lr_cfg=_stage(root,'LR',lr_jobs,'global',lambda c:c['job']['config']['lr'],workers);lr=lr_cfg['lr']
    mus={}
    for method in MU_METHODS:
        jobs=[_job(root,'neural',config=_config(method,42,lr,mu)) for mu in MU_CANDIDATES]
        identity='global' if method in ('caaa','fedprox') else 'personal'
        cfg=_stage(root,'MU_'+method,jobs,identity,lambda c:c['job']['config']['prox_mu'],workers);mus[method]=cfg['prox_mu']
    jobs=[_job(root,'neural',config=_config('caaa',42,lr,mus['caaa'],coeff)) for coeff in COEFFICIENTS]
    def coeff_preference(c):
        cfg=c['job']['config'];value=tuple(cfg[k] for k in ('alpha','beta','gamma'))
        return (value!=(.3,.3,.4),value)
    ca_cfg=_stage(root,'COEFFICIENTS',jobs,'global',coeff_preference,workers)
    coeff=tuple(ca_cfg[k] for k in ('alpha','beta','gamma'))
    zero_components=[name for name,value in zip(('alpha_size','beta_performance','gamma_climate'),coeff) if value==0]
    immutable_json(root/'development'/'hyperparameters_frozen.json',dict(lr=lr,mu=mus,coefficients=coeff,selected_primary_zero_aggregation_components=zero_components,
                  primary_interpretation='Selected aggregation components equal to zero are inactive; climate inputs/embedding/adaptive proximity remain as separately configured.',
                  ablation_coefficients=(.3,.3,.4),selection=selection,effective_group_aliases=effective,evaluation_labels_accessed=False))
    jobs=[];outputs=[];experts={}
    def neural(method,group,seed,variant=None,scenario='MAIN',overrides=None,climate_free=False):
        actual_group=effective.get(group,group)
        cfg=_config(method,seed,lr,mus.get(method,.01),coeff,actual_group,overrides)
        job=_job(root,'neural',scenario,actual_group,config=cfg,climate_free=climate_free);jobs.append(job)
        identities=('global','personal') if method in ('fedavg','fedprox','caaa') else (('global',) if method=='central' else ('personal',))
        for identity in identities: outputs.append(dict(label=_model_label(method,identity,group,variant,scenario),identity=identity,job_sha256=job['job_sha256'],kind='neural',seed=seed,scenario=scenario,group=actual_group,alias_of_effective_group=actual_group if group=='TRAIN_SELECTED' else None))
    for group in GROUPS:
        for seed in SEEDS:
            for method in ('fedavg','fedprox','caaa'):neural(method,group,seed)
            for family in ('Ridge','ET','RF'):
                for scope in ('pooled','local'):
                    cfg=mods['classical'].ControlConfig(family,scope,effective[group],seed=seed,n_jobs=2)
                    job=_job(root,'classical',group=effective[group],config=asdict(cfg));jobs.append(job)
                    label=mods['classical'].ControlConfig(family,scope,group,seed=seed,n_jobs=2).label()
                    outputs.append(dict(label=label,job_sha256=job['job_sha256'],kind='classical',seed=seed,scenario='MAIN',group=effective[group],alias_of_effective_group=effective[group] if group=='TRAIN_SELECTED' else None))
                    if group in ('CORE','FULL'): experts[(group,seed,family,scope)]=job['job_sha256']
    for seed in SEEDS:
        for method in ('central','local','ditto','graph_original','graph_enriched','graph_uniform'):neural(method,'FULL',seed)
        anchor={'alpha':.3,'beta':.3,'gamma':.4}
        neural('caaa','FULL',seed,'ANCHOR',overrides=anchor)
        for flag,overrides in CA_FLAGS.items():neural('caaa','FULL',seed,flag,overrides={**anchor,**overrides})
        neural('caaa','CORE',seed,'CLIMATE_FREE',overrides={'use_climate':False,'use_embedding':False,'adaptive_prox':False},climate_free=True)
        for scenario in ('LATENCY_REFERENCE','LATENCY_LAG1'):
            for method in ('fedavg','fedprox','caaa'):neural(method,'FULL',seed,scenario=scenario)
        for scope in ('pooled','local'):
            cfg=mods['classical'].ControlConfig('ET',scope,'FULL',representation='flat_sequence',seed=seed,n_jobs=2)
            job=_job(root,'classical',config=asdict(cfg));jobs.append(job)
            outputs.append(dict(label=cfg.label(),job_sha256=job['job_sha256'],kind='classical',seed=seed,scenario='MAIN',group='FULL'))
        cfg=mods['classical'].ControlConfig('ET','pooled','FULL',seed=seed,n_jobs=2,target_mode='client_normalized')
        job=_job(root,'classical',config=asdict(cfg));jobs.append(job)
        outputs.append(dict(label=cfg.label(),job_sha256=job['job_sha256'],kind='classical',seed=seed,scenario='MAIN',group='FULL',target_mode='client_normalized'))
    baseline=_job(root,'baselines');jobs.append(baseline)
    immutable_json(root/'resolved_fit_schedule.json',dict(jobs=list({j['job_sha256']:j for j in jobs}.values()),outputs=outputs,effective_group_aliases=effective))
    _dispatch(jobs,workers)
    routers=[]
    for group in ('CORE','FULL'):
        for seed in SEEDS:
            expert_jobs=[experts[(group,seed,family,scope)] for family in ('Ridge','ET','RF') for scope in ('pooled','local')]
            job=_job(root,'router',group=group,seed=seed,expert_jobs=expert_jobs);routers.append(job)
            for label,suite in ROUTER_SUITES.items():
                outputs.append(dict(label=f'ROUTER_{group}_{label}',job_sha256=job['job_sha256'],kind='router',suite=suite,seed=seed,scenario='MAIN',group=group,expert_jobs=expert_jobs))
    immutable_json(root/'resolved_router_schedule.json',routers);_dispatch(routers,workers)
    developmental_jobs=[candidate['job'] for path in (root/'development').glob('*.json')
                        for candidate in read_json(path).get('candidates',[])]
    all_jobs=list({j['job_sha256']:j for j in [*developmental_jobs,*jobs,*routers]}.values())
    fits={j['job_sha256']:_complete(root,j) for j in all_jobs}
    frozen=dict(run_sha256=read_json(root/'run_identity.json')['run_sha256'],lr=lr,mu=mus,coefficients=coeff,selected_primary_zero_aggregation_components=zero_components,ablation_coefficients=(.3,.3,.4),effective_group_aliases=effective,
                outputs=outputs,baseline_job_sha256=baseline['job_sha256'],all_fit_jobs=all_jobs,
                fit_complete_hashes={key:file_hash(fit_path(root,key)/'complete.json') for key in fits},
                evaluation_labels_accessed=False,choices_including_checkpoints_and_router_weights_frozen=True)
    immutable_json(root/'choices_frozen.json',frozen)
    return frozen


def evaluate(root,pvalues=False):
    """Evaluation truth is opened only after validating every frozen fit receipt."""
    import joblib
    root=Path(root);mods=_mods();frozen=read_json(root/'choices_frozen.json')
    _check_runtime_code(root)
    if frozen['run_sha256']!=read_json(root/'run_identity.json')['run_sha256']:raise ValueError('Frozen choices belong to another run')
    _check_files(root,read_json(root/'prepared.json')['files'])
    for job in frozen['all_fit_jobs']:
        _complete(root,job)
        if file_hash(fit_path(root,job['job_sha256'])/'complete.json')!=frozen['fit_complete_hashes'][job['job_sha256']]:raise ValueError('Frozen fit receipt changed')
    final_destination=root/'evaluation'
    if (final_destination/'complete.json').exists():
        receipt=read_json(final_destination/'complete.json');_check_files(final_destination,receipt['files'])
        if receipt['pvalues']!=pvalues:raise ValueError('Evaluation already completed with a different pvalue policy')
        return receipt
    if final_destination.exists(): raise ValueError('Stale evaluation directory exists without a completed receipt')
    destination=root/'incomplete'/('eval_'+uuid.uuid4().hex[:8]);destination.mkdir(parents=True)
    truth=pd.read_csv(root/'datasets'/'evaluation_truth.csv',dtype={'target_month':str}).set_index(['client','target_month'])
    cache={};frames=[]
    def eval_data(scenario,group):
        key=(scenario,group)
        if key not in cache:
            data,schema=_load_dataset(root,scenario,group,evaluation=True)
            for c,d in data.items():d['y']=truth.loc[(c,d['target_month']), 'y'].to_numpy(float)
            cache[key]=(data,schema)
        return cache[key]
    pred_cache={};model_export=[]
    for output in frozen['outputs']:
        key=output['job_sha256'];path=fit_path(root,key)
        prediction_key=(key,output.get('identity'),output.get('suite'))
        if prediction_key not in pred_cache:
            if output['kind']=='neural':
                result=mods['neural_benchmark'].load_fit(path/'model',load_round_history=False)
                data,_=eval_data(output['scenario'],output['group']);rows=[];n=mods['neural_benchmark']
                for c,d in data.items():
                    job=read_json(path/'complete.json')['job'];drivers=np.zeros_like(d['drivers']) if job.get('climate_free') else d['drivers']
                    batch=n.FeatureBatch(d['scaled_x'],d['scaled_climate'],drivers,tuple(k+'-01' for k in d['target_month']))
                    pred=result.predict(c,batch,output['identity'])
                    rows.extend(dict(client=c,target_month=k[:7],y=float(y),prediction=float(p),heavy_probability=float(h),checkpoint_round=pred.checkpoint_round) for k,y,p,h in zip(pred.keys,d['y'],pred.rainfall_mm,pred.heavy_probability))
                frame=pd.DataFrame(rows)
                del result  # Keep only keyed predictions, never all model/history tensors.
            elif output['kind']=='classical':
                result=joblib.load(path/'model.joblib');data,_=eval_data(output['scenario'],'FULL');frame=result.predict(data,'2017-01','2024-12')
            else:
                router=joblib.load(path/'model.joblib');wide=None;data,_=eval_data('MAIN','FULL')
                for expert_id in output['expert_jobs']:
                    control=joblib.load(fit_path(root,expert_id)/'model.joblib');part=control.predict(data,'2017-01','2024-12');label=part.model.iloc[0]
                    part=part[['client','target_month','y','prediction']].rename(columns={'prediction':label})
                    wide=part if wide is None else wide.merge(part,on=['client','target_month','y'],validate='one_to_one')
                routed=router.predict(wide,'2017-01','2024-12');frame=routed[['client','target_month','y',output['suite']]].rename(columns={output['suite']:'prediction'})
            if not np.isfinite(frame[['y','prediction']].to_numpy(float)).all():raise ValueError('Nonfinite prediction; no replacement')
            pred_cache[prediction_key]=frame
        frame=pred_cache[prediction_key].copy();frame['model']=output['label'];frame['seed']=output['seed'];frames.append(frame)
        model_export.append(output)
    baseline=joblib.load(fit_path(root,frozen['baseline_job_sha256'])/'model.joblib');data,_=eval_data('MAIN','FULL')
    for seed in SEEDS:frames.append(baseline.predict(data,'2017-01','2024-12',seed))
    frame=pd.concat(frames,ignore_index=True)
    frame=mods['evaluation'].validate_prediction_frame(frame,EVAL_MONTHS,SEEDS)
    frame.to_csv(destination/'raw_predictions.csv',index=False,float_format='%.17g')
    main_schema=read_json(root/'datasets'/'MAIN_FULL_schema.json');thresholds={c:d['heavy80_threshold_mm'] for c,d in main_schema['statistics']['clients'].items()}
    latency_models=frame.model.str.contains('_LATENCY_')
    metrics=mods['evaluation'].evaluate_predictions(frame.loc[~latency_models],thresholds,EVAL_MONTHS,SEEDS)
    lat_schema=read_json(root/'datasets'/'LATENCY_REFERENCE_FULL_schema.json');lat_thresholds={c:d['heavy80_threshold_mm'] for c,d in lat_schema['statistics']['clients'].items()}
    lat_metrics=mods['evaluation'].evaluate_predictions(frame.loc[latency_models],lat_thresholds,EVAL_MONTHS,SEEDS)
    for name in ('per_client','aggregate_per_seed','seed_summary'):metrics[name]=pd.concat([metrics[name],lat_metrics[name]],ignore_index=True)
    metrics['PR_curves']+=lat_metrics['PR_curves']
    metrics['receipt']['latency_threshold_contract']='All latency models use their common matched298 training q80; main methods use main299 training q80'
    for name in ('per_client','aggregate_per_seed','seed_summary'):metrics[name].to_csv(destination/(name+'.csv'),index=False,float_format='%.17g')
    save_json(destination/'PR_curves.json',metrics['PR_curves']);save_json(destination/'metrics_receipt.json',metrics['receipt'])
    # Matched latency thresholds were fit on 298, not main299, training months.
    if latency_models.any():
        lat_metrics['per_client'].to_csv(destination/'matched_latency_per_client.csv',index=False,float_format='%.17g')
        save_json(destination/'matched_latency_PR_curves.json',lat_metrics['PR_curves'])
    negative=[]
    for (model,seed,client),part in frame.groupby(['model','seed','client']):
        p=part.prediction.to_numpy(float);negative.append(dict(model=model,seed=int(seed),client=client,n=len(p),negative_count=int((p<0).sum()),minimum_raw_prediction_mm=float(p.min()),negative_total_magnitude_mm=float(-p[p<0].sum()),policy='retained raw for all primary metrics'))
    pd.DataFrame(negative).to_csv(destination/'negative_forecast_diagnostics.csv',index=False,float_format='%.17g')
    comparisons=protocol()['comparisons'];boot=[];per_seed=[];tests=[]
    for i,(a,b) in enumerate(comparisons):
        results=mods['evaluation'].paired_block_bootstrap(frame,a,b,repetitions=2000,bootstrap_seed=731)
        boot.append(results['summary']);per_seed.append(results['per_seed']);results['draws'].to_csv(destination/f'bootstrap_draws_{i:02d}.csv',index=False,float_format='%.17g');save_json(destination/f'bootstrap_receipt_{i:02d}.json',results['receipt'])
        if pvalues: tests.append(mods['evaluation'].block_sign_randomization(frame,a,b,block_length=12))
    pd.concat(boot,ignore_index=True).to_csv(destination/'paired_block_bootstrap_summary.csv',index=False,float_format='%.17g');pd.concat(per_seed,ignore_index=True).to_csv(destination/'paired_per_seed_summary.csv',index=False,float_format='%.17g')
    if tests:
        adjusted=mods['evaluation'].holm_adjust({str(i):v['two_sided_p'] for i,v in enumerate(tests)})
        for i,test in enumerate(tests):test['Holm_p']=adjusted[str(i)]
        save_json(destination/'block_sign_randomization_Holm.json',tests)
    save_json(destination/'model_identities.json',model_export)
    hardware=dict(platform=platform.platform(),python=sys.version,logical_cpu_count=os.cpu_count(),max_worker_processes=2,threads_per_worker=2,maximum_active_threads=4,
                  execution='CPU simulation; time per saved fit receipt, not hardware-independent FLOPs',package_versions={name:importlib.import_module(name).__version__ for name in ('numpy','pandas','sklearn','torch','scipy')})
    save_json(destination/'hardware.json',hardware)
    fit_receipts=[read_json(fit_path(root,key)/'complete.json') for key in frozen['fit_complete_hashes']]
    pd.DataFrame([dict(job_sha256=r['job']['job_sha256'],kind=r['job']['kind'],group=r['job']['group'],seed=r['job'].get('seed',r['job'].get('config',{}).get('seed')),elapsed_wall_seconds=r['elapsed_wall_seconds'],worker_process_cpu_seconds=r.get('worker_process_cpu_seconds'),worker_pid=r['worker_pid'],threads=2) for r in fit_receipts]).to_csv(destination/'fit_timing.csv',index=False)
    stage_roles={}
    for path in (root/'development').glob('*.json'):
        for candidate in read_json(path).get('candidates',[]):stage_roles.setdefault(candidate['job']['job_sha256'],set()).add('development_'+path.stem)
    for output in frozen['outputs']:stage_roles.setdefault(output['job_sha256'],set()).add('final_'+output['kind'])
    stage_roles.setdefault(frozen['baseline_job_sha256'],set()).add('final_baselines')
    cost_rows=[]
    for fit_receipt in fit_receipts:
        job=fit_receipt['job'];key=job['job_sha256'];path=fit_path(root,key)
        row=dict(job_sha256=key,kind=job['kind'],stage_roles=';'.join(sorted(stage_roles.get(key,set()))),elapsed_wall_seconds=fit_receipt['elapsed_wall_seconds'],worker_process_cpu_seconds=fit_receipt.get('worker_process_cpu_seconds'),optimizer_steps=0,training_examples_presented=0,message_uplink_bytes=0,message_downlink_bytes=0,graph_matrix_setup_delivery_alias_bytes=0,input_scaler_setup_bytes=0,target_decode_summary_bytes=0,centralized_training_array_access_bytes=0)
        if job['kind']=='neural':
            steps=read_json(path/'model'/'local_steps.json');messages=read_json(path/'model'/'messages.json')
            row['optimizer_steps']=sum(s['optimizer_steps'] for s in steps);row['training_examples_presented']=sum(s['examples'] for s in steps)
            row['message_uplink_bytes']=sum(s['payload_bytes'] for s in messages if s['direction']=='uplink')
            row['message_downlink_bytes']=sum(s['payload_bytes'] for s in messages if s['direction']=='downlink')
            row['graph_matrix_setup_delivery_alias_bytes']=sum(s['payload_bytes'] for s in messages if s['direction']=='setup_artifact')
            setup=read_json(path/'setup_ledger.json');row['input_scaler_setup_bytes']=setup['input_scaler']['total_bytes'];row['target_decode_summary_bytes']=setup['target_stats_export']['total_bytes']
            if job['config']['method']=='central':
                ds,schema=_load_dataset(root,job['scenario'],job['group']);count=sum(int((d['target_month']<='2007-11').sum()) for d in ds.values());width=len(schema['names']);context=len(schema['climate_names'])
                row['centralized_training_array_access_bytes']=count*4*(12*width+context+24+2)
        elif job['kind']=='classical':
            control=read_json(path/'fit_receipt.json')
            if job['config']['scope']=='pooled':row['centralized_training_array_access_bytes']=control['fit_n_total']*8*(control['input_dimensions']+1)
            row['target_decode_summary_bytes']=control.get('training_target_summary_sharing',{}).get('bytes',0)
        cost_rows.append(row)
    pd.DataFrame(cost_rows).to_csv(destination/'fit_cost_ledger.csv',index=False)
    save_json(destination/'fit_cost_receipt.json',dict(each_unique_complete_fit_counted_once=True,unique_fit_bundles=len(cost_rows),
              sum_worker_wall_seconds=sum(row['elapsed_wall_seconds'] for row in cost_rows),
              sum_process_cpu_seconds=sum(row['worker_process_cpu_seconds'] for row in cost_rows if row['worker_process_cpu_seconds'] is not None),
              preprocessing_wall_seconds=read_json(root/'prepared.json')['preparation_wall_seconds'],
              timing_scope='Sum of per-process fit/validation/checkpoint-I/O wall times, not elapsed wall time of the parallel schedule. Actual process CPU time is recorded separately.',
              graph_setup='MAIN offline graph-family messages are separate; model graph-matrix setup fields alias existing artifact delivery and are not additional charges',
              input_scaler='Listed per independently deployed neural algorithm; shared precomputed scaler artifact is reused in this benchmark execution',
              centralized_array_access='Minimum numeric array bytes for deliberately centralized training access, not a measured network transfer; excludes key/header serialization',
              sensitivity='CAAA summary included once in uplink messages; setup_ledger descriptive alias contributes zero further bytes'))
    graph_setup=read_json(root/'datasets'/'MAIN_graphs.json')['messages']
    save_json(destination/'shared_graph_setup_ledger.json',dict(count_each_artifact_once=True,applies_only_to='MAIN graph-family offline preparation; three graph variants share this artifact',
              messages=graph_setup,graph_matrix_delivery_alias='FitResult.messages public_training_graph is the same setup-artifact matrix delivery and must not be charged again',
              CAAA_local_summary_alias='CAAA sensitivity uploads occur once in each model/messages.json; setup_ledger summary is descriptive and has zero additional charge',
              privacy='Privileged raw aligned training rainfall anomaly sharing is required by enriched graph computation, and does not apply to CAAA/FedAvg/FedProx; no secure aggregation or DP claim'))
    files={str(p.relative_to(destination)):file_hash(p) for p in destination.rglob('*') if p.is_file()}
    receipt=dict(run_sha256=frozen['run_sha256'],choices_frozen_sha256=file_hash(root/'choices_frozen.json'),files=files,pvalues=pvalues,
                 models=int(frame.model.nunique()),seeds=SEEDS,months=EVAL_MONTHS,raw_predictions=len(frame),evaluation_truth_opened_only_after_all_choices=True,
                 aggregate_latency_heavy_note='All latency heavy-event metrics and PR curves use the common matched298 training threshold')
    receipt=_jsonable(receipt)
    save_json(destination/'complete.json',receipt)
    destination.rename(final_destination)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','prepare','run','evaluate'))
    parser.add_argument('--monthly',type=Path);parser.add_argument('--indices',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--execute',action='store_true',help='Explicit fit/evaluation gate. Omit for source preparation only.')
    parser.add_argument('--pvalues',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='plan':
        args.output.mkdir(parents=True,exist_ok=True);immutable_json(args.output/'protocol.json',protocol());immutable_json(args.output/'schedule.json',schedule());print(json.dumps(schedule()['counts_by_stage']));return
    if any(x is None for x in (args.monthly,args.indices,args.manifest)):parser.error('--monthly, --indices and --manifest are required for hash reconciliation on every invocation')
    prepare(args.monthly,args.indices,args.manifest,args.output)
    if args.command=='prepare':print('Source/calendar preparation complete; zero model fits and evaluation predictions.');return
    if not args.execute:parser.error('Real model fitting/evaluation requires --execute; preparation is complete and reviewable')
    if args.command=='run':run_development(args.output,args.workers)
    receipt=evaluate(args.output,args.pvalues)
    print(json.dumps(dict(run_sha256=receipt['run_sha256'],models=receipt['models'],raw_predictions=receipt['raw_predictions'])))


if __name__=='__main__': main()
