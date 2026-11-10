"""Separate frozen matched-cohort supplement launcher.

Shares the installed caadapfl runner/model interfaces without modifying them.
Three sources x two matched arms x three methods x five seeds = 90 fits.
Each fit emits separately selected global and personal model identities.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib,importlib,json,os
from pathlib import Path
import sys,time,uuid
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SEEDS=(42,123,2024,7,999)
METHODS=('fedavg','fedprox','caaa')
CASES=('IMD','solar','GLDAS21')
ARMS=('REFERENCE','ADDED')
VERSION='matched_supplements_frozen_primary_choices_v1'

def configure(package_root):
    path=str(Path(package_root).resolve())
    if path not in sys.path:sys.path.insert(0,path)
    r=importlib.import_module('caadapfl.runner')
    return r,r._mods()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def canonical(value):
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    if isinstance(value,np.ndarray):return canonical(value.tolist())
    if isinstance(value,np.generic):return canonical(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def digest(value):return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('tmp_'+uuid.uuid4().hex[:12]);tmp.write_text(json.dumps(canonical(value),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');tmp.replace(p)
def immutable(path,value):
    p=Path(path)
    if p.exists():
        if read(p)!=canonical(value):raise ValueError(f'Stale immutable receipt: {p}')
    else:save(p,value)
def verify_files(root,files):
    for name,expected in files.items():
        if sha(Path(root)/name)!=expected:raise ValueError(f'Changed prepared/fit artifact: {name}')
def code_hashes(r,mods):return {**{name:sha(m.__file__) for name,m in mods.items()},'runner':sha(r.__file__)}
def check_primary_environment(identity,current):
    if identity.get('runtime_environment')!=current:raise ValueError('Supplement runtime environment differs from frozen primary environment')
def check_code(root,r,mods):
    ident=read(Path(root)/'run_identity.json')
    if code_hashes(r,mods)!=ident['module_code'] or sha(__file__)!=ident['launcher_sha256']:raise ValueError('Runtime code changed from frozen supplement identity')
    if ident.get('runtime_environment')!=r.runtime_environment():raise ValueError('Runtime environment changed from frozen supplement identity')
def expected_keys(clients,start,end):return {(c,str(m)) for c in clients for m in pd.period_range(start,end,freq='M')}
def keyed(path,clients,start,end):
    d=pd.read_csv(path);d['month']=pd.PeriodIndex(d['month'],freq='M')
    if d.duplicated(['client','month']).any() or set((c,str(m)) for c,m in d[['client','month']].to_numpy())!=expected_keys(clients,start,end):raise ValueError(f'Supplement key/calendar failure: {path}')
    return d.set_index(['client','month']).sort_index()
def admitted_hash(receipt,name):
    claimed=receipt['outputs'][name]
    return claimed['sha256'] if isinstance(claimed,dict) else claimed

def source_inputs(args,manifest,f):
    """Validate source admission before reading outcomes for feature preparation."""
    sources={};plans={};hashes={}
    for case,directory,planpath in [('IMD',args.imd_dir,args.imd_plan),('GLDAS21',args.gldas_dir,args.gldas_plan)]:
        directory=Path(directory);receipt=read(directory/'manifest.json')
        required_status='PASS_WITH_EXPLICIT_INVALID_SOURCE' if case=='IMD' else 'PASS'
        if receipt['status']!=required_status:raise ValueError(f'{case}: source has not passed admission')
        name='imd_supplement.csv' if case=='IMD' else 'gldas21_supplement.csv'
        quality_name='imd_supplement_quality.csv' if case=='IMD' else 'gldas21_supplement_quality.csv'
        for leaf in (name,quality_name):
            if sha(directory/leaf)!=admitted_hash(receipt,leaf):raise ValueError(f'{case}: admitted source bytes changed')
            hashes[f'{case}/{leaf}']=sha(directory/leaf)
        start='1981-01' if case=='IMD' else '2000-01'
        d=keyed(directory/name,f.CLIENTS,start,'2024-12');q=keyed(directory/quality_name,f.CLIENTS,start,'2024-12')
        names=('imd_rainfall_mm',) if case=='IMD' else tuple(receipt['met_columns'])
        if list(d.columns)!=list(names):raise ValueError(f'{case}: admitted feature schema/order changed')
        good=q.feature_admitted.eq(1)&q.weighted_coverage.ge(.95)&q.weighted_coverage.le(1+1e-10)
        finite=np.isfinite(d[list(names)].to_numpy(float)).all(axis=1)
        if not np.array_equal(good.to_numpy(),finite):raise ValueError(f'{case}: admitted source/NA mask mismatch')
        plans[case]=read(planpath);hashes[f'{case}/plan']=sha(planpath);hashes[f'{case}/receipt']=sha(directory/'manifest.json')
        sources[case]=(d,names,{(c,str(m)):bool(v) for (c,m),v in good.items()})
        if case=='GLDAS21' and plans[case].get('source_quality_receipt_sha256')!=sha(directory/'manifest.json'):raise ValueError('GLDAS matched plan belongs to another source admission')
    solarpath=Path(args.canonical_dir)/'solar_supplement.csv'
    claims={**manifest.get('inputs',{}),**manifest.get('supplement_inputs',{})}
    solar_claim=next((claims[k] for k in ('solar','solar_supplement','solar_supplement.csv') if k in claims),None)
    if not isinstance(solar_claim,dict) or solar_claim.get('sha256')!=sha(solarpath):raise ValueError('Canonical manifest must authenticate emitted solar supplement bytes')
    if not manifest.get('units',{}).get('power_solar_mj_m2_day','').startswith('MJ m-2 day-1'):raise ValueError('Solar converted-unit contract missing')
    d=keyed(solarpath,f.CLIENTS,'1981-01','2024-12');name='power_solar_mj_m2_day'
    if name not in d:raise ValueError('Solar feature absent')
    finite=np.isfinite(d[name].to_numpy(float));expected=np.array([str(m)>='1984-01' for c,m in d.index])
    if not np.array_equal(finite,expected) or not d.loc[finite,name].between(0,100).all():raise ValueError('Solar structural/converted-unit gate failed')
    sources['solar']=(d[[name]],(name,),{(c,str(m)):bool(v) for (c,m),v in zip(d.index,finite)})
    plans['solar']=read(args.solar_plan);hashes['solar/plan']=sha(args.solar_plan);hashes['solar/feature_csv']=sha(solarpath)
    for case,plan in plans.items():
        if plan.get('status')!='SOURCE_QUALITY_COHORT_DERIVED' or plan.get('extra_lag')!=0:raise ValueError(f'{case}: only admitted frozen default source cohort can be fitted')
        if plan['shared_clients']!=list(f.CLIENTS):raise ValueError('All-client source match changed')
        valid=sources[case][2];derived=[]
        for t in pd.period_range('1983-01','2024-12',freq='M'):
            if all(valid.get((c,str(t-j-3)),False) for c in f.CLIENTS for j in range(1,13)):derived.append(str(t))
        if plan['target_months']!=derived:raise ValueError(f'{case}: target keys must derive solely from all-client source availability')
        evaluation=[m for m in derived if m>='2017-01']
        if plan['expected_evaluation_months']!=evaluation:raise ValueError('Explicit matched evaluation keys changed')
        if evaluation!=list(pd.period_range(evaluation[0],evaluation[-1],freq='M').astype(str)):raise ValueError('Frozen common-date bootstrap cannot bridge excluded calendar gaps')
    return sources,plans,hashes

def prepare(args):
    r,mods=configure(args.package_root);f=mods['features'];output=Path(args.output).resolve();canonical_dir=Path(args.canonical_dir)
    main_ident=read(Path(args.primary_run)/'run_identity.json');hp_path=Path(args.primary_run)/'development/hyperparameters_frozen.json';hp=read(hp_path)
    if hp.get('evaluation_labels_accessed') is not False:raise ValueError('Primary choices must be frozen without evaluation label access')
    module_code=code_hashes(r,mods)
    if module_code!=main_ident['code']:raise ValueError('Supplement modules differ from frozen primary run')
    environment=r.runtime_environment();check_primary_environment(main_ident,environment)
    manifest_path=canonical_dir/'manifest.json';manifest=read(manifest_path)
    if manifest.get('source_gate',{}).get('passed') is not True:raise ValueError('Canonical source gates have not passed')
    for name in ('monthly','indices'):
        if sha(canonical_dir/(name+'.csv'))!=manifest['inputs'][name]['sha256'] or manifest['inputs'][name]['sha256']!=main_ident['source'][name+'_sha256']:raise ValueError('Supplement canonical input differs from frozen primary source')
    if sha(manifest_path)!=main_ident['source']['manifest_sha256']:raise ValueError('Canonical manifest differs from frozen primary run')
    met=tuple(manifest['feature_contract']['met_columns'])
    if len(met)!=22 or len(set(met))!=22:raise ValueError('Main22 meteorology contract changed')
    if hp['lr'] not in r.LR_CANDIDATES or any(hp['mu'][m] not in r.MU_CANDIDATES for m in ('fedprox','caaa')) or tuple(hp['coefficients']) not in r.COEFFICIENTS:raise ValueError('Primary selected choices outside frozen candidate contract')
    sources,plans,source_hashes=source_inputs(args,manifest,f)
    protocol={'version':VERSION,'cases':CASES,'arms':ARMS,'methods':METHODS,'seeds':SEEDS,'fit_count':90,
       'rounds':60,'local_epochs':3,'batch_size':32,'threads_per_worker':2,'max_workers':2,
       'choices':'Primary FULL selected LR, FedProx/CA mu and CA coefficients; no supplemental retuning or old three-fold selector',
       'scalers':'Own gradient-TRAIN input scalers, local target scalers, q80 and climate sensitivity per matched case/arm',
       'matching':'Same target/date/client axes and target statistics for both arms; any invalid source removes target globally',
       'evaluation':'Global/personal predictions separately identified; raw inverse mm, no clipping; truth opened after all90fits/checkpoints freeze',
       'subset_selection':'Source calendar and quality only; no outcome-based subset selection',
       'revised_archive_disclosure':'Retrospective archived products and conservative lag scenario, not historical real-time vintages',
       'comparisons':[(f'{m.upper()}_{identity.upper()}_{case}_ADDED',f'{m.upper()}_{identity.upper()}_{case}_REFERENCE') for case in CASES for m in METHODS for identity in ('global','personal')]}
    ident={'version':VERSION,'primary_run_sha256':main_ident['run_sha256'],'primary_hyperparameters_sha256':sha(hp_path),
       'primary_choices':{'lr':hp['lr'],'mu':{m:hp['mu'][m] for m in ('fedprox','caaa')},'coefficients':hp['coefficients']},
       'canonical_input_hashes':{name:sha(canonical_dir/filename) for name,filename in [('monthly','monthly.csv'),('indices','indices.csv'),('manifest','manifest.json')]},
       'source_hashes':source_hashes,'module_code':module_code,'runtime_environment':environment,'launcher_sha256':sha(__file__),'protocol_hash':digest(protocol)}
    ident['run_sha256']=digest(ident)
    output.mkdir(parents=True,exist_ok=True);immutable(output/'run_identity.json',ident);immutable(output/'protocol.json',protocol);immutable(output/'primary_choices_frozen.json',hp)
    if (output/'prepared.json').exists():
        receipt=read(output/'prepared.json');verify_files(output,receipt['files']);return receipt
    if (output/'datasets').exists():raise ValueError('Incomplete prior preparation; use a fresh supplement output directory')
    panel=f.checked_panel(canonical_dir/'monthly.csv',ident['canonical_input_hashes']['monthly'],['rainfall_mm',*met]);indices=pd.read_csv(canonical_dir/'indices.csv')
    dataset_dir=output/'datasets';dataset_dir.mkdir();cases={}
    for case in CASES:
        extra,names,valid=sources[case];plan=plans[case];truth=None;target_stats=None;reference_sensitivity=None
        immutable(output/'cohorts'/(case+'.json'),plan)
        for arm in ARMS:
            scenario=case+'_'+arm;columns=met if arm=='REFERENCE' else met+names
            current=panel if arm=='REFERENCE' else panel.join(extra[list(names)],validate='one_to_one')
            rows,fs=f.build_rows(current,indices,met_columns=columns,with_climate=True)
            raw=f.samples(current,rows,fs,target_months=tuple(plan['target_months']),expected_evaluation_months=tuple(plan['expected_evaluation_months']))
            scaled,stats=f.train_transform(raw,fs)
            declared={k:v for k,v in plan['role_counts_per_client'].items() if k!='embargo'}
            if any(v['role_counts']!=declared for v in stats['clients'].values()):raise ValueError('Matched role counts/embargo changed')
            if len(fs.names)!=(43 if arm=='REFERENCE' else 43+len(names)) or len(fs.climate_names)!=7:raise ValueError('Supplement neural feature/context dimensions changed')
            prefix=dataset_dir/(scenario+'_FULL');r._array_archive(str(prefix)+'_fit.npz',raw,scaled,fs)
            r._array_archive(str(prefix)+'_evaluation_features.npz',raw,scaled,fs,evaluation=True)
            save(str(prefix)+'_schema.json',{'names':fs.names,'climate_names':fs.climate_names,'support':fs.support,'met_columns':fs.met_columns,'extra_lag':0,'statistics':stats})
            local={}
            for c,d in raw.items():
                train=f.split_roles(d['target_month'])['train'];y=d['y'][train];calendar=np.array([int(k[-2:]) for k in d['target_month'][train]])
                anomaly=y-np.array([y[calendar==m].mean() for m in calendar]);flags=[]
                sensitivity=[mods['graph_profiles']._corr(anomaly,d['drivers'][train,-1,i],label=c+':supplement_sensitivity:'+str(i),flags=flags,min_pairs=24) for i in (0,1)]
                local[c]={'sensitivity':sensitivity,'fit_end':'2007-11','n':int(train.sum()),'flags':flags,'source':'matched client-local TRAIN calendar anomalies and lag-eligible raw drivers'}
            save(dataset_dir/(scenario+'_local_sensitivity.json'),{'clients':local,'privacy':'Two client-local training correlations; no pairwise raw rainfall sharing'})
            target_now={c:(v['target_mean'],v['target_std'],v['heavy80_threshold_mm']) for c,v in stats['clients'].items()}
            if target_stats is None:target_stats=target_now;reference_sensitivity=local
            elif target_stats!=target_now or reference_sensitivity!=local:raise ValueError('Matched arms have different target scaling/thresholds/sensitivities')
            frame=pd.concat([pd.DataFrame({'client':c,'target_month':d['target_month'][d['target_month']>='2017-01'],'y':d['y'][d['target_month']>='2017-01']}) for c,d in raw.items()],ignore_index=True).sort_values(['client','target_month']).reset_index(drop=True)
            if truth is None:truth=frame
            elif not truth.equals(frame):raise ValueError('Matched evaluation truths differ')
        truth.to_csv(dataset_dir/(case+'_evaluation_truth.csv'),index=False,float_format='%.17g')
        cases[case]={'role_counts':declared,'evaluation_months':plan['expected_evaluation_months'],'supplement_met_columns':names,'samples_per_client':len(plan['target_months'])}
    files={str(p.relative_to(output)):sha(p) for p in dataset_dir.iterdir() if p.is_file()}
    receipt={'run_sha256':ident['run_sha256'],'cases':cases,'files':files,'model_fits':0,'evaluation_predictions':0,
       'evaluation_separation':'Fit archives contain only train/monitor/validation labels; evaluation feature archives exclude y/y_z/heavy; sealed truth read only after all fits/checkpoints freeze'}
    save(output/'prepared.json',receipt);return receipt

def job_core(job):return {k:v for k,v in job.items() if k not in ('root','package_root')}
def fit_path(root,key):return Path(root)/'fits'/key[:16]
def complete(root,job):
    path=fit_path(root,job['job_sha256'])
    if not path.exists():return None
    receipt=read(path/'complete.json')
    if job_core(receipt['job'])!=job_core(job):raise ValueError('Stale fit or hash-prefix collision')
    verify_files(path,receipt['files']);return receipt
def worker(job):
    os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
    from threadpoolctl import threadpool_limits
    r,mods=configure(job['package_root']);root=Path(job['root']);check_code(root,r,mods)
    previous=complete(root,job)
    if previous is not None:return previous
    verify_files(root,read(root/'prepared.json')['files'])
    temporary=root/'incomplete'/(job['job_sha256'][:16]+'_'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    started=time.perf_counter();cpu=time.process_time();n=mods['neural_benchmark']
    try:
        with threadpool_limits(limits=2):
            clients,graphs,data,schema=r._neural_inputs(root,job['scenario'],'FULL')
            result=n.fit(clients,n.StrictConfig(**job['config']),graphs);result.save(temporary/'model')
            validation=r._predict_neural_validation(result,data,n,['global','personal'])
            validation.to_csv(temporary/'validation_predictions.csv',index=False,float_format='%.17g')
            scores={i:mods['classical'].macro_r2(validation[validation.identity==i],'prediction') for i in ('global','personal')}
            save(temporary/'setup_ledger.json',{'input_scaler':schema['statistics']['setup_communication'],
              'target_stats_export':{'bytes_per_client':16,'total_bytes':80,'wire':'Plain two-float64 target inverse-transform summary'},
              'sensitivity_summary':{'additional_charge_bytes':0,'alias_of':'model/messages.json sensitivity upload'} if job['method']=='caaa' else None,
              'privacy':'Federation simulation; no secure aggregation or differential privacy'})
        files={str(p.relative_to(temporary)):sha(p) for p in temporary.rglob('*') if p.is_file()}
        receipt={'job':job,'files':files,'validation_macro_R2':scores,'worker_pid':os.getpid(),'cpu_threads':2,
          'elapsed_wall_seconds':time.perf_counter()-started,'worker_process_cpu_seconds':time.process_time()-cpu,
          'evaluation_labels_accessed':False,'supplement_hyperparameters_retuned':False}
        save(temporary/'complete.json',receipt);destination=fit_path(root,job['job_sha256']);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise ValueError('Concurrent duplicate fit destination')
        temporary.rename(destination);return receipt
    except BaseException as error:
        save(temporary/'failure.json',{'job':job,'error_type':type(error).__name__,'error':str(error)});raise

def run(args):
    r,mods=configure(args.package_root);root=Path(args.output).resolve();check_code(root,r,mods);prepared=read(root/'prepared.json');verify_files(root,prepared['files'])
    identity=read(root/'run_identity.json');hp=identity['primary_choices'];jobs=[]
    for case in CASES:
        for arm in ARMS:
            for method in METHODS:
                for seed in SEEDS:
                    config=r._config(method,seed,hp['lr'],hp['mu'].get(method,.01),tuple(hp['coefficients']),'FULL')
                    payload={'run_sha256':identity['run_sha256'],'case':case,'arm':arm,'scenario':case+'_'+arm,'method':method,'config':config}
                    payload['job_sha256']=digest(payload);jobs.append({**payload,'root':str(root),'package_root':str(Path(args.package_root).resolve())})
    assert len(jobs)==90
    immutable(root/'resolved_fit_schedule.json',{'jobs':[job_core(j) for j in jobs],'count':90,'retuning':False})
    if not args.execute:return {'scheduled_fits':90,'execute':False,'note':'Pass --execute after coordinating at most two total model worker processes with the primary run'}
    if not 1<=args.workers<=2:raise ValueError('Maximum two workers, two threads each')
    pending_jobs=[j for j in jobs if complete(root,j) is None]
    if args.workers==1:
        for j in pending_jobs:worker(j)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending={};iterator=iter(pending_jobs)
            for _ in range(args.workers):
                try:j=next(iterator)
                except StopIteration:break
                pending[pool.submit(worker,j)]=j
            while pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    future.result();pending.pop(future)
                    try:j=next(iterator)
                    except StopIteration:continue
                    pending[pool.submit(worker,j)]=j
    receipts={j['job_sha256']:sha(fit_path(root,j['job_sha256'])/'complete.json') for j in jobs if complete(root,j) is not None}
    if len(receipts)!=90:raise ValueError('Supplement fit schedule incomplete')
    immutable(root/'choices_frozen.json',{'run_sha256':identity['run_sha256'],'jobs':[job_core(j) for j in jobs],
      'fit_complete_hashes':receipts,'hyperparameters_frozen_from_primary':True,'all_checkpoints_frozen':True,'evaluation_labels_accessed':False})
    return {'complete_fits':90,'choices_frozen':True,'evaluation_predictions':0}

def label(method,identity,case,arm):return f'{method.upper()}_{identity.upper()}_{case}_{arm}'
def evaluate(args):
    r,mods=configure(args.package_root);root=Path(args.output);check_code(root,r,mods);frozen=read(root/'choices_frozen.json');prepared=read(root/'prepared.json');verify_files(root,prepared['files'])
    if frozen['run_sha256']!=read(root/'run_identity.json')['run_sha256'] or frozen.get('all_checkpoints_frozen') is not True or len(frozen['jobs'])!=90:raise ValueError('Fit/checkpoint choices incomplete')
    for key,value in frozen['fit_complete_hashes'].items():
        if sha(fit_path(root,key)/'complete.json')!=value:raise ValueError('Frozen checkpoint receipt changed')
    destination=root/'evaluation'
    if destination.exists():
        receipt=read(destination/'complete.json');verify_files(destination,receipt['files']);return receipt
    temporary=root/'incomplete'/('evaluation_'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    all_predictions=[];all_metrics=[];all_summary=[];costs=[]
    for case in CASES:
        truth=pd.read_csv(root/'datasets'/(case+'_evaluation_truth.csv'));frames=[]
        thresholds=read(root/'datasets'/(case+'_REFERENCE_FULL_schema.json'))['statistics']['clients']
        threshold={c:v['heavy80_threshold_mm'] for c,v in thresholds.items()}
        for job in frozen['jobs']:
            if job['case']!=case:continue
            restored={**job,'root':str(root),'package_root':str(args.package_root)};receipt=complete(root,restored)
            result=mods['neural_benchmark'].load_fit(fit_path(root,job['job_sha256'])/'model');data,schema=r._load_dataset(root,job['scenario'],'FULL',evaluation=True)
            for identity_name in ('global','personal'):
                for c,d in data.items():
                    batch=mods['neural_benchmark'].FeatureBatch(d['scaled_x'],d['scaled_climate'],d['drivers'],tuple(k+'-01' for k in d['target_month']))
                    prediction=result.predict(c,batch,identity_name)
                    frame=pd.DataFrame({'client':c,'target_month':[k[:7] for k in prediction.keys],
                      'prediction':prediction.rainfall_mm,'heavy_probability':prediction.heavy_probability,
                      'model':label(job['method'],identity_name,case,job['arm']),'seed':job['config']['seed'],'checkpoint_round':prediction.checkpoint_round})
                    frame=frame.merge(truth,on=['client','target_month'],validate='many_to_one')
                    if len(frame)!=len(batch.keys):raise ValueError('Evaluation truth/key join failed')
                    frames.append(frame)
            path=fit_path(root,job['job_sha256']);messages=read(path/'model/messages.json');steps=read(path/'model/local_steps.json');setup=read(path/'setup_ledger.json')
            costs.append({'case':case,'arm':job['arm'],'method':job['method'],'seed':job['config']['seed'],'job_sha256':job['job_sha256'],
              'wall_seconds':receipt['elapsed_wall_seconds'],'cpu_seconds':receipt['worker_process_cpu_seconds'],
              'optimizer_steps':sum(v['optimizer_steps'] for v in steps),'training_examples':sum(v['examples'] for v in steps),
              'uplink_bytes':sum(v['payload_bytes'] for v in messages if v['direction']=='uplink'),
              'downlink_bytes':sum(v['payload_bytes'] for v in messages if v['direction']=='downlink'),
              'scaler_setup_bytes':setup['input_scaler']['total_bytes'],'target_decode_summary_bytes':80})
        frame=pd.concat(frames,ignore_index=True);months=prepared['cases'][case]['evaluation_months']
        frame=mods['evaluation'].validate_prediction_frame(frame,months,SEEDS);all_predictions.append(frame.assign(case=case))
        case_dir=temporary/case;case_dir.mkdir();frame.to_csv(case_dir/'raw_predictions.csv',index=False,float_format='%.17g')
        metrics=mods['evaluation'].evaluate_predictions(frame,threshold,months,SEEDS)
        for name in ('per_client','aggregate_per_seed','seed_summary'):metrics[name].to_csv(case_dir/(name+'.csv'),index=False,float_format='%.17g')
        save(case_dir/'PR_curves.json',metrics['PR_curves']);save(case_dir/'metrics_receipt.json',metrics['receipt'])
        all_metrics.append(metrics['aggregate_per_seed'].assign(case=case));all_summary.append(metrics['seed_summary'].assign(case=case))
        summaries=[];perseed=[]
        for method in METHODS:
            for identity_name in ('global','personal'):
                a=label(method,identity_name,case,'ADDED');b=label(method,identity_name,case,'REFERENCE')
                result=mods['evaluation'].paired_block_bootstrap(frame,a,b,repetitions=2000,bootstrap_seed=731)
                prefix=method+'_'+identity_name;result['draws'].to_csv(case_dir/(prefix+'_bootstrap_draws.csv'),index=False,float_format='%.17g')
                save(case_dir/(prefix+'_bootstrap_receipt.json'),result['receipt']);summaries.append(result['summary']);perseed.append(result['per_seed'])
        pd.concat(summaries,ignore_index=True).to_csv(case_dir/'paired_block_bootstrap_summary.csv',index=False,float_format='%.17g')
        pd.concat(perseed,ignore_index=True).to_csv(case_dir/'paired_per_seed_summary.csv',index=False,float_format='%.17g')
    pd.concat(all_predictions,ignore_index=True).to_csv(temporary/'raw_predictions_all_cases.csv',index=False,float_format='%.17g')
    pd.concat(all_metrics,ignore_index=True).to_csv(temporary/'aggregate_per_seed_all_cases.csv',index=False,float_format='%.17g')
    pd.concat(all_summary,ignore_index=True).to_csv(temporary/'seed_summary_all_cases.csv',index=False,float_format='%.17g')
    pd.DataFrame(costs).to_csv(temporary/'fit_cost_ledger.csv',index=False,float_format='%.17g')
    receipt={'run_sha256':frozen['run_sha256'],'complete_fits':90,'raw_prediction_rows':sum(len(f) for f in all_predictions),
      'fit_choices_or_subsets_selected_from_evaluation':False,'supplement_retuning':False,'clipped_predictions':False,
      'comparisons':'Within-case added minus own matched reference; global and personal identities remain separate',
      'bootstrap':'Five paired seeds/common-client-date circular blocks12,6,24;2000repetitions; no cross-cohort score comparison',
      'files':{str(p.relative_to(temporary)):sha(p) for p in temporary.rglob('*') if p.is_file()}}
    save(temporary/'complete.json',receipt);temporary.rename(destination);return receipt

def main(argv=None):
    ap=argparse.ArgumentParser();commands=ap.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','evaluate'):
        p=commands.add_parser(name);p.add_argument('--package-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if name=='prepare':
            p.add_argument('--canonical-dir',type=Path,required=True);p.add_argument('--primary-run',type=Path,required=True)
            p.add_argument('--imd-dir',type=Path,default=HERE/'derived/imd');p.add_argument('--gldas-dir',type=Path,default=HERE/'derived/gldas_full')
            p.add_argument('--imd-plan',type=Path,default=HERE/'derived/cohorts/IMD_cohort_plan.json')
            p.add_argument('--solar-plan',type=Path,default=HERE/'derived/cohorts/solar_cohort_plan.json')
            p.add_argument('--gldas-plan',type=Path,default=HERE/'derived/cohorts_gldas_admitted/GLDAS21_cohort_plan.json')
        if name=='run':p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,choices=[1,2],default=1)
    args=ap.parse_args(argv);result={'prepare':prepare,'run':run,'evaluate':evaluate}[args.command](args)
    print(json.dumps(canonical(result),indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
