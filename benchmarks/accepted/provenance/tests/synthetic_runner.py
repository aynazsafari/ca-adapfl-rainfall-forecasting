"""Synthetic-only verification of schedule, label sealing and normalized control.

Copies current read-only integration modules into a disposable package inside
this scratch directory. Never reads the study panel or runs the real schedule.
"""
from pathlib import Path
import importlib
import json
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
TMP=Path(tempfile.mkdtemp(prefix='synthetic_',dir=HERE/'synthetic_only'))
PKG=TMP/'package'/'caadapfl';PKG.mkdir(parents=True)
(PKG/'__init__.py').write_text('',encoding='utf-8')
for name,source in {'features':WORK/'benchmark_stage'/'caadapfl'/'features.py',
                    'classical':WORK/'classical_impl'/'classical.py','evaluation':WORK/'classical_impl'/'evaluation.py',
                    'neural_benchmark':WORK/'benchmark_impl'/'neural_benchmark.py',
                    'graph_profiles':WORK/'benchmark_impl'/'graph_profiles.py','runner':HERE/'runner.py'}.items():
    shutil.copy2(source,PKG/(name+'.py'))
sys.path.insert(0,str(PKG.parent))
from caadapfl import runner as r, features as f, classical as c

checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def fails(name,call,exception=ValueError):
    try:call()
    except exception:checks.append(name)
    else:raise AssertionError(name+' did not reject')

months=pd.period_range('1981-01','2024-12',freq='M')
met=tuple('synthetic_met_'+str(i) for i in range(22))
rows=[]
for ci,client in enumerate(f.CLIENTS):
    for t,month in enumerate(months):
        record=dict(client=client,month=str(month),rainfall_mm=80+ci*9+35*np.sin(t*2*np.pi/12)+10*np.sin(t*.13)+.02*t)
        record.update({name:ci*.3+t*.002+np.sin(t/(4+i)) for i,name in enumerate(met)})
        rows.append(record)
panel=TMP/'synthetic_monthly.csv';pd.DataFrame(rows).to_csv(panel,index=False,float_format='%.17g')
idx=TMP/'synthetic_indices.csv';pd.DataFrame({'month':months.astype(str),**{n:np.sin(np.arange(len(months))*(.02+i*.009))+.1*np.cos(np.arange(len(months))*.11) for i,n in enumerate(f.INDEX_NAMES)}}).to_csv(idx,index=False,float_format='%.17g')
manifest=TMP/'synthetic_manifest.json'
r.save_json(manifest,dict(source_gate={'passed':True,'label':'SYNTHETIC ONLY'},inputs={'monthly':{'sha256':r.file_hash(panel)},'indices':{'sha256':r.file_hash(idx)}},feature_contract={'met_columns':met}))
out=TMP/'prepared_synthetic'
receipt=r.prepare(panel,idx,manifest,out)
check('preparation has zero fits and evaluation predictions',receipt['model_fits']==receipt['evaluation_predictions']==0)
check('preparation leaves no choices frozen or fit artifacts',not(out/'choices_frozen.json').exists() and not(out/'fits').exists())
check('manifest source checksums and code hashes in immutable identity',len(r.read_json(out/'run_identity.json')['code'])==6)
check('schedule precedes score artifacts',(out/'schedule.json').exists() and not(out/'development').exists())
check('quarter-simplex has exactly15 points plus anchor',len(r.COEFFICIENTS)==16 and len(set(r.COEFFICIENTS))==16 and all(np.isclose(sum(x),1) for x in r.COEFFICIENTS))
check('exact five paired seeds',r.SEEDS==(42,123,2024,7,999))
check('candidate tuning is sequential2+18+16, not Cartesian',r.schedule()['counts_by_stage']['LR']==2 and r.schedule()['counts_by_stage']['MU']==18 and r.schedule()['counts_by_stage']['COEFFICIENTS']==16)
check('all zero-component coefficient boundary candidates preserved',(0.,1.,0.) in r.COEFFICIENTS and (0.,0.,1.) in r.COEFFICIENTS and (1.,0.,0.) in r.COEFFICIENTS)
check('five ablation pairs reference fixed nonzero anchor parent',all(a=='CAAA_GLOBAL_FULL_ANCHOR' for a,b in r.protocol()['comparisons'] if any(b=='CAAA_GLOBAL_FULL_'+flag for flag in r.CA_FLAGS)))
for boundary in ((0.,1.,0.),(0.,0.,1.)):
    for flag,overrides in r.CA_FLAGS.items():
        cfg=r._config('caaa',42,.001,coeff=boundary,overrides={'alpha':.3,'beta':.3,'gamma':.4,**overrides})
        active_sum=cfg['alpha']+(cfg['beta'] if cfg['use_performance'] else 0)+(cfg['gamma'] if cfg['use_climate'] else 0)
        check(f'{boundary} {flag} anchored weights stay nonzero and valid',active_sum>0 and tuple(cfg[k] for k in ('alpha','beta','gamma'))==(.3,.3,.4))
n=r._mods()['neural_benchmark']
metadata=([10,20],[.1,2.],[[1.,.2],[.1,2.]],[[.4,.6],[.9,.2]])
fails('pure-performance parent removing performance would degenerate',lambda:n.caaa_weights(*metadata,n.StrictConfig(**r._config('caaa',42,.001,coeff=(0.,1.,0.),overrides={'use_performance':False}))))
fails('pure-climate parent removing climate would degenerate',lambda:n.caaa_weights(*metadata,n.StrictConfig(**r._config('caaa',42,.001,coeff=(0.,0.,1.),overrides={'use_climate':False}))))
parent_weights=n.caaa_weights(*metadata,n.StrictConfig(**r._config('caaa',42,.001)))
for flag in ('NO_PERFORMANCE_WEIGHT','NO_CLIMATE_WEIGHT'):
    weights=n.caaa_weights(*metadata,n.StrictConfig(**r._config('caaa',42,.001,overrides=r.CA_FLAGS[flag])))
    check(flag+' anchor changes actual finite normalized weights',np.isfinite(weights).all() and np.isclose(weights.sum(),1) and np.max(np.abs(weights-parent_weights))>1e-3)
check('CPU bound2processes2threads',r.protocol()['cpu']['maximum_active_threads']==4)
for scenario,counts in (('MAIN',(299,35,71)),('LATENCY_REFERENCE',(298,35,71)),('LATENCY_LAG1',(298,35,71))):
    for group,width in (('CORE',14),('CLIMATE',21),('FULL',43)):
        data,schema=r._load_dataset(out,scenario,group)
        check(f'{scenario} {group} width and finite fit arrays',all(d['x'].shape==(sum(counts),12,width) and np.isfinite(d['x']).all() for d in data.values()))
        check(f'{scenario} {group} training-only transform counts',all(tuple(v['role_counts'][k] for k in ('train','monitor','validation'))==counts for v in schema['statistics']['clients'].values()))
        with np.load(out/'datasets'/(scenario+'_'+group+'_evaluation_features.npz'),allow_pickle=False) as archive:
            check(f'{scenario} {group} evaluation archives contain zero target or event labels',not any(key.split('::')[-1] in ('y','y_z','heavy') for key in archive.files))
        check(f'{scenario} {group} fit archives exclude eval and boundaryDecember labels',all(not any(k>='2017-01' or k in ('2007-12','2010-12','2016-12') for k in d['target_month']) for d in data.values()))
clients,graphs,data,schema=r._neural_inputs(out,'MAIN','CORE')
check('CORE has zero-dimensional context and embedding disabled',all(v.train.climate.shape[1]==0 for v in clients) and r._config('caaa',42,.001,group='CORE')['use_embedding'] is False)
check('fit API payload includes no evaluation arrays',all(v.validation.keys[-1]=='2016-11-01' for v in clients))
lat0,_=r._load_dataset(out,'LATENCY_REFERENCE','FULL');lat1,_=r._load_dataset(out,'LATENCY_LAG1','FULL')
check('latency cohorts exactly aligned with same298 training dates',all(np.array_equal(lat0[client]['target_month'],lat1[client]['target_month']) for client in f.CLIENTS))
check('latency source changes one calendar month and keeps evaluation96',lat1[f.CLIENTS[0]]['x'][0,-1,4]!=lat0[f.CLIENTS[0]]['x'][0,-1,4] and len(r.EVAL_MONTHS)==96)
config=r._config('fedavg',42,.001)
check('tuning and final identical configurations deduplicate jobs',r._job(out,'neural',config=config)['job_sha256']==r._job(out,'neural',config=config,climate_free=False)['job_sha256'])
fails('no evaluation without frozen choices',lambda:r.evaluate(out),FileNotFoundError)
bad=TMP/'bad_manifest.json';bad_data=r.read_json(manifest);bad_data['source_gate']['passed']=False;r.save_json(bad,bad_data)
fails('source gate fails hard before reading models',lambda:r.prepare(panel,idx,bad,TMP/'bad_prepare'))
bad_data=r.read_json(manifest);bad_data['inputs']['indices']['sha256']='0'*64;r.save_json(bad,bad_data)
fails('index hash mismatch fails hard',lambda:r.prepare(panel,idx,bad,TMP/'bad_hash'))
check('resume preparation exactly identical and fit free',r.prepare(panel,idx,manifest,out)==receipt)
stale=r.read_json(out/'run_identity.json');stale['run_sha256']='0'*64;r.save_json(out/'run_identity.json',stale)
fails('stale run refuses overwrite',lambda:r.prepare(panel,idx,manifest,out))

# Manual decoding example. Both clients have the same standardized relationship,
# despite a tenfold difference in rainfall scale; client identity is never input.
train_months=np.array(['1990-01','1990-02','1990-03','1990-04','2017-01','2017-02'])
x=np.array([0,1,2,3,0,3],float)[:,None,None]*np.ones((1,12,1))
manual={'A':dict(x=x,y=np.array([10,12,14,16,10,16],float),target_month=train_months),
        'B':dict(x=x,y=np.array([100,120,140,160,100,160],float),target_month=train_months)}
cfg=c.ControlConfig('ET','pooled','FULL',seed=42,n_estimators=8,min_samples_leaf=1,target_mode='client_normalized')
control=c.fit_control(manual,('rain_a',),('rain_a',),cfg)
prediction=control.predict(manual,'2017-01','2017-02')
check('normalized pooled ET manual endpoint inverse transforms',np.allclose(prediction.prediction,[10,16,100,160],rtol=0,atol=1e-12))
check('normalized control declares training summary sharing and no clientIDinput',control.receipt['training_target_summary_sharing']['bytes']==32 and control.receipt['client_identity_input'] is False)
check('normalized versus raw pooled labels remain distinct',cfg.label().startswith('client_normalized_pooled_ET') and c.ControlConfig('ET','pooled','FULL').label().startswith('raw_pooled_ET'))
changed={client:{**d,'y':np.where(d['target_month']>='2017-01',9999,d['y'])} for client,d in manual.items()}
same=c.fit_control(changed,('rain_a',),('rain_a',),cfg)
check('future evaluation truth cannot change normalized fit or stats',same.target_stats==control.target_stats and np.array_equal(same.predict(manual,'2017-01','2017-02').prediction,prediction.prediction))
fails('normalized local scope rejected instead of ambiguous control',lambda:c.fit_control(manual,('rain_a',),('rain_a',),c.ControlConfig('ET','local',target_mode='client_normalized')))

# Exact ties are policy, not approximate tolerances.
candidates=[dict(job={'config':{'lr':v}},receipt={'validation_macro_R2':{'global':.5}}) for v in (.003,.001)]
check('exact equal validation scores prefer smaller LR',r.select_candidate(candidates,'global',lambda q:q['job']['config']['lr'])['job']['config']['lr']==.001)
candidates[0]['receipt']['validation_macro_R2']['global']=.5+1e-12
check('nonexact tiny score difference remains a real difference',r.select_candidate(candidates,'global',lambda q:q['job']['config']['lr'])['job']['config']['lr']==.003)
r.save_json(HERE/'synthetic_only'/'verification_receipt.json',dict(label='SYNTHETIC ONLY; zero study data reads, zero real fits',checks_passed=len(checks),checks=checks,temporary_directory=str(TMP),schedule_counts=r.schedule()['counts_by_stage'],real_data_scores=0))
print(json.dumps(dict(synthetic_checks_passed=len(checks),temporary_directory=str(TMP))))
