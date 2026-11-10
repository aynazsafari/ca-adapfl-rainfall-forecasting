"""Finite, reproducible revision experiment; labels are sealed from fitting."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,sys,time,concurrent.futures,copy
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from caadapfl import neural_revision as n, features_revision as f, runner as accepted
RUN=ROOT/'runs/revision_v3'
SEEDS=(42,123,2024,7,999)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as q:
  for b in iter(lambda:q.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def save(p,x,immutable=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(x,indent=2,sort_keys=True,allow_nan=False)
 if immutable and p.exists():
  assert json.loads(p.read_text())==x,f'Immutable artifact mismatch: {p}'
 else:
  tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');os.replace(tmp,p)
def load(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def protocol():
 return dict(version='revision_v3.0',historical_evaluation='2017–2024 already inspected; exploratory only',paired_seeds=SEEDS,
  rounds=60,local_epochs=3,batch_size=32,lr=.001,prox_mu=.1,alpha_beta_gamma=[1.,0.,0.],selection_role='outer validation 2011–2016 excluding December2016; never evaluation',
  development_candidates=['FULL_H48','FULL_H24','FULL_H12','COMPACT_H12','REGULARIZED_H12'],
  compact_rule='Greedy correlation thinning on pooled TRAIN final sequence rows; abs(r)>=.95; fixed source feature order; always retain four known-calendar harmonics. Recompute TRAIN-only scaling. Raw two-driver adaptation remains unchanged.',
  regularization={'hidden':12,'recurrent_dropout':.35,'head_dropout':.5,'weight_decay':.0001},
  common_architecture_selection='Highest mean paired-seed selected-global VALIDATION macroR2 over the five development candidates; ties within1e-4 favor fewer trainable parameters, then fixed candidate order. All four algorithm controls receive exactly this common architecture and fixed optimizer settings. This is a matched-mechanism comparison, not independent best-of-search tuning of every algorithm.',
  selected_parent_factorial={'embedding':[True,False],'adaptive_prox':[True,False],'fixed_mu':.1,'climate_sequence_channels_and_adaptive_drivers':'retained in all arms; embedding-off removes its branch only'},
  fair_methods=['caaa','fedavg','fedprox','central'],output_loss_variants=['RAW','NONNEGATIVE','NONNEGATIVE_TAIL2'],
  head='softplus(decoded rainfall_mm,beta=.1/mm), mapping back to local z only for objective; per-row local mean/std in central pool; identical training/checkpoint/inference',
  tail_loss='Weighted standardized squared error, weight2 for observed rainfall>TRAIN localq80, normalized by fixed mean weight over each client TRAIN (pooled mean weight for central). Auxiliary=.3 and proximal coefficients unchanged.',
  loss_selection='A common variant is chosen by the highest mean VALIDATION macroR2 across four methods and five seeds; tie within1e-4 favors RAW, then NONNEGATIVE, then NONNEGATIVE_TAIL2. RAW also receives common max(0,mm) inference sensitivity, with its already selected raw checkpoint explicitly retained.',
  delay2='Two extra months of observed-input latency at the original target-first-day issuance. Latest rain/index t−4, met t−6. Both reference and delayed arms start1983-03 and have297/35/71/96 labels. Known target calendar unchanged. This does not claim true issuance two months earlier; that would require earlier label embargoes.',
  uncertainty={'paired_seed_date_bootstrap':2000,'common_date_circular_block_months':[12,6,24],'report_primary_block':12,'annual_sign_blocks':8,'sign_patterns':256,'holm_family_size':36,'CI_policy':'Unadjusted percentile95% intervals, exploratory; Holm only exact annual sign p-values; no equivalence claim from nonrejection','contrast_family':[
   ['FULL_H24','FULL_H48'],['FULL_H12','FULL_H48'],['COMPACT_H12','FULL_H12'],['REGULARIZED_H12','FULL_H12'],
   ['CAAA_RAW','EMBEDDING_OFF'],['CAAA_RAW','ADAPTIVE_OFF'],['CAAA_RAW','BOTH_OFF'],['EMBEDDING_OFF','BOTH_OFF'],
   ['CAAA_RAW','FEDAVG_RAW'],['CAAA_RAW','FEDPROX_RAW'],['CAAA_RAW','CENTRAL_RAW'],
   *[[m+'_NONNEGATIVE',m+'_RAW'] for m in ['CAAA','FEDAVG','FEDPROX','CENTRAL']],
   *[[m+'_NONNEGATIVE_TAIL2',m+'_NONNEGATIVE'] for m in ['CAAA','FEDAVG','FEDPROX','CENTRAL']],
   *[[m+'_SELECTED_LOSS',b] for m in ['CAAA','FEDAVG','FEDPROX','CENTRAL'] for b in ['TRAIN_MONTHLY_CLIMATOLOGY','POOLED_EXTRATREES_V2']],
   ['DELAY2_OBSERVATION','DELAY2_REFERENCE'],
   *[[m+'_CLIPPED_RAW',m+'_RAW'] for m in ['CAAA','FEDAVG','FEDPROX','CENTRAL']],
   *[['CAAA_GLOBAL_FULL_V2',b] for b in ['TRAIN_MONTHLY_CLIMATOLOGY','POOLED_EXTRATREES_V2','STATIC_OOF_RIDGE_V2','STATIC_OOF_RIDGE_CORE_V2']]]},
  new_fit_budget={'development':25,'selected_factorial_additional':15,'matched_fedavg_central':10,'nonnegative_and_tail':40,'delay':10,'maximum':100},
  stopping='Complete fixed100 fits (or exact reuse of selected-parent/FedProx-equivalent identity); no seed expansion, no evaluation-led retuning; stop only for finite known errors requiring reviewed code repair and new identity.',
  new_holdout='Only if accessible, source-consistent, complete and verified unused. No2025/2026 period is called independent without provenance. Not required to invent a holdout.',
  git_policy='Prepare code/data-manifests/raw predictions/environment/README and checks for GitHub; no invented public repository or DOI.')

def prepare():
 RUN.mkdir(parents=True,exist_ok=True)
 manifest=load(ROOT/'data/manifest.json');panel=f.checked_panel(ROOT/'data/monthly.csv',manifest['inputs']['monthly']['sha256'],['rainfall_mm',*manifest['feature_contract']['met_columns']]);indices=pd.read_csv(ROOT/'data/indices.csv')
 schema=load(RUN/'datasets/MAIN_FULL_schema.json')
 data,_=accepted._load_dataset(RUN,'MAIN','FULL')
 names=schema['names'];matrix=np.concatenate([d['x'][f.split_roles(d['target_month'])['train'],-1,:] for d in data.values()])
 corr=np.corrcoef(matrix,rowvar=False);keep=[];removed=[]
 for j,name in enumerate(names):
  redundant=next((i for i in keep if abs(corr[i,j])>=.95),None) if j>=4 else None
  if redundant is None: keep.append(j)
  else: removed.append({'removed':name,'retained':names[redundant],'correlation':float(corr[redundant,j])})
 climate=tuple(x for x in schema['climate_names'] if names.index(x) in keep)
 fs=f.FeatureSet(tuple(names[i] for i in keep),climate,{names[i]:schema['support'][names[i]] for i in keep},tuple(x for x in schema['met_columns'] if names.index(x) in keep),0)
 selected={c:{'x':d['x'][:,:,keep],'climate':d['x'][:,-1,[names.index(x) for x in climate]],'drivers':d['drivers'],'y':d['y'],'target_month':d['target_month'],'origin_month':d['origin_month']} for c,d in data.items()}
 scaled,stats=f.train_transform(selected,fs)
 accepted._array_archive(RUN/'datasets/MAIN_COMPACT_fit.npz',selected,scaled,fs)
 # Evaluation input is transformed using exactly the training-selected channels/scalers, with no labels.
 evaldata,_=accepted._load_dataset(RUN,'MAIN','FULL',evaluation=True)
 arr={};means=np.array(stats['global_input_mean']);stds=np.array(stats['global_input_std']);ci=[fs.names.index(x) for x in climate]
 for c,d in evaldata.items():
  x=d['x'][:,:,keep];context=d['x'][:,-1,[names.index(x) for x in climate]]
  for key,val in [('x',x),('climate',context),('drivers',d['drivers']),('target_month',d['target_month']),('origin_month',d['origin_month'])]:arr[c+'::raw_'+key]=val
  arr[c+'::scaled_x']=((x-means)/stds).astype(np.float32);arr[c+'::scaled_climate']=((context-means[ci])/stds[ci]).astype(np.float32)
  arr[c+'::target_stats']=np.array([stats['clients'][c]['target_mean'],stats['clients'][c]['target_std']])
 np.savez_compressed(RUN/'datasets/MAIN_COMPACT_evaluation_features.npz',**arr)
 for c,d in evaldata.items():
  assert np.array_equal(d['target_month'],np.array(pd.period_range('2017-01','2024-12',freq='M').astype(str)))
  stats['clients'][c]['fit_archive_role_counts']=dict(stats['clients'][c]['role_counts'])
  stats['clients'][c]['role_counts']['evaluation']=len(d['target_month'])
 save(RUN/'datasets/MAIN_COMPACT_schema.json',dict(names=fs.names,climate_names=climate,support=fs.support,statistics=stats,met_columns=fs.met_columns,extra_lag=0,compact_selection={'source':'TRAIN only','threshold':.95,'removed':removed}))
 for scenario,extra in [('DELAY2_REFERENCE',0),('DELAY2_OBSERVATION',2)]:
  rows,fs=f.build_rows(panel,indices,tuple(manifest['feature_contract']['met_columns']),extra_lag=extra)
  raw=f.samples(panel,rows,fs,start_target='1983-03');scaled,stats=f.train_transform(raw,fs)
  assert all(v['role_counts']==dict(train=297,monitor=35,validation=71,evaluation=96) for v in stats['clients'].values())
  accepted._array_archive(RUN/f'datasets/{scenario}_FULL_fit.npz',raw,scaled,fs)
  accepted._array_archive(RUN/f'datasets/{scenario}_FULL_evaluation_features.npz',raw,scaled,fs,evaluation=True)
  save(RUN/f'datasets/{scenario}_FULL_schema.json',dict(names=fs.names,climate_names=fs.climate_names,support=fs.support,statistics=stats,met_columns=fs.met_columns,extra_lag=extra))
  profiles={}
  for c,d in raw.items():
   mask=d['target_month']<='2007-11';y=d['y'][mask];cal=np.array([int(k[-2:]) for k in d['target_month'][mask]]);anomaly=y-np.array([y[cal==month].mean() for month in cal]);flags=[]
   profiles[c]={'sensitivity':[accepted._mods()['graph_profiles']._corr(anomaly,d['drivers'][mask,-1,i],label=c,flags=flags,min_pairs=24) for i in (0,1)],'n':len(y),'fit_end':'2007-11'}
  save(RUN/f'datasets/{scenario}_local_sensitivity.json',{'clients':profiles})
 truth=panel[['rainfall_mm']].reset_index();truth['month']=truth['month'].astype(str);truth=truth[truth['month']>='2017-01'].rename(columns={'month':'target_month','rainfall_mm':'observed_mm'})
 truth.to_csv(RUN/'datasets/SEALED_evaluation_truth.csv',index=False)
 files={p.relative_to(ROOT).as_posix():sha(p) for p in (RUN/'datasets').iterdir() if p.is_file()}
 save(RUN/'prepared.json',dict(files=files,evaluation_label_policy='Truth stored separately; workers load fit.npz only. Historical labels already inspected in previous assessment.',compact_removed=removed))
 save(RUN/'protocol.json',protocol(),immutable=True)
 code={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'caadapfl/neural_revision.py',ROOT/'caadapfl/features_revision.py',Path(__file__),ROOT/'caadapfl/neural_benchmark.py',ROOT/'caadapfl/runner.py',ROOT/'caadapfl/features.py',ROOT/'caadapfl/graph_profiles.py']}
 save(RUN/'run_identity.json',{'created_utc':now(),'code':code,'datasets':files,'environment':accepted.runtime_environment(),'protocol_sha256':sha(RUN/'protocol.json'),'version':'revision_v3.0','previous_run_artifacts_not_overwritten':True},immutable=True)
 print('PREPARED',len(files),'files; compact',len(keep),'features; no model fits',flush=True)

def config(method,seed,**over):
 vals=accepted._config(method,seed,.001,.1,(1.,0.,0.),'FULL');vals.update(over)
 return asdict(n.StrictConfig(**vals))
def job(label,seed,scenario='MAIN',group='FULL',**over):
 method=over.pop('method','caaa');cfg=config(method,seed,**over)
 obj=dict(label=label,seed=seed,scenario=scenario,group=group,config=cfg)
 obj['id']=hashlib.sha256(json.dumps({'config':cfg,'scenario':scenario,'group':group,'run_identity_sha256':sha(RUN/'run_identity.json')},sort_keys=True).encode()).hexdigest()
 return obj
def inputs(j):
 data,schema=accepted._load_dataset(RUN,j['scenario'],j['group']);profiles=load(RUN/f"datasets/{j['scenario']}_local_sensitivity.json")
 clients=[]
 for c,d in data.items():
  roles=f.split_roles(d['target_month'])
  def batch(role):
   a=roles[role];return n.LabeledBatch(d['scaled_x'][a],d['scaled_climate'][a],d['drivers'][a],tuple(k+'-01' for k in d['target_month'][a]),d['y_z'][a],d['heavy'][a])
  clients.append(n.ClientInput(c,batch('train'),batch('monitor'),batch('validation'),tuple(profiles['clients'][c]['sensitivity']),d['target_mean'],d['target_std']))
 return clients,schema
def worker(j):
 assert accepted.runtime_environment()==load(RUN/'run_identity.json')['environment'],'Runtime changed after freeze'
 directory=RUN/'fits'/j['id'];done=directory/'complete.json'
 if done.exists():
  rec=load(done);assert rec['job']==j
  for name,h in rec['files'].items():assert sha(directory/name)==h
  return rec
 if directory.exists():raise ValueError('Incomplete existing fit; explicit repair required: '+str(directory))
 clients,schema=inputs(j);cfg=n.StrictConfig(**j['config']);fit=n.fit(clients,cfg);fit.save(directory)
 curve=next(c for c in fit.curves if c['kind']=='evaluation' and c['identity']=='global' and c['round']==fit.selected_rounds['global'])
 dev=[]
 for c in clients:
  b=c.validation;prediction=fit.predict(c.client_id,n.FeatureBatch(b.x,b.climate,b.drivers,b.keys),'global')
  for k,y,p,h in zip(b.keys,c.target_mean+c.target_std*b.y_z,prediction.rainfall_mm,b.heavy):dev.append(dict(label=j['label'],seed=j['seed'],client=c.client_id,target_month=k[:7],observed_mm=float(y),predicted_mm=float(p),heavy80=int(h)))
 pd.DataFrame(dev).to_csv(directory/'validation_predictions.csv',index=False)
 params=n.ALCE(fit.features,fit.context,cfg);counts={'total':sum(p.numel() for p in params.parameters()),'trainable':sum(p.numel() for p in params.parameters() if p.requires_grad)}
 rec=dict(job=j,finished_utc=now(),validation_macro_r2=curve['validation_macro_r2'],selected_round=fit.selected_rounds['global'],parameters=counts,elapsed_seconds=fit.elapsed_seconds,optimizer_steps=sum(x['optimizer_steps'] for x in fit.local_steps),examples=sum(x['examples'] for x in fit.local_steps),files={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
 save(done,rec);return rec
def execute(jobs,phase):
 save(RUN/f'{phase}_jobs_frozen.json',{'frozen_utc':now(),'jobs':jobs,'evaluation_labels_used':False},immutable=False)
 results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(worker,j):j for j in jobs}
  for future in concurrent.futures.as_completed(futures):
   r=future.result();results.append(r)
   print(json.dumps({'utc':now(),'phase':phase,'complete':len(results),'total':len(jobs),'label':r['job']['label'],'seed':r['job']['seed'],'val_r2':r['validation_macro_r2'],'seconds':round(r['elapsed_seconds'],2)}),flush=True)
   save(RUN/'progress.json',dict(utc=now(),phase=phase,complete=len(results),total=len(jobs)))
 save(RUN/f'{phase}_complete.json',{'jobs':sorted(results,key=lambda x:(x['job']['label'],x['job']['seed'])),'finished_utc':now()})
 return results
def train():
 assert (RUN/'implementation_qa.json').exists() and load(RUN/'implementation_qa.json')['status']=='PASS'
 identity=load(RUN/'run_identity.json')
 assert accepted.runtime_environment()==identity['environment'],'Runtime changed after freeze'
 for name,h in identity['code'].items():assert sha(ROOT/name)==h,'Code changed after freeze: '+name
 for name,h in identity['datasets'].items():assert sha(ROOT/name)==h,'Dataset changed after freeze: '+name
 jobs=[]
 for seed in SEEDS:
  for width in (48,24,12):jobs.append(job(f'FULL_H{width}',seed,hidden=width))
  jobs.append(job('COMPACT_H12',seed,group='COMPACT',hidden=12))
  jobs.append(job('REGULARIZED_H12',seed,hidden=12,recurrent_dropout=.35,head_dropout=.5,weight_decay=.0001))
 development=execute(jobs,'development')
 summaries=[]
 for label in protocol()['development_candidates']:
  a=[r for r in development if r['job']['label']==label]
  summaries.append(dict(label=label,mean_validation_r2=float(np.mean([r['validation_macro_r2'] for r in a])),trainable_parameters=a[0]['parameters']['trainable']))
 maximum=max(s['mean_validation_r2'] for s in summaries);eligible=[s for s in summaries if s['mean_validation_r2']>=maximum-1e-4];selected=min(eligible,key=lambda s:(s['trainable_parameters'],protocol()['development_candidates'].index(s['label'])))
 exemplar=next(r['job'] for r in development if r['job']['label']==selected['label']); common={k:exemplar['config'][k] for k in ['hidden','head_dropout','recurrent_dropout','weight_decay']};group=exemplar['group']
 save(RUN/'development_choice_frozen.json',dict(frozen_utc=now(),summaries=summaries,selected=selected,common_config=common,group=group,selected_using_evaluation=False,selection_rule=protocol()['common_architecture_selection']))
 new=[];aliases=[]
 for seed in SEEDS:
  parent=next(r['job'] for r in development if r['job']['label']==selected['label'] and r['job']['seed']==seed)
  aliases.append({'label':'CAAA_RAW','job':parent,'seed':seed})
  for emb,adp,label in [(False,True,'EMBEDDING_OFF'),(True,False,'ADAPTIVE_OFF'),(False,False,'BOTH_OFF')]:new.append(job(label,seed,group=group,use_embedding=emb,adaptive_prox=adp,**common))
  for method in ('fedavg','central'):new.append(job(method.upper()+'_RAW',seed,group=group,method=method,**common))
  # Uniform CAAA with fixed mu and the embedding retained is exactly FedProx's training mechanism.
  fp=job('ADAPTIVE_OFF',seed,group=group,use_embedding=True,adaptive_prox=False,**common);aliases.append({'label':'FEDPROX_RAW','job':fp,'seed':seed,'alias_explanation':'Equal-count uniform CAAA aggregation with fixed mu=.1 equals matched FedProx updates; exact small-fixture equality independently verified.'})
  for method in ('caaa','fedavg','fedprox','central'):
   for variant,tw in [('NONNEGATIVE',1.),('NONNEGATIVE_TAIL2',2.)]:new.append(job(method.upper()+'_'+variant,seed,group=group,method=method,nonnegative_head=True,tail_weight=tw,**common))
  for scenario in ('DELAY2_REFERENCE','DELAY2_OBSERVATION'):new.append(job(scenario,seed,scenario=scenario,hidden=48))
 results=execute(new,'matched_controls')
 allresults=development+results
 lossrows=[]
 for variant in ['RAW','NONNEGATIVE','NONNEGATIVE_TAIL2']:
  vals=[]
  for method in ['CAAA','FEDAVG','FEDPROX','CENTRAL']:
   label=method+'_'+variant
   if variant=='RAW' and method in ('CAAA','FEDPROX'):
    vals.extend(next(r['validation_macro_r2'] for r in allresults if r['job']['id']==a['job']['id']) for a in aliases if a['label']==label)
   else:vals.extend(r['validation_macro_r2'] for r in allresults if r['job']['label']==label)
  assert len(vals)==20,(variant,len(vals))
  lossrows.append({'variant':variant,'mean_validation_macro_r2':float(np.mean(vals))})
 vmax=max(r['mean_validation_macro_r2'] for r in lossrows);best=next(r for r in lossrows if r['mean_validation_macro_r2']>=vmax-1e-4)
 save(RUN/'choices_frozen.json',dict(frozen_utc=now(),run_identity_sha256=sha(RUN/'run_identity.json'),development_choice=load(RUN/'development_choice_frozen.json'),aliases=aliases,all_jobs=[r['job'] for r in allresults],fit_complete_hashes={r['job']['id']:sha(RUN/'fits'/r['job']['id']/'complete.json') for r in allresults},loss_choices=lossrows,selected_common_loss=best,evaluation_label_used=False))
 save(RUN/'fit_complete.json',dict(status='COMPLETE',unique_fits=len(allresults),expected_unique_fits=100,optimizer_steps=sum(r['optimizer_steps'] for r in allresults),examples=sum(r['examples'] for r in allresults),finished_utc=now()))
 print('ALL FITS COMPLETE AND CHOICES FROZEN',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train']);a=parser.parse_args()
 {'prepare':prepare,'train':train}[a.action]()
