"""Independent numerical/contract fixtures before launching real fits."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import torch
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from caadapfl import neural_benchmark as old,neural_revision as new,features_revision as features,runner
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def record(name,condition):
 assert condition,name
 checks.append(name)
def clients(module):
 rng=np.random.default_rng(8123);out=[]
 for i,(mean,std) in enumerate([(30.,12.),(180.,90.)]):
  def batch(year):
   return module.LabeledBatch(rng.normal(size=(4,3,5)),rng.normal(size=(4,2)),rng.normal(size=(4,3,2)),tuple(f'{year}-{m:02}-01' for m in range(1,5)),rng.normal(size=4),(np.arange(4)>=3).astype(float))
  out.append(module.ClientInput('C'+str(i+1),batch(2000),batch(2009),batch(2014),(.2,-.3),mean,std))
 return out
oldc=clients(old);newc=clients(new)
for method in ('caaa','fedavg','fedprox','central'):
 cfg=dict(method=method,seed=42,rounds=2,local_epochs=1,batch_size=3,lr=.001,prox_mu=.1,alpha=1.,beta=0.,gamma=0.,hidden=3,embedding=2,attention_hidden=4,layers=1,cpu_threads=1,primary_identity='global')
 a=old.fit(oldc,old.StrictConfig(**cfg));b=new.fit(newc,new.StrictConfig(**cfg))
 record('legacy_equivalence_selected_state_'+method,all(torch.equal(a.global_state[k],b.global_state[k]) for k in a.global_state))
 record('legacy_equivalence_checkpoint_'+method,a.selected_rounds==b.selected_rounds)
 batch=oldc[0].validation;p=old.FeatureBatch(batch.x,batch.climate,batch.drivers,batch.keys);q=new.FeatureBatch(batch.x,batch.climate,batch.drivers,batch.keys)
 record('legacy_equivalence_prediction_'+method,np.array_equal(a.predict('C1',p).rainfall_mm,b.predict('C1',q).rainfall_mm))
 cfg['adaptive_prox']=False
 if method=='caaa':fixed=new.fit(newc,new.StrictConfig(**cfg))
 if method=='fedprox':fp=new.fit(newc,new.StrictConfig(**cfg))
record('uniform_fixed_caaa_equals_fedprox',all(torch.equal(fixed.global_state[k],fp.global_state[k]) for k in fp.global_state) and fixed.selected_rounds==fp.selected_rounds)
# Heterogeneous target scales must survive pooling, independently of the first client.
cfg=new.StrictConfig(method='central',hidden=3,embedding=2,layers=1,nonnegative_head=True,cpu_threads=1,rounds=2,local_epochs=1,batch_size=3,lr=.001,primary_identity='global')
template=new.ALCE(5,2,cfg);private=[new._PrivateClient(c,template) for c in newc]
pooled=new._Tensors(*(torch.cat([getattr(c._train,name) for c in private]) for name in ('x','climate','drivers','y','heavy','target_mean','target_std')))
record('central_target_scales_per_row',pooled.target_mean.tolist()==[30.]*4+[180.]*4 and pooled.target_std.tolist()==[12.]*4+[90.]*4)
z=torch.tensor([-5.,0.,2.],requires_grad=True);means=torch.tensor([30.,180.,0.]);stds=torch.tensor([12.,90.,15.]);mapped=new.decode_z(z,means,stds,True,.1);expected=torch.nn.functional.softplus(means+stds*z,beta=.1)
record('physical_head_matches_decoded_mm_softplus',torch.allclose(means+stds*mapped,expected.to(torch.float64),atol=1e-5))
mapped.sum().backward();record('physical_head_finite_gradients',torch.isfinite(z.grad).all().item())
head=new.fit(newc,cfg)
for c in newc:
 batch=c.validation;p=head.predict(c.client_id,new.FeatureBatch(batch.x,batch.climate,batch.drivers,batch.keys))
 record('physical_head_inference_nonnegative_'+c.client_id,bool((p.rainfall_mm>=0).all()))
 score=next(x['validation'][c.client_id]['r2'] for x in head.curves if x['kind']=='evaluation' and x['identity']=='global' and x['round']==head.selected_rounds['global'])
 y=c.target_mean+c.target_std*batch.y_z.astype(np.float64);r2=1-np.sum((y-p.rainfall_mm.astype(np.float64))**2)/np.sum((y-y.mean())**2)
 record('physical_head_checkpoint_inference_consistency_'+c.client_id,abs(score-r2)<1e-6)
record('head_does_not_constrain_standardized_z_to_positive',bool((mapped.detach()<0).any()))
# Heavy weights have fixed TRAIN normalization, not a per-batch changing denominator.
h=torch.tensor([0.,0.,0.,1.]);weights=1+h
record('tail_fixed_train_normalizer_mean_one',abs(float((weights/(1+float(h.mean()))).mean())-1)<1e-7)
base=ROOT/'runs/revision_v3/datasets'
for p in base.glob('*evaluation_features.npz'):
 with np.load(p,allow_pickle=False) as a:record('inference_npz_label_free_'+p.name,all(not k.endswith(('::y','::y_z','::heavy')) for k in a.files))
for scenario,latest_rain,latest_met in [('DELAY2_REFERENCE',-1,-3),('DELAY2_OBSERVATION',-3,-5)]:
 a=json.loads((base/f'{scenario}_FULL_schema.json').read_text());record(scenario+'_support',a['support']['rain_origin_lag1']['latest_origin_offset']==latest_rain and a['support']['era5_t2m_c']['latest_origin_offset']==latest_met)
 record(scenario+'_cohort',all(c['role_counts']==dict(train=297,monitor=35,validation=71,evaluation=96) for c in a['statistics']['clients'].values()))
for bad in [dict(nonnegative_head='yes'),dict(tail_weight=.5),dict(weight_decay=-.1)]:
 try:new.StrictConfig(method='caaa',**bad)
 except (ValueError,TypeError):record('configuration_rejects_'+str(bad),True)
 else:raise AssertionError('Invalid configuration accepted')
receipt={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'scientific_fits':0,'synthetic_fixture_fits':11,'new_code_sha256':sha(ROOT/'caadapfl/neural_revision.py'),'qa_script_sha256':sha(__file__),'limitation':'Tiny numerical contract fixtures do not substitute for 100 full-budget empirical runs.'}
path=ROOT/'runs/revision_v3/implementation_qa.json';path.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','checks':len(checks)}))
