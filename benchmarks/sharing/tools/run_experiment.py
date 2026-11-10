"""Guarded prepare/train pipeline. Training does not open EVAL features or truth."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, platform, sys, tempfile, time
from dataclasses import replace
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from caadapfl import neural_revision as n,features_revision as f
import sharing_engine
RUN=ROOT/'run'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as q:
        for b in iter(lambda:q.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),cpu_only=True)

def raw_data(evaluation=False):
    schema=load(ROOT/'datasets/MAIN_FULL_schema.json');data={}
    name='MAIN_FULL_evaluation_features.npz' if evaluation else 'MAIN_FULL_fit.npz'
    with np.load(ROOT/'datasets'/name,allow_pickle=False) as a:
        for c in schema['statistics']['clients']:
            d={k:a[c+'::raw_'+k].copy() for k in ('x','climate','drivers','target_month','origin_month')}
            if not evaluation:
                d.update({k:a[c+'::'+k].copy() for k in ('y','y_z','heavy')})
            data[c]=d
    return data,schema

def scaling(data,schema,mode):
    names=schema['names'];ix=[names.index(k) for k in schema['climate_names']]
    summaries={}
    for c,d in data.items():
        tr=f.split_roles(d['target_month'])['train'];x=d['x'][tr,-1,:]
        summaries[c]=dict(count=len(x),sum=x.sum(0),squares=(x*x).sum(0))
    def stats(count,sums,squares):
        mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0));std[std<1e-12]=1.
        return mean,std
    if mode=='shared':
        count=sum(q['count'] for q in summaries.values());sums=sum(q['sum'] for q in summaries.values());sq=sum(q['squares'] for q in summaries.values())
        mean,std=stats(count,sums,sq);vectors={c:(mean,std) for c in data}
    elif mode=='local':vectors={c:stats(q['count'],q['sum'],q['squares']) for c,q in summaries.items()}
    else:raise ValueError(mode)
    record={};scaled={}
    for c,d in data.items():
        mean,std=vectors[c];tr=f.split_roles(d['target_month'])['train'];y=d['y'][tr]
        ym=float(y.mean());ys=float(y.std(ddof=0));q80=float(np.quantile(y,.8,method='linear'))
        assert ys>0
        scaled[c]=dict(d,scaled_x=((d['x']-mean)/std).astype(np.float32),scaled_climate=((d['climate']-mean[ix])/std[ix]).astype(np.float32),
            y_z=((d['y']-ym)/ys).astype(np.float32),heavy=(d['y']>q80).astype(np.float32),target_mean=ym,target_std=ys)
        record[c]=dict(input_mean=mean.tolist(),input_std=std.tolist(),TRAIN_count=summaries[c]['count'],TRAIN_sum=summaries[c]['sum'].tolist(),
            TRAIN_squared_sum=summaries[c]['squares'].tolist(),target_mean=ym,target_std=ys,q80_mm=q80)
    return scaled,record

def clients_for(job):
    data,schema=raw_data();data,record=scaling(data,schema,job['scaling'])
    sensitivity=load(ROOT/'datasets/MAIN_local_sensitivity.json')['clients'];clients=[]
    for c,d in data.items():
        roles=f.split_roles(d['target_month'])
        def batch(role):
            mask=roles[role]
            return n.LabeledBatch(d['scaled_x'][mask],d['scaled_climate'][mask],d['drivers'][mask],tuple(k+'-01' for k in d['target_month'][mask]),d['y_z'][mask],d['heavy'][mask])
        clients.append(n.ClientInput(c,batch('train'),batch('monitor'),batch('validation'),tuple(sensitivity[c]['sensitivity']),d['target_mean'],d['target_std']))
    return clients,data,schema,record

def implementation_qa():
    checks=[];rng=np.random.default_rng(29)
    def batch(keys):
        size=len(keys);return n.LabeledBatch(rng.normal(size=(size,3,4)),rng.normal(size=(size,2)),rng.normal(size=(size,3,2)),tuple(keys),rng.normal(size=size),np.zeros(size))
    clients=[n.ClientInput(c,batch(['2000-01-01','2000-02-01','2000-03-01','2000-04-01']),batch(['2008-01-01','2008-02-01']),batch(['2011-01-01','2011-02-01']),(0.,0.),50.,10.) for c in ('A','B')]
    cfg=n.StrictConfig(method='fedavg',seed=19,rounds=2,local_epochs=1,batch_size=4,lr=.001,hidden=2,layers=1,embedding=2,attention_hidden=2,
        recurrent_dropout=0.,head_dropout=0.,cpu_threads=1,state_checkpoint_interval=1,nonnegative_head=True,use_prox=False,prox_mu=0.,adaptive_prox=False)
    eq=lambda a,b:all(torch.equal(a[k],b[k]) for k in a)
    parent=n.fit(clients,cfg);new,_,cost=sharing_engine.fit(clients,cfg,1)
    assert eq(new.global_state,parent.round_states[-1]['global_state'])
    assert all(eq(new.personal_states[c],parent.round_states[-1]['personal_states'][c]) for c in ('A','B'))
    checks.append('cadence1 terminal states bitwise equal to inherited FedAvg')
    local,_,_=sharing_engine.fit(clients,replace(cfg,method='local'),0)
    sparse,_,_=sharing_engine.fit(clients,cfg,2)
    assert all(eq(local.personal_states[c],sparse.personal_states[c]) for c in ('A','B'))
    assert eq(sparse.global_state,n.aggregate_states(list(local.personal_states.values()),[.5,.5]))
    checks.append('cadence2 retains both local ticks/RNG/resets and averages only at2')
    changed=replace(clients[1],train=replace(clients[1].train,y_z=clients[1].train.y_z+10))
    perturbed,_,_=sharing_engine.fit([clients[0],changed],replace(cfg,method='local'),0)
    assert eq(local.personal_states['A'],perturbed.personal_states['A']);assert not local.messages
    checks.append('fixed terminal local state independent of another client target; zero model messages')
    ca=replace(cfg,method='caaa',prox_mu=.1,use_prox=True,adaptive_prox=True,alpha=1.,beta=0.,gamma=0.)
    oldca=n.fit(clients,ca);newca,_,_=sharing_engine.fit(clients,ca,1)
    assert eq(newca.global_state,oldca.round_states[-1]['global_state'])
    checks.append('uniform-weight CAAA terminal parity despite omitting inactive telemetry')
    params=n.state_bytes(new.global_state);assert sum(q['payload_bytes'] for q in new.messages)==2*params+2*8+2*2*2*params
    assert sum(q['optimizer_steps'] for q in new.local_steps)==4 and sum(q['examples'] for q in new.local_steps)==16
    checks.append('exact update/example budgets and model/count payload formula including final broadcast')
    data,schema=raw_data();shared,record=scaling(data,schema,'shared');iso,private_stats=scaling(data,schema,'local')
    with np.load(ROOT/'datasets/MAIN_FULL_fit.npz',allow_pickle=False) as a:
        assert all(np.array_equal(d['scaled_x'],a[c+'::scaled_x']) for c,d in shared.items())
        assert all(np.array_equal(d['scaled_climate'],a[c+'::scaled_climate']) for c,d in shared.items())
    checks.append('recomputed shared scaling exactly matches authenticated parent tensors')
    c0,c1=list(data)[:2];changed={c:dict(d) for c,d in data.items()};changed[c1]=dict(data[c1],x=data[c1]['x'].copy())
    changed[c1]['x'][f.split_roles(changed[c1]['target_month'])['validation']]+=1000
    _,valstats=scaling(changed,schema,'shared');assert valstats==record
    changed[c1]['x'][f.split_roles(changed[c1]['target_month'])['train']]+=3
    _,private2=scaling(changed,schema,'local');_,shared2=scaling(changed,schema,'shared')
    assert private2[c0]==private_stats[c0] and shared2[c0]['input_mean']!=record[c0]['input_mean']
    checks.append('VAL perturbation cannot change scaling; TRAIN cross-client dependency confined to shared arm')
    for c,d in data.items():assert [int(v.sum()) for v in f.split_roles(d['target_month']).values()][:3]==[299,35,71]
    with np.load(ROOT/'datasets/MAIN_FULL_evaluation_features.npz',allow_pickle=False) as a:assert not any(k.endswith(('::y','::y_z','::heavy')) for k in a.files)
    checks.append('split counts299/35/71 and separate label-free EVAL artifact')
    with tempfile.TemporaryDirectory(prefix='sharing_synthetic_') as td:
        dst=Path(td)/'fit';new.save(dst);restored=n.load_fit(dst)
        b=clients[0].validation;features=n.FeatureBatch(b.x,b.climate,b.drivers,b.keys)
        assert np.array_equal(new.predict('A',features).rainfall_mm,restored.predict('A',features).rainfall_mm)
    checks.append('save/load exact inference parity')
    save(RUN/'implementation_qa.json',dict(status='PASS',utc=now(),checks=checks,scope='Synthetic fixtures and TRAIN preprocessing only; no empirical fits/EVAL outcomes'))
    print('IMPLEMENTATION_QA_PASS',len(checks),flush=True)

def prepare():
    assert not (RUN/'identity.json').exists()
    p=load(ROOT/'protocol.json');assert load(RUN/'implementation_qa.json')['status']=='PASS'
    for name,h in p['inputs'].items():assert sha(ROOT/name)==h
    code={q.relative_to(ROOT).as_posix():sha(q) for q in [Path(__file__),ROOT/'tools/sharing_engine.py',*sorted((ROOT/'caadapfl').glob('*.py'))]}
    save(RUN/'identity.json',dict(utc=now(),protocol_sha256=sha(ROOT/'protocol.json'),inputs=p['inputs'],code=code,environment=environment(),implementation_qa_sha256=sha(RUN/'implementation_qa.json')))
    jobs=[]
    for j in p['jobs']:
        j=dict(j);j['id']=hashlib.sha256(json.dumps(dict(job=j,identity_sha256=sha(RUN/'identity.json')),sort_keys=True).encode()).hexdigest();jobs.append(j)
    save(RUN/'jobs_frozen.json',dict(utc=now(),jobs=jobs,evaluation_labels_used=False))
    print('PREPARED40_FIXED_JOBS',flush=True)

def guard():
    i=load(RUN/'identity.json');assert environment()==i['environment'];assert sha(ROOT/'protocol.json')==i['protocol_sha256']
    assert sha(RUN/'implementation_qa.json')==i['implementation_qa_sha256']
    for name,h in {**i['code'],**i['inputs']}.items():assert sha(ROOT/name)==h,name
    frozen=load(RUN/'jobs_frozen.json')['jobs'];expected=load(ROOT/'protocol.json')['jobs']
    assert len(frozen)==len(expected)==40
    assert len({(j['label'],j['seed']) for j in frozen})==40
    for j,source in zip(frozen,expected):
        assert {k:v for k,v in j.items() if k!='id'}==source,'Job differs from sealed protocol'
        key=hashlib.sha256(json.dumps(dict(job=source,identity_sha256=sha(RUN/'identity.json')),sort_keys=True).encode()).hexdigest()
        assert j['id']==key,'Invalid frozen job identity'

def worker(job):
    dst=RUN/'fits'/job['id']
    if (dst/'complete.json').exists():
        r=load(dst/'complete.json');assert r['job']==job
        for name,h in r['files'].items():assert sha(dst/name)==h
        return r
    assert not dst.exists(),'Partial fit must be recorded and repaired, never silently overwritten'
    clients,data,schema,stats=clients_for(job);cfg=n.StrictConfig(**job['config'])
    terminal,selected,cost=sharing_engine.fit(clients,cfg,job['cadence'])
    terminal.save(dst/'terminal');selected.save(dst/'validation_selected')
    save(dst/'scaler_statistics.json',dict(mode=job['scaling'],clients=stats,fit_role='TRAIN only'))
    audit=cost.pop('validation_audit_messages');save(dst/'research_audit_messages.json',audit)
    width=len(schema['names']);messages=[]
    if job['method']=='central':
        for c,d in data.items():
            tr=f.split_roles(d['target_month'])['train'];count=int(tr.sum())
            # Source tensors converted explicitly to the declared numeric wire format.
            payload=d['x'][tr].astype(np.float64).nbytes+d['climate'][tr].astype(np.float64).nbytes+d['y'][tr].astype(np.float64).nbytes+8
            messages.append(dict(client_id=c,direction='uplink',kind='engineered_TRAIN_example_bundle',payload_bytes=int(payload)))
            messages.append(dict(client_id=c,direction='downlink',kind='deployed_model',payload_bytes=n.state_bytes(terminal.global_state)))
    else:
        if job['scaling']=='shared':
            for c in data:
                messages.append(dict(client_id=c,direction='uplink',kind='TRAIN_scaler_count_sum_squaredsum',payload_bytes=8+16*width))
                messages.append(dict(client_id=c,direction='downlink',kind='TRAIN_scaler_mean_std',payload_bytes=16*width))
        messages.extend(terminal.messages)
    save(dst/'required_training_messages.json',messages)
    examples=sum(q['examples'] for q in terminal.local_steps);steps=sum(q['optimizer_steps'] for q in terminal.local_steps)
    assert examples==269100 and steps==(8460 if job['method']=='central' else 9000)
    model=n.ALCE(terminal.features,terminal.context,cfg);assert sum(q.numel() for q in model.parameters())==17330
    cost.update(logical_required_payload_bytes=sum(q['payload_bytes'] for q in messages),uplink_bytes=sum(q['payload_bytes'] for q in messages if q['direction']=='uplink'),
        downlink_bytes=sum(q['payload_bytes'] for q in messages if q['direction']=='downlink'),
        scaler_setup_bytes=sum(q['payload_bytes'] for q in messages if 'scaler' in q['kind']),
        model_tensor_bytes=sum(q['payload_bytes'] for q in messages if q['kind'] in {'initial_model','model_update','aggregated_model','deployed_model'}),
        all_round_VAL_audit_bytes=sum(q['payload_bytes'] for q in audit),
        three_checkpoint_VAL_selection_bytes_per_identity=5*3*16+5*8,
        three_checkpoint_VAL_selection_bytes_all_identities=(5*3*16+5*8)*len(selected.selected_rounds),
        aggregation_count=0 if not job['cadence'] else 60//job['cadence'],parameters=17330,model_state_bytes=n.state_bytes(terminal.global_state or next(iter(terminal.personal_states.values()))))
    save(dst/'cost.json',cost)
    r=dict(job=job,finished_utc=now(),examples=examples,optimizer_steps=steps,selected_rounds=selected.selected_rounds,cost=cost,
        files={q.relative_to(dst).as_posix():sha(q) for q in dst.rglob('*') if q.is_file()})
    save(dst/'complete.json',r)
    return r

def train():
    guard();jobs=load(RUN/'jobs_frozen.json')['jobs'];results=[]
    save(RUN/'training_started.json',dict(utc=now(),jobs_sha256=sha(RUN/'jobs_frozen.json'),EVAL_labels_read=False))
    for j in jobs:
        guard();r=worker(j);results.append(r)
        progress=dict(utc=now(),complete=len(results),total=len(jobs),last_arm=j['label'],last_seed=j['seed'],last_train_seconds=r['cost']['train_wall_seconds'])
        save(RUN/'progress.json',progress);print(json.dumps(progress),flush=True)
    save(RUN/'choices_frozen.json',dict(utc=now(),jobs=jobs,fit_complete_hashes={r['job']['id']:sha(RUN/'fits'/r['job']['id']/'complete.json') for r in results},EVAL_used_for_selection=False))
    save(RUN/'fit_complete.json',dict(utc=now(),status='COMPLETE',unique_new_fits=len(results),total_examples=sum(r['examples'] for r in results),total_steps=sum(r['optimizer_steps'] for r in results)))
    print('ALL40_FITS_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qa','prepare','train','guard']);a=p.parse_args()
    {'qa':implementation_qa,'prepare':prepare,'train':train,'guard':guard}[a.action]()
