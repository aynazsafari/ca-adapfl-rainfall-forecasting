"""Cadence intervention retaining parent ALCE, Adam resets and local RNG ticks.

Fitting returns terminal states plus separately audited validation selections.
Evaluation labels are never accepted here.
"""
from dataclasses import replace
import copy, time
import numpy as np
import torch
from caadapfl import neural_revision as n

def fit(clients, config, cadence):
    ids = tuple(c.client_id for c in clients)
    assert len(ids)==len(set(ids)) and len(ids)>=2
    features, context, window = clients[0].train.x.shape[2],clients[0].train.climate.shape[1],clients[0].train.x.shape[1]
    assert config.method in {'local','fedavg','caaa','central'}
    assert cadence==0 if config.method in {'local','central'} else cadence>0 and config.rounds%cadence==0
    previous_threads=torch.get_num_threads(); previous_det=torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(config.cpu_threads);torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed); template=n.ALCE(features,context,config)
        initial=n.clone_state(template); private=[n._PrivateClient(c,template) for c in clients]
        global_state=n.clone_state(initial); personal={c:n.clone_state(initial) for c in ids}
        server=n._Server(ids); counts=np.array([c.count for c in private],float);weights=counts/counts.sum()
        has_global=config.method!='local';has_personal=config.method!='central'
        if config.method in {'fedavg','caaa'}:
            for c in private:
                server.send(0,c.client_id,'initial_model',initial,'downlink')
                server.send(0,c.client_id,'train_count',np.array(c.count,np.int64),'uplink')
        central=None
        if config.method=='central':
            central=copy.deepcopy(private[0]);central.client_id='__central_pooled__';central.count=int(counts.sum())
            central._train=n._Tensors(*(torch.cat([getattr(c._train,key) for c in private]) for key in ('x','climate','drivers','y','heavy','target_mean','target_std')))
        curves=[];steps=[];round_states=[];audit=[];best={'global':-np.inf,'personal':-np.inf}
        best_global=None;best_personal={};selected={};train_wall=0.;train_cpu=0.;audit_wall=0.;audit_cpu=0.
        for rd in range(1,config.rounds+1):
            t=time.perf_counter();ct=time.process_time()
            updates=[]
            if config.method=='central':
                global_state,log,_=central.update(global_state,None,config,rd);steps.append(log)
            else:
                for c in private:
                    cid=c.client_id
                    if config.method=='local':start=personal[cid];reference=None;mu=0.;adaptive=False
                    else:
                        start=global_state if (rd-1)%cadence==0 else personal[cid]
                        reference=global_state;mu=config.prox_mu if config.method=='caaa' and config.use_prox else 0.
                        adaptive=config.method=='caaa' and config.adaptive_prox and mu>0
                    update,log,strength=c.update(start,reference,config,rd,adaptive=adaptive,mu=mu)
                    steps.append(log);updates.append(update);personal[cid]=update
                if config.method!='local' and rd%cadence==0:
                    for cid,update in zip(ids,updates):server.send(rd,cid,'model_update',update,'uplink')
                    # All declared CAAA aggregation weights are size-only in the selected configuration.
                    assert config.method!='caaa' or (config.alpha,config.beta,config.gamma)==(1.,0.,0.)
                    global_state=server.aggregate(updates,weights)
                    for cid in ids:server.send(rd,cid,'aggregated_model',global_state,'downlink')
                    curves.append(dict(round=rd,kind='aggregation',weights={c:float(w) for c,w in zip(ids,weights)}))
            train_wall+=time.perf_counter()-t;train_cpu+=time.process_time()-ct
            # Audit never affects optimization or the fixed terminal endpoint.
            t=time.perf_counter();ct=time.process_time()
            for identity,available in [('global',has_global),('personal',has_personal)]:
                if not available:continue
                scores={c.client_id:c.evaluate(global_state if identity=='global' else personal[c.client_id],'validation') for c in private}
                macro=float(np.mean([v['r2'] for v in scores.values()]))
                curves.append(dict(round=rd,kind='evaluation',identity=identity,validation_macro_r2=macro,validation=scores))
                for cid in ids:
                    audit.append(dict(round=rd,client_id=cid,kind='VAL_'+identity,direction='uplink',payload_bytes=16))
                if (rd in (20,40,60) or rd==config.rounds) and macro>best[identity]+1e-12:
                    best[identity]=macro;selected[identity]=rd
                    if identity=='global':best_global=n.clone_state(global_state)
                    else:best_personal={c:n.clone_state(s) for c,s in personal.items()}
            audit_wall+=time.perf_counter()-t;audit_cpu+=time.process_time()-ct
            if rd==1 or rd==config.rounds or rd%config.state_checkpoint_interval==0:
                round_states.append(dict(round=rd,global_state=n.clone_state(global_state) if has_global else None,
                    personal_states={c:n.clone_state(s) for c,s in personal.items()} if has_personal else {}))
        manifest={c.client_id:{s:dict(count=len(getattr(c,s).keys),first=getattr(c,s).keys[0],last=getattr(c,s).keys[-1],keys=getattr(c,s).keys) for s in ('train','monitor','validation')} for c in clients}
        stats={c.client_id:(c.target_mean,c.target_std) for c in clients}
        terminal=n.FitResult(config,features,context,window,ids,stats,n.clone_state(global_state) if has_global else None,
            {c:n.clone_state(s) for c,s in personal.items()} if has_personal else {},
            {x:config.rounds for x in ('global','personal') if (has_global if x=='global' else has_personal)},
            curves,steps,server.messages,manifest,time.perf_counter()-started,round_states)
        sensitivity=n.FitResult(config,features,context,window,ids,stats,best_global,best_personal,selected,curves,steps,server.messages,manifest,terminal.elapsed_seconds,[])
        for identity in selected:
            for cid in ids:audit.append(dict(round=selected[identity],client_id=cid,kind='chosen_round_'+identity,direction='downlink',payload_bytes=8))
        cost=dict(train_wall_seconds=train_wall,train_cpu_seconds=train_cpu,validation_audit_wall_seconds=audit_wall,validation_audit_cpu_seconds=audit_cpu,
            total_engine_wall_seconds=terminal.elapsed_seconds,validation_audit_messages=audit)
        return terminal,sensitivity,cost
    finally:
        torch.set_num_threads(previous_threads);torch.use_deterministic_algorithms(previous_det)
