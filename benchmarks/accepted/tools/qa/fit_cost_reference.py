"""Independent saved-receipt chronology, checkpoint and cost checks."""
from collections import Counter
import math
import numpy as np
import pandas as pd

def require(c,m):
    if not c:raise AssertionError(m)
def months(a,b):return list(pd.period_range(a,b,freq='M').astype(str))

def neural_cost(path,receipt,read):
    cfg=receipt['job']['config'];meta=read(path/'model/manifest.json');steps=read(path/'model/local_steps.json');messages=read(path/'model/messages.json');curves=read(path/'model/curves.json')
    require(meta['config']==cfg and cfg['rounds']==60 and cfg['local_epochs']==3 and cfg['batch_size']==32 and cfg['cpu_threads']==2,'Neural protocol/config mismatch')
    ids=meta['client_ids'];require(len(ids)==5 and len(set(ids))==5,'Five clients required')
    start='1983-01' if receipt['job']['scenario']=='MAIN' else '1983-02'
    role_keys={'train':months(start,'2007-11'),'monitor':months('2008-01','2010-11'),'validation':months('2011-01','2016-11')}
    ntrain=len(role_keys['train']);require(ntrain in (299,298),'Training cohort mismatch')
    for client in ids:
        for role,keys in role_keys.items():
            data=meta['data_manifest'][client][role]
            require(data['keys']==[k+'-01' for k in keys] and data['count']==len(keys),'Neural role calendar mismatch')
        require(meta['data_manifest'][client]['train']['last']=='2007-11-01' and meta['data_manifest'][client]['monitor']['last']=='2010-11-01' and meta['data_manifest'][client]['validation']['last']=='2016-11-01','Unreleased December labels admitted')
    require(cfg['seed'] in (42,123,2024,7,999),'Nonfrozen random seed')
    require(meta['rainfall_prediction_policy']=='raw inverse-transformed mm; no clipping in checkpoint scores or inference','Prediction clipping policy changed')
    method=cfg['method'];streams=['main','ditto_personal'] if method=='ditto' else ['main']
    if method=='central':expected={(r,'__central_pooled__','main') for r in range(1,61)};per_count=ntrain*5
    else:expected={(r,c,s) for r in range(1,61) for c in ids for s in streams};per_count=ntrain
    require(len(steps)==len(expected) and {(r['round'],r['client_id'],r['stream']) for r in steps}==expected,'Actual local update coverage mismatch')
    for row in steps:
        require(row['optimizer_steps']==3*math.ceil(per_count/32) and row['examples']==3*per_count,'Actual optimizer steps/examples differ')
        require(row['optimizer']==cfg['optimizer'] and row['sgd_steps']==(row['optimizer_steps'] if cfg['optimizer']=='sgd' else 0),'Optimizer ledger mislabel')
        require(all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))),'Nonfinite update cost/objective record')
        if method in ('fedavg','central','local') or not cfg['use_prox']:
            require(row['lambda_mean']==row['prox_mean']==0,'Disabled proximity remains active')
        elif method=='ditto' and row['stream']=='main':
            require(row['lambda_mean']==row['prox_mean']==0,'Ditto global stream applies a personalized penalty')
        elif method in ('fedprox','ditto','climate_graph') or not cfg['adaptive_prox']:
            require(math.isclose(row['lambda_mean'],cfg['prox_mu'],abs_tol=1e-7) and math.isclose(row['lambda_min'],cfg['prox_mu'],abs_tol=1e-7) and math.isclose(row['lambda_max'],cfg['prox_mu'],abs_tol=1e-7),'Fixed proximal coefficient changed')
        elif method=='caaa':
            require(row['lambda_min']>=.3*cfg['prox_mu']-1e-7 and row['lambda_max']<=cfg['prox_mu']+1e-7 and row['lambda_max']>=row['lambda_min'],'Adaptive proximity outside declared [.3mu,mu] bounds')
    hasglobal=method not in ('local','climate_graph');haspersonal=method!='central'
    available=[name for name,flag in [('global',hasglobal),('personal',haspersonal)] if flag]
    selected={}
    for identity in available:
        rows=[r for r in curves if r.get('kind')=='evaluation' and r.get('identity')==identity]
        require([r['round'] for r in rows]==list(range(1,61)),'Not all60 checkpoint rounds completed')
        best=-math.inf;selected_round=None
        for r in rows:
            require(set(r['validation'])==set(ids) and all(v['n']==71 for v in r['validation'].values()),'Checkpoint outer validation cohort changed')
            require(set(r['inner_monitor'])==set(ids) and all(v['n']==35 for v in r['inner_monitor'].values()),'Checkpoint monitor cohort changed')
            score=sum(v['r2'] for v in r['validation'].values())/5
            require(math.isclose(score,r['validation_macro_r2'],rel_tol=0,abs_tol=1e-14),'Checkpoint macro arithmetic')
            if score>best+1e-12:best=score;selected_round=r['round']
        require(meta['selected_rounds'][identity]==selected_round,'Wrong checkpoint chosen')
        selected[identity]=selected_round
    counts=Counter(m['kind'] for m in messages)
    if method in ('central','local'):expected_messages={}
    else:
        expected_messages={'initial_model':5,'train_count':5,'model_update':300}
        expected_messages['personal_graph_reference' if method=='climate_graph' else 'global_reference']=300
        for identity in available:
            expected_messages['outer_validation_metrics_'+identity]=300
            expected_messages['inner_monitor_diagnostic_metrics_'+identity]=300
        if method=='climate_graph':expected_messages['public_training_graph']=1
        if method=='caaa':
            expected_messages['sensitivity']=5
            if cfg['use_performance']:expected_messages['inner_monitor_loss']=300
            if cfg['use_climate']:expected_messages['round_climate_strength']=300
    require(dict(counts)==expected_messages,'Algorithm communication messages differ from actual protocol')
    model_sizes={m['payload_bytes'] for m in messages if m['kind'] in ('initial_model','model_update','global_reference','personal_graph_reference')}
    if model_sizes:require(len(model_sizes)==1 and next(iter(model_sizes))>0,'Parameter wire sizes differ')
    for m in messages:
        kind=m['kind'];wire=m['payload_bytes']
        if kind=='train_count':require(wire==8,'Training count is not int64')
        if kind.startswith(('outer_validation_metrics_','inner_monitor_diagnostic_metrics_')):require(wire==16 and m['direction']=='uplink','Metric summary wire differs')
        if kind in ('sensitivity','round_climate_strength'):require(wire==16 and m['direction']=='uplink','Climate summary wire differs')
        if kind=='inner_monitor_loss':require(wire==8 and m['direction']=='uplink','Monitor loss wire differs')
        if kind=='public_training_graph':require(wire==100 and m['direction']=='setup_artifact','Graph matrix alias wire differs')
    setup=read(path/'setup_ledger.json');width=meta['features'];scaler=setup['input_scaler']
    require(scaler['total_bytes']==sum(m['bytes'] for m in scaler['messages'])==5*(8+32*width),'Scaler summary bytes differ')
    require(setup['target_stats_export']['total_bytes']==80,'Target decode summary bytes differ')
    if method!='climate_graph':require(setup['graph_setup_artifact'] is None,'NonGraph algorithm charged privileged graph')
    if method=='caaa':require(setup['sensitivity_summary']['additional_charge_bytes']==0,'Sensitivity summary double charged')
    row={'optimizer_steps':sum(r['optimizer_steps'] for r in steps),'training_examples_presented':sum(r['examples'] for r in steps),'message_uplink_bytes':sum(m['payload_bytes'] for m in messages if m['direction']=='uplink'),'message_downlink_bytes':sum(m['payload_bytes'] for m in messages if m['direction']=='downlink'),'graph_matrix_setup_delivery_alias_bytes':sum(m['payload_bytes'] for m in messages if m['direction']=='setup_artifact'),'input_scaler_setup_bytes':scaler['total_bytes'],'target_decode_summary_bytes':80,'centralized_training_array_access_bytes':0}
    if method=='central':row['centralized_training_array_access_bytes']=ntrain*5*4*(12*width+meta['context']+24+2)
    return row,{'method':method,'seed':cfg['seed'],'training_count_per_client':ntrain,'selected_rounds':selected,'actual_update_streams':streams,'messages':dict(counts)}

def other_cost(path,receipt,read):
    kind=receipt['job']['kind'];data=read(path/'fit_receipt.json')
    empty={k:0 for k in ['optimizer_steps','training_examples_presented','message_uplink_bytes','message_downlink_bytes','graph_matrix_setup_delivery_alias_bytes','input_scaler_setup_bytes','target_decode_summary_bytes','centralized_training_array_access_bytes']}
    if kind=='classical':
        cfg=receipt['job']['config'];require(data['config']==cfg and data['fit_end']=='2007-11' and data['future_labels_accessed'] is False,'Classical fit boundaries/config')
        require(data['fit_months']==months('1983-01','2007-11') and data['fit_n_total']==1495,'Classical training cohort')
        require(data['negative_prediction_policy']=='retained without clipping' and data['client_identity_input'] is False,'Control policy mislabel')
        require(data['target_mode']==cfg.get('target_mode','raw'),'Control target population')
        if cfg['scope']=='pooled':empty['centralized_training_array_access_bytes']=data['fit_n_total']*8*(data['input_dimensions']+1)
        empty['target_decode_summary_bytes']=data.get('training_target_summary_sharing',{}).get('bytes',0)
        require(empty['target_decode_summary_bytes']==(80 if cfg.get('target_mode')=='client_normalized' else 0),'Normalized control summary bytes')
    elif kind=='baselines':
        require(data['fit_end']=='2007-11' and data['future_labels_accessed'] is False and data['latest_available_target_offset']==-2 and data['same_calendar_previous_year_target_offset']==-12,'Baseline source availability')
    elif kind=='router':
        require(data['routing_fit_fold']==['2011-01','2013-11'] and data['routing_score_fold']==['2014-01','2016-11'] and data['final_refit_period']==['2011-01','2016-11'],'Router fit/score embargo')
        require(data['same_pool_for_all_routers'] is True and data['evaluation_labels_accessed'] is False and data['output_clipping'] is False and len(data['expert_pool'])==6,'Router pool or target leak')
        frame=pd.read_csv(path/'training_oof_predictions.csv',dtype={'target_month':str,'OOF_fit_end':str})
        require(not frame.duplicated(['client','target_month']).any() and (pd.PeriodIndex(frame.OOF_fit_end,freq='M')<=pd.PeriodIndex(frame.target_month,freq='M')-2).all() and frame.target_month.le('2007-11').all(),'OOF release embargo or duplication')
        oof=read(path/'oof_fit_receipts.json');require(len(oof)==18,'Six experts × three developmental OOF folds required')
        for r in oof:require(pd.Period(r['fit_end'])<=pd.Period(r['OOF_score_start'])-2 and r['OOF_score_end']<='2007-11','OOF receipt chronology')
        held=pd.read_csv(path/'routing_heldout_predictions.csv',usecols=['client','target_month'])
        require(len(held)==175 and held.target_month.min()=='2014-01' and held.target_month.max()=='2016-11','Router heldout scope')
    else:raise ValueError('Unknown fit kind')
    return empty,{'kind':kind,'receipt_admitted':True}
