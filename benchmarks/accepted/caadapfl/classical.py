"""Portable classical controls. Fit functions see only explicitly admitted months.

All targets and predictions are raw millimetres. Pooled controls optimize raw
pooled squared error and are labeled accordingly; local controls fit separately.
No prediction clipping, target fallback, missing-value imputation, or evaluation
label access occurs in fitting, feature selection or router calibration.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Mapping,Sequence
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

TRAIN_END='2007-11'
DEVELOPMENT_FOLDS=(('1995-11','1996-01','1998-12'),('2000-11','2001-01','2003-12'),('2004-11','2005-01','2007-11'))
ROUTING_FIT=('2011-01','2013-11')
ROUTING_SCORE=('2014-01','2016-11')

def _months(values):
    result=np.asarray(values,dtype=str)
    if result.ndim!=1 or len(set(result))!=len(result):raise ValueError('Unique month vector required')
    if not np.array_equal(result,np.asarray(pd.PeriodIndex(result,freq='M').astype(str))):raise ValueError('Canonical YYYY-MM keys required')
    if len(result)>1 and not (result[1:]>result[:-1]).all():raise ValueError('Months must be chronological')
    return result

def _admitted(samples,start=None,end=TRAIN_END):
    """Read future key strings solely to gate them; do not inspect future x/y."""
    output={}
    for client,d in samples.items():
        keys=_months(d['target_month']);mask=keys<=end
        if start is not None:mask &= keys>=start
        if not mask.any():raise ValueError(f'{client}: empty admitted period')
        x=np.asarray(d['x'])[mask];y=np.asarray(d['y'])[mask]
        if x.ndim!=3 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError(f'{client}: invalid admitted arrays')
        output[client]={'x':x,'y':y,'target_month':keys[mask]}
    reference=next(iter(output.values()))['target_month']
    if any(not np.array_equal(d['target_month'],reference) for d in output.values()):raise ValueError('Clients must share identical admitted months')
    return output

def _design(x,indices,representation):
    if representation=='latest':return np.asarray(x[:, -1, indices],dtype=np.float64)
    if representation=='flat_sequence':return np.asarray(x[:,:,indices],dtype=np.float64).reshape(len(x),-1)
    raise ValueError('Representation must be latest or flat_sequence')

def _r2(y,p):
    denominator=float(np.sum((y-np.mean(y))**2))
    if denominator<=0:raise ValueError('R2 undefined for constant observations; no denominator floor')
    return 1-float(np.sum((y-p)**2))/denominator

def macro_r2(frame,column):
    return float(np.mean([_r2(g.y.to_numpy(float),g[column].to_numpy(float)) for _,g in frame.groupby('client',sort=True)]))

def default_feature_groups(names,climate_names=()):
    names=tuple(names);core=tuple(n for n in names if n.startswith(('season_','rain_')))
    if not core:raise ValueError('CORE calendar/rainfall features absent')
    climate=tuple(n for n in names if n in set(core)|set(climate_names))
    return {'CORE':core,'CLIMATE':climate,'FULL':names}

@dataclass(frozen=True)
class ControlConfig:
    family:str
    scope:str
    feature_group:str='CORE'
    representation:str='latest'
    seed:int=0
    ridge_alpha:float=10.0
    n_estimators:int=256
    min_samples_leaf:int=3
    max_features:float=1.0
    n_jobs:int=1
    target_mode:str='raw'
    def label(self):
        scope=('client_normalized_pooled' if self.target_mode=='client_normalized' else 'raw_pooled') if self.scope=='pooled' else 'local'
        sensitivity='_flat_sequence_sensitivity' if self.representation=='flat_sequence' else '_latest'
        return f'{scope}_{self.family}_{self.feature_group}{sensitivity}'

def _estimator(config):
    if config.scope not in ('pooled','local'):raise ValueError('Explicit pooled/local scope required')
    if config.family=='Ridge':return make_pipeline(StandardScaler(),Ridge(alpha=config.ridge_alpha))
    arguments=dict(n_estimators=config.n_estimators,min_samples_leaf=config.min_samples_leaf,max_features=config.max_features,random_state=config.seed,n_jobs=config.n_jobs)
    if config.family=='ET':return ExtraTreesRegressor(**arguments)
    if config.family=='RF':return RandomForestRegressor(**arguments)
    raise ValueError('Families: Ridge, ET, RF')

@dataclass
class FittedControl:
    config:ControlConfig
    indices:tuple[int,...]
    feature_names:tuple[str,...]
    models:dict
    receipt:dict
    target_stats:dict=field(default_factory=dict)
    def predict(self,samples,start,end):
        selected=_admitted(samples,start,end);rows=[]
        for client,d in selected.items():
            model=self.models['__pooled__'] if self.config.scope=='pooled' else self.models[client]
            prediction=np.asarray(model.predict(_design(d['x'],self.indices,self.config.representation)),float)
            if self.config.target_mode=='client_normalized':
                mean,std=self.target_stats[client];prediction=mean+std*prediction
            if prediction.shape!=d['y'].shape or not np.isfinite(prediction).all():raise ValueError('Missing/nonfinite prediction; no fallback')
            rows.extend({'client':client,'target_month':month,'y':float(y),'prediction':float(pred),'model':self.config.label(),'seed':self.config.seed} for month,y,pred in zip(d['target_month'],d['y'],prediction))
        return pd.DataFrame(rows)

def fit_control(samples,names,group,config,end=TRAIN_END):
    if end>TRAIN_END:raise ValueError('Classical controls may only fit gradient-training months')
    names=tuple(names);group=tuple(group)
    if len(set(group))!=len(group) or any(n not in names for n in group):raise ValueError('Invalid feature group')
    if config.target_mode not in ('raw','client_normalized'):raise ValueError('Unknown target mode')
    if config.target_mode=='client_normalized' and config.scope!='pooled':raise ValueError('Normalized pooled control requires pooled scope')
    selected=_admitted(samples,end=end);indices=tuple(names.index(n) for n in group);models={};target_stats={}
    if config.scope=='pooled':
        x=np.concatenate([_design(d['x'],indices,config.representation) for d in selected.values()])
        if config.target_mode=='client_normalized':
            for client,d in selected.items():
                mean,std=float(d['y'].mean()),float(d['y'].std(ddof=0))
                if std<=0:raise ValueError('Degenerate client training target; no normalization floor')
                target_stats[client]=(mean,std)
            y=np.concatenate([(d['y']-target_stats[client][0])/target_stats[client][1] for client,d in selected.items()])
        else:y=np.concatenate([d['y'] for d in selected.values()])
        models['__pooled__']=_estimator(config).fit(x,y)
    else:
        for client,d in selected.items():models[client]=_estimator(config).fit(_design(d['x'],indices,config.representation),d['y'])
    sample=next(iter(selected.values()));receipt={'label':config.label(),'fit_end':end,'fit_clients':list(selected),'fit_months':sample['target_month'].tolist(),'fit_n_total':sum(len(d['y']) for d in selected.values()),'input_feature_names':list(group),'representation':config.representation,'input_dimensions':len(group)*(sample['x'].shape[1] if config.representation=='flat_sequence' else 1),'client_identity_input':False,'target':'raw rainfall mm','loss_population':'raw pooled samples' if config.scope=='pooled' else 'separate client samples','preprocessing':'Ridge StandardScaler fit on admitted fold only; trees raw inputs','config':vars(config),'negative_prediction_policy':'retained without clipping','future_labels_accessed':False}
    receipt['target_mode']=config.target_mode
    if config.target_mode=='client_normalized':
        receipt['target']='client training-standardized rainfall; decode with same client training mean/std'
        receipt['loss_population']='pooled client-normalized squared error, matching neural central target population'
        receipt['training_target_summary_sharing']={'client_stats':target_stats,'wire_format':'two float64 training mean/std per client','bytes':16*len(selected),'privacy':'Plain training summaries, no DP or secure aggregation claimed'}
    return FittedControl(config,indices,group,models,receipt,target_stats)

def select_feature_group(samples,names,groups,minimum_macro_r2_gain=.002,selection_end=TRAIN_END):
    """Three fixed groups, three chronological folds, Ridge only; no broad screen."""
    if selection_end>TRAIN_END:raise ValueError('Feature selection must remain training only')
    admitted=_admitted(samples,end=selection_end);results=[]
    folds=[fold for fold in DEVELOPMENT_FOLDS if fold[2]<=selection_end]
    if not folds:
        receipt={'selected_source_group':'CORE','selected_feature_names':list(groups['CORE']),'rule':'No fully admitted frozen development fold: retain CORE without selection','minimum_gain':minimum_macro_r2_gain,'mean_fold_macro_R2':{},'fold_results':[],'candidate_groups':{g:list(groups[g]) for g in ('CORE','CLIMATE','FULL')},'selection_end':selection_end,'future_labels_accessed':False,'selector_model':'raw pooled Ridge alpha10, latest row','selection_changes_main_groups':False}
        return tuple(groups['CORE']),receipt
    for group in ('CORE','CLIMATE','FULL'):
        if group not in groups:raise ValueError('Three fixed groups required')
        for train_end,start,end in folds:
            cfg=ControlConfig('Ridge','pooled',group)
            fitted=fit_control(admitted,names,groups[group],cfg,train_end)
            frame=fitted.predict(admitted,start,end)
            results.append({'group':group,'train_end':train_end,'score_start':start,'score_end':end,'macro_R2':macro_r2(frame,'prediction'),'feature_count':len(groups[group]),'preprocessing_fit_end':train_end})
    table=pd.DataFrame(results);means=table.groupby('group').macro_R2.mean().to_dict();ordered=sorted(means,key=lambda g:(-means[g],len(groups[g]),g));best=ordered[0]
    chosen=best if means[best]-means['CORE']>=minimum_macro_r2_gain else 'CORE'
    receipt={'selected_source_group':chosen,'selected_feature_names':list(groups[chosen]),'rule':f'Highest mean chronological-fold macro R2, requiring >= {minimum_macro_r2_gain} improvement over CORE; ties prefer fewer features','minimum_gain':minimum_macro_r2_gain,'mean_fold_macro_R2':means,'fold_results':results,'candidate_groups':{g:list(groups[g]) for g in ('CORE','CLIMATE','FULL')},'selection_end':selection_end,'future_labels_accessed':False,'selector_model':'raw pooled Ridge alpha10, latest row','selection_changes_main_groups':False}
    return tuple(groups[chosen]),receipt

def temporal_oof_controls(samples,names,groups,configs):
    """Each OOF row is predicted by a model fit strictly before its target fold."""
    admitted=_admitted(samples,end=TRAIN_END);wide=[];receipts=[]
    for train_end,start,end in DEVELOPMENT_FOLDS:
        fold=None
        nested_group,nested_receipt=None,None
        if any(cfg.feature_group=='TRAIN_SELECTED' for cfg in configs):
            nested_group,nested_receipt=select_feature_group(admitted,names,groups,selection_end=train_end)
        for cfg in configs:
            group=nested_group if cfg.feature_group=='TRAIN_SELECTED' else groups[cfg.feature_group]
            fitted=fit_control(admitted,names,group,cfg,train_end);pred=fitted.predict(admitted,start,end);label=cfg.label();part=pred[['client','target_month','y','prediction']].rename(columns={'prediction':label})
            fold=part if fold is None else fold.merge(part,on=['client','target_month','y'],how='outer',validate='one_to_one')
            receipts.append(dict(fitted.receipt,OOF_score_start=start,OOF_score_end=end,OOF_nested_feature_selection=nested_receipt if cfg.feature_group=='TRAIN_SELECTED' else None))
        if fold is None:raise ValueError('Empty expert pool')
        fold['OOF_fit_end']=train_end;wide.append(fold)
    result=pd.concat(wide,ignore_index=True).sort_values(['client','target_month']);expert_names=[c.label() for c in configs]
    if len(set(expert_names))!=len(expert_names) or result.duplicated(['client','target_month']).any() or not np.isfinite(result[['y']+expert_names].to_numpy(float)).all():raise ValueError('Invalid OOF pool or keys')
    return result,receipts

@dataclass
class Baselines:
    seasonal:dict
    latest_index:int
    previous_year_index:int
    receipt:dict
    def predict(self,samples,start,end,seed=0):
        admitted=_admitted(samples,start,end);rows=[]
        for client,d in admitted.items():
            latest=d['x'][:,-1,self.latest_index];last_year=d['x'][:,-1,self.previous_year_index]
            for i,month in enumerate(d['target_month']):
                values={'training_seasonal_mean':self.seasonal[client][int(month[-2:])],'available_rainfall_target_minus2':float(latest[i]),'same_calendar_target_minus12':float(last_year[i])}
                for model,pred in values.items():
                    if not np.isfinite(pred):raise ValueError('Nonfinite baseline; no fallback')
                    rows.append({'client':client,'target_month':month,'y':float(d['y'][i]),'prediction':pred,'model':model,'seed':seed})
        return pd.DataFrame(rows)

def fit_baselines(samples,names,support=None):
    names=tuple(names);last='rain_origin_lag1';year='rain_origin_lag11'
    if support is not None:
        if support[last]['latest_origin_offset']!=-1 or support[year]['latest_origin_offset']!=-11:raise ValueError('This baseline contract requires target-2 and target-12 support; pass separately keyed values for another latency scenario')
    admitted=_admitted(samples,end=TRAIN_END);seasonal={}
    for client,d in admitted.items():
        months=np.array([int(m[-2:]) for m in d['target_month']]);seasonal[client]={}
        for m in range(1,13):
            values=d['y'][months==m]
            if not len(values):raise ValueError('Seasonal mean training month missing')
            seasonal[client][m]=float(values.mean())
    return Baselines(seasonal,names.index(last),names.index(year),{'fit_end':TRAIN_END,'seasonal_means':seasonal,'latest_available_target_offset':-2,'same_calendar_previous_year_target_offset':-12,'future_labels_accessed':False})

def _expert_frame(frame,experts,start,end):
    if not {'client','target_month','y',*experts}.issubset(frame.columns):raise ValueError('Missing expert/target columns')
    # Gate by keys before reading any covariates/targets.
    result=frame.loc[(frame.target_month>=start)&(frame.target_month<=end),['client','target_month','y',*experts]].copy().sort_values(['client','target_month'])
    expected=np.asarray(pd.period_range(start,end,freq='M').astype(str))
    if result.duplicated(['client','target_month']).any():raise ValueError('Duplicate router keys')
    for c,g in result.groupby('client'):
        if not np.array_equal(g.target_month.to_numpy(),expected):raise ValueError(f'{c}: incomplete common router period')
    if result.empty or not np.isfinite(result[['y',*experts]].to_numpy(float)).all():raise ValueError('Nonfinite/empty expert pool; no fallback')
    return result

def _convex_weights(y,x,robust=False):
    full=float(np.sum((y-y.mean())**2));half=len(y)//2;recent=float(np.sum((y[half:]-y[half:].mean())**2))
    if full<=0 or (robust and recent<=0):raise ValueError('Router R2 undefined; no floors')
    observation_weight=np.full(len(y),(.65 if robust else 1.)/full)
    if robust:observation_weight[half:]+=.35/recent
    objective=lambda w:float(np.sum(observation_weight*(x@w-y)**2))
    gradient=lambda w:2*x.T@(observation_weight*(x@w-y))
    initial=np.full(x.shape[1],1/x.shape[1]);solution=minimize(objective,initial,jac=gradient,bounds=[(0.,1.)]*len(initial),constraints=[{'type':'eq','fun':lambda w:float(w.sum()-1),'jac':lambda w:np.ones_like(w)}],method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    if not solution.success or abs(solution.x.sum()-1)>1e-8 or solution.x.min()<-1e-8:raise RuntimeError('Convex router optimization failed: '+solution.message)
    return solution.x

@dataclass
class RouterSuite:
    experts:tuple[str,...]
    clients:tuple[str,...]
    weights:dict
    stackers:dict
    receipt:dict
    def predict(self,frame,start,end):
        admitted=_expert_frame(frame,self.experts,start,end);parts=[]
        for client,g in admitted.groupby('client',sort=True):
            if client not in self.clients:raise ValueError('Unseen router client')
            x=g[list(self.experts)].to_numpy(float);result=g[['client','target_month','y']].copy()
            for name,byclient in self.weights.items():result[name]=x@byclient[client]
            result['temporal_OOF_Ridge_stack']=self.stackers[client].predict(x)
            if not np.isfinite(result.drop(columns=['client','target_month']).to_numpy(float)).all():raise ValueError('Nonfinite router predictions')
            parts.append(result)
        return pd.concat(parts,ignore_index=True)

def _routing_weights(calibration,experts):
    weights={name:{} for name in ['validation_best_expert','equal_weight_experts','validation_convex_stack','robust_validation_router']}
    for client,g in calibration.groupby('client',sort=True):
        y=g.y.to_numpy(float);x=g[list(experts)].to_numpy(float);scores=np.array([_r2(y,x[:,i]) for i in range(x.shape[1])]);best=np.zeros(x.shape[1]);best[int(np.argmax(scores))]=1.
        weights['validation_best_expert'][client]=best;weights['equal_weight_experts'][client]=np.full(x.shape[1],1/x.shape[1]);weights['validation_convex_stack'][client]=_convex_weights(y,x);weights['robust_validation_router'][client]=_convex_weights(y,x,True)
    return weights

def fit_router_suite(validation_frame,training_oof_frame,expert_names,stack_alpha=10.):
    experts=tuple(expert_names)
    if not experts or len(set(experts))!=len(experts):raise ValueError('Unique common expert pool required')
    # This function rejects evaluation rows rather than silently admitting them.
    if (validation_frame.target_month<'2011-01').any() or (validation_frame.target_month>'2016-11').any():raise ValueError('Router input must contain safe-label validation only; December2016 is unreleased at January2017 issue')
    if (training_oof_frame.target_month>TRAIN_END).any():raise ValueError('Stacker OOF labels must be training only')
    if 'OOF_fit_end' not in training_oof_frame:raise ValueError('OOF fitting provenance is required')
    fit_end=pd.PeriodIndex(training_oof_frame.OOF_fit_end,freq='M');latest_released=pd.PeriodIndex(training_oof_frame.target_month,freq='M')-2
    if not (fit_end<=latest_released).all():raise ValueError('OOF fit end must be <= target minus2: previous-month CHIRPS labels are unreleased at first-day issue')
    if training_oof_frame.duplicated(['client','target_month']).any() or not np.isfinite(training_oof_frame[['y',*experts]].to_numpy(float)).all():raise ValueError('Invalid temporal OOF pool')
    calibration=_expert_frame(validation_frame,experts,*ROUTING_FIT);score=_expert_frame(validation_frame,experts,*ROUTING_SCORE);clients=tuple(sorted(calibration.client.unique()));stackers={}
    if set(training_oof_frame.client.unique())!=set(clients):raise ValueError('OOF client pool differs')
    for c,g in training_oof_frame.groupby('client',sort=True):stackers[c]=make_pipeline(StandardScaler(),Ridge(alpha=stack_alpha)).fit(g[list(experts)].to_numpy(float),g.y.to_numpy(float))
    fit_weights=_routing_weights(calibration,experts);provisional=RouterSuite(experts,clients,fit_weights,stackers,{})
    score_predictions=provisional.predict(score,*ROUTING_SCORE);columns=list(fit_weights)+['temporal_OOF_Ridge_stack'];scores={col:{'macro_R2':macro_r2(score_predictions,col),'per_client_R2':{c:_r2(g.y.to_numpy(float),g[col].to_numpy(float)) for c,g in score_predictions.groupby('client')}} for col in columns}
    full=_expert_frame(validation_frame,experts,'2011-01','2016-11');final_weights=_routing_weights(full,experts)
    receipt={'expert_pool':list(experts),'same_pool_for_all_routers':True,'weight_scope':'separate client convex/static weights','routing_fit_fold':ROUTING_FIT,'routing_score_fold':ROUTING_SCORE,'heldout_routing_scores':scores,'robust_objective':'.65 overall calibration R2 + .35 chronological last-half calibration R2','final_refit_period':['2011-01','2016-11'],'heldout_routing_scores_describe_provisional_weights_not_final_refit':True,'final_weights':{name:{c:w.tolist() for c,w in byclient.items()} for name,byclient in final_weights.items()},'provisional_weights':{name:{c:w.tolist() for c,w in byclient.items()} for name,byclient in fit_weights.items()},'OOF_stack_alpha':stack_alpha,'OOF_months':sorted(training_oof_frame.target_month.unique()),'OOF_only_before':TRAIN_END,'OOF_label_release_embargo':'fit end <= heldout target minus2; CHIRPS assumed day21 following month','evaluation_labels_accessed':False,'output_clipping':False,'router_architecture':'Static validation-calibrated expert router; no climate-dependent neural gating claim'}
    return RouterSuite(experts,clients,final_weights,stackers,receipt),score_predictions
