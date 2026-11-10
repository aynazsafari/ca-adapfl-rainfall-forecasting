"""Meaningful unit cases and SYNTHETIC smoke only. No actual climate data read."""
from pathlib import Path
import sys,json,copy
import numpy as np,pandas as pd
from classical import (ControlConfig,fit_control,default_feature_groups,select_feature_group,fit_baselines,temporal_oof_controls,fit_router_suite)
from evaluation import (regression_metrics,evaluate_predictions,training_heavy_thresholds,paired_block_bootstrap,circular_date_block_indices,block_sign_randomization,holm_adjust,validate_prediction_frame)

OUT=Path(__file__).resolve().parent/'synthetic_only';OUT.mkdir(exist_ok=True);checks=[]
def check(name,passed,details=None):
    checks.append({'check':name,'passed':bool(passed),'details':details})
    if not passed:raise AssertionError(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,RuntimeError):check(name,True);return
    raise AssertionError(name+' did not reject')

# Different client scales make macro and pooled fundamentally different.
rows=[]
for c,y,p in [('A',[0,1,2,3],[0,1,2,2]),('B',[100,200,300,400],[250,250,250,250])]:
    rows.extend({'model':'manual','seed':0,'client':c,'target_month':m,'y':a,'prediction':b} for m,a,b in zip(pd.period_range('2017-01','2017-04',freq='M').astype(str),y,p))
manual=evaluate_predictions(pd.DataFrame(rows),{'A':1.5,'B':250});a=manual['aggregate_per_seed'];macro=a.loc[a.scope=='macro_equal_client'].iloc[0];pooled=a.loc[a.scope=='pooled_all_client_months'].iloc[0]
check('manual_macro_R2_is_mean_client_R2',abs(macro.R2-.4)<1e-14,{'macro_R2':float(macro.R2),'pooled_R2':float(pooled.R2)})
check('manual_macro_RMSE_is_mean_client_RMSE',abs(macro.RMSE_mm-(.5+np.sqrt(12500))/2)<1e-14)
check('macro_and_pooled_differ',abs(macro.R2-pooled.R2)>.2 and abs(macro.RMSE_mm-pooled.RMSE_mm)>10)
metric=regression_metrics(np.array([1,2,3]),np.array([-1,-2,-3]));check('negative_predictions_retained_and_PBIAS_sign_explicit',metric['PBIAS_signed_percent']==200 and metric['forecast_bias_signed_percent']==-200 and abs(metric['RMSE_mm']-np.sqrt(56/3))<1e-14)
metric=regression_metrics(np.ones(3),np.ones(3));check('constant_metric_undefined_without_hidden_floor',np.isnan(metric['R2']) and 'constant observations' in metric['undefined_reasons'])
client=manual['per_client'].set_index('client');check('actual_heavy_confusion_uses_observed_training_threshold',client.loc['A','heavy_TP']==2 and client.loc['B','heavy_FN']==2 and client.loc['B','heavy_TP']==0)
proba=pd.DataFrame(rows);proba['heavy_probability']=np.tile([.9,.9,.1,.1],2);pmetrics=evaluate_predictions(proba,{'A':1.5,'B':250});pc=pmetrics['per_client'].set_index('client');check('heavy_probability_confusion_uses_actual_events_and_declared_decision',pc.loc['A','heavy_TP']==0 and pc.loc['A','heavy_FP']==2 and pc.loc['A','heavy_FN']==2)
check('Holm_manual_case',holm_adjust({'a':.01,'b':.03,'c':.04})=={'a':.03,'b':.06,'c':.06})

# All following data are generated here; no scientific data files are opened.
rng=np.random.default_rng(64);calendar=pd.period_range('1981-01','2024-12',freq='M');names=('season_sin1','season_cos1','season_sin2','season_cos2','rain_origin_lag1','rain_origin_lag2','rain_origin_lag5','rain_origin_lag11','fake_nino','fake_T2M');samples={}
for ci,c in enumerate(['SYNTH_A','SYNTH_B']):
    season=np.sin(2*np.pi*np.arange(len(calendar))/12);signal=60+25*season+15*ci+np.arange(len(calendar))*.03+rng.normal(0,2,len(calendar));climate=rng.normal(0,1,len(calendar));met=10+.05*np.arange(len(calendar))+rng.normal(0,1,len(calendar));x=[];y=[];target=[]
    for t in range(24,len(calendar)):
        sequence=[]
        for shift in range(11,-1,-1):
            row_t=t-shift;month=calendar[row_t].month
            sequence.append([np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12),np.sin(4*np.pi*month/12),np.cos(4*np.pi*month/12),signal[row_t-2],signal[row_t-3],signal[row_t-6],signal[row_t-12],climate[row_t-2],met[row_t-4]])
        x.append(sequence);y.append(signal[t]);target.append(str(calendar[t]))
    samples[c]={'x':np.asarray(x),'y':np.asarray(y),'target_month':np.asarray(target),'origin_month':np.asarray([str(pd.Period(t,'M')-1) for t in target]),'climate':np.asarray(x)[:,-1,-2:-1],'drivers':np.asarray(x)[:,:,-2:]}
groups=default_feature_groups(names,('fake_nino',));selected,selection=select_feature_group(samples,names,groups);groups['TRAIN_SELECTED']=selected
cfg=ControlConfig('Ridge','pooled','CORE');fit=fit_control(samples,names,groups['CORE'],cfg)
altered=copy.deepcopy(samples)
for d in altered.values():
    future=d['target_month']>'2007-11';d['x'][future]=1e9;d['y'][future]=1e9
other=fit_control(altered,names,groups['CORE'],cfg);check('fitting_invariant_to_all_future_x_and_y',np.array_equal(fit.models['__pooled__'][0].mean_,other.models['__pooled__'][0].mean_) and np.array_equal(fit.models['__pooled__'][1].coef_,other.models['__pooled__'][1].coef_))
_,selection2=select_feature_group(altered,names,{k:groups[k] for k in ('CORE','CLIMATE','FULL')});check('training_feature_selection_invariant_to_future_arrays',selection==selection2)
baselines=fit_baselines(samples,names);baselines2=fit_baselines(altered,names);check('seasonal_baseline_invariant_to_future_y',baselines.seasonal==baselines2.seasonal)
baseline_predictions=baselines.predict(samples,'2017-01','2024-12');month='2017-01';d=samples['SYNTH_A'];i=np.flatnonzero(d['target_month']==month)[0];r=baseline_predictions.set_index(['client','target_month','model']);check('availability_baselines_use_exact_declared_latest_and_lastyear_features',r.loc[('SYNTH_A',month,'available_rainfall_target_minus2'),'prediction']==d['x'][i,-1,names.index('rain_origin_lag1')] and r.loc[('SYNTH_A',month,'same_calendar_target_minus12'),'prediction']==d['x'][i,-1,names.index('rain_origin_lag11')])
thresholds=training_heavy_thresholds(samples);check('heavy_threshold_invariant_to_future_y',thresholds==training_heavy_thresholds(altered))
rejects('fit_end_cannot_admit_monitor_or_later',lambda:fit_control(samples,names,groups['CORE'],cfg,'2008-01'))

configs=[ControlConfig(family,scope,'CORE',seed=3,n_estimators=20,min_samples_leaf=2) for scope in ['pooled','local'] for family in ['Ridge','ET','RF']];experts=[cfg.label() for cfg in configs];validation=None;evaluation=[];fit_receipts=[]
for cfg in configs:
    fitted=fit_control(samples,names,groups['CORE'],cfg);pred=fitted.predict(samples,'2011-01','2016-11');part=pred[['client','target_month','y','prediction']].rename(columns={'prediction':cfg.label()});validation=part if validation is None else validation.merge(part,on=['client','target_month','y'],validate='one_to_one');evaluation.append(fitted.predict(samples,'2017-01','2024-12'));fit_receipts.append(fitted.receipt)
flat=fit_control(samples,names,groups['CORE'],ControlConfig('Ridge','pooled','CORE','flat_sequence'));check('flat_sensitivity_label_and_dimension_explicit','flat_sequence_sensitivity' in flat.config.label() and flat.receipt['input_dimensions']==12*len(groups['CORE']))
oof,oof_receipts=temporal_oof_controls(samples,names,groups,configs);check('temporal_OOF_every_row_strictly_earlier_fit',(oof.OOF_fit_end<oof.target_month).all())
_,nested=temporal_oof_controls(samples,names,groups,[ControlConfig('Ridge','pooled','TRAIN_SELECTED')]);check('OOF_train_selected_group_reselected_without_heldout_fold_labels',all(r['OOF_nested_feature_selection']['selection_end']==r['fit_end'] and all(f['score_end']<=r['fit_end'] for f in r['OOF_nested_feature_selection']['fold_results']) for r in nested))
routers,routing_score=fit_router_suite(validation,oof,experts);check('all_router_weights_finite_and_simplex',all(np.isfinite(w).all() and w.min()>=-1e-8 and abs(w.sum()-1)<1e-8 for byclient in routers.weights.values() for w in byclient.values()))
badval=pd.concat([validation,validation.iloc[:1].assign(target_month='2017-01')]);rejects('router_rejects_evaluation_rows',lambda:fit_router_suite(badval,oof,experts))
bad_oof=oof.copy();bad_oof.loc[0,'OOF_fit_end']=bad_oof.loc[0,'target_month'];rejects('stacking_rejects_in_sample_predictions',lambda:fit_router_suite(validation,bad_oof,experts))
unreleased=oof.copy();unreleased.loc[0,'OOF_fit_end']=str(pd.Period(unreleased.loc[0,'target_month'],'M')-1);rejects('OOF_previous_month_label_is_unreleased_at_first_day_issue',lambda:fit_router_suite(validation,unreleased,experts))
december=validation.copy();december=pd.concat([december,december.iloc[:1].assign(target_month='2016-12')]);rejects('router_December2016_label_is_unreleased_at_January2017_issue',lambda:fit_router_suite(december,oof,experts))
check('main_fit_released_label_boundary_is_November2007',fit.receipt['fit_months'][-1]=='2007-11' and fit.receipt['fit_n_total']==299*2)
check('fold_fit_end_is_exact_target_minus_two_months',all(pd.Period(r['fit_end'],'M')==pd.Period(r['OOF_score_start'],'M')-2 for r in oof_receipts))

frame=pd.concat(evaluation,ignore_index=True);metrics=evaluate_predictions(frame,thresholds,expected_months=pd.period_range('2017-01','2024-12',freq='M').astype(str),expected_seeds=[3]);check('synthetic_all_strong_controls_evaluate_common_keys',len(metrics['per_client'])==12 and len(metrics['aggregate_per_seed'])==12)
paired=[]
reference=frame.loc[frame.model==experts[0]]
for seed in range(5):
    part=reference.assign(seed=seed,model='SYNTH_IDENTICAL_A',prediction=reference.prediction+seed*.5);paired.extend([part,part.assign(model='SYNTH_IDENTICAL_B')])
paired=pd.concat(paired,ignore_index=True);boot=paired_block_bootstrap(paired,'SYNTH_IDENTICAL_A','SYNTH_IDENTICAL_B',repetitions=50);check('identity_bootstrap_all_draws_and_CI_exact_zero',(boot['draws'].filter(like='delta_').to_numpy()==0).all() and (boot['summary'][['delta_a_minus_b','CI95_low','CI95_high']].to_numpy()==0).all())
dates=circular_date_block_indices(96,12,np.random.default_rng(3));check('circular_blocks_preserve_consecutive_dates_within_each_block',all(np.array_equal((dates[j:j+12]-dates[j])%96,np.arange(12)) for j in range(0,96,12)))
rejects('paired_missing_one_client_date_fails',lambda:paired_block_bootstrap(paired.drop(paired.index[-1]),'SYNTH_IDENTICAL_A','SYNTH_IDENTICAL_B',repetitions=1))
wrongtruth=paired.copy();wrongtruth.loc[wrongtruth.model=='SYNTH_IDENTICAL_B','y']+=.1;rejects('paired_changed_truth_fails',lambda:paired_block_bootstrap(wrongtruth,'SYNTH_IDENTICAL_A','SYNTH_IDENTICAL_B',repetitions=1))
badprediction=paired.copy();badprediction.loc[0,'prediction']=np.nan;rejects('missing_prediction_cannot_fallback',lambda:validate_prediction_frame(badprediction))
sign=block_sign_randomization(paired,'SYNTH_IDENTICAL_A','SYNTH_IDENTICAL_B');check('identity_block_randomization_p_is_one',sign['two_sided_p']==1 and sign['delta_macro_R2_a_minus_b']==0)
dominance=paired.copy();dominance.loc[dominance.model=='SYNTH_IDENTICAL_A','prediction']=dominance.loc[dominance.model=='SYNTH_IDENTICAL_A','y'];dominance.loc[dominance.model=='SYNTH_IDENTICAL_B','prediction']=dominance.loc[dominance.model=='SYNTH_IDENTICAL_B','y']+1;exact=block_sign_randomization(dominance,'SYNTH_IDENTICAL_A','SYNTH_IDENTICAL_B');check('manual_eight_positive_block_sign_p_is_two_over_256',exact['n_blocks']==8 and exact['two_sided_p']==2/256)

receipt={'status':'PASS','data_kind':'SYNTHETIC_ONLY_NO_SCIENTIFIC_RESULTS','actual_climate_data_read':False,'actual_experiment_fits':0,'synthetic_fit_and_router_smoke':True,'checks':checks,'feature_selection_smoke':selection,'router_smoke':routers.receipt,'control_fit_smoke':fit_receipts,'OOF_smoke':oof_receipts,'block_bootstrap_smoke':boot['receipt'],'block_sign_smoke':sign}
(OUT/'verification_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');manual['aggregate_per_seed'].to_csv(OUT/'manual_macro_vs_pooled.csv',index=False);routing_score.to_csv(OUT/'synthetic_only_router_fold_score_predictions.csv',index=False);boot['summary'].to_csv(OUT/'synthetic_identity_bootstrap.csv',index=False)
print(json.dumps({'status':receipt['status'],'data_kind':receipt['data_kind'],'passed_checks':len(checks),'scientific_results':False},indent=2))
