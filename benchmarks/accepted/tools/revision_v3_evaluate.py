"""Open archived evaluation only after all fresh choices and fits are authenticated."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import revision_v3_runner as r
from caadapfl import neural_revision as n,evaluation as e,runner as accepted
RUN=r.RUN;DEST=RUN/'evaluation'
LEGACY_SOURCE=ROOT/'runs/accepted_reference/raw_predictions.csv'
MAPPING={'CAAA_GLOBAL_FULL_V2':'CAAA_GLOBAL_FULL','CAAA_PERSONAL_FULL_V2':'CAAA_PERSONAL_FULL','TRAIN_MONTHLY_CLIMATOLOGY':'training_seasonal_mean','POOLED_EXTRATREES_V2':'raw_pooled_ET_FULL_latest','STATIC_OOF_RIDGE_V2':'ROUTER_FULL_temporal_OOF_Ridge_stack','STATIC_OOF_RIDGE_CORE_V2':'ROUTER_CORE_temporal_OOF_Ridge_stack'}
def evaluate():
 assert r.load(RUN/'fit_complete.json')['status']=='COMPLETE'
 frozen=r.load(RUN/'choices_frozen.json');identity=r.load(RUN/'run_identity.json')
 assert accepted.runtime_environment()==identity['environment']
 assert frozen['run_identity_sha256']==r.sha(RUN/'run_identity.json')
 assert r.sha(RUN/'protocol.json')==identity['protocol_sha256'],'Protocol changed after training freeze'
 gate=r.load(RUN/'evaluation_prelaunch_identity.json')
 assert gate['archive_protocol_sha256']==r.sha(RUN/'archive_analysis_protocol.json')
 assert gate['archive_feature_protocol_sha256']==r.sha(RUN/'archive_feature_CI_protocol.json')
 assert gate['choices_sha256']==r.sha(RUN/'choices_frozen.json')
 for name,h in gate['evaluation_code'].items():assert r.sha(ROOT/name)==h,'Evaluation code changed after freeze: '+name
 assert r.load(RUN/'archive_analysis_protocol.json')['legacy_label_mapping']==MAPPING
 feature_protocol=r.load(RUN/'archive_feature_CI_protocol.json')
 combined_mapping={**MAPPING,**feature_protocol['additional_legacy_label_mapping']}
 for name,h in identity['code'].items():assert r.sha(ROOT/name)==h
 for name,h in identity['datasets'].items():assert r.sha(ROOT/name)==h
 for j in frozen['all_jobs']:
  p=RUN/'fits'/j['id'];assert r.sha(p/'complete.json')==frozen['fit_complete_hashes'][j['id']]
  receipt=r.load(p/'complete.json');assert receipt['job']==j
  for name,h in receipt['files'].items():assert r.sha(p/name)==h
 assert r.sha(LEGACY_SOURCE)==r.load(RUN/'archive_analysis_protocol.json')['legacy_raw_sha256']
 if DEST.exists():raise RuntimeError('Evaluation directory already exists; never overwrite a partial or accepted evaluation.')
 DEST.mkdir()
 truth=pd.read_csv(RUN/'datasets/SEALED_evaluation_truth.csv',float_precision='round_trip').set_index(['client','target_month'])['observed_mm']
 frames=[];cache={};inference_receipts=[]
 for j in frozen['all_jobs']:
  fit=n.load_fit(RUN/'fits'/j['id'],load_round_history=False);data,schema=accepted._load_dataset(RUN,j['scenario'],j['group'],evaluation=True);rows=[]
  for c,d in data.items():
   batch=n.FeatureBatch(d['scaled_x'],d['scaled_climate'],d['drivers'],tuple(k+'-01' for k in d['target_month']));prediction=fit.predict(c,batch,'global')
   for key,p,h in zip(prediction.keys,prediction.rainfall_mm,prediction.heavy_probability):
    month=key[:7];rows.append(dict(model=j['label'],seed=j['seed'],client=c,target_month=month,y=float(truth.loc[(c,month)]),prediction=float(p),heavy_probability=float(h),checkpoint_round=prediction.checkpoint_round,job_id=j['id'],origin='new_fit_v3'))
  part=pd.DataFrame(rows);frames.append(part);cache[j['id']]=part
  inference_receipts.append({'label':j['label'],'seed':j['seed'],'id':j['id'],'rows':len(part),'scenario':j['scenario'],'group':j['group']})
 for alias in frozen['aliases']:
  part=cache[alias['job']['id']].copy();part['model']=alias['label'];part['origin']='exact_matched_identity_alias';frames.append(part)
 frame=pd.concat(frames,ignore_index=True)
 for method in ('CAAA','FEDAVG','FEDPROX','CENTRAL'):
  part=frame[frame.model==method+'_RAW'].copy();assert len(part)==2400
  part['model']=method+'_CLIPPED_RAW';part['prediction']=np.maximum(part.prediction,0.);part['origin']='common_inference_projection_same_raw_checkpoint';frames.append(part)
  selected=method+'_'+frozen['selected_common_loss']['variant'];part=frame[frame.model==selected].copy();assert len(part)==2400
  part['model']=method+'_SELECTED_LOSS';part['origin']='validation_selected_common_loss_alias';frames.append(part)
 legacy=pd.read_csv(LEGACY_SOURCE,float_precision='round_trip')
 inverse={v:k for k,v in combined_mapping.items()};routers=r.load(RUN/'archive_analysis_protocol.json')['router_labels']
 keep=legacy.model.isin([*inverse,*routers]);old=legacy[keep].copy()
 assert set(old.model.unique())==set([*inverse,*routers]),set([*inverse,*routers])-set(old.model.unique())
 max_truth_discrepancy=0.
 for label,part in old.groupby('model',sort=True):
  canonical=np.array([truth.loc[(c,k)] for c,k in zip(part.client,part.target_month)])
  discrepancy=float(np.max(np.abs(part.y.to_numpy()-canonical)));max_truth_discrepancy=max(max_truth_discrepancy,discrepancy);assert discrepancy<1e-10
  part=part.copy();part['y']=canonical;part['model']=inverse.get(label,label);part['origin']='authenticated_accepted_v2_predictions';frames.append(part)
 frame=pd.concat(frames,ignore_index=True);frame=e.validate_prediction_frame(frame,accepted.EVAL_MONTHS,r.SEEDS)
 frame.to_csv(DEST/'raw_predictions.csv',index=False,float_format='%.17g')
 schema=r.load(RUN/'datasets/MAIN_FULL_schema.json');thresholds={c:x['heavy80_threshold_mm'] for c,x in schema['statistics']['clients'].items()}
 delay_labels=['DELAY2_REFERENCE','DELAY2_OBSERVATION']
 scores=e.evaluate_predictions(frame[~frame.model.isin(delay_labels)],thresholds,accepted.EVAL_MONTHS,r.SEEDS)
 delay_schema=r.load(RUN/'datasets/DELAY2_REFERENCE_FULL_schema.json');delay_thresholds={c:x['heavy80_threshold_mm'] for c,x in delay_schema['statistics']['clients'].items()}
 delayed_schema=r.load(RUN/'datasets/DELAY2_OBSERVATION_FULL_schema.json')
 assert delay_thresholds=={c:x['heavy80_threshold_mm'] for c,x in delayed_schema['statistics']['clients'].items()}
 delay_scores=e.evaluate_predictions(frame[frame.model.isin(delay_labels)],delay_thresholds,accepted.EVAL_MONTHS,r.SEEDS)
 for name in ('per_client','aggregate_per_seed','seed_summary'):scores[name]=pd.concat([scores[name],delay_scores[name]],ignore_index=True)
 scores['receipt']['auxiliary_event_thresholds']='MAIN299 local TRAINq80 except both DELAY2 arms: matched297 local TRAINq80. Common MAIN299q80 used only for separately labelled regression-tail bias diagnostics.'
 for name in ('per_client','aggregate_per_seed','seed_summary'):scores[name].to_csv(DEST/(name+'.csv'),index=False,float_format='%.17g')
 r.save(DEST/'PR_curves.json',scores['PR_curves']+delay_scores['PR_curves'])
 # Quantities that expose dry-month artifacts and heavy-rain bias, independently of auxiliary scores.
 tail=[]
 for (model,seed,client),g in frame.groupby(['model','seed','client'],sort=True):
  threshold=thresholds[client];a=g.y.to_numpy();p=g.prediction.to_numpy();mask=a>threshold;error=p-a;clim=frame[(frame.model=='TRAIN_MONTHLY_CLIMATOLOGY')&(frame.seed==seed)&(frame.client==client)].sort_values('target_month');g=g.sort_values('target_month')
  denominator=float(np.mean((clim.y-clim.prediction)**2));mse=float(np.mean((g.y-g.prediction)**2))
  tail.append(dict(model=model,seed=int(seed),client=client,n=len(a),unique_heavy_months=int(mask.sum()),negative_predictions=int((p<0).sum()),minimum_prediction_mm=float(p.min()),mean_bias_mm=float(error.mean()),tail80_bias_mm=float(error[mask].mean()),tail80_RMSE_mm=float(np.sqrt(np.mean(error[mask]**2))),tail80_underprediction_fraction=float(np.mean(p[mask]<a[mask])),nonheavy_RMSE_mm=float(np.sqrt(np.mean(error[~mask]**2))),MSESS_vs_climatology=1-mse/denominator))
 tail=pd.DataFrame(tail);tail.to_csv(DEST/'tail_bias_by_client_seed.csv',index=False,float_format='%.17g')
 ts=tail.groupby('model')[['tail80_bias_mm','tail80_RMSE_mm','tail80_underprediction_fraction','nonheavy_RMSE_mm','MSESS_vs_climatology']].mean().reset_index()
 ts=ts.rename(columns={'tail80_bias_mm':'tail80_bias_macro_mm','tail80_RMSE_mm':'tail80_RMSE_macro_mm','tail80_underprediction_fraction':'tail80_underprediction_macro_fraction','nonheavy_RMSE_mm':'nonheavy_RMSE_macro_mm'})
 negatives=tail.groupby('model')['negative_predictions'].sum().rename('negative_predictions_total_over_2400_seed_specific_rows');ts=ts.merge(negatives,on='model',validate='one_to_one')
 pooled=[]
 for model,g in frame.groupby('model'):
  mask=g.y.to_numpy()>np.array([thresholds[c] for c in g.client]);pooled.append({'model':model,'tail80_underprediction_pooled_seed_row_fraction':float(np.mean(g.prediction.to_numpy()[mask]<g.y.to_numpy()[mask])),'minimum_prediction_mm':float(g.prediction.min()),'unique_observed_heavy_client_months':int(g.loc[mask,['client','target_month']].drop_duplicates().shape[0])})
 ts=ts.merge(pd.DataFrame(pooled),on='model',validate='one_to_one')
 ts.to_csv(DEST/'tail_bias_summary.csv',index=False,float_format='%.17g')
 protocol=r.load(RUN/'protocol.json');family=protocol['uncertainty']['contrast_family'];assert len(family)==36
 summaries=[];tests=[];all_draws=[];receipts=[]
 for index,(a,b) in enumerate(family):
  label=f'C{index+1:02}'
  result=e.paired_block_bootstrap(frame,a,b,block_lengths=(12,6,24),repetitions=2000,bootstrap_seed=731)
  summary=result['summary'];summary.insert(0,'contrast_id',label);summaries.append(summary)
  draws=result['draws'];draws.insert(0,'contrast_id',label);all_draws.append(draws)
  test=e.block_sign_randomization(frame,a,b);test['contrast_id']=label;tests.append(test);receipts.append({'contrast_id':label,**result['receipt']})
  print('PAIRED',label,a,b,flush=True)
 adjusted=e.holm_adjust({x['contrast_id']:x['two_sided_p'] for x in tests})
 for t in tests:t['holm_p_36']=adjusted[t['contrast_id']];t['reject_at_05']=adjusted[t['contrast_id']]<.05
 pd.concat(summaries).to_csv(DEST/'paired_CI_36.csv',index=False,float_format='%.17g');pd.concat(all_draws).to_csv(DEST/'bootstrap_draws_36.csv',index=False,float_format='%.17g');pd.DataFrame(tests).to_csv(DEST/'annual_sign_Holm36.csv',index=False,float_format='%.17g')
 r.save(DEST/'paired_receipts_36.json',receipts)
 extra=[];extra_draws=[]
 archive_protocol=r.load(RUN/'archive_analysis_protocol.json')
 for i,(a,b) in enumerate(archive_protocol['additional_unadjusted_pairs']):
  result=e.paired_block_bootstrap(frame,a,b,block_lengths=(12,6,24),repetitions=2000,bootstrap_seed=731)
  label=f'A{i+1:02}';summary=result['summary'];summary.insert(0,'contrast_id',label);extra.append(summary)
  draws=result['draws'];draws.insert(0,'contrast_id',label);extra_draws.append(draws)
 pd.concat(extra).to_csv(DEST/'archived_additional_CI_11_unadjusted.csv',index=False,float_format='%.17g');pd.concat(extra_draws).to_csv(DEST/'archived_additional_draws_11.csv',index=False,float_format='%.17g')
 feature_summary=[];feature_draws=[]
 for i,(a,b) in enumerate(feature_protocol['pairs']):
  result=e.paired_block_bootstrap(frame,a,b,block_lengths=(12,6,24),repetitions=2000,bootstrap_seed=731)
  label=f'F{i+1:02}';summary=result['summary'];summary.insert(0,'contrast_id',label);feature_summary.append(summary)
  draws=result['draws'];draws.insert(0,'contrast_id',label);feature_draws.append(draws)
 pd.concat(feature_summary).to_csv(DEST/'archived_feature_CI_2_unadjusted.csv',index=False,float_format='%.17g');pd.concat(feature_draws).to_csv(DEST/'archived_feature_draws_2.csv',index=False,float_format='%.17g')
 r.save(DEST/'inference_receipts.json',inference_receipts)
 r.save(DEST/'metric_definitions.json',{'overall':scores['receipt'],'tail_bias_sign':'prediction minus observation, mm; negative means underprediction','tail_event':'Monthly total >local TRAINq80; not a daily extreme/flood event','negative_count':'Seed-specific predictions; repeated seeds do not multiply independent observed events','matched_delay_thresholds':'Auxiliary detection uses matched297 local TRAINq80 for both delay arms; separate rainfall-tail/bias diagnostics use common MAIN299 TRAINq80 only for cross-model comparability','new_and_legacy_truth_max_abs_difference_mm':max_truth_discrepancy,'archival_rows_truth_policy':'Verified difference below1e-10 then aligned to one canonical frozen source truth; predictions unchanged','new_family_36':'Exploratory frozen before evaluation; does not replace accepted31-comparison family','additional_archive_11':'Posthoc, unadjusted95% percentile CIs, no new confirmatory/significance claim','global_primary_retained':'The accepted v2 headline remains its own historical run; new validation-selected posthoc exploration is separately identified'})
 r.save(DEST/'complete.json',{'status':'COMPLETE','utc':r.now(),'choices_sha256':r.sha(RUN/'choices_frozen.json'),'evaluation_script_sha256':r.sha(__file__),'evaluation_helper_sha256':r.sha(ROOT/'caadapfl/evaluation.py'),'archive_protocol_sha256':r.sha(RUN/'archive_analysis_protocol.json'),'archive_feature_protocol_sha256':r.sha(RUN/'archive_feature_CI_protocol.json'),'evaluation_prelaunch_sha256':r.sha(RUN/'evaluation_prelaunch_identity.json'),'models':int(frame.model.nunique()),'rows':len(frame),'fresh_unique_fits':len(frozen['all_jobs']),'comparisons_v3':36,'archived_extra_CIs':11,'archived_feature_CIs':2,'bootstrap_draws':(36+11+2)*6000,'annual_sign_minimum_two_sided_p':2/256,'Holm36_minimum_first_adjusted_p':36*2/256,'independent_holdout_status':'UNVERIFIED; user uncertain about prior2025 inspection','files':{p.name:r.sha(p) for p in DEST.iterdir() if p.is_file()}})
 print('EVALUATION COMPLETE',frame.model.nunique(),len(frame),flush=True)
if __name__=='__main__':evaluate()
