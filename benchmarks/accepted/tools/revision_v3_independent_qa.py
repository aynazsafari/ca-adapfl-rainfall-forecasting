"""Independent CSV-based recomputation: no production scoring helpers imported."""
from pathlib import Path
import csv,json,hashlib,itertools,math,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/revision_v3';EVAL=RUN/'evaluation';OUT=RUN/'independent_qa'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8',newline='') as stream:yield from csv.DictReader(stream)
def close(saved,value,label):
 if np.isnan(value):assert saved=='' or np.isnan(float(saved)),label
 else:assert abs(float(saved)-value)<1e-10,(label,saved,value)
def regression(y,p):
 err=p-y;r2=1-(err@err)/np.sum((y-y.mean())**2);bias=-100*err.sum()/y.sum();corr=np.corrcoef(y,p)[0,1];alpha=p.std(ddof=1)/y.std(ddof=1);beta=p.mean()/y.mean()
 return {'R2':r2,'RMSE_mm':np.sqrt(np.mean(err**2)),'MAE_mm':np.mean(np.abs(err)),'PBIAS_signed_percent':bias,'PBIAS_absolute_percent':abs(bias),'forecast_bias_signed_percent':-bias,'Pearson_r':corr,'KGE_2009':1-np.sqrt((corr-1)**2+(alpha-1)**2+(beta-1)**2),'NSE':r2}
def event_metrics(event,score,predicted):
 tp=int(np.sum(event&predicted));fp=int(np.sum(~event&predicted));fn=int(np.sum(event&~predicted));tn=int(np.sum(~event&~predicted))
 if not event.any():return {'actual_heavy_n':0,'heavy_TP':tp,'heavy_FP':fp,'heavy_FN':fn,'heavy_TN':tn,'heavy_precision':tp/(tp+fp) if tp+fp else np.nan,'heavy_recall':np.nan,'heavy_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,'heavy_average_precision':np.nan,'heavy_ROC_AUC':np.nan},np.array([]),np.array([]),np.array([])
 cuts=np.unique(score);precision=[];recall=[]
 for cutoff in cuts:
  flag=score>=cutoff;hits=np.sum(flag&event);precision.append(hits/flag.sum());recall.append(hits/event.sum())
 precision=np.r_[precision,1.];recall=np.r_[recall,0.];ap=-np.sum(np.diff(recall)*precision[:-1]);pos=score[event];neg=score[~event];auc=np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)) if len(neg) else np.nan
 return {'actual_heavy_n':int(event.sum()),'heavy_TP':tp,'heavy_FP':fp,'heavy_FN':fn,'heavy_TN':tn,'heavy_precision':tp/(tp+fp) if tp+fp else np.nan,'heavy_recall':tp/(tp+fn),'heavy_F1':2*tp/(2*tp+fp+fn),'heavy_average_precision':ap,'heavy_ROC_AUC':auc},precision,recall,cuts
def stats(y,p):
 errors=(p-y)**2;center=y-np.mean(y,axis=2,keepdims=True)
 r2=1-np.sum(errors,axis=2)/np.sum(center**2,axis=2)
 rmse=np.sqrt(np.mean(errors,axis=2))
 return np.array([np.mean(r2),np.mean(rmse)])
def main(destination=None):
 out=Path(destination).resolve() if destination else OUT
 if out.exists():raise RuntimeError('Independent verification never replaces an existing receipt: '+str(out))
 complete=load(EVAL/'complete.json');assert complete['status']=='COMPLETE'
 for name,h in complete['files'].items():assert sha(EVAL/name)==h
 source=list(rows(EVAL/'raw_predictions.csv'));models=sorted(set(x['model'] for x in source));seeds=sorted(set(int(x['seed']) for x in source));clients=sorted(set(x['client'] for x in source));months=sorted(set(x['target_month'] for x in source))
 expected_clients=['Client1_NW_Arid','Client2_Indo_Gangetic','Client3_NE_Himalaya','Client4_South_Peninsular','Client5_Central_India']
 assert seeds==[7,42,123,999,2024] and clients==expected_clients and len(clients)==5
 assert months==[f'{year}-{month:02}' for year in range(2017,2025) for month in range(1,13)]
 shape=(len(seeds),len(clients),len(months));prediction={m:np.full(shape,np.nan) for m in models};observed=np.full(shape,np.nan);prob={m:np.full(shape,np.nan) for m in models};seen=set()
 for x in source:
  key=(x['model'],int(x['seed']),x['client'],x['target_month']);assert key not in seen;seen.add(key)
  i,j,k=seeds.index(int(x['seed'])),clients.index(x['client']),months.index(x['target_month']);value=float(x['y'])
  if np.isfinite(observed[i,j,k]):assert observed[i,j,k]==value
  observed[i,j,k]=value;prediction[x['model']][i,j,k]=float(x['prediction'])
  if x.get('heavy_probability'):prob[x['model']][i,j,k]=float(x['heavy_probability'])
 assert len(source)==len(models)*2400 and all(np.isfinite(p).all() for p in prediction.values())
 assert np.array_equal(observed,np.broadcast_to(observed[0:1],observed.shape))
 summaries={x['model']:x for x in rows(EVAL/'seed_summary.csv')};max_point_error=0.
 for m,p in prediction.items():
  per_seed=np.array([stats(observed[i:i+1],p[i:i+1]) for i in range(5)])
  for j,name in enumerate(['R2','RMSE_mm']):
   for suffix,value in [('mean',per_seed[:,j].mean()),('std',per_seed[:,j].std(ddof=1))]:
    discrepancy=abs(float(summaries[m][name+'_'+suffix])-value);max_point_error=max(max_point_error,discrepancy);assert discrepancy<1e-10,(m,name,suffix,discrepancy)
  if '_NONNEGATIVE' in m:assert (p>=0).all(),m
 # Replicated aliases/projections are checked explicitly; they are not independent fits.
 frozen=load(RUN/'choices_frozen.json')
 assert len(frozen['all_jobs'])==100 and len(set(j['id'] for j in frozen['all_jobs']))==100
 assert complete['choices_sha256']==sha(RUN/'choices_frozen.json') and complete['evaluation_prelaunch_sha256']==sha(RUN/'evaluation_prelaunch_identity.json')
 assert complete['archive_protocol_sha256']==sha(RUN/'archive_analysis_protocol.json') and complete['archive_feature_protocol_sha256']==sha(RUN/'archive_feature_CI_protocol.json')
 for job in frozen['all_jobs']:
  folder=RUN/'fits'/job['id'];receipt=load(folder/'complete.json');assert receipt['job']==job and sha(folder/'complete.json')==frozen['fit_complete_hashes'][job['id']]
  for name,h in receipt['files'].items():assert sha(folder/name)==h
 selected_arch=frozen['development_choice']['selected']['label']
 assert np.array_equal(prediction['CAAA_RAW'],prediction[selected_arch])
 assert np.array_equal(prediction['FEDPROX_RAW'],prediction['ADAPTIVE_OFF'])
 assert np.array_equal(prob['CAAA_RAW'],prob[selected_arch]) and np.array_equal(prob['FEDPROX_RAW'],prob['ADAPTIVE_OFF'])
 for method in ['CAAA','FEDAVG','FEDPROX','CENTRAL']:
  assert np.array_equal(prediction[method+'_CLIPPED_RAW'],np.maximum(0,prediction[method+'_RAW']))
  assert np.array_equal(prediction[method+'_SELECTED_LOSS'],prediction[method+'_'+frozen['selected_common_loss']['variant']])
  assert np.array_equal(prob[method+'_CLIPPED_RAW'],prob[method+'_RAW']) and np.array_equal(prob[method+'_SELECTED_LOSS'],prob[method+'_'+frozen['selected_common_loss']['variant']])
 protocols=[(load(RUN/'protocol.json')['uncertainty']['contrast_family'],'C','bootstrap_draws_36.csv','paired_CI_36.csv'),(load(RUN/'archive_analysis_protocol.json')['additional_unadjusted_pairs'],'A','archived_additional_draws_11.csv','archived_additional_CI_11_unadjusted.csv'),(load(RUN/'archive_feature_CI_protocol.json')['pairs'],'F','archived_feature_draws_2.csv','archived_feature_CI_2_unadjusted.csv')]
 maximum_draw_discrepancy=0.;count=0;sign_results=[]
 for pairs,prefix,drawfile,summaryfile in protocols:
  summary_rows=list(rows(EVAL/summaryfile));assert len(summary_rows)==len(pairs)*6
  saved_summaries={(x['contrast_id'],int(x['block_length_months']),x['metric']):x for x in summary_rows};assert len(saved_summaries)==len(summary_rows)
  grouped=iter(itertools.groupby(rows(EVAL/drawfile),key=lambda x:x['contrast_id']))
  for index,(a,b) in enumerate(pairs):
   cid=f'{prefix}{index+1:02}';key,stream=next(grouped);assert key==cid;saved=list(stream);assert len(saved)==6000
   pa,pb=prediction[a],prediction[b];point=stats(observed,pa)-stats(observed,pb);rng=np.random.default_rng(731);position=0
   for length in [12,6,24]:
    draws=[]
    for rep in range(2000):
     start=rng.integers(0,96,size=math.ceil(96/length));dates=np.concatenate([(int(s)+np.arange(length))%96 for s in start])[:96];sampled=rng.integers(0,5,size=5)
     delta=stats(observed[sampled][:,:,dates],pa[sampled][:,:,dates])-stats(observed[sampled][:,:,dates],pb[sampled][:,:,dates]);draws.append(delta)
     row=saved[position];position+=1;assert int(row['block_length_months'])==length and int(row['replicate'])==rep
     saved_delta=np.array([float(row['delta_macro_R2_a_minus_b']),float(row['delta_macro_RMSE_mm_a_minus_b'])]);d=float(np.max(np.abs(delta-saved_delta)));maximum_draw_discrepancy=max(maximum_draw_discrepancy,d);assert d<1e-10,(cid,length,rep,d);count+=1
    lo,hi=np.quantile(np.array(draws),[.025,.975],axis=0)
    for j,metric in enumerate(['macro_R2','macro_RMSE_mm']):
     row=saved_summaries[(cid,length,metric)]
     assert row['model_a']==a and row['model_b']==b
     assert max(abs(float(row['delta_a_minus_b'])-point[j]),abs(float(row['CI95_low'])-lo[j]),abs(float(row['CI95_high'])-hi[j]))<1e-10
   if prefix=='C':
    tss=np.sum((observed-observed.mean(axis=2,keepdims=True))**2,axis=2);contrib=np.mean(((observed-pb)**2-(observed-pa)**2)/tss[:,:,None],axis=(0,1));blocks=contrib.reshape(8,12).sum(axis=1)
    null=np.array([sum(s*v for s,v in zip(signs,blocks)) for signs in itertools.product([-1,1],repeat=8)]);p=float(np.mean(np.abs(null)>=abs(float(blocks.sum()))-1e-14));sign_results.append((cid,p))
   print('INDEPENDENT',cid,flush=True)
  try:next(grouped)
  except StopIteration:pass
  else:raise AssertionError('Unexpected extra bootstrap contrast')
 saved_sign={x['contrast_id']:x for x in rows(EVAL/'annual_sign_Holm36.csv')};ordered=sorted(sign_results,key=lambda x:x[1]);last=0.
 for i,(cid,p) in enumerate(ordered):
  last=max(last,min(1.,(36-i)*p));assert abs(float(saved_sign[cid]['two_sided_p'])-p)<1e-14;assert abs(float(saved_sign[cid]['holm_p_36'])-last)<1e-14
  pair=load(RUN/'protocol.json')['uncertainty']['contrast_family'][int(cid[1:])-1];assert [saved_sign[cid]['model_a'],saved_sign[cid]['model_b']]==pair and saved_sign[cid]['reject_at_05']==str(last<.05)
 assert len(saved_sign)==36
 # TRAIN thresholds are verified independently from fit-only raw training labels.
 schema=load(RUN/'datasets/MAIN_FULL_schema.json');thresholds={c:schema['statistics']['clients'][c]['heavy80_threshold_mm'] for c in clients}
 with np.load(RUN/'datasets/MAIN_FULL_fit.npz',allow_pickle=False) as a:
  for c in clients:
   target=a[c+'::raw_target_month'];y=a[c+'::y'];assert abs(float(np.quantile(y[target<='2007-11'],.8))-thresholds[c])<1e-12
 delay_thresholds={}
 for scenario in ['DELAY2_REFERENCE','DELAY2_OBSERVATION']:
  ds=load(RUN/f'datasets/{scenario}_FULL_schema.json');values={c:ds['statistics']['clients'][c]['heavy80_threshold_mm'] for c in clients}
  with np.load(RUN/f'datasets/{scenario}_FULL_fit.npz',allow_pickle=False) as a:
   for c in clients:assert abs(np.quantile(a[c+'::y'][a[c+'::raw_target_month']<='2007-11'],.8)-values[c])<1e-12
  delay_thresholds[scenario]=values
 assert delay_thresholds['DELAY2_REFERENCE']==delay_thresholds['DELAY2_OBSERVATION']
 client_scores={(x['model'],int(x['seed']),x['client']):x for x in rows(EVAL/'per_client.csv')};aggregates={(x['model'],int(x['seed']),x['scope']):x for x in rows(EVAL/'aggregate_per_seed.csv')};pr_saved={(x['model'],int(x['seed']),x['client']):x for x in load(EVAL/'PR_curves.json')}
 for m,p in prediction.items():
  macro_records=[]
  for i,s in enumerate(seeds):
   regressions=[]
   for j,c in enumerate(clients):
    y=observed[i,j];pr=p[i,j];row=client_scores[(m,s,c)];reg=regression(y,pr);regressions.append(reg);assert int(row['n'])==96
    for key,value in reg.items():close(row[key],value,(m,s,c,key))
    q=delay_thresholds[m][c] if m in delay_thresholds else thresholds[c];event=y>q;h=prob[m][i,j]
    if np.isnan(h).all():score=pr;flag=pr>q
    else:assert np.isfinite(h).all() and ((h>=0)&(h<=1)).all();score=h;flag=h>=.5
    metrics,precision,recall,cut=event_metrics(event,score,flag);metrics.update({'heavy80_training_threshold_mm':q,'tail80_RMSE_mm':np.sqrt(np.mean((pr[event]-y[event])**2)),'tail80_MAE_mm':np.mean(np.abs(pr[event]-y[event]))})
    for key,value in metrics.items():close(row[key],value,(m,s,c,key))
    saved=pr_saved[(m,s,c)]
    for key,value in [('precision',precision),('recall',recall),('score_threshold',cut)]:assert np.allclose(saved[key],value,rtol=0,atol=1e-12),(m,s,c,key)
   macro=aggregates[(m,s,'macro_equal_client')];pooled=aggregates[(m,s,'pooled_all_client_months')]
   reconstructed={key:np.mean([x[key] for x in regressions]) for key in regressions[0]};macro_records.append(reconstructed)
   assert int(macro['n'])==480 and int(pooled['n'])==480 and int(macro['n_clients'])==5
   for key in regressions[0]:close(macro[key],reconstructed[key],(m,s,key));close(pooled[key],regression(observed[i].ravel(),p[i].ravel())[key],(m,s,'pooled',key))
  for key in macro_records[0]:
   values=np.array([x[key] for x in macro_records]);values=values[np.isfinite(values)]
   quantities=[('mean',values.mean() if len(values) else np.nan),('std',values.std(ddof=1) if len(values)>1 else np.nan),('min',values.min() if len(values) else np.nan),('max',values.max() if len(values) else np.nan)]
   for suffix,value in quantities:close(summaries[m][key+'_'+suffix],value,(m,key,suffix))
 saved_tail={(x['model'],int(x['seed']),x['client']):x for x in rows(EVAL/'tail_bias_by_client_seed.csv')}
 for m,p in prediction.items():
  for i,s in enumerate(seeds):
   for j,c in enumerate(clients):
    y=observed[i,j];pr=p[i,j];mask=y>thresholds[c];err=pr-y;row=saved_tail[(m,s,c)]
    assert int(row['negative_predictions'])==int((pr<0).sum()) and int(row['unique_heavy_months'])==int(mask.sum())
    expected={'mean_bias_mm':err.mean(),'tail80_bias_mm':err[mask].mean(),'tail80_RMSE_mm':np.sqrt(np.mean(err[mask]**2)),'tail80_underprediction_fraction':np.mean(pr[mask]<y[mask]),'nonheavy_RMSE_mm':np.sqrt(np.mean(err[~mask]**2)),'MSESS_vs_climatology':1-np.mean(err**2)/np.mean((prediction['TRAIN_MONTHLY_CLIMATOLOGY'][i,j]-y)**2)}
    for key,value in expected.items():assert abs(float(row[key])-value)<1e-10,(m,s,c,key)
 saved_summary={x['model']:x for x in rows(EVAL/'tail_bias_summary.csv')}
 for m,p in prediction.items():
  records=[saved_tail[(m,s,c)] for s in seeds for c in clients];row=saved_summary[m]
  mapping={'tail80_bias_macro_mm':'tail80_bias_mm','tail80_RMSE_macro_mm':'tail80_RMSE_mm','tail80_underprediction_macro_fraction':'tail80_underprediction_fraction','nonheavy_RMSE_macro_mm':'nonheavy_RMSE_mm','MSESS_vs_climatology':'MSESS_vs_climatology'}
  for output,source_col in mapping.items():close(row[output],np.mean([float(x[source_col]) for x in records]),(m,output))
  event=observed>np.array([thresholds[c] for c in clients])[None,:,None]
  close(row['tail80_underprediction_pooled_seed_row_fraction'],np.mean(p[event]<observed[event]),(m,'pooled_tail'))
  assert int(row['unique_observed_heavy_client_months'])==int(event[0].sum()) and int(row['negative_predictions_total_over_2400_seed_specific_rows'])==int((p<0).sum())
  close(row['minimum_prediction_mm'],p.min(),(m,'minimum'))
 out.mkdir(parents=True,exist_ok=False)
 receipt={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':len(models),'raw_rows':len(source),'unique_fresh_fits':len(frozen['all_jobs']),'independently_recomputed_bootstrap_draws':count,'independently_recomputed_annual_sign_contrasts':len(sign_results),'maximum_point_discrepancy':max_point_error,'maximum_bootstrap_draw_discrepancy':maximum_draw_discrepancy,'production_scoring_helpers_imported':False,'raw_predictions_sha256':sha(EVAL/'raw_predictions.csv'),'evaluation_complete_sha256':sha(EVAL/'complete.json'),'independent_qa_script_sha256':sha(__file__),'acceptance_scope':'Numeric reproducibility on fixed historical cohort; not independentfuturetest, equivalence, privacy/security or institutionaldeployment validation.'}
 (out/'complete.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,help='Fresh directory for a new verification receipt; accepted evidence is never replaced.');args=parser.parse_args();main(args.output)
