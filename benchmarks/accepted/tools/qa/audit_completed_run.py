"""Independent completed-result QA. Refuses all result reads before completion.

Standalone audit formulas live beside this file; no caadapfl modules are
imported and no model is constructed, fit, loaded, or used for inference.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,math,statistics,sys,time
import numpy as np
import pandas as pd
from metric_reference import regression_reference,heavy_reference
from uncertainty_reference import bootstrap_reference,sign_reference,holm_reference
from fit_cost_reference import neural_cost,other_cost

SEEDS=sorted([42,123,2024,7,999])
CLIENTS=sorted(['Client1_NW_Arid','Client2_Indo_Gangetic','Client3_NE_Himalaya','Client4_South_Peninsular','Client5_Central_India'])
MONTHS=list(pd.period_range('2017-01','2024-12',freq='M').astype(str))
REG=['R2','RMSE_mm','MAE_mm','PBIAS_signed_percent','PBIAS_absolute_percent','forecast_bias_signed_percent','KGE_2009','Pearson_r','NSE']
def require(c,m):
    if not c:raise AssertionError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pathkey(k):return Path(str(k).replace('\\','/'))
def scalar_close(a,b,where,atol=1e-9):
    a=float(a);b=float(b)
    if math.isnan(a) and math.isnan(b):return 0.
    require(math.isfinite(a) and math.isfinite(b),where+': undefined status differs')
    delta=abs(a-b);require(math.isclose(a,b,rel_tol=1e-10,abs_tol=atol),where+f': numerical difference {delta}')
    return delta
def exported_table_equal(actual,expected,where):
    """Compare keyed CSV values without treating inferred int/float as drift."""
    require(actual.index.equals(expected.index) and actual.columns.equals(expected.columns),where+': key or column schema differs')
    for name in actual.columns:
        a,b=actual[name],expected[name]
        numeric_a=pd.api.types.is_numeric_dtype(a);numeric_b=pd.api.types.is_numeric_dtype(b)
        require(numeric_a==numeric_b,where+': numeric/string schema differs for '+name)
        if numeric_a:
            for key,x,y in zip(actual.index,a,b):scalar_close(x,y,where+'/'+str(key)+'/'+name)
        else:
            require(a.fillna('<MISSING>').equals(b.fillna('<MISSING>')),where+': values differ for '+name)
def frame(p):return pd.read_csv(p,float_precision='round_trip',dtype={'target_month':str})
def hash_tree(base,files):
    for k,h in files.items():require(sha(base/pathkey(k))==h,'Changed file: '+str(base/pathkey(k)))
def keyed_sha(records):
    h=hashlib.sha256()
    for row in records:h.update(('\t'.join(str(v) for v in row)+'\n').encode('utf-8'))
    return h.hexdigest()
def independent_heavy_curve(observed,scores):
    labels=np.asarray(observed,bool);scores=np.asarray(scores,float);positive=int(labels.sum())
    if not positive:return {'precision':[],'recall':[],'threshold':[],'AP':math.nan,'AUC':math.nan}
    thresholds=np.unique(scores);precision=[];recall=[]
    for cut in thresholds:
        selected=scores>=cut;tp=int((labels&selected).sum())
        precision.append(tp/int(selected.sum()));recall.append(tp/positive)
    precision.append(1.);recall.append(0.)
    ap=math.fsum((recall[i]-recall[i+1])*precision[i] for i in range(len(thresholds)))
    a=scores[labels];b=scores[~labels]
    auc=float(((a[:,None]>b[None,:]).sum()+.5*(a[:,None]==b[None,:]).sum())/(len(a)*len(b))) if len(b) else math.nan
    return {'precision':precision,'recall':recall,'threshold':thresholds.tolist(),'AP':ap,'AUC':auc}
def linear_quantile(values,q):
    ordered=sorted(float(v) for v in values);pos=(len(ordered)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)

def gates(run,report):
    evaluation=run/'evaluation'
    # This check precedes opening any scientific run inputs or result content.
    if not (evaluation/'complete.json').is_file():raise RuntimeError('Evaluation completion gate is absent. No result/truth files were opened.')
    completed=read(evaluation/'complete.json');identity=read(run/'run_identity.json');frozen=read(run/'choices_frozen.json');protocol=read(run/'protocol.json')
    expected=dict(identity);expected.pop('run_sha256')
    require(digest(expected)==identity['run_sha256']==completed['run_sha256']==frozen['run_sha256'],'Scientific run identity differs')
    require(completed['choices_frozen_sha256']==sha(run/'choices_frozen.json'),'Frozen choices hash differs')
    require(completed['evaluation_truth_opened_only_after_all_choices'] is True and frozen['evaluation_labels_accessed'] is False and frozen['choices_including_checkpoints_and_router_weights_frozen'] is True,'Evaluation sealing receipts differ')
    require(digest(protocol)==identity['protocol_hash'] and digest(read(run/'schedule.json'))==identity['schedule_hash'],'Prespecified protocol/schedule changed')
    require(protocol['seeds']==[42,123,2024,7,999] and len(protocol['comparisons'])==31,'Five seeds and31 frozen comparisons required')
    hash_tree(evaluation,completed['files']);prepared=read(run/'prepared.json');hash_tree(run,prepared['files'])
    require(prepared['model_fits']==prepared['evaluation_predictions']==0 and prepared['run_sha256']==identity['run_sha256'],'Prepare stage fitted/scored or belongs to another run')
    package=run.parent.parent
    for module,h in identity['code'].items():require(sha(package/'caadapfl'/(module+'.py'))==h,'Frozen scientific source differs: '+module)
    for name,h in [('monthly',identity['source']['monthly_sha256']),('indices',identity['source']['indices_sha256']),('manifest',identity['source']['manifest_sha256'])]:require(sha(package/'data'/(name+'.json' if name=='manifest' else name+'.csv'))==h,'Frozen canonical source differs')
    require(set(frozen['fit_complete_hashes'])=={r['job_sha256'] for r in frozen['all_fit_jobs']},'Unique fit identity map differs')
    hardware=read(evaluation/'hardware.json');require(hardware['max_worker_processes']==2 and hardware['threads_per_worker']==2,'Actual hardware execution cap differs')
    for label,distribution in [('numpy','numpy'),('pandas','pandas'),('sklearn','scikit-learn'),('torch','torch'),('scipy','scipy')]:require(hardware['package_versions'][label]==identity['runtime_environment']['packages'][distribution]['loaded_version'],'Hardware runtime signature differs')
    require(protocol['bootstrap']['blocks']==[12,6,24] and protocol['bootstrap']['repetitions']==2000 and protocol['bootstrap']['seed']==731,'Uncertainty protocol differs')
    report['identity']={'run_sha256':identity['run_sha256'],'choices_frozen_sha256':sha(run/'choices_frozen.json'),'evaluation_completion_sha256':sha(evaluation/'complete.json'),'authenticated_evaluation_files':len(completed['files']),'authenticated_prepared_files':len(prepared['files']),'scientific_modules':identity['code'],'runtime_environment':identity['runtime_environment'],'prespecified_comparisons':31}
    report.update(run_sha256=identity['run_sha256'],evaluation_complete_sha256=sha(evaluation/'complete.json'),checked_files={'evaluation':completed['files'],'prepared':prepared['files'],'fit_completion_receipts':frozen['fit_complete_hashes'],'scientific_module_files':identity['code']})
    report['formulas']={'R2_NSE':'1-sum((prediction-observed)^2)/sum((observed-mean(observed))^2)','RMSE':'sqrt(mean((prediction-observed)^2))','MAE':'mean(abs(prediction-observed))','PBIAS':'100*sum(observed-prediction)/sum(observed); positive underprediction','KGE2009':'1-sqrt((Pearson_r-1)^2+(sd(pred)/sd(obs)-1)^2+(mean(pred)/mean(obs)-1)^2)','macro':'Equal-client average of metrics, followed by mean of five paired seeds','heavy':'Observed strictly > training q80; head probability >=.5 or no-head predicted rainfall > q80','uncertainty':'Common circular date blocks and paired seed resampling; PCG64 seed731,2000 draws each12/6/24 months; exact eight-block sign test and31-comparison Holm family when requested'}
    report['scope']={'evaluation_period':['2017-01','2024-12'],'clients':CLIENTS,'seeds':SEEDS,'model_source':'Exactly frozen outputs plus three training-only baselines','predictions':'Raw values with negative forecasts retained; no clipping or denominator floors','source_and_code':'Scientific run hash, canonical-source hashes, frozen runtime signature and active module file hashes authenticated','statistical_scope':'Five fixed regions and final revised archive cohort; previously inspected retrospective evaluation'}
    report['limitations']=['This recalculation checks saved results and receipts; it does not independently recreate optimizer trajectories or rerun models.','Tree/Ridge estimator-operation counts follow admitted pooled/local recipe receipts, while neural update counts are actual telemetry.','Communication bytes are simulated numeric payload sizes and declared setup aliases, not network measurements or a privacy guarantee.','The paired block analysis is conditional on these five regions and paired seeds; block sign tests assume sign symmetry/exchangeability.','No untouched prospective-test or historical operational-vintage claim is established.']
    return evaluation,completed,identity,frozen,protocol

def prediction_keys(evaluation,completed,frozen,report):
    predictions=frame(evaluation/'raw_predictions.csv')
    key=['model','seed','client','target_month'];require(not predictions.duplicated(key).any(),'Duplicate prediction keys')
    require(np.isfinite(predictions[['y','prediction']].to_numpy()).all() and (predictions.y>=0).all(),'Nonfinite predictions or invalid observations')
    require(len(predictions)==completed['raw_predictions'],'Raw prediction receipt row count differs')
    models=sorted(predictions.model.unique());require(len(models)==completed['models'],'Model count differs')
    exported={o['label'] for o in frozen['outputs']}|{'training_seasonal_mean','available_rainfall_target_minus2','same_calendar_target_minus12'}
    require(set(models)==exported,'Missing/extra evaluated method labels')
    expected=pd.MultiIndex.from_product([SEEDS,CLIENTS,MONTHS],names=['seed','client','target_month']);truth=None
    for model in models:
        part=predictions[predictions.model==model].set_index(['seed','client','target_month']).sort_index()
        require(part.index.equals(expected),'Incomplete96dates/5clients/5seeds: '+model)
        values=part.y.to_numpy()
        if truth is None:truth=values
        else:require(np.array_equal(values,truth),'Compared methods have different keyed truths')
    for output in frozen['outputs']:
        group=predictions[(predictions.model==output['label'])&(predictions.seed==output['seed'])]
        require(len(group)==480,'Evaluated output cohort differs')
        if output['kind']=='neural':
            path=evaluation.parent/'fits'/output['job_sha256'][:16];meta=read(path/'model/manifest.json');cfg=meta['config']
            require((group.checkpoint_round==meta['selected_rounds'][output['identity']]).all(),'Evaluation checkpoint round differs from frozen selection')
            if cfg['use_aux']:require(group.heavy_probability.notna().all(),'Auxiliary head probabilities absent')
            else:require(group.heavy_probability.isna().all(),'No-head method fabricates probability')
    arrays={model:predictions[predictions.model==model].set_index(['seed','client','target_month']).reindex(expected).prediction.to_numpy().reshape(5,5,96) for model in models}
    truth=truth.reshape(5,5,96);require(np.array_equal(truth,np.broadcast_to(truth[0:1],truth.shape)),'Observed labels differ across seeds')
    sealed=frame(evaluation.parent/'datasets/evaluation_truth.csv').set_index(['client','target_month']).sort_index()
    truthkeys=pd.MultiIndex.from_product([CLIENTS,MONTHS],names=['client','target_month']);require(sealed.index.equals(truthkeys),'Sealed truth calendar differs')
    sealed_delta=float(np.max(np.abs(sealed.y.to_numpy()-truth[0].reshape(-1))));require(sealed_delta<1e-10,'Prediction observations differ from sealed truth')
    actual_key_hash=keyed_sha((r.model,int(r.seed),r.client,r.target_month) for r in predictions.sort_values(['model','seed','client','target_month']).itertuples())
    expected_key_hash=keyed_sha((m,s,c,t) for m in models for s in SEEDS for c in CLIENTS for t in MONTHS);require(actual_key_hash==expected_key_hash,'Canonical key-calendar hash differs')
    truth_hash=keyed_sha((c,t,float(truth[0,i,j]).hex()) for i,c in enumerate(CLIENTS) for j,t in enumerate(MONTHS))
    # Alias labels reuse a source fit, identical feature representation and identity.
    aliases=defaultdict(list)
    for output in frozen['outputs']:aliases[(output['job_sha256'],output.get('identity'),output.get('suite'),output['seed'])].append(output['label'])
    aliaschecks=0
    for labels in aliases.values():
        if len(labels)>1:
            first=predictions[(predictions.model==labels[0])].sort_values(['seed','client','target_month'])
            for label in labels[1:]:
                other=predictions[predictions.model==label].sort_values(['seed','client','target_month'])
                require(np.array_equal(first.prediction.to_numpy(),other.prediction.to_numpy()),'Declared identical-fit alias predictions differ');aliaschecks+=1
    require(read(evaluation/'model_identities.json')==frozen['outputs'],'Exported model identities differ from frozen outputs')
    report['prediction_keys']={'models':len(models),'raw_rows':len(predictions),'seeds':SEEDS,'clients':CLIENTS,'months':96,'exact_truth_and_key_pairing':True,'actual_key_calendar_sha256':actual_key_hash,'expected_key_calendar_sha256':expected_key_hash,'shared_observation_calendar_sha256_floathex':truth_hash,'sealed_truth_file_sha256':sha(evaluation.parent/'datasets/evaluation_truth.csv'),'maximum_sealed_truth_decimal_parse_difference_mm':sealed_delta,'identical_fit_alias_checks':aliaschecks,'negative_forecasts_retained':int((predictions.prediction<0).sum())}
    return predictions,truth,arrays

def thresholds(run):
    output={}
    for scenario in ['MAIN','LATENCY_REFERENCE']:
        schema=read(run/'datasets'/(scenario+'_FULL_schema.json'));values={}
        with np.load(run/'datasets'/(scenario+'_FULL_fit.npz'),allow_pickle=False) as data:
            for c in CLIENTS:
                dates=data[c+'::raw_target_month'];train=dates<='2007-11'
                require(int(train.sum())==(299 if scenario=='MAIN' else 298),'Heavy threshold training cohort')
                value=linear_quantile(data[c+'::y'][train],.8)
                scalar_close(value,schema['statistics']['clients'][c]['heavy80_threshold_mm'],'Training-only q80',atol=1e-10);values[c]=value
        output[scenario]=values
    return output

def metrics(run,evaluation,predictions,report):
    heavy=thresholds(run);saved=frame(evaluation/'per_client.csv').set_index(['model','seed','client']);aggregate=frame(evaluation/'aggregate_per_seed.csv').set_index(['model','seed','scope']);summary=frame(evaluation/'seed_summary.csv').set_index('model')
    curves=read(evaluation/'PR_curves.json');curve={(r['model'],int(r['seed']),r['client']):r for r in curves};negative=frame(evaluation/'negative_forecast_diagnostics.csv').set_index(['model','seed','client'])
    require(saved.index.is_unique and aggregate.index.is_unique and summary.index.is_unique and len(curve)==len(curves)==len(saved),'Metric/PR grain differs')
    model_names=set(predictions.model.unique());expected_client_keys={(m,s,c) for m in model_names for s in SEEDS for c in CLIENTS}
    require(set(saved.index)==set(negative.index)==set(curve)==expected_client_keys,'Per-client/negative/PR output cohorts differ')
    require(set(summary.index)==model_names and set(aggregate.index)=={(m,s,scope) for m in model_names for s in SEEDS for scope in ['macro_equal_client','pooled_all_client_months']},'Aggregate/seed summary cohort differs')
    metric_receipt=read(evaluation/'metrics_receipt.json');require(metric_receipt['no_clipping_or_metric_floors'] is True and metric_receipt['pooled_reported_separately'] is True,'Metric policy declaration differs')
    latency_saved=frame(evaluation/'matched_latency_per_client.csv').set_index(['model','seed','client']).sort_index();latency_expected=saved.loc[[k for k in saved.index if '_LATENCY_' in k[0]]].sort_index()
    exported_table_equal(latency_saved,latency_expected,'Matched latency table versus combined metrics')
    latcurves=read(evaluation/'matched_latency_PR_curves.json');require({(r['model'],int(r['seed']),r['client']):r for r in latcurves}=={k:v for k,v in curve.items() if '_LATENCY_' in k[0]},'Matched latency PR curves differ')
    perclient={};maxdiff=defaultdict(float);recordcount=0;prchecks=0
    for key,g in predictions.groupby(['model','seed','client'],sort=True):
        model,seed,client=key;g=g.sort_values('target_month');y=g.y.to_numpy();p=g.prediction.to_numpy();base=regression_reference(y,p)
        threshold=heavy['LATENCY_REFERENCE' if '_LATENCY_' in model else 'MAIN'][client]
        probability=None
        if 'heavy_probability' in g and g.heavy_probability.notna().any():
            require(g.heavy_probability.notna().all(),'Partially missing probability');probability=g.heavy_probability.to_numpy();require(((probability>=0)&(probability<=1)).all(),'Probability out of range')
        event=heavy_reference(y,p,threshold,probability);scores=p if probability is None else probability;pr=independent_heavy_curve(y>threshold,scores)
        event['heavy80_training_threshold_mm']=threshold;event['heavy_average_precision']=pr['AP'];event['heavy_ROC_AUC']=pr['AUC']
        actual=saved.loc[key]
        for name,v in {**base,**event}.items():maxdiff[name]=max(maxdiff[name],scalar_close(v,actual[name],f'{key}/{name}'))
        require(actual['NSE']==actual['R2'] or (math.isnan(actual['NSE']) and math.isnan(actual['R2'])),'NSE must exactly equal R2')
        c=curve[key]
        for name,a in [('precision',pr['precision']),('recall',pr['recall']),('score_threshold',pr['threshold'])]:
            require(np.shape(a)==np.shape(c[name]) and np.allclose(a,c[name],rtol=1e-10,atol=1e-10,equal_nan=True),'Independent PR curve differs')
        expected_kind='continuous predicted rainfall mm; not calibrated probability' if probability is None else 'predicted heavy-event probability'
        require(actual['heavy_score_kind']==c['score_kind']==expected_kind,'Heavy score population mislabel')
        n=negative.loc[key];require(int(n.negative_count)==int((p<0).sum()) and int(n.n)==96,'Negative forecast count clipped/misreported')
        scalar_close(n.minimum_raw_prediction_mm,p.min(),'Raw minimum');scalar_close(n.negative_total_magnitude_mm,-p[p<0].sum(),'Negative magnitude')
        perclient[key]=base;recordcount+=1;prchecks+=1
    macro_by_model=defaultdict(list)
    for key,g in predictions.groupby(['model','seed'],sort=True):
        model,seed=key;client_values=[perclient[(model,seed,c)] for c in CLIENTS]
        macro={name:statistics.fmean([c[name] for c in client_values]) if all(math.isfinite(c[name]) for c in client_values) else math.nan for name in REG}
        pooled=regression_reference(g.y,g.prediction)
        for scope,expected in [('macro_equal_client',macro),('pooled_all_client_months',pooled)]:
            for name in REG:maxdiff[name]=max(maxdiff[name],scalar_close(expected[name],aggregate.loc[(model,seed,scope),name],f'{model}/{seed}/{scope}/{name}'))
        macro_by_model[model].append(macro)
    for model,rows in macro_by_model.items():
        for name in REG:
            values=[r[name] for r in rows];clean=[v for v in values if math.isfinite(v)]
            expected={'mean':statistics.fmean(clean) if clean else math.nan,'std':statistics.stdev(clean) if len(clean)>1 else math.nan,'min':min(clean) if clean else math.nan,'max':max(clean) if clean else math.nan}
            for suffix,v in expected.items():scalar_close(v,summary.loc[model,name+'_'+suffix],f'{model}/seed_summary/{name}/{suffix}')
    report['scalar_metrics']={'per_client_groups':recordcount,'PR_curves':prchecks,'independent_training_q80':heavy,'maximum_absolute_metric_discrepancies':dict(maxdiff),'primary_macro_and_pooled_separate':True,'signed_PBIAS':'100*sum(obs-pred)/sum(obs), positive underprediction','NSE_equals_R2':True,'raw_negative_forecasts_included':True,'no_head_probabilities_not_fabricated':True,'seed_summary_from_metrics_not_averaged_predictions':True,'matched_latency_tables_and_curves_checked':True}

def uncertainty(evaluation,protocol,truth,arrays,report):
    summary=frame(evaluation/'paired_block_bootstrap_summary.csv');perseed=frame(evaluation/'paired_per_seed_summary.csv');randomization=read(evaluation/'block_sign_randomization_Holm.json') if (evaluation/'block_sign_randomization_Holm.json').exists() else None
    comparisons=protocol['comparisons'];records=[];pvalues=[]
    require(len(summary)==31*3*2 and len(perseed)==31*5,'Paired uncertainty output count differs')
    for i,(a,b) in enumerate(comparisons):
        reference=bootstrap_reference(truth,arrays[a],arrays[b]);saved_draws=frame(evaluation/f'bootstrap_draws_{i:02d}.csv');receipt=read(evaluation/f'bootstrap_receipt_{i:02d}.json')
        require(receipt['models']==[a,b] and receipt['seed_keys']==SEEDS and receipt['client_keys']==CLIENTS and receipt['month_keys']==MONTHS and receipt['bootstrap_seed']==731 and receipt['blocks']==[12,6,24] and receipt['iid_rows_resampled'] is False,'Uncertainty pairing/RNG receipt differs')
        require(len(saved_draws)==6000 and not saved_draws.duplicated(['block_length_months','replicate']).any(),'Bootstrap draw count/grain differs')
        max_error=0.
        for length,deltas in reference['draws'].items():
            observed=saved_draws[saved_draws.block_length_months==length].sort_values('replicate');require(list(observed.replicate)==list(range(2000)),'Bootstrap replicate numbering')
            actual=observed[['delta_macro_R2_a_minus_b','delta_macro_RMSE_mm_a_minus_b']].to_numpy();delta=float(np.max(np.abs(deltas-actual)));require(delta<1e-9,'Reconstructed PCG64 paired draws differ');max_error=max(max_error,delta)
        centers=[]
        for row in reference['summary']:
            for j,metric in enumerate(['macro_R2','macro_RMSE_mm']):
                match=summary[(summary.model_a==a)&(summary.model_b==b)&(summary.metric==metric)&(summary.block_length_months==row['block'])];require(len(match)==1,'Missing comparison summary');s=match.iloc[0]
                for field,expected in [('delta_a_minus_b',row['point'][j]),('CI95_low',row['CI95_low'][j]),('CI95_high',row['CI95_high'][j])]:scalar_close(expected,s[field],f'bootstrap/{i}/{metric}/{field}')
                require(s.CI95_low<=s.CI95_high and bool(s.resample_seeds) is True,'CI interval or seed resampling')
                centers.append({'block':row['block'],'metric':metric,'point':row['point'][j],'draw_mean':row['draw_mean'][j],'CI_midpoint':row['CI_midpoint'][j],'CI_midpoint_minus_point':row['CI_midpoint'][j]-row['point'][j]})
        seed_rows=perseed[(perseed.model_a==a)&(perseed.model_b==b)].sort_values('seed');require(list(seed_rows.seed)==SEEDS,'Paired per-seed summary cohort')
        require(np.allclose(seed_rows[['delta_macro_R2_a_minus_b','delta_macro_RMSE_mm_a_minus_b']].to_numpy(),reference['perseed'],atol=1e-9,rtol=1e-10),'Per-seed delta direction differs')
        for j,metric in enumerate(['R2','RMSE_mm']):require(math.isclose(reference['perseed'][:,j].mean(),reference['point'][j],abs_tol=1e-12),'Point is not mean per-seed macro delta')
        if randomization is not None:
            s=sign_reference(truth,arrays[a],arrays[b]);r=randomization[i];require(r['model_a']==a and r['model_b']==b and r['block_length_months']==12 and r['n_blocks']==8,'Sign comparison family differs');scalar_close(s['delta'],r['delta_macro_R2_a_minus_b'],'Sign delta');scalar_close(s['p'],r['two_sided_p'],'Exact256 sign p',atol=0);pvalues.append(s['p'])
        records.append({'comparison':i,'models':[a,b],'paired_draws_recomputed':6000,'maximum_draw_discrepancy':max_error,'CI_centers':centers,'direction':'R2 positive favors A; RMSE negative favors A'})
        print(f'Independent paired uncertainty {i+1}/31 verified',flush=True)
    if randomization is not None:
        for expected,r in zip(holm_reference(pvalues),randomization):scalar_close(expected,r['Holm_p'],'Holm31 family',atol=0)
    report['paired_uncertainty']={'comparisons':records,'paired_bootstrap_draws':31*6000,'PCG64_seed':731,'joint_dates_paired_clients_and_seeds':True,'pvalues_recomputed':randomization is not None,'Holm_family_size':31 if randomization is not None else None,'CI_center_note':'Interval midpoint and bootstrap mean are recorded as diagnostics; neither is required to equal the observed paired point estimate.'}

def fit_audit(run,evaluation,identity,frozen,report):
    ledger=frame(evaluation/'fit_cost_ledger.csv').set_index('job_sha256');timing=frame(evaluation/'fit_timing.csv').set_index('job_sha256');cost_receipt=read(evaluation/'fit_cost_receipt.json');require(ledger.index.is_unique and set(ledger.index)==set(frozen['fit_complete_hashes']) and len(ledger)==cost_receipt['unique_fit_bundles'],'Fit cost dedup map differs')
    rows=[];wall=0;cpu=0
    for job in frozen['all_fit_jobs']:
        key=job['job_sha256'];path=run/'fits'/key[:16];receipt=read(path/'complete.json')
        payload={k:v for k,v in job.items() if k not in ['root','job_sha256']}
        require(digest({'run_sha256':identity['run_sha256'],'payload':payload})==key,'Fit job hash does not bind active scientific run identity')
        require(sha(path/'complete.json')==frozen['fit_complete_hashes'][key],'Frozen fit completion hash changed')
        original=dict(receipt['job']);original.pop('root',None);expected=dict(job);expected.pop('root',None);require(original==expected,'Fit config identity differs')
        hash_tree(path,receipt['files']);require(receipt['evaluation_labels_accessed'] is False and receipt['cpu_process_thread_cap']==2,'Fit access scope/cpu cap differs')
        if job['kind']=='neural':
            expected_cost,detail=neural_cost(path,receipt,read);meta=read(path/'model/manifest.json')
            require(meta['torch_version']==identity['runtime_environment']['packages']['torch']['loaded_version'] and meta['numpy_version']==identity['runtime_environment']['packages']['numpy']['loaded_version'],'Scientific checkpoint runtime differs')
        else:expected_cost,detail=other_cost(path,receipt,read)
        for name,expected in expected_cost.items():scalar_close(expected,ledger.loc[key,name],'Fit cost/'+key+'/'+name,atol=0)
        scalar_close(receipt['elapsed_wall_seconds'],ledger.loc[key,'elapsed_wall_seconds'],'Fit wall time');scalar_close(receipt['worker_process_cpu_seconds'],ledger.loc[key,'worker_process_cpu_seconds'],'Fit CPU time')
        scalar_close(receipt['elapsed_wall_seconds'],timing.loc[key,'elapsed_wall_seconds'],'Fit timing duplicate');wall+=receipt['elapsed_wall_seconds'];cpu+=receipt['worker_process_cpu_seconds']
        rows.append({'job_sha256':key,**expected_cost,**detail})
    scalar_close(wall,cost_receipt['sum_worker_wall_seconds'],'Unique summed worker wall time');scalar_close(cpu,cost_receipt['sum_process_cpu_seconds'],'Unique summed CPU time')
    require(cost_receipt['each_unique_complete_fit_counted_once'] is True,'Duplicate cost counting')
    shared=read(evaluation/'shared_graph_setup_ledger.json');require(shared['count_each_artifact_once'] is True and shared['applies_only_to']=='MAIN graph-family offline preparation; three graph variants share this artifact','Privileged graph setup charged unrelated methods')
    counts={kind:sum(job['kind']==kind for job in frozen['all_fit_jobs']) for kind in ['neural','classical','router','baselines']}
    estimator_operations=sum(1 if job['config']['scope']=='pooled' else 5 for job in frozen['all_fit_jobs'] if job['kind']=='classical')
    report['fit_integrity_and_cost']={'unique_completed_fits':len(rows),'all_artifact_hashes_verified':True,'all_fit_label_roles_train_monitor_validation_only':True,'all_neural60rounds_checked':True,'deduplicated_cost_rows':rows,'fit_bundle_counts':counts,'scientific_work_counts':{'direct_classical_estimator_fit_operations_implied_by_declared_scope':estimator_operations,'OOF_control_fit_bundles':18*counts['router'],'OOF_estimator_fit_operations_implied_by_declared_scope':54*counts['router'],'router_client_Ridge_stack_fit_operations':5*counts['router'],'actual_neural_optimizer_steps':sum(r['optimizer_steps'] for r in rows),'actual_neural_training_examples_presented':sum(r['training_examples_presented'] for r in rows),'count_scope':'Unique saved bundles; tree/Ridge estimator operations are implied by admitted pooled/local scope, not independent optimizer telemetry.'},'sum_worker_wall_seconds':wall,'sum_process_cpu_seconds':cpu,'graph_artifact_once_and_sensitivity_alias_zero_additional':True,'model_training_or_checkpoint_inference_in_this_audit':False}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();run=args.run.resolve();output=args.output.resolve()
    if not (run/'evaluation/complete.json').is_file():ap.error('Evaluation completion gate absent; no result/truth content read.')
    require(not output.exists(),'Audit output already exists; use a fresh directory')
    report={'status':'RUNNING','audit_python':sys.version,'audit_program_sha256':sha(__file__),'helper_sha256':{p.name:sha(p) for p in [Path(__file__).with_name('metric_reference.py'),Path(__file__).with_name('uncertainty_reference.py'),Path(__file__).with_name('fit_cost_reference.py')]},'scientific_source_mutations':False,'training_or_checkpoint_inference_performed':False,'findings':[]};started=time.perf_counter()
    try:
        evaluation,completed,identity,frozen,protocol=gates(run,report)
        report['sections_completed']=['completion_source_runtime_file_gates']
        predictions,truth,arrays=prediction_keys(evaluation,completed,frozen,report)
        report['sections_completed'].append('all_prediction_key_calendars_truth_and_identities')
        metrics(run,evaluation,predictions,report)
        report['sections_completed'].append('all_scalar_heavy_PR_negative_metrics')
        fit_audit(run,evaluation,identity,frozen,report)
        report['sections_completed'].append('all_fit_hashes_chronology_checkpoint_and_cost')
        uncertainty(evaluation,protocol,truth,arrays,report)
        report['sections_completed'].append('all31_paired_bootstrap_and_requested_sign_Holm')
        require(completed['pvalues']==report['paired_uncertainty']['pvalues_recomputed'],'P-value policy receipt differs')
        report['status']='PASS'
    except Exception as error:
        report['status']='FAIL';report['findings'].append({'severity':'blocking','error':str(error)});raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        report['checked_table_coverage']={'hash_authentication':'Every evaluation file named in evaluation/complete.json; every prepared dataset/feature/schema/graph file and every file of every frozen unique fit','numerical_recalculation':['raw_predictions.csv key calendars/truth/prediction finiteness/checkpoint identities/alias equality','per_client.csv all scalar and heavy metrics','aggregate_per_seed.csv all macro and pooled scalar metrics','seed_summary.csv all mean/std/min/max values','PR_curves.json all PR/AP/AUC curves','matched_latency_per_client.csv and matched_latency_PR_curves.json identity and recalculation through combined tables','negative_forecast_diagnostics.csv all counts/minima/magnitudes','paired_block_bootstrap_summary.csv all31 comparisons×3 blocks×2 metrics','paired_per_seed_summary.csv all31×5 paired seed deltas','bootstrap_draws_00..30.csv all186000 paired draws and corresponding receipts','block_sign_randomization_Holm.json exact eight-block sign and full-family adjustment when requested','fit_cost_ledger.csv and fit_timing.csv every unique fit cost/timing row','fit_cost_receipt.json summed unique costs and shared_graph_setup_ledger.json alias/scope','model_identities.json exact frozen outputs','model curves/local_steps/messages/manifest JSON chronology,selection,protocol arithmetic for every neural fit','classical/baseline/router fit and OOF receipts chronology/population/same-pool gates'],'not_recomputed':['Training trajectories or inference from checkpoints','Raw reanalysis aggregation, already covered by separate source audit','Figure rendering, handled by the figure task']}
        if report['status']!='PASS':
            report['checked_table_coverage']['planned_numerical_recalculation']=report['checked_table_coverage'].pop('numerical_recalculation')
            report['checked_table_coverage']['coverage_note']='Audit failed; only sections_completed identifies fully completed checks. Planned coverage is not asserted as completed.'
        output.mkdir(parents=True,exist_ok=False)
        content=json.dumps(report,indent=2,allow_nan=False)
        (output/'completed_result_audit.json').write_text(content,encoding='utf-8');(output/'complete.json').write_text(content,encoding='utf-8')
    print(json.dumps({'status':report['status'],'models':report['prediction_keys']['models'],'unique_fits':report['fit_integrity_and_cost']['unique_completed_fits'],'paired_draws':186000,'output':str(output/'completed_result_audit.json')}))

if __name__=='__main__':main()
