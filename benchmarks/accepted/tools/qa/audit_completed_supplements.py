"""Independent completed-supplement QA; no models, fitting or inference.

Completion precedes every study source/array/prediction/score read. Contrasts
remain within each source: ADDED minus its own matched REFERENCE.
"""
from pathlib import Path
from collections import Counter
import argparse,json,math,statistics,time
import numpy as np
import pandas as pd
from audit_completed_run import read,sha,digest,hash_tree,frame,scalar_close,exported_table_equal,independent_heavy_curve,keyed_sha,REG
from metric_reference import regression_reference,heavy_reference
from supplement_reference import role_calendars,derived_targets,contiguous_calendar,own_train_statistics,cohort_bootstrap_reference

CASES=('IMD','solar','GLDAS21');ARMS=('REFERENCE','ADDED');METHODS=('fedavg','fedprox','caaa');IDENTITIES=('global','personal')
SEEDS=sorted([42,123,2024,7,999])
CLIENTS=sorted(['Client1_NW_Arid','Client2_Indo_Gangetic','Client3_NE_Himalaya','Client4_South_Peninsular','Client5_Central_India'])
def require(condition,message):
    if not condition:raise AssertionError(message)
def label(method,identity,case,arm):return f'{method.upper()}_{identity.upper()}_{case}_{arm}'
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def write_receipt(output,report):
    content=json.dumps(clean(report),indent=2,sort_keys=True,allow_nan=False)
    (output/'completed_supplement_audit.json').write_text(content,encoding='utf-8')
    (output/'complete.json').write_text(content,encoding='utf-8')

def gates(run,primary,report):
    evaluation=run/'evaluation'
    if not (evaluation/'complete.json').is_file():raise RuntimeError('Supplement evaluation completion absent. No study scientific files were opened.')
    completed=read(evaluation/'complete.json');identity=read(run/'run_identity.json');protocol=read(run/'protocol.json');frozen=read(run/'choices_frozen.json');prepared=read(run/'prepared.json')
    payload=dict(identity);payload.pop('run_sha256')
    require(digest(payload)==identity['run_sha256']==completed['run_sha256']==frozen['run_sha256']==prepared['run_sha256'],'Supplement scientific run identity differs')
    require(identity['protocol_hash']==digest(protocol),'Supplement protocol changed')
    require(protocol['cases']==list(CASES) and protocol['arms']==list(ARMS) and protocol['methods']==list(METHODS) and protocol['seeds']==[42,123,2024,7,999],'Supplement methods/cases/seeds changed')
    require(protocol['fit_count']==90 and protocol['rounds']==60 and protocol['local_epochs']==3 and protocol['batch_size']==32 and protocol['threads_per_worker']==2 and protocol['max_workers']==2,'Supplement bounded schedule changed')
    comparisons=[[label(m,i,c,'ADDED'),label(m,i,c,'REFERENCE')] for c in CASES for m in METHODS for i in IDENTITIES]
    require(protocol['comparisons']==comparisons,'Within-case 18-comparison family differs')
    require(completed['complete_fits']==90 and completed['fit_choices_or_subsets_selected_from_evaluation'] is False and completed['supplement_retuning'] is False and completed['clipped_predictions'] is False,'Completed supplement policy differs')
    require(frozen['hyperparameters_frozen_from_primary'] is True and frozen['all_checkpoints_frozen'] is True and frozen['evaluation_labels_accessed'] is False,'Supplement checkpoint/evaluation sealing differs')
    require(prepared['model_fits']==prepared['evaluation_predictions']==0,'Preparation fitted/scored models')
    hash_tree(evaluation,completed['files']);hash_tree(run,prepared['files'])
    primary_identity=read(primary/'run_identity.json');hp_path=primary/'development/hyperparameters_frozen.json'
    require(primary_identity['run_sha256']==identity['primary_run_sha256'],'Primary run linkage changed')
    require(sha(hp_path)==identity['primary_hyperparameters_sha256']==sha(run/'primary_choices_frozen.json'),'Primary frozen HP receipt changed')
    hp=read(hp_path);expected_hp={'lr':hp['lr'],'mu':{m:hp['mu'][m] for m in ('fedprox','caaa')},'coefficients':hp['coefficients']}
    require(hp['evaluation_labels_accessed'] is False and identity['primary_choices']==expected_hp,'Supplement HP retuned or differs from primary')
    require(identity['module_code']==primary_identity['code'] and identity['runtime_environment']==primary_identity['runtime_environment'],'Primary scientific code/environment parity differs')
    package=run.parent.parent
    for name,h in identity['module_code'].items():require(sha(package/'caadapfl'/(name+'.py'))==h,'Active frozen scientific module differs: '+name)
    require(sha(package/'tools/supplement_runner.py')==identity['launcher_sha256'],'Frozen supplement launcher differs')
    for name,h in identity['canonical_input_hashes'].items():
        require(h==primary_identity['source'][name+'_sha256'] and sha(package/'data'/(name+'.json' if name=='manifest' else name+'.csv'))==h,'Primary canonical source linkage differs')
    jobs=frozen['jobs'];require(len(jobs)==90 and len({j['job_sha256'] for j in jobs})==90,'Fits duplicated or missing')
    require({(j['case'],j['arm'],j['method'],j['config']['seed']) for j in jobs}=={(c,a,m,s) for c in CASES for a in ARMS for m in METHODS for s in SEEDS},'Case/arm/method/seed schedule differs')
    require(set(frozen['fit_complete_hashes'])=={j['job_sha256'] for j in jobs},'Unique completion map differs')
    schedule=read(run/'resolved_fit_schedule.json');require(schedule['jobs']==jobs and schedule['count']==90 and schedule['retuning'] is False,'Resolved frozen schedule differs')
    report.update(run_sha256=identity['run_sha256'],primary_run_sha256=identity['primary_run_sha256'],evaluation_complete_sha256=sha(evaluation/'complete.json'),checked_files={'evaluation':completed['files'],'prepared':prepared['files'],'fit_completion_receipts':frozen['fit_complete_hashes'],'scientific_modules':identity['module_code']},scope={'cases':list(CASES),'arms':list(ARMS),'methods':list(METHODS),'identities':list(IDENTITIES),'seeds':SEEDS,'clients':CLIENTS,'paired_contrasts':18,'comparison_policy':'Within-case ADDED minus own matched REFERENCE only; no cross-case ranking','prediction_policy':'Raw mm; negative predictions retained; no metric floors'},limitations=['Saved files/receipts are checked without optimizer recreation or inference.','Simulated numeric communication payloads are neither network measurements nor a privacy guarantee. Target inverse summaries total80bytes per fit; schema q80 is exported, regardless of older locality prose.','Circular blocks are conditional on five fixed regions and revised archives; interior gaps cause rejection.','Constant-observation bootstrap R2 causes explicit rejection, following frozen evaluator; scalar/heavy undefined metrics remain undefined.','No new hypothesis tests, multiplicity family, fitted choices or cross-source rankings.'],formulas={'regression':'Independent scalar R2/NSE, RMSE, MAE, signed PBIAS, correlation and KGE2009','heavy':'Own gradient-TRAIN linear q80; observed >q80, probability >=.5; tied PR/AP/AUC','bootstrap':'Same common dates and paired5seeds for both arms; PCG64 seed731,2000 replicates each12/6/24 months'})
    report['identity']={'canonical_input_hashes':identity['canonical_input_hashes'],'source_hashes':identity['source_hashes'],'primary_hyperparameters_sha256':identity['primary_hyperparameters_sha256'],'primary_choices':expected_hp,'launcher_sha256':identity['launcher_sha256'],'runtime_environment':identity['runtime_environment'],'choices_frozen_sha256':sha(run/'choices_frozen.json')}
    return evaluation,package,identity,frozen,prepared,completed

def authenticated_plan(run,package,case,expected_sha256):
    # Preparation canonically serializes a parsed plan; original source bytes
    # and the run copy can differ in key order while holding identical values.
    leaf=case+'_cohort_plan.json'
    candidates=[package/'tools/derived/cohorts'/leaf,package/'tools/derived/cohorts_gldas_admitted'/leaf]
    originals=[p for p in candidates if p.is_file() and sha(p)==expected_sha256]
    require(bool(originals),'Original admitted cohort source hash differs')
    original=read(originals[0]);stored_path=run/'cohorts'/(case+'.json')
    require(read(stored_path)==original,'Frozen run cohort content differs from authenticated source plan')
    return original,sha(stored_path)

def source_cohorts(run,package,identity,prepared,report):
    hashes=identity['source_hashes'];sources={};plans={};paths={}
    for case,directory in [('IMD',package/'tools/derived/imd'),('GLDAS21',package/'tools/derived/gldas_full')]:
        meta=read(directory/'manifest.json');require(sha(directory/'manifest.json')==hashes[case+'/receipt'],'Admitted source receipt hash changed')
        require(meta['status']==('PASS_WITH_EXPLICIT_INVALID_SOURCE' if case=='IMD' else 'PASS'),'Source admission status differs')
        feature='imd_supplement.csv' if case=='IMD' else 'gldas21_supplement.csv';quality='imd_supplement_quality.csv' if case=='IMD' else 'gldas21_supplement_quality.csv'
        for leaf in (feature,quality):
            claim=meta['outputs'][leaf];h=claim['sha256'] if isinstance(claim,dict) else claim
            require(sha(directory/leaf)==h==hashes[case+'/'+leaf],'Source/quality hash differs')
        data=frame(directory/feature).set_index(['client','month']).sort_index();q=frame(directory/quality).set_index(['client','month']).sort_index();start='1981-01' if case=='IMD' else '2000-01'
        expected=pd.MultiIndex.from_product([CLIENTS,list(pd.period_range(start,'2024-12',freq='M').astype(str))],names=['client','month'])
        require(data.index.equals(expected) and q.index.equals(expected) and data.index.is_unique and q.index.is_unique,'Source full calendar/grain differs')
        names=['imd_rainfall_mm'] if case=='IMD' else meta['met_columns'];require(list(data.columns)==names,'Supplement source columns/order differ')
        valid=q.feature_admitted.eq(1)&q.weighted_coverage.ge(.95)&q.weighted_coverage.le(1+1e-10)
        require(np.array_equal(valid,np.isfinite(data.to_numpy(float)).all(1)),'Source admission and NA pattern differ')
        if case=='IMD':require(set(q.index[~valid])=={('Client1_NW_Arid','2023-10')},'IMD invalid source pattern differs')
        else:require(valid.all(),'GLDAS full source admission unexpectedly missing')
        sources[case]=(names,dict(zip(data.index,valid)));paths[case]={'receipt':sha(directory/'manifest.json'),'feature':sha(directory/feature),'quality':sha(directory/quality)}
    solar=package/'data/solar_supplement.csv';require(sha(solar)==hashes['solar/feature_csv'],'Solar source hash differs')
    data=frame(solar).set_index(['client','month']).sort_index();expected=pd.MultiIndex.from_product([CLIENTS,list(pd.period_range('1981-01','2024-12',freq='M').astype(str))],names=['client','month'])
    require(data.index.equals(expected) and data.index.is_unique,'Solar calendar differs')
    name='power_solar_mj_m2_day';valid=np.isfinite(data[name].to_numpy(float));structural=np.array([m>='1984-01' for c,m in data.index]);require(np.array_equal(valid,structural),'Solar structural1981–83 absence differs')
    manifest=read(package/'data/manifest.json');require(manifest['supplement_inputs']['solar']['sha256']==sha(solar) and manifest['units'][name].startswith('MJ m-2 day-1'),'Solar conversion/source contract differs')
    sources['solar']=([name],dict(zip(data.index,valid)));paths['solar']={'feature':sha(solar)}
    for case in CASES:
        plan,stored_sha=authenticated_plan(run,package,case,hashes[case+'/plan']);names,valid=sources[case];targets=derived_targets(valid,CLIENTS);roles=role_calendars(targets)
        require(plan['status']=='SOURCE_QUALITY_COHORT_DERIVED' and plan['extra_lag']==0 and plan['shared_clients']==CLIENTS,'Cohort source-quality contract differs')
        require(plan['target_months']==targets and plan['expected_evaluation_months']==roles['evaluation'],'Cohort differs from source-only t-15..t-4 availability')
        contiguous_calendar(roles['evaluation']);counts={r:len(k) for r,k in roles.items()}
        require({r:plan['role_counts_per_client'][r] for r in roles}==counts and prepared['cases'][case]['role_counts']==counts,'Source cohort role counts differ')
        require(prepared['cases'][case]['evaluation_months']==roles['evaluation'] and prepared['cases'][case]['supplement_met_columns']==names and prepared['cases'][case]['samples_per_client']==len(targets),'Prepared matched case differs')
        expected_end='2024-01' if case=='IMD' else '2024-12'
        require(roles['evaluation']==list(pd.period_range('2017-01',expected_end,freq='M').astype(str)),'Frozen case evaluation cohort differs')
        plans[case]={'plan':plan,'roles':roles,'names':names};paths[case].update(original_cohort_sha256=hashes[case+'/plan'],run_cohort_copy_sha256=stored_sha)
    report['source_cohorts']={c:{'role_counts':{r:len(k) for r,k in plans[c]['roles'].items()},'source_met_columns':plans[c]['names'],'target_calendar_sha256':keyed_sha((m,) for m in plans[c]['plan']['target_months']),'evaluation_months':plans[c]['roles']['evaluation'],'hashes':paths[c]} for c in CASES}
    return plans

def archive(run,case,arm,evaluation=False):
    path=run/'datasets'/(case+'_'+arm+'_FULL_'+('evaluation_features' if evaluation else 'fit')+'.npz')
    fields=['raw_x','raw_climate','raw_drivers','raw_target_month','raw_origin_month','scaled_x','scaled_climate','target_stats']+([] if evaluation else ['y','y_z','heavy'])
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={c+'::'+f for c in CLIENTS for f in fields},'Unexpected label/hidden field or missing NPZ field')
        data={c:{f:z[c+'::'+f].copy() for f in fields} for c in CLIENTS}
    return data

def dataset_statistics(run,plans,report):
    thresholds={};records={}
    canonical=frame(run.parent.parent/'data/monthly.csv').set_index(['client','month']).rainfall_mm
    for case in CASES:
        roles=plans[case]['roles'];fit_months=sorted(roles['train']+roles['monitor']+roles['validation']);targets=None;sensitivity=None;reference=None;records[case]={}
        for arm in ARMS:
            schema=read(run/'datasets'/(case+'_'+arm+'_FULL_schema.json'));names=schema['names'];context=schema['climate_names'];data=archive(run,case,arm);evaluation=archive(run,case,arm,True)
            require(len(names)==(43 if arm=='REFERENCE' else 43+len(plans[case]['names'])) and len(context)==7 and schema['extra_lag']==0,'Feature dimensions/context/latency differ')
            if arm=='ADDED':require(names[:43]==reference['names'] and names[43:]==plans[case]['names'],'Added arm changes common inputs')
            for c in CLIENTS:
                d=data[c];e=evaluation[c]
                require(d['raw_target_month'].tolist()==fit_months and e['raw_target_month'].tolist()==roles['evaluation'],'Fit/evaluation calendar separation differs')
                require(np.allclose(d['y'],[canonical.loc[(c,m)] for m in fit_months],rtol=0,atol=1e-10),'Fit labels differ from canonical CHIRPS targets')
                for v in (d,e):
                    dates=v['raw_target_month'];require(v['raw_origin_month'].tolist()==[str(pd.Period(t,freq='M')-1) for t in dates],'Origin calendar differs')
                    require(v['raw_x'].shape==(len(dates),12,len(names)) and v['raw_drivers'].shape==(len(dates),12,2) and v['raw_climate'].shape==(len(dates),7),'Array dimensions differ')
                    require(all(np.isfinite(a).all() for a in v.values() if a.dtype.kind in 'fiu'),'Nonfinite feature/label/scaling arrays')
                if arm=='ADDED':
                    for key in ('y','raw_target_month','raw_origin_month','raw_drivers','raw_climate','target_stats','y_z','heavy'):require(np.array_equal(d[key],reference['fit'][c][key]),'Matched fit arms differ in labels/axes/context')
                    for key in ('raw_target_month','raw_origin_month','raw_drivers','raw_climate','target_stats'):require(np.array_equal(e[key],reference['evaluation'][c][key]),'Matched evaluation arm axes/context differ')
                    require(np.array_equal(d['raw_x'][:,:,:43],reference['fit'][c]['raw_x']) and np.array_equal(e['raw_x'][:,:,:43],reference['evaluation'][c]['raw_x']),'Matched common raw features differ')
            own=own_train_statistics(CLIENTS,data,names,context,roles,schema['statistics']);now=own['target_statistics']
            if targets is None:targets=now;thresholds[case]={c:v[2] for c,v in now.items()}
            else:require(targets==now,'Arms have different own-TRAIN target statistics/q80')
            mean=np.asarray(schema['statistics']['global_input_mean']);std=np.asarray(schema['statistics']['global_input_std']);ci=[names.index(n) for n in context]
            for c,e in evaluation.items():
                require(np.array_equal(e['scaled_x'],((e['raw_x']-mean)/std).astype(np.float32)) and np.array_equal(e['scaled_climate'],((e['raw_climate']-mean[ci])/std[ci]).astype(np.float32)),'Evaluation input scaling differs')
                require(np.array_equal(e['target_stats'],data[c]['target_stats']),'Evaluation inverse-target scaling differs')
            local=read(run/'datasets'/(case+'_'+arm+'_local_sensitivity.json'))
            for c,v in local['clients'].items():require(v['fit_end']=='2007-11' and v['n']==len(roles['train']) and len(v['sensitivity'])==2,'TRAIN sensitivity receipt differs')
            if sensitivity is None:sensitivity=local
            else:require(sensitivity==local,'Arms have different training target sensitivities')
            setup=schema['statistics']['setup_communication'];require(setup['total_bytes']==sum(v['bytes'] for v in setup['messages'])==5*(8+32*len(names)),'Own-scaler setup ledger differs')
            records[case][arm]={'features':len(names),'TRAIN_count_all_clients':own['count'],'target_statistics':now,'scaler_source':'Latest-row gradient TRAIN only','q80_source':'Own gradient TRAIN targets','setup_bytes':setup['total_bytes'],'evaluation_archive_has_labels':False}
            if arm=='REFERENCE':reference={'names':names,'fit':data,'evaluation':evaluation}
    report['dataset_statistics']=records
    return thresholds

def predictions_and_metrics(run,evaluation,plans,thresholds,frozen,completed,report):
    combined=frame(evaluation/'raw_predictions_all_cases.csv');require(len(combined)==completed['raw_prediction_rows'],'Combined prediction count differs')
    require(set(combined.case)==set(CASES),'Combined case cohort differs');arrays={};observed={};negative=[];counts={};case_frames=[];aggregate_frames=[];summary_frames=[]
    canonical=frame(run.parent.parent/'data/monthly.csv').set_index(['client','month']).rainfall_mm
    for case in CASES:
        directory=evaluation/case;pred=frame(directory/'raw_predictions.csv');months=plans[case]['roles']['evaluation'];expected=pd.MultiIndex.from_product([SEEDS,CLIENTS,months],names=['seed','client','target_month']);models=[label(m,i,case,a) for m in METHODS for i in IDENTITIES for a in ARMS]
        require(not pred.duplicated(['model','seed','client','target_month']).any() and set(pred.model)==set(models),'Case prediction models/key grain differ')
        require(np.isfinite(pred[['y','prediction']].to_numpy(float)).all() and (pred.y>=0).all() and len(pred)==len(models)*len(expected),'Nonfinite or incomplete case predictions')
        truth=frame(run/'datasets'/(case+'_evaluation_truth.csv')).set_index(['client','target_month']).sort_index();truthkeys=pd.MultiIndex.from_product([CLIENTS,months],names=['client','target_month']);require(truth.index.equals(truthkeys),'Sealed case truth calendar differs')
        require(np.allclose(truth.y,[canonical.loc[(c,m)] for c,m in truthkeys],rtol=0,atol=1e-10),'Sealed case labels differ from canonical CHIRPS targets')
        first=None;arrays[case]={}
        for model in models:
            part=pred[pred.model==model].set_index(['seed','client','target_month']).sort_index();require(part.index.equals(expected),'Case five-seed/five-client calendar missing')
            y=part.y.to_numpy().reshape(5,5,len(months));require(np.array_equal(y,np.broadcast_to(y[:1],y.shape)),'Case observations vary across seeds')
            if first is None:first=y
            else:require(np.array_equal(first,y),'Models/arms have different keyed truths')
            require(np.allclose(y[0].ravel(),truth.y.to_numpy(),rtol=0,atol=1e-10),'Observations differ from sealed case truth')
            arrays[case][model]=part.prediction.to_numpy().reshape(5,5,len(months))
        observed[case]=first
        for job in frozen['jobs']:
            if job['case']!=case:continue
            meta=read(run/'fits'/job['job_sha256'][:16]/'model/manifest.json')
            for identity in IDENTITIES:
                g=pred[(pred.model==label(job['method'],identity,case,job['arm']))&(pred.seed==job['config']['seed'])]
                require(g.checkpoint_round.eq(meta['selected_rounds'][identity]).all(),'Evaluation checkpoint differs')
                require(g.heavy_probability.notna().all() if job['config']['use_aux'] else g.heavy_probability.isna().all(),'Auxiliary/no-head policy differs')
        per=frame(directory/'per_client.csv').set_index(['model','seed','client']);aggregate=frame(directory/'aggregate_per_seed.csv').set_index(['model','seed','scope']);summary=frame(directory/'seed_summary.csv').set_index('model');curves=read(directory/'PR_curves.json');curve={(r['model'],r['seed'],r['client']):r for r in curves}
        expected_client={(m,s,c) for m in models for s in SEEDS for c in CLIENTS}
        require(per.index.is_unique and set(per.index)==set(curve)==expected_client and len(curves)==len(expected_client),'Case scalar/PR cohorts differ')
        require(aggregate.index.is_unique and set(aggregate.index)=={(m,s,k) for m in models for s in SEEDS for k in ['macro_equal_client','pooled_all_client_months']} and summary.index.is_unique and set(summary.index)==set(models),'Aggregate/seed cohort differs')
        policy=read(directory/'metrics_receipt.json');require(policy['no_clipping_or_metric_floors'] is True and policy['pooled_reported_separately'] is True,'Case metric policy differs')
        local={};max_difference=0.;undefined_scalars=0;no_event_groups=0
        for key,g in pred.groupby(['model','seed','client'],sort=True):
            g=g.sort_values('target_month');y=g.y.to_numpy();p=g.prediction.to_numpy();prob=None if g.heavy_probability.isna().all() else g.heavy_probability.to_numpy();threshold=thresholds[case][key[2]]
            require(prob is None or (np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()),'Invalid/partially missing probability')
            scalar=regression_reference(y,p);event=heavy_reference(y,p,threshold,prob);pr=independent_heavy_curve(y>threshold,p if prob is None else prob);event.update(heavy80_training_threshold_mm=threshold,heavy_average_precision=pr['AP'],heavy_ROC_AUC=pr['AUC'])
            for name,value in {**scalar,**event}.items():max_difference=max(max_difference,scalar_close(value,per.loc[key,name],case+'/'+str(key)+'/'+name))
            undefined_scalars+=sum(not math.isfinite(v) for v in {**scalar,**event}.values())
            if any(not math.isfinite(v) for v in scalar.values()):require(pd.notna(per.loc[key,'undefined_reasons']) and bool(str(per.loc[key,'undefined_reasons'])),'Undefined scalar metric lacks stated reason')
            require(per.loc[key,'NSE']==per.loc[key,'R2'] or (math.isnan(per.loc[key,'NSE']) and math.isnan(per.loc[key,'R2'])),'NSE differs from R2')
            saved=curve[key];kind='continuous predicted rainfall mm; not calibrated probability' if prob is None else 'predicted heavy-event probability'
            if not event['actual_heavy_n']:
                no_event_groups+=1;require(saved.get('undefined_reason')=='No actual heavy events','No-event PR lacks undefined reason')
            require(saved['score_kind']==per.loc[key,'heavy_score_kind']==kind,'Heavy score type mislabeled')
            for name,values in [('precision',pr['precision']),('recall',pr['recall']),('score_threshold',pr['threshold'])]:require(np.shape(saved[name])==np.shape(values) and np.allclose(saved[name],values,rtol=1e-10,atol=1e-10,equal_nan=True),'Case tied PR differs')
            local[key]=scalar;negative.append({'case':case,'model':key[0],'seed':key[1],'client':key[2],'n':len(g),'negative_count':int((p<0).sum()),'minimum_raw_prediction_mm':float(p.min()),'negative_total_magnitude_mm':float(-p[p<0].sum())})
        bymodel={m:[] for m in models}
        for (model,seed),g in pred.groupby(['model','seed'],sort=True):
            cs=[local[(model,seed,c)] for c in CLIENTS];macro={k:statistics.fmean(r[k] for r in cs) if all(math.isfinite(r[k]) for r in cs) else math.nan for k in REG};pooled=regression_reference(g.y,g.prediction)
            for scope,values in [('macro_equal_client',macro),('pooled_all_client_months',pooled)]:
                for name in REG:max_difference=max(max_difference,scalar_close(values[name],aggregate.loc[(model,seed,scope),name],case+'/'+model+'/'+scope+'/'+name))
            bymodel[model].append(macro)
        for model,rows in bymodel.items():
            for name in REG:
                v=[r[name] for r in rows if math.isfinite(r[name])];expected_values={'mean':statistics.fmean(v) if v else math.nan,'std':statistics.stdev(v) if len(v)>1 else math.nan,'min':min(v) if v else math.nan,'max':max(v) if v else math.nan}
                for suffix,value in expected_values.items():scalar_close(value,summary.loc[model,name+'_'+suffix],case+'/'+model+'/seed_summary/'+name+'/'+suffix)
        case_frames.append(pred.assign(case=case));aggregate_frames.append(aggregate.reset_index().assign(case=case));summary_frames.append(summary.reset_index().assign(case=case))
        counts[case]={'models':12,'seeds':5,'clients':5,'dates':len(months),'prediction_rows':len(pred),'per_client_metric_PR_groups':len(per),'undefined_numeric_metric_entries':undefined_scalars,'no_actual_heavy_event_groups':no_event_groups,'maximum_scalar_difference':max_difference,'key_calendar_sha256':keyed_sha(tuple(r) for r in pred.sort_values(['model','seed','client','target_month'])[['model','seed','client','target_month']].itertuples(index=False,name=None))}
    for filename,expected,keys in [('raw_predictions_all_cases.csv',pd.concat(case_frames),['case','model','seed','client','target_month']),('aggregate_per_seed_all_cases.csv',pd.concat(aggregate_frames),['case','model','seed','scope']),('seed_summary_all_cases.csv',pd.concat(summary_frames),['case','model'])]:
        actual=frame(evaluation/filename).set_index(keys).sort_index();expected=expected.set_index(keys).sort_index()
        require(set(actual.columns)==set(expected.columns),'Combined export column schema differs');exported_table_equal(actual,expected.reindex(columns=actual.columns),'Combined-versus-case '+filename)
    report['predictions_and_metrics']=counts
    return observed,arrays,pd.DataFrame(negative)

def uncertainty(evaluation,plans,observed,arrays,report):
    details=[]
    for case in CASES:
        directory=evaluation/case;summary=frame(directory/'paired_block_bootstrap_summary.csv');perseed=frame(directory/'paired_per_seed_summary.csv');months=plans[case]['roles']['evaluation']
        require(len(summary)==36 and len(perseed)==30,'Six contrasts × metric/block/seed row count differs')
        require(not summary.duplicated(['model_a','model_b','metric','block_length_months']).any() and not perseed.duplicated(['model_a','model_b','seed']).any(),'Duplicated paired summary key')
        for method in METHODS:
            for identity in IDENTITIES:
                a=label(method,identity,case,'ADDED');b=label(method,identity,case,'REFERENCE');prefix=method+'_'+identity;draws=frame(directory/(prefix+'_bootstrap_draws.csv'));receipt=read(directory/(prefix+'_bootstrap_receipt.json'))
                require(receipt['models']==[a,b] and receipt['seed_keys']==SEEDS and receipt['client_keys']==CLIENTS and receipt['month_keys']==months and receipt['blocks']==[12,6,24] and receipt['bootstrap_seed']==731 and receipt['iid_rows_resampled'] is False,'Within-case bootstrap receipt differs')
                own=cohort_bootstrap_reference(observed[case],arrays[case][a],arrays[case][b],months);max_difference=0.
                require(len(draws)==6000 and not draws.duplicated(['block_length_months','replicate']).any(),'Bootstrap replicate count/grain differs')
                for block,expected_draws in own['draws'].items():
                    group=draws[draws.block_length_months==block].sort_values('replicate');require(group.replicate.tolist()==list(range(2000)),'Paired replicate keys differ');actual=group[['delta_macro_R2_a_minus_b','delta_macro_RMSE_mm_a_minus_b']].to_numpy(float)
                    require(np.allclose(expected_draws,actual,rtol=1e-10,atol=1e-10),'Independent paired draws differ');max_difference=max(max_difference,float(np.max(np.abs(expected_draws-actual))))
                own_summary=[]
                for row in own['summary']:
                    for i,metric in enumerate(['macro_R2','macro_RMSE_mm']):
                        selected=summary[(summary.model_a==a)&(summary.model_b==b)&(summary.metric==metric)&(summary.block_length_months==row['block'])];require(len(selected)==1,'Paired summary row missing');saved=selected.iloc[0]
                        require(saved.repetitions==2000 and saved.n_seeds==saved.n_clients==5 and saved.n_months==len(months) and saved.resample_seeds==True,'Paired cohort/resampling metadata differs')
                        for field,value in [('delta_a_minus_b',row['point'][i]),('CI95_low',row['CI95_low'][i]),('CI95_high',row['CI95_high'][i])]:scalar_close(value,saved[field],case+'/'+a+'/'+metric+'/'+field)
                        own_summary.append({'block':row['block'],'metric':metric,'point':row['point'][i],'CI95_low':row['CI95_low'][i],'CI95_high':row['CI95_high'][i],'draw_mean':row['draw_mean'][i],'CI_midpoint':row['CI_midpoint'][i]})
                group=perseed[(perseed.model_a==a)&(perseed.model_b==b)].sort_values('seed');require(group.seed.tolist()==SEEDS,'Per-seed paired keys differ')
                for i,saved in enumerate(group.itertuples()):
                    for j,metric in enumerate(['R2','RMSE_mm']):
                        for suffix,value in [('a',own['perseed'][i]['a'][j]),('b',own['perseed'][i]['b'][j]),('delta',own['perseed'][i]['delta'][j])]:
                            field='delta_macro_'+metric+'_a_minus_b' if suffix=='delta' else 'macro_'+metric+'_'+suffix;scalar_close(value,getattr(saved,field),'Per-seed paired '+field)
                details.append({'case':case,'model_a':a,'model_b':b,'maximum_draw_difference':max_difference,'summaries':own_summary,'per_seed_deltas':[row['delta'] for row in own['perseed']]})
    require(len(details)==18,'Exactly18 within-case paired contrasts required')
    report['paired_uncertainty']={'comparisons':18,'bootstrap_draws_checked':108000,'interior_gaps_bridged':False,'new_pvalues_or_cross_case_rankings':False,'details':details}

def fit_cost(path,receipt,roles):
    cfg=receipt['job']['config'];meta=read(path/'model/manifest.json');steps=read(path/'model/local_steps.json');messages=read(path/'model/messages.json');curves=read(path/'model/curves.json');setup=read(path/'setup_ledger.json');ntrain=len(roles['train'])
    require(meta['config']==cfg and cfg['rounds']==60 and cfg['local_epochs']==3 and cfg['batch_size']==32 and cfg['cpu_threads']==2 and cfg['method'] in METHODS,'Supplement scientific fit config differs')
    for key,value in {'train_end':'2007-11-01','monitor_start':'2008-01-01','monitor_end':'2010-11-01','validation_start':'2011-01-01','validation_end':'2016-11-01'}.items():require(cfg[key]==value,'Strict config label-release boundary differs')
    require(meta['client_ids']==CLIENTS and meta['rainfall_prediction_policy']=='raw inverse-transformed mm; no clipping in checkpoint scores or inference','Client/prediction policy differs')
    for c in CLIENTS:
        for role in ('train','monitor','validation'):
            saved=meta['data_manifest'][c][role];keys=[m+'-01' for m in roles[role]]
            require(saved['keys']==keys and saved['count']==len(keys) and saved['last']==keys[-1],'Fit label release calendar differs')
    expected={(r,c,'main') for r in range(1,61) for c in CLIENTS};require(len(steps)==300 and {(r['round'],r['client_id'],r['stream']) for r in steps}==expected,'Actual60×5 update coverage differs')
    for row in steps:
        require(row['optimizer_steps']==3*math.ceil(ntrain/32) and row['examples']==3*ntrain and row['optimizer']==cfg['optimizer'],'Actual optimizer steps/examples differ')
        require(row['sgd_steps']==(row['optimizer_steps'] if cfg['optimizer']=='sgd' else 0),'Optimizer telemetry mislabeled')
        require(all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))),'Nonfinite optimizer/proximity telemetry')
        if cfg['method']=='fedavg' or not cfg['use_prox']:require(row['lambda_mean']==row['prox_mean']==0,'Disabled/FedAvg proximity active')
        elif cfg['method']=='fedprox' or not cfg['adaptive_prox']:
            for key in ('lambda_mean','lambda_min','lambda_max'):require(math.isclose(row[key],cfg['prox_mu'],abs_tol=1e-7),'Fixed proximity differs')
        else:require(.3*cfg['prox_mu']-1e-7<=row['lambda_min']<=row['lambda_mean']<=row['lambda_max']<=cfg['prox_mu']+1e-7,'Adaptive [.3mu,mu] bound differs')
    selected={}
    for identity in IDENTITIES:
        values=[r for r in curves if r.get('kind')=='evaluation' and r.get('identity')==identity];require([v['round'] for v in values]==list(range(1,61)),'Not all60 checkpoints saved')
        best=-math.inf;chosen=None
        for row in values:
            require(set(row['validation'])==set(CLIENTS) and set(row['inner_monitor'])==set(CLIENTS),'Checkpoint client keys differ')
            require(all(v['n']==len(roles['validation']) for v in row['validation'].values()) and all(v['n']==len(roles['monitor']) for v in row['inner_monitor'].values()),'Checkpoint validation/monitor calendars differ')
            score=statistics.fmean(v['r2'] for v in row['validation'].values());scalar_close(score,row['validation_macro_r2'],'Checkpoint macro arithmetic',atol=1e-14)
            if score>best+1e-12:best=score;chosen=row['round']
        require(meta['selected_rounds'][identity]==chosen,'Checkpoint selection differs');selected[identity]=chosen
    counts=Counter(m['kind'] for m in messages);expected_messages={'initial_model':5,'train_count':5,'model_update':300,'global_reference':300}
    for identity in IDENTITIES:expected_messages.update({'outer_validation_metrics_'+identity:300,'inner_monitor_diagnostic_metrics_'+identity:300})
    if cfg['method']=='caaa':
        expected_messages['sensitivity']=5
        if cfg['use_performance']:expected_messages['inner_monitor_loss']=300
        if cfg['use_climate']:expected_messages['round_climate_strength']=300
    require(dict(counts)==expected_messages,'Protocol message counts differ')
    sizes={m['payload_bytes'] for m in messages if m['kind'] in ('initial_model','global_reference','model_update')};require(len(sizes)==1 and next(iter(sizes))>0,'Full model payload inconsistent')
    for msg in messages:
        kind=msg['kind'];wire=msg['payload_bytes']
        if kind=='train_count':require(wire==8 and msg['direction']=='uplink','Count wire differs')
        if kind.startswith(('outer_validation_metrics_','inner_monitor_diagnostic_metrics_')) or kind in ('sensitivity','round_climate_strength'):require(wire==16 and msg['direction']=='uplink','Metric/climate wire differs')
        if kind=='inner_monitor_loss':require(wire==8 and msg['direction']=='uplink','Monitor loss wire differs')
    scaler=setup['input_scaler'];require(scaler['total_bytes']==sum(m['bytes'] for m in scaler['messages'])==5*(8+32*meta['features']),'Scaler bytes differ')
    require(setup['target_stats_export']['total_bytes']==80,'Target inverse summary differs')
    if cfg['method']=='caaa':require(setup['sensitivity_summary']['additional_charge_bytes']==0,'Sensitivity double charged')
    require(not any(m['direction']=='setup_artifact' for m in messages),'Supplement charged graph artifact')
    return {'optimizer_steps':sum(s['optimizer_steps'] for s in steps),'training_examples':sum(s['examples'] for s in steps),'uplink_bytes':sum(m['payload_bytes'] for m in messages if m['direction']=='uplink'),'downlink_bytes':sum(m['payload_bytes'] for m in messages if m['direction']=='downlink'),'scaler_setup_bytes':scaler['total_bytes'],'target_decode_summary_bytes':80},{'selected_rounds':selected,'message_counts':dict(counts),'ntrain_per_client':ntrain}

def fit_audit(run,evaluation,identity,frozen,plans,report):
    ledger=frame(evaluation/'fit_cost_ledger.csv').set_index('job_sha256');require(ledger.index.is_unique and set(ledger.index)==set(frozen['fit_complete_hashes']) and len(ledger)==90,'Cost ledger duplicated/missing')
    details=[];totals=Counter();hp=identity['primary_choices']
    for job in frozen['jobs']:
        payload={k:v for k,v in job.items() if k not in ('root','package_root','job_sha256')};require(digest(payload)==job['job_sha256'] and payload['run_sha256']==identity['run_sha256'],'Scientific job hash differs')
        case=job['case'];arm=job['arm'];method=job['method'];cfg=job['config']
        coeff=hp['coefficients'] if method=='caaa' else [.3,.3,.4]
        require(cfg['method']==method and cfg['lr']==hp['lr'] and [cfg[k] for k in ('alpha','beta','gamma')]==coeff,'Fit choices differ from primary reuse/canonical unused coefficients')
        require(cfg['prox_mu']==hp['mu'].get(method,.01),'Proximal coefficient differs from primary reuse')
        path=run/'fits'/job['job_sha256'][:16];require(sha(path/'complete.json')==frozen['fit_complete_hashes'][job['job_sha256']],'Fit completion hash changed');receipt=read(path/'complete.json')
        stored={k:v for k,v in receipt['job'].items() if k not in ('root','package_root')};require(stored==job,'Relocatable fit identity differs');hash_tree(path,receipt['files'])
        require(receipt['evaluation_labels_accessed'] is False and receipt['supplement_hyperparameters_retuned'] is False and receipt['cpu_threads']==2,'Fit leakage/retuning/thread cap differs')
        expected,detail=fit_cost(path,receipt,plans[case]['roles']);saved=ledger.loc[job['job_sha256']]
        require(saved['case']==case and saved.arm==arm and saved.method==method and saved.seed==cfg['seed'],'Cost scientific identity differs')
        for key,value in expected.items():scalar_close(value,saved[key],job['job_sha256']+'/'+key);totals[key]+=value
        for key,field in [('wall_seconds','elapsed_wall_seconds'),('cpu_seconds','worker_process_cpu_seconds')]:
            require(math.isfinite(receipt[field]) and receipt[field]>0,'Invalid fit timing');scalar_close(receipt[field],saved[key],'Fit timing '+key)
        validation=frame(path/'validation_predictions.csv');require(not validation.duplicated(['identity','client','target_month']).any(),'Validation keys duplicated')
        for identity_name in IDENTITIES:
            selected=validation[validation.identity==identity_name];require(len(selected)==5*len(plans[case]['roles']['validation']),'Selected validation cohort differs');scores=[]
            for c in CLIENTS:
                g=selected[selected.client==c].sort_values('target_month');require(g.target_month.tolist()==plans[case]['roles']['validation'],'Validation calendar differs');scores.append(regression_reference(g.y,g.prediction)['R2'])
            scalar_close(statistics.fmean(scores),receipt['validation_macro_R2'][identity_name],'Selected validation score')
        details.append({'job_sha256':job['job_sha256'],'case':case,'arm':arm,'method':method,'seed':cfg['seed'],**detail,**expected})
    report['fit_costs']={'unique_fit_bundles':90,'prediction_identities_per_fit':2,'identities_not_double_charged':True,'totals_unique_fits':dict(totals),'fit_details':details,'privacy':'80-byte target inverse summaries and saved q80 exports are authoritative; no secure aggregation/DP claimed'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--primary-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();output=args.output.resolve();require(not output.exists(),'Use a fresh audit output directory');output.mkdir(parents=True)
    report={'status':'RUNNING','audit_code_sha256':sha(__file__),'sections_completed':[]};started=time.perf_counter()
    try:
        evaluation,package,identity,frozen,prepared,completed=gates(run,args.primary_run.resolve(),report);report['sections_completed'].append('completion_code_runtime_frozen_identity')
        plans=source_cohorts(run,package,identity,prepared,report);report['sections_completed'].append('source_only_cohort_derivation')
        thresholds=dataset_statistics(run,plans,report);report['sections_completed'].append('own_TRAIN_scalers_q80_archive_separation')
        observed,arrays,negative=predictions_and_metrics(run,evaluation,plans,thresholds,frozen,completed,report);negative.to_csv(output/'independent_negative_forecast_diagnostics.csv',index=False,float_format='%.17g');report['sections_completed'].append('prediction_keys_scalar_heavy_PR_combined_tables')
        uncertainty(evaluation,plans,observed,arrays,report);report['sections_completed'].append('18_within_case_paired_block_contrasts')
        fit_audit(run,evaluation,identity,frozen,plans,report);report['sections_completed'].append('90_unique_fit_chronology_checkpoints_costs')
        report.update(status='PASS',checked_table_coverage={'cases':3,'models':36,'per_client_scalar_PR_groups':900,'paired_contrasts':18,'paired_bootstrap_draws':108000,'unique_fit_cost_rows':90,'all_completed_evaluation_file_hashes_authenticated':True},elapsed_seconds=time.perf_counter()-started)
    except Exception as error:
        report.update(status='FAIL',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started);write_receipt(output,report);raise
    write_receipt(output,report);print(json.dumps({'status':'PASS','output':str(output),'run_sha256':report['run_sha256']},indent=2))

if __name__=='__main__':main()
