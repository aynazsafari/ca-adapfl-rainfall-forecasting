"""Post-completion descriptive diagnostics; no training, tuning or inference.

Scientific CSVs are opened only after the evaluation completion and independent
QA gates bind the run and evaluation bytes. ONI value bins are not NOAA event
classifications. IMD/CHIRPS agreement is a separate source-product comparison.
"""
from pathlib import Path
import argparse,hashlib,json,math,statistics
import numpy as np
import pandas as pd
from metric_reference import regression_reference

SEEDS=sorted([42,123,2024,7,999])
CLIENTS=sorted(['Client1_NW_Arid','Client2_Indo_Gangetic','Client3_NE_Himalaya','Client4_South_Peninsular','Client5_Central_India'])
MONTHS=list(pd.period_range('2017-01','2024-12',freq='M').astype(str))
BINS=('negative_ONI_value','neutral_ONI_value','positive_ONI_value')
METRICS=('R2','NSE','RMSE_mm','MAE_mm','PBIAS_signed_percent','PBIAS_absolute_percent','forecast_bias_signed_percent','Pearson_r','KGE_2009')
ROLE_PERIODS={'train':('1983-01','2007-11'),'monitor':('2008-01','2010-11'),'validation':('2011-01','2016-11'),'evaluation':('2017-01','2024-12')}
INVALID_IMD={('Client1_NW_Arid','2023-10')}

def require(condition,message):
    if not condition:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def frame(path):return pd.read_csv(path,float_precision='round_trip',dtype={'month':str,'target_month':str})
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def write_json(path,value):Path(path).write_text(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
def keyed_hash(rows):
    h=hashlib.sha256()
    for row in rows:h.update(('\t'.join(map(str,row))+'\n').encode())
    return h.hexdigest()
def safe_relative(base,leaf):
    p=(base/str(leaf).replace('\\','/')).resolve()
    require(p.is_relative_to(base.resolve()),'Receipt path escapes declared source directory')
    return p

def completion_gate(run,imd_directory,expected_imd_manifest_sha256):
    evaluation=run/'evaluation';qa_path=run/'independent_qa/complete.json'
    # No scientific source, prediction, label or score read precedes these checks.
    require((evaluation/'complete.json').is_file(),'Evaluation completion gate absent; no scientific CSV read')
    require(qa_path.is_file(),'Independent QA gate absent; no scientific CSV read')
    completed=read(evaluation/'complete.json');qa=read(qa_path);identity=read(run/'run_identity.json')
    payload=dict(identity);payload.pop('run_sha256')
    require(digest(payload)==identity['run_sha256']==completed['run_sha256']==qa.get('run_sha256'),'Run identity gate differs')
    require(qa.get('status')=='PASS' and bool(qa.get('scope')),'Independent QA is not a scoped PASS')
    require(qa.get('evaluation_complete_sha256')==sha(evaluation/'complete.json'),'QA completion hash differs')
    require(qa.get('checked_files',{}).get('evaluation')==completed['files'],'QA evaluated file set differs')
    require(sha(run/'choices_frozen.json')==completed['choices_frozen_sha256'],'Frozen choices changed')
    frozen=read(run/'choices_frozen.json')
    require(frozen['evaluation_labels_accessed'] is False and completed['evaluation_truth_opened_only_after_all_choices'] is True,'Evaluation sealing receipt differs')
    package=run.parent.parent;data=package/'data'
    for name in ('monthly','indices','manifest'):
        file=data/(name+'.json' if name=='manifest' else name+'.csv')
        require(sha(file)==identity['source'][name+'_sha256'],'Frozen canonical source hash differs')
    for name,expected in identity['code'].items():require(sha(package/'caadapfl'/(name+'.py'))==expected,'Active scientific source differs')
    require(sha(evaluation/'raw_predictions.csv')==completed['files']['raw_predictions.csv'],'Prediction bytes changed after QA')
    manifest_path=imd_directory/'manifest.json'
    require(sha(manifest_path)==expected_imd_manifest_sha256,'Frozen IMD admission manifest hash differs')
    imd=read(manifest_path)
    require(imd['status']=='PASS_WITH_EXPLICIT_INVALID_SOURCE' and imd['unit']=='mm month-1' and imd['minimum_weighted_area_coverage']==.95,'IMD source admission contract differs')
    for leaf in ('imd_supplement.csv','imd_supplement_quality.csv'):
        claim=imd['outputs'][leaf];expected=claim['sha256'] if isinstance(claim,dict) else claim
        require(sha(imd_directory/leaf)==expected,'Admitted IMD source bytes changed')
    manifest=read(data/'manifest.json')
    require(manifest['source_gate']['passed'] is True,'Canonical source gate not PASS')
    for name in ('monthly','indices'):require(manifest['inputs'][name]['sha256']==identity['source'][name+'_sha256'],'Canonical source receipt differs')
    target_path=package/'provenance/source_tables/rainfall_panel.csv'
    require(sha(target_path)==manifest['frozen_target_sha256'],'Frozen CHIRPS quality/source bytes differ')
    evidence={'run_sha256':identity['run_sha256'],'evaluation_complete_sha256':sha(evaluation/'complete.json'),'independent_qa_sha256':sha(qa_path),'raw_predictions_sha256':sha(evaluation/'raw_predictions.csv'),'source_sha256':identity['source'],'CHIRPS_source_quality_sha256':sha(target_path),'imd_manifest_sha256':sha(manifest_path),'imd_output_sha256':{leaf:sha(imd_directory/leaf) for leaf in ('imd_supplement.csv','imd_supplement_quality.csv')}}
    return evaluation,data,target_path,frozen,evidence

def checked_predictions(predictions,frozen):
    keys=['model','seed','client','target_month']
    require(not predictions.duplicated(keys).any(),'Duplicate prediction key')
    require(np.isfinite(predictions[['y','prediction']].to_numpy(float)).all(),'Nonfinite observation/forecast')
    require((predictions.y>=0).all(),'Negative observed rainfall')
    models=sorted(predictions.model.unique())
    labels={o['label'] for o in frozen['outputs']}|{'training_seasonal_mean','available_rainfall_target_minus2','same_calendar_target_minus12'}
    require(set(models)==labels,'Saved evaluated model set differs')
    expected=pd.MultiIndex.from_product([SEEDS,CLIENTS,MONTHS],names=keys[1:]);truth=None
    for model in models:
        part=predictions[predictions.model==model].set_index(keys[1:]).sort_index()
        require(part.index.equals(expected),'All 96dates/5clients/5seeds must be retained for '+model)
        if truth is None:truth=part.y.to_numpy()
        else:require(np.array_equal(truth,part.y.to_numpy()),'Models have different keyed observations')
    truth=truth.reshape(5,5,96)
    require(np.array_equal(truth,np.broadcast_to(truth[:1],truth.shape)),'Observed rainfall differs across seeds')
    return models

def oni_bin(value):
    require(math.isfinite(float(value)),'ONI value is missing')
    return BINS[0] if value<=-.5 else BINS[2] if value>=.5 else BINS[1]

def metric_record(observed,predicted):
    y=list(observed);p=list(predicted)
    if y:metrics=regression_reference(y,p)
    else:metrics={'n':0,**{name:math.nan for name in METRICS}}
    metrics.update({name+'_undefined':not math.isfinite(metrics[name]) for name in METRICS})
    metrics['any_metric_undefined']=any(metrics[name+'_undefined'] for name in METRICS)
    metrics['negative_forecast_n']=sum(float(v)<0 for v in p)
    return metrics

def oni_tables(predictions,indices,frozen):
    models=checked_predictions(predictions,frozen)
    require('oni_anomaly_3mo' in indices and not indices.month.duplicated().any(),'ONI source schema/calendar differs')
    expected=list(pd.period_range('1981-01','2024-12',freq='M').astype(str))
    require(indices.month.tolist()==expected,'Canonical ONI support calendar differs')
    lookup=indices.set_index('month')['oni_anomaly_3mo'];rows=[]
    for target in MONTHS:
        support=str(pd.Period(target,freq='M')-2);value=float(lookup.loc[support])
        rows.append({'target_month':target,'eligible_ONI_support_month':support,'oni_anomaly_3mo':value,'ONI_value_bin':oni_bin(value)})
    mapping=pd.DataFrame(rows)
    annotated=predictions[['model','seed','client','target_month']].merge(mapping,on='target_month',validate='many_to_one').sort_values(['model','seed','client','target_month'])
    require(len(annotated)==len(predictions) and not annotated.ONI_value_bin.isna().any(),'ONI bin assignment lost saved keys')
    joined=predictions.merge(mapping[['target_month','ONI_value_bin']],on='target_month',validate='many_to_one')
    client_rows=[];aggregate_rows=[]
    for model in models:
        for seed in SEEDS:
            for bin_name in BINS:
                part=joined[(joined.model==model)&(joined.seed==seed)&(joined.ONI_value_bin==bin_name)]
                local=[];date_count=int(mapping.ONI_value_bin.eq(bin_name).sum())
                for client in CLIENTS:
                    group=part[part.client==client].sort_values('target_month')
                    require(len(group)==date_count,'Within-bin calendar differs across clients')
                    row={'model':model,'seed':seed,'client':client,'ONI_value_bin':bin_name,'unique_dates':date_count,**metric_record(group.y,group.prediction)}
                    client_rows.append(row);local.append(row)
                pooled=metric_record(part.y,part.prediction)
                macro={'n':len(part),'negative_forecast_n':sum(r['negative_forecast_n'] for r in local)}
                for name in METRICS:
                    undefined=sum(r[name+'_undefined'] for r in local)
                    macro[name]=statistics.fmean(r[name] for r in local) if not undefined else math.nan
                    macro[name+'_undefined']=bool(undefined);macro[name+'_undefined_clients']=undefined
                macro['any_metric_undefined']=any(macro[name+'_undefined'] for name in METRICS)
                for scope,values in [('macro_equal_five_clients',macro),('pooled_all_client_months',pooled)]:
                    aggregate_rows.append({'model':model,'seed':seed,'ONI_value_bin':bin_name,'scope':scope,'unique_dates':date_count,'client_count':5,'clients_with_pairs':5 if date_count else 0,**values})
    aggregate=pd.DataFrame(aggregate_rows);seed_rows=[]
    for (model,bin_name,scope),part in aggregate.groupby(['model','ONI_value_bin','scope'],sort=True):
        row={'model':model,'ONI_value_bin':bin_name,'scope':scope,'paired_seeds':5,'unique_dates':int(part.unique_dates.iloc[0]),'client_count':5}
        for name in METRICS:
            values=part[name].to_numpy(float);defined=np.isfinite(values)
            row[name+'_undefined_seeds']=int((~defined).sum())
            row[name+'_mean']=float(values.mean()) if defined.all() else math.nan
            row[name+'_std']=float(values.std(ddof=1)) if defined.all() else math.nan
        seed_rows.append(row)
    return {'oni_month_bins':mapping,'oni_prediction_key_bins':annotated,'oni_per_client_seed_bin':pd.DataFrame(client_rows),'oni_aggregate_per_seed_bin':aggregate,'oni_five_seed_summary':pd.DataFrame(seed_rows)}

def source_pair_tables(monthly,imd,quality,chirps_quality):
    calendar=list(pd.period_range('1981-01','2024-12',freq='M').astype(str))
    expected=pd.MultiIndex.from_product([CLIENTS,calendar],names=['client','month'])
    frames=[]
    for label,data,required_columns in [('CHIRPS',monthly,['rainfall_mm']),('IMD',imd,['imd_rainfall_mm']),('IMD_quality',quality,['feature_admitted','weighted_coverage','physical_complete_cell_mean_mm','quality_reason','complete_cells']),('CHIRPS_quality',chirps_quality,['rainfall_mm','weighted_coverage','geometry_pixel_count','valid_pixel_count','geometry_area_m2','valid_area_m2'])]:
        require(not data.duplicated(['client','month']).any() and set(required_columns).issubset(data.columns),label+' source grain/schema differs')
        keyed=data.set_index(['client','month']).sort_index();require(keyed.index.equals(expected),label+' full native source calendar differs');frames.append(keyed)
    monthly,imd,quality,chirps_quality=frames
    require(np.isfinite(monthly.rainfall_mm).all() and (monthly.rainfall_mm>=0).all(),'CHIRPS source target missing/negative')
    require(np.allclose(monthly.rainfall_mm,chirps_quality.rainfall_mm,rtol=0,atol=1e-10),'CHIRPS quality source versus canonical target differs')
    require(np.isfinite(chirps_quality[['weighted_coverage','geometry_pixel_count','valid_pixel_count','geometry_area_m2','valid_area_m2']].to_numpy(float)).all(),'Missing CHIRPS spatial quality metadata')
    require(chirps_quality.weighted_coverage.between(.95,1+1e-10).all(),'Frozen CHIRPS spatial coverage gate differs')
    valid=quality.feature_admitted.eq(1)&quality.weighted_coverage.ge(.95)&quality.weighted_coverage.le(1+1e-10)
    require(np.array_equal(valid.to_numpy(),np.isfinite(imd.imd_rainfall_mm)),'IMD admission versus NA pattern differs')
    require(set(quality.index[~valid])==INVALID_IMD,'Prespecified invalid IMD key changed')
    require(np.isfinite(quality.weighted_coverage).all() and np.isfinite(quality.physical_complete_cell_mean_mm).all(),'Missing physical IMD quality metadata')
    require((quality.physical_complete_cell_mean_mm>=0).all(),'Invalid physical IMD rainfall')
    require(np.array_equal(imd.loc[valid,'imd_rainfall_mm'].to_numpy(),quality.loc[valid,'physical_complete_cell_mean_mm'].to_numpy()),'Admitted versus physical IMD source value differs')
    pairs=[];client_rows=[];aggregate_rows=[]
    for role,(start,end) in ROLE_PERIODS.items():
        dates=list(pd.period_range(start,end,freq='M').astype(str));role_records=[]
        for client in CLIENTS:
            for month in dates:
                key=(client,month);admitted=bool(valid.loc[key])
                row={'role':role,'client':client,'target_month':month,'CHIRPS_rainfall_mm':float(monthly.loc[key,'rainfall_mm']),'CHIRPS_weighted_coverage':float(chirps_quality.loc[key,'weighted_coverage']),'CHIRPS_geometry_pixel_count':int(chirps_quality.loc[key,'geometry_pixel_count']),'CHIRPS_valid_pixel_count':int(chirps_quality.loc[key,'valid_pixel_count']),'CHIRPS_geometry_area_m2':float(chirps_quality.loc[key,'geometry_area_m2']),'CHIRPS_valid_area_m2':float(chirps_quality.loc[key,'valid_area_m2']),'IMD_admitted_rainfall_mm':float(imd.loc[key,'imd_rainfall_mm']),'IMD_physical_complete_cell_mean_mm':float(quality.loc[key,'physical_complete_cell_mean_mm']),'IMD_weighted_coverage':float(quality.loc[key,'weighted_coverage']),'IMD_complete_cells':int(quality.loc[key,'complete_cells']),'IMD_feature_admitted':admitted,'pair_included':admitted,'exclusion_reason':str(quality.loc[key,'quality_reason']) if not admitted else ''}
                pairs.append(row);role_records.append(row)
        role_data=pd.DataFrame(role_records);locals=[]
        for client in CLIENTS:
            group=role_data[role_data.client==client];admitted=group[group.pair_included]
            row={'role':role,'client':client,'raw_same_month_pair_keys':len(group),'valid_pair_n':len(admitted),'excluded_invalid_IMD_n':len(group)-len(admitted),'missing_CHIRPS_n':0,'missing_admitted_IMD_n':int(group.IMD_admitted_rainfall_mm.isna().sum()),**metric_record(admitted.CHIRPS_rainfall_mm,admitted.IMD_admitted_rainfall_mm)}
            client_rows.append(row);locals.append(row)
        included=role_data[role_data.pair_included];pooled=metric_record(included.CHIRPS_rainfall_mm,included.IMD_admitted_rainfall_mm)
        macro={'n':len(included),'negative_forecast_n':0}
        for name in METRICS:
            count=sum(r[name+'_undefined'] for r in locals)
            macro[name]=statistics.fmean(r[name] for r in locals) if not count else math.nan
            macro[name+'_undefined']=bool(count);macro[name+'_undefined_clients']=count
        macro['any_metric_undefined']=any(macro[name+'_undefined'] for name in METRICS)
        for scope,values in [('macro_equal_five_clients',macro),('pooled_all_valid_client_months',pooled)]:
            # Regression-reference names describe CHIRPS-as-reference arithmetic,
            # not a model forecast; rename the generic negative-count field.
            values=dict(values);values['negative_IMD_source_n']=values.pop('negative_forecast_n')
            aggregate_rows.append({'role':role,'scope':scope,'client_count':5,'raw_same_month_pair_keys':len(role_data),'valid_pair_n':len(included),'excluded_invalid_IMD_n':len(role_data)-len(included),**values})
    client=pd.DataFrame(client_rows).rename(columns={'negative_forecast_n':'negative_IMD_source_n'})
    return {'CHIRPS_IMD_same_month_pairs':pd.DataFrame(pairs),'CHIRPS_IMD_agreement_per_client_role':client,'CHIRPS_IMD_agreement_macro_pooled_role':pd.DataFrame(aggregate_rows)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--imd-dir',type=Path,required=True);parser.add_argument('--imd-manifest-sha256',required=True)
    args=parser.parse_args();run=args.run.resolve();output=args.output.resolve();imd_dir=args.imd_dir.resolve()
    require(not output.exists(),'Use a fresh diagnostics output directory')
    evaluation,data,target_path,frozen,evidence=completion_gate(run,imd_dir,args.imd_manifest_sha256)
    # Scientific CSV reads begin only after completion_gate returns.
    predictions=frame(evaluation/'raw_predictions.csv');indices=frame(data/'indices.csv');monthly=frame(data/'monthly.csv')
    imd=frame(imd_dir/'imd_supplement.csv');quality=frame(imd_dir/'imd_supplement_quality.csv');chirps_quality=frame(target_path)
    tables={**oni_tables(predictions,indices,frozen),**source_pair_tables(monthly,imd,quality,chirps_quality)}
    output.mkdir(parents=True)
    for name,table in tables.items():table.to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
    oni_keys=tables['oni_prediction_key_bins']
    bins=tables['oni_month_bins'];pairs=tables['CHIRPS_IMD_same_month_pairs']
    receipt={'status':'PASS','label':'POST_COMPLETION_DESCRIPTIVE_DIAGNOSTICS','gates':evidence,'scientific_core_changed':False,'new_fits_inference_or_pvalues':False,'ONI':{'definition':'Eligible oni_anomaly_3mo at target minus two months; revised archived value','bins':{'negative_ONI_value':'value <= -0.5','neutral_ONI_value':'-0.5 < value < +0.5','positive_ONI_value':'value >= +0.5'},'event_classification':False,'NOAA_five_overlapping_season_persistence_applied':False,'eligible_months':96,'bin_date_counts':bins.ONI_value_bin.value_counts().reindex(BINS,fill_value=0).to_dict(),'all_saved_prediction_keys_retained':len(oni_keys),'prediction_key_calendar_sha256':keyed_hash(oni_keys[['model','seed','client','target_month']].itertuples(index=False,name=None)),'macro':'Equal five clients, then separate five-seed summary; any undefined constituent keeps macro undefined','pooled':'All within-bin client-month pairs; separate from macro','no_new_inference_family':True},'CHIRPS_IMD':{'interpretation':'Same-month product agreement with CHIRPS as numerical reference; neither independent validation nor forecast evaluation','source_grids_and_inputs':'Same frozen country-center domain and native WGS84 cell-area protocol; each source spatial coverage retained','product_dependence':'CHIRPS station blended and IMD gauge gridded; agreement does not establish statistical independence or target-free verification','units':'mm month-1','roles':ROLE_PERIODS,'boundary_December_labels_excluded_from_roles':['2007-12','2010-12','2016-12'],'earlier_native_months_outside_target_roles':'1981-01 through1982-12','raw_pair_keys':len(pairs),'valid_pairs':int(pairs.pair_included.sum()),'excluded_pairs':pairs.loc[~pairs.pair_included,['role','client','target_month','exclusion_reason']].to_dict('records'),'direct_source_exclusion_month':'Client1/2023-10 only','distinct_forecasting_IMD_sensitivity_exclusion':'2024-02 through2024-12 due to target-4 availability and12-step history; not applied to this source-product comparison','forecast_cohort_selection_used':False,'invalid_IMD_filled':False,'PBIAS_definition':'100*sum(CHIRPS-IMD)/sum(CHIRPS); positive smaller IMD values','R2_NSE_interpretation':'Descriptive squared-difference agreement relative to CHIRPS variance; no predictive-skill claim'},'undefined_values':'CSV NaN retained with explicit per-metric undefined flags; JSON undefined numbers encoded null','files':{name+'.csv':{'rows':len(table),'sha256':sha(output/(name+'.csv'))} for name,table in tables.items()},'diagnostic_code_sha256':sha(__file__),'metric_reference_sha256':sha(Path(__file__).with_name('metric_reference.py'))}
    write_json(output/'complete.json',receipt)
    print(json.dumps({'status':'PASS','output':str(output),'tables':len(tables)},indent=2))

if __name__=='__main__':main()
