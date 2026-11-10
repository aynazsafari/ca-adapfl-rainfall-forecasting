"""Independent output arithmetic/cost QA without redistributing IMD inputs.

Uses frozen independent reference formulas, not production scoring helpers.
This deliberately does NOT claim to rederive IMD source admission/scaling.
"""
from pathlib import Path
import argparse, hashlib, json, sys, datetime
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True

IMD_OMISSIONS={
 'datasets/IMD_ADDED_FULL_fit.npz',
 'datasets/IMD_ADDED_FULL_evaluation_features.npz',
 'datasets/IMD_ADDED_FULL_schema.json',
}

def safe_prepared(run,relative):
 normalized=relative.replace('\\','/')
 if ':' in normalized or normalized.startswith('/'):
  raise ValueError('Nonrelative prepared artifact path')
 path=(run/normalized).resolve()
 if run.resolve() not in path.parents:raise ValueError('Prepared artifact traversal refused')
 return normalized,path

def public_gates(root,base,run,evaluation,q,report):
 complete=q.read(evaluation/'complete.json');identity=q.read(run/'run_identity.json')
 protocol=q.read(run/'protocol.json');frozen=q.read(run/'choices_frozen.json');prepared=q.read(run/'prepared.json')
 payload=dict(identity);payload.pop('run_sha256')
 q.require(q.digest(payload)==identity['run_sha256']==complete['run_sha256']==frozen['run_sha256']==prepared['run_sha256'],'Supplement scientific run identity differs')
 q.require(identity['protocol_hash']==q.digest(protocol),'Supplement protocol changed')
 q.require(protocol['cases']==list(q.CASES) and protocol['arms']==list(q.ARMS) and protocol['methods']==list(q.METHODS) and protocol['seeds']==[42,123,2024,7,999],'Supplement methods/cases/seeds changed')
 q.require(protocol['fit_count']==90 and protocol['rounds']==60 and protocol['local_epochs']==3 and protocol['batch_size']==32 and protocol['threads_per_worker']==2 and protocol['max_workers']==2,'Supplement bounded schedule changed')
 comparisons=[[q.label(m,i,c,'ADDED'),q.label(m,i,c,'REFERENCE')] for c in q.CASES for m in q.METHODS for i in q.IDENTITIES]
 q.require(protocol['comparisons']==comparisons,'Within-case 18-comparison family differs')
 q.require(complete['complete_fits']==90 and complete['fit_choices_or_subsets_selected_from_evaluation'] is False and complete['supplement_retuning'] is False and complete['clipped_predictions'] is False,'Completed supplement policy differs')
 q.require(frozen['hyperparameters_frozen_from_primary'] is True and frozen['all_checkpoints_frozen'] is True and frozen['evaluation_labels_accessed'] is False,'Supplement checkpoint/evaluation sealing differs')
 q.require(prepared['model_fits']==prepared['evaluation_predictions']==0,'Preparation fitted/scored models')
 q.hash_tree(evaluation,complete['files'])
 omitted={}
 for asset in q.read(root/'metadata/release_assets.json')['assets']:
  if asset['target']!='benchmarks/accepted':continue
  for item in asset['omitted_inputs']:
   key=item['path']
   if key.startswith('runs/supplements_v2/'):
    rel=key[len('runs/supplements_v2/'):]
    q.require(rel not in omitted,'Duplicate declared omitted artifact')
    omitted[rel]=item['sha256']
 q.require(set(omitted)==IMD_OMISSIONS,'Exactly the three declared IMD ADDED artifacts must be omitted')
 missing=[]
 for rel,value in prepared['files'].items():
  normalized,target=safe_prepared(run,rel)
  if normalized in omitted:q.require(value==omitted[normalized],'Omitted prepared artifact expected SHA differs')
  if target.is_file():q.require(q.sha(target)==value,'Prepared public artifact changed: '+normalized)
  else:q.require(normalized in omitted,'Unexpected absent prepared artifact: '+normalized);missing.append(normalized)
 q.require(IMD_OMISSIONS.issubset({r.replace('\\','/') for r in prepared['files']}),'Expected omitted artifacts absent from prepared inventory')
 primary=q.read(base/'runs/revision_v2/run_identity.json');hp_path=base/'runs/revision_v2/development/hyperparameters_frozen.json'
 q.require(primary['run_sha256']==identity['primary_run_sha256'],'Primary run linkage changed')
 q.require(q.sha(hp_path)==identity['primary_hyperparameters_sha256']==q.sha(run/'primary_choices_frozen.json'),'Primary frozen HP receipt changed')
 hp=q.read(hp_path);expected_hp={'lr':hp['lr'],'mu':{m:hp['mu'][m] for m in ('fedprox','caaa')},'coefficients':hp['coefficients']}
 q.require(hp['evaluation_labels_accessed'] is False and identity['primary_choices']==expected_hp,'Supplement HP retuned or differs from primary')
 q.require(identity['module_code']==primary['code'] and identity['runtime_environment']==primary['runtime_environment'],'Primary scientific code/environment parity differs')
 for name,value in identity['module_code'].items():q.require(q.sha(base/'caadapfl'/(name+'.py'))==value,'Frozen scientific code changed')
 q.require(q.sha(base/'tools/supplement_runner.py')==identity['launcher_sha256'],'Frozen launcher changed')
 for name,value in identity['canonical_input_hashes'].items():
  q.require(value==primary['source'][name+'_sha256'] and q.sha(base/'data'/(name+'.json' if name=='manifest' else name+'.csv'))==value,'Primary canonical source linkage differs')
 jobs=frozen['jobs'];q.require(len(jobs)==90 and len({j['job_sha256'] for j in jobs})==90,'Fits duplicated or missing')
 q.require({(j['case'],j['arm'],j['method'],j['config']['seed']) for j in jobs}=={(c,a,m,s) for c in q.CASES for a in q.ARMS for m in q.METHODS for s in q.SEEDS},'Case/arm/method/seed schedule differs')
 q.require(set(frozen['fit_complete_hashes'])=={j['job_sha256'] for j in jobs},'Unique completion map differs')
 schedule=q.read(run/'resolved_fit_schedule.json');q.require(schedule['jobs']==jobs and schedule['count']==90 and schedule['retuning'] is False,'Resolved frozen schedule differs')
 report.update(run_sha256=identity['run_sha256'],primary_run_sha256=identity['primary_run_sha256'],missing_declared_inputs=missing,evaluation_complete_sha256=q.sha(evaluation/'complete.json'),protocol_sha256=q.sha(run/'protocol.json'),prepared_sha256=q.sha(run/'prepared.json'),resolved_schedule_sha256=q.sha(run/'resolved_fit_schedule.json'),primary_choices_sha256=q.sha(run/'primary_choices_frozen.json'))
 return complete,identity,frozen,prepared

def public_source_cohorts(base,run,identity,prepared,q,report):
 hashes=identity['source_hashes'];sources={};paths={};plans={}
 directory=base/'tools/derived/gldas_full';meta=q.read(directory/'manifest.json')
 q.require(q.sha(directory/'manifest.json')==hashes['GLDAS21/receipt'] and meta['status']=='PASS','GLDAS source admission receipt differs')
 for leaf in ('gldas21_supplement.csv','gldas21_supplement_quality.csv'):
  claim=meta['outputs'][leaf];value=claim['sha256'] if isinstance(claim,dict) else claim
  q.require(q.sha(directory/leaf)==value==hashes['GLDAS21/'+leaf],'GLDAS source/quality hash differs')
 data=q.frame(directory/'gldas21_supplement.csv').set_index(['client','month']).sort_index();quality=q.frame(directory/'gldas21_supplement_quality.csv').set_index(['client','month']).sort_index()
 expected=pd.MultiIndex.from_product([q.CLIENTS,list(pd.period_range('2000-01','2024-12',freq='M').astype(str))],names=['client','month'])
 q.require(data.index.equals(expected) and quality.index.equals(expected) and data.index.is_unique and quality.index.is_unique,'GLDAS full calendar/grain differs')
 names=meta['met_columns'];q.require(list(data.columns)==names,'GLDAS columns/order differ')
 valid=quality.feature_admitted.eq(1)&quality.weighted_coverage.ge(.95)&quality.weighted_coverage.le(1+1e-10)
 q.require(np.array_equal(valid,np.isfinite(data.to_numpy(float)).all(1)) and valid.all(),'GLDAS source admission differs')
 sources['GLDAS21']=(names,dict(zip(data.index,valid)));paths['GLDAS21']={'receipt':q.sha(directory/'manifest.json'),'feature':q.sha(directory/'gldas21_supplement.csv'),'quality':q.sha(directory/'gldas21_supplement_quality.csv')}
 solar=base/'data/solar_supplement.csv';q.require(q.sha(solar)==hashes['solar/feature_csv'],'Solar source hash differs')
 data=q.frame(solar).set_index(['client','month']).sort_index();expected=pd.MultiIndex.from_product([q.CLIENTS,list(pd.period_range('1981-01','2024-12',freq='M').astype(str))],names=['client','month'])
 q.require(data.index.equals(expected) and data.index.is_unique,'Solar full calendar/grain differs')
 name='power_solar_mj_m2_day';valid=np.isfinite(data[name].to_numpy(float));structural=np.array([m>='1984-01' for c,m in data.index])
 q.require(np.array_equal(valid,structural),'Solar structural1981–83 absence differs')
 manifest=q.read(base/'data/manifest.json');q.require(manifest['supplement_inputs']['solar']['sha256']==q.sha(solar) and manifest['units'][name].startswith('MJ m-2 day-1'),'Solar conversion/source contract differs')
 sources['solar']=([name],dict(zip(data.index,valid)));paths['solar']={'feature':q.sha(solar)}
 for case in q.CASES:
  plan,stored_sha=q.authenticated_plan(run,base,case,hashes[case+'/plan'])
  if case=='IMD':names=['imd_rainfall_mm'];targets=plan['target_months']
  else:names,valid=sources[case];targets=q.derived_targets(valid,q.CLIENTS)
  roles=q.role_calendars(targets)
  q.require(plan['status']=='SOURCE_QUALITY_COHORT_DERIVED' and plan['extra_lag']==0 and plan['shared_clients']==q.CLIENTS,'Cohort source-quality contract differs')
  q.require(plan['target_months']==targets and plan['expected_evaluation_months']==roles['evaluation'],'Cohort/calendar differs')
  q.contiguous_calendar(roles['evaluation']);counts={r:len(k) for r,k in roles.items()}
  q.require({r:plan['role_counts_per_client'][r] for r in roles}==counts and prepared['cases'][case]['role_counts']==counts,'Source cohort role counts differ')
  q.require(prepared['cases'][case]['evaluation_months']==roles['evaluation'] and prepared['cases'][case]['supplement_met_columns']==names and prepared['cases'][case]['samples_per_client']==len(targets),'Prepared matched case differs')
  end='2024-01' if case=='IMD' else '2024-12';q.require(roles['evaluation']==list(pd.period_range('2017-01',end,freq='M').astype(str)),'Frozen evaluation cohort differs')
  plans[case]={'plan':plan,'roles':roles,'names':names};paths.setdefault(case,{}).update(original_cohort_sha256=hashes[case+'/plan'],run_cohort_copy_sha256=stored_sha)
 report['source_cohorts']={c:{'role_counts':{r:len(k) for r,k in plans[c]['roles'].items()},'source_met_columns':plans[c]['names'],'source_admission_rederived':c!='IMD','hashes':paths[c]} for c in q.CASES}
 return plans

def public_dataset_statistics(base,run,plans,q,report):
 thresholds={};records={};canonical=q.frame(base/'data/monthly.csv').set_index(['client','month']).rainfall_mm
 for case in q.CASES:
  roles=plans[case]['roles'];fit_months=sorted(roles['train']+roles['monitor']+roles['validation']);targets=None;sensitivity=None;reference=None;records[case]={}
  for arm in (('REFERENCE',) if case=='IMD' else q.ARMS):
   schema=q.read(run/'datasets'/(case+'_'+arm+'_FULL_schema.json'));names=schema['names'];context=schema['climate_names'];data=q.archive(run,case,arm);evaluation=q.archive(run,case,arm,True)
   q.require(len(names)==(43 if arm=='REFERENCE' else 43+len(plans[case]['names'])) and len(context)==7 and schema['extra_lag']==0,'Feature dimensions/context/latency differ')
   if arm=='ADDED':q.require(names[:43]==reference['names'] and names[43:]==plans[case]['names'],'Added arm changes common inputs')
   for c in q.CLIENTS:
    d=data[c];e=evaluation[c]
    q.require(d['raw_target_month'].tolist()==fit_months and e['raw_target_month'].tolist()==roles['evaluation'],'Fit/evaluation calendar separation differs')
    q.require(np.allclose(d['y'],[canonical.loc[(c,m)] for m in fit_months],rtol=0,atol=1e-10),'Fit labels differ from canonical CHIRPS targets')
    for v in (d,e):
     dates=v['raw_target_month'];q.require(v['raw_origin_month'].tolist()==[str(pd.Period(t,freq='M')-1) for t in dates],'Origin calendar differs')
     q.require(v['raw_x'].shape==(len(dates),12,len(names)) and v['raw_drivers'].shape==(len(dates),12,2) and v['raw_climate'].shape==(len(dates),7),'Array dimensions differ')
     q.require(all(np.isfinite(a).all() for a in v.values() if a.dtype.kind in 'fiu'),'Nonfinite feature/label/scaling arrays')
    if arm=='ADDED':
     for key in ('y','raw_target_month','raw_origin_month','raw_drivers','raw_climate','target_stats','y_z','heavy'):q.require(np.array_equal(d[key],reference['fit'][c][key]),'Matched fit arms differ in labels/axes/context')
     for key in ('raw_target_month','raw_origin_month','raw_drivers','raw_climate','target_stats'):q.require(np.array_equal(e[key],reference['evaluation'][c][key]),'Matched evaluation arms differ in axes/context')
     q.require(np.array_equal(d['raw_x'][:,:,:43],reference['fit'][c]['raw_x']) and np.array_equal(e['raw_x'][:,:,:43],reference['evaluation'][c]['raw_x']),'Matched common raw features differ')
   own=q.own_train_statistics(q.CLIENTS,data,names,context,roles,schema['statistics']);now=own['target_statistics']
   if targets is None:targets=now;thresholds[case]={c:v[2] for c,v in now.items()}
   else:q.require(targets==now,'Arms have different own-TRAIN target statistics/q80')
   mean=np.asarray(schema['statistics']['global_input_mean']);std=np.asarray(schema['statistics']['global_input_std']);ci=[names.index(n) for n in context]
   for c,e in evaluation.items():
    q.require(np.array_equal(e['scaled_x'],((e['raw_x']-mean)/std).astype(np.float32)) and np.array_equal(e['scaled_climate'],((e['raw_climate']-mean[ci])/std[ci]).astype(np.float32)),'Evaluation input scaling differs')
    q.require(np.array_equal(e['target_stats'],data[c]['target_stats']),'Evaluation inverse-target scaling differs')
   local=q.read(run/'datasets'/(case+'_'+arm+'_local_sensitivity.json'))
   for c,v in local['clients'].items():q.require(v['fit_end']=='2007-11' and v['n']==len(roles['train']) and len(v['sensitivity'])==2,'TRAIN sensitivity receipt differs')
   if sensitivity is None:sensitivity=local
   else:q.require(sensitivity==local,'Arms have different TRAIN target sensitivities')
   setup=schema['statistics']['setup_communication'];q.require(setup['total_bytes']==sum(v['bytes'] for v in setup['messages'])==5*(8+32*len(names)),'Own-scaler setup ledger differs')
   records[case][arm]={'features':len(names),'TRAIN_count_all_clients':own['count'],'target_statistics':now,'scaler_source':'Latest-row gradient TRAIN only','q80_source':'Own gradient TRAIN targets','setup_bytes':setup['total_bytes'],'evaluation_archive_has_labels':False}
   if arm=='REFERENCE':reference={'names':names,'fit':data,'evaluation':evaluation}
  if case=='IMD':records[case]['ADDED']={'status':'NOT_RECONSTRUCTED','reason':'Provider-derived source/schema/feature inputs excluded from public assets'}
 report['dataset_statistics']=records
 return thresholds
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.repo_root.resolve();base=root/'benchmarks/accepted';run=base/'runs/supplements_v2';evaluation=run/'evaluation'
 if a.output.exists():raise FileExistsError('Use a fresh output receipt')
 if base in a.output.resolve().parents:raise ValueError('Receipt must be outside accepted evidence')
 sys.path.insert(0,str(base/'tools/qa'));import audit_completed_supplements as q
 report={'status':'RUNNING','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_scope':'Provider-derived IMD features omitted. Protocol, frozen choices, prepared files, primary bindings, output arithmetic and fit/cost authentication checked. Solar/GLDAS source admission and both-arm scaling independently rederived; IMD REFERENCE statistics checked. IMD source admission and ADDED scaler/feature reconstruction are not checked in this mode.','new_fits':0,'checkpoint_inference':False,'helper_sha256':{f.name:q.sha(f) for f in (base/'tools/qa').glob('*.py')}}
 complete,identity,frozen,prepared=public_gates(root,base,run,evaluation,q,report)
 plans=public_source_cohorts(base,run,identity,prepared,q,report)
 thresholds=public_dataset_statistics(base,run,plans,q,report)
 observed,arrays,negative=q.predictions_and_metrics(run,evaluation,plans,thresholds,frozen,complete,report)
 q.uncertainty(evaluation,plans,observed,arrays,report)
 q.fit_audit(run,evaluation,identity,frozen,plans,report)
 report.update(status='PASS_OUTPUT_ARITHMETIC',cases=3,output_labels=36,unique_fits=90,forecasts=83100,contrasts=18,bootstrap_draws=108000,full_source_admission_rechecked=False)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(q.clean(report),indent=2,allow_nan=False),encoding='utf-8')
 print('PASS_OUTPUT_ARITHMETIC: 36 labels, 83100 forecasts, 108000 draws; IMD source reconstruction remains external.')
if __name__=='__main__':main()
