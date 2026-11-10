"""Independent pre-fit source/calendar/TRAIN-array audit; sealed evaluation truth unopened."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

CLIENTS = ['Client1_NW_Arid','Client2_Indo_Gangetic','Client3_NE_Himalaya','Client4_South_Peninsular','Client5_Central_India']
CASES = ['IMD','solar','GLDAS21']
ROLES = {'train':('1983-01','2007-11'),'monitor':('2008-01','2010-11'),'validation':('2011-01','2016-11'),'evaluation':('2017-01','2024-12')}
INDEX_NAMES = ['nino34_anomaly','dmi_hadisst11_anomaly','soi_cpc_standardized','oni_anomaly_3mo','meiv2_bimonthly','amv_ersstv6_global_removed_1951_1980']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def dates(start,end):return [str(p) for p in pd.period_range(start,end,freq='M')]
def offset(month,n):return str(pd.Period(month,freq='M')+n)
def keys(start):return {(c,m) for c in CLIENTS for m in dates(start,'2024-12')}
def source_rows(path,expected):
    """Read keys/raw strings; interpret source numeric values only at/before TRAIN end."""
    with Path(path).open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    keyed={(r['client'],r['month']):r for r in rows}
    assert len(rows)==len(keyed) and set(keyed)==expected
    return keyed
def independent_corr(u,v):
    um=math.fsum(float(x) for x in u)/len(u);vm=math.fsum(float(x) for x in v)/len(v)
    a=[float(x)-um for x in u];b=[float(x)-vm for x in v]
    aa=math.fsum(x*x for x in a);bb=math.fsum(x*x for x in b)
    assert aa>0 and bb>0
    return math.fsum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb)
def header(path,member):
    with ZipFile(path) as z:
        with z.open(member+'.npy') as f:
            version=np.lib.format.read_magic(f)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
    return list(shape),str(dtype)


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();package=Path(a.package_root).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    run=package/'runs/supplements_v2';primary=package/'runs/revision_v2'
    ident,main_ident,prepared=read(run/'run_identity.json'),read(primary/'run_identity.json'),read(run/'prepared.json')
    hp_path=primary/'development/hyperparameters_frozen.json';hp=read(hp_path);manifest=read(package/'data/manifest.json')
    assert digest({k:v for k,v in ident.items() if k!='run_sha256'})==ident['run_sha256']==prepared['run_sha256']
    assert ident['primary_run_sha256']==main_ident['run_sha256']
    assert ident['runtime_environment']==main_ident['runtime_environment']
    assert ident['module_code']==main_ident['code']
    auth_files={}
    for name,expected in ident['module_code'].items():
        path=package/'caadapfl'/(name+'.py');assert sha(path)==expected;auth_files['caadapfl/'+name+'.py']=expected
    assert sha(package/'tools/supplement_runner.py')==ident['launcher_sha256']
    auth_files['tools/supplement_runner.py']=ident['launcher_sha256']
    assert hp['evaluation_labels_accessed'] is False
    assert sha(hp_path)==ident['primary_hyperparameters_sha256']
    assert read(run/'primary_choices_frozen.json')==hp
    expected_choices={'lr':hp['lr'],'mu':{m:hp['mu'][m] for m in ['fedprox','caaa']},'coefficients':hp['coefficients']}
    assert ident['primary_choices']==expected_choices=={'lr':.001,'mu':{'fedprox':.1,'caaa':.1},'coefficients':[1.,0.,0.]}
    assert hp['ablation_coefficients']==[.3,.3,.4] and hp['effective_group_aliases']['TRAIN_SELECTED']=='FULL'
    protocol=read(run/'protocol.json');assert digest(protocol)==ident['protocol_hash']
    assert protocol['fit_count']==90 and protocol['cases']==CASES and protocol['arms']==['REFERENCE','ADDED']
    assert protocol['methods']==['fedavg','fedprox','caaa'] and protocol['seeds']==[42,123,2024,7,999]
    assert (protocol['rounds'],protocol['local_epochs'],protocol['batch_size'],protocol['threads_per_worker'],protocol['max_workers'])==(60,3,32,2,2)
    assert len(protocol['comparisons'])==18 and prepared['model_fits']==prepared['evaluation_predictions']==0
    assert not (run/'fits').exists() and not (run/'evaluation').exists()
    assert manifest['source_gate']['passed'] is True
    for name in ['monthly','indices','manifest']:
        path=package/'data'/(name+('.json' if name=='manifest' else '.csv'))
        actual=sha(path);assert actual==ident['canonical_input_hashes'][name]==main_ident['source'][name+'_sha256']
        auth_files['data/'+path.name]=actual
    verified_prepared={};sealed_truth=[]
    for relative,expected in prepared['files'].items():
        relative=relative.replace('\\','/')
        if relative.endswith('_evaluation_truth.csv'):
            sealed_truth.append({'file':relative,'claimed_sha256':expected,'opened':False,'hash_recomputed':False})
            assert (run/relative).is_file()
        else:
            assert sha(run/relative)==expected;verified_prepared[relative]=expected
    assert len(verified_prepared)==24 and len(sealed_truth)==3
    sources,quality,source_receipts,plans={}, {}, {}, {}
    source_claim_paths={
       'IMD/imd_supplement.csv':'tools/derived/imd/imd_supplement.csv',
       'IMD/imd_supplement_quality.csv':'tools/derived/imd/imd_supplement_quality.csv',
       'IMD/receipt':'tools/derived/imd/manifest.json','IMD/plan':'tools/derived/cohorts/IMD_cohort_plan.json',
       'GLDAS21/gldas21_supplement.csv':'tools/derived/gldas_full/gldas21_supplement.csv',
       'GLDAS21/gldas21_supplement_quality.csv':'tools/derived/gldas_full/gldas21_supplement_quality.csv',
       'GLDAS21/receipt':'tools/derived/gldas_full/manifest.json','GLDAS21/plan':'tools/derived/cohorts_gldas_admitted/GLDAS21_cohort_plan.json',
       'solar/feature_csv':'data/solar_supplement.csv','solar/plan':'tools/derived/cohorts/solar_cohort_plan.json'}
    assert set(source_claim_paths)==set(ident['source_hashes'])
    for claim,relative in source_claim_paths.items():
        assert sha(package/relative)==ident['source_hashes'][claim];auth_files[relative]=ident['source_hashes'][claim]
    starts={'IMD':'1981-01','solar':'1981-01','GLDAS21':'2000-01'}
    source_names={}
    quality_summary={}
    for case in CASES:
        plans[case]=read(package/source_claim_paths[case+'/plan'])
        assert read(run/'cohorts'/(case+'.json'))==plans[case]
        if case=='solar':
            sources[case]=source_rows(package/'data/solar_supplement.csv',keys('1981-01'))
            source_names[case]=['power_solar_mj_m2_day']
            valid={(c,m):m>='1984-01' for c,m in sources[case]}
            assert manifest['units']['power_solar_mj_m2_day'].startswith('MJ m-2 day-1')
            for key,row in sources[case].items():
                available=row['power_solar_mj_m2_day'].strip().lower() not in ['', 'nan','na','inf','-inf','+inf']
                assert available==valid[key]
            quality_summary[case]={'source_rows':2640,'admitted_rows':2460,'structural_missing_rows':180,'first_valid_source':'1984-01'}
        else:
            folder='imd' if case=='IMD' else 'gldas_full';base=package/'tools/derived'/folder
            receipt=read(base/'manifest.json');source_receipts[case]=receipt
            assert receipt['status']==('PASS_WITH_EXPLICIT_INVALID_SOURCE' if case=='IMD' else 'PASS')
            name='imd_supplement.csv' if case=='IMD' else 'gldas21_supplement.csv'
            qname='imd_supplement_quality.csv' if case=='IMD' else 'gldas21_supplement_quality.csv'
            for leaf in [name,qname]:
                claimed=receipt['outputs'][leaf];claimed=claimed['sha256'] if isinstance(claimed,dict) else claimed
                assert sha(base/leaf)==claimed
            sources[case]=source_rows(base/name,keys(starts[case]));quality[case]=source_rows(base/qname,keys(starts[case]))
            source_names[case]=['imd_rainfall_mm'] if case=='IMD' else receipt['met_columns']
            valid={}
            for key,row in quality[case].items():
                admitted=int(row['feature_admitted'])==1
                coverage=float(row['weighted_coverage'])
                assert admitted==(.95<=coverage<=1+1e-10)
                present=all(sources[case][key][n].strip().lower() not in ['','nan','na','inf','-inf','+inf'] for n in source_names[case])
                assert admitted==present;valid[key]=admitted
            bad=[{'client':c,'month':m,'coverage':float(quality[case][(c,m)]['weighted_coverage'])} for (c,m),v in valid.items() if not v]
            if case=='IMD':assert bad==[{'client':CLIENTS[0],'month':'2023-10','coverage':.9306549597623043}]
            else:
                assert not bad and plans[case]['source_quality_receipt_sha256']==sha(base/'manifest.json')
                assert all(int(r['flow_records'])==int(r['expected_records']) for r in quality[case].values())
                assert all(int(r['state_records'])==(int(r['expected_records'])-1 if m=='2000-01' else int(r['expected_records'])) for (c,m),r in quality[case].items())
            quality_summary[case]={'source_rows':len(valid),'admitted_rows':sum(valid.values()),'invalid_rows':bad,
               'minimum_weighted_coverage':min(float(r['weighted_coverage']) for r in quality[case].values())}
        derived=[t for t in dates('1983-01','2024-12') if all(valid.get((c,offset(t,-j-3)),False) for c in CLIENTS for j in range(1,13))]
        plan=plans[case]
        assert plan['status']=='SOURCE_QUALITY_COHORT_DERIVED' and plan['shared_clients']==CLIENTS and plan['extra_lag']==0
        assert plan['target_months']==derived
        eval_months=[m for m in derived if m>='2017-01']
        assert plan['expected_evaluation_months']==eval_months==prepared['cases'][case]['evaluation_months']
        assert eval_months==dates(eval_months[0],eval_months[-1])
        counts={role:sum(lo<=t<=hi for t in derived) for role,(lo,hi) in ROLES.items()}
        assert counts=={k:v for k,v in plan['role_counts_per_client'].items() if k!='embargo'}==prepared['cases'][case]['role_counts']
        assert len(derived)==sum(counts.values())+3==plan['samples_per_client']==prepared['cases'][case]['samples_per_client']
        assert prepared['cases'][case]['supplement_met_columns']==source_names[case]
        quality_summary[case].update(first_target=derived[0],last_target=derived[-1],role_counts=counts,embargo_months=['2007-12','2010-12','2016-12'],
            candidate_samples_including_embargo=len(derived),active_samples_excluding_embargo=sum(counts.values()),evaluation_keys=len(eval_months)*5,source_window='t-15 through t-4 inclusive')
    source_receipt={'status':'PASS','run_sha256':ident['run_sha256'],'primary_run_sha256':main_ident['run_sha256'],
       'primary_choices_reused':expected_choices,'retuning':False,'runtime_identity_equals_primary':True,'authenticated_files':auth_files,
       'prepared_nontruth_files_verified':verified_prepared,'sealed_truth_not_opened':sealed_truth,'source_and_cohort_checks':quality_summary,
       'source_subset_selection_uses_target_values':False,'model_fits_run_by_audit':0}
    save(out/'source_cohort_authentication.json',source_receipt)
    print('Source/cohort/hash/HP PASS; independent TRAIN array/scaler checks follow.',flush=True)
    # Canonical numeric TRAIN history only. Evaluation target values are never converted, displayed or used.
    monthly={};monthly_keys=[];train_source_lines=[]
    with (package/'data/monthly.csv').open(encoding='utf-8',newline='') as f:
        source_header=f.readline();train_source_lines.append(source_header)
        columns=next(csv.reader([source_header]))
        assert columns[:3]==['client','month','rainfall_mm']
        for line in f:
            c,m,_=line.split(',',2);monthly_keys.append((c,m))
            # Future target fields are not CSV-parsed or interpreted. Only key prefixes are retained.
            if m<='2007-11':
                train_source_lines.append(line)
                row=dict(zip(columns,next(csv.reader([line]))))
                monthly[(c,m)]={n:float(v) for n,v in row.items() if n not in ['client','month']}
    assert len(monthly_keys)==len(set(monthly_keys))==2640 and set(monthly_keys)==keys('1981-01')
    # The source contract uses pandas' CSV parser; verify its exact TRAIN target reproduction
    # separately from the independent Python-float source reconstruction above.
    pandas_train_targets=pd.read_csv(io.StringIO(''.join(train_source_lines)),usecols=['client','month','rainfall_mm']).set_index(['client','month'])['rainfall_mm'].to_dict()
    del train_source_lines
    with (package/'data/indices.csv').open(encoding='utf-8',newline='') as f:
        index_strings=list(csv.DictReader(f))
    assert len(index_strings)==528 and len({r['month'] for r in index_strings})==528
    indices={r['month']:{n:float(r[n]) for n in INDEX_NAMES} for r in index_strings if r['month']<='2007-11'}
    del index_strings
    extra_values={case:{key:{n:float(row[n]) for n in source_names[case]} for key,row in rows.items() if key[1]<='2007-11' and all(row[n].strip() for n in source_names[case])} for case,rows in sources.items()}
    main_schema=read(primary/'datasets/MAIN_FULL_schema.json');base_names=main_schema['names'];base_met=main_schema['met_columns']
    assert len(base_names)==43 and len(base_met)==22
    base_cache={}
    def independent_base_row(c,s):
        if (c,s) in base_cache:return base_cache[(c,s)]
        target=pd.Period(s,freq='M')+1;record={}
        for h in [1,2]:
            record[f'season_sin{h}']=float(np.sin(2*np.pi*h*target.month/12));record[f'season_cos{h}']=float(np.cos(2*np.pi*h*target.month/12))
        for lag in [1,2,5,11]:record[f'rain_origin_lag{lag}']=monthly[(c,offset(s,-lag))]['rainfall_mm']
        for w in [3,6,12]:
            v=[monthly[(c,offset(s,-j))]['rainfall_mm'] for j in range(w,0,-1)];mu=math.fsum(v)/w
            record[f'rain_roll{w}_mean']=mu;record[f'rain_roll{w}_std']=math.sqrt(math.fsum((z-mu)**2 for z in v)/w)
        record.update(indices[offset(s,-1)]);record['enso_x_iod']=record['nino34_anomaly']*record['dmi_hadisst11_anomaly']
        record.update({n:monthly[(c,offset(s,-3))][n] for n in base_met})
        result=np.array([record[n] for n in base_names],dtype=np.float64);base_cache[(c,s)]=result;return result
    results,threshold_rows,accessed=[],[],[]
    for case in CASES:
        arm_data={};case_stats={};case_sensitivity={}
        for arm in ['REFERENCE','ADDED']:
            pref=run/'datasets'/f'{case}_{arm}_FULL';schema=read(str(pref)+'_schema.json');names=schema['names'];width=43+(len(source_names[case]) if arm=='ADDED' else 0)
            assert names==base_names+(source_names[case] if arm=='ADDED' else []) and len(names)==width
            assert schema['climate_names']==main_schema['climate_names'] and schema['extra_lag']==0
            assert schema['met_columns']==base_met+(source_names[case] if arm=='ADDED' else [])
            assert {n:schema['support'][n] for n in base_names}==main_schema['support']
            for n in names[43:]:assert schema['support'][n]=={'kind':'archived_regional_month','latest_origin_offset':-3,'source':n}
            plan=plans[case];fit_months=[t for t in plan['target_months'] if any(lo<=t<=hi for role,(lo,hi) in ROLES.items() if role!='evaluation')]
            eval_months=plan['expected_evaluation_months'];clients={};max_source_diff=0.;max_target_diff=0.;max_target_stat_diff=0.;max_sensitivity_diff=0.
            fit_path=Path(str(pref)+'_fit.npz');eval_path=Path(str(pref)+'_evaluation_features.npz')
            with np.load(eval_path,allow_pickle=False) as ar:
                assert not any(k.endswith('::y') or k.endswith('::y_z') or k.endswith('::heavy') for k in ar.files)
                for c in CLIENTS:
                    d=ar[c+'::raw_target_month'];o=ar[c+'::raw_origin_month']
                    assert d.tolist()==eval_months and o.tolist()==[offset(m,-1) for m in eval_months]
                    assert header(eval_path,c+'::raw_x')[0]==[len(eval_months),12,width]
                    assert header(eval_path,c+'::scaled_x')[0]==[len(eval_months),12,width]
                    assert header(eval_path,c+'::raw_climate')[0]==[len(eval_months),7]
                    accessed.extend([f'{case}_{arm}:evaluation:{c}:raw_target_month',f'{case}_{arm}:evaluation:{c}:raw_origin_month'])
            local=read(run/'datasets'/f'{case}_{arm}_local_sensitivity.json')['clients']
            with np.load(fit_path,allow_pickle=False) as ar:
                for c in CLIENTS:
                    all_dates=ar[c+'::raw_target_month'];origins=ar[c+'::raw_origin_month']
                    assert all_dates.tolist()==fit_months and origins.tolist()==[offset(m,-1) for m in fit_months]
                    assert max(all_dates)<='2016-11' and not any(m in all_dates for m in ['2007-12','2010-12','2016-12'])
                    train=all_dates<='2007-11';train_dates=all_dates[train];x=ar[c+'::raw_x'][train].copy();y=ar[c+'::y'][train].copy()
                    scaled=ar[c+'::scaled_x'][train].copy();clim=ar[c+'::raw_climate'][train].copy();scaled_clim=ar[c+'::scaled_climate'][train].copy();driver=ar[c+'::raw_drivers'][train].copy()
                    yz=ar[c+'::y_z'][train].copy();heavy=ar[c+'::heavy'][train].copy()
                    assert x.shape==(plan['role_counts_per_client']['train'],12,width) and np.isfinite(x).all() and np.isfinite(y).all()
                    expected=[]
                    for t in train_dates:
                        sequence=[]
                        for k in range(12,0,-1):
                            s=offset(t,-k);v=independent_base_row(c,s)
                            if arm=='ADDED':v=np.concatenate([v,[extra_values[case][(c,offset(s,-3))][n] for n in source_names[case]]])
                            sequence.append(v)
                        expected.append(sequence)
                    expected=np.array(expected);diff=float(np.max(np.abs(x-expected)));assert diff<=1e-10;max_source_diff=max(max_source_diff,diff)
                    expected_y=np.array([monthly[(c,t)]['rainfall_mm'] for t in train_dates])
                    target_diff=float(np.max(np.abs(y-expected_y)));assert target_diff<=1e-10;max_target_diff=max(max_target_diff,target_diff)
                    assert np.array_equal(y,np.array([pandas_train_targets[(c,t)] for t in train_dates]))
                    cids=[names.index(n) for n in schema['climate_names']];assert np.array_equal(clim,x[:,-1,cids])
                    assert np.array_equal(driver,x[:,:,[names.index('nino34_anomaly'),names.index('dmi_hadisst11_anomaly')]])
                    ym=math.fsum(float(z) for z in y)/len(y);ys=math.sqrt(math.fsum((float(z)-ym)**2 for z in y)/len(y))
                    sorted_y=sorted(float(z) for z in y);q=.8*(len(y)-1);lo=math.floor(q);hi=math.ceil(q);q80=sorted_y[lo]+(q-lo)*(sorted_y[hi]-sorted_y[lo])
                    declared=schema['statistics']['clients'][c]
                    assert declared['role_counts']==prepared['cases'][case]['role_counts']
                    err=max(abs(ym-declared['target_mean']),abs(ys-declared['target_std']),abs(q80-declared['heavy80_threshold_mm']));assert err<=1e-10;max_target_stat_diff=max(max_target_stat_diff,err)
                    assert np.array_equal(ar[c+'::target_stats'],np.array([declared['target_mean'],declared['target_std']]))
                    assert np.array_equal(yz,((y-declared['target_mean'])/declared['target_std']).astype(np.float32))
                    assert np.array_equal(heavy,(y>q80).astype(np.float32))
                    months=np.array([int(t[-2:]) for t in train_dates]);monthly_mean={m:math.fsum(float(z) for z in y[months==m])/int((months==m).sum()) for m in set(months)}
                    anomaly=np.array([float(z)-monthly_mean[m] for z,m in zip(y,months)])
                    sensitivities=[independent_corr(anomaly,driver[:,-1,i]) for i in [0,1]]
                    sensitivity_diff=max(abs(a-b) for a,b in zip(sensitivities,local[c]['sensitivity']));assert sensitivity_diff<=1e-12;max_sensitivity_diff=max(max_sensitivity_diff,sensitivity_diff)
                    assert local[c]['n']==len(y) and local[c]['fit_end']=='2007-11' and local[c]['flags']==[]
                    clients[c]={'dates':train_dates,'x':x,'scaled':scaled,'scaled_climate':scaled_clim,'y':y}
                    threshold_rows.append({'case':case,'arm':arm,'client':c,'train_rows':len(y),'target_mean_mm':declared['target_mean'],'target_std_mm':declared['target_std'],'q80_mm':declared['heavy80_threshold_mm'],'independent_max_abs_stat_error_mm':err})
                    accessed.append(f'{case}_{arm}:fit:{c}:TRAIN-only covariates/labels')
            count=sum(len(d['x']) for d in clients.values());sums=np.zeros(width);squares=np.zeros(width)
            for d in clients.values():
                last=d['x'][:,-1,:];sums+=last.sum(0);squares+=(last*last).sum(0)
            mu=sums/count;sd=np.sqrt(np.maximum(squares/count-mu*mu,0));sd[sd<1e-12]=1
            stored_mu=np.array(schema['statistics']['global_input_mean']);stored_sd=np.array(schema['statistics']['global_input_std'])
            assert count==schema['statistics']['count']==plan['role_counts_per_client']['train']*5
            assert np.array_equal(mu,stored_mu) and np.array_equal(sd,stored_sd)
            for d in clients.values():
                assert np.array_equal(d['scaled'],((d['x']-mu)/sd).astype(np.float32))
                assert np.array_equal(d['scaled_climate'],d['scaled'][:,-1,cids])
            expected_bytes=5*(8+32*width)
            assert schema['statistics']['setup_communication']['total_bytes']==expected_bytes
            arm_data[arm]=clients;case_stats[arm]=schema['statistics'];case_sensitivity[arm]=local
            results.append({'case':case,'arm':arm,'feature_count':width,'climate_context_count':7,'fit_months_per_client':len(fit_months),'evaluation_months_per_client':len(eval_months),
               'train_rows_all_clients':count,'independent_source_feature_comparisons':count*12*width,'max_source_abs_difference':max_source_diff,
               'scaler_count_sum_squares_exact':True,'all_TRAIN_history_float32_scaling_exact':True,'TRAIN_source_target_pandas_parser_exact':True,
               'max_TRAIN_target_python_float_source_difference_mm':max_target_diff,
               'max_target_stat_abs_difference_mm':max_target_stat_diff,'max_sensitivity_abs_difference':max_sensitivity_diff,'input_scaler_setup_bytes':expected_bytes,
               'evaluation_numeric_arrays_read':False,'evaluation_truth_read':False})
        assert case_stats['REFERENCE']['clients']==case_stats['ADDED']['clients'] and case_sensitivity['REFERENCE']==case_sensitivity['ADDED']
        assert case_stats['REFERENCE']['global_input_mean']==case_stats['ADDED']['global_input_mean'][:43]
        assert case_stats['REFERENCE']['global_input_std']==case_stats['ADDED']['global_input_std'][:43]
        for c in CLIENTS:
            r,b=arm_data['REFERENCE'][c],arm_data['ADDED'][c]
            assert np.array_equal(r['dates'],b['dates']) and np.array_equal(r['y'],b['y'])
            assert np.array_equal(r['x'],b['x'][:,:,:43]) and np.array_equal(r['scaled'],b['scaled'][:,:,:43])
        if case!='IMD':assert case_stats['REFERENCE']['global_input_mean']!=main_schema['statistics']['global_input_mean']
    with (out/'TRAIN_target_scaler_q80_checks.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(threshold_rows[0]));w.writeheader();w.writerows(threshold_rows)
    receipt={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'run_sha256':ident['run_sha256'],
       'scope':'Independent pre-fit prepared-source/cohort/TRAIN-array/scaler/q80/sensitivity audit; no fits or evaluation numeric reads',
       'primary_hyperparameters_sha256':sha(hp_path),'source_cohort_receipt_sha256':sha(out/'source_cohort_authentication.json'),
       'cases':results,'matched_arm_base_inputs_and_targets_exact':True,'input_scalers_own_matched_TRAIN':True,'targets_and_q80_own_local_TRAIN':True,
       'independent_source_comparisons':sum(r['independent_source_feature_comparisons'] for r in results),'source_raw_difference_tolerance':1e-10,
       'independent_target_stat_tolerance_mm':1e-10,'independent_sensitivity_tolerance':1e-12,
       'evaluation_numeric_arrays_read':False,'evaluation_truth_files_opened':[],'fits_run':0,'accessed_array_scope':accessed,
       'scientific_limits':['Selected CAAA aggregation weights are [1,0,0]: performance/climate aggregation terms inactive; embedding/adaptive proximity remain configured.',
          'Both arms are refitted on their own matched cohort; no score comparison across different source cohorts is authorized by this audit.',
          'Plan sample counts include three embargo dates, while fit/evaluation archives exclude those labels.',
          'GLDAS native validation and published preceding-3h convention are preserved; original raw time_bnds/custom GEE hour metadata origin remains unverified.',
          'Frozen metadata identities and code match; the launcher also rechecks the current scientific runtime signature before execution.',
          'Absolute original source-path strings in source/cohort receipts are provenance annotations, not loader dependencies; portable CLI uses packaged relative sources.']}
    save(out/'prepared_supplement_QA.json',receipt)
    text=['# Prepared supplement audit','','PASS: all six prepared case/arm inputs satisfy the pre-fit audit. No fits, predictions, evaluation numeric arrays or sealed truth values were read.','',
       '| Case | TRAIN/MON/VAL/EVAL per client | Reference/added channels | First target | Evaluation keys |', '|---|---|---|---|---:|']
    for case in CASES:
        q=quality_summary[case];c=q['role_counts'];text.append(f"| {case} | {c['train']}/{c['monitor']}/{c['validation']}/{c['evaluation']} | 43/{43+len(source_names[case])} | {q['first_target']} | {q['evaluation_keys']} |")
    text+=['','Primary choices reused exactly: LR .001, FedProx/CA mu .1, CAAA coefficients [1,0,0]. No supplemental retuning. Both identities are evaluated later for each of the 90 fits.','',
       'The source-only formula independently derives every target key from all-client valid support t-15 through t-4. IMD excludes 2024-02 through 2024-12; its retained evaluation axis is contiguous through 2024-01.','',
       'Each case/arm input scaler is recomputed from one latest row per gradient-TRAIN sample. All 12 TRAIN history rows and context arrays match the stored float32 transform exactly. Local target scalers, strict q80 labels and deseasonalized climate sensitivities were independently checked.','',
       'Twenty-four nontruth prepared files match their claimed SHA256. Three sealed evaluation truth files were not opened or independently hashed; their existing claimed hashes are recorded separately. Evaluation archive name lists, headers and date keys were checked, without loading numeric evaluation feature arrays.','']
    (out/'README.md').write_text('\n'.join(text),encoding='utf-8')
    save(out/'file_hashes.json',{f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='file_hashes.json'})
    print(json.dumps({'status':'PASS','run_sha256':ident['run_sha256'],'cases':results,'source_cohort_checks':quality_summary,'output':str(out)},indent=2))


if __name__=='__main__':main()
