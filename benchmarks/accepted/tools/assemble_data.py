"""Assemble admitted regional sources; fail before model fitting on source gaps."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CLIENTS=('Client1_NW_Arid','Client2_Indo_Gangetic','Client3_NE_Himalaya','Client4_South_Peninsular','Client5_Central_India')
TARGET=ROOT/'provenance/source_tables/rainfall_panel.csv'
# Frozen target checksum (not a rebuilt target with a new product definition).
TARGET_HASH='5ab390484e367ffc634b11fb900cd9c05134b501de697bce849cc2d5f1b9fb03'
POWER=ROOT/'provenance/source_tables/power_native_region_panel_1981_2024.csv'
INDICES=ROOT/'provenance/source_tables/corrected_indices_with_amv.csv'
SOURCE_MAP={
 'ERA5':{
  'temperature_2m':('era5_t2m_c','degC',1,-273.15),
  'dewpoint_temperature_2m':('era5_d2m_c','degC',1,-273.15),
  'surface_pressure':('era5_sp_hpa','hPa',.01,0),
  'mean_sea_level_pressure':('era5_mslp_hpa','hPa',.01,0),
  'u_component_of_wind_10m':('era5_u10_m_s','m s-1',1,0),
  'v_component_of_wind_10m':('era5_v10_m_s','m s-1',1,0),
  'total_precipitation':('era5_tp_mm','mm month-1',1000,0)},
 'ERA5Land':{
  'temperature_2m':('era5land_t2m_c','degC',1,-273.15),
  'dewpoint_temperature_2m':('era5land_d2m_c','degC',1,-273.15),
  'skin_temperature':('era5land_skin_c','degC',1,-273.15),
  **{f'volumetric_soil_water_layer_{i}':(f'era5land_soil_m3_m3_layer{i}','m3 m-3',1,0) for i in (1,2,3,4)},
  'total_precipitation_sum':('era5land_tp_mm','mm month-1',1000,0),
  'total_evaporation_sum':('era5land_evaporation_mm_signed','mm month-1 (negative evaporation)',1000,0),
  'surface_runoff_sum':('era5land_surface_runoff_mm','mm month-1',1000,0),
  'sub_surface_runoff_sum':('era5land_subsurface_runoff_mm','mm month-1',1000,0)},
 'GLDAS21':{
  'Tair_f_inst':('gldas21_tair_c','degC',1,-273.15),
  'Qair_f_inst':('gldas21_qair_kg_kg','kg kg-1',1,0),
  'Wind_f_inst':('gldas21_wind_m_s','m s-1',1,0),
  **{f'SoilMoi{layer}_inst':(f'gldas21_soil_{layer}_kg_m2','kg m-2',1,0) for layer in ('0_10cm','10_40cm','40_100cm','100_200cm')},
  'Rainf_f_tavg':('gldas21_total_rainfall_mm','mm month-1',1,0),
  'Rainf_tavg':('gldas21_liquid_rainfall_mm','mm month-1',1,0),
  'Evap_tavg':('gldas21_evaporation_mm','mm month-1',1,0),
  'Qs_acc':('gldas21_surface_runoff_mm','mm month-1',1,0),
  'Qsb_acc':('gldas21_subsurface_runoff_mm','mm month-1',1,0)}}
POWER_MAP={'T2M':('power_t2m_c','degC'),'RH2M':('power_rh2m_pct','percent'),'WS10M':('power_ws10m_m_s','m s-1'),'PRECTOTCORR':('power_prectotcorr_mm_day','mm day-1 (rate)')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def era5_admission(source_dir):
    """Reconcile the admitted composition against immutable source bytes."""
    recovery=ROOT/'provenance/era5_recovery'
    paths={'composition':source_dir/'ERA5_retrieval.json',
      'provider':recovery/'derived/provider_admission_receipt.json',
      'native':recovery/'derived/native_recovery_receipt.json',
      'completion':recovery/'recovery_completion_receipt.json',
      'contract':recovery/'frozen_recovery_contract.json',
      'recovered':recovery/'derived/monthly_source_rows.csv',
      'original':source_dir/'era5_chunks/assembled_for_admission.csv',
      'merged':source_dir/'CAAdapFL_ERA5_completed.csv'}
    composition=json.loads(paths['composition'].read_text())
    provider=json.loads(paths['provider'].read_text());native=json.loads(paths['native'].read_text())
    completion=json.loads(paths['completion'].read_text())
    if composition.get('status')!='SOURCE_GATES_PASS' or provider.get('status')!='PASS' or completion.get('status')!='PASS':raise ValueError('ERA5 recovery admission failed')
    required={'raw_ee_sha256':'original','recovered_source_sha256':'recovered','provider_admission_receipt_sha256':'provider','native_recovery_receipt_sha256':'native','frozen_recovery_contract_sha256':'contract','merged_csv_sha256':'merged'}
    if any(composition.get(k)!=sha(paths[v]) for k,v in required.items()):raise ValueError('ERA5 composition lineage changed')
    provider_hashes={'earth_engine_sha256':'original','earthmover_sha256':'recovered','native_receipt_sha256':'native','frozen_contract_sha256':'contract'}
    if any(provider.get('files',{}).get(k)!=sha(paths[v]) for k,v in provider_hashes.items()):raise ValueError('ERA5 provider admission lineage changed')
    if completion.get('receipts',{}).get('derived/provider_admission_receipt.json')!=sha(paths['provider']) or completion.get('receipts',{}).get('derived/native_recovery_receipt.json')!=sha(paths['native']):raise ValueError('ERA5 completion lineage changed')
    months=('2022-04','2022-05','2022-07','2022-08','2024-01','2024-02')
    if tuple(composition.get('recovered_months',()))!=months or completion.get('repair_rows')!=30 or provider.get('control_month')!='2024-03':raise ValueError('ERA5 frozen repair scope changed')
    if not (composition.get('no_statistical_imputation') is True and composition.get('no_monthly_value_clipping') is True and provider.get('no_imputation') is True and native.get('all_native_cells_all_required_hours_valid') is True and native.get('imputed_cells')==0):raise ValueError('ERA5 recovery validity policy failed')
    if native.get('snapshot_id')!='ZFKDHBCTBVHVXM3BQFV0' or native.get('source_expver')!={'1':5139} or any(v!={'0':5139} for v in native.get('source_flags',{}).values()) or len(native.get('source_flags',{}))!=7:raise ValueError('ERA5 native source status changed')
    original=pd.read_csv(paths['original'],float_precision='round_trip').set_index(['client','month'])
    recovered=pd.read_csv(paths['recovered'],float_precision='round_trip').set_index(['client','month'])
    merged=pd.read_csv(paths['merged'],float_precision='round_trip').set_index(['client','month'])
    expected={(c,m) for c in CLIENTS for m in months}
    if set(original.index[original.valid_area_m2<=0])!=expected or len(merged)!=2640 or merged.index.duplicated().any():raise ValueError('ERA5 original rejected keys changed')
    numeric=list(SOURCE_MAP['ERA5'])+['geometry_area_m2','geometry_cells','valid_area_m2','valid_cells','weighted_coverage','expected_records','state_records','flow_records']
    metadata=['source','source_interval_start','source_interval_end']
    for key in merged.index:
        source=recovered if key in expected else original
        if not np.array_equal(merged.loc[key,numeric].to_numpy(float),source.loc[key,numeric].to_numpy(float)) or not merged.loc[key,metadata].equals(source.loc[key,metadata]):raise ValueError('ERA5 source row composition changed')
    return {'composition_receipt_sha256':sha(paths['composition']),'provider_admission_receipt_sha256':sha(paths['provider']),'native_recovery_receipt_sha256':sha(paths['native']),'recovery_completion_receipt_sha256':sha(paths['completion']),'frozen_recovery_contract_sha256':sha(paths['contract']),'original_ee_csv_sha256':sha(paths['original']),'recovered_csv_sha256':sha(paths['recovered']),'merged_csv_sha256':sha(paths['merged']),'earth_engine_months':522,'native_archive_months':6,'repaired_rows':30,'repair_months':months,'snapshot_id':native['snapshot_id'],'control_month':'2024-03','all_seven_variables_replaced_on_repaired_months':True,'no_statistical_imputation':True,'compatibility_scope':'Independent January1981 and March2024 regional controls; no authenticated CDS bitwise equivalence claim'}
def key_check(df,start,end):
    if df.duplicated(['client','month']).any():raise ValueError('Duplicate source key')
    expected={(c,str(m)) for c in CLIENTS for m in pd.period_range(start,end,freq='M')}
    if set(map(tuple,df[['client','month']].to_numpy()))!=expected:raise ValueError('Source calendar/client gap or extra row')
def source_gate(path,source):
    df=pd.read_csv(path);df['month']=df['month'].astype(str)
    key_check(df,'2000-01' if source=='GLDAS21' else '1981-01','2024-12')
    if not (df['source']==source).all():raise ValueError('Wrong source identity')
    if not np.isfinite(df[list(SOURCE_MAP[source])].to_numpy()).all():raise ValueError('Nonfinite required source value')
    geom=json.loads((ROOT/'provenance/validation/local_geometry_receipt.json').read_text())['definitions'][source]
    quality_fields=['valid_area_m2','geometry_area_m2','valid_cells','geometry_cells','weighted_coverage','state_records','flow_records','expected_records']
    if not np.isfinite(df[quality_fields].to_numpy(float)).all():raise ValueError('Nonfinite quality metadata')
    if not ((df.valid_area_m2>0)&(df.geometry_area_m2>0)).all():raise ValueError('Nonpositive source areas')
    for field in ('valid_cells','geometry_cells','state_records','flow_records','expected_records'):
        if not (df[field]==np.floor(df[field])).all():raise ValueError('Fractional native counts')
    for client,part in df.groupby('client'):
        if not (part.geometry_cells==geom['counts'][client]).all():raise ValueError('Frozen source geometry cell count changed')
        if not np.allclose(part.geometry_area_m2,geom['areas'][client],rtol=1e-10,atol=.001):raise ValueError('Frozen source geometry area changed')
    monthkeys=pd.PeriodIndex(df.month,freq='M')
    expected_start=[str(m.start_time.strftime('%Y-%m-%dT%H:%M')) for m in monthkeys]
    expected_end=[str((m+1).start_time.strftime('%Y-%m-%dT%H:%M')) for m in monthkeys]
    if list(df.source_interval_start)!=expected_start or list(df.source_interval_end)!=expected_end:raise ValueError('Source physical calendar interval changed')
    # Broad regional India monthly native-unit checks detect unit/sentinel errors.
    # Small signed archive precipitation artifacts are retained, never clipped.
    bounds={'ERA5':{'temperature_2m':(200,350),'dewpoint_temperature_2m':(150,350),'surface_pressure':(30000,120000),'mean_sea_level_pressure':(85000,115000),'u_component_of_wind_10m':(-80,80),'v_component_of_wind_10m':(-80,80),'total_precipitation':(-.001,10)},
            'ERA5Land':{'temperature_2m':(200,350),'dewpoint_temperature_2m':(150,350),'skin_temperature':(200,350),**{f'volumetric_soil_water_layer_{i}':(0,1) for i in (1,2,3,4)},'total_precipitation_sum':(-.001,10),'total_evaporation_sum':(-10,10),'surface_runoff_sum':(-.001,10),'sub_surface_runoff_sum':(-.001,10)},
            'GLDAS21':{'Tair_f_inst':(200,350),'Qair_f_inst':(0,.1),'Wind_f_inst':(0,80),**{f'SoilMoi{layer}_inst':(0,2000) for layer in ('0_10cm','10_40cm','40_100cm','100_200cm')},**{v:(0,10000) for v in ('Rainf_f_tavg','Rainf_tavg','Evap_tavg','Qs_acc','Qsb_acc')}}}
    for band,(low,high) in bounds[source].items():
        if not df[band].between(low,high).all():raise ValueError(f'{source}/{band}: native-unit/sentinel range gate failed')
    coverage=df['weighted_coverage'].to_numpy(float)
    if not ((coverage>=.95)&(coverage<=1+1e-10)).all():raise ValueError('Required source coverage gate95% failed')
    if not ((df.valid_cells>0)&(df.valid_cells<=df.geometry_cells)).all():raise ValueError('Source native valid-cell count invalid')
    if not np.allclose(df.valid_area_m2/df.geometry_area_m2,coverage,rtol=0,atol=1e-12):raise ValueError('Area/coverage identity failed')
    if source=='ERA5Land':
        if not ((df.state_records==1)&(df.flow_records==1)&(df.expected_records==1)).all():raise ValueError('Monthly asset gap')
    else:
        months=pd.PeriodIndex(df.month,freq='M');expected=months.days_in_month*(8 if source=='GLDAS21' else 24)
        state_expected=expected-(np.asarray(df.month=='2000-01',int) if source=='GLDAS21' else 0)
        if not ((df.expected_records==expected)&(df.flow_records==expected)&(df.state_records==state_expected)).all():raise ValueError('Hourly state/flow interval gap')
    renamed=df[['client','month']].copy()
    for raw,(name,unit,scale,offset) in SOURCE_MAP[source].items():renamed[name]=df[raw]*scale+offset
    return renamed,df
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--with-gldas',action='store_true');args=ap.parse_args()
    if args.output_dir.exists():raise ValueError('Use a new canonical output directory')
    if sha(TARGET)!=TARGET_HASH:raise ValueError('Frozen target changed')
    era5_lineage=era5_admission(args.source_dir)
    monthly=pd.read_csv(TARGET,usecols=['client','month','rainfall_mm']);key_check(monthly,'1981-01','2024-12')
    if not np.isfinite(monthly.rainfall_mm).all() or (monthly.rainfall_mm<0).any():raise ValueError('Target gap')
    sources,quality,hashes={},[],{}
    for source in ('ERA5','ERA5Land')+(('GLDAS21',) if args.with_gldas else ()):
        path=args.source_dir/f'CAAdapFL_{source}_completed.csv'
        sources[source],q=source_gate(path,source);quality.append(q);hashes[source]=sha(path)
        if source!='GLDAS21':monthly=monthly.merge(sources[source],on=['client','month'],validate='one_to_one')
    power=pd.read_csv(POWER);key_check(power,'1981-01','2024-12')
    if not np.isfinite(power[list(POWER_MAP)].to_numpy()).all():raise ValueError('POWER meteorology gap')
    power_receipts=[ROOT/'provenance/source_tables'/name for name in ('power_native_extraction_receipt.json','power_independent_validation.json')]
    power_hash_manifest=ROOT/'provenance/source_tables/power_rebuild_complete_file_hashes.json'
    recorded={r['path']:r['sha256'] for r in json.loads(power_hash_manifest.read_text())}
    for path in [POWER]+power_receipts:
        if recorded.get(path.name)!=sha(path):raise ValueError('POWER validated artifact lineage changed')
    if any(json.loads(p.read_text()).get('status')!='PASS' for p in power_receipts):raise ValueError('POWER independent extraction QA failed')
    for field,limits in {'T2M':(-50,80),'RH2M':(0,100),'WS10M':(0,80),'PRECTOTCORR':(0,1000)}.items():
        if not power[field].between(*limits).all():raise ValueError('POWER native unit/range gate failed')
    solar_available=power.month.ge('1984-01')
    if power.loc[~solar_available,'ALLSKY_SFC_SW_DWN'].notna().any() or not np.isfinite(power.loc[solar_available,'ALLSKY_SFC_SW_DWN']).all():raise ValueError('Solar structural calendar changed')
    if not power.loc[solar_available,'ALLSKY_SFC_SW_DWN'].between(0,100).all():raise ValueError('Solar converted unit/range gate failed')
    power_main=power[['client','month']+list(POWER_MAP)].rename(columns={k:v[0] for k,v in POWER_MAP.items()})
    monthly=monthly.merge(power_main,on=['client','month'],validate='one_to_one').sort_values(['client','month'])
    met=[v[0] for s in ('ERA5','ERA5Land') for v in SOURCE_MAP[s].values()]+[v[0] for v in POWER_MAP.values()]
    if len(met)!=22 or not np.isfinite(monthly[met].to_numpy()).all():raise ValueError('Main feature contract failed')
    index_names=('nino34_anomaly','dmi_hadisst11_anomaly','soi_cpc_standardized','oni_anomaly_3mo','meiv2_bimonthly','amv_ersstv6_global_removed_1951_1980')
    indices=pd.read_csv(INDICES)
    if list(indices.columns)!=['month']+list(index_names):raise ValueError('Corrected index definition/order changed')
    if indices.month.duplicated().any() or list(indices.month)!=list(pd.period_range('1981-01','2024-12',freq='M').astype(str)):raise ValueError('Index support calendar gap')
    if not np.isfinite(indices[list(index_names)].to_numpy(float)).all():raise ValueError('Required index value missing')
    qa=ROOT/'provenance/validation/preflight_acceptance.json'
    target_qa=ROOT/'provenance/validation/upstream_verification_receipt.json'
    if not qa.exists() or not target_qa.exists():raise ValueError('Independent source QA receipt missing')
    accepted=json.loads(qa.read_text());target_accepted=json.loads(target_qa.read_text())
    if accepted.get('native_spatial_reduction_accepted') is not True or target_accepted.get('status')!='pass':raise ValueError('Independent source QA failed')
    geometry_hash='4cdbe7cba953c55e4c6591b779128484c2e203ff0c3433b1826bafca95a4d090'
    if accepted.get('geometry_sha256')!=geometry_hash:raise ValueError('Preflight country geometry changed')
    if target_accepted.get('frozen_inputs',{}).get('panel_sha256')!=TARGET_HASH or target_accepted.get('frozen_inputs',{}).get('country_geometry',{}).get('sha256')!=geometry_hash:raise ValueError('Target upstream QA lineage changed')
    args.output_dir.mkdir(parents=True)
    dest=args.output_dir/'monthly.csv';monthly.to_csv(dest,index=False,float_format='%.17g')
    indices.to_csv(args.output_dir/'indices.csv',index=False,float_format='%.17g')
    if args.with_gldas:sources['GLDAS21'].to_csv(args.output_dir/'gldas21_supplement.csv',index=False,float_format='%.17g')
    solar=power[['client','month','ALLSKY_SFC_SW_DWN','solar_missing_reason']].rename(columns={'ALLSKY_SFC_SW_DWN':'power_solar_mj_m2_day'})
    solar.to_csv(args.output_dir/'solar_supplement.csv',index=False,float_format='%.17g')
    pd.concat(quality,ignore_index=True).to_csv(args.output_dir/'source_quality.csv',index=False,float_format='%.17g')
    units={v[0]:v[1] for mapping in SOURCE_MAP.values() for v in mapping.values()};units.update({v[0]:v[1] for v in POWER_MAP.values()})
    units['power_solar_mj_m2_day']='MJ m-2 day-1 (mean daily irradiation)'
    manifest={'status':'SOURCE_GATES_PASS','source_gate':{'passed':True,'policy':'completecalendar,finitevalues,nativejointareacoverage95%,completephysicaltimeintervalcounts;independentpreflightsaccepted','preflight_receipt_sha256':sha(qa),'target_upstream_receipt_sha256':sha(target_qa),'power_receipt_sha256':{p.name:sha(p) for p in power_receipts},'power_hash_manifest_sha256':sha(power_hash_manifest)},'inputs':{'monthly':{'path':'monthly.csv','sha256':sha(dest)},'indices':{'path':'indices.csv','sha256':sha(args.output_dir/'indices.csv')}},
      'feature_contract':{'met_columns':met,'full_features':43,'climate_features':21,'climate_context_features':7,'climate_features_definition':'CORE14 plus six climate indices and ENSO-times-IOD interaction','core_features':14,'issue_time':'targetmonth firstday00:00UTC','source_support':'rain/indexperiodend target-2;reanalysis/POWER target-4;revisedarchivesavailabilityscenario','embargoed_label_months':['2007-12','2010-12','2016-12']},
      'units':units,'frozen_target_sha256':TARGET_HASH,'source_csv_sha256':hashes,'power_sha256':sha(POWER),
      'country_geometry_sha256':geometry_hash,
      'geometry_policy':'native centers inside frozen country and half-openboxes;full-cellWGS84quadrilateralareas;jointsourcevalidcells;95%minareacoverage;C1/C5overlapretained',
      'structural_sources':{'solar':'1984+ only; separate matched-cohort supplement','GLDAS21':{'period':'2000+ only; separate matched-cohort supplement','admitted':args.with_gldas}},
      'IMD':'One suppliedrawmonthC1Oct2023coverage93.065%fails95%. No synthetic repair; primary excludesIMD. Separate coverage-aware matched sensitivity required.',
      'GLDAS_temporal_note':'Past3h physicalinterval semantics documented byNASAproducer andGooglecatalog;start/endhourassetproperties retained as unexplained ingestionmetadata,raw timebounds notnumericallyverified.',
      'evaluation':'2017-2024 retrospective;previously inspectedperiod,notuntouchedconfirmation',
      'privacy':'Federation simulationplainmodel/summarysharing;nodefinitionalprivacyguarantee'}
    manifest['supplement_inputs']={'solar':{'path':'solar_supplement.csv','sha256':sha(args.output_dir/'solar_supplement.csv')}}
    manifest['source_gate']['era5_recovery']=era5_lineage
    if args.with_gldas:manifest['supplement_inputs']['GLDAS21']={'path':'gldas21_supplement.csv','sha256':sha(args.output_dir/'gldas21_supplement.csv')}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'status':manifest['status'],'main_rows':len(monthly),'met_columns':met,'output':str(args.output_dir)}))
if __name__=='__main__':main()
