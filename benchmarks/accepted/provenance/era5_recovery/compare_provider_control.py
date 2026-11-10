"""Read-only frozen March2024 provider admission check, no forecast outcomes."""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
import pandas as pd
from recover_selected_months import ROOT,NATIVE,VARS,TOLERANCES,sha

def main(ee_path=None):
    workspace=ROOT.parents[2]
    ee=Path(ee_path).resolve() if ee_path else workspace/'work/ee_rebuild/completed_raw/era5_chunks/assembled_for_admission.csv'
    aws=ROOT/'derived/monthly_source_rows.csv';receipt=ROOT/'derived/native_recovery_receipt.json'
    frozen=json.loads((ROOT/'frozen_recovery_contract.json').read_text());native=json.loads(receipt.read_text())
    if frozen['provider_compatibility_absolute_tolerances_native_units']!=TOLERANCES:raise ValueError('Frozen tolerances changed')
    if native['status']!='NATIVE_RECOVERY_PASS_COMPATIBILITY_PENDING':raise ValueError('Native source receipt not ready')
    output_hashes={k.replace('\\','/'):v for k,v in native['output_files'].items()}
    if output_hashes['derived/monthly_source_rows.csv']!=sha(aws.read_bytes()):raise ValueError('Native source output hash changed')
    a=pd.read_csv(aws).set_index(['client','month']);b=pd.read_csv(ee).set_index(['client','month'])
    comparisons=[];gates=[]
    for c in frozen['clients']:
        ar=a.loc[(c,'2024-03')];br=b.loc[(c,'2024-03')]
        if not isinstance(ar,pd.Series) or not isinstance(br,pd.Series):raise ValueError('Duplicate source control row')
        for var in VARS:
            field=NATIVE[var];difference=abs(float(ar[field])-float(br[field]));passed=difference<=TOLERANCES[var]
            comparisons.append({'client':c,'month':'2024-03','variable':var,'native_band':field,'earthmover_value':float(ar[field]),'earth_engine_value':float(br[field]),'absolute_difference':difference,'absolute_tolerance':TOLERANCES[var],'pass':passed})
        for field in ['expected_records','state_records','flow_records','geometry_cells','valid_cells']:
            gates.append({'client':c,'field':field,'pass':float(ar[field])==float(br[field]),'earthmover':float(ar[field]),'earth_engine':float(br[field])})
        for field in ['geometry_area_m2','valid_area_m2']:
            gates.append({'client':c,'field':field,'pass':math.isclose(float(ar[field]),float(br[field]),rel_tol=1e-10,abs_tol=.001),'earthmover':float(ar[field]),'earth_engine':float(br[field])})
        for field in ['weighted_coverage','state_temporal_coverage']:
            gates.append({'client':c,'field':field,'pass':abs(float(ar[field])-float(br[field]))<=1e-10,'earthmover':float(ar[field]),'earth_engine':float(br[field])})
        for field in ['source_interval_start','source_interval_end','source']:
            gates.append({'client':c,'field':field,'pass':str(ar[field])==str(br[field]),'earthmover':str(ar[field]),'earth_engine':str(br[field])})
    pd.DataFrame(comparisons).to_csv(ROOT/'derived/march2024_provider_compatibility.csv',index=False,float_format='%.17g')
    pd.DataFrame(gates).to_csv(ROOT/'derived/march2024_metadata_compatibility.csv',index=False)
    passed=all(c['pass'] for c in comparisons) and all(g['pass'] for g in gates)
    result={'status':'PASS' if passed else 'FAIL','control_month':'2024-03','native_comparisons':35,'metadata_comparisons':len(gates),
      'frozen_tolerances':TOLERANCES,'maximum_native_absolute_differences_by_variable':{v:max(c['absolute_difference'] for c in comparisons if c['variable']==v) for v in VARS},
      'files':{'earth_engine_table':str(ee),'earth_engine_sha256':sha(ee.read_bytes()),'earthmover_table':str(aws),'earthmover_sha256':sha(aws.read_bytes()),'native_receipt_sha256':sha(receipt.read_bytes()),'frozen_contract_sha256':sha((ROOT/'frozen_recovery_contract.json').read_bytes())},
      'source_repair_scope':'Six rejected ERA5 months, all seven variables; March2024 is verification control only',
      'cause_verified':'Earth Engine six rejected months have21 missing tp validity records in total; state bands complete',
      'numeric_difference_explanation':'Small cross-provider differences are measured; no proven encoding cause and no bitwise equivalence claim',
      'no_imputation':True,'historical_availability':'Retrospective final reanalysis, no historical operational vintage reconstruction'}
    (ROOT/'derived/provider_admission_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not passed:raise SystemExit('Mixed-provider admission FAIL; leave original EE rows rejected')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ee-table',type=Path);args=ap.parse_args();main(args.ee_table)
