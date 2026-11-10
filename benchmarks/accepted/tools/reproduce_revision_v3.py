"""Prepare, fit or verify the finite revision study in a fresh repository checkout."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,sys,datetime
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/revision_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def invoke(name,*args):subprocess.run([sys.executable,str(ROOT/'tools'/name),*args],cwd=ROOT,check=True)
def prepare():
 if RUN.exists():raise RuntimeError('runs/revision_v3 already exists. Use a fresh extracted checkout; this command never replaces empirical evidence.')
 source=ROOT/'reproduction_inputs';manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
 for name,h in manifest['files'].items():assert sha(source/name)==h,name
 inputs=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))['inputs']
 for key,name in [('monthly','monthly.csv'),('indices','indices.csv')]:assert sha(ROOT/'data'/name)==inputs[key]['sha256'],name
 target=RUN/'datasets';target.mkdir(parents=True)
 for name in ['MAIN_FULL_fit.npz','MAIN_FULL_evaluation_features.npz','MAIN_FULL_schema.json','MAIN_local_sensitivity.json']:shutil.copyfile(source/name,target/name)
 for name in ['archive_analysis_protocol.json','archive_feature_CI_protocol.json']:shutil.copyfile(source/name,RUN/name)
 archive=ROOT/'runs/accepted_reference';archive.mkdir(parents=True,exist_ok=True)
 if (archive/'raw_predictions.csv').exists():assert sha(archive/'raw_predictions.csv')==manifest['files']['accepted_raw_predictions.csv']
 else:shutil.copyfile(source/'accepted_raw_predictions.csv',archive/'raw_predictions.csv')
 invoke('revision_v3_runner.py','prepare');invoke('revision_implementation_qa.py')
 print('Prepared fresh identity and numerical contracts. Zero empirical fits; evaluation truth remains unused by training.')
def evaluate():
 assert json.loads((RUN/'fit_complete.json').read_text())['status']=='COMPLETE'
 gate=RUN/'evaluation_prelaunch_identity.json'
 if gate.exists():raise RuntimeError('Evaluation gate already exists; inspect/resume the authenticated run manually without overwriting evidence.')
 obj={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'choices_sha256':sha(RUN/'choices_frozen.json'),'archive_protocol_sha256':sha(RUN/'archive_analysis_protocol.json'),'archive_feature_protocol_sha256':sha(RUN/'archive_feature_CI_protocol.json'),'evaluation_code':{name:sha(ROOT/name) for name in ['tools/revision_v3_evaluate.py','caadapfl/evaluation.py']},'independent_qa_code_sha256':sha(ROOT/'tools/revision_v3_independent_qa.py')}
 gate.write_text(json.dumps(obj,indent=2),encoding='utf-8');invoke('revision_v3_evaluate.py');invoke('revision_v3_independent_qa.py')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','train','evaluate','verify']);parser.add_argument('--execute',action='store_true',help='Required for100 empirical fits or opening archived evaluation truth.');a=parser.parse_args()
 if a.action in ('train','evaluate') and not a.execute:parser.error('--execute is required for this action')
 if a.action=='prepare':prepare()
 elif a.action=='train':
  if any((RUN/name).exists() for name in ['fit_complete.json','choices_frozen.json','evaluation/complete.json']):raise RuntimeError('Training is complete; use verify or a fresh checkout. Accepted choices and evaluation bindings must not be rewritten.')
  invoke('revision_v3_runner.py','train')
 elif a.action=='evaluate':evaluate()
 else:
  destination=RUN/'independent_rechecks'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
  invoke('revision_v3_independent_qa.py','--output',str(destination))
