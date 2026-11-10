"""Portable entry point. Accepted science and evidence are never overwritten."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, zipfile

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
ACCEPTED=ROOT/'benchmarks/accepted'
SHARING=ROOT/'benchmarks/sharing'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def safe(base,rel):
 if '\\' in rel or ':' in rel:raise ValueError('Nonportable archive member: '+rel)
 p=(base/rel).resolve()
 if base.resolve() not in p.parents:raise ValueError('Archive traversal rejected')
 return p
def invoke(script,*args,cwd=None):
 env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run([sys.executable,str(script),*map(str,args)],cwd=cwd or ROOT,env=env,check=True)
def invoke_module(module,*args,cwd):
 env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
 subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=cwd,env=env,check=True)
def install(a):
 manifest=load(ROOT/'metadata/release_assets.json')
 # Authenticate ALL supplied archives before any extraction. No remote URLs guessed.
 for entry in manifest['assets']:
  safe(ROOT,entry['target']+'/__destination_check__')
  safe(a.assets,entry['file'])
  p=a.assets/entry['file']
  if not p.is_file():raise FileNotFoundError(f'Missing {p}. Download all six assets from the published Release, or use the supplied local release_assets directory.')
  if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise RuntimeError('Asset authentication failed: '+entry['file'])
 for entry in manifest['assets']:
  base=ROOT/entry['target'];count=0
  with zipfile.ZipFile(a.assets/entry['file']) as z:
   if len(z.infolist())!=entry['members']:raise RuntimeError('Asset member count mismatch')
   seen=set()
   for info in z.infolist():
    if info.is_dir():continue
    if info.filename in seen:raise RuntimeError('Duplicate member')
    seen.add(info.filename)
    if not info.filename.startswith(entry['strip_prefix']):raise RuntimeError('Unexpected member root')
    rel=info.filename[len(entry['strip_prefix']):];target=safe(base,rel)
    # Never replace different content. Compare existing bytes against actual entry.
    if target.exists():
     with z.open(info) as f:h=hashlib.sha256();h.update(f.read())
     if sha(target)!=h.hexdigest():raise RuntimeError('Refusing changed existing evidence: '+str(target))
     continue
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(info) as f,target.open('xb') as g:shutil.copyfileobj(f,g,4<<20)
    count+=1
  print('Installed',entry['file'],count,'new files',flush=True)
 if a.private_inputs:
  permitted={'benchmarks/accepted/'+e['path']:e['sha256'] for e in load(ROOT/'metadata/frozen_source_inventory.json')['omissions'] if 'IMD input' in e['reason']}
  for asset in manifest['assets']:
   for item in asset['omitted_inputs']:
    permitted[asset['target']+'/'+item['path']]=item['sha256']
  with zipfile.ZipFile(a.private_inputs) as z:
   private=json.loads(z.read('PRIVATE_INPUTS_MANIFEST.json'))
   paths=[entry['path'] for entry in private['files']]
   if len(paths)!=len(set(paths)):raise RuntimeError('Duplicate private input identity')
   for entry in private['files']:
    if permitted.get(entry['path'])!=entry['sha256']:raise RuntimeError('Undeclared private input refused: '+entry['path'])
    data=z.read(entry['path'])
    if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise RuntimeError('Private input content hash mismatch')
    p=safe(ROOT,entry['path'])
    if p.exists():
     if sha(p)!=entry['sha256']:raise RuntimeError('Refusing changed existing private input')
     continue
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream:stream.write(data)
  print('Restored local IMD sidecar; do not stage or publish its inputs.')
 print('Results installed. No fitting or scoring occurred.')
def doctor(a):
 import platform,importlib.metadata as md
 expected={'numpy':'2.3.5','pandas':'3.0.1','torch':'2.14.1','scikit-learn':'1.9.1','scipy':'1.18.1'}
 observed={}
 for name in expected:
  try:observed[name]=md.version(name)
  except md.PackageNotFoundError:observed[name]=None
 frozen=load(ROOT/'metadata/frozen_source_inventory.json');bad=[]
 for entry in frozen['files']:
  p=ROOT/entry['path']
  if not p.is_file() or sha(p)!=entry['sha256']:bad.append(entry['path'])
 if bad:raise RuntimeError('Frozen source files missing/changed: '+str(bad[:10]))
 signature_match=None
 if all(observed.values()) and (ROOT/'metadata/reference_runtime_signature.json').is_file():
  sys.path.insert(0,str(ACCEPTED))
  from caadapfl.runner import runtime_environment
  signature_match=runtime_environment()==load(ROOT/'metadata/reference_runtime_signature.json')
 print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'observed':observed,'reference':expected,'full_reference_runtime_signature_match':signature_match,'frozen_source_files_verified':len(frozen['files']),'declared_omissions':len(frozen['omissions']),'IMD_local_inputs_available':(ACCEPTED/'tools/derived/imd/imd_supplement.csv').exists(),'result_directories':{name:(ACCEPTED/'runs'/name/'evaluation/complete.json').is_file() for name in ['revision_v2','supplements_v2','revision_v3']},'sharing_results':(SHARING/'evaluation/complete.json').is_file(),'exact_runtime_required_for_saved_state_replay':True},indent=2))
 if any(observed[n]!=v for n,v in expected.items()):print('Reference-library mismatch. Arithmetic QA may still work; exact replay requires its recorded runtime.')
def verify(a):
 suite=['primary','sources','capacity','sharing'] if a.suite=='all' else [a.suite]
 out=ROOT/'work'/('verify_'+stamp());out.mkdir(parents=True)
 for name in suite:
  if name=='primary':invoke(ACCEPTED/'tools/qa/audit_completed_run.py','--run',ACCEPTED/'runs/revision_v2','--output',out/'primary')
  elif name=='sources':
   required=[ACCEPTED/'tools/derived/imd/manifest.json',ACCEPTED/'runs/supplements_v2/datasets/IMD_ADDED_FULL_fit.npz']
   if all(p.is_file() for p in required):invoke(ACCEPTED/'tools/qa/audit_completed_supplements.py','--run',ACCEPTED/'runs/supplements_v2','--primary-run',ACCEPTED/'runs/revision_v2','--output',out/'sources_full')
   else:invoke(ROOT/'scripts/verify_source_outputs.py','--repo-root',ROOT,'--output',out/'sources_output_only.json')
  elif name=='capacity':invoke(ACCEPTED/'tools/revision_v3_independent_qa.py','--output',out/'capacity')
  else:invoke(SHARING/'tools/independent_qa.py','--root',SHARING,'--output',out/'sharing.json')
 print('Independent arithmetic checks saved under',out)
def replay(a):
 suite=['primary','sources','capacity','sharing'] if a.suite=='all' else [a.suite]
 out=ROOT/'work'/('replay_'+stamp());out.mkdir(parents=True)
 partial=False
 for name in suite:
  if name=='sharing':invoke(SHARING/'tools/replay_predictions.py','--output',out/'sharing.json')
  else:
   command=[sys.executable,str(ROOT/'scripts/replay_saved.py'),'--repo-root',str(ROOT),'--suite',name,'--output',str(out/name)]
   result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'))
   if result.returncode==2 and name=='sources':partial=True
   elif result.returncode:raise RuntimeError('Replay failed: '+name)
 print('State replay receipts:',out)
 if partial:print('PARTIAL: IMD-added inference needs locally acquired native inputs. Remaining scopes replayed.');return 2
 return 0
def train(a):
 frozen=SHARING if a.suite=='sharing' else ACCEPTED
 workspace=a.workdir.resolve()
 if workspace.exists():raise FileExistsError('Use a new nonexistent --workdir')
 if any(p in workspace.parents for p in [ACCEPTED.resolve(),SHARING.resolve()]):raise RuntimeError('Fresh training must stay outside preserved science/evidence roots')
 if not a.execute:
  print('Plan:',a.suite,'fresh checkout at',workspace,'. Add --execute to run empirical fits.');return
 entries=load(ROOT/'metadata/frozen_source_inventory.json')['files']
 for entry in entries:
  p=safe(ROOT,entry['path'])
  if frozen.resolve() in p.resolve().parents and (not p.is_file() or sha(p)!=entry['sha256']):raise RuntimeError('Missing or changed frozen training source: '+entry['path'])
 if a.suite=='sources':
  missing=[r for r in ['data/imd_supplement.csv','data/imd_supplement_quality.csv','tools/derived/imd/imd_supplement.csv','tools/derived/imd/imd_supplement_quality.csv','tools/derived/imd/manifest.json'] if not (ACCEPTED/r).is_file()]
  if missing:raise FileNotFoundError('Source training needs local IMD acquisition/import before replication: '+str(missing))
 workspace.mkdir(parents=True)
 for entry in entries:
  p=ROOT/entry['path']
  if frozen.resolve() in p.resolve().parents:
   target=workspace/p.relative_to(frozen);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 if a.suite=='sources':
  for rel in ['data/imd_supplement.csv','data/imd_supplement_quality.csv','tools/derived/imd/imd_supplement.csv','tools/derived/imd/imd_supplement_quality.csv','tools/derived/imd/manifest.json']:
   if not (ACCEPTED/rel).is_file():raise FileNotFoundError('Source training needs local IMD acquisition/import before replication.')
   target=workspace/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ACCEPTED/rel,target)
  invoke(workspace/'tools/supplement_runner.py','prepare','--package-root',workspace,'--canonical-dir',workspace/'data','--primary-run',ACCEPTED/'runs/revision_v2','--output',workspace/'runs/supplements_v2',cwd=workspace)
  invoke(workspace/'tools/supplement_runner.py','run','--execute','--workers',a.workers,'--package-root',workspace,'--output',workspace/'runs/supplements_v2',cwd=workspace)
  invoke(workspace/'tools/supplement_runner.py','evaluate','--package-root',workspace,'--output',workspace/'runs/supplements_v2',cwd=workspace)
 elif a.suite=='primary':
  args=['--monthly','data/monthly.csv','--indices','data/indices.csv','--manifest','data/manifest.json','--output','runs/revision_v2']
  invoke_module('caadapfl.runner','prepare',*args,cwd=workspace)
  invoke_module('caadapfl.runner','run','--execute','--workers',a.workers,'--pvalues',*args,cwd=workspace)
 elif a.suite=='capacity':
  for action in ['prepare','train','evaluate']:invoke(workspace/'tools/reproduce_revision_v3.py',action,*(['--execute'] if action!='prepare' else []),cwd=workspace)
 else:
  invoke(workspace/'tools/run_experiment.py','qa',cwd=workspace)
  invoke(workspace/'tools/run_experiment.py','prepare',cwd=workspace)
  invoke(workspace/'tools/seal_evaluation.py',cwd=workspace)
  invoke(workspace/'tools/run_experiment.py','train',cwd=workspace)
  invoke(workspace/'tools/verify_training_completion.py',cwd=workspace)
  invoke(workspace/'tools/evaluate_experiment.py',cwd=workspace)
  invoke(workspace/'tools/independent_qa.py','--root',workspace,'--output',workspace/'qa/fresh_independent_numeric_qa.json',cwd=workspace)
def main():
 if not __debug__:raise RuntimeError('Optimized Python disables assertions in preserved scientific modules; use ordinary Python for this benchmark.')
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 sub.add_parser('doctor')
 q=sub.add_parser('install');q.add_argument('--assets',type=Path,required=True);q.add_argument('--private-inputs',type=Path)
 for command in ['verify','replay']:
  q=sub.add_parser(command);q.add_argument('--suite',choices=['all','primary','sources','capacity','sharing'],default='all')
 q=sub.add_parser('figures');q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('tables');q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('train');q.add_argument('--suite',choices=['primary','sources','capacity','sharing'],required=True);q.add_argument('--workdir',type=Path,required=True);q.add_argument('--execute',action='store_true');q.add_argument('--workers',type=int,choices=[1,2],default=2)
 a=p.parse_args()
 if a.command=='doctor':doctor(a)
 elif a.command=='install':install(a)
 elif a.command=='verify':verify(a)
 elif a.command=='replay':return replay(a)
 elif a.command=='figures':invoke(ROOT/'scripts/publication_export.py','--repo-root',ROOT,'--output',a.output.resolve())
 elif a.command=='tables':invoke(ROOT/'scripts/export_tables.py','--repo-root',ROOT,'--output',a.output.resolve())
 else:train(a)
 return 0
if __name__=='__main__':raise SystemExit(main())
