"""Small synthetic fits exercise checkpoint/router I/O, never study data.

This is a scratch verifier, not a public short-run setting. It changes the
synthetic job's architecture/rounds explicitly and labels every result synthetic.
"""
from dataclasses import asdict
from pathlib import Path
import json
import runpy
import shutil
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ns=runpy.run_path(str(HERE/'synthetic_verification.py'))
r,c,out=ns['r'],ns['c'],ns['out']
identity=r.read_json(out/'run_identity.json');identity['run_sha256']=ns['receipt']['run_sha256'];r.save_json(out/'run_identity.json',identity)
checks=[]
def check(name,truth):
    if not truth:raise AssertionError(name)
    checks.append(name)

cfg=r._config('fedavg',42,.001)
cfg.update(rounds=2,local_epochs=1,batch_size=512,hidden=4,embedding=4,attention_hidden=4,layers=1,recurrent_dropout=0.,head_dropout=0.)
job=r._job(out,'neural',config=cfg)
original_loader=r._load_dataset
def guarded_loader(root,scenario,group,evaluation=False):
    if evaluation:raise AssertionError('Fit worker attempted to open evaluation features')
    return original_loader(root,scenario,group,evaluation)
r._load_dataset=guarded_loader
receipt=r._worker(job)
check('synthetic neural worker sees no evaluation archives',receipt['evaluation_labels_accessed'] is False)
check('synthetic neural worker emits both global/personal validation scores',set(receipt['validation_macro_R2'])=={'global','personal'})
path=r.fit_path(out,job['job_sha256'])
check('selected and round checkpoints preserved',(path/'model'/'selected_states.pt').exists() and(path/'model'/'round_states.pt').exists())
check('message and actual-step ledgers saved',(path/'model'/'messages.csv').exists() and(path/'model'/'local_steps.csv').exists())
check('resume returns same fit receipt without rewriting',r._worker(job)==receipt)
jobs=[]
for family in ('Ridge','ET','RF'):
    for scope in ('pooled','local'):
        config=c.ControlConfig(family,scope,'FULL',seed=42,n_estimators=8,min_samples_leaf=2,n_jobs=2)
        small=r._job(out,'classical',config=asdict(config));r._worker(small);jobs.append(small)
router_job=r._job(out,'router',group='FULL',seed=42,expert_jobs=[small['job_sha256'] for small in jobs])
router=r._worker(router_job)
check('router worker preserves genuine OOF provenance',(r.fit_path(out,router_job['job_sha256'])/'training_oof_predictions.csv').exists())
check('router scores all five concrete suites',set(router['validation_macro_R2'])==set(r.ROUTER_SUITES.values()))
oof_receipts=r.read_json(r.fit_path(out,router_job['job_sha256'])/'oof_fit_receipts.json')
check('OOF uses exact same expert configs, including synthetic tree budget',len(oof_receipts)==18 and all(x['config']['n_estimators']==8 for x in oof_receipts))
baseline=r._job(out,'baselines');r._worker(baseline)
check('baseline fit is sealed without evaluation labels',r._complete(out,baseline)['evaluation_labels_accessed'] is False)
r._load_dataset=original_loader

# Exercise all evaluate interfaces on the synthetic fitted model without
# pretending aliases are distinct algorithms or independent training seeds.
# Every label is an explicit fixture; this verifies serialization and pairing.
fitted=r._mods()['neural_benchmark'].load_fit(path/'model')
data,schema=r._load_dataset(out,'MAIN','FULL',evaluation=True)
n=r._mods()['neural_benchmark'];client=next(iter(data));d=data[client]
pred=fitted.predict(client,n.FeatureBatch(d['scaled_x'],d['scaled_climate'],d['drivers'],tuple(k+'-01' for k in d['target_month'])),'global')
check('evaluation inference is label-free with96finite outputs',len(pred.keys)==96 and np.isfinite(pred.rainfall_mm).all() and 'y' not in d)
labels=sorted(set(value for pair in r.protocol()['comparisons'] for value in pair))
outputs=[dict(label=label,identity='global',job_sha256=job['job_sha256'],kind='neural',seed=seed,scenario='MAIN',group='FULL',fixture='SYNTHETIC_ALIAS_ONLY_NO_ALGORITHM_COMPARISON') for label in labels for seed in r.SEEDS]
frozen=dict(run_sha256=identity['run_sha256'],outputs=outputs,baseline_job_sha256=baseline['job_sha256'],all_fit_jobs=[job,baseline],
            fit_complete_hashes={j['job_sha256']:r.file_hash(r.fit_path(out,j['job_sha256'])/'complete.json') for j in (job,baseline)},
            evaluation_labels_accessed=False,choices_including_checkpoints_and_router_weights_frozen=True,label='SYNTHETIC ALIASES; NO SCIENTIFIC RESULTS')
r.save_json(out/'choices_frozen.json',frozen)
evaluation=r.evaluate(out,pvalues=True)
check('all evaluation outputs have a completed hash receipt',evaluation['evaluation_truth_opened_only_after_all_choices'] and evaluation['raw_predictions']>0)
check('evaluation resume validates receipts without rewriting',r.evaluate(out,pvalues=True)==evaluation)
check('all31declared comparison receipts emitted',len(list((out/'evaluation').glob('bootstrap_receipt_*.json')))==31)
moved=out.parent/'relocated';shutil.copytree(out,moved)
relocated_job={**job,'root':str(moved.resolve())}
check('relocated fit resume ignores only runtime root',r._worker(relocated_job)==receipt)
check('relocated receipt preserves original root provenance',r._complete(moved,relocated_job)['original_root']==str(out))
check('relocated preparation validates same input/code identity',r.prepare(ns['panel'],ns['idx'],ns['manifest'],moved)==ns['receipt'])
check('relocated frozen evaluation validates all receipts',r.evaluate(moved,pvalues=True)==evaluation)
changed={**relocated_job,'config':{**relocated_job['config'],'lr':.003}}
try:r._complete(moved,changed)
except ValueError:checks.append('relocation does not hide a changed learning rate')
else:raise AssertionError('Changed configuration incorrectly resumed')
r.immutable_json(moved/'relocation_fixture.json',{'job':job});r.immutable_json(moved/'relocation_fixture.json',{'job':relocated_job})
check('immutable schedules permit relocation without changing original provenance',r.read_json(moved/'relocation_fixture.json')['job']['root']==job['root'])
r.save_json(HERE/'synthetic_only'/'worker_smoke_receipt.json',dict(label='SYNTHETIC ONLY; fixtures alias one tiny2-round model, no scientific results',
           checks_passed=len(checks),checks=checks,temporary_directory=str(out),relocated_directory=str(moved),real_data_reads=0,real_data_fits=0,real_data_scores=0))
print(json.dumps(dict(synthetic_worker_checks=len(checks),directory=str(out))))
