"""Seal the evaluation implementation and analysis plan before evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import run_experiment as r

def main():
    r.guard()
    assert not (r.RUN/'evaluation_identity.json').exists()
    assert not (ROOT/'evaluation').exists() and not (r.RUN/'evaluation_prelaunch.json').exists()
    amendment=r.load(ROOT/'analysis_amendment_v1.json')
    assert amendment['original_protocol_sha256']==r.sha(ROOT/'protocol.json')
    files=['tools/evaluate_experiment.py','protocol.json','analysis_amendment_v1.json','ANALYSIS_ORIGIN.json']
    r.save(r.RUN/'evaluation_identity.json',dict(utc=r.now(),training_identity_sha256=r.sha(r.RUN/'identity.json'),
        files={name:r.sha(ROOT/name) for name in files},new_EVAL_outputs_inspected=False,
        scope='Additional pre-evaluation implementation seal; original training identity and jobs unchanged'))
    print('EVALUATION_IDENTITY_SEALED',flush=True)

if __name__=='__main__':main()
