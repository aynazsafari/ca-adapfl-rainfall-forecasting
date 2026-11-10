"""Verify completed fit artifacts and conserved budgets before scoring truth."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import run_experiment as r

def main():
    r.guard();choices=r.load(r.RUN/'choices_frozen.json');jobs=r.load(r.RUN/'jobs_frozen.json')['jobs']
    assert choices['jobs']==jobs and len(jobs)==40
    examples=steps=files=0
    for job in jobs:
        d=r.RUN/'fits'/job['id'];c=r.load(d/'complete.json')
        assert c['job']==job and r.sha(d/'complete.json')==choices['fit_complete_hashes'][job['id']]
        for name,h in c['files'].items():
            assert r.sha(d/name)==h;files+=1
        examples+=c['examples'];steps+=c['optimizer_steps']
    assert examples==10764000 and steps==357300
    r.save(ROOT/'qa/training_completion_review.json',dict(status='PASS',utc=r.now(),new_fits=40,hash_checked_files=files,
        examples=examples,steps=steps,frozen_choices=True,evaluation_labels_scored=(ROOT/'evaluation').exists(),
        scope='Artifact integrity and budget conservation; independent semantic/numeric QA is separate'))
    print('TRAINING_COMPLETION_REVIEW_PASS',files,flush=True)

if __name__=='__main__':main()
