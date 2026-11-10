"""Replay every released neural forecast from saved states, without fitting."""
from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_experiment as r
from caadapfl import neural_revision as n

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'qa/state_prediction_replay.json');args=p.parse_args()
    r.guard();frozen=r.load(r.RUN/'choices_frozen.json');data,schema=r.raw_data(evaluation=True)
    assert frozen['jobs']==r.load(r.RUN/'jobs_frozen.json')['jobs']
    launch=r.load(r.RUN/'evaluation_prelaunch.json')
    assert launch['choices_sha256']==r.sha(r.RUN/'choices_frozen.json')
    assert launch['evaluation_identity_sha256']==r.sha(r.RUN/'evaluation_identity.json')
    sealed=r.load(r.RUN/'evaluation_identity.json')
    for name,h in sealed['files'].items():assert r.sha(ROOT/name)==h
    complete=r.load(ROOT/'evaluation/complete.json')
    for name,h in complete['files'].items():assert r.sha(ROOT/'evaluation'/name)==h
    source=pd.read_csv(ROOT/'evaluation/raw_predictions_ALL_endpoints.csv',float_precision='round_trip')
    neural=source[source.job_id!='NO_FIT']
    assert len(source)==64800 and len(neural)==62400
    assert not neural.duplicated(['job_id','endpoint','client','target_month']).any()
    assert set(neural.job_id)=={j['id'] for j in frozen['jobs']}
    context_ix=[schema['names'].index(x) for x in schema['climate_names']];checked=0;maximum=0.;prob_max=0.
    for job in frozen['jobs']:
        path=r.RUN/'fits'/job['id'];receipt=r.load(path/'complete.json')
        assert r.sha(path/'complete.json')==frozen['fit_complete_hashes'][job['id']]
        for name,h in receipt['files'].items():assert r.sha(path/name)==h
        stats=r.load(path/'scaler_statistics.json')['clients']
        for branch in ['terminal','validation_selected']:
            fit=n.load_fit(path/branch,load_round_history=False)
            for identity in ['global','personal']:
                if (identity=='global' and fit.global_state is None) or (identity=='personal' and not fit.personal_states):continue
                role='fixed60' if branch=='terminal' else 'VAL20_40_60';endpoint=role if identity==fit.config.primary else role+'_personal_diagnostic'
                for c,d in data.items():
                    mean=np.array(stats[c]['input_mean']);sd=np.array(stats[c]['input_std'])
                    b=n.FeatureBatch(((d['x']-mean)/sd).astype(np.float32),((d['climate']-mean[context_ix])/sd[context_ix]).astype(np.float32),d['drivers'],tuple(k+'-01' for k in d['target_month']))
                    pred=fit.predict(c,b,identity)
                    expected=source[(source.job_id==job['id'])&(source.client==c)&(source.endpoint==endpoint)].sort_values('target_month')
                    assert len(expected)==96 and np.array_equal(expected.target_month.to_numpy(),d['target_month']) and (expected.checkpoint_tick==pred.checkpoint_round).all()
                    assert (expected.model==job['label']).all() and (expected.seed==job['seed']).all() and (expected.identity==identity).all()
                    diff=float(np.max(np.abs(pred.rainfall_mm-expected.prediction.to_numpy())));pdiff=float(np.max(np.abs(pred.heavy_probability-expected.heavy_probability.to_numpy())))
                    maximum=max(maximum,diff);prob_max=max(prob_max,pdiff);checked+=len(expected)
                    assert diff==0. and pdiff==0.,(job['label'],job['seed'],c,endpoint,diff,pdiff)
    assert checked==len(neural)==62400
    r.save(args.output,dict(status='PASS',utc=r.now(),new_fits=0,forecasts_replayed=checked,maximum_rainfall_difference_mm=maximum,maximum_heavy_probability_difference=prob_max,
        all_saved_neural_endpoints=True,scope='Exact state/feature replay on the recorded environment; independent mathematical QA is separate'))
    print('STATE_REPLAY_PASS',checked,'forecasts;zero_difference;zero_fits',flush=True)

if __name__=='__main__':main()
