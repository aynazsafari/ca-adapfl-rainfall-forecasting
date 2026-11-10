"""Frozen endpoint evaluation, paired factorial contrasts and auditable cost."""
from pathlib import Path
import itertools,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_experiment as r
from caadapfl import neural_revision as n, evaluation as e
DEST=ROOT/'evaluation'
METRICS=['macro_MSE_mm2','macro_RMSE_mm','macro_R2','macro_MSESS']

def metrics(y,p,clim):
    # p [model,seed,client,date], y/clim [client,date]. Clients are fixed.
    err=p-y[None,None,:,:];mse=np.mean(err*err,axis=-1)
    den=np.sum((y-y.mean(axis=-1,keepdims=True))**2,axis=-1)
    cden=np.mean((y-clim)**2,axis=-1)
    assert np.all(den>0) and np.all(cden>0),'Undefined sampled skill denominator'
    return np.stack([mse.mean(axis=(1,2)),np.sqrt(mse).mean(axis=(1,2)),
        (1-np.sum(err*err,axis=-1)/den).mean(axis=(1,2)),(1-mse/cden).mean(axis=(1,2))],axis=-1)

def make_predictions():
    r.guard();assert r.load(r.RUN/'fit_complete.json')['status']=='COMPLETE'
    assert not DEST.exists(),'Do not replace a partial/accepted evaluation'
    frozen=r.load(r.RUN/'choices_frozen.json');assert frozen['jobs']==r.load(r.RUN/'jobs_frozen.json')['jobs']
    amendment=r.load(ROOT/'analysis_amendment_v1.json')
    assert amendment['original_protocol_sha256']==r.sha(ROOT/'protocol.json') and not amendment['new_EVAL_outputs_inspected']
    sealed=r.load(r.RUN/'evaluation_identity.json')
    assert sealed['training_identity_sha256']==r.sha(r.RUN/'identity.json')
    for name,h in sealed['files'].items():assert r.sha(ROOT/name)==h,name
    assert not (r.RUN/'evaluation_prelaunch.json').exists(),'Do not overwrite an evaluation launch receipt'
    r.save(r.RUN/'evaluation_prelaunch.json',dict(utc=r.now(),evaluator_sha256=r.sha(__file__),choices_sha256=r.sha(r.RUN/'choices_frozen.json'),protocol_sha256=r.sha(ROOT/'protocol.json'),analysis_amendment_sha256=r.sha(ROOT/'analysis_amendment_v1.json'),evaluation_identity_sha256=r.sha(r.RUN/'evaluation_identity.json')))
    DEST.mkdir()
    train,schema=r.raw_data();eval_data,_=r.raw_data(evaluation=True)
    panel=pd.read_csv(ROOT/'datasets/monthly.csv',float_precision='round_trip').set_index(['client','month'])['rainfall_mm']
    clim={};q80={};rows=[];train_stats=[];cost=[]
    for c,d in train.items():
        mask=d['target_month']<='2007-11';y=d['y'][mask];keys=d['target_month'][mask];q80[c]=float(np.quantile(y,.8))
        for m in range(1,13):
            vals=y[np.array([int(k[-2:])==m for k in keys])];clim[c,m]=float(vals.mean())
            train_stats.append(dict(client=c,calendar_month=m,TRAIN_count=len(vals),climatology_mm=clim[c,m],q80_mm=q80[c]))
    for job in frozen['jobs']:
        fitdir=r.RUN/'fits'/job['id'];assert r.sha(fitdir/'complete.json')==frozen['fit_complete_hashes'][job['id']]
        receipt=r.load(fitdir/'complete.json')
        for name,h in receipt['files'].items():assert r.sha(fitdir/name)==h,name
        scalers=r.load(fitdir/'scaler_statistics.json')['clients'];context_ix=[schema['names'].index(k) for k in schema['climate_names']]
        for branch in ['terminal','validation_selected']:
            fit=n.load_fit(fitdir/branch,load_round_history=False)
            for identity in ['global','personal']:
                if identity=='global' and fit.global_state is None:continue
                if identity=='personal' and not fit.personal_states:continue
                role='fixed60' if branch=='terminal' else 'VAL20_40_60'
                endpoint=role if identity==fit.config.primary else role+'_personal_diagnostic'
                for c,d in eval_data.items():
                    s=scalers[c];mean=np.array(s['input_mean']);sd=np.array(s['input_std'])
                    features=n.FeatureBatch(((d['x']-mean)/sd).astype(np.float32),((d['climate']-mean[context_ix])/sd[context_ix]).astype(np.float32),d['drivers'],tuple(k+'-01' for k in d['target_month']))
                    pred=fit.predict(c,features,identity)
                    for month,pp,prob in zip(d['target_month'],pred.rainfall_mm,pred.heavy_probability):
                        rows.append(dict(endpoint=endpoint,model=job['label'],seed=job['seed'],client=c,target_month=month,y=float(panel.loc[c,month]),prediction=float(pp),
                            climatology_mm=clim[c,int(month[-2:])],q80_mm=q80[c],heavy_probability=float(prob),checkpoint_tick=pred.checkpoint_round,job_id=job['id'],identity=identity))
        cost.append(dict(model=job['label'],seed=job['seed'],job_id=job['id'],optimizer_steps=receipt['optimizer_steps'],examples=receipt['examples'],**receipt['cost']))
    # Fixed climatology is repeated only for paired seed bookkeeping.
    for seed in r.load(ROOT/'protocol.json')['seeds']:
        for c,d in eval_data.items():
            for month in d['target_month']:
                rows.append(dict(endpoint='fixed60',model='TRAIN_CLIMATOLOGY',seed=seed,client=c,target_month=month,y=float(panel.loc[c,month]),prediction=clim[c,int(month[-2:])],
                    climatology_mm=clim[c,int(month[-2:])],q80_mm=q80[c],heavy_probability=np.nan,checkpoint_tick=0,job_id='NO_FIT',identity='fixed_reference'))
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'raw_predictions_ALL_endpoints.csv',index=False,float_format='%.17g')
    primary=frame[frame.endpoint=='fixed60'].copy();e.validate_prediction_frame(primary,expected_months=list(pd.period_range('2017-01','2024-12',freq='M').astype(str)),expected_seeds=r.load(ROOT/'protocol.json')['seeds'])
    assert len(primary)==21600 and primary[['client','target_month']].drop_duplicates().shape[0]==480
    primary.to_csv(DEST/'raw_predictions_PRIMARY.csv',index=False,float_format='%.17g')
    pd.DataFrame(train_stats).to_csv(DEST/'TRAIN_climatology.csv',index=False,float_format='%.17g')
    pd.DataFrame(cost).to_csv(DEST/'fit_cost_40.csv',index=False,float_format='%.17g')
    return frame,primary

def summaries(frame):
    rows=[]
    for (endpoint,model,seed,client),g in frame.groupby(['endpoint','model','seed','client']):
        y=g.y.to_numpy();p=g.prediction.to_numpy();clim=g.climatology_mm.to_numpy();err=p-y;tail=y>g.q80_mm.to_numpy()
        rows.append(dict(endpoint=endpoint,model=model,seed=int(seed),client=client,n=len(y),MSE_mm2=float(np.mean(err*err)),
            RMSE_mm=float(np.sqrt(np.mean(err*err))),R2=float(1-np.sum(err*err)/np.sum((y-y.mean())**2)),MSESS=float(1-np.sum(err*err)/np.sum((y-clim)**2)),
            bias_mm=float(err.mean()),MAE_mm=float(np.abs(err).mean()),tail80_bias_mm=float(err[tail].mean()),tail80_RMSE_mm=float(np.sqrt(np.mean(err[tail]**2))),tail_n=int(tail.sum()),negative_predictions=int((p<0).sum())))
    per=pd.DataFrame(rows);per.to_csv(DEST/'per_client_seed_ALL_endpoints.csv',index=False,float_format='%.17g')
    values=['MSE_mm2','RMSE_mm','R2','MSESS','bias_mm','MAE_mm','tail80_bias_mm','tail80_RMSE_mm']
    byseed=per.groupby(['endpoint','model','seed'])[values].mean().reset_index();byseed.to_csv(DEST/'macro_per_seed_ALL_endpoints.csv',index=False,float_format='%.17g')
    summary=byseed.groupby(['endpoint','model'])[values].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(DEST/'model_summary_ALL_endpoints.csv',index=False,float_format='%.17g')
    summary.loc['fixed60'].reset_index().to_csv(DEST/'principal_comparison.csv',index=False,float_format='%.17g')

def uncertainty(primary):
    protocol=r.load(ROOT/'protocol.json');amendment=r.load(ROOT/'analysis_amendment_v1.json');family=amendment['amended_primary_family']
    models=sorted(primary.model.unique());seeds=sorted(primary.seed.unique());clients=sorted(primary.client.unique());months=sorted(primary.target_month.unique())
    pred=np.stack([primary[primary.model==m].sort_values(['seed','client','target_month']).prediction.to_numpy().reshape(5,5,96) for m in models])
    first=primary[(primary.model==models[0])&(primary.seed==seeds[0])].sort_values(['client','target_month']);y=first.y.to_numpy().reshape(5,96);clim=first.climatology_mm.to_numpy().reshape(5,96)
    weights=np.array([[contrast['terms'].get(m,0.) for m in models] for contrast in protocol['primary_contrasts']])
    point=weights@metrics(y,pred,clim);rng=np.random.default_rng(731);draws=[];summary=[];resamples=[]
    for length in (12,6,24):
        vals=[]
        for repeat in range(2000):
            dates=e.circular_date_block_indices(96,length,rng);ss=rng.integers(0,5,size=5)
            pp=pred[:,ss,:,:][:,:,:,dates];mm=metrics(y[:,dates],pp,clim[:,dates]);delta=weights@mm;vals.append(delta)
            resamples.append(dict(block_length_months=length,replicate=repeat,seeds=[int(z) for z in ss],dates=[int(z) for z in dates]))
            for ci,contrast in enumerate(protocol['primary_contrasts']):
                draws.append(dict(contrast_id=contrast['id'],block_length_months=length,replicate=repeat,**{METRICS[k]:float(delta[ci,k]) for k in range(4)}))
        vals=np.array(vals)
        for ci,contrast in enumerate(protocol['primary_contrasts']):
            for k,metric in enumerate(METRICS):
                lo,hi=np.quantile(vals[:,ci,k],[.025,.975])
                summary.append(dict(contrast_id=contrast['id'],title=contrast['title'],metric=metric,delta=float(point[ci,k]),CI95_low=float(lo),CI95_high=float(hi),block_length_months=length,repetitions=2000,
                    interval_role='primary_block' if length==12 else 'block_sensitivity',metric_role='primary' if k==0 and contrast['id'] in family else 'secondary',multiplicity='Unadjusted percentile intervals; amended sharing family Holm6; original Holm7 sensitivity retained'))
    pd.DataFrame(draws).to_csv(DEST/'bootstrap_draws_7.csv',index=False,float_format='%.17g');pd.DataFrame(summary).to_csv(DEST/'paired_CI_7.csv',index=False,float_format='%.17g')
    np.savez_compressed(DEST/'bootstrap_resamples.npz',block=np.array([q['block_length_months'] for q in resamples]),replicate=np.array([q['replicate'] for q in resamples]),seeds=np.array([q['seeds'] for q in resamples]),dates=np.array([q['dates'] for q in resamples]))
    monthly_losses=np.mean((pred-y[None,None,:,:])**2,axis=(1,2));contributions=weights@monthly_losses
    signs=np.array(list(itertools.product([-1.,1.],repeat=8)));tests=[];annual=[]
    for ci,contrast in enumerate(protocol['primary_contrasts']):
        blocks=contributions[ci].reshape(8,12).sum(axis=1)/96;observed=float(blocks.sum());assert np.isclose(observed,point[ci,0],rtol=1e-12,atol=1e-9)
        null=signs@blocks;p=float(np.mean(np.abs(null)>=abs(observed)-1e-10))
        tests.append(dict(contrast_id=contrast['id'],delta_macro_MSE_mm2=observed,two_sided_p=p,n_annual_blocks=8,sign_assignments=256,assumption='Annual paired loss contributions sign-exchangeable; fixed clients and inspected retrospective outcomes'))
        for year,block in zip(range(2017,2025),blocks):annual.append(dict(contrast_id=contrast['id'],year=year,contribution_to_full_MSE_delta=float(block),annual_mean_delta=float(block*8)))
    holm7=e.holm_adjust({q['contrast_id']:q['two_sided_p'] for q in tests})
    holm6=e.holm_adjust({q['contrast_id']:q['two_sided_p'] for q in tests if q['contrast_id'] in family})
    for q in tests:q.update(holm_p_7_sensitivity=holm7[q['contrast_id']],holm_p_6_primary=holm6.get(q['contrast_id'],np.nan),
        primary_reject_05=holm6.get(q['contrast_id'],1.)<.05,contrast_role='amended_primary_sharing' if q['contrast_id'] in family else 'secondary_proximal_context')
    pd.DataFrame(tests).to_csv(DEST/'annual_MSE_sign_tests.csv',index=False,float_format='%.17g');pd.DataFrame(annual).to_csv(DEST/'annual_loss_contributions.csv',index=False,float_format='%.17g')
    r.save(DEST/'uncertainty_receipt.json',dict(utc=r.now(),models=models,seeds=[int(s) for s in seeds],clients=clients,months=months,common_resampling=True,draw_rows=42000,
        primary='macro_MSE_mm2',primary_family=family,original_family_Holm7_retained=True,seed='731',blocks=[12,6,24],repetitions=2000,interaction_recomputed_in_each_draw=True,clients_fixed=True,unique_observations=480,independent_year_blocks_not_established=True))

def main():
    frame,primary=make_predictions();summaries(frame);uncertainty(primary)
    r.save(DEST/'complete.json',dict(status='COMPLETE_PENDING_INDEPENDENT_QA',utc=r.now(),new_fits=40,primary_forecast_rows=21600,all_endpoint_rows=len(frame),unique_observations=480,
        reported_contrasts=7,primary_sharing_contrasts=6,secondary_contextual_contrasts=1,bootstrap_draw_rows=42000,files={p.name:r.sha(p) for p in DEST.iterdir() if p.is_file()}))
    print('EVALUATION_COMPLETE',len(frame),'rows;21600primary;42000draws',flush=True)

if __name__=='__main__':main()
