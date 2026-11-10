"""Keyed rainfall metrics and paired seed/common-date block uncertainty.

Positive PBIAS means underprediction. Undefined metrics remain NaN with reasons;
there are no denominator floors. Bootstrap samples identical date blocks for
every client, seed and compared model, then pairs resampled training seeds.
"""
from __future__ import annotations
import itertools,math
from typing import Mapping
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score,roc_auc_score

KEYS=['model','seed','client','target_month']

def training_heavy_thresholds(samples,quantile=.8,train_end='2007-11'):
    if train_end>'2007-11':raise ValueError('Heavy threshold must be released gradient-training labels only')
    result={}
    for c,d in samples.items():
        mask=np.asarray(d['target_month'],str)<=train_end;values=np.asarray(d['y'])[mask]
        if not len(values) or not np.isfinite(values).all():raise ValueError('Invalid admitted threshold targets')
        result[c]=float(np.quantile(values,quantile,method='linear'))
    return result

def validate_prediction_frame(frame,expected_months=None,expected_seeds=None):
    required=set(KEYS+['y','prediction'])
    if not required.issubset(frame.columns):raise ValueError('Missing prediction columns: '+str(required-set(frame.columns)))
    result=frame.copy()
    if result.empty or result.duplicated(KEYS).any():raise ValueError('Empty or duplicate prediction keys')
    if not np.isfinite(result[['y','prediction']].to_numpy(float)).all():raise ValueError('Nonfinite observed/predicted value; no fallback')
    if (result.y<0).any():raise ValueError('Negative observed rainfall')
    canonical=np.asarray(pd.PeriodIndex(result.target_month,freq='M').astype(str))
    if not np.array_equal(result.target_month.to_numpy(str),canonical):raise ValueError('Canonical monthly keys required')
    first=None;truth=None;models=sorted(result.model.unique());seedset=None
    for model in models:
        seeds=sorted(result.loc[result.model==model,'seed'].unique())
        if seedset is None:seedset=seeds
        if seeds!=seedset:raise ValueError('Model seed cohorts differ')
        if expected_seeds is not None and seeds!=sorted(expected_seeds):raise ValueError('Expected seed cohort differs')
        for seed in seeds:
            part=result.loc[(result.model==model)&(result.seed==seed)].sort_values(['client','target_month']);keys=part[['client','target_month']].to_numpy(str);values=part.y.to_numpy(float)
            if first is None:first=keys;truth=values
            if not np.array_equal(keys,first) or not np.array_equal(values,truth):raise ValueError('Models/seeds must share exactly identical client/date keys and truths')
    # Client calendars are common and contiguous, including all requested dates.
    sample=result.loc[(result.model==models[0])&(result.seed==seedset[0])]
    calendar=None
    for client,g in sample.groupby('client',sort=True):
        dates=np.sort(g.target_month.to_numpy(str))
        if calendar is None:calendar=dates
        if not np.array_equal(dates,calendar):raise ValueError('Clients must share common dates')
        expected=np.asarray(pd.period_range(dates[0],dates[-1],freq='M').astype(str)) if expected_months is None else np.asarray(expected_months,str)
        if not np.array_equal(dates,expected):raise ValueError('Incomplete calendar or unexpected cohort')
    return result.sort_values(KEYS).reset_index(drop=True)

def regression_metrics(y,prediction):
    y=np.asarray(y,float);p=np.asarray(prediction,float)
    if y.shape!=p.shape or y.ndim!=1 or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Finite aligned one-dimensional vectors required')
    error=p-y;tss=float(np.sum((y-y.mean())**2));sy=float(y.sum());std_y=float(y.std(ddof=1)) if len(y)>1 else np.nan;std_p=float(p.std(ddof=1)) if len(p)>1 else np.nan;reasons=[]
    r2=1-float(np.sum(error**2))/tss if tss>0 else np.nan
    if tss<=0:reasons.append('R2/NSE undefined: constant observations')
    pbias=-100*float(error.sum())/sy if sy!=0 else np.nan
    if sy==0:reasons.append('PBIAS undefined: zero observed total')
    corr=float(np.corrcoef(y,p)[0,1]) if std_y>0 and std_p>0 else np.nan
    if not np.isfinite(corr):reasons.append('Correlation/KGE undefined: constant vector or insufficient count')
    alpha=std_p/std_y if std_y>0 else np.nan;beta=float(p.mean()/y.mean()) if y.mean()!=0 else np.nan
    kge=1-math.sqrt((corr-1)**2+(alpha-1)**2+(beta-1)**2) if np.isfinite([corr,alpha,beta]).all() else np.nan
    if not np.isfinite(kge) and not any('KGE' in r for r in reasons):reasons.append('KGE undefined: zero observed mean')
    return {'n':len(y),'R2':r2,'RMSE_mm':float(np.sqrt(np.mean(error**2))),'MAE_mm':float(np.mean(np.abs(error))),'PBIAS_signed_percent':pbias,'PBIAS_absolute_percent':abs(pbias),'forecast_bias_signed_percent':-pbias,'KGE_2009':kge,'Pearson_r':corr,'NSE':r2,'undefined_reasons':'; '.join(reasons)}

def _heavy_metrics(y,p,threshold,probability=None,probability_decision_threshold=.5):
    observed=np.asarray(y)>threshold
    if probability is None:
        score=np.asarray(p,float);predicted=score>threshold;score_kind='continuous predicted rainfall mm; not calibrated probability';decision=f'predicted rainfall > training q80 ({threshold})'
    else:
        score=np.asarray(probability,float)
        if not np.isfinite(score).all() or ((score<0)|(score>1)).any():raise ValueError('Invalid heavy-event probabilities')
        predicted=score>=probability_decision_threshold;score_kind='predicted heavy-event probability';decision=f'probability >= {probability_decision_threshold}'
    tp=int(np.sum(observed&predicted));fp=int(np.sum(~observed&predicted));fn=int(np.sum(observed&~predicted));tn=int(np.sum(~observed&~predicted));precision=tp/(tp+fp) if tp+fp else np.nan;recall=tp/(tp+fn) if tp+fn else np.nan;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    tail=np.asarray(p)[observed]-np.asarray(y)[observed];ap=float(average_precision_score(observed,score)) if observed.any() else np.nan;auc=float(roc_auc_score(observed,score)) if observed.any() and (~observed).any() else np.nan
    if observed.any():pr,rc,cut=precision_recall_curve(observed,score);curve={'precision':pr.tolist(),'recall':rc.tolist(),'score_threshold':cut.tolist(),'score_kind':score_kind}
    else:curve={'precision':[],'recall':[],'score_threshold':[],'score_kind':score_kind,'undefined_reason':'No actual heavy events'}
    values={'heavy80_training_threshold_mm':threshold,'actual_heavy_n':int(observed.sum()),'heavy_TP':tp,'heavy_FP':fp,'heavy_FN':fn,'heavy_TN':tn,'heavy_precision':precision,'heavy_recall':recall,'heavy_F1':f1,'heavy_average_precision':ap,'heavy_ROC_AUC':auc,'tail80_RMSE_mm':float(np.sqrt(np.mean(tail**2))) if len(tail) else np.nan,'tail80_MAE_mm':float(np.mean(np.abs(tail))) if len(tail) else np.nan,'heavy_score_kind':score_kind,'heavy_decision_rule':decision}
    return values,curve

def evaluate_predictions(frame,heavy_thresholds,expected_months=None,expected_seeds=None,probability_column='heavy_probability',probability_decision_threshold=.5):
    frame=validate_prediction_frame(frame,expected_months,expected_seeds);client_rows=[];curves=[];aggregate=[]
    for (model,seed,client),g in frame.groupby(['model','seed','client'],sort=True):
        if client not in heavy_thresholds:raise ValueError('Training-only client heavy threshold absent')
        probability=None
        if probability_column in g:
            if g[probability_column].notna().any():
                if not g[probability_column].notna().all():raise ValueError('Partially missing heavy probability; no per-row fallback')
                probability=g[probability_column].to_numpy(float)
        y=g.y.to_numpy(float);p=g.prediction.to_numpy(float);metrics=regression_metrics(y,p);heavy,curve=_heavy_metrics(y,p,float(heavy_thresholds[client]),probability,probability_decision_threshold)
        client_rows.append({'model':model,'seed':seed,'client':client,**metrics,**heavy});curves.append({'model':model,'seed':int(seed),'client':client,**curve})
    byclient=pd.DataFrame(client_rows)
    regression_names=['R2','RMSE_mm','MAE_mm','PBIAS_signed_percent','PBIAS_absolute_percent','forecast_bias_signed_percent','KGE_2009','Pearson_r','NSE']
    for (model,seed),g in frame.groupby(['model','seed'],sort=True):
        c=byclient.loc[(byclient.model==model)&(byclient.seed==seed)]
        macro={'model':model,'seed':seed,'scope':'macro_equal_client','n':len(g),'n_clients':len(c)}
        for metric in regression_names:
            values=c[metric].to_numpy(float);macro[metric]=float(values.mean()) if np.isfinite(values).all() else np.nan
        macro['undefined_reasons']='; '.join(sorted(set(v for v in c.undefined_reasons if v)))
        aggregate.append(macro);aggregate.append({'model':model,'seed':seed,'scope':'pooled_all_client_months','n_clients':len(c),**regression_metrics(g.y.to_numpy(float),g.prediction.to_numpy(float))})
    aggregate=pd.DataFrame(aggregate);primary=aggregate[aggregate.scope=='macro_equal_client'];seed_summary=primary.groupby('model')[regression_names].agg(['mean','std','min','max']);seed_summary.columns=['_'.join(col) for col in seed_summary.columns];seed_summary=seed_summary.reset_index()
    return {'per_client':byclient,'aggregate_per_seed':aggregate,'seed_summary':seed_summary,'PR_curves':curves,'receipt':{'primary_metrics':['macro_equal_client R2','macro_equal_client RMSE_mm'],'pooled_reported_separately':True,'PBIAS_sign':'100*sum(obs-pred)/sum(obs); positive means underprediction, matching manuscript EqP0324','forecast_bias_sign':'100*sum(pred-obs)/sum(obs); positive means overprediction','KGE_definition':'2009: r, std(pred)/std(obs), mean(pred)/mean(obs)','heavy_truth':'observed rainfall strictly greater than client training q80','threshold_fitting':'Caller supplies training-only thresholds; helper training_heavy_thresholds admits gradient-training labels only','no_clipping_or_metric_floors':True}}

def _paired_arrays(frame,model_a,model_b,expected_seed_count=5):
    selected=frame.loc[frame.model.isin([model_a,model_b])].copy();selected=validate_prediction_frame(selected)
    if set(selected.model.unique())!=set([model_a,model_b]):raise ValueError('Compared model absent')
    seeds=sorted(int(s) for s in selected.seed.unique());clients=sorted(str(c) for c in selected.client.unique());months=sorted(str(m) for m in selected.target_month.unique())
    if len(seeds)!=expected_seed_count:raise ValueError(f'Expected {expected_seed_count} paired seeds; found {len(seeds)}')
    arrays={}
    for model in [model_a,model_b]:
        group=selected.loc[selected.model==model].set_index(['seed','client','target_month']);index=pd.MultiIndex.from_product([seeds,clients,months],names=['seed','client','target_month']);group=group.reindex(index)
        arrays[model]=group.prediction.to_numpy(float).reshape(len(seeds),len(clients),len(months));truth=group.y.to_numpy(float).reshape(len(seeds),len(clients),len(months))
    if not np.isfinite(truth).all() or not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonrectangular paired cohort')
    if not np.array_equal(truth,np.broadcast_to(truth[0:1],truth.shape)):raise ValueError('Observed truth differs across paired seeds')
    return truth,arrays[model_a],arrays[model_b],seeds,clients,months

def _statistics(y,p):
    denominator=np.sum((y-y.mean(axis=-1,keepdims=True))**2,axis=-1)
    if (denominator<=0).any():raise ValueError('Resample has constant client observations; R2 undefined, no floor')
    r2=1-np.sum((y-p)**2,axis=-1)/denominator;rmse=np.sqrt(np.mean((y-p)**2,axis=-1));return float(r2.mean()),float(rmse.mean())

def circular_date_block_indices(n,block_length,rng):
    if not 1<=block_length<=n:raise ValueError('Invalid circular block length')
    starts=rng.integers(0,n,size=math.ceil(n/block_length));return ((starts[:,None]+np.arange(block_length))%n).ravel()[:n]

def paired_block_bootstrap(frame,model_a,model_b,block_lengths=(12,6,24),repetitions=2000,bootstrap_seed=731,expected_seed_count=5,resample_seeds=True):
    y,a,b,seeds,clients,months=_paired_arrays(frame,model_a,model_b,expected_seed_count);n=len(months);rng=np.random.default_rng(bootstrap_seed);point_a=_statistics(y,a);point_b=_statistics(y,b);summary=[];draws=[];perseed=[]
    for i,seed in enumerate(seeds):
        sa=_statistics(y[i:i+1],a[i:i+1]);sb=_statistics(y[i:i+1],b[i:i+1]);perseed.append({'seed':seed,'model_a':model_a,'model_b':model_b,'macro_R2_a':sa[0],'macro_R2_b':sb[0],'delta_macro_R2_a_minus_b':sa[0]-sb[0],'macro_RMSE_mm_a':sa[1],'macro_RMSE_mm_b':sb[1],'delta_macro_RMSE_mm_a_minus_b':sa[1]-sb[1]})
    for length in block_lengths:
        values=[]
        for repeat in range(repetitions):
            dates=circular_date_block_indices(n,length,rng);sample_seeds=rng.integers(0,len(seeds),size=len(seeds)) if resample_seeds else np.arange(len(seeds));ys=y[sample_seeds][:,:,dates];aa=a[sample_seeds][:,:,dates];bb=b[sample_seeds][:,:,dates];stat_a=_statistics(ys,aa);stat_b=_statistics(ys,bb);delta=(stat_a[0]-stat_b[0],stat_a[1]-stat_b[1]);values.append(delta);draws.append({'block_length_months':length,'replicate':repeat,'delta_macro_R2_a_minus_b':delta[0],'delta_macro_RMSE_mm_a_minus_b':delta[1]})
        values=np.asarray(values)
        for j,name in enumerate(['macro_R2','macro_RMSE_mm']):
            lo,hi=np.quantile(values[:,j],[.025,.975]);summary.append({'model_a':model_a,'model_b':model_b,'metric':name,'delta_a_minus_b':point_a[j]-point_b[j],'CI95_low':float(lo),'CI95_high':float(hi),'block_length_months':length,'role':'primary' if length==12 else 'prespecified_block_length_sensitivity','repetitions':repetitions,'n_seeds':len(seeds),'n_clients':len(clients),'n_months':n,'resample_seeds':resample_seeds})
    receipt={'models':[model_a,model_b],'seed_keys':seeds,'client_keys':clients,'month_keys':months,'date_resampling':'Same circular date block indices for all clients, paired seeds and both models','seed_resampling':'Same sampled seed indices for both models; observations identical across seeds' if resample_seeds else 'Fixed five paired seeds','estimand':'Mean of per-seed macro equal-client metrics, not metric of ensemble-averaged predictions','blocks':list(block_lengths),'bootstrap_seed':bootstrap_seed,'iid_rows_resampled':False,'spatial_generalization_claim':False,'delta_direction':'R2: positive favors A; RMSE: negative favors A','conditional_scope':'Five fixed study regions and this revised-archive evaluation cohort; no untouched prospective-test claim'}
    return {'summary':pd.DataFrame(summary),'draws':pd.DataFrame(draws),'per_seed':pd.DataFrame(perseed),'receipt':receipt}

def block_sign_randomization(frame,model_a,model_b,block_length=12,repetitions=10000,random_seed=732,expected_seed_count=5):
    """Paired macro-R2 loss contributions; each common date block gets one sign.

    Exact enumeration for <=16 nonoverlapping blocks; otherwise Monte Carlo
    plus-one correction. Requires blockwise exchangeability/symmetry under H0.
    """
    y,a,b,seeds,clients,months=_paired_arrays(frame,model_a,model_b,expected_seed_count);den=np.sum((y-y.mean(axis=-1,keepdims=True))**2,axis=-1)
    if (den<=0).any():raise ValueError('Undefined R2 randomization')
    contribution=np.mean(((y-b)**2-(y-a)**2)/den[:,:,None],axis=(0,1));blocks=np.array([contribution[start:start+block_length].sum() for start in range(0,len(months),block_length)]);observed=float(blocks.sum());count=len(blocks)
    if count<=16:
        signs=np.array(list(itertools.product([-1.,1.],repeat=count)));null=signs@blocks;p=float(np.mean(np.abs(null)>=abs(observed)-1e-14));method='exact common-block sign enumeration'
    else:
        rng=np.random.default_rng(random_seed);signs=rng.choice([-1.,1.],size=(repetitions,count));null=signs@blocks;p=float((1+np.sum(np.abs(null)>=abs(observed)-1e-14))/(repetitions+1));method='Monte Carlo common-block sign randomization with plus-one correction'
    return {'model_a':model_a,'model_b':model_b,'delta_macro_R2_a_minus_b':observed,'two_sided_p':p,'block_length_months':block_length,'n_blocks':count,'method':method,'null_assumption':'Paired date-block loss contributions have sign symmetry/exchangeability; this is an assumption, not established independence','multiplicity':'Apply Holm across the prespecified reported comparison family','not_iid_Wilcoxon':True}

def holm_adjust(pvalues:Mapping[str,float]):
    ordered=sorted(pvalues,key=pvalues.get);n=len(ordered);result={};previous=0.
    for i,key in enumerate(ordered):
        p=float(pvalues[key])
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Finite p in [0,1] required')
        previous=max(previous,min(1.,(n-i)*p));result[key]=previous
    return result
