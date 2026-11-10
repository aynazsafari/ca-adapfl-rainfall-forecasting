"""Independent supplement calendar, normalization and paired-block arithmetic."""
import math
import numpy as np
import pandas as pd
from uncertainty_reference import equal_client_statistics
from audit_completed_run import scalar_close,linear_quantile

def require(condition,message):
    if not condition:raise AssertionError(message)
def role_calendars(targets):
    return {'train':[m for m in targets if m<='2007-11'],
            'monitor':[m for m in targets if '2008-01'<=m<='2010-11'],
            'validation':[m for m in targets if '2011-01'<=m<='2016-11'],
            'evaluation':[m for m in targets if '2017-01'<=m<='2024-12']}
def contiguous_calendar(calendar):
    require(bool(calendar),'Empty evaluation calendar')
    require(calendar==list(pd.period_range(calendar[0],calendar[-1],freq='M').astype(str)),
            'Interior calendar gap: circular date blocks cannot bridge excluded months')
def derived_targets(valid,clients):
    return [str(t) for t in pd.period_range('1983-01','2024-12',freq='M')
            if all(valid.get((c,str(t-offset)),False) for c in clients for offset in range(4,16))]

def own_train_statistics(clients,arrays,names,climate_names,roles,statistics):
    """Recalculate summaries from only latest-row gradient TRAIN covariates."""
    sums=squares=None;count=0;local={};climate_indices=[names.index(n) for n in climate_names]
    for client in clients:
        data=arrays[client];train=np.isin(data['raw_target_month'],roles['train']);x=data['raw_x'][train,-1,:];y=data['y'][train]
        require(len(y)==len(roles['train']) and np.isfinite(x).all() and np.isfinite(y).all(),'TRAIN archive/calendar missing')
        sums=x.sum(0) if sums is None else sums+x.sum(0);squares=(x*x).sum(0) if squares is None else squares+(x*x).sum(0);count+=len(y)
        mean=float(np.mean(y));std=float(np.std(y,ddof=0));q80=linear_quantile(y,.8)
        require(std>=1e-12,'Degenerate training target')
        local[client]=(mean,std,q80)
        saved=statistics['clients'][client]
        for name,value in zip(['target_mean','target_std','heavy80_threshold_mm'],local[client]):scalar_close(value,saved[name],client+'/'+name,atol=1e-10)
        require(saved['role_counts']=={r:len(k) for r,k in roles.items()},'TRAIN/monitor/validation/evaluation role count differs')
        require(np.allclose(data['target_stats'],[mean,std],rtol=1e-10,atol=1e-10),'Archived target inverse summary differs')
        require(np.array_equal(data['y_z'],((data['y']-saved['target_mean'])/saved['target_std']).astype(np.float32)),'Target z-values differ from TRAIN-only scaling')
        require(np.array_equal(data['heavy'],(data['y']>saved['heavy80_threshold_mm']).astype(np.float32)),'Strict TRAIN q80 heavy flags differ')
    mean=sums/count;std=np.sqrt(np.maximum(squares/count-mean*mean,0));std[std<1e-12]=1.
    require(statistics['count']==count and statistics['source_role']=='gradient_train','Input scaler population differs')
    require(np.allclose(mean,statistics['global_input_mean'],rtol=1e-10,atol=1e-10) and np.allclose(std,statistics['global_input_std'],rtol=1e-10,atol=1e-10),'Input summaries are not own TRAIN-only statistics')
    for client in clients:
        data=arrays[client];saved_mean=np.asarray(statistics['global_input_mean']);saved_std=np.asarray(statistics['global_input_std'])
        require(np.array_equal(data['scaled_x'],((data['raw_x']-saved_mean)/saved_std).astype(np.float32)),'Archived input normalization differs')
        require(np.array_equal(data['raw_climate'],data['raw_x'][:,-1,climate_indices]),'Climate context versus latest feature row differs')
        require(np.array_equal(data['scaled_climate'],((data['raw_climate']-saved_mean[climate_indices])/saved_std[climate_indices]).astype(np.float32)),'Archived climate normalization differs')
    return {'count':count,'input_mean':mean,'input_std':std,'target_statistics':local}

def cohort_bootstrap_reference(observed,a,b,calendar,lengths=(12,6,24),repeats=2000,seed=731):
    contiguous_calendar(calendar);n=len(calendar)
    require(observed.shape==a.shape==b.shape==(5,5,n),'Five paired seeds/clients and declared date axis required')
    require(all(np.isfinite(v).all() for v in [observed,a,b]),'Finite paired arrays required')
    require(np.array_equal(observed,np.broadcast_to(observed[:1],observed.shape)),'Truth varies across paired seeds')
    rng=np.random.Generator(np.random.PCG64(seed));stat_a=equal_client_statistics(observed,a);stat_b=equal_client_statistics(observed,b);point=stat_a-stat_b
    perseed=[]
    for i in range(5):
        sa=equal_client_statistics(observed[i:i+1],a[i:i+1]);sb=equal_client_statistics(observed[i:i+1],b[i:i+1]);perseed.append({'a':sa,'b':sb,'delta':sa-sb})
    draws={};summary=[]
    for length in lengths:
        require(1<=length<=n,'Invalid date block length');deltas=np.empty((repeats,2));blocks=(n+length-1)//length
        for r in range(repeats):
            starts=rng.integers(0,n,blocks);dates=np.concatenate([np.arange(int(s),int(s)+length)%n for s in starts])[:n];seeds=rng.integers(0,5,5)
            y=observed[seeds][:,:,dates];aa=a[seeds][:,:,dates];bb=b[seeds][:,:,dates]
            # Undefined R2 raises explicitly, following the frozen evaluator;
            # no denominator floor or silent removal of bootstrap replicates.
            deltas[r]=equal_client_statistics(y,aa)-equal_client_statistics(y,bb)
        draws[length]=deltas;ci=np.quantile(deltas,[.025,.975],axis=0)
        summary.append({'block':length,'point':point,'CI95_low':ci[0],'CI95_high':ci[1],'draw_mean':deltas.mean(0),'CI_midpoint':ci.mean(0)})
    return {'draws':draws,'summary':summary,'perseed':perseed,'point':point}
