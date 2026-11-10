"""Independent paired uncertainty arithmetic; no dataset reads or model imports."""
import itertools,math
import numpy as np

def equal_client_statistics(observed,forecast):
    """Mean of seed/client metrics; last dimension is common resampled dates."""
    residual=forecast-observed
    centered=observed-observed.mean(axis=2,keepdims=True)
    denominators=np.einsum('sct,sct->sc',centered,centered)
    if (denominators<=0).any():raise ValueError('Undefined constant-client R2; no floor')
    errors=np.einsum('sct,sct->sc',residual,residual)
    return np.array([(1-errors/denominators).mean(),np.sqrt(errors/observed.shape[2]).mean()])

def bootstrap_reference(observed,a,b,lengths=(12,6,24),repeats=2000,seed=731):
    if observed.shape!=a.shape or a.shape!=b.shape or observed.shape!=(5,5,96):raise ValueError('Exactly paired5seed/5client/96date arrays required')
    if not all(np.isfinite(v).all() for v in [observed,a,b]):raise ValueError('Finite arrays required')
    rng=np.random.Generator(np.random.PCG64(seed));point=equal_client_statistics(observed,a)-equal_client_statistics(observed,b)
    perseed=np.vstack([equal_client_statistics(observed[i:i+1],a[i:i+1])-equal_client_statistics(observed[i:i+1],b[i:i+1]) for i in range(5)])
    draws={};summaries=[]
    for length in lengths:
        block_total=(96+length-1)//length;deltas=np.empty((repeats,2))
        for r in range(repeats):
            starts=rng.integers(0,96,block_total)
            dates=np.concatenate([np.arange(int(s),int(s)+length)%96 for s in starts])[:96]
            paired_seeds=rng.integers(0,5,5)
            yy=observed[paired_seeds][:,:,dates];aa=a[paired_seeds][:,:,dates];bb=b[paired_seeds][:,:,dates]
            deltas[r]=equal_client_statistics(yy,aa)-equal_client_statistics(yy,bb)
        draws[length]=deltas
        ci=np.quantile(deltas,[.025,.975],axis=0)
        summaries.append({'block':length,'point':point.tolist(),'CI95_low':ci[0].tolist(),'CI95_high':ci[1].tolist(),'draw_mean':deltas.mean(axis=0).tolist(),'CI_midpoint':ci.mean(axis=0).tolist()})
    return {'draws':draws,'summary':summaries,'perseed':perseed,'point':point}

def sign_reference(observed,a,b):
    centered=observed-observed.mean(axis=2,keepdims=True);tss=(centered*centered).sum(axis=2)
    if (tss<=0).any():raise ValueError('Undefined R2 randomization')
    contributions=(((observed-b)**2-(observed-a)**2)/tss[:,:,None]).mean(axis=(0,1))
    blocks=np.array([math.fsum(contributions[i:i+12]) for i in range(0,96,12)])
    observed_delta=math.fsum(blocks)
    values=[math.fsum(sign*value for sign,value in zip(signs,blocks)) for signs in itertools.product((-1,1),repeat=8)]
    p=sum(abs(v)>=abs(observed_delta)-1e-14 for v in values)/256
    return {'delta':observed_delta,'p':p,'sign_patterns':256,'blocks':8}

def holm_reference(pvalues):
    ordered=sorted(enumerate(pvalues),key=lambda pair:pair[1]);adjusted=np.empty(len(pvalues));running=0
    for rank,(i,p) in enumerate(ordered):
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid p')
        running=max(running,min(1,(len(pvalues)-rank)*p));adjusted[i]=running
    return adjusted.tolist()
