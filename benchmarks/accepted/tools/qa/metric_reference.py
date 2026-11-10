"""Independent scalar formulas prepared for later completed-result QA.

This module has no file reads, imports from caadapfl, fitting, or execution side
effects. Its inputs must be admitted paired observations/raw predictions.
"""
import math
import statistics

def regression_reference(observed,predicted):
    y=list(map(float,observed));p=list(map(float,predicted));n=len(y)
    if not n or n!=len(p) or not all(math.isfinite(v) for v in y+p):
        raise ValueError('Finite aligned vectors required')
    ym=statistics.fmean(y);pm=statistics.fmean(p)
    ys=math.fsum((v-ym)**2 for v in y);ps=math.fsum((v-pm)**2 for v in p)
    errors=[b-a for a,b in zip(y,p)];total=math.fsum(y)
    r2=1-math.fsum(e*e for e in errors)/ys if ys>0 else math.nan
    bias=-100*math.fsum(errors)/total if total else math.nan
    correlation=math.fsum((a-ym)*(b-pm) for a,b in zip(y,p))/math.sqrt(ys*ps) if n>1 and ys>0 and ps>0 else math.nan
    alpha=math.sqrt(ps/ys) if n>1 and ys>0 else math.nan
    beta=pm/ym if ym else math.nan
    kge=1-math.sqrt((correlation-1)**2+(alpha-1)**2+(beta-1)**2) if all(math.isfinite(v) for v in [correlation,alpha,beta]) else math.nan
    return {'n':n,'R2':r2,'NSE':r2,'RMSE_mm':math.sqrt(math.fsum(e*e for e in errors)/n),'MAE_mm':math.fsum(abs(e) for e in errors)/n,'PBIAS_signed_percent':bias,'PBIAS_absolute_percent':abs(bias),'forecast_bias_signed_percent':-bias,'Pearson_r':correlation,'KGE_2009':kge}

def heavy_reference(observed,predicted,training_q80,probabilities=None):
    y=list(map(float,observed));p=list(map(float,predicted))
    if len(y)!=len(p) or not y:raise ValueError('Aligned vectors required')
    actual=[v>training_q80 for v in y]
    if probabilities is None:decision=[v>training_q80 for v in p]
    else:
        probability=list(map(float,probabilities))
        if len(probability)!=len(y) or not all(math.isfinite(v) and 0<=v<=1 for v in probability):raise ValueError('Valid aligned probabilities required')
        decision=[v>=.5 for v in probability]
    tp=sum(a and d for a,d in zip(actual,decision));fp=sum(not a and d for a,d in zip(actual,decision));fn=sum(a and not d for a,d in zip(actual,decision));tn=sum(not a and not d for a,d in zip(actual,decision))
    tail=[b-a for a,b,h in zip(y,p,actual) if h]
    return {'actual_heavy_n':sum(actual),'heavy_TP':tp,'heavy_FP':fp,'heavy_FN':fn,'heavy_TN':tn,'heavy_precision':tp/(tp+fp) if tp+fp else math.nan,'heavy_recall':tp/(tp+fn) if tp+fn else math.nan,'heavy_F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else math.nan,'tail80_RMSE_mm':math.sqrt(math.fsum(v*v for v in tail)/len(tail)) if tail else math.nan,'tail80_MAE_mm':math.fsum(abs(v) for v in tail)/len(tail) if tail else math.nan}
