"""Inspect retained native hourly precipitation without altering source values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pcodec.standalone as pcodec
from recover_selected_months import ROOT,CHUNK,frozen_contract,sha
pre,masks,clients,weights,areas,rr,cc,periods,tiles,contract=frozen_contract()
rows=[json.loads(l) for l in (ROOT/'source_chunk_ledger.jsonl').read_text().splitlines() if json.loads(l)['variable']=='tp']
summary={p['month']:{'native_minimum':float('inf'),'native_maximum':float('-inf'),'native_hour_cell_values':0,'negative_hour_cell_values':0,'zero_hour_cell_values':0,'nonfinite_values':0} for p in periods};examples=[]
for row in rows:
    path=ROOT/row['raw_path'];raw=path.read_bytes();side=json.loads(path.with_suffix('.json').read_text())
    if sha(raw)!=side['sha256']:raise ValueError('Raw cache hash mismatch')
    t,y,x=row['chunk_indices'];data=pcodec.simple_decompress(raw).reshape(CHUNK)
    idx=np.where((rr>=y*12)&(rr<y*12+12)&(cc>=x*12)&(cc<x*12+12))[0]
    for p in periods:
        a=max(p['start']+1,t*CHUNK[0]);b=min(p['end_inclusive']+1,(t+1)*CHUNK[0])
        if a>=b:continue
        sub=data[a-t*CHUNK[0]:b-t*CHUNK[0],rr[idx]-y*12,cc[idx]-x*12];s=summary[p['month']]
        s['native_minimum']=min(s['native_minimum'],float(sub.min()));s['native_maximum']=max(s['native_maximum'],float(sub.max()))
        s['native_hour_cell_values']+=sub.size;s['negative_hour_cell_values']+=int((sub<0).sum());s['zero_hour_cell_values']+=int((sub==0).sum());s['nonfinite_values']+=int((~np.isfinite(sub)).sum())
        for h,k in np.argwhere(sub<0)[:max(0,20-len(examples))]:examples.append({'month':p['month'],'valid_time':str(pd.Timestamp('1940-01-01')+pd.Timedelta(hours=a+int(h))),'native_row':int(rr[idx[k]]),'native_column':int(cc[idx[k]]),'value_m':float(sub[h,k])})
for p in periods:
    if summary[p['month']]['native_hour_cell_values']!=p['hours']*3287:raise ValueError('Frozen hourly cell coverage mismatch')
receipt={'status':'PASS_NONNEGATIVE_COMPLETE' if all(s['negative_hour_cell_values']==s['nonfinite_values']==0 for s in summary.values()) else 'EXPLICIT_SOURCE_REVIEW_REQUIRED',
 'by_month':summary,'negative_examples':examples,'source_values_changed':False,'raw_tp_chunks':len(rows),'required_flow_semantics':contract['precipitation_time_semantics']}
(ROOT/'precipitation_native_probe.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
