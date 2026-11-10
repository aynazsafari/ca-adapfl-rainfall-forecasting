"""Recover seven fixed monthly ERA5 panels from a pinned, read-only public snapshot.

Six source-defective GEE months plus a frozen March 2024 compatibility control.
Keep every compressed native chunk, byte range and source hash. No imputation.
Two network workers, one numerical thread, and a hard 2 GB climate-byte limit.
"""
from pathlib import Path
import argparse, asyncio, calendar, datetime, json, math, os, shutil, sys, time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import icechunk, zarr, netCDF4, requests
import pcodec.standalone as pcodec
from threadpoolctl import threadpool_limits
from native_geometry import VARS, OLD, CHUNK, sha, build_weights

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
MONTHS=['2022-04','2022-05','2022-07','2022-08','2024-01','2024-02','2024-03']
CAP=2_000_000_000
NATIVE={'t2m':'temperature_2m','d2m':'dewpoint_temperature_2m','sp':'surface_pressure','msl':'mean_sea_level_pressure','u10':'u_component_of_wind_10m','v10':'v_component_of_wind_10m','tp':'total_precipitation'}
TOLERANCES={'t2m':1e-4,'d2m':1e-4,'sp':.01,'msl':.01,'u10':1e-4,'v10':1e-4,'tp':1e-5}

def immutable(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'Immutable bytes changed: {path}')
    else:
        tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(data);os.replace(tmp,path)

def frozen_contract():
    pre=json.loads((ROOT/'source_evidence/preflight.json').read_text())
    maskfile=ROOT/'source_evidence/frozen_native_masks.npz'
    masks=np.load(maskfile)
    clients,weights,areas=build_weights(masks)
    li=masks['latitude_indices'];lj=masks['longitude_indices'];rr,cc=np.where(masks['union']);rr=li[rr];cc=lj[cc]
    if len(rr)!=3287 or [int((weights[i]>0).sum()) for i in range(5)]!=[690,912,419,466,864]:raise ValueError('Frozen mask changed')
    periods=[];epoch=pd.Timestamp('1940-01-01')
    for m in MONTHS:
        start=pd.Timestamp(m+'-01');end=start+pd.offsets.MonthBegin(1)
        a=int((start-epoch)/pd.Timedelta(hours=1));b=int((end-epoch)/pd.Timedelta(hours=1))
        periods.append({'month':m,'start':a,'end_inclusive':b,'hours':b-a,'source_interval_start':start.strftime('%Y-%m-%dT00:00'),'source_interval_end':end.strftime('%Y-%m-%dT00:00')})
    tiles=sorted({i//CHUNK[0] for p in periods for i in range(p['start'],p['end_inclusive']+1)})
    contract={'months':MONTHS,'recovery_months':MONTHS[:-1],'compatibility_month':'2024-03','snapshot_id':pre['snapshot_id'],
      'country_sha256':pre['geo_sha256'],'mask_sha256':sha(maskfile.read_bytes()),'temporal_tiles':tiles,
      'union_cells':3287,'clients':clients,'native_counts':dict(zip(clients,[690,912,419,466,864])),
      'native_areas_m2':dict(zip(clients,[float(x) for x in areas])),'periods':periods,'cap_climate_bytes':CAP,'network_workers':2,
      'instantaneous_time_semantics':'[00 first day,00 first day of next month)',
      'precipitation_time_semantics':'One-hour amount ending validity timestamp; (00 first day,00 first day of next month]',
      'provider_compatibility_absolute_tolerances_native_units':TOLERANCES,'tolerances_frozen_from':'January 1981 check, before any forecast outcomes',
      'source_admission':'all selected native cells finite; all required hourly timestamps; source status0 and expver1; March compatibility PASS',
      'historical_availability':'Retrospective final reanalysis archive, no historical operational vintage claim','no_imputation':True}
    immutable(ROOT/'frozen_recovery_contract.json',json.dumps(contract,indent=2).encode())
    original_metadata=json.loads((ROOT/'raw_metadata_manifest.json').read_text())
    for item in original_metadata:
        raw=(ROOT/'raw/metadata'/item['key']).read_bytes()
        if sha(raw)!=item['sha256']:raise ValueError('Pinned original source metadata hash mismatch')
    if sha((ROOT/'source_evidence/india_country_ne_5_1_1.geojson').read_bytes())!=pre['geo_sha256']:raise ValueError('Country hash mismatch')
    return pre,masks,clients,weights,areas,rr,cc,periods,tiles,contract

async def inventory(session,pre,tiles):
    path=ROOT/'source_chunk_ledger.jsonl'
    targets={(t,y,x) for t in tiles for y,x in pre['spatial_tiles']}
    if path.exists():
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        if {(r['variable'],*r['chunk_indices']) for r in rows}!={(v,*c) for v in VARS for c in targets}:raise ValueError('Inventory coverage mismatch')
        if any(r['snapshot_id']!=pre['snapshot_id'] for r in rows):raise ValueError('Inventory snapshot mismatch')
        return rows
    rows=[]
    for var in VARS:
        got=set();it=session.store.array_chunk_iterator('/single/temporal/'+var,batch_size=100000)
        try:
            async for coords,kinds,paths,offsets,lengths,inlined in it:
                for i in np.where(np.isin(coords[:,0],tiles))[0]:
                    co=tuple(int(x) for x in coords[i])
                    if co not in targets:continue
                    if co in got:raise ValueError('Duplicate native chunk reference')
                    typ=int(kinds[i]);location=str(paths[i])
                    if typ not in (1,2):raise ValueError('Unsupported inline or missing source chunk')
                    url='https://earthmover-icechunk-era5.s3.us-east-1.amazonaws.com/icechunkV2/chunks/'+location if typ==1 else location
                    rows.append({'variable':var,'chunk_indices':list(co),'zarr_key':f'single/temporal/{var}/c/{co[0]}/{co[1]}/{co[2]}',
                      'storage_type':typ,'storage_path':location,'source_url':url,'byte_offset':int(offsets[i]),'byte_length':int(lengths[i]),
                      'snapshot_id':pre['snapshot_id'],'raw_path':f'raw/chunks/{var}/{co[0]}_{co[1]}_{co[2]}.pcodec'})
                    got.add(co)
                if got==targets:break
        finally:await it.aclose()
        if got!=targets:raise ValueError(f'Missing source chunks for {var}: {targets-got}')
        print('INVENTORY',var,len(got),sum(r['byte_length'] for r in rows if r['variable']==var),flush=True)
    rows.sort(key=lambda r:(r['variable'],*r['chunk_indices']))
    immutable(path,(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows)).encode())
    return rows

def fetch(row):
    path=ROOT/row['raw_path'];side=path.with_suffix('.json')
    if path.exists():
        raw=path.read_bytes();saved=json.loads(side.read_text())
        if len(raw)!=row['byte_length'] or sha(raw)!=saved['sha256']:raise ValueError('Raw immutable cache integrity failure')
        return saved
    off=row['byte_offset'];length=row['byte_length']
    for attempt in range(3):
        try:
            response=requests.get(row['source_url'],headers={'Range':f'bytes={off}-{off+length-1}','Accept-Encoding':'identity'},timeout=(30,180))
            response.raise_for_status()
            if response.status_code!=206 or len(response.content)!=length:raise ValueError('Native source byte-range contract failed')
            raw=response.content;break
        except (requests.ConnectionError,requests.Timeout):
            if attempt==2:raise
            time.sleep(2*(attempt+1))
    saved={**row,'sha256':sha(raw),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'content_range':response.headers.get('Content-Range'),'etag':response.headers.get('ETag')}
    immutable(path,raw);immutable(side,json.dumps(saved,indent=2).encode());return saved

async def main(execute):
    threadpool_limits(limits=1)
    pre,masks,clients,weights,areas,rr,cc,periods,tiles,contract=frozen_contract()
    repo=icechunk.Repository.open(icechunk.s3_storage(bucket='earthmover-icechunk-era5',prefix='icechunkV2',region='us-east-1',anonymous=True))
    session=repo.readonly_session(snapshot_id=pre['snapshot_id']);group=zarr.open_group(session.store,path='single/temporal',mode='r')
    if not np.array_equal(group['latitude'][:],masks['latitude']) or not np.array_equal(group['longitude'][:],masks['longitude']):raise ValueError('Source native coordinates changed')
    hours=sorted({i for p in periods for i in range(p['start'],p['end_inclusive']+1)})
    coords=group['valid_time'][:]
    if not np.array_equal(coords[hours],np.asarray(hours)):raise ValueError('Source time coordinates or gaps mismatch')
    status=zarr.open_group(session.store,path='single/temporal/status',mode='r')
    flags={v:{str(int(k)):int(n) for k,n in zip(*np.unique(status[v][:][hours],return_counts=True))} for v in VARS}
    versions={str(int(k)):int(n) for k,n in zip(*np.unique(status['expver'][:][hours],return_counts=True))}
    if any(f!={'0':len(hours)} for f in flags.values()) or versions!={'1':len(hours)}:raise ValueError('Required source hours are not final valid records')
    evidence={'snapshot_id':pre['snapshot_id'],'selected_hours_with_boundary':len(hours),'variable_status_counts':flags,'expver_counts':versions,
      'source_units':{v:dict(group[v].attrs)['units'] for v in VARS},'time_gaps':0,'coordinate_exact_matches_frozen_mask':True,
      'source_variable_attributes':{v:dict(group[v].attrs) for v in VARS}}
    if evidence['source_units']!={'t2m':'K','d2m':'K','sp':'Pa','u10':'m s**-1','v10':'m s**-1','tp':'m','msl':'Pa'}:raise ValueError('Native source units changed')
    immutable(ROOT/'source_quality_metadata.json',json.dumps(evidence,indent=2).encode())
    rows=await inventory(session,pre,tiles);total=sum(r['byte_length'] for r in rows)
    summary={'snapshot_id':pre['snapshot_id'],'chunk_count':len(rows),'temporal_tiles':tiles,'source_compressed_bytes':total,
      'source_compressed_bytes_by_variable':{v:sum(r['byte_length'] for r in rows if r['variable']==v) for v in VARS},'climate_byte_ceiling':CAP,
      'ledger_sha256':sha((ROOT/'source_chunk_ledger.jsonl').read_bytes()),'contract_sha256':sha((ROOT/'frozen_recovery_contract.json').read_bytes())}
    immutable(ROOT/'source_inventory_summary.json',json.dumps(summary,indent=2).encode());print('EXACT_SOURCE_BYTES',json.dumps(summary),flush=True)
    if total>CAP:raise SystemExit(f'Exact required chunks {total} exceed frozen climate-byte ceiling {CAP}')
    if not execute:return
    started=time.monotonic();records=[]
    for i in range(0,len(rows),2):
        records.extend(await asyncio.gather(*(asyncio.to_thread(fetch,r) for r in rows[i:i+2])))
        if i%28==0:print('RAW_CACHE',min(i+2,len(rows)),len(rows),'bytes',sum(r['byte_length'] for r in records),'elapsed_s',round(time.monotonic()-started,1),flush=True)
    immutable(ROOT/'raw_retrieval_manifest.jsonl',(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records)).encode())
    derived=ROOT/'derived';derived.mkdir(exist_ok=True);nw=weights[:,rr,cc]
    ncfiles={};hourly_main={};monthly={p['month']:{} for p in periods};checks=[];rangechecks=[]
    for p in periods:
        nc=netCDF4.Dataset(derived/f"native_hourly_{p['month']}.nc",'w',format='NETCDF4')
        nc.createDimension('valid_time',p['hours']+1);nc.createDimension('cell',len(rr));nc.createDimension('client',5)
        tv=nc.createVariable('valid_time','i8',('valid_time',));tv[:]=np.arange(p['start'],p['end_inclusive']+1);tv.units='hours since 1940-01-01 00:00:00';tv.calendar='proleptic_gregorian'
        nc.createVariable('latitude','f8',('cell',))[:]=masks['latitude'][rr];nc.createVariable('longitude','f8',('cell',))[:]=masks['longitude'][cc]
        nc.createVariable('native_latitude_index','i4',('cell',))[:]=rr;nc.createVariable('native_longitude_index','i4',('cell',))[:]=cc
        nc.createVariable('normalized_client_area_weight','f8',('client','cell'))[:]=nw
        nc.source_snapshot_id=pre['snapshot_id'];nc.country_geometry_sha256=pre['geo_sha256'];nc.client_names=json.dumps(clients)
        nc.precipitation_semantics=contract['precipitation_time_semantics'];nc.instantaneous_semantics=contract['instantaneous_time_semantics'];ncfiles[p['month']]=nc
    for var in VARS:
        selected={p['month']:np.full((p['hours']+1,len(rr)),np.nan,dtype=np.float32) for p in periods}
        main_hourly={p['month']:np.zeros((p['hours']+1,5),dtype=np.float64) for p in periods}
        for row in (r for r in rows if r['variable']==var):
            t,y,x=row['chunk_indices'];raw=(ROOT/row['raw_path']).read_bytes();data=pcodec.simple_decompress(raw)
            if data.dtype!=np.float32 or data.size!=np.prod(CHUNK):raise ValueError('Native payload float32/shape contract failed')
            data=data.reshape(CHUNK);idx=np.where((rr>=y*12)&(rr<y*12+12)&(cc>=x*12)&(cc<x*12+12))[0]
            for p in periods:
                a=max(p['start'],t*CHUNK[0]);b=min(p['end_inclusive']+1,(t+1)*CHUNK[0])
                if a>=b:continue
                sub=data[a-t*CHUNK[0]:b-t*CHUNK[0],rr[idx]-y*12,cc[idx]-x*12]
                if not np.isfinite(sub).all() or (np.abs(sub)>3e38).any():raise ValueError(f'Invalid native source cell {var} {p["month"]}')
                if var=='tp' and (sub<0).any():raise ValueError('Negative native precipitation; explicit source review required')
                selected[p['month']][a-p['start']:b-p['start'],idx]=sub
                main_hourly[p['month']][a-p['start']:b-p['start']]+=sub.astype(np.float64)@nw[:,idx].T
        for p in periods:
            m=p['month'];data=selected[m]
            if not np.isfinite(data).all():raise ValueError(f'Incomplete frozen country-cell support {var} {m}')
            use=data[1:] if var=='tp' else data[:-1]
            cellmonthly=use.sum(axis=0,dtype=np.float64) if var=='tp' else use.mean(axis=0,dtype=np.float64)
            independent=np.asarray([math.fsum(float(value)*float(weight) for value,weight in zip(cellmonthly,nw[i])) for i in range(5)])
            main=main_hourly[m][1:].sum(axis=0,dtype=np.float64) if var=='tp' else main_hourly[m][:-1].mean(axis=0,dtype=np.float64)
            if not np.allclose(main,independent,rtol=2e-13,atol=2e-12):raise ValueError('Independent spatial/temporal aggregation disagreement')
            for i,c in enumerate(clients):checks.append({'month':m,'client':c,'variable':var,'main':float(main[i]),'cell_then_fsum':float(independent[i]),'absolute_difference':float(abs(main[i]-independent[i]))})
            rangechecks.append({'month':m,'variable':var,'native_minimum':float(use.min()),'native_maximum':float(use.max()),'valid_hours':p['hours'],'valid_native_cells':len(rr),'all_finite':True,'missing_cells':0})
            out=ncfiles[m].createVariable(var,'f4',('valid_time','cell'),zlib=True,complevel=4,shuffle=True,chunksizes=(24,len(rr)));out[:]=data;out.units=evidence['source_units'][var]
            monthly[m][var]=main;hourly_main[m+'_'+var]=main_hourly[m]
        print('DECODED',var,'all7months complete',flush=True)
    for nc in ncfiles.values():nc.close()
    output=[];canonical=[]
    for p in periods:
        m=p['month']
        for i,c in enumerate(clients):
            row={'client':c,'month':m,'source':'ERA5','provider':'Earthmover pinned public Icechunk','source_snapshot_id':pre['snapshot_id'],
              'geometry_area_m2':float(areas[i]),'geometry_cells':contract['native_counts'][c],'valid_area_m2':float(areas[i]),'valid_cells':contract['native_counts'][c],
              'weighted_coverage':1.,'expected_records':p['hours'],'state_records':p['hours'],'flow_records':p['hours'],'state_temporal_coverage':1.,
              'source_interval_start':p['source_interval_start'],'source_interval_end':p['source_interval_end']}
            row.update({NATIVE[v]:float(monthly[m][v][i]) for v in VARS});output.append(row)
            canonical.append({'client':c,'date':m+'-01',**{OLD[v]:float(monthly[m][v][i]) for v in VARS},'era5_tp_mm_per_month':float(monthly[m]['tp'][i]*1000)})
    pd.DataFrame(output).to_csv(derived/'monthly_source_rows.csv',index=False,float_format='%.17g')
    pd.DataFrame(canonical).to_csv(derived/'monthlycomparisonexpected.csv',index=False,float_format='%.17g')
    pd.DataFrame(checks).to_csv(derived/'independent_aggregation_check.csv',index=False,float_format='%.17g')
    pd.DataFrame(rangechecks).to_csv(derived/'native_support_unit_checks.csv',index=False,float_format='%.17g')
    np.savez_compressed(derived/'regional_hourly.npz',**hourly_main)
    receipt={**summary,'status':'NATIVE_RECOVERY_PASS_COMPATIBILITY_PENDING','rows':len(output),'independent_checks':len(checks),
      'max_independent_absolute_difference':max(c['absolute_difference'] for c in checks),'all_native_cells_all_required_hours_valid':True,
      'source_flags':evidence['variable_status_counts'],'source_expver':evidence['expver_counts'],'imputed_cells':0,
      'elapsed_seconds':time.monotonic()-started,'output_files':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in derived.iterdir() if p.is_file()}}
    (derived/'native_recovery_receipt.json').write_text(json.dumps(receipt,indent=2));print('NATIVE_RECOVERY_COMPLETE',json.dumps(receipt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args();asyncio.run(main(args.execute))
