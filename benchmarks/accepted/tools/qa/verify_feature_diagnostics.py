"""Bounded independent fsum check of saved TRAIN-only correlations."""
from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.package_root), Path(a.output)
    def rows(name):
        with (out/name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    dictionary = rows('feature_dictionary.csv')
    profiles, pairs, high = rows('profile_by_client_and_global.csv'), rows('pairwise_correlations.csv'), rows('high_correlation_pairs_abs_ge_0p95.csv')
    assert (len(dictionary),len(profiles),len(pairs),len(high)) == (43,258,5418,99)
    assert sum(d['core_member']=='True' for d in dictionary)==14
    assert sum(d['climate_member']=='True' for d in dictionary)==21
    symmetry = []
    for scope in sorted({r['scope'] for r in pairs}):
        matrix = rows('correlation_matrix_'+scope+'.csv')
        z = np.array([[float(r[d['feature']]) for d in dictionary] for r in matrix])
        n = int(next(r['n_complete'] for r in pairs if r['scope']==scope))
        max_diag = float(np.max(np.abs(np.diag(z)-1)))
        bound = n*np.finfo(np.float64).eps
        assert np.array_equal(z,z.T) and max_diag<=bound and np.isfinite(z).all() and np.max(np.abs(z))<=1
        symmetry.append(dict(scope=scope,exact_symmetry=True,diagonal_max_abs_error=max_diag,roundoff_bound_n_epsilon=bound))
    archive_path=root/'runs/revision_v2/datasets/MAIN_FULL_fit.npz'
    auth=json.loads((out/'authentication.json').read_text())
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest()==auth['files']['datasets/MAIN_FULL_fit.npz']
    clients=[r['scope'] for r in symmetry if r['scope']!='GLOBAL_POOLED']
    x={}
    with np.load(archive_path,allow_pickle=False) as ar:
        for c in clients:
            dates=ar[c+'::raw_target_month']
            mask=dates<='2007-11'
            assert int(mask.sum())==299
            x[c]=ar[c+'::scaled_x'][mask,-1,:].astype(np.float64)
    x['GLOBAL_POOLED']=np.concatenate([x[c] for c in clients])
    names=[d['feature'] for d in dictionary]
    checks=[]
    requests=[(c,'era5_tp_mm','era5land_tp_mm') for c in clients]+[('GLOBAL_POOLED','era5_tp_mm','era5land_tp_mm'),('GLOBAL_POOLED','nino34_anomaly','oni_anomaly_3mo')]
    for scope,f1,f2 in requests:
        u=x[scope][:,names.index(f1)].tolist();v=x[scope][:,names.index(f2)].tolist()
        n=len(u);um=math.fsum(u)/n;vm=math.fsum(v)/n
        du=[z-um for z in u];dv=[z-vm for z in v]
        independent=math.fsum(a*b for a,b in zip(du,dv))/math.sqrt(math.fsum(a*a for a in du)*math.fsum(b*b for b in dv))
        stored=float(next(r['pearson_r'] for r in pairs if r['scope']==scope and r['feature_a']==f1 and r['feature_b']==f2))
        error=abs(independent-stored)
        assert error <= n*np.finfo(np.float64).eps
        checks.append(dict(scope=scope,feature_a=f1,feature_b=f2,independent_math_fsum_r=independent,stored_r=stored,absolute_error=error))
    receipt=dict(status='PASS',counts={'dictionary':43,'profiles':258,'pairs':5418,'high_pairs':99,'scopes':6},
                 matrix_checks=symmetry,independent_fsum_checks=checks,labels_read=False,evaluation_files_read=False,
                 note='A preliminary fixed 1e-14 diagonal assertion was overly strict for pooled BLAS arithmetic (observed 1.28e-14); final verification uses the conservative n*float64-epsilon summation bound. Scientific |r|>=0.95 threshold is unchanged.')
    (out/'output_verification_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob('*')) if f.is_file() and f.name!='file_hashes.json'}
    (out/'file_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
