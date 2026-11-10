"""Create paired portable archives and verify every archive member."""
from pathlib import Path
import hashlib,json,uuid,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def members(roots):
    result=[]
    for rel in roots:
        p=ROOT/rel
        for q in ([p] if p.is_file() else sorted(p.rglob('*'))):
            if q.is_file() and '__pycache__' not in q.parts and q.suffix!='.pyc':
                if q.name in {'setup_experiment.py','update_report_app.py','polish_review_copy.py'}:continue # One-time workspace authoring depends on prior artifacts.
                result.append(q)
    assert len(result)==len(set(result))
    return sorted(result)

def archive(name,paths):
    target=OUT/name
    entries={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    manifest=dict(schema_version=1,root='information_sharing_benchmark',archive=name,members=entries,
        scope='Logical source/result separation; original datasets and frozen receipts retained')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,'information_sharing_benchmark/'+p.relative_to(ROOT).as_posix())
        z.writestr('information_sharing_benchmark/'+name.replace('.zip','_MEMBERS.json'),json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for rel,entry in entries.items():
            b=z.read('information_sharing_benchmark/'+rel)
            assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
    return dict(file=name,sha256=sha(target),bytes=target.stat().st_size,member_count=len(entries)+1)

def main():
    assert json.loads((ROOT/'qa/independent_numeric_qa.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'qa/state_prediction_replay.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'qa/notebook_execution.json').read_text())['status']=='PASS'
    source=members(['README.md','requirements.txt','protocol.json','analysis_amendment_v1.json','ANALYSIS_ORIGIN.json',
                    'PARENT_EVIDENCE_BINDING.json','datasets','caadapfl','tools'])
    results=members(['run','evaluation','qa','figures','notebook','reports'])
    assert not {p.relative_to(ROOT) for p in source}&{p.relative_to(ROOT) for p in results}
    a=archive('Information_Sharing_SOURCE.zip',source)
    b=archive('Information_Sharing_RESULTS.zip',results)
    # Short Windows paths avoid legacy MAX_PATH limits in nested state directories.
    parent=Path.home()/'Documents/Codex'/('sharecheck_'+uuid.uuid4().hex[:8]);parent.mkdir()
    for item in [a,b]:
        with zipfile.ZipFile(OUT/item['file']) as z:z.extractall(parent)
    combined=parent/'information_sharing_benchmark'
    for p in source+results:
        assert sha(combined/p.relative_to(ROOT))==sha(p)
    fresh_parent=Path.home()/'Documents/Codex'/('sharesource_'+uuid.uuid4().hex[:8]);fresh_parent.mkdir()
    with zipfile.ZipFile(OUT/a['file']) as z:z.extractall(fresh_parent)
    fresh=fresh_parent/'information_sharing_benchmark'
    assert not (fresh/'run').exists() and not (fresh/'evaluation').exists()
    receipt=dict(status='ARCHIVE_INTEGRITY_PASS_PENDING_PORTABLE_REPLAY',archives=[a,b],
        extracted_member_hashes_match=True,combined_root=str(combined),source_only_root=str(fresh),
        no_fits_in_packaging=True)
    save(OUT/'release_integrity.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
