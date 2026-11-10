"""Execute the fixed saved-results notebook after independent study acceptance."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();repo=args.repo.resolve();out=args.output.resolve()
    run=repo/'runs/revision_v2';completion=run/'evaluation/complete.json'
    qa_path=run/'independent_qa/complete.json'
    if not completion.is_file() or not qa_path.is_file():
        ap.error('Completed evaluation and independent PASS receipt required; notebook was not opened or executed.')
    identity=read(run/'run_identity.json');qa=read(qa_path)
    assert qa['status']=='PASS' and qa['run_sha256']==identity['run_sha256']
    assert qa['evaluation_complete_sha256']==sha(completion) and qa.get('scope')
    source=repo/'notebooks/inspect_results.ipynb'
    assert sha(source)=='80c650a1173baca33efdfb5d6183f80c041197f502cdb93a0ae2582f5c0f884d'
    assert not out.exists(),'Preserve previous notebook execution; use a fresh directory'
    out.mkdir(parents=True)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    nb=nbformat.read(source,as_version=4)
    nbformat.validate(nb)
    specs=out/'temporary_kernel_specs';kernel=specs/'caadapfl-task'
    kernel.mkdir(parents=True)
    specification={'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
      'display_name':'CAAdapFL task-local verification','language':'python',
      'env':{'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
             'NUMEXPR_NUM_THREADS':'2','MPLBACKEND':'module://matplotlib_inline.backend_inline'}}
    (kernel/'kernel.json').write_text(json.dumps(specification,indent=2),encoding='utf-8')
    ksm=KernelSpecManager(kernel_dirs=[str(specs)],ensure_native_kernel=False,
                         allowed_kernelspecs={'caadapfl-task'})
    assert set(ksm.find_kernel_specs())=={'caadapfl-task'}
    km=KernelManager(kernel_name='caadapfl-task',kernel_spec_manager=ksm)
    client=NotebookClient(nb,km=km,kernel_name='caadapfl-task',timeout=900,
                         allow_errors=False,resources={'metadata':{'path':str(repo)}})
    started=time.perf_counter()
    try:
        executed=client.execute()
    finally:
        try:
            if km.has_kernel:km.shutdown_kernel(now=True)
        finally:
            if client.kc is not None:client.kc.stop_channels()
    assert not km.has_kernel
    assert client.kc is None or not client.kc.channels_running
    nbformat.validate(executed)
    assert all(o.get('output_type')!='error' for c in executed.cells for o in c.get('outputs',[]))
    code=[c for c in executed.cells if c.cell_type=='code']
    assert len(code)==7 and all(c.execution_count is not None for c in code)
    assert sha(source)=='80c650a1173baca33efdfb5d6183f80c041197f502cdb93a0ae2582f5c0f884d'
    saved=out/'inspect_results_executed.ipynb';nbformat.write(executed,saved)
    receipt={'status':'PASS','scope':'Actual execution of all seven fixed read-only saved-result notebook cells',
      'run_sha256':identity['run_sha256'],'evaluation_complete_sha256':sha(completion),
      'independent_qa_sha256':sha(qa_path),'source_notebook_sha256':sha(source),
      'executed_notebook_sha256':sha(saved),'elapsed_seconds':time.perf_counter()-started,
      'code_cells_executed':len(code),'error_outputs':0,'temporary_kernel_only':True,
      'global_kernel_registration_performed':False,'kernel_shutdown_confirmed':not km.has_kernel,
      'kernel_client_channels_stopped':client.kc is None or not client.kc.channels_running,
      'models_fitted_or_inferred':False,'source_notebook_unchanged':True,
      'python':sys.version,'verification_wrapper_sha256':sha(__file__)}
    (out/'execution_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['status','code_cells_executed','elapsed_seconds','source_notebook_unchanged']}))

if __name__=='__main__':main()
