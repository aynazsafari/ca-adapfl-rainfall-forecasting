"""Execute the companion in an isolated temporary kernel registration; no fitting."""
from pathlib import Path
import argparse, json, sys, tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve()
 if out.exists():raise FileExistsError('Use a fresh executed notebook output')
 if out.is_relative_to(ROOT/'benchmarks'):raise RuntimeError('Notebook output must stay outside frozen evidence')
 notebook=nbformat.read(ROOT/'notebooks/Reproduce_Paper.ipynb',as_version=4)
 for cell in notebook.cells:
  if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
 with tempfile.TemporaryDirectory(prefix='caadapfl_kernel_') as folder:
  kernel=Path(folder)/'caadapfl';kernel.mkdir()
  (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'CAAdapFL reference','language':'python'}),encoding='utf-8')
  manager=KernelSpecManager(kernel_dirs=[folder])
  km=KernelManager(kernel_name='caadapfl',kernel_spec_manager=manager)
  NotebookClient(notebook,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x',encoding='utf-8') as stream:nbformat.write(notebook,stream)
 print('NOTEBOOK_EXECUTED',sum(c.cell_type=='code' for c in notebook.cells),'code cells; zero fits')
if __name__=='__main__':main()
