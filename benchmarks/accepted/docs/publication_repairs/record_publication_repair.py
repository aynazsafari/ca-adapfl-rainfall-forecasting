from pathlib import Path
import datetime
import difflib
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'outputs/03_benchmark/caadapfl_benchmark'
DIR=REPO/'docs/publication_repairs'
RUN=REPO/'runs/revision_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parent=DIR/'pre_actual_qa_make_figures.py'
source=REPO/'tools/make_figures.py'
old=b"    close(graphs['graphs']['enriched'], np.mean(np.asarray(components, float), axis=0), 'enriched equal-component mean')"
new=(b"    require(np.array_equal(np.asarray(graphs['graphs']['enriched'], float),\n"
     b"                           np.mean(np.asarray(components, float), axis=0).astype(np.float32).astype(float)),\n"
     b"            'Enriched graph differs from the equal-component mean serialized as float32')")
before=parent.read_bytes()
assert sha(parent)=='316a77b750ba5703532874976cfb6ca3f4c4a32e22dc67a44685573d98c92b09'
if b'\r\n' in before:new=new.replace(b'\n',b'\r\n')
assert before.count(old)==1
expected=before.replace(old,new)
assert expected.replace(b'\r\n',b'\n')==source.read_bytes().replace(b'\r\n',b'\n')
source.write_bytes(expected)
assert source.read_bytes()==expected
helper=REPO/'tools/make_publication_companions.py'
helper_parent=DIR/'pre_actual_qa_make_publication_companions.py'
if b'\r\n' in helper_parent.read_bytes():
    helper.write_bytes(helper.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
plan=REPO/'tools/layout_plan.json'
assert sha(plan)==sha(DIR/'pre_actual_qa_layout_plan.json')=='7c62f5c40d4f1d2a55c3f8d763dd6dae2b50c2cb3b9fcb61bc3e21884ff200a0'
identity=json.loads((RUN/'run_identity.json').read_text())
for name,digest in identity['code'].items():assert sha(REPO/'caadapfl'/(name+'.py'))==digest
graph_path=RUN/'datasets/MAIN_graphs.json'
graph=json.loads(graph_path.read_text())
prepared=json.loads((RUN/'prepared.json').read_text())
prepared_files={k.replace('\\','/'):v for k,v in prepared['files'].items()}
assert sha(graph_path)==prepared_files['datasets/MAIN_graphs.json']
mean=np.mean(np.asarray(list(graph['components'].values()),float),axis=0)
saved=np.asarray(graph['graphs']['enriched'],float)
assert np.array_equal(saved,mean.astype(np.float32).astype(float))
receipt={
    'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Post-evaluation plotting verification/provenance repair only; exact float32 serialization check. No metric, graph, data, fit, plot choice, plate order, comparison family or layout changes.',
    'original_plotter_sha256':sha(parent),'repaired_plotter_sha256':sha(source),
    'original_companion_helper_sha256':sha(helper_parent),'repaired_companion_helper_sha256':sha(helper),
    'frozen_layout_plan_sha256':sha(plan),'frozen_layout_plan_unchanged':True,
    'exact_allowed_plotter_byte_replacement_count':1,
    'scientific_modules_unchanged':True,'scientific_module_sha256':identity['code'],
    'run_sha256':identity['run_sha256'],'prepared_graph_sha256':sha(graph_path),
    'saved_equals_float32_component_mean_exactly':True,
    'maximum_unserialized_float64_difference':float(np.max(np.abs(saved-mean))),
    'model_fitting_or_inference':False,
    'companion_change':'Gate verifies this receipt, original source and exact allowed plotter byte delta; original layout PLAN_SHA256 remains unchanged.',
    'writer_sha256':sha(Path(__file__)),
}
target=DIR/'graph_precision_repair.json'
assert not target.exists()
target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
for current,original,label in [(source,parent,'make_figures'),(helper,helper_parent,'companion_gate')]:
    diff=''.join(difflib.unified_diff(original.read_text().splitlines(keepends=True),current.read_text().splitlines(keepends=True),fromfile='pre_actual_qa/'+original.name,tofile=current.name))
    (DIR/(label+'.diff')).write_text(diff,encoding='utf-8')
print(json.dumps({'status':'PASS','repair_receipt_sha256':sha(target),'plotter_sha256':sha(source),'companion_helper_sha256':sha(helper),'layout_unchanged':True}))
