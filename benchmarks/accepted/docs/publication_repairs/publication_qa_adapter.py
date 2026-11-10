"""Textual gate view; preserve the original independent audit byte-for-byte."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/03_benchmark/caadapfl_benchmark/runs/revision_v2'
DIR=RUN/'independent_qa'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=DIR/'complete.json'
before=sha(original)
assert before=='c379f9b16af248f6c34f709767ff0b66274789057c4a0bcf9e2d0e4b3cbe1d3a'
receipt=json.loads(original.read_text(encoding='utf-8'))
assert receipt['status']=='PASS' and isinstance(receipt['scope'],dict) and receipt['scope']
assert receipt['evaluation_complete_sha256']==sha(RUN/'evaluation/complete.json')
assert receipt['run_sha256']==json.loads((RUN/'run_identity.json').read_text())['run_sha256']
adapted=dict(receipt)
adapted['scope_structured']=receipt['scope']
adapted['scope']='Independent saved-artifact audit: 96 prediction identities, 283 unique fits, all keyed regression/event metrics and PR curves, exact paired common-date bootstrap draws at 12/6/24 months, sign enumeration/Holm31, checkpoint provenance and deduplicated cost/message receipts. No model fitting or inference.'
adapted['gate_view_provenance']={
    'original_audit_receipt':'complete.json',
    'original_audit_receipt_sha256':before,
    'purpose':'Serialize a nonempty textual scope for existing publication gates; original detailed scope retained unchanged in scope_structured.',
    'new_scientific_calculations':False,
    'adapter_sha256':sha(Path(__file__)),
}
target=DIR/'publication_gate.json'
assert not target.exists()
target.write_text(json.dumps(adapted,indent=2)+'\n',encoding='utf-8')
assert sha(original)==before
assert json.loads(target.read_text())['scope_structured']==receipt['scope']
weights=ROOT/'work/real_run_qa/completed_revision_v2/selected_primary_caaa_weights.json'
assert sha(weights)=='03b2aa3cd32b1a9a023a6bedde79e5499131cd2af13735f12c94c9b16116bd42'
shutil.copy2(weights,DIR/weights.name)
assert sha(DIR/weights.name)==sha(weights)
print(json.dumps({'status':'PASS','publication_gate_sha256':sha(target),'original_qa_unchanged':sha(original)==before}))
