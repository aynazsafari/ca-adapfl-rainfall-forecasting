"""Portable textual publication view of a completed independent PASS receipt.

Original audit bytes and structured scope are retained. No metrics are computed.
"""
from pathlib import Path
import argparse,hashlib,json

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--audit',type=Path,help='Default: RUN/independent_qa/complete.json')
    ap.add_argument('--output',type=Path,help='Default: RUN/independent_qa/publication_gate.json')
    args=ap.parse_args();run=args.run.resolve()
    source=args.audit or run/'independent_qa/complete.json'
    target=args.output or run/'independent_qa/publication_gate.json'
    before=sha(source);original=json.loads(source.read_text(encoding='utf-8'))
    if original.get('status')!='PASS' or not original.get('scope'):
        raise ValueError('A complete independent PASS receipt with scope is required')
    identity=json.loads((run/'run_identity.json').read_text(encoding='utf-8'))
    if original.get('run_sha256')!=identity['run_sha256']:
        raise ValueError('Independent audit and run identity differ')
    if original.get('evaluation_complete_sha256')!=sha(run/'evaluation/complete.json'):
        raise ValueError('Independent audit does not bind the exact completed evaluation')
    if target.exists():raise FileExistsError('Use a fresh output; never replace an accepted receipt')
    adapted=dict(original)
    adapted['scope_structured']=original['scope']
    adapted['scope']='Independent saved-artifact audit of the completed run, with exact detailed scope retained in scope_structured. No new scientific calculations, model fitting or inference.'
    adapted['gate_view_provenance']={
        'original_audit_receipt':source.name,'original_audit_receipt_sha256':before,
        'purpose':'Nonempty textual scope for publication consumers; original scope retained unchanged.',
        'new_scientific_calculations':False,'adapter_sha256':sha(Path(__file__)),
    }
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(adapted,indent=2)+'\n',encoding='utf-8')
    if sha(source)!=before:raise RuntimeError('Original audit changed unexpectedly')
    print(json.dumps({'status':'PASS','output':str(target),'original_audit_sha256':before,'publication_gate_sha256':sha(target)}))

if __name__=='__main__':main()
