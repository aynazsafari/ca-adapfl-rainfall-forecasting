"""Verify, archive or explicitly stage the reviewed public source whitelist."""
from pathlib import Path
import argparse, hashlib, json, subprocess, zipfile

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'metadata/source_manifest.json'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def checked():
 manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));paths=[];seen=set()
 for entry in manifest['files']:
  name=entry['path']
  if name in seen or '\\' in name or ':' in name or Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Invalid/duplicate source identity')
  seen.add(name);original=ROOT/name;path=original.resolve()
  if name=='metadata/source_manifest.json':raise RuntimeError('Source manifest self file must be excluded from files list')
  if not path.is_relative_to(ROOT) or path==ROOT or original.is_symlink():raise RuntimeError('Escaping/symlink source identity')
  if any(parent.is_symlink() for parent in original.parents if parent.is_relative_to(ROOT)):raise RuntimeError('Source symlink directory refused')
  forbidden={'runs','run','work','evaluation','private_inputs_DO_NOT_UPLOAD','imd_import','imd_prepared_rebuild'}
  if forbidden.intersection(Path(name).parts) or name.endswith(('.nc','.nc4','.zip')):raise RuntimeError('Private/result content in source whitelist: '+name)
  if 'imd_supplement' in Path(name).name or name=='benchmarks/accepted/tools/derived/imd/manifest.json':raise RuntimeError('IMD values must not be deposited')
  if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise RuntimeError('Changed deposited source: '+name)
  if path.stat().st_size>=50*1024**2:raise RuntimeError('Git source file exceeds conservative50MiB boundary')
  paths.append(name)
 paths.append('metadata/source_manifest.json')
 return paths

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 sub.add_parser('verify')
 s=sub.add_parser('stage');s.add_argument('--execute',action='store_true');s.add_argument('--init-git',action='store_true')
 s=sub.add_parser('archive');s.add_argument('--output',type=Path,required=True)
 a=p.parse_args();paths=checked()
 if a.action=='verify':print('SOURCE_WHITELIST_PASS',len(paths),'files');return
 if a.action=='stage':
  if not a.execute:print('Verified staging preview:',len(paths),'explicit source files; no mutation.');return
  if not (ROOT/'.git').exists():
   if not a.init_git:raise RuntimeError('Use --init-git with --execute for a new local repository')
   subprocess.run(['git','init','-b','main',str(ROOT)],check=True)
  # Force only the authenticated whitelist past nested frozen ignore rules.
  for offset in range(0,len(paths),50):subprocess.run(['git','-C',str(ROOT),'add','-f','--',*paths[offset:offset+50]],check=True)
  tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode('utf-8').split('\0')
  unexpected=set(filter(None,tracked))-set(paths)
  if unexpected:raise RuntimeError('Unexpected tracked files; inspect manually: '+str(sorted(unexpected)))
  print('STAGED_SOURCE_WHITELIST',len(paths),'files; no commit, push or remote write.')
 else:
  destination=a.output.resolve()
  if destination.is_relative_to(ROOT/'benchmarks'):raise RuntimeError('Archive cannot be written into frozen benchmark roots')
  destination.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
   for name in paths:out.write(ROOT/name,'CAAdapFL/'+name)
  print('SOURCE_ARCHIVE_PASS',destination.name,destination.stat().st_size,sha(destination))

if __name__=='__main__':main()
