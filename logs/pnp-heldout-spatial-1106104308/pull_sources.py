"""Read-only supplemental source capture; never import a simulator or policy."""
import base64
import hashlib
import json
import pathlib
import shlex
import subprocess

root = pathlib.Path('/example/project/logs/pnp-heldout-spatial-1106104308')
script = '''import base64,hashlib,json,pathlib
import torch
root=pathlib.Path('/root/autodl-tmp/logs/pnp-heldout-spatial-1106104308')
p=json.loads((root/'protocol.json').read_text())
m=json.loads((root/'heldout/manifest.json').read_text())
sources=[('/root/autodl-tmp/datasets/pnp_policy_spatial_center/summary.json',m['collection_summary_sha256']),
         ('/root/autodl-tmp/scripts/collect_pnp_policy_expert.py',m['collector_sha256'])]
sources += [(x['path'],x['sha256']) for x in p['source_files']]
sources += [(x['path'],x['sha256']) for g in p['groups'].values() for x in g['normalization']]
sources += [(str(pathlib.Path('/root/autodl-tmp/scripts')/name),sha) for name,sha in p['eval_scripts_sha256'].items()]
sources += [(t['init_file'],t['init_file_sha256']) for t in m['tasks']]
seen=set(); files=[]
for name,expected in sources:
 if name in seen: continue
 seen.add(name); raw=pathlib.Path(name).read_bytes(); sha=hashlib.sha256(raw).hexdigest()
 assert sha==expected,name
 files.append({'source':name,'sha256':sha,'bytes':len(raw),'base64':base64.b64encode(raw).decode()})
task=m['tasks'][2]
states=torch.load(task['init_file'],map_location='cpu',weights_only=False)
used={'source':task['init_file'],'source_sha256':task['init_file_sha256'],
      'indices':list(range(31)),'states':[[float(x) for x in state] for state in states[:31]]}
print(json.dumps({'files':files,'collection_initial_states':used},allow_nan=False))
'''
payload = json.loads(subprocess.check_output([
    'ssh-wrapper',
    '/root/autodl-tmp/openpi/.venv/bin/python -c ' + shlex.quote(script),
]))
report = []
for entry in payload['files']:
    name = pathlib.PurePosixPath(entry['source']).relative_to('/root/autodl-tmp')
    raw = base64.b64decode(entry.pop('base64'), validate=True)
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    destination = root / 'source-snapshots' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    report.append(dict(entry, local=str(destination.relative_to(root))))
(root / 'source-snapshots.json').write_text(json.dumps(report, indent=2) + '\n')
(root / 'collection-used-initial-states.json').write_text(json.dumps(payload['collection_initial_states'], indent=2, allow_nan=False) + '\n')
print('SOURCE_SNAPSHOTS_VERIFIED', len(report))
