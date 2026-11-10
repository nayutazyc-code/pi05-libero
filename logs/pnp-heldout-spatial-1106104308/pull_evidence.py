import base64
import hashlib
import io
import json
import pathlib
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone

remote = '/root/autodl-tmp/logs/pnp-heldout-spatial-1106104308'
local = pathlib.Path('/example/project/logs/pnp-heldout-spatial-1106104308')
wrapper = 'ssh-wrapper'
script = '''import hashlib,json,pathlib
root=pathlib.Path(%r)
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file():
  files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps({'root':str(root),'files':files},indent=2))
''' % remote
inventory_bytes = subprocess.check_output([wrapper, '/root/autodl-tmp/openpi/.venv/bin/python -c ' + shlex.quote(script)])
inventory = json.loads(inventory_bytes)
print('REMOTE_INVENTORY files=%d bytes=%d' % (len(inventory['files']), sum(p['bytes'] for p in inventory['files'])), flush=True)
encoded = subprocess.check_output([wrapper, 'set -o pipefail; tar -czf - -C ' + shlex.quote(remote) + ' . | base64 -w0'])
archive = base64.b64decode(encoded, validate=True)
print('DOWNLOADED compressed_bytes=%d' % len(archive), flush=True)
expected = {entry['path']: entry for entry in inventory['files']}
extracted = set()
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
 for member in bundle:
  relative = pathlib.PurePosixPath(member.name)
  assert not relative.is_absolute() and '..' not in relative.parts, member.name
  if member.isdir():
   continue
  assert member.isfile(), member.name
  name = str(relative)
  assert name in expected and name not in extracted, name
  contents = bundle.extractfile(member).read()
  entry = expected[name]
  assert len(contents) == entry['bytes'] and hashlib.sha256(contents).hexdigest() == entry['sha256'], name
  dest = local / name
  dest.parent.mkdir(parents=True, exist_ok=True)
  if not dest.exists() or dest.read_bytes() != contents:
   dest.write_bytes(contents)
  extracted.add(name)
assert extracted == set(expected)
for name, entry in expected.items():
 p = local / name
 assert p.stat().st_size == entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'], name
(local / 'remote-inventory.json').write_bytes(inventory_bytes)
report = {'source': remote, 'transport': wrapper + ' (configured numeric IP)', 'finished_utc': datetime.now(timezone.utc).isoformat(), 'files_verified': len(extracted), 'source_bytes': sum(p['bytes'] for p in expected.values()), 'archive_sha256': hashlib.sha256(archive).hexdigest(), 'all_files_match_remote_sha256': True, 'remote_operations': 'read-only inventory and tar; no evaluation or training', 'deletions': 0}
(local / 'transfer-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
