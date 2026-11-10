#!/usr/bin/env bash
# Phase1: fetch official gs://openpi-assets/checkpoints/pi05_base (12.4GB) via parallel HTTP range requests
# (single-stream GCS ~0.4MB/s on AutoDL). Verify each file vs official GCS md5Hash/crc32c.
set -eo pipefail
source /root/autodl-tmp/env.sh
NAME=${1:-pi05_base}
D=$OPENPI_DATA_HOME/openpi-assets/checkpoints/$NAME
/root/miniconda3/bin/python3 -m pip install -q google-crc32c
/root/miniconda3/bin/python3 - "$D" "$NAME" <<'PY'
import json,os,sys,urllib.request,urllib.parse,hashlib,base64,subprocess
from concurrent.futures import ThreadPoolExecutor
import google_crc32c
D,NAME=sys.argv[1],sys.argv[2];pre=f'checkpoints/{NAME}/';items=[];tok=None
while True:
    j=json.load(urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix='+pre+('&pageToken='+tok if tok else ''),timeout=60))
    items+=j.get('items',[]);tok=j.get('nextPageToken')
    if not tok:break
print('files',len(items),'GB',sum(int(i['size']) for i in items)/1e9,flush=True)
def ok(p,i):
    h=hashlib.md5() if 'md5Hash' in i else google_crc32c.Checksum()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<24),b''):h.update(b)
    return base64.b64encode(h.digest()).decode()==i.get('md5Hash',i.get('crc32c'))
CH=64<<20
def part(a):
    url,p,s,e=a
    for k in range(20):
        if os.path.exists(p) and os.path.getsize(p)==e-s+1:return
        subprocess.call(['curl','-sSfL','--retry','5','-m','900','-r',f'{s}-{e}','-o',p,url])
    raise SystemExit('part failed '+p)
pool=ThreadPoolExecutor(32)
for i in items:
    n=i['name'];r=n[len(pre):]
    if not r or n.endswith('/'):continue
    p=os.path.join(D,r);os.makedirs(os.path.dirname(p),exist_ok=True);sz=int(i['size'])
    if os.path.exists(p) and os.path.getsize(p)==sz and ok(p,i):print('OK(cached)',r,flush=True);continue
    url='https://storage.googleapis.com/openpi-assets/'+urllib.parse.quote(n)
    if sz==0: open(p,'wb').close()
    else:
        parts=[(url,f'{p}.part{k}',s,min(s+CH,sz)-1) for k,s in enumerate(range(0,sz,CH))]
        list(pool.map(part,parts))
        with open(p,'wb') as o:
            for a in parts:
                with open(a[1],'rb') as f:
                    for b in iter(lambda:f.read(1<<24),b''):o.write(b)
                os.remove(a[1])
    print(('OK ' if ok(p,i) else 'BAD ')+r,flush=True)
print('FETCH_DONE',flush=True)
PY
grep -c '^BAD' $W/logs/phase1-base-fetch.log 2>/dev/null || true
