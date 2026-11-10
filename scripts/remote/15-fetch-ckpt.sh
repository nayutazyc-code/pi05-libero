#!/usr/bin/env bash
# Download gs://openpi-assets/checkpoints/pi05_libero to $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero (openpi cache layout)
set -eo pipefail
source /root/autodl-tmp/env.sh
D=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
python3 - "$D" <<'PY'
import json,os,sys,urllib.request,urllib.parse,subprocess
D=sys.argv[1];pre='checkpoints/pi05_libero/';items=[];tok=None
while True:
    u='https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix='+pre+('&pageToken='+tok if tok else '')
    j=json.load(urllib.request.urlopen(u,timeout=60));items+=j.get('items',[]);tok=j.get('nextPageToken')
    if not tok:break
tot=sum(int(i['size']) for i in items);print('files',len(items),'GB',tot/1e9,flush=True)
for i in items:
    n=i['name'];p=os.path.join(D,n[len(pre):])
    if n.endswith('/'):continue
    if os.path.exists(p) and os.path.getsize(p)==int(i['size']):continue
    os.makedirs(os.path.dirname(p),exist_ok=True)
    url='https://storage.googleapis.com/openpi-assets/'+urllib.parse.quote(n)
    for k in range(10):
        if subprocess.call(['curl','-sSfL','--retry','5','-o',p,url])==0 and os.path.getsize(p)==int(i['size']):break
    else: sys.exit('FAILED '+n)
print('CKPT_DONE',flush=True)
PY
du -sh $D
