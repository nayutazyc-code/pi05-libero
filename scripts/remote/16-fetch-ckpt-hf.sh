#!/usr/bin/env bash
# GCS direct ~0.4MB/s from AutoDL. Fetch the same file list (from official gs://openpi-assets listing)
# from HF mirror ryanhoangt/pi05_libero via hf-mirror.com, verify each file md5 == official GCS md5.
set -eo pipefail
source /root/autodl-tmp/env.sh
D=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
pip install -q google-crc32c
python3 - "$D" <<'PY'
import json,os,sys,urllib.request,urllib.parse,hashlib,base64,subprocess
from concurrent.futures import ThreadPoolExecutor
D=sys.argv[1];pre='checkpoints/pi05_libero/';items=[];tok=None
while True:
    j=json.load(urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix='+pre+('&pageToken='+tok if tok else ''),timeout=60))
    items+=j.get('items',[]);tok=j.get('nextPageToken')
    if not tok:break
import google_crc32c
def md5(p,i=None):
    # official hash: md5Hash if present, else crc32c (composite objects)
    if 'md5Hash' in i:
        h=hashlib.md5()
    else:
        h=google_crc32c.Checksum()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<24),b''):h.update(b)
    d=h.digest()
    return base64.b64encode(d).decode()==i.get('md5Hash',i.get('crc32c'))
def get(i):
    r=i['name'][len(pre):]
    if not r or r.endswith('/'):return 'skip'
    p=os.path.join(D,r);os.makedirs(os.path.dirname(p),exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p)==int(i['size']) and md5(p,i):return 'OK(cached) '+r
    u='https://hf-mirror.com/ryanhoangt/pi05_libero/resolve/main/'+urllib.parse.quote(r)
    for k in range(10):
        subprocess.call(['curl','-sSfL','--retry','5','-C','-','-o',p,u])
        if os.path.exists(p) and os.path.getsize(p)==int(i['size']):
            return ('OK ' if md5(p,i) else 'MD5_MISMATCH ')+r
        if os.path.exists(p) and os.path.getsize(p)>int(i['size']): os.remove(p)
    return 'FAILED '+r
with ThreadPoolExecutor(6) as ex:
    res=list(ex.map(get,items))
for x in res: print(x,flush=True)
bad=[x for x in res if x.startswith(('MD5','FAILED'))];print('VERIFY_BAD',len(bad));sys.exit(1 if bad else 0)
PY
echo CKPT_VERIFIED_DONE
