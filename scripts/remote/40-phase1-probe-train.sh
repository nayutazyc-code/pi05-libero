#!/usr/bin/env bash
# Phase1 chain: wait downloads -> data audit -> compute_norm_stats (+compare to official) -> 50-step probe -> 30k LoRA train.
source /root/autodl-tmp/env.sh
cd $W/openpi
CFG=pi05_libero_lora
L=$W/logs
until grep -q FETCH_DONE $L/phase1-base-fetch.log && grep -q DATA_DONE $L/phase1-data-dl.log; do
  grep -q DATA_FAILED $L/phase1-data-dl.log && { echo CHAIN_ABORT data; exit 1; }
  sleep 60; done
grep -q '^BAD' $L/phase1-base-fetch.log && { echo CHAIN_ABORT base_md5; exit 1; }
echo "AUDIT_START $(date -Is)" > $L/phase1-audit.log
.venv/bin/python - >> $L/phase1-audit.log 2>&1 <<'PY'
import json,glob,os,pyarrow.parquet as pq,numpy as np
R=os.environ['W']+'/cache/hf/lerobot/physical-intelligence/libero'
info=json.load(open(R+'/meta/info.json'));print('info',{k:info[k] for k in ('total_episodes','total_frames','fps') if k in info})
fs=sorted(glob.glob(R+'/data/*/*.parquet'));print('parquet_files',len(fs))
bad=0;nonfin=0;eps=set()
for f in fs:
    try:t=pq.read_table(f)
    except Exception as e:bad+=1;print('BADFILE',f,e);continue
    d=t.to_pydict();eps.update(d['episode_index'])
    for k in ('state','actions'):
        if k in d:
            a=np.asarray(d[k],dtype=np.float32)
            nonfin+=int((~np.isfinite(a)).sum())
print('bad_files',bad,'nonfinite',nonfin,'episodes',len(eps))
print('AUDIT_OK' if bad==0 and nonfin==0 else 'AUDIT_FAIL')
PY
grep -q AUDIT_OK $L/phase1-audit.log || { echo CHAIN_ABORT audit; exit 1; }
.venv/bin/python scripts/compute_norm_stats.py --config-name $CFG > $L/phase1-norm.log 2>&1 || { echo CHAIN_ABORT norm; exit 1; }
.venv/bin/python - >> $L/phase1-norm.log 2>&1 <<'PY'
import json,glob,os,numpy as np
W=os.environ['W']
mine=glob.glob(W+'/openpi/assets/pi05_libero_lora/**/norm_stats.json',recursive=True)
off=W+'/cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json'
a=json.load(open(mine[0]))['norm_stats'];b=json.load(open(off))['norm_stats']
for k in a:
    for s in ('mean','std'):
        x=np.array(a[k][s]);y=np.array(b[k][s])[:len(x)]
        print('NORMCMP',k,s,'maxabsdiff',float(np.abs(x-y).max()))
PY
.venv/bin/python scripts/train.py $CFG --exp-name phase1_probe --overwrite --no-wandb-enabled \
  --num-train-steps 50 --log-interval 10 --save-interval 50 --keep-period 50 > $L/phase1-probe.log 2>&1 || { echo CHAIN_ABORT probe; exit 1; }
echo PROBE_DONE >> $L/phase1-probe.log
nvidia-smi --query-gpu=memory.used --format=csv >> $L/phase1-probe.log
.venv/bin/python scripts/train.py $CFG --exp-name phase1_lora_baseline --overwrite --no-wandb-enabled \
  --num-train-steps 30000 --save-interval 1000 --keep-period 5000 > $L/phase1-train.log 2>&1
echo "TRAIN_EXIT $? $(date -Is)" >> $L/phase1-train.log
