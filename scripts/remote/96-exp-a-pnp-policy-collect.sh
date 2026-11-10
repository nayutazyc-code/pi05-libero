#!/usr/bin/env bash
# A-policy expert PnP collection + later clean/eval hooks
set -eo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
CFG=pi05_libero_lora
CKPT=$W/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/29999
OUT=$W/datasets/pnp_policy_spatial_center
LOG=$W/logs/pnp-policy-collect.log
mkdir -p $W/logs $OUT $W/scripts
cd $W/openpi
pkill -f "serve_policy.py.*LIBERO" 2>/dev/null || true
pkill -f collect_pnp_policy_expert.py 2>/dev/null || true
sleep 2
.venv/bin/python scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=$CFG --policy.dir=$CKPT > $W/logs/pnp-a-server.log 2>&1 &
SP=$!
trap "kill $SP 2>/dev/null || true" EXIT
for i in $(seq 1 360); do
  grep -q "server listening\|Uvicorn running\|Creating server" $W/logs/pnp-a-server.log 2>/dev/null \
    && (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break
  sleep 5
done
echo "server ready after $i polls pid=$SP" | tee -a "$LOG"
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
{
  echo "PNP_POLICY_COLLECT_START $(date -Iseconds)"
  echo "CFG=$CFG CKPT=$CKPT replan=5 want_success=30 dataset_ref=lerobot/libero_spatial_image"
  python $W/scripts/collect_pnp_policy_expert.py \
    --out "$OUT" --num-success 30 --max-attempts 80 --seed 7 --replan-steps 5
  echo "PNP_POLICY_COLLECT_EXIT $? $(date -Iseconds)"
} 2>&1 | tee -a "$LOG"
