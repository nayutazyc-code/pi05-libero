#!/usr/bin/env bash
# Exp B action-LoRA eval: ckpt 29999, pi05_libero_lora_action_only, libero_spatial, 10x50, seed 7
set -eo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
CFG=pi05_libero_lora_action_only
CKPT=$W/openpi/checkpoints/pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999
LOG=$W/logs/phase1-exp-b-eval-spatial.log
mkdir -p $W/logs $W/logs/videos_exp_b_spatial
cd $W/openpi
pkill -f "serve_policy.py.*LIBERO" 2>/dev/null || true
sleep 2
.venv/bin/python scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=$CFG --policy.dir=$CKPT > $W/logs/phase1-exp-b-server.log 2>&1 &
SP=$!
trap "kill $SP 2>/dev/null || true" EXIT
for i in $(seq 1 360); do
  grep -q "server listening\|Uvicorn running\|Creating server" $W/logs/phase1-exp-b-server.log 2>/dev/null \
    && (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break
  sleep 10
done
echo "server ready after $i polls" | tee -a "$LOG"
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero
{
  echo "EXP_B_EVAL_START $(date -Iseconds)"
  echo "CFG=$CFG CKPT=$CKPT seed=7 trials=50 suite=libero_spatial"
  python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 50 \
    --args.seed 7 --args.video-out-path $W/logs/videos_exp_b_spatial
  echo EVAL_DONE
  echo "EXP_B_EVAL_EXIT 0 $(date -Iseconds)"
} 2>&1 | tee -a "$LOG"
