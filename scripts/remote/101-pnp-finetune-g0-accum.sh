#!/usr/bin/env bash
# G0 effective batch32 = micro8 x 4; preserve previous OOM runs and existing G1.
set -euo pipefail
source /root/autodl-tmp/env.sh
source /root/.config/pi05/wandb.env
cd /root/autodl-tmp/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
export PNP_MICROBATCH_SIZE=8
LOG=/root/autodl-tmp/logs/pnp-g0-accum-mb8-train.log
echo "TRAIN_G0_ACCUM_START $(date -Iseconds) effective_batch=32 microbatch=8 accum=4 optimizer_updates=2500 wandb=$WANDB_MODE" | tee -a "$LOG"
set +e
.venv/bin/python scripts/train_pnp_grad_accum.py pi05_libero_lora_pnp \
  --exp-name pnp_g0_accum_bs32_mb8 \
  --batch-size 32 --num-train-steps 2500 \
  --log-interval 50 --save-interval 2500 --keep-period 2500 >> "$LOG" 2>&1
rc=$?
set -e
echo "TRAIN_G0_ACCUM_EXIT $rc $(date -Iseconds)" | tee -a "$LOG"
exit "$rc"
