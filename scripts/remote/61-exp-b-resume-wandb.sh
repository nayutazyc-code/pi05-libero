#!/usr/bin/env bash
# Resume experiment B with wandb + loss logging enabled.
set -euo pipefail
source /root/autodl-tmp/env.sh
# shellcheck disable=SC1091
source /root/.config/pi05/wandb.env
cd /root/autodl-tmp/openpi
L=/root/autodl-tmp/logs
CFG=pi05_libero_lora_action_only
EXP=phase1_exp_b_action_lora
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Prefer online wandb; fail clearly if key/network broken.
export WANDB_MODE="${WANDB_MODE:-online}"
echo "RESUME_B_WANDB_START $(date -Is) key_set=$([ -n "${WANDB_API_KEY:-}" ] && echo yes || echo no)" | tee -a $L/phase1-exp-b-wandb.log
.venv/bin/python scripts/train.py "$CFG" \
  --exp-name "$EXP" \
  --resume --no-overwrite \
  --batch-size 16 \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 1000 \
  --keep-period 5000 \
  >> $L/phase1-exp-b-wandb.log 2>&1
echo "TRAIN_B_EXIT $? $(date -Is)" | tee -a $L/phase1-exp-b-wandb.log
