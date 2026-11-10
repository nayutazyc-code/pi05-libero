#!/usr/bin/env bash
# Continue the authorized mb8 run without restarting training or A/G1 evaluation.
set -euo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
TRAIN_LOG=$W/logs/pnp-g0-accum-mb8-train.log
CKPT=$W/openpi/checkpoints/pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499

until grep -q '^TRAIN_G0_ACCUM_EXIT ' "$TRAIN_LOG"; do
  sleep 15
done

cd "$W/openpi"
.venv/bin/python "$W/scripts/100-pnp-export-loss.py"
if grep -q '^TRAIN_G0_ACCUM_EXIT 0 ' "$TRAIN_LOG" && [[ -d "$CKPT/params" ]]; then
  PNP_EVAL_GROUPS="G0 G2" bash "$W/scripts/99-pnp-eval-g0-g1-g2.sh"
else
  echo "G0_MB8_FAILED: no G0/G2 evaluation; preserve A/G1 results"
  PNP_EVAL_GROUPS=" " bash "$W/scripts/99-pnp-eval-g0-g1-g2.sh"
fi
echo "PNP_MB8_FOLLOWUP_DONE $(date -Iseconds)"
