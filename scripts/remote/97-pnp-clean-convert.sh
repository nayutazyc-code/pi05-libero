#!/usr/bin/env bash
# Clean policy PnP demos + convert to local LeRobot; write stats md.
set -euo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
L=$W/logs
mkdir -p "$L"
cd "$W/openpi"
echo "PNP_CLEAN_START $(date -Iseconds)" | tee -a "$L/pnp-clean-convert.log"
.venv/bin/python "$W/scripts/clean_convert_pnp_lerobot.py" \
  --src "$W/datasets/pnp_policy_spatial_center" \
  --clean-out "$W/datasets/pnp_policy_spatial_center_clean" \
  --repo-id "local/pnp_policy_spatial_center" \
  --stats-md "$L/pnp-clean-stats.md" \
  2>&1 | tee -a "$L/pnp-clean-convert.log"
echo "PNP_CLEAN_EXIT $? $(date -Iseconds)" | tee -a "$L/pnp-clean-convert.log"
# also copy stats to box-visible path via stdout marker
wc -l "$L/pnp-clean-stats.md"
df -h "$W" | tee -a "$L/pnp-clean-convert.log"
