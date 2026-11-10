#!/usr/bin/env bash
# Scripted expert PnP collection: libero_spatial task_id=2, LeRobot-compatible npy episodes
set -eo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
OUT=$W/datasets/pnp_scripted_spatial_center
LOG=$W/logs/pnp-collect.log
mkdir -p $W/logs $OUT $W/scripts
cd $W/openpi
export PYTHONPATH=$PWD/third_party/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
{
  echo "PNP_COLLECT_START $(date -Iseconds)"
  echo "dataset_ref=lerobot/libero_spatial_image task_id=2 seed=7 n=40"
  echo "OUT=$OUT"
  /root/autodl-tmp/openpi/examples/libero/.venv/bin/python $W/scripts/collect_pnp_scripted.py \
    --out "$OUT" --num-episodes 40 --seed 7 --max-steps 400
  echo "PNP_COLLECT_EXIT $? $(date -Iseconds)"
} 2>&1 | tee -a "$LOG"
