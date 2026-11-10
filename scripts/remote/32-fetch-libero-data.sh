#!/usr/bin/env bash
# Phase1: resumable download of physical-intelligence/libero (LeRobot) via hf-mirror, with retries.
source /root/autodl-tmp/env.sh; cd $W/openpi
for n in $(seq 1 30); do
  .venv/bin/huggingface-cli download physical-intelligence/libero --repo-type dataset \
    --local-dir $W/cache/hf/lerobot/physical-intelligence/libero --max-workers 8 && { echo DATA_DONE; exit 0; }
  echo "retry $n"; sleep 20
done
echo DATA_FAILED
