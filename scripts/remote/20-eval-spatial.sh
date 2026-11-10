#!/usr/bin/env bash
# Phase0 frozen protocol: official pi05_libero ckpt, libero_spatial, 10 tasks x 50 trials = 500, seed 7,
# openpi examples/libero/main.py defaults (replan_steps=5, num_steps_wait=10, resize 224). Pass >= 494/500.
set -eo pipefail
source /root/autodl-tmp/env.sh
cd $W/openpi
.venv/bin/python scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config=pi05_libero \
  --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero > $W/logs/server.log 2>&1 &
SP=$!; trap "kill $SP" EXIT
for i in $(seq 1 360); do grep -q "server listening\|Uvicorn running\|Creating server" $W/logs/server.log 2>/dev/null && (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null && break; sleep 10; done
echo "server ready after $i polls"
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero
python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 50 \
  --args.seed 7 --args.video-out-path $W/logs/videos_spatial
echo EVAL_DONE
