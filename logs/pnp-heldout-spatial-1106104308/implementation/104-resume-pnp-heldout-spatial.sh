#!/usr/bin/env bash
# Error-gate-approved Py3.8 environment capture, then preflight and formal eval.
set -euo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
RUN=$W/logs/pnp-heldout-spatial-1106104308
cd "$W/openpi"
mkdir -p "$RUN"
exec 8>"$RUN/.resume.lock"
flock -n 8
trap 'rc=$?; echo "ERROR_GATE rc=$rc phase=environment-or-evaluation line=$LINENO"; exit "$rc"' ERR

# Keep the original failed capture. Publish a new complete file only on success.
ENV_CAPTURE=$(mktemp "$RUN/environment-compatible.XXXXXX")
date -Iseconds > "$ENV_CAPTURE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "$ENV_CAPTURE"
df -h "$W" >> "$ENV_CAPTURE"
for interpreter in .venv/bin/python examples/libero/.venv/bin/python; do
  "$interpreter" - >> "$ENV_CAPTURE" <<'PY'
import importlib.metadata as metadata
import json
import sys

versions = {}
for name in ("numpy", "imageio", "jax", "jaxlib", "mujoco", "torch", "websockets", "robosuite", "bddl"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(sys.executable, sys.version)
print(json.dumps(versions, sort_keys=True))
PY
done
mv "$ENV_CAPTURE" "$RUN/environment-compatible.txt"
echo "ENVIRONMENT_CAPTURE_DONE"
bash "$W/scripts/103-pnp-heldout-spatial.sh" preflight
bash "$W/scripts/103-pnp-heldout-spatial.sh" formal
