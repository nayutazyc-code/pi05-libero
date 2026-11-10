#!/usr/bin/env bash
# Read-only training-machine inventory. No installation, download, or cleanup.
export PATH=/root/miniconda3/bin:$PATH
set -euo pipefail

printf '=== timestamp ===\n'
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf '=== operating system ===\n'
uname -a
cat /etc/os-release
printf '=== data disk ===\n'
if [[ ! -d /root/autodl-tmp ]]; then
    printf 'STOP: /root/autodl-tmp does not exist\n' >&2
    exit 20
fi
df -B1 /root/autodl-tmp
findmnt -T /root/autodl-tmp
printf '=== memory ===\n'
free -b
printf '=== GPU and active GPU processes ===\n'
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
printf '=== available tools ===\n'
for tool in python3 uv conda git curl docker; do
    if command -v "$tool"; then
        "$tool" --version
    else
        printf '%s: unavailable\n' "$tool"
    fi
done
printf '=== storage stop rule ===\n'
python3 - <<'PY'
import shutil
import sys

disk = shutil.disk_usage('/root/autodl-tmp')
projected_bytes = 105 * 1024**3
minimum_free = 20 * 1024**3
print(f'total_bytes={disk.total} used_bytes={disk.used} free_bytes={disk.free}')
print(f'planned_additional_bytes={projected_bytes}')
if disk.free - projected_bytes < minimum_free or (disk.used + projected_bytes) / disk.total > 0.85:
    print('STOP: planned peak exceeds the approved storage budget', file=sys.stderr)
    sys.exit(21)
PY
printf '=== network endpoint HTTP status (no assets downloaded) ===\n'
for endpoint in https://github.com https://storage.googleapis.com/openpi-assets/ https://huggingface.co https://hf-mirror.com; do
    # A bucket root may return 403; retain the status for diagnosis.
    curl --silent --show-error --location --head --connect-timeout 10 --max-time 20 \
        --output /dev/null --write-out '%{url_effective} %{http_code}\n' "$endpoint"
done
