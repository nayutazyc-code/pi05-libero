#!/usr/bin/env bash
# Small FT from Exp A 29999 on cleaned local PnP set. G0 batch=32, G1 batch=16.
# MUST enable wandb (offline ok). Same step count for both.
set -euo pipefail
source /root/autodl-tmp/env.sh
# shellcheck disable=SC1091
source /root/.config/pi05/wandb.env
W=/root/autodl-tmp
L=$W/logs
CFG=pi05_libero_lora_pnp
REPO=local/pnp_policy_spatial_center
A_PARAMS=$W/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/29999/params
STEPS=${PNP_FT_STEPS:-2500}
SAVE_INTERVAL=${PNP_SAVE_INTERVAL:-500}
KEEP_PERIOD=${PNP_KEEP_PERIOD:-2500}
mkdir -p "$L"
cd "$W/openpi"

echo "PNP_FT_SETUP $(date -Iseconds) steps=$STEPS" | tee -a "$L/pnp-finetune.log"
df -h "$W" | tee -a "$L/pnp-finetune.log"

# Free Exp C intermediate ckpts if present (keep 29999)
CROOT=$W/openpi/checkpoints/pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora
if [[ -d "$CROOT" ]]; then
  for s in 5000 10000 15000 20000 25000; do
    if [[ -d "$CROOT/$s" ]]; then
      echo "DEL_EXP_C_INTERMEDIATE $s" | tee -a "$L/pnp-finetune.log"
      rm -rf "$CROOT/$s"
    fi
  done
fi
df -h "$W" | tee -a "$L/pnp-finetune.log"

# Inject TrainConfig if missing (pass values via env)
export PNP_REPO="$REPO"
export PNP_A_PARAMS="$A_PARAMS"
export PNP_STEPS="$STEPS"
.venv/bin/python <<'PY'
import os
from pathlib import Path
p = Path("src/openpi/training/config.py")
s = p.read_text()
name = "pi05_libero_lora_pnp"
repo = os.environ["PNP_REPO"]
a_params = os.environ["PNP_A_PARAMS"]
steps = int(os.environ["PNP_STEPS"])
if f'name="{name}"' in s:
    print("CONFIG_EXISTS", name)
else:
    block = f"""
    # A-PnP small FT: same LoRA recipe as pi05_libero_lora, local cleaned PnP demos,
    # weight_loader = Exp A final 29999 params.
    TrainConfig(
        name="{name}",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="{repo}",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=5e-5, decay_steps={steps}, decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("{a_params}"),
        num_train_steps={steps},
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
"""
    anchor = 'name="pi05_libero_lora_vlm_only"'
    i = s.find(anchor)
    if i < 0:
        anchor = 'name="pi05_libero_lora"'
        i = s.find(anchor)
    if i < 0:
        raise SystemExit("anchor TrainConfig not found")
    needle = "ema_decay=None,\n    ),"
    j = s.find(needle, i)
    if j < 0:
        raise SystemExit("end of anchor block not found")
    j = s.find("\n", j + len(needle))
    s = s[: j + 1] + block + s[j + 1 :]
    p.write_text(s)
    print("CONFIG_ADDED", name)

import sys
sys.path.insert(0, "src")
from openpi.training import config as cfg
c = cfg.get_config(name)
print("CONFIG_OK", c.name, "batch", c.batch_size, "steps", c.num_train_steps, "repo", c.data.repo_id)
PY

echo "NORM_START $(date -Iseconds)" | tee -a "$L/pnp-finetune.log"
.venv/bin/python scripts/compute_norm_stats.py --config-name "$CFG" 2>&1 | tee -a "$L/pnp-norm.log"
echo "NORM_EXIT $? $(date -Iseconds)" | tee -a "$L/pnp-finetune.log"
ls -la "assets/$CFG/$REPO/" 2>&1 | tee -a "$L/pnp-finetune.log"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- G0: official batch=32 ---
EXP0=pnp_g0_bs32
echo "TRAIN_G0_START $(date -Iseconds) batch=32 steps=$STEPS wandb=$WANDB_MODE" | tee -a "$L/pnp-finetune.log"
set +e
.venv/bin/python scripts/train.py "$CFG" \
  --exp-name "$EXP0" \
  --overwrite \
  --batch-size 32 \
  --num-train-steps "$STEPS" \
  --log-interval 50 \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  >> "$L/pnp-g0-train.log" 2>&1
G0_RC=$?
set -e
echo "TRAIN_G0_EXIT $G0_RC $(date -Iseconds)" | tee -a "$L/pnp-finetune.log" | tee -a "$L/pnp-g0-train.log"
if [[ $G0_RC -ne 0 ]]; then
  if grep -qiE "out of memory|RESOURCE_EXHAUSTED|OOM|MEMORY" "$L/pnp-g0-train.log"; then
    echo "G0_OOM_RECORDED batch=32 failed; will still run G1 batch=16" | tee -a "$L/pnp-finetune.log"
  else
    echo "G0_FAIL non-OOM rc=$G0_RC" | tee -a "$L/pnp-finetune.log"
  fi
fi
df -h "$W" | tee -a "$L/pnp-finetune.log"
# drop intermediate of this small run if any (keep final only)
G0ROOT=$W/openpi/checkpoints/$CFG/$EXP0
if [[ -d "$G0ROOT" ]]; then
  final=$(ls -1 "$G0ROOT" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
  for d in "$G0ROOT"/*; do
    base=$(basename "$d")
    if [[ "$base" =~ ^[0-9]+$ && "$base" != "$final" ]]; then
      echo "DEL_G0_INTERMEDIATE $base" | tee -a "$L/pnp-finetune.log"
      rm -rf "$d"
    fi
  done
fi

# --- G1: batch=16 only ---
EXP1=pnp_g1_bs16
echo "TRAIN_G1_START $(date -Iseconds) batch=16 steps=$STEPS wandb=$WANDB_MODE" | tee -a "$L/pnp-finetune.log"
set +e
.venv/bin/python scripts/train.py "$CFG" \
  --exp-name "$EXP1" \
  --overwrite \
  --batch-size 16 \
  --num-train-steps "$STEPS" \
  --log-interval 50 \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  >> "$L/pnp-g1-train.log" 2>&1
G1_RC=$?
set -e
echo "TRAIN_G1_EXIT $G1_RC $(date -Iseconds)" | tee -a "$L/pnp-finetune.log" | tee -a "$L/pnp-g1-train.log"
G1ROOT=$W/openpi/checkpoints/$CFG/$EXP1
if [[ -d "$G1ROOT" ]]; then
  final=$(ls -1 "$G1ROOT" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
  for d in "$G1ROOT"/*; do
    base=$(basename "$d")
    if [[ "$base" =~ ^[0-9]+$ && "$base" != "$final" ]]; then
      echo "DEL_G1_INTERMEDIATE $base" | tee -a "$L/pnp-finetune.log"
      rm -rf "$d"
    fi
  done
fi

# export loss CSV + plots
.venv/bin/python <<'PY'
import re, csv
from pathlib import Path
L = Path("/root/autodl-tmp/logs")
plot = L / "plots"
plot.mkdir(parents=True, exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have_plt = True
except Exception as e:
    print("PLOT_SKIP_IMPORT", e)
    have_plt = False

for tag, logname, title in [
    ("g0", "pnp-g0-train.log", "PnP G0 batch=32 loss"),
    ("g1", "pnp-g1-train.log", "PnP G1 batch=16 loss"),
]:
    log = L / logname
    rows = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"Step\s+(\d+):\s+.*?\bloss=([0-9.eE+-]+)", line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    d = {s: l for s, l in rows}
    rows = sorted(d.items())
    csv_path = plot / f"pnp_{tag}_loss.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"]); w.writerows(rows)
    print("CSV", csv_path, "N", len(rows))
    if have_plt and rows:
        xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, lw=1.2)
        plt.xlabel("step"); plt.ylabel("loss"); plt.title(title)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        png = plot / f"pnp_{tag}_loss_curve.png"
        plt.savefig(png, dpi=120); plt.close()
        print("PNG", png)
print("LOSS_EXPORT_DONE")
PY

echo "PNP_FT_ALL_DONE $(date -Iseconds)" | tee -a "$L/pnp-finetune.log"
df -h "$W" | tee -a "$L/pnp-finetune.log"
