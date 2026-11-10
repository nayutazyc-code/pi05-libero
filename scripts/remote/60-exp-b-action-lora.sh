#!/usr/bin/env bash
# Experiment B: action-expert LoRA only; freeze VLM LoRA. Same data/budget/protocol as baseline.
set -euo pipefail
source /root/autodl-tmp/env.sh
cd "$W/openpi"
L=$W/logs
CFG=pi05_libero_lora_action_only
EXP=phase1_exp_b_action_lora

echo "EXP_B_SETUP $(date -Is)" | tee -a $L/phase1-exp-b.log

# --- restore-safe cache clean (keep lerobot data, pi05_base, uvpy, final ckpt 29999) ---
echo "CLEAN_START $(date -Is)" | tee -a $L/phase1-exp-b-clean.log
df -h /root/autodl-tmp | tee -a $L/phase1-exp-b-clean.log
# duplicate HF datasets parquet cache (train uses lerobot path)
rm -rf /root/autodl-tmp/cache/hf/datasets/parquet || true
# intermediate ckpt 25000 (keep final 29999 only)
rm -rf /root/autodl-tmp/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/25000 || true
# tiny eval videos ok to keep; clear tmp
rm -rf /root/autodl-tmp/tmp/* || true
df -h /root/autodl-tmp | tee -a $L/phase1-exp-b-clean.log
echo "CLEAN_DONE $(date -Is)" | tee -a $L/phase1-exp-b-clean.log

# --- add TrainConfig if missing ---
.venv/bin/python - <<'PY'
from pathlib import Path
p = Path("src/openpi/training/config.py")
s = p.read_text()
name = 'pi05_libero_lora_action_only'
if f'name="{name}"' in s:
    print("CONFIG_EXISTS", name)
else:
    if "import openpi.shared.nnx_utils as nnx_utils" not in s:
        s = s.replace(
            "import openpi.shared.normalize as _normalize",
            "import openpi.shared.normalize as _normalize\nimport openpi.shared.nnx_utils as nnx_utils",
        )
    block = '''
    # Experiment B: train action-expert LoRA only; freeze VLM/PaliGemma LoRA (+ bases).
    # Same data/budget/lr as pi05_libero_lora baseline; do not change loss or data.
    TrainConfig(
        name="pi05_libero_lora_action_only",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        # Freeze everything except action-expert LoRA params (.*llm.*_1.*lora.*).
        freeze_filter=nnx.Not(nnx_utils.PathRegex(r".*llm.*_1.*lora.*")),
        ema_decay=None,
    ),
'''
    anchor = 'name="pi05_libero_lora"'
    i = s.find(anchor)
    if i < 0:
        raise SystemExit("anchor pi05_libero_lora not found")
    # insert after the closing of that TrainConfig: find the "),\n    #" after the ema_decay=None of that block
    j = s.find("ema_decay=None,\n    ),", i)
    if j < 0:
        raise SystemExit("end of pi05_libero_lora block not found")
    j = s.find("\n", j + len("ema_decay=None,\n    ),"))
    s = s[: j + 1] + block + s[j + 1 :]
    p.write_text(s)
    print("CONFIG_ADDED", name)

# smoke: config importable
import sys
sys.path.insert(0, "src")
from openpi.training import config as cfg
c = cfg.get_config(name)
print("CONFIG_OK", c.name, "freeze", c.freeze_filter)
PY

# reuse norm_stats from baseline config name (same data)
mkdir -p "assets/$CFG/physical-intelligence/libero"
cp -n assets/pi05_libero_lora/physical-intelligence/libero/norm_stats.json \
  "assets/$CFG/physical-intelligence/libero/norm_stats.json" || \
  cp assets/pi05_libero_lora/physical-intelligence/libero/norm_stats.json \
  "assets/$CFG/physical-intelligence/libero/norm_stats.json"
echo "ASSETS_OK"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "TRAIN_B_START $(date -Is)" | tee -a $L/phase1-exp-b.log
.venv/bin/python scripts/train.py "$CFG" \
  --exp-name "$EXP" \
  --overwrite \
  --batch-size 16 \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 1000 \
  --keep-period 5000 \
  >> $L/phase1-exp-b.log 2>&1
echo "TRAIN_B_EXIT $? $(date -Is)" | tee -a $L/phase1-exp-b.log
