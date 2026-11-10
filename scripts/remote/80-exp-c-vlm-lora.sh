#!/usr/bin/env bash
# Experiment C: VLM/PaliGemma LoRA only; freeze action-expert LoRA. Symmetric to B.
# Must enable wandb + step loss (no --no-wandb-enabled). Use WANDB_MODE=offline if api.wandb.ai 403.
set -euo pipefail
source /root/autodl-tmp/env.sh
# shellcheck disable=SC1091
source /root/.config/pi05/wandb.env
cd "$W/openpi"
L=$W/logs
CFG=pi05_libero_lora_vlm_only
EXP=phase1_exp_c_vlm_lora
mkdir -p "$L"

echo "EXP_C_SETUP $(date -Is)" | tee -a "$L/phase1-exp-c.log"

# --- add TrainConfig if missing ---
.venv/bin/python - <<'PY'
from pathlib import Path
p = Path("src/openpi/training/config.py")
s = p.read_text()
name = "pi05_libero_lora_vlm_only"
if f'name="{name}"' in s:
    print("CONFIG_EXISTS", name)
else:
    if "import openpi.shared.nnx_utils as nnx_utils" not in s:
        s = s.replace(
            "import openpi.shared.normalize as _normalize",
            "import openpi.shared.normalize as _normalize\nimport openpi.shared.nnx_utils as nnx_utils",
        )
    block = '''
    # Experiment C: train VLM/PaliGemma LoRA only; freeze action-expert LoRA (+ bases).
    # Symmetric to B. Same data/budget/lr as pi05_libero_lora baseline; do not change loss or data.
    # Trainable: .*llm.*lora.* AND NOT .*llm.*_1.*  (10 VLM LoRA tensors; action uses *_1 / mlp_1).
    TrainConfig(
        name="pi05_libero_lora_vlm_only",
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
        freeze_filter=nnx.Not(
            nnx.All(
                nnx_utils.PathRegex(r".*llm.*lora.*"),
                nnx.Not(nnx_utils.PathRegex(r".*llm.*_1.*")),
            )
        ),
        ema_decay=None,
    ),
'''
    # Prefer insert after experiment B block; else after baseline.
    anchor = 'name="pi05_libero_lora_action_only"'
    i = s.find(anchor)
    if i < 0:
        anchor = 'name="pi05_libero_lora"'
        i = s.find(anchor)
    if i < 0:
        raise SystemExit("anchor TrainConfig not found")
    j = s.find("ema_decay=None,\n    ),", i)
    if j < 0:
        raise SystemExit("end of anchor TrainConfig block not found")
    j = s.find("\n", j + len("ema_decay=None,\n    ),"))
    s = s[: j + 1] + block + s[j + 1 :]
    p.write_text(s)
    print("CONFIG_ADDED", name)

import sys
sys.path.insert(0, "src")
from openpi.training import config as cfg
c = cfg.get_config(name)
print("CONFIG_OK", c.name, "freeze", c.freeze_filter)
PY

# reuse norm_stats from baseline (same data)
mkdir -p "assets/$CFG/physical-intelligence/libero"
cp -f assets/pi05_libero_lora/physical-intelligence/libero/norm_stats.json \
  "assets/$CFG/physical-intelligence/libero/norm_stats.json"
echo "ASSETS_OK"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Prefer offline when api.wandb.ai is blocked (training machine historically 403).
export WANDB_MODE="${WANDB_MODE:-offline}"
echo "TRAIN_C_START $(date -Is) wandb_mode=$WANDB_MODE key_set=$([ -n "${WANDB_API_KEY:-}" ] && echo yes || echo no)" \
  | tee -a "$L/phase1-exp-c.log"
.venv/bin/python scripts/train.py "$CFG" \
  --exp-name "$EXP" \
  --overwrite \
  --batch-size 16 \
  --num-train-steps 30000 \
  --log-interval 100 \
  --save-interval 1000 \
  --keep-period 5000 \
  >> "$L/phase1-exp-c.log" 2>&1
echo "TRAIN_C_EXIT $? $(date -Is)" | tee -a "$L/phase1-exp-c.log"
