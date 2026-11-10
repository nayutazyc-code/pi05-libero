# Phase 1 plan (official-compatible baseline)

- Init route: **pi05_libero_lora** (honest name) — same as upstream `pi05_libero` (pi05=True, discrete_state_input=False, extra_delta_transform=False, weight from `pi05_base`) but LoRA variants + freeze_filter + no EMA + batch 32 for single 4090D 24GB (mirrors upstream `pi0_libero_low_mem_finetune` pattern). Full `pi05_libero` batch 256 not feasible on 24GB.
- Data: `physical-intelligence/libero` LeRobot via hf-mirror → `$HF_HOME` / openpi expected path.
- Steps: download pi05_base (GCS verify) + dataset → compute_norm_stats → short probe (≤100 steps) → freeze budget → nohup train 30k.
- Gate note (for ds-gpt-gate README/skill later): if 3 reviews still fail, modify + open new review round / reset budget instead of hard-stop. Applied going forward; Phase 0 already PASS this task.
