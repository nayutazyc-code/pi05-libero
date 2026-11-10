# Exp B libero_spatial eval

- Config: `pi05_libero_lora_action_only` / `phase1_exp_b_action_lora`
- Checkpoint: `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999`
- Protocol: libero_spatial, 10 tasks × 50 trials, seed 7 (same as Phase1 / Phase0)
- Result: **315/500 (63.0%)**
- Exit: `EXP_B_EVAL_EXIT 0` @ 2026-11-05T04:28:02+08:00
- Baseline Phase1 LoRA (`pi05_libero_lora` / `phase1_lora_baseline` / 29999): **496/500 (99.2%)**
- Delta vs Phase1 baseline: **−181 / 500 (−36.2 pp)**
- Training: `TRAIN_B_EXIT 0` (action-expert LoRA only; VLM LoRA frozen)
- Log: `/root/autodl-tmp/logs/phase1-exp-b-eval-spatial.log`
- Workdir evidence: `logs/phase1-exp-b-eval-spatial-summary.txt`, `logs/phase1-exp-b-status.md`

## Per-task (50 trials each)

| # | Success | Rate | Task |
|---|---------|------|------|
| 01 | 21/50 | 42% | black bowl between plate and ramekin → plate |
| 02 | 36/50 | 72% | black bowl next to ramekin → plate |
| 03 | 42/50 | 84% | black bowl from table center → plate |
| 04 | 39/50 | 78% | black bowl on cookie box → plate |
| 05 | 19/50 | 38% | black bowl in top drawer → plate |
| 06 | 27/50 | 54% | black bowl on ramekin → plate |
| 07 | 38/50 | 76% | black bowl next to cookie box → plate |
| 08 | 33/50 | 66% | black bowl on stove → plate |
| 09 | 29/50 | 58% | black bowl next to plate → plate |
| 10 | 31/50 | 62% | black bowl on wooden cabinet → plate |
| **Σ** | **315/500** | **63.0%** | |

## Notes
- No retrain requested; this task is eval → acceptance → reporter only.
- Night gate auto-pass authorized by user; do not restart training.
