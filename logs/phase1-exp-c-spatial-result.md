# Exp C libero_spatial eval

- Config: `pi05_libero_lora_vlm_only` / `phase1_exp_c_vlm_lora`
- Checkpoint: `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora/29999`
- Protocol: libero_spatial, 10 tasks × 50 trials, seed 7 (same as Phase1 / Exp A / Exp B)
- Result: **456/500 (91.2%)**
- Exit: `EXP_C_EVAL_EXIT 0` @ 2026-11-05T23:30:27+08:00
- Control A Phase1 LoRA (`pi05_libero_lora` / `phase1_lora_baseline` / 29999): **496/500 (99.2%)**
- Control B action-only (`pi05_libero_lora_action_only` / `phase1_exp_b_action_lora` / 29999): **315/500 (63.0%)**
- Delta vs A: **−40 / 500 (−8.0 pp)**
- Delta vs B: **+141 / 500 (+28.2 pp)**
- Training: VLM LoRA only; action-expert LoRA frozen (symmetric ablation to Exp B)
- Log: `/root/autodl-tmp/logs/phase1-exp-c-eval-spatial.log`
- Workdir evidence: `logs/phase1-exp-c-eval-spatial-summary.txt`, `logs/phase1-exp-c-eval-spatial-tail.txt`

## Per-task (50 trials each)

| # | Success | Rate | Task |
|---|---------|------|------|
| 01 | 50/50 | 100% | black bowl between plate and ramekin → plate |
| 02 | 48/50 | 96% | black bowl next to ramekin → plate |
| 03 | 49/50 | 98% | black bowl from table center → plate |
| 04 | 49/50 | 98% | black bowl on cookie box → plate |
| 05 | 39/50 | 78% | black bowl in top drawer → plate |
| 06 | 41/50 | 82% | black bowl on ramekin → plate |
| 07 | 49/50 | 98% | black bowl next to cookie box → plate |
| 08 | 44/50 | 88% | black bowl on stove → plate |
| 09 | 46/50 | 92% | black bowl next to plate → plate |
| 10 | 41/50 | 82% | black bowl on wooden cabinet → plate |
| **Σ** | **456/500** | **91.2%** | |

## Notes
- No retrain requested; this task is eval → acceptance → reporter only.
- Night gate auto-pass authorized by user; do not restart training.
- Remote verify (SSH `ssh-wrapper`): `Total success rate: 0.912`; `Total episodes: 500`; ckpt dir `phase1_exp_c_vlm_lora/29999` exists.
