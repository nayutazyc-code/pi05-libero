# Phase1 LoRA libero_spatial 评测结果

- Time: 2026-11-04 07:12 Asia/Shanghai
- Suite: libero_spatial
- Checkpoint: `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/29999`
- Config: pi05_libero_lora / phase1_lora_baseline
- Episodes: 500
- Successes: **496**
- Rate: **99.2%** (0.992)
- Exit: `EVAL_DONE` / `PHASE1_EVAL_EXIT 0` @ 2026-11-04T07:12:10+08:00
- Remote log: `/root/autodl-tmp/logs/phase1-eval-spatial.log`
- Videos: `/root/autodl-tmp/logs/videos_phase1_spatial` (13 files)
- Phase0 baseline for reference: 495/500 (99.0%)

## Next (user-authorized, do not wait)
1. Acceptance + reporter (user-facing numbers)
2. Clear autodl-tmp caches; keep final ckpt 29999 (+25000 if needed for B)
3. Start experiment B: action-expert-only LoRA, freeze VLM LoRA
