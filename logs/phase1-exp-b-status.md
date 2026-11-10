# Experiment B status

- Started: 2026-11-04 ~07:53 Asia/Shanghai
- Config: `pi05_libero_lora_action_only` (freeze all except action-expert LoRA `.*llm.*_1.*lora.*`)
- Exp: `phase1_exp_b_action_lora`
- Batch 16, steps 30000, save 1000 / keep 5000, init `pi05_base`
- Log: `/root/autodl-tmp/logs/phase1-exp-b.log`
- Script: `scripts/remote/60-exp-b-action-lora.sh`
- Baseline Phase1 spatial: 496/500 (99.2%) @ ckpt 29999
