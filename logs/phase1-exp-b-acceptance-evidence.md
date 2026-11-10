# Experiment B — acceptance evidence (2026-11-05 Asia/Shanghai)

## Scope
Acceptance for **Experiment B eval only** (action-expert LoRA). Training already finished (`TRAIN_B_EXIT 0`). Do **not** retrain.

## Numbers
- Exp B spatial: **315/500 (63.0%)**
- Phase1 LoRA baseline (control): **496/500 (99.2%)**
- Exit: `EXP_B_EVAL_EXIT 0` @ 2026-11-05T04:28:02+08:00
- Remote verify (box SSH `ssh-wrapper`): `Total success rate: 0.63`; ckpt dir exists at `.../phase1_exp_b_action_lora/29999`

## Artifacts
| Item | Path |
|------|------|
| Result summary | workdir `logs/phase1-exp-b-spatial-result.md` |
| Eval summary extract | workdir `logs/phase1-exp-b-eval-spatial-summary.txt` |
| Status | workdir `logs/phase1-exp-b-status.md` |
| Remote eval log | `/root/autodl-tmp/logs/phase1-exp-b-eval-spatial.log` |
| Checkpoint | `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999` |
| Phase1 baseline result | workdir `logs/phase1-spatial-result.md` (496/500) |

## Protocol pins (unchanged vs Phase1)
- Suite: libero_spatial; 10×50; seed 7
- Config name: `pi05_libero_lora_action_only` / exp `phase1_exp_b_action_lora`
- Init: `pi05_base`; freeze all except action-expert LoRA `.*llm.*_1.*lora.*`

## User authorization
Night gate review auto-pass; executor only forwards results; no retrain.
See `logs/phase1-exp-b-user-waiver.md`.
