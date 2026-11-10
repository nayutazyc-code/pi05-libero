# Experiment C — acceptance evidence (2026-11-06 Asia/Shanghai)

## Scope
Acceptance for **Experiment C eval only** (VLM LoRA only). Training already finished at ckpt 29999. Do **not** retrain.

## Numbers
- Exp C spatial: **456/500 (91.2%)**
- Control A Phase1 LoRA baseline: **496/500 (99.2%)** → Δ **−8.0 pp**
- Control B action-only: **315/500 (63.0%)** → Δ **+28.2 pp**
- Exit: `EXP_C_EVAL_EXIT 0` @ 2026-11-05T23:30:27+08:00
- Remote verify (box SSH `ssh-wrapper`): `Total success rate: 0.912`; ckpt dir exists at `.../phase1_exp_c_vlm_lora/29999`

## Artifacts
| Item | Path |
|------|------|
| Result summary | workdir `logs/phase1-exp-c-spatial-result.md` |
| Eval summary extract | workdir `logs/phase1-exp-c-eval-spatial-summary.txt` |
| Eval log tail | workdir `logs/phase1-exp-c-eval-spatial-tail.txt` |
| Remote eval log | `/root/autodl-tmp/logs/phase1-exp-c-eval-spatial.log` |
| Checkpoint | `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora/29999` |
| Control A | workdir `logs/phase1-spatial-result.md` (496/500) |
| Control B | workdir `logs/phase1-exp-b-spatial-result.md` (315/500) |

## Protocol pins (unchanged vs Phase1 / Exp B)
- Suite: libero_spatial; 10×50; seed 7
- Config name: `pi05_libero_lora_vlm_only` / exp `phase1_exp_c_vlm_lora`
- Init: `pi05_base`; train VLM LoRA only; freeze action-expert LoRA

## User authorization
Night gate review auto-pass; executor only forwards results; no retrain.
See `logs/phase1-exp-c-user-waiver.md`.
