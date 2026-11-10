# PnP G0/G1/G2 results (Asia/Shanghai)

- generated: 2026-11-06T09:06:47
- task: libero_spatial task_id=2 (bowl from table center → plate)
- FT data: local/pnp_policy_spatial_center (cleaned A-policy demos)
- start weights: Exp A pi05_libero_lora/phase1_lora_baseline/29999
- protocol: seed=7, task2, max_steps=220, wait=10, action_horizon=10; replan and actual trial counts below
- G0 native batch32 originally OOM before step0; micro16 x2 also OOM; replacement uses micro8 x4, averaged gradients, one clipping/AdamW update per effective batch. W&B offline enabled. Budget: 2500 optimizer updates per train group; a completed G0 would see 80000 samples, G1 saw 40000 (repeated sampling).
- state: xyz + quaternion(4) + first gripper joint(1), matching collection/training; differs from official xyz + axis-angle(3) + both gripper joints(2). Error gate approved retaining this contract.
- Evaluation uses init_states[0:50]; collection used the first 31 states (30 successful demos). This includes training initial states and is not held-out generalization or the official 10-task benchmark.

## Official baseline config (G0 train/eval)
| item | value | source |
|---|---|---|
| batch_size | 32 | openpi TrainConfig pi05_libero_lora |
| action_horizon | 10 | Pi0Config(pi05=True, action_horizon=10) |
| replan_steps | 5 | examples/libero/main.py Args.replan_steps |
| async RTC | not enabled in this tree | — |

## Results
| group | replan | successes | trials | success_rate | diff vs G0 | note |
|---|---:|---:|---:|---:|---|---|
| A29999 | 5 | 50 | 50 | 100.0% | +0.0pp | Exp A untouched reference |
| G0 | 5 | 50 | 50 | 100.0% | — | FT grad_accum effective batch32 = micro8 x4; action_horizon=10 |
| G1 | 5 | 50 | 50 | 100.0% | +0.0pp | FT native batch16; same 2500 optimizer updates; half the G0 sample exposure |
| G2 | 10 | 50 | 50 | 100.0% | +0.0pp | G0 weights, eval replan=10 only |

## Artifacts
- train logs: `/root/autodl-tmp/logs/pnp-g0-train.log` (native OOM), `pnp-g0-accum-train.log` (micro16 OOM), `pnp-g0-accum-mb8-train.log`, `pnp-g1-train.log`
- loss CSV/PNG: `/root/autodl-tmp/logs/plots/pnp_g{0,1}_wandb_loss.csv` / `pnp_g{0,1}_wandb_loss_curve.png`
- eval summaries: `/root/autodl-tmp/logs/videos_pnp_eval/*_summary.json`
- G0 vs G1 compares effective batch at equal optimizer updates, not equal sample exposure; microbatch RNG streams also differ. G0 uses accumulation rather than native batch32. A vs FT also changes data and normalization. G0 vs G2 changes only synchronous replan, not asynchronous RTC inpainting.
