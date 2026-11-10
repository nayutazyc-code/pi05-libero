# PnP held-out spatial results

Complete: 2000/2000 valid trials

G0 is an internal control using some official hyperparameters: batch32 (micro8 × accum4), horizon10, synchronous replan5. Official installed-source evidence: logs/pnp-config-evidence.txt; config.py TrainConfig.batch_size and examples/libero/main.py Args.replan_steps.

| Group | Successes / trials | Rate | Δ vs G0 | Difference from G0 |
|---|---:|---:|---:|---|
| A29999 | 487/500 | 97.4% | +47.8pp | A/29999, no PnP fine-tuning; different data/normalization |
| G0 | 248/500 | 49.6% | +0.0pp | internal control, micro8 ×4 |
| G1 | 285/500 | 57.0% | +7.4pp | native batch16; different training exposure/RNG |
| G2 | 223/500 | 44.6% | -5.0pp | same G0 weights/normalization; synchronous replan10 only |

| Task | Group | Successes / trials | Rate | Δ vs G0 |
|---:|---|---:|---:|---:|
| 0 | A29999 | 50/50 | 100.0% | +2.0pp |
| 0 | G0 | 49/50 | 98.0% | +0.0pp |
| 0 | G1 | 50/50 | 100.0% | +2.0pp |
| 0 | G2 | 43/50 | 86.0% | -12.0pp |
| 1 | A29999 | 49/50 | 98.0% | +98.0pp |
| 1 | G0 | 0/50 | 0.0% | +0.0pp |
| 1 | G1 | 2/50 | 4.0% | +4.0pp |
| 1 | G2 | 0/50 | 0.0% | +0.0pp |
| 2 | A29999 | 50/50 | 100.0% | +0.0pp |
| 2 | G0 | 50/50 | 100.0% | +0.0pp |
| 2 | G1 | 49/50 | 98.0% | -2.0pp |
| 2 | G2 | 50/50 | 100.0% | +0.0pp |
| 3 | A29999 | 47/50 | 94.0% | +14.0pp |
| 3 | G0 | 40/50 | 80.0% | +0.0pp |
| 3 | G1 | 41/50 | 82.0% | +2.0pp |
| 3 | G2 | 31/50 | 62.0% | -18.0pp |
| 4 | A29999 | 48/50 | 96.0% | +94.0pp |
| 4 | G0 | 1/50 | 2.0% | +0.0pp |
| 4 | G1 | 0/50 | 0.0% | -2.0pp |
| 4 | G2 | 1/50 | 2.0% | +0.0pp |
| 5 | A29999 | 47/50 | 94.0% | +78.0pp |
| 5 | G0 | 8/50 | 16.0% | +0.0pp |
| 5 | G1 | 24/50 | 48.0% | +32.0pp |
| 5 | G2 | 3/50 | 6.0% | -10.0pp |
| 6 | A29999 | 49/50 | 98.0% | +8.0pp |
| 6 | G0 | 45/50 | 90.0% | +0.0pp |
| 6 | G1 | 41/50 | 82.0% | -8.0pp |
| 6 | G2 | 25/50 | 50.0% | -40.0pp |
| 7 | A29999 | 50/50 | 100.0% | +16.0pp |
| 7 | G0 | 42/50 | 84.0% | +0.0pp |
| 7 | G1 | 49/50 | 98.0% | +14.0pp |
| 7 | G2 | 49/50 | 98.0% | +14.0pp |
| 8 | A29999 | 47/50 | 94.0% | +68.0pp |
| 8 | G0 | 13/50 | 26.0% | +0.0pp |
| 8 | G1 | 29/50 | 58.0% | +32.0pp |
| 8 | G2 | 21/50 | 42.0% | +16.0pp |
| 9 | A29999 | 50/50 | 100.0% | +100.0pp |
| 9 | G0 | 0/50 | 0.0% | +0.0pp |
| 9 | G1 | 0/50 | 0.0% | +0.0pp |
| 9 | G2 | 0/50 | 0.0% | +0.0pp |

Protocol SHA256: b16e0bf433f132d6a9ceef3a55ac03b54eabc44120d6bd34624e80c5447efe32
Execution errors: 0; interrupted attempts: 0; extra attempts: 0.

Held-out only against all 31 PnP collection attempts; no claim of isolation from A's earlier training data. Task2 includes 31 native-reset generated states, fixed before policy scores. Nonofficial xyz+quaternion+first-gripper state encoding is retained under the prior gate approval. Results are internal comparisons. G0/G1 exposure differs and cannot isolate batch causality; 50 trials/task have sampling uncertainty. G2 is synchronous replanning, not async RTC.

Raw events: ../trials/; frozen inputs: ../heldout/manifest.json and ../protocol.json; videos: ../videos/. No prior evaluation results are included.
