# PnP clean stats (2026-11-06 Asia/Shanghai)

- src: `/root/autodl-tmp/datasets/pnp_policy_spatial_center`
- clean_out: `/root/autodl-tmp/datasets/pnp_policy_spatial_center_clean`
- repo_id (LeRobot): `local/pnp_policy_spatial_center`

## Before → After
| | episodes | frames |
|---|---:|---:|
| before | 30 | 3327 |
| after | 30 | 3327 |
| delta | 0 | 0 |

## Filters
- drop fail / nonfinite / bad shape / after-clean len < 20
- drop near-noop frames: max(|action[:6]|) < 0.001
- compress idle runs > 8: eef_delta < 0.001 and action_l2_6d < 0.05 (keep ends)

Dropped episodes: 0


## LeRobot convert
- path: `/root/autodl-tmp/cache/hf/lerobot/local/pnp_policy_spatial_center`
- flipped_180: true
