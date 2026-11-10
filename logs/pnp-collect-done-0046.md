## Progress update (2026-11-06 00:46 Asia/Shanghai)
POLICY_COLLECT_DONE already on training machine:
- ok=30 fail=1 attempts=31 @ 00:34
- out=/root/autodl-tmp/datasets/pnp_policy_spatial_center/ (~1.3G)
- log=/root/autodl-tmp/logs/pnp-policy-collect.log
- EXIT 0

Please continue immediately:
1. Clean dataset (drop fail/noop/bad frames; before/after stats)
2. Small fine-tune from Exp A ckpt 29999 + eval with G0/G1/G2
   - G0 official: batch=32, replan=5, action_horizon=10
   - G1: batch=16 only
   - G2: replan=10 only
3. Keep A/B/C final 29999; disk ~14G free
4. Then acceptance → reporter
