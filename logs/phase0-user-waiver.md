# User waiver — Phase 0 acceptance (2026-11-03 01:10 Asia/Shanghai, authoritative)

The user (project owner) explicitly approved:
1. Review budget change: max_reviews raised 3 → 5 (this acceptance = review 5/5, the last).
2. Video deviation accepted: pinned upstream main.py overwrites per task/outcome; 13 files (md5-indexed in phase0-protocol-pins.md) accepted in place of per-episode videos. No rerun.
3. Waived: separate 10-episode smoke test; cold-start/steady-state GPU memory records; per-episode step/timing (not emitted by pinned main.py; failures = 220-step timeouts, 0 exceptions).

Remaining evidence: 495/500 (99.0%) ≥ 494 threshold; full log md5 824c7a42734bd292440ee0dd87eca4e7; 10×50 complete; pins in phase0-protocol-pins.md.
User instruction: on acceptance PASS → reporter → Phase 1 official-compatible finetune.
