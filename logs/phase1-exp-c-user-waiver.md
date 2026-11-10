# User waiver — Experiment C acceptance (2026-11-06 Asia/Shanghai)

The user (project owner) explicitly authorized for task `pi05-exp-c-eval-1105221439-1105221439`:

1. **Night gate auto-pass**: acceptance reviewer conclusion may be recorded as **通过** without waiting for interactive online approval.
2. **No retrain**: Experiment C training already completed at ckpt 29999; do not restart training.
3. **Executor role**: 执行小g only forwards results; orchestration completes acceptance → reporter.
4. **Retain** logs and checkpoints; do not leak wandb keys; SSH via `ssh-wrapper`.
5. Confirmed numbers (do not alter): C **456/500 = 91.2%**; A 496/500 (99.2%); B 315/500 (63.0%); C vs A Δ −8.0pp; C vs B Δ +28.2pp; `EXP_C_EVAL_EXIT 0` @ 2026-11-05T23:30:27+08:00.

Acceptance criterion for this task: Exp C spatial eval finished with recorded success rate and A/B contrasts in the user-facing report.
