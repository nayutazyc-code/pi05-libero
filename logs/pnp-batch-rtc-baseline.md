# 官方 batch / RTC 基线配置表（2026-11-06）

| 项 | 官方基线 G0 | 出处 |
|---|---|---|
| batch_size | **32** | `openpi/src/openpi/training/config.py` → `TrainConfig(name="pi05_libero_lora", batch_size=32)` |
| action_horizon | **10** | 同上 `Pi0Config(pi05=True, action_horizon=10, …)` |
| replan_steps（同步 chunk） | **5** | `examples/libero/main.py` `Args.replan_steps=5` |
| 异步 RTC inpainting | 本树 π0.5 LIBERO **未启用**；基线=同步 replan | PI RTC 博文/论文；经验笔记 rtc_policy 拒 π0.5 |

## 计划对照（采数/清洗后执行）
| 组 | 相对 G0 唯一改动 | 用途 |
|---|---|---|
| G0 | — | 官方基线训评 |
| G1 | batch=16 | 对照 A 实跑曾用 16 |
| G2 | replan_steps=10 | 更长 open-loop chunk |

采数当前用 G0 评测侧默认 replan=5。
