# PnP 数据集选定与场景方案（2026-11-06 00:19 Asia/Shanghai）

## 选定公开数据集
- **主选**：Hugging Face [`lerobot/libero_spatial_image`](https://huggingface.co/datasets/lerobot/libero_spatial_image)
  - robot: Franka Panda；fps 10；256×256 `image` + `wrist_image`；state 8D；action 7D
  - 任务含 pick black bowl → place on plate（与 A 同 embodiment）
- **训练同源对照**：`physical-intelligence/libero`（`pi05_libero_lora` 的 `repo_id`；已在 `/root/autodl-tmp/cache/hf/lerobot`）

## 仿真场景对齐（不另起异构仿真）
| 维度 | 方案 |
|---|---|
| 仿真栈 | 现有 LIBERO `OffScreenRenderEnv`（openpi `examples/libero` + `third_party/libero`） |
| suite / task | `libero_spatial` / `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`（task_id=2） |
| 物体 | BDDL：`akita_black_bowl_*`、`plate_1`、ramekin/cookies 等干扰物 |
| 相机 | `agentview_image`、`robot0_eye_in_hand_image` @ 256×256（对齐公开集） |
| 动作空间 | 7D OSC（dx,dy,dz,dax,day,daz,gripper），与 A / LeRobot LIBERO 一致 |
| 成功判定 | `env.check_success()`（bowl on plate） |

## 官方 batch / RTC 基线（对照用）
| 项 | 值 | 出处 |
|---|---|---|
| batch_size | 32 | `config.py` `pi05_libero_lora` |
| action_horizon | 10 | 同上 `Pi0Config(pi05=True, action_horizon=10)` |
| replan_steps（同步 chunk，本树 LIBERO 默认） | 5 | `examples/libero/main.py` |
| 完整异步 RTC | 本树 π0.5 LIBERO 未启用；基线记为同步 replan | PI RTC 论文 / 经验笔记 |

对照：G1 batch=16；G2 replan_steps=10。

## 采数策略更新（2026-11-06 00:27 Asia/Shanghai）
- 纯几何 OSC 脚本专家：碗接触后 xy 漂移、无法闭环抓取（调试 0/5）。
- **改用实验 A 策略作专家**（REQUIREMENTS 允许「程序化/专家策略」）：`serve_policy` + ckpt 29999，`replan_steps=5`（官方 LIBERO 默认），采成功 episode → `/root/autodl-tmp/datasets/pnp_policy_spatial_center/`。
- 脚本：`scripts/remote/96-exp-a-pnp-policy-collect.sh`、`collect_pnp_policy_expert.py`。
