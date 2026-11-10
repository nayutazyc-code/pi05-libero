# PnP executor：报错证据与待审交接

任务：pi05-a-pnp-1106001525；核验时间：2026-11-06 01:37–01:38 Asia/Shanghai。

## 门禁状态

已读任务目录 HANDOFF.md、state.json、planning 门禁及项目 REQUIREMENTS.md。
planning 已通过，state.json 仍为 reviews_used=1/max_reviews=3、phase=executing。
当前按用户要求停止执行，提交报错重现节点给小co / Architect；未自行登记通过或追加审查轮次。
任务目录不在本轮可写范围，交接证据保存在本工作区。

## 远端实测

SSH 已恢复，使用 `ssh-wrapper`，未使用域名或提权。

- 数据盘：150G，总使用约 48G，可用 103G，32%。
- 远端清洗统计：30 episodes / 3327 frames → 30 / 3327；LeRobot 路径 `/root/autodl-tmp/cache/hf/lerobot/local/pnp_policy_spatial_center`。
- A/B/C 最终 29999 及各自 params 目录均存在；本轮未删除任何远端文件。
- G1 已有进程：PID 743911，父进程 741248；batch16、2500 steps、log_interval50、save_interval500、keep_period2500。未重复开训。
- G1 最新已记录点：step250，loss=0.0296（01:37:56）；step0 loss=0.1121。GPU 核验时约 22079MiB / 24564MiB、100% 使用。
- G0/G1 日志均有 `Tracking run with wandb version 0.19.11`；启动日志明确 `wandb=offline`。

## G0 错误证据（读取已有失败日志，未再次开训）

远端 `/root/autodl-tmp/logs/pnp-g0-train.log`：

```text
2026-11-06 01:25:34: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 13238702560 bytes.
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 13238702560 bytes.
TRAIN_G0_EXIT 1 2026-11-06T01:25:37+08:00
```

G0 目录 `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_pnp/pnp_g0_bs32` 仅有 `wandb_id.txt`，无数字 checkpoint。
G2 按约定必须使用 G0 权重，不能用 G1 权重冒充；当前无 G0/G2 成功率或 batch 对照结论。
planning/HANDOFF 已允许 batch32 OOM 后记录失败并运行 batch16，现有 G1 正按此回退运行。

## 代码核验发现（待本次同一节点审查）

1. **state 数据契约不一致**。远端官方 `examples/libero/main.py:135–137` 拼接 `robot0_eef_pos`、`_quat2axisangle(robot0_eef_quat)`、完整 `robot0_gripper_qpos`，即 3+3+2。
   采数 `collect_pnp_policy_expert.py:20–25` 和评测 `eval_libero_task.py:35–40` 拼接 3+4+1。形状同为 8，但旋转与夹爪分量语义不一致。
   当前清洗仅过滤，没有修复该表示，不能据此声称数据契约已与 A 对齐。应审查是否转换既有 state（四元数→轴角；补齐夹爪需验证关节约束或重采），并处理现有 G1 的有效性；不能只改评测而沿用不匹配训练数据。
2. **A 参考评测会被跳过**。`99-pnp-eval-g0-g1-g2.sh` 的 `A_CKPT` 已指向 29999 叶目录，`run_one` 却对其调用 `pick_final`，再次寻找数字子目录。应允许完整 checkpoint 路径或传入实验根目录，并验证最终解析到 A/29999。
3. 清洗脚本尚未完整验证 wrist shape、action/state shape、动作范围；现有 30→30 统计不能替代这些数据契约检查。

## 审查后续执行入口

先裁决 state 契约及现有 G1 处理方案，修复 A checkpoint 解析，再继续有效训练/评测。
保留 G0 原始 OOM 记录与 A/B/C 最终 29999、原始 PnP 数据、logs、wandb。
最终 loss CSV 必须按 step 去重升序；G0 无有效 loss 点应明确为空，不能绘制伪曲线。
G0/G2 若仍无权重应明确报告不可评；如补做其他重规划对照须单独命名并只改 replan。
完成后再进入最终验收节点并交 reporter。

本轮仅核验远端与更新本地交接文件，未启动清洗、训练、评测或修改现有远端进程。
