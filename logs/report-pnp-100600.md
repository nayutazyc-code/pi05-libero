# pi05 A-PnP 流水线汇报

2026-11-06 · 任务 `pi05-a-pnp-1106001525` · acceptance 5/5 通过，架构巡检「继续」

## 1) 改了什么
以实验 A 权重（`pi05_libero_lora/phase1_lora_baseline/29999`）为起点，在 LIBERO `libero_spatial` 单任务 task_id=2（桌中央黑碗→盘）跑通「公开数据参考→仿真对齐→采数→清洗→微调→评测」：

- **采数**：几何脚本因接触漂移失败，改用 A 策略专家，31 次尝试 30 成功，仅成功轨迹入库（30 eps / 3327 帧）。
- **清洗+转换**：30→30 eps、3327→3327 帧，未删任何轨迹（专家轨迹无 near-noop）；LeRobot 仓库 `local/pnp_policy_spatial_center`。
- **G0 官方基线**：名义 batch32 / replan5 / action_horizon10。原生 32 与 micro16×2 均首步 OOM；按你授权改为 **micro8×累积4 等效 batch32**，2500 次更新，08:53 退出 0，ckpt 2499。
- **G1**：仅改原生 batch16，2500 次更新，03:13 退出 0。
- **G2**：复用 G0 权重，仅改 replan=10。
- **清盘**：先清 HF 可重建缓存，保留 A/B/C 最终 29999、PnP 数据、logs、W&B；收尾约 85G 可用、GPU 空闲。
- 新增梯度累积 adapter、训练/评测/导出脚本并实测微批 8/16/32 与整批梯度一致、裁剪/AdamW/EMA 各一次。

## 2) 结果
协议：task2、seed7、`init_states[0:50]`、max_steps220、wait10、action_horizon10。

| 组 | 训练 batch | replan | 成功 | 平均环境步数 |
|---|---|---:|---:|---:|
| A-29999（参考） | — | 5 | 50/50 (100%) | 101.86 |
| G0 | micro8×4=32 | 5 | 50/50 (100%) | 105.20 |
| G1 | 原生 16 | 5 | 50/50 (100%) | 99.60 |
| G2 | 复用 G0 | 10 | 50/50 (100%) | 102.20 |

四组成功率相同，当前协议下**未观察到 batch 或重规划差异**；步数仅描述，不是推理时延。loss：G0 0.08826→0.01012，G1 0.11210→0.01196（每 50 更新记录，末点非最后一步）。

## 3) 结果在哪
本汇报：`logs/report-pnp-100600.md`。工作区 `logs/` 下：
- 结果表 `pnp-g0-g1-g2-result.md`；验收证据 `pnp-acceptance-evidence.md`
- 清洗统计 `pnp-clean-stats.md`；审计 `pnp-data-audit.json`、`pnp-lerobot-audit.json`；独立校验 `pnp-final-verification.json`
- 授权/门禁 `pnp-user-auth.md`；梯度累积实测 `pnp-grad-accum-adapter-test.md/.txt`
- loss 曲线 `plots/pnp_g{0,1}_wandb_loss_curve.png`；视频 `videos_pnp_eval/`（每组 3 段成功视频+summary）
- 远端：G0/G2 `…/checkpoints/pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499`，G1 `…/pnp_g1_bs16/2499`；日志/视频/W&B offline 在 `/root/autodl-tmp/logs/`、`/root/autodl-tmp/openpi/wandb/`（G0 `laea5vht`，G1 `v6rh1riu`）

## 4) 风险
1. **非官方 benchmark 复现**：state 为 xyz+四元数+单指，异于官方 xyz+轴角+双指（error 门禁已批准披露）。
2. **非独立泛化**：评测初态含采数用过的前 31 个，且仅单任务 50 回合。
3. **batch 效应未隔离**：G0 是梯度累积等效 32，非原生；样本暴露 80000 vs 40000、随机数路径不同。
4. **无 RTC 收益证据**：G2 仅同步 replan，非异步 RTC inpainting。
5. **天花板效应**：四组全 100%，比较不敏感，不能反推「微调/batch 无效」。

## 5) 下一步（均未执行）
1. 换采数未用的 `init_states`（如 50–99）并扩到 10 任务，检验泛化。
2. 补等样本暴露对照，并在更大显存上验证原生 batch32。
3. 若要谈 RTC，实现异步路径并用推理时延/吞吐评估。
4. 可选清中间 ckpt；建议保留 A/B/C 29999、G0/G1 2499、数据与日志。
5. 多 seed 报均值±方差，并对失败视频归因。
