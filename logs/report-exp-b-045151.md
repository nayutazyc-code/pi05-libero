# 实验 B 评测汇报（action-expert-only LoRA）

2026-11-05 04:52 Asia/Shanghai · 门禁 acceptance 已通过（夜间授权自动通过）

## 1) 改了什么

- 实验 B 训练已完成（`TRAIN_B_EXIT 0`）：只训 action expert LoRA，冻结 VLM LoRA，其余沿用 Phase1 配置，最终 ckpt `phase1_exp_b_action_lora/29999`。本次任务**没有重训**。
- 在 `libero_spatial` 上按与 Phase1 完全相同的协议评测：10 任务 × 50 回合、seed 7、单轮无补跑。
- **结果：315/500 = 63.0%**，`EXP_B_EVAL_EXIT 0`（2026-11-05 04:28）。
- 对照 Phase1 全 LoRA 基线（`phase1_lora_baseline/29999`，同协议）：**496/500 = 99.2%**。
- 差值：**−181 个成功 / −36.2 个百分点**。

### 逐任务成功率（每任务 50 回合）

| 任务 | 成功 | 成功率 |
|---|---|---|
| 01 盘与烤盅之间的碗 → 盘子 | 21 | 42% |
| 02 烤盅旁的碗 → 盘子 | 36 | 72% |
| 03 桌子中央的碗 → 盘子 | 42 | 84% |
| 04 饼干盒上的碗 → 盘子 | 39 | 78% |
| 05 最上层抽屉里的碗 → 盘子 | 19 | 38% |
| 06 烤盅上的碗 → 盘子 | 27 | 54% |
| 07 饼干盒旁的碗 → 盘子 | 38 | 76% |
| 08 炉灶上的碗 → 盘子 | 33 | 66% |
| 09 盘子旁的碗 → 盘子 | 29 | 58% |
| 10 木柜上的碗 → 盘子 | 31 | 62% |
| **合计** | **315** | **63.0%** |

最弱：05 抽屉（38%）、01 盘与烤盅之间（42%）；最强：03 桌中央（84%）、04 饼干盒上（78%）。

## 2) 结果在哪 / 如何打开

工作区 `/example/project`：

- 本汇报：`logs/report-exp-b-045151.md`
- 实验 B 数字与逐任务：`logs/phase1-exp-b-spatial-result.md`
- 验收证据：`logs/phase1-exp-b-acceptance-evidence.md`；夜间授权：`logs/phase1-exp-b-user-waiver.md`
- 评测摘要/日志尾部：`logs/phase1-exp-b-eval-spatial-summary.txt`、`logs/phase1-exp-b-status.md`
- 基线：`logs/phase1-spatial-result.md`（496/500）
- 远端（SSH `ssh-wrapper`）：日志 `/root/autodl-tmp/logs/phase1-exp-b-eval-spatial.log`；ckpt `/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999`

打开方式：本地 `less logs/phase1-exp-b-spatial-result.md`；远端 `ssh-wrapper 'tail -n 100 /root/autodl-tmp/logs/phase1-exp-b-eval-spatial.log'`。

## 3) 风险

- **单 suite、单 seed**：只评了 spatial（seed 7），结论不能外推到 Object/Goal/Long 等 suite，也不能当作统计显著差异。
- **63% 显著低于 99.2%**：这与「冻结 VLM、只训 action expert」的消融预期方向一致，但当前只有一个点位，无法区分是 VLM 冻结导致表征不足、还是 action expert 容量/训练步数不足。不要把它读成「action-only 一定不行」。
- **视频证据有限**：沿用上游按任务+成败覆写，只能抽查，不能逐回合复盘。
- **失败模式未细查**：失败集中在抽屉、遮挡类任务，但尚未逐条看视频归因（抓取失败 / 定位偏差 / 超时）。
- 未发现训练或评测异常退出；磁盘与 ckpt 完整性本次未新增检查项。

## 4) 下一步（建议，未执行）

1. 抽查失败任务（05/01）远端视频，给失败归因（抓取 vs 定位 vs 超时）。
2. 若继续消融：在相同协议下补一个中间点（如解冻部分 VLM 层或更长训练），用于判断差距来源；这属于可选后续实验，不是本次验收的返工。
3. 需要更强结论时，扩到 Object/Goal 或补 seed 做均值±方差。
4. 保持 ckpt 29999 与基线 ckpt 不删，便于复现对照。
