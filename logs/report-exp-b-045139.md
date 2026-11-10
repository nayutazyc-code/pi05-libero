完成。汇报已写入工作区 `logs/report-exp-b-045151.md`，验收门禁按夜间授权通过，未重训。

## 1) 改了什么
- 实验 B：只训 action expert LoRA、冻结 VLM LoRA，训练已完成（`TRAIN_B_EXIT 0`，ckpt `phase1_exp_b_action_lora/29999`）；本次只做评测验收，**未重训**。
- 在与 Phase1 相同协议下评测 `libero_spatial`：10 任务 × 50 回合、seed 7、单轮无补跑。
- **实验 B：315/500 = 63.0%**，`EXP_B_EVAL_EXIT 0`（2026-11-05 04:28）。
- **对照 Phase1 全 LoRA 基线：496/500 = 99.2%** → **Δ −181 / −36.2 个百分点**。
- 最弱任务：05 抽屉 38%、01 盘与烤盅之间 42%；最强：03 桌中央 84%。

## 2) 结果在哪 / 如何打开
工作区 `/example/project/logs/`：
- 本汇报：`report-exp-b-045151.md`
- 数字与逐任务表：`phase1-exp-b-spatial-result.md`
- 验收证据：`phase1-exp-b-acceptance-evidence.md`、`phase1-exp-b-user-waiver.md`
- 基线：`phase1-spatial-result.md`（496/500）
- 远端：`ssh-wrapper 'tail -n 100 /root/autodl-tmp/logs/phase1-exp-b-eval-spatial.log'`；ckpt 在 `.../pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999`

## 3) 风险
- 只评了 spatial 单 suite、单 seed，不能外推到其他 suite，也不构成显著差异结论。
- 63% 远低于基线，方向符合「冻 VLM、只训 action expert」的消融预期，但目前只有一个点位，无法区分是 VLM 冻结导致表征不足，还是 action expert 容量/步数不足；不宜读成「action-only 路线失败」。
- 视频沿用上游按任务+成败覆写，只能抽查；失败尚未逐条归因。
- 训练/评测均正常退出，未见异常。

## 4) 下一步（建议，均未执行）
1. 抽查 05/01 失败视频，归因抓取/定位/超时。
2. 若要继续消融，补一个中间点（部分解冻 VLM 或延长训练）判断差距来源——属可选后续，不是返工。
3. 需要更强结论时扩 Object/Goal 或补 seed 做均值±方差。
4. 保留 B ckpt 与基线 ckpt 以便复现。
