## 实验 C（仅 VLM LoRA）评测汇报

验收已通过（按你的夜间授权自动通过，见 `logs/phase1-exp-c-user-waiver.md`），本任务**未重训**。

### 1) 改了什么
- 用已训好的实验 C ckpt（`pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora/29999`）在 `libero_spatial` 上评测：10 任务 × 50 回合、seed 7、单轮无补跑，协议与 A/B 完全一致。
- 设置：只训 VLM LoRA、冻结 action-expert LoRA（与实验 B 对称消融）。
- 结果：**456/500 = 91.2%**，`EXP_C_EVAL_EXIT 0`（2026-11-05 23:30），评测摘要中 `Total success rate: 0.912`、`Total episodes: 500`。

| 实验 | 训练范围 | 成功率 | 与 C 的差 |
|---|---|---|---|
| A Phase1 | 全 LoRA | 496/500 = 99.2% | C 低 **8.0 pp**（−40 回合）|
| **C 本次** | **仅 VLM LoRA** | **456/500 = 91.2%** | — |
| B | 仅 action LoRA | 315/500 = 63.0% | C 高 **28.2 pp**（+141 回合）|

- 逐任务：最强 01（盘与烤盅之间）50/50；最弱 05（抽屉内）39/50 = 78%，06（烤盅上）、10（木柜上）各 41/50 = 82%。

### 2) 结果在哪 / 如何打开
工作区 `logs/` 下：
- 汇报正文：`report-exp-c-000424.md`
- 数字与逐任务表：`phase1-exp-c-spatial-result.md`；摘要/尾部：`phase1-exp-c-eval-spatial-summary.txt`、`phase1-exp-c-eval-spatial-tail.txt`
- 验收证据/授权：`phase1-exp-c-acceptance-evidence.md`、`phase1-exp-c-user-waiver.md`
- 对照：`phase1-spatial-result.md`（A）、`phase1-exp-b-spatial-result.md`（B）
- 远端日志：`ssh-wrapper 'tail -n 100 /root/autodl-tmp/logs/phase1-exp-c-eval-spatial.log'`

### 3) 风险
- 单 suite（spatial）+ 单 seed + 单轮，不能外推到 Object/Goal/Long，也不构成显著性结论。
- 91.2% 低于全 LoRA 基线 8 pp：说明只训 VLM 有代价，但远高于 action-only；差距成因（容量/步数/表征）单点无法区分，别读成「VLM-only 不可行」。
- 失败集中在抽屉、堆叠等精细操作，尚未逐条归因；视频按任务+成败覆写，只能抽查。
- 交付侧：验收提交已在本地（领先 origin/main 1 个提交），但推 GitHub 被仓库侧 `commit_refs_rules` 拒绝；不影响评测数字。

### 4) 下一步（建议，均未执行）
1. 抽查 05/06/10 失败视频，归因抓取/定位/超时。
2. 如需解释 8 pp 差距，可补中间点（部分解冻 action expert 或延长训练），属可选后续，非返工。
3. 需更强结论时扩到 Object/Goal 或补 seed 报均值±方差。
4. 保留 A/B/C 三个 ckpt 以便复现；按授权不重训。
