汇报已写入 `logs/report-exp-c-000424.md`（已本地提交 `59fd9bb`），验收按夜间授权通过，未重训。

## 1) 改了什么
- 对已训好的实验 C 权重（`pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora/29999`）在 `libero_spatial` 上评测：10 任务 × 50 回合、seed 7，协议与 A/B 完全一致，单轮无补跑。
- 设置：只训 VLM LoRA、冻结 action-expert LoRA（与实验 B 对称）。
- 结果：**456/500 = 91.2%**，正常退出（EXIT 0，11-05 23:30）。

| 实验 | 训练范围 | 成功率 | 与 C 的差 |
|---|---|---|---|
| A Phase1 | 全 LoRA | 496/500 = 99.2% | C 低 **8.0 pp**（−40 题）|
| **C 本次** | **仅 VLM LoRA** | **456/500 = 91.2%** | — |
| B | 仅 action LoRA | 315/500 = 63.0% | C 高 **28.2 pp**（+141 题）|

- 最强任务 01（碗在盘与烤盅之间）50/50 = 100%；最弱 05（抽屉内）39/50 = 78%，06（碗在烤盅上）、10（木柜上）各 41/50 = 82%。

## 2) 结果在哪 / 如何打开
工作区 `/example/project/logs/`：
- 本汇报：`report-exp-c-000424.md`
- 数字+逐任务表：`phase1-exp-c-spatial-result.md`
- 评测输出摘要/尾部：`phase1-exp-c-eval-spatial-summary.txt`、`phase1-exp-c-eval-spatial-tail.txt`
- 验收证据与授权：`phase1-exp-c-acceptance-evidence.md`、`phase1-exp-c-user-waiver.md`
- 对照：`phase1-spatial-result.md`（A）、`phase1-exp-b-spatial-result.md`（B）
- 远端日志：`ssh-wrapper 'tail -n 100 /root/autodl-tmp/logs/phase1-exp-c-eval-spatial.log'`

## 3) 风险
- 单 suite、单 seed、单轮 500 回合：不能外推到 Object/Goal/Long，也不构成显著性结论。
- 比全 LoRA 基线低 8 pp，说明「只训 VLM」有代价，但仍远高于 action-only；差距来源无法从单点区分，别读成「VLM-only 不可行」。
- 失败集中于抽屉、堆叠放置等精细操作，尚未逐条归因；视频按任务+成败覆写，只能抽查。
- 日志尾部有 EGL 上下文清理报错，属退出阶段噪音，退出码 0，不影响统计。
- 交付侧：本地提交已就绪，但推送到 GitHub 被仓库侧规则拒绝（本地领先 2 个提交），不影响评测数字，稍后重推即可。

## 4) 下一步（建议，均未执行）
1. 抽查 05/06/10 的失败视频，归因抓取、定位或超时。
2. 若要解释 8 pp 差距，补一个中间点（部分解冻 action expert 或延长训练）——可选后续，非返工。
3. 需要更强结论时扩到 Object/Goal 或补 seed 报均值±方差。
4. 保留 A/B/C 三个 ckpt 以便复现；按授权不重训。

备注：任务收尾的 `handoff → done` 我这边无法写入（任务目录只读），需由编排侧记录；报告与证据均已落地工作区。
