完成。汇报已写入工作区 `logs/report-phase1-074830.md`（门禁任务目录的 `out-reporter-*.md` 因沙箱限制写不进去，需编排侧转存或复制）。

## 汇报内容

**1) 改了什么**
- 在训练机上完成第一轮官方兼容微调：以 `pi05_base` 官方权重初始化，套官方 LIBERO 配置（openpi 固定版），训练 PaliGemma 2B LoRA + action expert 300M LoRA。
- 数据用官方 `physical-intelligence/libero`（1693 episodes），审计通过：Parquet 坏文件 0、非有限值 0，归一化与官方 norm_stats 对齐。
- 训到 30,000 步（batch 16）。中途 27k 存盘时磁盘满崩溃，清旧 ckpt 后从 26k 续训至 30k，`TRAIN_EXIT 0`，最终 ckpt `29999`。
- 用与 Phase 0 完全相同的协议评测 libero_spatial：500 回合、seed 7、replan 5、wait 10、上限 224，单轮无补跑。
- **结果：496/500 = 99.2%**（Phase 0 基线 495/500），`EVAL_DONE`，退出码 0。

**2) 在哪看**
- 摘要 `logs/phase1-spatial-result.md`；汇总与退出码 `logs/phase1-eval-spatial-summary.txt`；日志尾部 `logs/phase1-eval-spatial-tail.txt`；数据审计 `logs/phase1-audit-and-basefetch.md`。
- 远端：`/root/autodl-tmp/logs/phase1-eval-spatial.log`，ckpt `.../phase1_lora_baseline/29999`，视频 `/root/autodl-tmp/logs/videos_phase1_spatial`。

**3) 风险**
- 99.2% vs 99.0% 只差 1 个成功（4 vs 5 次失败），在噪声范围内，不能据此宣称微调有效。
- 只评了 spatial；4 个失败的任务分布仅在远端完整日志，工作区尚无逐任务明细。
- 视频仍只 13 个（上游脚本按任务+成败覆盖写）；单次评测单 seed，结论不可外推。
- 数据盘曾写满，现约 74%（剩 ~40G），`cache` 约 80G 待清。

**4) 下一步（已获授权，不等用户）**
清缓存（保留 29999/25000）→ 开实验 B（只训 action expert LoRA、冻 VLM LoRA、其余对齐）→ 回填 4 个失败明细与逐任务成功率 → 视预算扩评其他 suite 做匹配对照。
