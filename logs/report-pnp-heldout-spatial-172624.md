已完成。汇报文件：`logs/report-pnp-heldout-172531.md`（已本地提交并推送 `e45be75`）。

## 1) 改了什么
不重训、不删任何资产，只做一次受控评测：4 组 × `libero_spatial` 10 任务 × 50 回合 = **2000 次正式评测**，2026-11-06 16:25 全部完成，2000/2000 有效（执行异常、重试、缺失、重复均为 0，16 次预跑单列不计分）。四组共用同一份 500 个 held-out 初态（每任务 50 个、任务内按状态内容唯一，排除采数全部 31 次尝试用过的初态），清单在看成绩前冻结。

| 组别 | 成功/总次数 | 成功率 | 相对 G0 |
|---|---:|---:|---:|
| A29999（微调前参考） | 487/500 | 97.4% | +47.8pp |
| G0（内部对照，有效 batch32） | 248/500 | 49.6% | — |
| G1（batch16） | 285/500 | 57.0% | +7.4pp |
| G2（同 G0 权重，replan10） | 223/500 | 44.6% | −5.0pp |

读法：采数任务 task2 四组都接近饱和（50/50）；G1 的增益集中在 task5、task8（各 +32pp），但 task6 掉 8pp；replan 从 5 升到 10 反而更差（task3 −18pp、task6 −40pp）；**task1/4/9 是共同短板**——三组微调合计仅 4 次成功，A29999 有 147/150，说明微调伤到了部分通用能力。

## 2) 结果在哪 / 如何打开
工作区 `logs/pnp-heldout-spatial-1106104308/` 下：
- 打开 `summary/report.md` 看总表与逐任务 Δ，`summary/per_task.csv` 是 40 行原始明细
- 证据说明：`logs/pnp-heldout-acceptance-evidence.md`；实现要点：`logs/pnp-heldout-code-packet.md`
- 逐 trial 原始事件 `trials/`、冻结清单 `heldout/manifest.json`、协议 `protocol.json`
- 代表视频：`local-video-index.md` → `videos/`（67 条正式，覆盖每个组×任务×成功/失败类别）
- 远端原始汇总（需 SSH）：`ssh-wrapper 'cat /root/autodl-tmp/logs/pnp-heldout-spatial-1106104308/summary/report.md'`

## 3) 风险
- G1 的 +7.4pp 不能归因为 batch：G0/G1 的样本暴露量与 RNG 不同，有效 batch32 是 micro8×4 累积出来的。
- held-out 只相对本轮 PnP 采数独立；task2 的 50 个初态里有 31 个是预先写死的规则生成的。
- state 编码沿用已批准的非官方协议，数字属内部比较，不能当官方 benchmark。
- 每任务 50 次有抽样不确定性，单 suite、单 seed、单轮，不能外推到其他 suite。

## 4) 下一步（建议，均未执行）
1. 查 task1/4/9 失败视频，归因抓取/定位/超时——这是微调副作用最值得解释的一块。
2. 复核 G1 在 task5/task8 的增益是否真实行为变化。
3. 若要坐实 batch 效应需控制变量重训；不做不影响本轮结论。
4. 决策建议：默认用 G1（batch16），replan 保持 5。
5. 归档并沿用同一份冻结清单做后续对比，避免换初态导致不可比。
