# HANDOFF 增量 — 证据打包完成

当前已完成证据回拷和本地审计，停于 **acceptance 门禁**，等待小co / Architect；未自行登记通过或追加审查。

- 17:11 reviewer 已通过回拷 error；清单采集改用已确认可执行的 openpi .venv 绝对路径。
- 4239 个远端文件逐文件 SHA256 相符；完整 formal/preflight/server 日志、trials、summary、heldout、protocol、environment、75 个视频已回拷。
- 本地独立复算 2000/2000：A29999 487/500（97.4%）；G0 248/500（49.6%）；G1 285/500（57.0%，+7.4pp）；G2 223/500（44.6%，−5.0pp）。每组每任务 50 次；无缺失、重复、执行异常或重试；采数内容交集为 0。
- 正式 67 + 预跑 8 个视频全帧解码通过；原 G0/G1 有效 loss CSV 各 50 个唯一递增 steps，3 PNG 解码通过，原文件保留。
- 验收材料：../pnp-heldout-acceptance-evidence.md；原始和独立汇总均在当前目录。冻结协议/评测脚本未改；未重训、重评或删除保护资产。
- state 只读快照（17:21 交接更新）：reviews_used=5 / max_reviews=6，剩余 1 次 acceptance；由编排管理后续验收，不沿用旧的固定三次表述。

原任务目录不可写；请编排同步该状态及最终验收材料。

17:16 本地补充：29 份来源快照与冻结哈希一致（含 10 任务原始初态文件、采数 summary/源码及归一化）；`collection-used-initial-states.json` 保存原 task2 索引 0–30 的内容，本地重算 31 个排除哈希与冻结集完全一致。完整复核脚本/结果为 `local_package_audit.py` / `local-package-audit.json`；正式 5895 帧、预跑 590 帧，视频入口 `local-video-index.md`。未修改已冻结的评测脚本、协议、清单或正式记录；保留既有验收与审计材料。


## acceptance 需改补件完成（2026-11-06）

- 依据 17:21 reviewer 退回意见，新增 `../pnp-heldout-code-packet.md`：从冻结 implementation.diff 对应源码摘录初态加载/内容排除、协议和资产校验、续跑身份及重复拒绝、原子事件落盘、正式汇总；附原文件行号及冻结 SHA256。
- `../pnp-heldout-acceptance-evidence.md` 已加入 local-package-audit.json、local-independent-audit.json 的关键字段表，涵盖 2000 条/40 分组、零缺失/重复/执行异常、31 采数状态排除、G0/G2 资产一致、协议/清单及审计 SHA、视频与 loss 验证。
- 本次只读复核：4239 个回拷文件、29 个来源快照、7 个冻结脚本 SHA 一致；独立重算 2000 条与原两份 audit 完全一致；代码摘录与 Diff/源码原文一致。未改冻结资产或审计 JSON；未重训、未重评。
- PLAN 测试结果与额度同步为 reviews_used=5 / max_reviews=6；执行者未更改编排计数、phase 或自行追加轮次。停在 acceptance 门禁，等待小co / Architect 最后 1 次既定审查。
