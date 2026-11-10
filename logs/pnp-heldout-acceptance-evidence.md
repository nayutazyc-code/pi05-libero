# PnP held-out spatial：acceptance 门禁材料

任务：`pi05-pnp-heldout-spatial-1106104308`。正式运行于 2026-11-06 16:25:18 Asia/Shanghai 完成；当前仅提交最终验收，未自行登记通过。

## 结果

四组各 500 次、每任务各 50 次，共 2000 条有效正式结果。1243 次成功，757 次达到 220 步上限并计任务失败；正式执行异常、未完成尝试、重试、重复计分及缺失均为 0。16 次预跑单独存放，未计入成绩。

| 组别 | 成功 / 总数 | 成功率 | 相对 G0 |
|---|---:|---:|---:|
| A29999 | 487/500 | 97.4% | +47.8pp |
| G0 | 248/500 | 49.6% | +0.0pp |
| G1 | 285/500 | 57.0% | +7.4pp |
| G2 | 223/500 | 44.6% | -5.0pp |

全任务比较已能区分效果。PnP 微调三组的总体成功率均低于 A29999；采数任务 task2 上仍接近饱和（A/G0/G2 为 50/50，G1 为 49/50）。这些观测不支持把 G1 优势简单归因为 batch；既有样本暴露量、数据及归一化差异须一并考虑。

| task | A29999 | G0 | G1（Δ vs G0） | G2（Δ vs G0） |
|---:|---:|---:|---:|---:|
| 0 | 50/50 | 49/50 | 50/50（+2.0pp） | 43/50（-12.0pp） |
| 1 | 49/50 | 0/50 | 2/50（+4.0pp） | 0/50（+0.0pp） |
| 2 | 50/50 | 50/50 | 49/50（-2.0pp） | 50/50（+0.0pp） |
| 3 | 47/50 | 40/50 | 41/50（+2.0pp） | 31/50（-18.0pp） |
| 4 | 48/50 | 1/50 | 0/50（-2.0pp） | 1/50（+0.0pp） |
| 5 | 47/50 | 8/50 | 24/50（+32.0pp） | 3/50（-10.0pp） |
| 6 | 49/50 | 45/50 | 41/50（-8.0pp） | 25/50（-40.0pp） |
| 7 | 50/50 | 42/50 | 49/50（+14.0pp） | 49/50（+14.0pp） |
| 8 | 47/50 | 13/50 | 29/50（+32.0pp） | 21/50（+16.0pp） |
| 9 | 50/50 | 0/50 | 0/50（+0.0pp） | 0/50（+0.0pp） |

每任务四组成功率及全部 Δ：[原始汇总报告](pnp-heldout-spatial-1106104308/summary/report.md)、[40 行 CSV](pnp-heldout-spatial-1106104308/summary/per_task.csv)。任务指令文本在冻结 manifest 的 `tasks[].language` 中。

## 配置和隔离证据

- G0 是“内部对照，沿用官方部分超参”：micro8×accum4、有效 batch32、horizon10、同步 replan5。G1 为原生 batch16；G2 与 G0 的权重和归一化完全一致，仅同步 replan10。A29999 使用本轮 PnP 微调前的 29999 权重。
- 官方参数出处沿用 [已安装源码快照](pnp-config-evidence.txt)：`TrainConfig.batch_size=32` 与 LIBERO `Args.replan_steps=5`；该树无 Git 元数据，出处边界已披露。精确四组路径、每个参数/归一化文件及评测脚本 SHA256 均在 [protocol.json](pnp-heldout-spatial-1106104308/protocol.json)。
- 固定 seed7、action_horizon10、max_steps220、wait10、256 相机/224 输入；每次请求按任务、状态、控制步派生同协议 RNG。四组共用 500 个任务内唯一状态和顺序。
- 原采数共 31 次（30 成功、1 失败），按原脚本 `(attempt-1)` 规则排除 task2 索引 0–30，并按状态内容排除全部 31 个指纹。collection-summary 及采数脚本哈希与冻结清单一致。
- task2 的 19 个现有状态不足，按预先写入的 seed7 单一 native-reset 候选流补 31 个；其他任务各保留 50 个现有状态。无策略检查加载、稳定、等待期间未成功，审计及规则在 [heldout/](pnp-heldout-spatial-1106104308/heldout/)。选择规则及清单在策略成绩前冻结。
- 清单 SHA256：`e70042f9ecd0fb870d05841569e9f2d3ad73d7533f6f103c3359cb15040f3f08`。
- 协议 SHA256：`b16e0bf433f132d6a9ceef3a55ac03b54eabc44120d6bd34624e80c5447efe32`。
- held-out 仅相对本轮 PnP 采数，不宣称相对 A 历史训练数据独立。保留已批准的非官方 xyz+quaternion+首个 gripper（3+4+1）状态协议；本结果为内部比较，不宣称官方 benchmark。G0/G1 样本暴露量不同；A 数据/归一化不同；每任务 50 次有抽样不确定性。

## 验证与材料

- 既有 3 项功能测试和 Python/Shell 语法检查通过。error 审查后仅新增 Py3.8 `metadata.version()` 采集及失败阻断入口；两解释器采集成功，注入采集失败时原码退出且 runner 未启动。冻结评测协议及脚本未改。
- 4239 个远端文件逐个比对 SHA256，通过：[transfer-audit.json](pnp-heldout-spatial-1106104308/transfer-audit.json)、remote-inventory.json。包含全部 formal/preflight/server 日志、4000 个正式 start/result 事件、冻结输入、环境证据及视频。
- 远端及本地独立脚本分别从原始事件重算 40 个分组和 4 个总体，全部与汇总一致；内容指纹复算交集为 0，G0/G2 资产一致：[local-independent-audit.json](pnp-heldout-spatial-1106104308/local-independent-audit.json)。最终全资产哈希复验通过：final-protocol-verify.log。
- 正式 67 个、预跑 8 个代表视频，本地 ffmpeg 全帧解码共 6485 帧，与远端 SHA256/帧数一致；覆盖所有实际出现的组×任务×成功/失败类别：[local-media-audit.json](pnp-heldout-spatial-1106104308/local-media-audit.json)。正式视频位于 [videos/](pnp-heldout-spatial-1106104308/videos/)。
- 原有 [loss CSV/PNG](plots/) 保留：有效 G0/G1 W&B CSV 各 50 点、step 0..2450 间隔 50、唯一且严格递增，provenance 首尾相符；3 张 PNG 完整解码通过。历史失败 G0 原生/micro16 的 header-only CSV 未伪造为有效曲线。证据包另存 training-loss/ 副本。
- 本轮实现代码及 Diff 在 [implementation.diff](pnp-heldout-spatial-1106104308/implementation.diff) 与 implementation/。未重训、未新增演示训练数据、未重开评测，未删除既有权重/数据/logs/wandb。正式前通过环境采集 error；后续回拷 PATH error 亦已有审查通过，均与正式任务失败分开记录。

## 审查补件：审计摘要

17:21 acceptance「需改」要求的关键实现已摘入 [代码审查包](pnp-heldout-code-packet.md)，覆盖显式初态加载、内容去重/排除、资产与协议校验、续跑身份/重复拒绝、异常落盘及正式汇总；每段注明源码行号并与 implementation.diff 原文核对。本次仅新增审查材料。

以下为现有 JSON 的关键字段摘要，便于直接审查；原始 JSON 未改。

| 来源 | 字段 / 核查项 | 结果 |
|---|---|---|
| local-independent-audit.json | valid_trials / start_events / per_task_cells | 2000 / 2000 / 40；每组 500、每任务 50 |
| 同上 | errors / duplicates / missing | 0 / 0 / 0 |
| 同上及 summary/audit.json | 执行异常 / 未完成 / 重试 | 0 / 0 / 0；全部 attempt=1 |
| 同上 | outcomes | success=1243；timeout=757（均 220 步，计失败） |
| 同上 | collection_overlap | 0；逐状态内容复算，任务内各 50 个唯一状态；四组完全共享清单 |
| 同上 | g0_g2_same_assets | true；G2 等于 G0 配置仅 replan 从 5 改 10（含路径、params、normalization 逐字段一致） |
| 同上 | summary_matches_independent_recount | true；40 行成功数/分母/Δ 和四组总体均一致 |
| local-package-audit.json | remote_files_verified / 日志数 | 4239；formal=40、preflight=8、server=4 |
| 同上 | preflight_trials_excluded | 16，独立目录存放，未进入正式 2000 条 |
| 同上 | collection_attempts_excluded | 31（30 成功+1 失败）；原 task2 索引 0–30 |
| 同上 | collection_state_hashes_recomputed_from_original_states | true；从保存的原始采数状态内容重算 31 个排除指纹，与冻结排除集完全相等 |
| 同上 | heldout_sources_by_task | task2：19 existing + 31 native_reset；其他任务各 50 existing |
| 同上 | frozen_eval_scripts_match | true；本次亦核对 implementation/ 与 scripts/remote/ 的冻结 7 文件一致 |
| 同上 | videos | 正式 67 个/5895 帧；预跑 8 个/590 帧；全部解码，实际结果类别覆盖完整 |
| 同上 | loss_csv / loss_png | 有效 G0/G1 W&B CSV 各 50 个唯一升序 step（0–2450）；3 PNG 完整解码 |

SHA256 复核入口（完整值）：

| 对象 | SHA256 |
|---|---|
| heldout/manifest.json | `e70042f9ecd0fb870d05841569e9f2d3ad73d7533f6f103c3359cb15040f3f08` |
| protocol.json | `b16e0bf433f132d6a9ceef3a55ac03b54eabc44120d6bd34624e80c5447efe32` |
| implementation.diff | `d7f776fedd9acc9c01a8de28b7368298014b3f95cadc9c3a00f0ebd34659470e` |
| local-package-audit.json | `2c75d163fedd8b338035f002d442d865af080fb90c5ef6795c9604be3a969874` |
| local-independent-audit.json | `3519b9a004415215e971b4f3b81b776635d63d3465408ee1ad7bd00abd072dcb` |

归一化 SHA256：

| 组 | norm_stats.json SHA256 |
|---|---|
| A29999 | `7e19abf21b2f8837e1e27c468f4021efe359e38a9f6158acb83b6f978dad6b5a` |
| G0 | `7c9ed137ed61140619cd0f839993711251d9fb117f064ee7a0c2f5e854f03f6a` |
| G1 | `7c9ed137ed61140619cd0f839993711251d9fb117f064ee7a0c2f5e854f03f6a` |
| G2 | `7c9ed137ed61140619cd0f839993711251d9fb117f064ee7a0c2f5e854f03f6a` |

protocol.json 保存全部权重分片与归一化条目的路径、大小和 SHA；G0/G2 同权重同归一化。远端最终权重/归一化/脚本全资产复验见 [final-protocol-verify.log](pnp-heldout-spatial-1106104308/final-protocol-verify.log) 的 `PROTOCOL_VERIFIED b16e0bf433f132d6a9ceef3a55ac03b54eabc44120d6bd34624e80c5447efe32`。本地不复制大权重；本次复核冻结条目、归一化快照及回拷文件，不声称重新读取远端权重。

本次补件验证：重新执行 `independent_audit.py`（仅读本地记录）所得 JSON 与 local-independent-audit.json、independent-audit.json 完全相等；再次核对 4239 个文件、29 个来源快照及 7 个冻结脚本的 SHA256，全部通过。视频/loss 直接引用已完成的完整审计结果，未重复运行解码或策略。代码摘录逐段与冻结源码和 Diff 原文一致。未重训、未新评测，冻结脚本、协议、清单、正式记录与原审计 JSON 均未修改。

## 交接状态

本地补充复核（2026-11-06 17:16 Asia/Shanghai）：另保存并逐项核对 **29 份冻结来源快照**，包括 10 任务原始初态文件、采数 summary/源码、归一化、评测与安装源码。见 [source-snapshots.json](pnp-heldout-spatial-1106104308/source-snapshots.json)。从原始 task2 初态文件只读导出的 [采数用过的 31 个状态内容](pnp-heldout-spatial-1106104308/collection-used-initial-states.json) 重新计算内容哈希，与冻结排除集完全一致；再与 500 个 held-out 内容指纹比对，交集为 0。

完整本地审计及可复核入口：[local-package-audit.json](pnp-heldout-spatial-1106104308/local-package-audit.json)、[local_package_audit.py](pnp-heldout-spatial-1106104308/local_package_audit.py)。该脚本同时核对全部远端文件、冻结源码、preflight 分离、原始采数初态内容、全部视频帧与 loss CSV/PNG；运行命令为 `/usr/bin/python3 logs/pnp-heldout-spatial-1106104308/local_package_audit.py`，需本地已有 NumPy、Pillow、FFmpeg，不运行策略或仿真。视频拆分为正式 5895 帧、预跑 590 帧，见 [可点击视频索引](pnp-heldout-spatial-1106104308/local-video-index.md)。完整回拷源文件逻辑大小为 9,631,370 字节；回拷与补充来源采集工具亦保存在证据目录。

停在 acceptance 门禁，等待小co / Architect 审查。按 17:21 交接，编排现有额度为 `reviews_used=5 / max_reviews=6`（剩余 1 次 acceptance）；未修改计数、phase 或自行追加审查。原任务目录只读，交接增量见 [HANDOFF-packaging.md](pnp-heldout-spatial-1106104308/HANDOFF-packaging.md)。
