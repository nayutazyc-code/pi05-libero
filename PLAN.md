# PLAN — PnP held-out + libero_spatial 全 10 任务评测

## 目标
不重训，用 held-out 初态 + libero_spatial 全 10 任务×50 trials，比较 A29999 / G0 / G1 / G2（共 2000 次正式评测），看能否拉开差距（拉开非验收硬条件）。

## 范围
- **G0（内部对照，沿用官方部分超参）**：ckpt 2499，micro8×accum4 有效 batch32，replan5
- G1：batch16，replan5
- G2：G0 权重，replan10
- A29999：参考组，replan5
- held-out **仅相对本次 PnP 采数**（排除全部 31 次尝试初态 + 状态内容去重）；不宣称相对 A 训练数据隔离
- 不做：重训；删 A/B/C 29999、G0/G1 ckpt、数据、logs、wandb；异步 RTC

## 步骤
1. 核验权重/归一化/采数记录；冻结 seed7、horizon10、max_steps220；评测前固定四组权重、归一化、脚本哈希
2. 构建 held-out 清单（排采数 31 次；内容去重；池不足预先固定补足规则；成绩前冻结+清单哈希；四组共用）
3. 扩展评测脚本（显式初态、逐 trial、断点续跑）+ 小规模验证
4. 训练机串行 4×10×50；按组×任务×初态唯一落盘；超时/执行异常/重试分记；禁静默替换或混入旧结果
5. 独立核算每任务/总体/Δ vs G0 → acceptance → reporter

## 测试结果
- 已冻结 500 个 held-out 状态：task2 原有 19 + native-reset 补足 31，其余任务各原有 50；按内容检查无任务内重复或采数泄漏。
- 已实现显式清单评测、逐 trial 原子落盘/续跑、协议冻结、独立汇总；Python/Shell 语法检查及远端 3 项测试通过。
- error 门禁通过后修 Py3.8 环境采集；preflight 16/16 流程通过；formal 于 16:25 Asia/Shanghai 完成：2000/2000 有效。A29999 487/500，G0 248/500，G1 285/500（+7.4pp），G2 223/500（-5.0pp）。远端 summary 已写；本地完整证据回拷与独立审计通过，详见 `logs/pnp-heldout-acceptance-evidence.md`。

- 证据打包 error 已于 17:11 审查通过：确认既有绝对解释器可执行后修复清单命令，4239 个文件逐项 SHA256 回拷核对通过。本地独立复算 2000 条结果、完整状态指纹去重/排除、G0/G2 同资产、75 视频全部帧和原 loss CSV/PNG 检查通过；无缺失/重复/执行异常/重试。已备齐验收材料及交接增量，停于 acceptance 等待审查；未重训或重评。

- acceptance 补件已完成：`logs/pnp-heldout-code-packet.md` 摘录冻结 Diff 的初态加载、协议/资产校验、续跑身份/防重及正式汇总代码；验收证据新增两份 local audit 的字段摘要与 SHA。重新只读复算 2000 条、4239 文件/29 来源快照/7 冻结脚本 SHA 校验通过；停于 acceptance，等待最后 1 次既定审查，未重训/重评。

## 风险与门禁
- 证据补充：29 份来源快照（含 10 个原始初态文件）与冻结哈希一致；从原始 task2 采数初态内容重算 31 个排除指纹通过。完整本地核验见 `logs/pnp-heldout-spatial-1106104308/local-package-audit.json`，视频入口 `local-video-index.md`。
- 初态池可能不足；生成补足须披露；state 非官方轴角，内部对照
- 本任务额度已由编排调整为 max_reviews=6，当前 reviews_used=5（acceptance 第1次需改；剩余 1 次）；执行者不修改计数、不自行追加审查。acceptance 等待小co / Architect。
- planning 第1次：需改 → 已补强（见上）
