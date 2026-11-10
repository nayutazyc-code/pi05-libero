# PnP held-out spatial：error 门禁交接

任务：`pi05-pnp-heldout-spatial-1106104308`。
核验时间：2026-11-06 10:58 Asia/Shanghai。

已确认 planning 第 2 次通过。按用户要求，出现实际执行错误后停止，等待小co / Architect 审查；本轮没有自行登记门禁通过或增加审查轮次。
任务目录不在可写范围，证据保存在项目 logs，供编排交接。
读取 state.json 时仍为 `reviews_used=2 / max_reviews=3, phase=executing`；额度及后续门禁由编排处理。

## 实际错误

远端环境版本采集命令中使用：

```python
import importlib.metadata as m
m.packages_distributions()
```

策略服务虚拟环境 Python 3.11 可执行；LIBERO 虚拟环境 Python 3.8.20 抛出：

```text
AttributeError: module 'importlib.metadata' has no attribute 'packages_distributions'
```

完整证据：`pnp-heldout-spatial-1106104308/environment.txt`。
这是执行者版本采集命令的兼容性错误。该采集命令与预跑启动位于同一条远端 shell 调用，采集错误未阻止后续启动；发现后已停止本轮 runner。
停止前仅完成协议冻结与校验；复核 runner 与其核验子进程均已退出，预跑和正式逐 trial 结果总数为 **0**，未启动策略服务。
没有自行修复版本采集命令或重启预跑。

## 已完成及验证

- 初态池原始数量：全 10 任务各 50；task2 排除全部 31 次采数尝试使用的索引 0–30 后剩 19。
- 预先独占写入 `heldout/selection_rules.json`，再进行无策略初态检查。task2 以原场景 native reset、seed7 单一候选流补 31 个，不修改场景/控制器。其余任务保留原有 50 个。
- 每个选中状态可加载、等待 10 步未预先成功、有限数值且满足预设稳定性阈值；全部候选接受/拒绝原因有独立审计。
- 冻结共 500 个状态，任务内唯一，与采数集合按状态内容指纹的交集为空。指纹去掉模拟器时间、float64 舍入至 12 位，避免仅时间不同导致伪去重。
- 采数排除集由现有初态文件和原采数脚本的 `(attempt-1)` 索引规则重建，包含失败尝试。held-out 仅针对本轮 PnP 采数，不宣称相对 A 既往训练数据隔离。
- 清单 SHA256：`e70042f9ecd0fb870d05841569e9f2d3ad73d7533f6f103c3359cb15040f3f08`。
- 协议 SHA256：`b16e0bf433f132d6a9ceef3a55ac03b54eabc44120d6bd34624e80c5447efe32`。已冻结并完整核验 A/G0/G1 精确 checkpoint params 与 normalization 文件哈希、评测脚本和关键源文件哈希；G0/G2 权重及归一化完全相同。
- 扩展原评测入口，新增显式清单、按组×任务×状态记录、原子事件落盘、协议一致性验证、同身份断点续跑、异常记录、代表视频写入及读帧验证。原入口未传清单时沿用原行为。
- 新增种子服务包装：每请求按 seed7、任务、状态和控制步确定 JAX key；四组及重试共用协议。未改变 state 表示或同步 replan 语义。
- 新增冻结、串行运行及独立汇总入口。预跑独立目录/事件 stage，正式汇总拒收预跑；正式运行要求同协议的预跑通过标记。
- `py_compile`、`bash -n` 通过。远端最终 3 项测试全部通过：清单越界/重复/泄漏拒绝；注入服务异常后原身份恢复且无重复计分/拒绝协议变化；2000 条合成正式记录独立重算及重复计分拒绝。测试数据位于 TemporaryDirectory，未混入正式记录。
- 下载证据后，本地重新计算全部状态指纹及交集、500 状态计数、脚本哈希和 G0/G2 资产一致性，均通过。
- 没有重训、修改或删除既有权重、数据、logs、wandb。磁盘约 85G 可用。

## 审查后待执行

建议只修复版本采集的 Python 3.8 兼容性：逐包调用 `importlib.metadata.version()`，捕获 `PackageNotFoundError`，避免使用新增 API。环境采集失败应阻断后续启动。此修复尚未执行。

当前还没有实机策略预跑通过标记；没有正式成功率，不能进入最终验收。
获本节点审查允许后，先补齐环境证据，再执行：

```bash
source /root/autodl-tmp/env.sh
bash /root/autodl-tmp/scripts/103-pnp-heldout-spatial.sh preflight
# 16 trials：四组 × task2/task4 × 2；task2 涵盖现有与补足初态。
# 全部完成并落 preflight-passed.json 后：
bash /root/autodl-tmp/scripts/103-pnp-heldout-spatial.sh formal
```

两阶段均失败即退出，不自动重试。formal 为 4×10×50，汇总从 trial 事件重新计算。
远端运行证据根：`/root/autodl-tmp/logs/pnp-heldout-spatial-1106104308/`。
本地证据副本：`logs/pnp-heldout-spatial-1106104308/`。
官方超参来源沿用 `logs/pnp-config-evidence.txt`；G0 必须称内部对照。已有非官方 state 协议、样本暴露量与归一化差异均已写入协议，后续报告仍须披露。
