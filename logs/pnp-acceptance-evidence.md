# pi05-a-pnp：最终验收证据包

任务：`pi05-a-pnp-1106001525`。执行完成时间：2026-11-06 09:16 Asia/Shanghai。
当前为 **acceptance 补件待复审**；planning/error 已通过，上次 acceptance 结论为需改。
补件时只读核对任务 state.json：reviews_used=3/max_reviews=5；额度由编排按 HANDOFF 更新，执行者未修改。
按用户三节点约定停止执行，等待编排登记 acceptance 并交小co / Architect 审查；未自行登记通过、增加审查轮次或运行 reporter。
PnP 实现与既有证据已由编排提交推送于 `efb9421`；本次仅补文件，不 commit/push，任务目录不在本轮可写范围。
已按要求尝试 `git add`，但 `.git` 为环境只读挂载，无法创建 `index.lock`，未能暂存；下文保留编排接手命令。

## 执行结果

- 公开参考：`lerobot/libero_spatial_image`；复用 LIBERO OffScreenRenderEnv、spatial task_id=2，Panda 把桌中央黑碗放入盘中。
- 场景原始双相机为256×256，模型输入resize224，动作7D；场景/BDDL/相机见 `pnp-dataset-and-scene.md`。
- 几何脚本专家接触漂移后停用；实际采用 A-29999 策略专家：31次尝试，30成功/1失败，仅成功轨迹入库。
- 清洗与LeRobot转换：30 episodes / 3327 frames → 30 / 3327；数据审计与转换审计通过，未冒称清洗删掉失败轨迹。
- 官方配置意图：LoRA batch32、action_horizon10、同步replan5；出处为远端openpi `src/openpi/training/config.py` / `examples/libero/main.py`，存档 `pnp-config-evidence.txt`。
- 原生batch32与micro16×2均首步OOM。按用户后续授权，G0改为 **micro8×4，有效batch32**，梯度平均后裁剪一次、AdamW更新一次；保留两次失败日志和W&B run。
- G0/G1均从 A/29999 起训，清洗数据/归一化/模型/学习率计划相同，2500次优化更新；G0于08:53:37退出0，G1于03:13:05退出0。
- W&B全程offline开启；G0 run=`laea5vht`，G1 run=`v6rh1riu`。

## 最终评测

共同协议：libero_spatial task2，seed7，init_states[0:50]，max_steps220，wait10，action_horizon10。

| 组 | 训练batch | replan | 成功 | 成功率 | 平均环境步数 |
|---|---|---:|---:|---:|---:|
| A-29999 | 原A权重参考 | 5 | 50/50 | 100% | 101.86 |
| G0 | micro8×4=有效32 | 5 | 50/50 | 100% | 105.20 |
| G1 | 原生16 | 5 | 50/50 | 100% | 99.60 |
| G2 | 复用G0权重 | 10 | 50/50 | 100% | 102.20 |

G0/G2评测退出码均0，后续脚本09:06:47完成。当前协议下未观察成功率差异；平均环境步数仅作描述，不代表推理时延或显著改善。

## 验证与改动范围

- 本轮小改：`101-pnp-finetune-g0-accum.sh`切换micro8、独立实验/日志；`99-pnp-eval-g0-g1-g2.sh`改G0路径和组别说明；`100-pnp-export-loss.py`分别导出native32/micro16失败run及micro8成功run；新增`102-pnp-after-g0-mb8.sh`顺序衔接导出与G0/G2评测，不重训/重评A/G1。
- `bash -n`与Python编译检查通过。远端CPU实际adapter测试通过：micro8/16/32梯度与整批一致，裁剪/AdamW/EMA各一次，非法micro size拒绝。
- 本地独立核验全部200条trial、episode编号、初始化状态索引、成功计数/比率、步数范围；G0/G2的checkpoint相同，其余记录协议字段一致。
- G0/G1 W&B各50点，step0–2450，唯一升序且有限；与文本日志逐点核对误差≤0.000051；native32/micro16各0点，不绘伪曲线。
- G0 loss记录首/末为0.0882594213→0.0101199802；G1为0.1120962277→0.0119647691。每50步记录一次，最后49次更新没有单独loss记录，不能称末点为最后一步loss。
- 每组3段成功视频已取回；各组ep00经ffprobe核验256×256且有帧，G0末帧及loss图已目视检查。

## 重要解释边界

1. G0为梯度累积等效batch32，不能称本卡原生batch32训练成功；micro8运行峰值仅观测约22.1GB，不是精确峰值测量。
2. G0/G1更新次数相同，样本暴露分别80000/40000（重复采样），微批RNG路径不同；不能归因为严格等样本预算的batch效应。A与微调组还改变了训练数据/归一化。
3. 数据/评测state为xyz+四元数+单指，异于官方xyz+轴角+双指；error门禁明确批准保留并披露，因此属于本任务内部协议，非官方benchmark复现。
4. 评测包含采数前31次初始状态，非独立泛化成绩；只覆盖单任务50次，不代表10任务套件表现。
5. G2仅改同步chunk重规划步数，未启用异步RTC inpainting。

## 证据位置与保留核验

本地（相对工作区）：

- 结果表：`logs/pnp-g0-g1-g2-result.md`。
- 验收补测原文：`logs/pnp-grad-accum-adapter-test.md`；完整 stdout/stderr：`logs/pnp-grad-accum-adapter-test.txt`。训练机 CPU、微批8，退出0；三份测试/adapter源码 SHA256 与本地一致。
- 本任务授权与 error 门禁原文：`logs/pnp-user-auth.md`；修正后的本任务计划：`PLAN.md`。
- 独立校验：`logs/pnp-final-verification.json`；原始/转换审计：`logs/pnp-data-audit.json`、`logs/pnp-lerobot-audit.json`。
- W&B来源与CSV核对：`logs/plots/pnp-loss-provenance.json`；成功组曲线：`logs/plots/pnp_g{0,1}_wandb_loss_curve.png`，同名CSV。
- 全部摘要/视频：`logs/videos_pnp_eval/`，每组独立tag；原生32与micro16错误日志、micro8训练日志、评测日志均已取回。

远端训练产物：

- G0/G2：`/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499`。
- G1：`/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_pnp/pnp_g1_bs16/2499`。
- 日志/视频/曲线：`/root/autodl-tmp/logs/`；W&B原记录：`/root/autodl-tmp/openpi/wandb/`。

09:14远端核验：A/B/C最终29999的params目录、G1/2499及原PnP数据均存在；GPU空闲，无残留训练/评测服务，数据盘85G可用。
本轮未删除缓存、数据或checkpoint，未覆盖原生32/micro16失败实验。

## 需改项闭环与下一次验收打包

1. PLAN 已按 A-PnP 修正；授权补件包含直接读取的 error 门禁原文与 HANDOFF 微批8授权原文。
2. 已于09:59在训练机运行实际 adapter 测试，完整输出附于上述补件；未重训、未重评、未删除 A/B/C 29999。
3. grad_accum / eval / export 脚本已在 `efb9421`，不存在待补的脚本修改；`git show efb9421 -- scripts/remote/` 可审查完整改动。PLAN/本证据包修改可用 `git diff` 查看；adapter/auth 补件目前为未跟踪文件，需编排暂存后用 `git diff --cached` 查看。

暂存阻断原文（退出128；环境权限限制，未重试绕过）：

```text
fatal: Unable to create '/example/project/.git/index.lock': Read-only file system
```

编排下一步登记既定 acceptance 时，请显式指定关键文件，使脚本 diff 分段进入瘦身包（默认关键文件列表不含 PnP 脚本）。执行者未运行此命令：

```bash
cd /example/project
git add PLAN.md logs/pnp-user-auth.md logs/pnp-grad-accum-adapter-test.md logs/pnp-grad-accum-adapter-test.txt logs/pnp-acceptance-evidence.md
GATE_KEY_FILES='PLAN.md scripts/remote/pnp_grad_accum.py scripts/remote/train_pnp_grad_accum.py scripts/remote/test_pnp_grad_accum.py scripts/remote/101-pnp-finetune-g0-accum.sh scripts/remote/99-pnp-eval-g0-g1-g2.sh scripts/remote/eval_libero_task.py scripts/remote/100-pnp-export-loss.py scripts/remote/102-pnp-after-g0-mb8.sh' /example/home/workflows/ds-gpt-gate/scripts/gate.sh pi05-a-pnp-1106001525 acceptance
```

已核对 `slim-context.sh`：PLAN 从工作区读取，adapter/auth/本证据包会进入第5节；上述脚本均已提交，关键文件 diff 可从 `HEAD~1..HEAD` 读取。补件保持待编排暂存/提交的状态。

审查通过后交reporter，面向用户说明四组均50/50、micro8×4成功及上述协议限制；本文件不是验收通过声明。
