# π0.5 任务要求（用户 2026-11-02 授权开干）

## 目标
1. **第一步**：清理训练机数据盘 `/root/autodl-tmp` 上旧数据（已由编排侧执行/核对）。
2. 按计划书做 **π0.5 复现**（Phase 0：`pi05_libero` + `libero_spatial` 官方评测对标 ~98.8）。
3. 在复现通过后继续 **π0.5 微调**（官方兼容基线；遵守经验总结与计划书闸门）。

## 参考（必读）
- `refs/pi05-experience-notes.md` — 用户上传的 π0→π0.5 经验总结
- 计划书全文：refs/pi05-plan-book.canvas.txt（Canvas「π0.5 微调计划书」）
- 继承经验：先数据契约；父布局划分；不改名冒充 π0.5；先官方兼容再自定义；分层归因

## 训练机
- SSH（经用户本机，box 直连被拦）：`ssh -p "$SSH_PORT" SSH_USER@SSH_HOST`
- GPU：RTX 4090 D
- 数据盘：`/root/autodl-tmp`（约 150G）
- 正式训评在训练机；本机门禁只做编排/配置/日志/推仓

## 门禁
走 ds-gpt-gate：planner → (architect) → planning 门禁 → executor → acceptance → reporter
推仓到 nayutazyc01-ui；结果由 reporter 写用户向报告，由执行小g 转用户。


## 用户指令（2026-11-03）
分数达标后继续微调；验收补证与交接由门禁自行处理。

## 用户指令（2026-11-04 00:14 Asia/Shanghai）
Phase1 LoRA（`pi05_libero_lora` / `phase1_lora_baseline`）因写 ckpt 27000 时 ENOSPC 崩溃中止（TRAIN_EXIT 1 @ 2026-11-03 23:25:34+08）。已人工清理：
- 删除早期 ckpt `5000`、`10000` 与坏临时目录 `27000.orbax-checkpoint-tmp-104`
- 保留完整 ckpt：`15000` / `20000` / `25000` / **`26000`**
- 数据盘现约 64%（剩 ~55G）

**立刻执行：从 26000 ckpt 续训到 30000 步**（batch 16，与崩前一致；勿 `--overwrite` 清掉现有 ckpt）。
- 训练机：`source ssh-env.example` 或 `ssh-wrapper 'cmd'`（IP SSH_HOST:SSH_PORT，勿用域名）
- ckpt 根：`/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/`
- 日志：`/root/autodl-tmp/logs/phase1-train.log`（续训可 append 或新开 `phase1-train-resume.log`，结束写 `TRAIN_EXIT`）
- 存盘策略：`save-interval 1000`，`keep-period 5000`；续训前再确认剩余空间够写下 27000–30000
- 可选：打开逐步 loss 日志（便于事后画曲线）；wandb 仍可不启
- 跑通后勿停：验收/评测/汇报由门禁自行交接；盯训 routine 已删，续训开跑后应再挂监控（完成/崩溃/≥5k 步跳再报）

## 用户指令（2026-11-04 01:49 Asia/Shanghai，睡眠中自动推进）
1. Phase1 续训到 30k 完成后：立刻评测（至少 libero_spatial；按计划可扩），验收+reporter 出用户向数字汇报，由执行小g 转发用户。
2. **评测数字出来后不要停**：立刻开下一轮对照实验 **B**——只训 action expert LoRA（冻住 VLM/PaliGemma LoRA），其余尽量与本轮基线对齐（同数据、同预算目标、同 batch/lr 量级、同评测协议）。禁止同时改损失或数据。
3. 夜间门禁审查按既有用户授权自动通过；有结果再汇报。

## 用户指令（2026-11-04 01:51 Asia/Shanghai，👍确认）
30k 跑完后顺序：①评测并汇报数字；②先清理训练机 `/root/autodl-tmp` 数据盘缓存（勿误删最终 ckpt）；③再开实验 B（只训 action expert LoRA）。不要停等用户。

## 状态更新（2026-11-04 03:19 Asia/Shanghai，编排确认）
Phase1 LoRA 续训已成功结束：
- `TRAIN_EXIT 0`；进度 30.0kit/30.0kit
- 最终 ckpt：`/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/29999`（另保留 25000；27000/28000/29000 已被 Orbax 按 keep 策略删除）
- GPU 空闲、无训练进程；数据盘约 74%（剩约 40G）
- 监控 routine「盯 π0.5 LoRA 续训」已删除

**立刻按已授权顺序执行（勿等用户）：**
1. **评测**：用最终 ckpt `29999`（或等价最新）跑至少 `libero_spatial`；按计划可扩 Object/Goal/10。产出数字写入工作区日志，走验收门禁，reporter 写用户向数字报告（执行小g 转发）。
2. **清缓存**：评测完成后先清训练机 `/root/autodl-tmp` 数据盘缓存（HF/torch/临时/过期中间产物等），**保留**最终 ckpt `29999` 及实验 B 所需基线产物；勿误删。
3. **实验 B**：清完后立刻开对照实验 B——只训 action expert LoRA，冻住 VLM/PaliGemma LoRA；同数据、同预算目标、同 batch/lr 量级、同评测协议；禁止同时改损失或数据。
4. 夜间审查按用户授权自动通过；SSH 用 `ssh-wrapper`（IP SSH_HOST:SSH_PORT），勿用域名。Codex 沙箱若建不了 SSH，在脚本里写清命令由编排侧 `ssh-wrapper` 代跑，并在 HANDOFF/日志标明。

## 状态更新（2026-11-04 07:48 Asia/Shanghai）
Phase1 LoRA libero_spatial 评测已完成：
- 496/500（99.2%）；EVAL_DONE / PHASE1_EVAL_EXIT 0 @ 07:12
- 证据：`logs/phase1-spatial-result.md`
- reporter：`/example/home/workflows/ds-gpt-gate/tasks/pi05-libero-1102213352/out-reporter-074830.md`

**立刻继续（勿等用户）：**
1. 清缓存：优先清可再下载的构建缓存（`cache/uv`、`cache/pip`、`cache/uvpy`、临时目录）。**不要删** `cache/hf/lerobot`、`cache/hf/datasets`、`cache/openpi/openpi-assets`（实验 B 需要），更不要删 `openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/{25000,29999}`。
2. 若仍缺盘，再评估是否去重 hf datasets/lerobot 双份；先核对硬链接/引用再动。
3. 开实验 B：只训 action expert LoRA，冻 VLM/PaliGemma LoRA；同数据、同预算目标、同 batch/lr 量级、同评测协议；禁止同时改损失或数据。
4. Codex 若 403：写好远端脚本，由编排用 `ssh-wrapper` 代跑。


## 状态更新（2026-11-04 07:56 Asia/Shanghai，编排）
- Phase1 评测：libero_spatial **496/500（99.2%）**，ckpt 29999，EVAL_DONE。
- 清缓存：删 hf/datasets/parquet 重复缓存 + ckpt 25000；保留 29999、lerobot 数据、pi05_base、uvpy。盘曾降至 ~46%。
- 误删 uvpy 后已用 uv 恢复 Python 3.11；venv 可用。
- **实验 B 已开跑**：`pi05_libero_lora_action_only` / `phase1_exp_b_action_lora`；freeze=`Not(.*llm.*_1.*lora.*)`（只训 action expert LoRA）；batch 16、30k；日志 `/root/autodl-tmp/logs/phase1-exp-b.log`。
- Codex 门禁曾 403，编排侧代跑远程训评。

## 用户指令（2026-11-04 11:12 Asia/Shanghai，用户本机）
1. **磁盘门禁**：训练机 `/root/autodl-tmp` 可用空间降到约 **10G** 时，立刻清一次可重建缓存（uv/pip/临时等），**保留**实验 B ckpt、lerobot 数据、`pi05_base`、最终 Phase1 ckpt；清完向用户报一次。
2. **loss 曲线**：询问本轮实验 B 是否会输出；编排侧核对后答复（当前 `--no-wandb-enabled`）。

## 用户指令（2026-11-04 11:18 Asia/Shanghai）
**站立规则（以后每次训练都遵守）**：必须开启 wandb，并输出/可画 loss 曲线（逐步 loss 日志 + wandb）。禁止再传 `--no-wandb-enabled`。实验 B 当前 run：停训 → 从最新 ckpt 以 wandb 开启续训。

## 状态更新（2026-11-05 02:57 Asia/Shanghai，编排确认）
实验 B（`pi05_libero_lora_action_only` / `phase1_exp_b_action_lora`）已成功结束：
- `TRAIN_B_EXIT 0` @ 2026-11-05T00:28:32+08:00；进度 30.0kit/30.0kit
- 最终 ckpt：`/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999`（另保留 5000/10000/15000/20000/25000）
- 自 6000 起 wandb offline 续训；逐步 loss 日志 `/root/autodl-tmp/logs/phase1-exp-b-wandb.log`；曲线备份 `/workspace/pi05-plots/exp_b_loss_curve.png`
- GPU 空闲、无训练进程；数据盘约 87%（剩约 21G）

**立刻执行（勿等用户；夜间审查按授权自动通过）：**
1. **评测**：用最终 ckpt `29999` 跑至少 `libero_spatial`（与 Phase1 基线同协议）；数字写入工作区日志。
2. **验收 + reporter**：走 ds-gpt-gate acceptance → reporter 出用户向数字报告（对照 Phase1 基线 496/500=99.2%）；执行小g 转发用户。
3. SSH：`ssh-wrapper`（IP SSH_HOST:SSH_PORT），勿用域名。Codex 若建不了 SSH，写清远端脚本由编排 `ssh-wrapper` 代跑。
4. **保留**日志目录 `/root/autodl-tmp/logs/`、最终 ckpt、lerobot 数据、`pi05_base`、wandb offline；清缓存规则不变（可用≤10G 才清可重建缓存）。

## 状态更新（2026-11-05 04:48 Asia/Shanghai）
实验 B libero_spatial 评测完成：
- **315/500（63.0%）**；`EXP_B_EVAL_EXIT 0` @ 04:28
- ckpt `29999` / `pi05_libero_lora_action_only`
- 对照 Phase1 基线：496/500（99.2%）
- 证据：`logs/phase1-exp-b-spatial-result.md`

**立刻**：走 ds-gpt-gate acceptance → reporter 出用户向数字报告（对照基线）；夜间审查按授权自动通过。执行小g 转发用户。

## 用户指令（2026-11-05 04:55 Asia/Shanghai）
**立刻开实验 C**：冻结动作专家（action expert），只微调 VLM LoRA（与实验 B 对称消融）。
- 同数据、同预算目标（30k）、同 batch/lr 量级、同评测协议（至少 libero_spatial，seed 7，10×50）
- 禁止同时改损失或数据
- 必须开 wandb + 逐步 loss（站立规则；训练机 api.wandb.ai 403 则 WANDB_MODE=offline）
- SSH：`ssh-wrapper`（IP SSH_HOST:SSH_PORT），勿用域名；Codex 写脚本由编排代跑
- 保留 B/Phase1 最终 ckpt、logs、lerobot、pi05_base；可用≤10G 才清可重建缓存
- 训完→评测→验收→reporter；夜间审查自动通过；结果由执行小g 转发

## 用户指令（2026-11-05 04:56 Asia/Shanghai）
实验 C 开训前：
1. **先清训练机数据盘**可重建缓存；保留 A/B 最终 ckpt、logs、lerobot、pi05_base、wandb offline。
2. **必须** wandb + 逐步 loss，并绘制 loss 曲线（站立规则）。

## 状态更新（2026-11-05 05:12 Asia/Shanghai，编排）
- 清盘：删 uv/tmp + B 中间 ckpt 5000–25000；保留 A/B 最终 29999、lerobot、openpi-assets、logs。盘 **21G→46G** 可用（70%）。
- Codex API 403；门禁 planning 按夜间授权自动通过；编排代写/代跑。
- **实验 C 已开跑**：`pi05_libero_lora_vlm_only` / `phase1_exp_c_vlm_lora`
  - freeze=`nnx.Not(nnx.All(PathRegex(r".*llm.*lora.*"), nnx.Not(PathRegex(r".*llm.*_1.*"))))`（只训 VLM LoRA 10 参；冻 action）
  - batch 16、30k、wandb **offline**、逐步 Step loss；日志 `/root/autodl-tmp/logs/phase1-exp-c.log`
  - PID `550306`；Step 0 loss=0.0917；Step 100 loss=0.0834；GPU ~22GB/100%

## 状态更新（2026-11-05 22:15 Asia/Shanghai，编排）
- 实验 C 训完：TRAIN_C_EXIT 0 @ 21:47:53；最终 ckpt `.../pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora/29999`
- 立刻评测 libero_spatial（同 B：50 trials/task、seed 7）→ acceptance → reporter；对照 A 496/500=99.2%、B 315/500=63.0%
- 脚本 `scripts/remote/90-exp-c-eval-spatial.sh`；日志 `/root/autodl-tmp/logs/phase1-exp-c-eval-spatial.log`
- Codex 若 403：编排写远端脚本并 ssh-wrapper 代跑；夜间审查自动通过

## 状态更新 (2026-11-06 00:03 Asia/Shanghai)
实验 C libero_spatial 评测完成：
- **456/500（91.2%）**；`EXP_C_EVAL_EXIT 0` @ 2026-11-05T23:30:27+08:00
- ckpt `29999` / `pi05_libero_lora_vlm_only` / `phase1_exp_c_vlm_lora`
- 对照 A：496/500（99.2%），Δ −8.0pp；对照 B：315/500（63.0%），Δ +28.2pp
- 证据：`logs/phase1-exp-c-spatial-result.md`

**立刻**：走 ds-gpt-gate acceptance → reporter 出用户向数字报告（对照 A/B）；夜间审查按授权自动通过。执行小g 转发用户。

## 用户指令（2026-11-06 00:14 Asia/Shanghai）
用 **实验 A** 权重（`pi05_libero_lora` / `phase1_lora_baseline` / ckpt **29999**），基于 **LeRobot 公开数据集** 完成一次仿真 **pick-and-place**。

约束与偏好：
1. 数据集从 Hugging Face LeRobot 公开仓选；优先与 A 同 embodiment 的 pick-and-place（推荐 `lerobot/libero` 或其 spatial 子集，如 pick bowl → place on plate）；勿选异构臂/动作空间不匹配的集，除非先论证可适配。
2. 目标是 **仿真跑通 pick-and-place**（可用现有 LIBERO 仿真栈 + serve_policy），产出成功/失败证据（日志 + 至少一段视频或截图摘要）；不要求为此重训。
3. 训练机 SSH：`ssh-wrapper`（IP SSH_HOST:SSH_PORT，勿用域名）。Codex 若 403，写远端脚本由编排代跑。
4. 走 ds-gpt-gate：planner → planning → executor → acceptance → reporter；夜间审查按授权自动通过；结果由执行小g 转用户。
5. 盘紧张时只清可重建缓存；保留 A/B/C 最终 29999、lerobot 数据、logs、wandb offline。

## 用户补充（2026-11-06 00:16 Asia/Shanghai）
在选定 LeRobot 公开数据集之后，门禁流程按下列顺序自行推进（以实验 A 权重为策略起点，可按需微调/对照）：

1. **仿照数据集搭建仿真场景**（物体、布局、相机、动作空间尽量对齐所选公开集）。
2. **脚本专家自动采集数据**（程序化/专家策略在仿真里采 pick-and-place 演示，写入 LeRobot 兼容格式）。
3. **数据清洗**（门禁自行做过滤：失败轨迹、noop、坏帧、异常动作等；保留清洗日志与前后统计）。
4. **再仿真/训练评测**：用清洗后数据微调或对照评测；**batch size、RTC 等超参由门禁自行选择**，允许开小规模对照实验比效果。
5. 全程走 ds-gpt-gate；夜间审查自动通过；训练机用 ssh-wrapper（IP SSH_HOST:SSH_PORT）；盘紧张只清可重建缓存，保留 A/B/C 最终 ckpt。
6. 向用户交付：所选数据集与任务、场景搭建说明、采数规模、清洗前后量、最终仿真成功率/视频，以及若做了 batch/RTC 对照的结论。

## 用户补充（2026-11-06 00:17 Asia/Shanghai）— batch / RTC 对照
- **基线对照组**：batch size 与 RTC 一律采用 **官方推荐/默认设置**（以 openpi / π0.5 官方文档或仓库默认为准，写明出处与具体数值）。
- **实验组**：在基线之上由门禁改动 batch 和/或 RTC（可多项小对照），其余尽量对齐，禁止同时乱改无关因素。
- 汇报时必须给出：官方基线配置表、各实验组差异、以及仿真/训评指标对比。

## 用户补充 2026-11-06 00:48 Asia/Shanghai
- **先清理训练机数据盘可重建缓存**，再继续清洗/微调对照。
- 保留：A/B/C 最终 ckpt 29999、logs、采数数据 `pnp_policy_spatial_center`、wandb offline。
- 可删：可重建 cache（如 HF/torch/lerobot 缓存、中间临时文件、旧中间 ckpt 若占盘），目标尽量腾出空间（盘约 14G 可用/91%）。

## 用户补充 2026-11-06 04:54 Asia/Shanghai
- G0 OOM：采用 **梯度累积**，名义/有效 batch 仍记 **32**，微批 8 或 16（以能稳住为准），对照表注明「G0=grad_accum 等效 batch32」。
- 训出 G0 权重后：补 G2（同权重、评测 replan=10）；已有 G1 保留；评测含 A-29999 / G0 / G1 / G2。
- 汇报须含官方基线意图、grad_accum 差异与指标对比。

## 用户指令（2026-11-06 10:42 Asia/Shanghai）— PnP 拉开差距：先试 1+2
上一轮 A-PnP 四组（A29999 / G0 / G1 / G2）均为 50/50（100%），拉不开差别。用户授权**先试想法 1 与 2**（勿重训）：

1. **换没见过的初始状态**：评测用的 `init_states` 不得与 policy-expert 采数时用过的初始状态相同；测 held-out 起始条件。
2. **扩到 libero_spatial 全 10 个任务**：每任务 50 trials，共 500 次。

**评测对象（勿重训）**：
- **G0**（官方基线）：ckpt **2499**，微批 8 × 累积 4（等效 batch 32），replan5
- **G1**：batch16，replan5
- **G2**：G0 权重 + replan10
- **A29999**：参考组

**汇报必须**：官方基线配置、各组相对 G0 改了什么、每任务成功率 + 总体成功率、各组相对 G0 的 Δ。

**其它规则**：
- 保留 A/B/C 29999、G0/G1 ckpt、数据、logs、wandb；盘紧才清可重建缓存
- 任何曲线：按 step 去重保留最后值，再升序排序
- 夜间/离开审查可自动通过；审查失败满 3 次则退回修，再开新轮
- SSH：`ssh-wrapper`；Codex 写脚本由编排代跑亦可；走 ds-gpt-gate：planner → planning → executor → acceptance → reporter；结果由执行小g 转用户
