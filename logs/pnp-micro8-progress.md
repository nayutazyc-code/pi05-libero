# G0 微批 8 续执行（2026-11-06 05:12 Asia/Shanghai）

- 已读任务目录 HANDOFF.md 与 state.json：planning/error 已通过，reviews_used=2/3；本轮沿用授权，不追加错误审查。
- 远端 GPU 空闲，无重复训练；可用盘约 94G。
- 已核验前次 micro16×2：`TRAIN_G0_ACCUM_EXIT 1 @04:59:42`，首步申请 14774015584 bytes 失败。
- 05:11:42 启动 micro8×4，effective batch32，2500 optimizer updates，W&B offline；shell PID 762517，训练 PID 762523。
- 独立实验：`/root/autodl-tmp/openpi/checkpoints/pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8`。
- 独立日志：`/root/autodl-tmp/logs/pnp-g0-accum-mb8-train.log`；保留 native32 / micro16 的 OOM 日志与原 W&B run。
- 日志确认 A/29999/params 恢复成功、梯度均值后裁剪一次并 AdamW 更新一次；微批仅用于分割 batch，step 仍为优化更新。
- CPU 验证实际 adapter（PNP_MICROBATCH_SIZE=8）：micro8/16/32 梯度与整批一致；clip/AdamW/EMA 一次；非法 micro size 拒绝，PASS。
- 已将 G0/G2 评测入口与 W&B 导出器切换到 mb8，并保留 mb16 失败 run 的单独空 CSV。
- A/G1 现有 summary 均为 50/50；微批8 Step0 已成功，训练尚未完成。

## 首步通过与后续编排（05:16）

- Step 0 @05:15:17：loss=0.0883，grad_norm=0.3270；GPU 22111MiB/24564MiB，100%。
- 后续脚本 `scripts/remote/102-pnp-after-g0-mb8.sh` 已启动，等候训练终态；成功后只评 G0/G2 各50次，不重复 A/G1。
- 后续日志 `/root/autodl-tmp/logs/pnp-g0-mb8-followup.log`；W&B 导出后核对文本日志、CSV step 去重升序。
- 仍需训练/评测结束后取回完整数字与视频，再提交最终验收门禁；本文件不是完成或验收通过声明。

## 08:14 核验

- SSH 重连后，远端日志确认训练约2050/2500，rate约5.2s/update；未见 OOM，预计剩余约39分钟。
- SSH 期间短暂失败不影响远端 nohup 作业；最终训练退出码和 G0/G2 分数仍待核实。

## 最终完成（09:16）

- G0 micro8×4：TRAIN_G0_ACCUM_EXIT 0 @08:53:37，checkpoint `pnp_g0_accum_bs32_mb8/2499`。
- G0/G2各50/50；eval退出码均0；PNP_MB8_FOLLOWUP_DONE @09:06:47。
- 本地验证4组summary的200条trial、同权重G0/G2、4组视频可解码、loss CSV step唯一升序且与日志一致，全通过。
- G0 W&B run `laea5vht`：50点，0.0882594213→0.0101199802；最终记录step2450。
- 原生32/micro16失败日志和W&B未覆盖；A/B/C最终29999、G1/2499与原数据存在；盘85G可用。
- 进入最终验收节点，等待审查；此处未声明acceptance通过或reporter完成。
