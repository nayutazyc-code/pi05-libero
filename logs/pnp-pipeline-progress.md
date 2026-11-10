# A-PnP 执行完成，待最终验收（2026-11-06 09:16 Asia/Shanghai）

| 环节 | 已核验结果 |
|---|---|
| 数据/场景 | lerobot/libero_spatial_image；LIBERO spatial task2，Panda 黑碗→盘 |
| 专家采数 | A-29999 策略专家，31 次尝试：30 成功 / 1 失败 |
| 清洗/转换 | 30 eps / 3327 frames → 30 / 3327；LeRobot local/pnp_policy_spatial_center |
| G0 原生 batch32 | OOM，保留原日志和 W&B run |
| G0 micro16×2 | OOM，保留原日志和 W&B run |
| G0 micro8×4 | 有效 batch32，2500 optimizer updates，08:53:37 退出0，最终2499 |
| G1 batch16 | 2500 optimizer updates，03:13:05 退出0，最终2499 |
| A-29999 / G0 / G1 / G2 | 各 50/50 = 100%；G2 用 G0 权重，replan10；其他replan5 |
| W&B loss | G0/G1 各50点，step0–2450，去重升序，与文本日志交叉核对通过 |
| 本地证据 | 表、4组JSON、每组3段成功视频、W&B CSV/PNG 已取回 |
| 远端保留 | A/B/C最终29999、G1/2499、原PnP数据、logs、W&B均保留；盘85G可用 |

G0 W&B loss：0.0882594 → 0.0101200；G1：0.1120962 → 0.0119648（末点step2450，不是最后一步单点loss）。
原生32、micro16×2各无有效loss点；各保留独立空CSV，不绘伪曲线。

关键限制：现有state为xyz+quat+单指，与官方axis-angle+双指语义有差异，error门禁已批准保留并披露。
评测前50个初始状态含采数前31次状态，非独立泛化/官方10任务benchmark。
G0/G1优化更新数相同，但样本暴露80000/40000且RNG路径不同；G0为梯度累积，不能宣称原生batch32一次通过。
重规划对照为同步chunk replan，非异步RTC inpainting；四组成功率相同，未观察到成功率改善。

planning/error均已有审查通过记录（2/3）。执行已完成，证据见`pnp-acceptance-evidence.md`；按用户要求停在最终验收节点，等待小co/Architect。
未自行登记acceptance通过、未增加报错审查、未运行reporter或commit/push。
