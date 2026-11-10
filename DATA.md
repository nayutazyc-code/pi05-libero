# PnP 数据集与仿真视频


本页仅包含本次 π0.5 复现/微调项目的自采仿真数据和已保存的仿真视频，不包含模型 checkpoint、凭据或上游 LIBERO 完整训练集。

## 内容与格式

采集任务为 `libero_spatial` 的 task 2：将桌中央黑碗放到盘子上。专家为 A29999，重规划间隔 5，10 fps，双相机 256×256。采集共尝试 31 次（30 成功、1 失败），保存的是 30 条成功轨迹、共 3327 帧；失败尝试仅在汇总记录中。

| Release 文件 | 字节数 | 内容 |
|---|---:|---|
| `pnp-policy-spatial-center-raw.tar.gz` | 502262699 | 原始数据；解压目录 `pnp_policy_spatial_center/`，含 `episode_0000`–`episode_0029`、`summary.json` |
| `pnp-policy-spatial-center-lerobot.tar.gz` | 416241362 | 清洗后的 LeRobot v2.1；解压目录 `local/pnp_policy_spatial_center/`，30 个 Parquet 及 `meta/` |
| `simulation-videos.tar.gz` | 10926085 | 训练机 logs 下现存的全部 152 条 MP4，保留来源目录 |
| `SHA256SUMS` | — | SHA-256 校验文件 |

原始每条轨迹含 `image.npy`、`wrist_image.npy`、`state.npy`、`action.npy`、`meta.json`、`replay.mp4`。LeRobot 图像嵌入 Parquet，`meta/info.json` 定义特征：`image`、`wrist_image`、`state`（float32, 8D）、`actions`（float32, 7D）及时间/索引字段。

原始数据文件合计 1,310,157,722 字节，清洗版 417,007,381 字节。均以 Release 附件提供，避免将近 GB 数据写入 Git 历史；无需 Git LFS，各附件均小于 2 GB。

## 下载与校验

发布页：<https://github.com/nayutazyc01-ui/pi05-libero/releases/tag/data-v1>

发布后运行：

```bash
gh release download data-v1 --repo nayutazyc01-ui/pi05-libero --dir pi05-data
cd pi05-data
sha256sum -c SHA256SUMS
mkdir -p raw lerobot
tar -xzf pnp-policy-spatial-center-raw.tar.gz -C raw
tar -xzf pnp-policy-spatial-center-lerobot.tar.gz -C lerobot
```

将 `HF_LEROBOT_HOME` 指向解压后的 `lerobot/`，dataset id 为 `local/pnp_policy_spatial_center`。

```text
e8872081f5effe2887673959554557a39742300b9c7a2869967e7075d5dbaff4  pnp-policy-spatial-center-raw.tar.gz
d651b5e0104c0304241bb1e3133431e82b999850471a106973cdcb58965f43ce  pnp-policy-spatial-center-lerobot.tar.gz
a04510854d464bdf5b75d132922504179bf5739409ecd6b5e8b909d3eb7a8502  simulation-videos.tar.gz
```

## 视频与解释边界

- 仓库 `media/` 计划保留所有 152 条日志视频，涵盖官方复现、A/B/C、PnP seen-init、held-out 正式与预跑；另含专家示例 `pnp-a-expert-ep0000.mp4`。视频本来就很小，无需二次编码。
- 30 条采数 replay 随原始数据包完整提供，不重复放进 Git。没有省略现有日志视频，但不能把示例视频数量当作评测回合数。
- held-out 正式 67 条、预跑 8 条只是抽样/类别代表录像，不是 2000 回合完整录像；定量结果请看项目报告。
- 这些是仿真画面，不是真机录像。原始和清洗版保留本次实验的 state 表示；8D state 不符合官方轴角接口，不应作为已修正的官方格式数据使用。
- 只有单任务采数，不代表覆盖 10 个任务的演示数据。held-out 隔离范围与训练预算混杂等限制见 [项目总结](PROJECT_SUMMARY.md)。
- 上游 LIBERO/LeRobot/模型资产的许可证仍须分别遵守；本发布不为上游资产另授许可证。
