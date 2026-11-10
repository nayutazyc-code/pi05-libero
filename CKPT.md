# pi0.5 最终检查点备份

下载：[GitHub Release ckpt-v1](https://github.com/nayutazyc01-ui/pi05-libero/releases/tag/ckpt-v1)。

训练机停机仅保留 15 天，因此将本项目五个最终检查点独立备份到 Release；不依赖训练机继续存在。G2 与 G0 共用权重，没有独立检查点。

## 备份范围

每个归档保留原始相对路径，包含完整 `params/`（包括冻结的主干参数，并非只有 LoRA adapter）、`assets/` 下的归一化统计及 `_CHECKPOINT_METADATA`。未合并、转换或量化权重。

**未包含 `train_state/`（优化器、训练状态）和中间检查点**，因此本备份用于推理/评测或作为新微调的初始化，不保证无缝恢复原优化器状态。五个最终检查点原目录合计约 39.71 GB，其中省略的 `train_state` 约 9.92 GB。原训练机文件未删除。

| 组别 | 配置名 | 解压后的相对目录 | 压缩归档 GB | 分卷数 |
|---|---|---|---:|---:|
| A29999 | `pi05_libero_lora` | `pi05_libero_lora/phase1_lora_baseline/29999` | 6.335772 | 4 |
| B29999 | `pi05_libero_lora_action_only` | `pi05_libero_lora_action_only/phase1_exp_b_action_lora/29999` | 5.382651 | 3 |
| C29999 | `pi05_libero_lora_vlm_only` | `pi05_libero_lora_vlm_only/phase1_exp_c_vlm_lora/29999` | 5.394970 | 3 |
| G0-2499 | `pi05_libero_lora_pnp` | `pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499` | 6.335755 | 4 |
| G1-2499 | `pi05_libero_lora_pnp` | `pi05_libero_lora_pnp/pnp_g1_bs16/2499` | 6.335747 | 4 |

总计 **29.784895 GB**，共 **18 卷**。

以上 GB 均为十进制，1 GB = 10^9 字节。每卷最大 1,900,000,000 字节，严格小于 GitHub 单附件 2 GB 限制。分卷是同一个 zstd 流的连续片段，不能单独解压。

## 下载、校验、重组

安装 GitHub CLI、GNU tar、zstd。仓库私有时，先用自己的 GitHub 账号登录 `gh`。

```bash
mkdir -p pi05-ckpt-download && cd pi05-ckpt-download
gh release download ckpt-v1 --repo nayutazyc01-ui/pi05-libero --pattern '*.part*' --pattern SHA256SUMS
sha256sum -c SHA256SUMS
mkdir -p restored
for name in A29999 B29999 C29999 G0-2499 G1-2499; do
  cat "${name}.tar.zst.part"* | zstd -d -c | tar -xf - -C restored
done
```

只下载一组，例如 G0：

```bash
gh release download ckpt-v1 --repo nayutazyc01-ui/pi05-libero --pattern 'G0-2499.tar.zst.part*' --pattern SHA256SUMS
# 仅验证已下载的 G0 卷；全量下载则使用上一节的完整校验。
grep '  G0-2499.tar.zst.part' SHA256SUMS | sha256sum -c -
mkdir -p restored
cat G0-2499.tar.zst.part* | zstd -d -c | tar -xf - -C restored
```

建议保留 SHA256SUMS 与所有分卷，在本地或独立存储留第二份副本。解压后的目录结构与 `openpi/checkpoints/` 下相同，放在哪里均可。

## 使用 openpi 加载

使用本仓库记录的 openpi 环境、模型配置及补丁；上游原版 openpi 没有本项目自定义配置名，不能只下载权重就直接运行。配置生成/安装步骤见 `scripts/remote/31-add-pi05-lora-config.py`、`60-exp-b-action-lora.sh`、`80-exp-c-vlm-lora.sh`、`98-pnp-finetune-g0-g1.sh`；其中 shell 脚本还包含训练命令，恢复环境时不要无意重启训练或执行其清理步骤。

在已配置的 openpi 工作目录中，替换下面的绝对路径，加载 G0：

```python
from openpi.training import config
from openpi.policies import policy_config

cfg = config.get_config("pi05_libero_lora_pnp")
policy = policy_config.create_trained_policy(
    cfg,
    "/absolute/path/pi05-ckpt-download/restored/"
    "pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499",
)
# 传入与 LIBERO 数据变换匹配的 observation 后调用 policy.infer(observation)。
```

或启动推理服务（命令在 openpi 目录执行）：

```bash
.venv/bin/python scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=pi05_libero_lora_pnp \
  --policy.dir=/absolute/path/pi05-ckpt-download/restored/pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499
```

其他组按表格替换配置名和路径。复现实验时还须保留对应观测/state 编码、动作 horizon 与 replan 设置；G0/G1 默认 replan=5，G2 使用 G0 权重、replan=10，服务加载权重并不自动更改客户端 replan。

## 完整性检查

上传前逐个完整解压 zstd 流并让 tar 检查归档目录，确认含 `params` 和 `norm_stats.json`。上传后核对 GitHub 附件大小与 SHA-256 摘要；`SHA256SUMS` 是每个 Release 分卷的校验和。此处不声称重新跑了 GPU 推理验收。
