# PnP Executor — 报错重现门禁待审

> 2026-11-06 01:38 更新：本文件中的 SSH 限制已解除，本轮已通过 `ssh-wrapper` 成功读取远端日志、GPU 和磁盘。当前待审事项为 G0 OOM 及 state 数据契约不一致，见 `logs/pnp-executor-error-gate.md`；下文保留为历史记录。

任务：`pi05-a-pnp-1106001525`。

## 已核对

- 已读任务目录 `HANDOFF.md`、项目 `REQUIREMENTS.md`、采数脚本与本地日志。
- HANDOFF 记录 planning 已于 2026-11-06 00:19 夜间自动通过；本次不重开规划审查。
- 交接记录采数完成：30 成功、1 失败、31 次尝试，退出码 0；远端数据目录 `/root/autodl-tmp/datasets/pnp_policy_spatial_center/`。这些来自交接记录，本次未能远端复核。

## 本次复现

执行只读命令：

```bash
ssh-wrapper 'date -Iseconds; df -h /root/autodl-tmp; tail -8 /root/autodl-tmp/logs/pnp-policy-collect.log'
```

退出码 1，关键错误：

```text
paramiko/client.py, line 358, in connect
    sock = socket.socket(af, socket.SOCK_STREAM)
socket.py, line 233, in __init__
    _socket.socket.__init__(self, family, type, proto, fileno)
PermissionError: [Errno 1] Operation not permitted
```

错误发生在本机创建 socket 阶段，未连接训练机，未读取远端磁盘或日志。本环境禁止提权申请，未尝试绕过限制。

## 门禁与交接

按用户要求停在“报错重现”节点，等待小co / Architect 审查。未自行宣布夜间审查通过，未启动额外审查轮次。

审查后可由具备 SSH 权限的编排侧代跑远端核验，并继续已授权流程：

1. 检查采数产物及训练数据契约。当前采数脚本保存 NPY episode；进入训练前需核对 LeRobot 格式、state 表示与图像方向。
2. 清洗失败、长 noop、坏帧、异常动作，输出真实前后统计。
3. 从 A 的 29999 小规模微调，启用 wandb 与逐步 loss；G0 batch32、G1 batch16，其余训练设置对齐。
4. 同协议评测 task_id=2、10 trials；G0/G1 replan5，G2 复用 G0 权重且仅改 replan10，保留视频。
5. 保留 A/B/C 最终 29999、lerobot、logs、wandb；磁盘紧张仅清可重建缓存。
6. 最终验收门禁通过后交 reporter。

本次未修改远端数据或权重，未执行清洗、微调或新增评测，不能给出这些阶段的结果。
