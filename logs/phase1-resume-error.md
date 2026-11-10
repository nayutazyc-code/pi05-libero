# Phase 1 续训：error 门禁交接

- 时间：2026-11-04 00:16 Asia/Shanghai。
- 当前授权：从 `phase1_lora_baseline/26000` 续训到 30000；batch 16，save_interval 1000，keep_period 5000；禁止 overwrite。
- 已读：任务目录 HANDOFF.md、REQUIREMENTS.md、经验总结、计划书全文和现有远端训练脚本。
- 旧脚本 `scripts/remote/40-phase1-probe-train.sh` 含 `--overwrite`，本次未运行。续训不能直接重跑该链。

## 预检及报错

通过 `source ssh-env.example` 加载现有凭据，调用 `ssh-wrapper`，目标为 `SSH_USER@SSH_HOST:SSH_PORT`。仅尝试读取时间、磁盘、GPU、训练进程、检查点目录和训练日志尾部。

命令退出码 1；本地创建 socket 时失败，尚未建立 SSH 连接：

```text
paramiko/client.py, line 358, in connect
    sock = socket.socket(af, socket.SOCK_STREAM)
socket.py, line 233, in __init__
    _socket.socket.__init__(self, family, type, proto, fileno)
PermissionError: [Errno 1] Operation not permitted
```

与 `logs/phase1-connection-error.txt` 的历史阻塞相同。当前执行环境 network restricted、approval policy never，无法申请提权；此证据不能用于判断远端主机或凭据是否异常。

## 执行边界及交接

- 本次未执行任何远端命令；未启动续训或监控，未修改或删除检查点，未输出或保存凭据。
- 26000 完整性、最新训练进程、可用空间和训练实际状态均未核实。REQUIREMENTS 中的清理及剩余约 55G 为用户提供的状态。
- 按用户要求停在 error 门禁，等待现有编排侧小co / Architect 审查；未另起审查轮次，未调整审查额度，未修改任务目录状态。
- 需要在允许 SSH 的执行环境继续：先核实无重复训练及 26000 可恢复，查固定版本的 resume/检查点保留逻辑，按实际 checkpoint 大小核算 27000–30000 写盘峰值且保留至少 20GB，再以原配置 batch 16 续训，不能使用 overwrite。
- 启动后核实恢复步数、loss 和保存进度，记录 PID、新日志及 TRAIN_EXIT，再恢复完成/崩溃/≥5k 步进度监控；执行完成后再进入验收及评测交接。
- Phase 0 已验收，不重开；本次不宣称 Phase 1 训练或验收完成。
