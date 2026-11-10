# Phase 0 远端执行

当前仅交付只读预检入口；环境安装、版本冻结、模型下载及评测均尚未执行。

目标为 `SSH_USER@SSH_HOST:SSH_PORT`，数据目录 `/root/autodl-tmp`。
不要使用旧主机名，不重复清空数据盘。

```bash
bash -n scripts/remote/00-preflight.sh
ssh -p "$SSH_PORT" SSH_USER@SSH_HOST 'bash -s' < scripts/remote/00-preflight.sh
```

自动化密码认证须由编排侧注入环境变量；密码不得写入脚本、日志、命令行或仓库。
不要开启 `set -x` 或输出完整环境变量。当前执行器使用 `python3`，本地无 `python` 命令。

脚本只读取系统、磁盘、内存、GPU、工具版本与网络状态；不安装、不下载权重、不删除数据。
预计新增占用按审批预算 105 GiB 保守估算，预计总占用超过 85% 或余量不足 20 GiB 时退出。
网络状态仅作证据：桶根目录 HTTP 403 不等于权重对象无法访问。

planning 已通过（2/3）。发生需处理的错误后停在 error 门禁，交小co／Architect 审查；
不得自行重试修复或增加审查轮次。Phase 0 门槛仍为固定协议下 ≥494/500，Phase 1 为条件阶段。
