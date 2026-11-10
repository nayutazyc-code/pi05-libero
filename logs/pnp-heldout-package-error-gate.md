# PnP held-out 证据回拷：error 门禁

任务：`pi05-pnp-heldout-spatial-1106104308`。记录时间：2026-11-06T17:09:58+08:00。

已停在 error 门禁，等待小co / Architect 审查；未修复、未重试、未自行追加审查轮次。
任务 state 只读核验：`reviews_used=3`、`max_reviews=6`；未改 phase 或计数。

## 报错与原因

通过已授权的 `ssh-wrapper` 读取远端 SHA256 文件清单，命令使用 `python3 -c ...`，返回：

```text
bash: line 1: python3: command not found
subprocess.CalledProcessError: remote inventory command returned non-zero exit status 127.
```

执行者错误假设远端非交互 shell 的 PATH 含 `python3`。此次仅证明该命令不可解析，不代表项目虚拟环境不可用。
原始摘要：`pnp-heldout-spatial-1106104308/packaging-error.txt`。
失败发生在下载 tar 之前，未开始回拷或覆盖本地已有证据；未执行远端写入、训练、仿真或评测。

## 已核对

- 已读任务原 HANDOFF、PLAN、REQUIREMENTS 及 state。
- 只读 SSH 成功；远端证据目录约 24 MB，40 份 formal 日志、8 份 preflight 日志、4 份 server 日志、summary、trials、videos、heldout、protocol 均列出。
- 远端 `summary/report.md` 为 Complete: 2000/2000 valid trials，A29999 487/500（97.4%），G0 248/500（49.6%），G1 285/500（57.0%，+7.4pp），G2 223/500（44.6%，-5.0pp）；这是远端汇总核对，尚未进行本轮本地独立复算。
- 既有 `logs/plots/` 中 G0/G1 loss CSV/PNG 和 provenance 仍在；本轮尚未验证 steps 顺序或图像解码。
- 本地依然缺完整 formal 记录、最终 summary 和 videos；不满足提交 acceptance 的材料条件。
- 没有重训、重评、删除受保护资产。未申请或执行额外审查。

## 建议审查范围与后续（尚未执行）

仅将回拷前清单采集器中的远端解释器由 `python3` 改为项目既有绝对路径 `/root/autodl-tmp/openpi/.venv/bin/python`。
该解释器路径已见冻结/续跑脚本；使用前只读确认可执行。无需修改评测脚本、冻结协议或环境安装。

获本节点审查允许后：
1. 修复本地临时回拷工具 `/tmp/pull_pi05_heldout_evidence.py` 的上述命令；通过 ssh-wrapper 只读列清单、tar 回拷完整目录，并按远端 SHA256 逐文件核对。
2. 本地运行既有 `independent_audit.py` 从全部 trial 重算、验证状态内容排除与 G0/G2 一致性；验证全部代表视频解码和 loss steps 去重升序。
3. 补 `logs/pnp-heldout-acceptance-evidence.md`、更新 PLAN 和交接；然后停在 acceptance 门禁等待审查。

任务原 HANDOFF 位于工作区可写范围之外，本轮交接增量保存于 `pnp-heldout-spatial-1106104308/HANDOFF-packaging.md`，供编排同步；不修改外部任务目录。
