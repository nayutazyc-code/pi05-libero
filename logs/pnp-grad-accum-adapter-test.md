# G0 grad-accum adapter 训练机测试原文

验收补测：2026-11-06 09:59 Asia/Shanghai。通过 `ssh-wrapper` 连接训练机（SSH_HOST:SSH_PORT），使用 openpi 的 `.venv/bin/python`；本次不是 box 上缺少 flax 的失败尝试。

## 测试入口
- 脚本: `scripts/remote/test_pnp_grad_accum.py`（依赖 flax.nnx，训练机 openpi 环境）
- 实现: `scripts/remote/pnp_grad_accum.py` / `train_pnp_grad_accum.py`
- 断言覆盖：micro8/16/32 梯度与整批一致；clip/AdamW/EMA 各一次；非法 micro size 拒绝

## 远端命令

```bash
source /root/autodl-tmp/env.sh
cd /root/autodl-tmp/openpi
export JAX_PLATFORMS=cpu
export PNP_MICROBATCH_SIZE=8
.venv/bin/python scripts/test_pnp_grad_accum.py
```

## 完整输出（stdout 与 stderr 合并）

以下原文另存 `logs/pnp-grad-accum-adapter-test.txt`。退出码为0。三份源码的 SHA256 与工作区 `scripts/remote/` 对应文件逐一核对一致；测试未修改远端源码。

```text
ADAPTER_TEST_START 2026-11-06T09:59:47+08:00
/root/autodl-tmp/openpi
Python 3.11.18
JAX_PLATFORMS=cpu PNP_MICROBATCH_SIZE=8
0d5719fd3b0c06b977c3b25a974dc81b7e1828dd088695b444b85b76afffa551  scripts/test_pnp_grad_accum.py
1e08f1d056ccf7481b5f675d60164372c95f591128e5f363042366dc434a85f9  scripts/pnp_grad_accum.py
b323fae4c785f99350af12cb627095e9e618cade486f2f6b67321e29e2e7f119  scripts/train_pnp_grad_accum.py
COMMAND: .venv/bin/python scripts/test_pnp_grad_accum.py
PASS: micro8/16/32 = full batch; adapter clips/updates/EMA once; invalid size rejected
ADAPTER_TEST_EXIT 0
ADAPTER_TEST_END 2026-11-06T09:59:57+08:00
```

本测试使用确定性的 TinyModel 核对数值；实际 `train_step` 以微批8运行。它不证明随机模型不同微批 RNG 路径与整批完全等价，相关限制仍见结果表。
