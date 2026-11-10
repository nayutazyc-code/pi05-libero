# PnP held-out spatial：关键代码审查包

任务：`pi05-pnp-heldout-spatial-1106104308`。本件回应 17:21 acceptance「需改」的代码可读性补件；仅摘录已冻结实现，不修改脚本、协议、清单或正式记录。审查额度 `reviews_used=5 / max_reviews=6`，剩余 1 次 acceptance；未自行登记通过。

完整 Diff：[implementation.diff](pnp-heldout-spatial-1106104308/implementation.diff)。以下代码逐段原文摘自该 Diff 对应的 implementation/ 快照；省略处以独立代码块分隔。代码片段、原文件行号和 SHA 可直接对照。审计结果与复核边界见 [验收证据](pnp-heldout-acceptance-evidence.md#审查补件审计摘要)。

## 1. 显式初态与内容隔离

入口带 init_manifest 时转入独立评测。内容身份忽略模拟器时间，按 float64、12 位小数规范化；每任务必须恰好 50 个唯一状态，全部排除采数 31 个内容指纹，索引必须为 0–49。加载指定状态后拒绝预先成功及等待期间成功。

来源：`implementation/eval_libero_task.py:50–53`；[源码](pnp-heldout-spatial-1106104308/implementation/eval_libero_task.py)。

<!-- excerpt eval_libero_task.py 50:53 -->
```python
def main(args: Args):
    if args.init_manifest:
        from pnp_heldout_eval import evaluate
        return evaluate(args)
```

来源：`implementation/build_pnp_heldout.py:39–45`；[源码](pnp-heldout-spatial-1106104308/implementation/build_pnp_heldout.py)。

<!-- excerpt build_pnp_heldout.py 39:45 -->
```python
def state_hash(state):
    # Time alone must not make a repeated physical state appear held-out.
    state = np.round(np.asarray(state, dtype="<f8")[1:], 12)
    state[state == 0] = 0  # canonicalize negative zero
    return hashlib.sha256(state.tobytes()).hexdigest()
```

来源：`implementation/pnp_heldout_eval.py:58–70`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 58:70 -->
```python
def verify_manifest(manifest):
    assert [t["task_id"] for t in manifest["tasks"]] == list(range(10))
    excluded = set(manifest["excluded_state_hashes"])
    assert len(excluded) == 31
    for task in manifest["tasks"]:
        states = task["states"]
        assert len(states) == 50
        identities = [state_hash(x["state"]) for x in states]
        assert len(set(identities)) == 50 and not (set(identities) & excluded)
        assert identities == [x["state_hash"] for x in states]
        assert [x["trial_index"] for x in states] == list(range(50))
```

来源：`implementation/pnp_heldout_eval.py:144–153`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 144:153 -->
```python
                np.random.seed(7)
                env.seed(7)
                env.reset()
                obs = env.set_init_state(np.asarray(entry["state"], dtype=np.float64))
                if env.check_success():
                    raise RuntimeError("frozen initial state already successful")
                for _ in range(args.num_steps_wait):
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    if done or env.check_success():
                        raise RuntimeError("frozen initial state succeeds during wait")
```

## 2. 协议、权重和归一化校验

每次启动/续跑先由入口调用 freeze_pnp_heldout.py --verify，核对清单、脚本、来源源码、全部权重分片及归一化文件。评测再核对参数、任务和服务 metadata；G2 在冻结时直接复用 G0 资产、只覆盖 replan=10。

来源：`implementation/103-pnp-heldout-spatial.sh:30–31`；[源码](pnp-heldout-spatial-1106104308/implementation/103-pnp-heldout-spatial.sh)。

<!-- excerpt 103-pnp-heldout-spatial.sh 30:31 -->
```bash
# Verify actual files on every start/resume, including model shards and normalizers.
.venv/bin/python "$W/scripts/freeze_pnp_heldout.py" --out "$PROTOCOL" --verify
```

来源：`implementation/freeze_pnp_heldout.py:30–43`；[源码](pnp-heldout-spatial-1106104308/implementation/freeze_pnp_heldout.py)。

<!-- excerpt freeze_pnp_heldout.py 30:43 -->
```python
def check(protocol):
    assert sha(protocol["manifest"]) == protocol["manifest_sha256"]
    for name, digest in protocol["eval_scripts_sha256"].items():
        assert sha(W / "scripts" / name) == digest, name
    for entry in protocol["source_files"]:
        assert sha(entry["path"]) == entry["sha256"], entry["path"]
    checked = set()
    for group in protocol["groups"].values():
        for entry in group["params"] + group["normalization"]:
            if entry["path"] not in checked:
                assert sha(entry["path"]) == entry["sha256"], entry["path"]
                checked.add(entry["path"])
```

来源：`implementation/pnp_heldout_eval.py:74–86`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 74:86 -->
```python
    protocol = json.loads(pathlib.Path(args.protocol_file).read_text())
    protocol_hash = file_hash(args.protocol_file)
    assert file_hash(args.init_manifest) == protocol["manifest_sha256"]
    for name, expected in protocol["eval_scripts_sha256"].items():
        assert file_hash(pathlib.Path(__file__).with_name(name)) == expected, name
    manifest = json.loads(pathlib.Path(args.init_manifest).read_text())
    verify_manifest(manifest)
    group = protocol["groups"][args.tag]
    assert args.checkpoint_path == group["checkpoint"] and args.policy_config == group["config"]
    for key in ("seed", "max_steps", "num_steps_wait", "resize_size", "task_suite_name"):
        assert getattr(args, key) == protocol[key], key
    assert args.replan_steps == group["replan"] and 0 <= args.task_id < 10
    assert args.num_trials == 50
```

来源：`implementation/pnp_heldout_eval.py:157–159`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 157:159 -->
```python
                    metadata = client.get_server_metadata()
                    assert metadata["heldout_protocol_sha256"] == protocol_hash
                    assert metadata["heldout_group"] == args.tag
```

来源：`implementation/freeze_pnp_heldout.py:72–72`；[源码](pnp-heldout-spatial-1106104308/implementation/freeze_pnp_heldout.py)。

<!-- excerpt freeze_pnp_heldout.py 72:72 -->
```python
    groups["G2"] = dict(groups["G0"], replan=10)
```

## 3. 续跑身份、重复拒绝与逐事件落盘

组×任务目录持排他锁；已有 result 按 trial_index 跳过，协议/stage 不符或重复 result 直接拒绝。trial_id 含组、任务、内容哈希；重试保留原身份并增加 attempt，start/result/error 分文件留存。durable_write 拒绝覆盖既有事件，fsync 后原子提交；函数余下部分还对父目录 fsync。

来源：`implementation/pnp_heldout_eval.py:99–111`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 99:111 -->
```python
    lock = (root / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    videos = pathlib.Path(args.video_out_path)
    videos.mkdir(parents=True, exist_ok=True)
    try:
        events = read_events(root)
        assert all(e["protocol_sha256"] == protocol_hash for e in events), "resume protocol mismatch"
        assert all(e["stage"] == stage for e in events), "resume stage mismatch"
        finished = {}
        for e in events:
            if e["event"] == "result":
                assert e["trial_index"] not in finished, "duplicate scored trial"
                finished[e["trial_index"]] = e
```

来源：`implementation/pnp_heldout_eval.py:124–137`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 124:137 -->
```python
            ep = entry["trial_index"]
            if ep in finished:
                continue
            past = [e for e in events if e["trial_index"] == ep and e["event"] == "start"]
            attempt = len(past) + 1
            # Incomplete previous starts remain visible as interrupted attempts.
            identity = f"{args.tag}:libero_spatial:{args.task_id}:{entry['state_hash']}"
            base = {"trial_id": identity, "group": args.tag, "task_id": args.task_id,
                    "trial_index": ep, "state_hash": entry["state_hash"], "attempt": attempt,
                    "protocol_sha256": protocol_hash, "seed": 7, "stage": stage}
            prefix = root / f"trial{ep:02d}_attempt{attempt:03d}"
            start = dict(base, event="start", time_unix=time.time())
            durable_write(str(prefix) + "_start.json", start)
            events.append(start)
```

来源：`implementation/pnp_heldout_eval.py:26–37`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 26:37 -->
```python
def durable_write(path, record):
    # One JSON file per event; rename commits complete records atomically.
    path = pathlib.Path(path)
    if path.exists():
        raise ValueError(f"refusing to replace trial event {path}")
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))
```

## 4. 任务失败与执行异常分开

默认控制步耗尽为 timeout（计失败），env.check_success() 成功才计成功；done 而未成功记 policy_failure。600 秒服务/执行墙钟上限抛异常，异常只落 error 并退出，不产生 scored result；重试须经 error 门禁。正式已存在的 757 条 timeout 全是 220 控制步耗尽，执行异常为 0。

来源：`implementation/pnp_heldout_eval.py:163–164`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 163:164 -->
```python
                outcome = "timeout"  # control horizon exhausted, counts as task failure
                steps = 0
```

来源：`implementation/pnp_heldout_eval.py:205–216`；[源码](pnp-heldout-spatial-1106104308/implementation/pnp_heldout_eval.py)。

<!-- excerpt pnp_heldout_eval.py 205:216 -->
```python
                result = dict(base, event="result", success=success, outcome=outcome, steps=steps,
                              video=str(video) if video else None, time_unix=time.time())
                durable_write(str(prefix) + "_result.json", result)
                events.append(result)
                finished[ep] = result
                print(f"HELDOUT_RESULT {identity} outcome={outcome} steps={steps} attempt={attempt}", flush=True)
            except Exception as exc:
                error = dict(base, event="error", phase=phase, error_type=type(exc).__name__,
                             message=str(exc), traceback=traceback.format_exc(), time_unix=time.time())
                durable_write(str(prefix) + "_error.json", error)
                logging.exception("ERROR_GATE trial=%s phase=%s", identity, phase)
                raise
```

## 5. 正式汇总从原始事件重算

汇总入口只读 trials/；从冻结清单构造 2000 个预期身份，拒绝 preflight、协议不符、非清单身份及重复计分。result 必须有合法 outcome/步数，缺失会阻断完整汇总；完整函数还检查 start/terminal 配对、attempt 连续性，并单列执行异常、未完成和额外尝试。每任务成功率分母来自有效记录，Δ 为相对 G0 的百分点差；四组总体由 40 行结果按 successes/trials 求和。

来源：`implementation/summarize_pnp_heldout.py:17–31`；[源码](pnp-heldout-spatial-1106104308/implementation/summarize_pnp_heldout.py)。

<!-- excerpt summarize_pnp_heldout.py 17:31 -->
```python
    expected = {}
    for group in protocol["group_order"]:
        for task in manifest["tasks"]:
            for state in task["states"]:
                key = f"{group}:libero_spatial:{task['task_id']}:{state['state_hash']}"
                assert key not in expected
                expected[key] = (group, task["task_id"], state["trial_index"])
    assert len(expected) == 2000
    results, starts, terminals, errors = {}, {}, {}, []
    for path in sorted(pathlib.Path(root).glob("*/task*/*.json")):
        event = json.loads(path.read_text())
        assert event["stage"] == "formal", "preflight must never enter formal scores"
        identity = event["trial_id"]
        assert identity in expected and event["protocol_sha256"] == digest, str(path)
        assert (event["group"], event["task_id"], event["trial_index"]) == expected[identity]
```

来源：`implementation/summarize_pnp_heldout.py:43–50`；[源码](pnp-heldout-spatial-1106104308/implementation/summarize_pnp_heldout.py)。

<!-- excerpt summarize_pnp_heldout.py 43:50 -->
```python
            assert identity not in results, f"duplicate scored identity: {identity}"
            assert isinstance(event["success"], bool)
            assert event["outcome"] in ("success", "timeout", "policy_failure")
            assert event["success"] == (event["outcome"] == "success")
            assert 0 < event["steps"] <= 220
            if event["outcome"] == "timeout":
                assert event["steps"] == 220
            results[identity] = event
```

来源：`implementation/summarize_pnp_heldout.py:59–72`；[源码](pnp-heldout-spatial-1106104308/implementation/summarize_pnp_heldout.py)。

<!-- excerpt summarize_pnp_heldout.py 59:72 -->
```python
    missing = sorted(set(expected) - set(results))
    if not partial:
        assert not missing, f"missing {len(missing)} valid trials"
    rows = []
    for task_id in range(10):
        base = [e for e in results.values() if e["group"] == "G0" and e["task_id"] == task_id]
        base_rate = sum(e["success"] for e in base) / len(base) if base else None
        for group in protocol["group_order"]:
            records = [e for e in results.values() if e["group"] == group and e["task_id"] == task_id]
            success = sum(e["success"] for e in records)
            rate = success / len(records) if records else None
            rows.append({"group": group, "task_id": task_id, "successes": success,
                         "trials": len(records), "success_rate": rate,
                         "delta_vs_g0_pp": 100 * (rate - base_rate) if rate is not None and base_rate is not None else None})
```

## 冻结 SHA256 与本次验证

以下评测代码哈希来自 protocol.json；本次逐一比对 implementation/ 与 scripts/remote/，7/7 一致。权重不在本地复制；远端最终全资产复验由 final-protocol-verify.log 的 `PROTOCOL_VERIFIED` 留证，本地只独立核对冻结资产条目及来源快照，不声称重新读取远端权重。

| 文件 | SHA256 |
|---|---|
| eval_libero_task.py | `b580728e4c2af02d1ebb85e05c1a8e3777b4be38c5ce6443ca62f6d46e9a38fc` |
| pnp_heldout_eval.py | `e8d6e54b4db6e0b38b6e8fd1be8a80a690179de6ca348220f8bcc6f513db151b` |
| build_pnp_heldout.py | `fcbcc631ae9532e2b893c74d6418cc9eb3e8fee77d473fce9e4ef80b26911f11` |
| serve_pnp_heldout.py | `2046dfd9486d451e46ec375533d172234f9762603c9c3e30e048f0d76a5e3252` |
| freeze_pnp_heldout.py | `2eac3558c4a577bce290252d3fde034b48e18ba34869243728712257c9950c0d` |
| summarize_pnp_heldout.py | `48dc78e9c97da4b357114b8114cc3ea9691dbd18ac7f16cd02cc4166dcf2cfc1` |
| 103-pnp-heldout-spatial.sh | `80fb9f09bb2fb81b578a6e3a1ede043a009acca9a4ec77c89c47c0ab9da81991` |

完整 implementation.diff SHA256：`d7f776fedd9acc9c01a8de28b7368298014b3f95cadc9c3a00f0ebd34659470e`。

本次本地只读验证：4239 个回拷文件与远端清单逐一核对 SHA256；29 个来源快照校验通过；独立重算 2000 条与已存两份 audit 一致。全部摘录逐段验证为 Diff 及对应源码原文。未运行策略/仿真、未重训或重评。
