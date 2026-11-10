# A-PnP 核心代码入包（acceptance 补件）
生成：2026-11-06 10:02 CST
说明：slim-context 默认只带最近 commit diff；下列为审查所需实现全文/关键脚本，来自工作区当前内容（与训练机 SHA256 已在 adapter 测试中核对）。

## scripts/remote/pnp_grad_accum.py
```
# lines: 21
"""Mean microbatch gradients, before clipping and a single optimizer update."""
import jax


def accumulate_gradients(loss_and_grad, batch, rng, microbatch_size):
    size = batch[1].shape[0]
    if microbatch_size < 1 or size % microbatch_size:
        raise ValueError("Effective batch must be divisible by microbatch size")
    count = size // microbatch_size

    def compute(index):
        microbatch = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index * microbatch_size, microbatch_size), batch
        )
        return loss_and_grad(jax.random.fold_in(rng, index), *microbatch)

    def add(index, total):
        return jax.tree.map(lambda a, b: a + b, total, compute(index))

    total = jax.lax.fori_loop(1, count, add, compute(0))
    return jax.tree.map(lambda x: x / count, total)
```

## scripts/remote/train_pnp_grad_accum.py
```
# lines: 81
"""PnP-only adapter for upstream scripts/train.py; batch_size stays effective."""
import logging
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import wandb

import train as upstream
from pnp_grad_accum import accumulate_gradients


MICROBATCH_SIZE = int(os.environ.get("PNP_MICROBATCH_SIZE", "16"))
_init_wandb = upstream.init_wandb


def init_wandb(config, **kwargs):
    if MICROBATCH_SIZE < 1 or config.batch_size % MICROBATCH_SIZE:
        raise ValueError("Effective batch must be divisible by microbatch size")
    _init_wandb(config, **kwargs)
    metadata = {
        "effective_batch_size": config.batch_size,
        "microbatch_size": MICROBATCH_SIZE,
        "gradient_accumulation_steps": config.batch_size // MICROBATCH_SIZE,
        "gradient_reduction": "mean, then clip once and AdamW once",
        "step_unit": "optimizer updates",
    }
    wandb.config.update(metadata)
    logging.info("PNP_GRAD_ACCUM %s", metadata)


def train_step(config, rng, state, batch):
    def loss_and_grad(micro_rng, observation, actions):
        # Merge inside the loop so NNX mutable graph objects stay within its trace.
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model):
            return jnp.mean(model.compute_loss(micro_rng, observation, actions, train=True))

        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(model)

    loss, grads = accumulate_gradients(
        loss_and_grad, batch, jax.random.fold_in(rng, state.step), MICROBATCH_SIZE
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, upstream.optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = upstream.dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = upstream.dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(upstream.nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return new_state, {
        "loss": loss,
        "grad_norm": upstream.optax.global_norm(grads),
        "param_norm": upstream.optax.global_norm(kernel_params),
    }


if __name__ == "__main__":
    upstream.init_wandb = init_wandb
    upstream.train_step = train_step
    upstream.main(upstream._config.cli())
```

## scripts/remote/test_pnp_grad_accum.py
```
# lines: 73
"""Check accumulated NNX gradients and AdamW update against a full batch on CPU."""
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pnp_grad_accum import accumulate_gradients
from train_pnp_grad_accum import train_step
from openpi.training import utils


class TinyModel(nnx.Module):
    def __init__(self):
        self.kernel = nnx.Param(jnp.array([[0.2, -0.1], [0.3, 0.4]]))

    def train(self):
        pass

    def compute_loss(self, rng, observation, actions, train=True):
        return (observation["x"] @ self.kernel.value - actions) ** 2


def assert_close(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for a, b in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def main():
    model = TinyModel()
    graph, params = nnx.split(model)
    batch = ({"x": jnp.arange(64, dtype=jnp.float32).reshape(32, 2) / 64, "optional": None},
             jnp.ones((32, 2)))
    rng = jax.random.key(42)

    def loss_and_grad(key, observation, actions):
        local = nnx.merge(graph, params)
        return nnx.value_and_grad(lambda m: jnp.mean(m.compute_loss(key, observation, actions)))(local)

    expected_loss, expected_grads = loss_and_grad(rng, *batch)
    for micro in (8, 16, 32):
        actual = jax.jit(lambda b: accumulate_gradients(loss_and_grad, b, rng, micro))(batch)
        assert_close(actual, (expected_loss, expected_grads))

    # Compare the actual training adapter, including clipping, optimizer count and EMA.
    tx = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3))
    state = utils.TrainState(step=jnp.array(0), params=params, model_def=graph, tx=tx,
                             opt_state=tx.init(params), ema_decay=0.9, ema_params=params)
    actual_state, info = jax.jit(lambda s, b: train_step(types.SimpleNamespace(trainable_filter=nnx.Param), rng, s, b))(state, batch)
    updates, expected_opt = tx.update(expected_grads, state.opt_state, params)
    expected_params = optax.apply_updates(params, updates)
    assert_close(actual_state.params, expected_params)
    assert_close(actual_state.opt_state, expected_opt)
    assert_close(actual_state.ema_params, jax.tree.map(lambda a, b: .9 * a + .1 * b, params, expected_params))
    assert_close(info["loss"], expected_loss)
    assert_close(info["grad_norm"], optax.global_norm(expected_grads))
    assert int(actual_state.step) == 1
    try:
        accumulate_gradients(loss_and_grad, batch, rng, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid microbatch accepted")
    print("PASS: micro8/16/32 = full batch; adapter clips/updates/EMA once; invalid size rejected")


if __name__ == "__main__":
    main()
```

## scripts/remote/101-pnp-finetune-g0-accum.sh
```
# lines: 20
#!/usr/bin/env bash
# G0 effective batch32 = micro8 x 4; preserve previous OOM runs and existing G1.
set -euo pipefail
source /root/autodl-tmp/env.sh
source /root/.config/pi05/wandb.env
cd /root/autodl-tmp/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
export PNP_MICROBATCH_SIZE=8
LOG=/root/autodl-tmp/logs/pnp-g0-accum-mb8-train.log
echo "TRAIN_G0_ACCUM_START $(date -Iseconds) effective_batch=32 microbatch=8 accum=4 optimizer_updates=2500 wandb=$WANDB_MODE" | tee -a "$LOG"
set +e
.venv/bin/python scripts/train_pnp_grad_accum.py pi05_libero_lora_pnp \
  --exp-name pnp_g0_accum_bs32_mb8 \
  --batch-size 32 --num-train-steps 2500 \
  --log-interval 50 --save-interval 2500 --keep-period 2500 >> "$LOG" 2>&1
rc=$?
set -e
echo "TRAIN_G0_ACCUM_EXIT $rc $(date -Iseconds)" | tee -a "$LOG"
exit "$rc"
```

## scripts/remote/102-pnp-after-g0-mb8.sh
```
# lines: 21
#!/usr/bin/env bash
# Continue the authorized mb8 run without restarting training or A/G1 evaluation.
set -euo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
TRAIN_LOG=$W/logs/pnp-g0-accum-mb8-train.log
CKPT=$W/openpi/checkpoints/pi05_libero_lora_pnp/pnp_g0_accum_bs32_mb8/2499

until grep -q '^TRAIN_G0_ACCUM_EXIT ' "$TRAIN_LOG"; do
  sleep 15
done

cd "$W/openpi"
.venv/bin/python "$W/scripts/100-pnp-export-loss.py"
if grep -q '^TRAIN_G0_ACCUM_EXIT 0 ' "$TRAIN_LOG" && [[ -d "$CKPT/params" ]]; then
  PNP_EVAL_GROUPS="G0 G2" bash "$W/scripts/99-pnp-eval-g0-g1-g2.sh"
else
  echo "G0_MB8_FAILED: no G0/G2 evaluation; preserve A/G1 results"
  PNP_EVAL_GROUPS=" " bash "$W/scripts/99-pnp-eval-g0-g1-g2.sh"
fi
echo "PNP_MB8_FOLLOWUP_DONE $(date -Iseconds)"
```

## scripts/remote/100-pnp-export-loss.py
```
# lines: 82
#!/usr/bin/env python3
"""Export PnP offline W&B loss without syncing or changing training artifacts."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re

import matplotlib
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = pathlib.Path("/root/autodl-tmp/openpi")
    logs = root.parent / "logs"
    out = logs / "plots"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for tag, exp, logname in [
        ("g0_native", "pnp_g0_bs32", "pnp-g0-train.log"),
        ("g0_mb16_failed", "pnp_g0_accum_bs32_mb16", "pnp-g0-accum-train.log"),
        ("g0", "pnp_g0_accum_bs32_mb8", "pnp-g0-accum-mb8-train.log"),
        ("g1", "pnp_g1_bs16", "pnp-g1-train.log"),
    ]:
        run_id = (root / "checkpoints/pi05_libero_lora_pnp" / exp / "wandb_id.txt").read_text().strip()
        sources = sorted({p.resolve() for p in (root / "wandb").glob(f"*/run-{run_id}.wandb")})
        if len(sources) != 1:
            raise ValueError(f"Expected one W&B source for {tag}: {sources}")
        store = DataStore()
        store.open_for_scan(str(sources[0]))
        values = {}
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("history"):
                row = {
                    item.key or "/".join(item.nested_key): json.loads(item.value_json)
                    for item in record.history.item
                }
                if "loss" in row:
                    step, loss = int(row["_step"]), float(row["loss"])
                    assert math.isfinite(loss)
                    values[step] = loss
        rows = sorted(values.items())
        logged = {}
        for match in re.finditer(r"Step\s+(\d+):[^\n]*?\bloss=([0-9.eE+-]+)", (logs / logname).read_text(errors="replace")):
            logged[int(match[1])] = float(match[2])
        assert set(values) == set(logged), (tag, "W&B/log step mismatch")
        assert all(abs(loss - logged[step]) <= 5.1e-5 for step, loss in rows)
        with (out / f"pnp_{tag}_wandb_loss.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "loss"])
            writer.writerows(rows)
        if rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(*zip(*rows), linewidth=1.2)
            ax.set(xlabel="Training step", ylabel="Logged mean loss", title=f"PnP {tag.upper()} offline W&B loss (interval=50)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"pnp_{tag}_wandb_loss_curve.png", dpi=150)
            plt.close(fig)
        provenance[tag] = {
            "run_id": run_id, "source": str(sources[0]), "points": len(rows),
            "first": rows[0] if rows else None, "last": rows[-1] if rows else None,
            "log_crosscheck": "passed; rounded log tolerance=0.000051",
            "note": "Logged interval means, every 50 optimizer updates; last 49 updates not logged. Native G0 and micro16 x2: OOM before first loss; G0 replacement: micro8 x4 effective batch32.",
        }
    (out / "pnp-loss-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
```

## scripts/remote/99-pnp-eval-g0-g1-g2.sh
```
# lines: 169
#!/usr/bin/env bash
# Eval A-29999 reference + G0/G1 ckpts on libero_spatial task_id=2 (50 trials).
# G2 = G0 weights with replan_steps=10. G0/G1/A use replan=5.
set -euo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
L=$W/logs
V=$L/videos_pnp_eval
CFG_BASE=pi05_libero_lora
CFG_PNP=pi05_libero_lora_pnp
A_CKPT=$W/openpi/checkpoints/pi05_libero_lora/phase1_lora_baseline/29999
G0_CKPT=$W/openpi/checkpoints/$CFG_PNP/pnp_g0_accum_bs32_mb8
G1_CKPT=$W/openpi/checkpoints/$CFG_PNP/pnp_g1_bs16
TRIALS=${PNP_EVAL_TRIALS:-50}
GROUPS_TO_EVAL=${PNP_EVAL_GROUPS:-"A29999 G0 G1 G2"}
mkdir -p "$L" "$V"
cd "$W/openpi"

pick_final() {
  local root=$1
  if [[ ! -d "$root" ]]; then echo ""; return; fi
  if [[ -d "$root/params" ]]; then
    echo "$root"
    return
  fi
  local step
  step=$(ls -1 "$root" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
  if [[ -n "$step" && -d "$root/$step/params" ]]; then
    echo "$root/$step"
  fi
}

run_one() {
  local tag=$1 cfg=$2 ckpt_root=$3 replan=$4
  local ckpt
  ckpt=$(pick_final "$ckpt_root")
  if [[ -z "$ckpt" ]]; then
    echo "SKIP_EVAL $tag missing ckpt under $ckpt_root" | tee -a "$L/pnp-eval.log"
    return 1
  fi
  echo "EVAL_START tag=$tag cfg=$cfg ckpt=$ckpt replan=$replan trials=$TRIALS $(date -Iseconds)" | tee -a "$L/pnp-eval.log"
  .venv/bin/python scripts/serve_policy.py --env LIBERO policy:checkpoint \
    --policy.config="$cfg" --policy.dir="$ckpt" > "$L/pnp-eval-server-${tag}.log" 2>&1 &
  local SP=$!
  local ready=0
  for i in $(seq 1 360); do
    if ! kill -0 "$SP" 2>/dev/null; then break; fi
    if grep -qE "server listening|Uvicorn running|Creating server" "$L/pnp-eval-server-${tag}.log" 2>/dev/null \
      && (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
      ready=1; break
    fi
    sleep 5
  done
  if [[ $ready -ne 1 ]]; then
    echo "EVAL_SERVER_FAIL $tag" | tee -a "$L/pnp-eval.log"
    kill $SP 2>/dev/null || true
    return 1
  fi
  echo "server ready tag=$tag polls=$i pid=$SP" | tee -a "$L/pnp-eval.log"
  # shellcheck disable=SC1091
  source examples/libero/.venv/bin/activate
  export PYTHONPATH=$PWD/third_party/libero
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  set +e
  python "$W/scripts/eval_libero_task.py" \
    --tag "$tag" \
    --task-suite-name libero_spatial \
    --task-id 2 \
    --num-trials "$TRIALS" \
    --replan-steps "$replan" \
    --seed 7 \
    --checkpoint-path "$ckpt" --policy-config "$cfg" \
    --video-out-path "$V" \
    2>&1 | tee -a "$L/pnp-eval-${tag}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  deactivate 2>/dev/null || true
  # restore openpi venv for next serve
  cd "$W/openpi"
  kill $SP 2>/dev/null || true
  wait $SP 2>/dev/null || true
  echo "EVAL_EXIT tag=$tag rc=$rc $(date -Iseconds)" | tee -a "$L/pnp-eval.log"
  return $rc
}

echo "PNP_EVAL_BATCH_START $(date -Iseconds) groups=$GROUPS_TO_EVAL" | tee -a "$L/pnp-eval.log"
eval_rc=0

for tag in $GROUPS_TO_EVAL; do
  case "$tag" in
    A29999) run_one "$tag" "$CFG_BASE" "$A_CKPT" 5 || eval_rc=1 ;;
    G0) run_one "$tag" "$CFG_PNP" "$G0_CKPT" 5 || eval_rc=1 ;;
    G1) run_one "$tag" "$CFG_PNP" "$G1_CKPT" 5 || eval_rc=1 ;;
    G2) run_one "$tag" "$CFG_PNP" "$G0_CKPT" 10 || eval_rc=1 ;;
    *) echo "Unknown eval group: $tag" >&2; exit 2 ;;
  esac
done

# Write results table
.venv/bin/python - <<'PY'
import json, pathlib, datetime
L = pathlib.Path("/root/autodl-tmp/logs")
V = L / "videos_pnp_eval"
rows = []
for tag, replan, note in [
    ("A29999", 5, "Exp A untouched reference"),
    ("G0", 5, "FT grad_accum effective batch32 = micro8 x4; action_horizon=10"),
    ("G1", 5, "FT native batch16; same 2500 optimizer updates; half the G0 sample exposure"),
    ("G2", 10, "G0 weights, eval replan=10 only"),
]:
    matches = sorted(V.glob(f"{tag}_task2_replan{replan}_summary.json"))
    if not matches:
        rows.append((tag, replan, note, None, None, None))
        continue
    s = json.loads(matches[-1].read_text())
    rows.append((tag, replan, note, s["successes"], s["trials"], s["success_rate"]))

md = []
md.append("# PnP G0/G1/G2 results (Asia/Shanghai)")
md.append("")
md.append(f"- generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
md.append("- task: libero_spatial task_id=2 (bowl from table center → plate)")
md.append("- FT data: local/pnp_policy_spatial_center (cleaned A-policy demos)")
md.append("- start weights: Exp A pi05_libero_lora/phase1_lora_baseline/29999")
md.append("- protocol: seed=7, task2, max_steps=220, wait=10, action_horizon=10; replan and actual trial counts below")
md.append("- G0 native batch32 originally OOM before step0; micro16 x2 also OOM; replacement uses micro8 x4, averaged gradients, one clipping/AdamW update per effective batch. W&B offline enabled. Budget: 2500 optimizer updates per train group; a completed G0 would see 80000 samples, G1 saw 40000 (repeated sampling).")
md.append("- state: xyz + quaternion(4) + first gripper joint(1), matching collection/training; differs from official xyz + axis-angle(3) + both gripper joints(2). Error gate approved retaining this contract.")
md.append("- Evaluation uses init_states[0:50]; collection used the first 31 states (30 successful demos). This includes training initial states and is not held-out generalization or the official 10-task benchmark.")
md.append("")
md.append("## Official baseline config (G0 train/eval)")
md.append("| item | value | source |")
md.append("|---|---|---|")
md.append("| batch_size | 32 | openpi TrainConfig pi05_libero_lora |")
md.append("| action_horizon | 10 | Pi0Config(pi05=True, action_horizon=10) |")
md.append("| replan_steps | 5 | examples/libero/main.py Args.replan_steps |")
md.append("| async RTC | not enabled in this tree | — |")
md.append("")
md.append("## Results")
md.append("| group | replan | successes | trials | success_rate | diff vs G0 | note |")
md.append("|---|---:|---:|---:|---:|---|---|")
g0_rate = None
for tag, replan, note, suc, trials, rate in rows:
    if tag == "G0" and rate is not None:
        g0_rate = rate
for tag, replan, note, suc, trials, rate in rows:
    if rate is None:
        reason = "evaluation missing/failed (not a zero success score)"
        md.append(f"| {tag} | {replan} | — | — | — | — | {note}; {reason} |")
    else:
        diff = ""
        if g0_rate is not None and tag != "G0":
            diff = f"{(rate - g0_rate)*100:+.1f}pp"
        elif tag == "G0":
            diff = "—"
        md.append(f"| {tag} | {replan} | {suc} | {trials} | {rate*100:.1f}% | {diff} | {note} |")
md.append("")
md.append("## Artifacts")
md.append("- train logs: `/root/autodl-tmp/logs/pnp-g0-train.log` (native OOM), `pnp-g0-accum-train.log` (micro16 OOM), `pnp-g0-accum-mb8-train.log`, `pnp-g1-train.log`")
md.append("- loss CSV/PNG: `/root/autodl-tmp/logs/plots/pnp_g{0,1}_wandb_loss.csv` / `pnp_g{0,1}_wandb_loss_curve.png`")
md.append("- eval summaries: `/root/autodl-tmp/logs/videos_pnp_eval/*_summary.json`")
md.append("- G0 vs G1 compares effective batch at equal optimizer updates, not equal sample exposure; microbatch RNG streams also differ. G0 uses accumulation rather than native batch32. A vs FT also changes data and normalization. G0 vs G2 changes only synchronous replan, not asynchronous RTC inpainting.")
out = L / "pnp-g0-g1-g2-result.md"
out.write_text("\n".join(md) + "\n")
print("RESULT_MD", out)
print(out.read_text())
PY

echo "PNP_EVAL_BATCH_DONE rc=$eval_rc $(date -Iseconds)" | tee -a "$L/pnp-eval.log"
exit "$eval_rc"
```


## git history (bounded)
commit efb94214b25b5765a2071aaaa343c1c60f448fca
Author: ds-gpt-gate <workflow-author>
Date:   Fri Nov 6 09:57:31 2026 +0800

    ds-gpt-gate: accept pi05-a-pnp-1106001525

 scripts/remote/101-pnp-finetune-g0-accum.sh | 20 ++++++++
 scripts/remote/pnp_grad_accum.py            | 21 +++++++++
 scripts/remote/test_pnp_grad_accum.py       | 73 +++++++++++++++++++++++++++++
 3 files changed, 114 insertions(+)
