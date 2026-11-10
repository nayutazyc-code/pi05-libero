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
