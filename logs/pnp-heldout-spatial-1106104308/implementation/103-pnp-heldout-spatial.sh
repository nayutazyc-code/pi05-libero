#!/usr/bin/env bash
# Frozen 4 groups × 10 spatial tasks × 50 trials; never launch training.
set -euo pipefail
source /root/autodl-tmp/env.sh
W=/root/autodl-tmp
RUN=$W/logs/pnp-heldout-spatial-1106104308
PROTOCOL=$RUN/protocol.json
PORT=8017
MODE=${1:-formal}
case "$MODE" in preflight|formal) ;; *) echo "unknown mode: $MODE"; exit 2 ;; esac
cd "$W/openpi"
export PYTHONPATH=$PWD/third_party/libero
mkdir -p "$RUN/trials" "$RUN/videos"
exec 9>"$RUN/.runner.lock"
flock -n 9
SP=""
cleanup() {
  if [[ -n "$SP" ]]; then
    kill "$SP" 2>/dev/null || true
    wait "$SP" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
trap 'rc=$?; echo "ERROR_GATE rc=$rc mode=$MODE line=$LINENO"; exit "$rc"' ERR
if [[ ! -f "$PROTOCOL" ]]; then
  .venv/bin/python "$W/scripts/freeze_pnp_heldout.py" \
    --manifest "$RUN/heldout/manifest.json" --out "$PROTOCOL"
fi
# Verify actual files on every start/resume, including model shards and normalizers.
.venv/bin/python "$W/scripts/freeze_pnp_heldout.py" --out "$PROTOCOL" --verify
if [[ "$MODE" == formal ]]; then
  .venv/bin/python - "$PROTOCOL" "$RUN/preflight-passed.json" <<'PY'
import hashlib, json, pathlib, sys
digest = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
assert json.loads(pathlib.Path(sys.argv[2]).read_text())["protocol_sha256"] == digest
PY
  TASKS="0 1 2 3 4 5 6 7 8 9"
  RESULTS=$RUN/trials
  VIDEOS=$RUN/videos
  EXTRA_ARGS=()
else
  TASKS="2 4"
  RESULTS=$RUN/preflight-trials
  VIDEOS=$RUN/preflight-videos
  EXTRA_ARGS=(--preflight-trials 2)
fi
for group in A29999 G0 G1 G2; do
  case "$group" in
    A29999) cfg=pi05_libero_lora; ckpt=$W/openpi/checkpoints/$cfg/phase1_lora_baseline/29999; replan=5 ;;
    G0|G2) cfg=pi05_libero_lora_pnp; ckpt=$W/openpi/checkpoints/$cfg/pnp_g0_accum_bs32_mb8/2499; replan=5 ;;
    G1) cfg=pi05_libero_lora_pnp; ckpt=$W/openpi/checkpoints/$cfg/pnp_g1_bs16/2499; replan=5 ;;
  esac
  if [[ "$group" == G2 ]]; then replan=10; fi
  # A dedicated port prevents accidental use of a previous evaluation server.
  if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    echo "ERROR_GATE port $PORT already in use"; exit 1
  fi
  .venv/bin/python "$W/scripts/serve_pnp_heldout.py" --protocol "$PROTOCOL" \
    --group "$group" --port "$PORT" >> "$RUN/server-$group.log" 2>&1 &
  SP=$!
  ready=0
  for ((poll=0; poll<360; poll++)); do
    if ! kill -0 "$SP" 2>/dev/null; then break; fi
    if curl --fail --silent "http://127.0.0.1:$PORT/healthz" >/dev/null; then ready=1; break; fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then echo "ERROR_GATE server $group failed"; exit 1; fi
  for task in $TASKS; do
    echo "EVAL_START mode=$MODE group=$group task=$task $(date -Iseconds)"
    examples/libero/.venv/bin/python "$W/scripts/eval_libero_task.py" \
      --tag "$group" --port "$PORT" --task-suite-name libero_spatial --task-id "$task" \
      --num-trials 50 --replan-steps "$replan" --seed 7 --max-steps 220 --num-steps-wait 10 \
      --checkpoint-path "$ckpt" --policy-config "$cfg" --save-video-n 1 \
      --init-manifest "$RUN/heldout/manifest.json" --protocol-file "$PROTOCOL" \
      --results-path "$RESULTS" --video-out-path "$VIDEOS" "${EXTRA_ARGS[@]}" \
      >> "$RUN/$MODE-eval-$group-task$task.log" 2>&1
  done
  cleanup
  SP=""
done
if [[ "$MODE" == preflight ]]; then
  .venv/bin/python - "$PROTOCOL" "$RUN" <<'PY'
import hashlib, json, pathlib, sys
protocol, run = map(pathlib.Path, sys.argv[1:])
results = [json.loads(p.read_text()) for p in (run / "preflight-trials").glob("*/task*/*_result.json")]
assert len(results) == 16 and all(e["stage"] == "preflight" for e in results)
assert len({e["trial_id"] for e in results}) == 16
marker = {"protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(), "preflight_trials": 16}
path = run / "preflight-passed.json"
if path.exists():
    assert json.loads(path.read_text()) == marker
else:
    with path.open("x") as f:
        json.dump(marker, f, indent=2)
print("PREFLIGHT_DONE (excluded from formal scores)")
PY
  exit 0
fi
.venv/bin/python "$W/scripts/summarize_pnp_heldout.py" \
  --root "$RUN/trials" --protocol "$PROTOCOL" --out "$RUN/summary"
echo "FORMAL_DONE $(date -Iseconds)"
