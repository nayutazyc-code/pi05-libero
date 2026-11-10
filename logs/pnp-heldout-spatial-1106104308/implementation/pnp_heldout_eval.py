"""Manifest-only evaluation with durable trial attempts and strict resume checks."""
from __future__ import annotations

import collections
import fcntl
import hashlib
import json
import logging
import os
import pathlib
import signal
import time
import traceback

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _ws

from build_pnp_heldout import file_hash, state_hash
from eval_libero_task import LIBERO_DUMMY_ACTION, proprio8


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
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_events(root):
    return [json.loads(p.read_text()) for p in sorted(pathlib.Path(root).glob("*.json"))]


def trial_seed(task_id, identity, step):
    text = f"7:{task_id}:{identity}:{step}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def trial_deadline(signum, frame):
    raise TimeoutError("trial execution exceeded fixed 600s wall deadline (unscored execution error)")


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


def evaluate(args):
    if not args.protocol_file or not args.results_path:
        raise ValueError("held-out evaluation requires protocol-file and results-path")
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
    assert 0 <= args.preflight_trials <= 3
    stage = "preflight" if args.preflight_trials else "formal"
    if args.preflight_trials:
        assert pathlib.Path(args.results_path).name == "preflight-trials"
    task_entry = manifest["tasks"][args.task_id]
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    assert task.language == task_entry["language"]
    bddl = suite.get_task_bddl_file_path(args.task_id)
    assert file_hash(bddl) == task_entry["bddl_sha256"]
    root = pathlib.Path(args.results_path) / args.tag / f"task{args.task_id}"
    root.mkdir(parents=True, exist_ok=True)
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
    except BaseException:
        lock.close()
        raise
    env = None
    client = None
    signal.signal(signal.SIGALRM, trial_deadline)
    try:
        if args.preflight_trials and args.task_id == 2:
            selected = [task_entry["states"][i] for i in (0, 19, 49)[:args.preflight_trials]]
        else:
            selected = task_entry["states"][:args.preflight_trials] if args.preflight_trials else task_entry["states"]
        for entry in selected:
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
            phase = "simulation_setup"
            signal.alarm(600)
            try:
                if env is None:
                    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
                # Reset episode/controller counters before restoring the same frozen state.
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
                phase = "service"
                if client is None:
                    client = _ws.WebsocketClientPolicy(args.host, args.port)
                    metadata = client.get_server_metadata()
                    assert metadata["heldout_protocol_sha256"] == protocol_hash
                    assert metadata["heldout_group"] == args.tag
                actions = collections.deque()
                frames = []
                success = False
                outcome = "timeout"  # control horizon exhausted, counts as task failure
                steps = 0
                while steps < args.max_steps:
                    phase = "service"
                    if not actions:
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        def resize(x):
                            return image_tools.convert_to_uint8(image_tools.resize_with_pad(x, args.resize_size, args.resize_size))
                        chunk = np.asarray(client.infer({
                            "observation/image": resize(img), "observation/wrist_image": resize(wrist),
                            "observation/state": proprio8(obs), "prompt": str(task.language),
                            "heldout_rng_seed": trial_seed(args.task_id, entry["state_hash"], steps),
                        })["actions"])
                        if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                            raise RuntimeError(f"invalid policy action chunk {chunk.shape}")
                        actions.extend(chunk[:args.replan_steps])
                    phase = "simulation"
                    frames.append(np.asarray(obs["agentview_image"]).copy())
                    obs, _, done, _ = env.step(np.asarray(actions.popleft(), dtype=np.float32).tolist())
                    steps += 1
                    if env.check_success():
                        success, outcome = True, "success"
                        break
                    if done:
                        outcome = "policy_failure"
                        break
                video = None
                # Representative first N successes and failures per group/task.
                previous_count = sum(e["success"] == success for e in finished.values())
                if previous_count < args.save_video_n:
                    phase = "video_io"
                    video = videos / f"{args.tag}_task{args.task_id}_trial{ep:02d}_{outcome}.mp4"
                    if video.exists():
                        video = videos / f"{args.tag}_task{args.task_id}_trial{ep:02d}_attempt{attempt}_{outcome}.mp4"
                    imageio.mimwrite(str(video), frames[::2], fps=10)
                    # Decode one frame; an unreadable representative is an execution error.
                    reader = imageio.get_reader(str(video))
                    try:
                        assert reader.get_data(0).size > 0
                    finally:
                        reader.close()
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
            finally:
                signal.alarm(0)
        print(f"HELDOUT_TASK_DONE stage={stage} group={args.tag} task={args.task_id} successes={sum(e['success'] for e in finished.values())}/{len(selected)}", flush=True)
    finally:
        if env is not None:
            env.close()
        lock.close()
