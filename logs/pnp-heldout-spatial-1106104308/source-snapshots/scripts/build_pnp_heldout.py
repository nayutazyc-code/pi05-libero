#!/usr/bin/env python3
"""Freeze policy-independent spatial initial states before any scored rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import signal

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


RULES = {
    "suite": "libero_spatial", "seed": 7, "trials_per_task": 50,
    "excluded_task": 2, "excluded_indices": list(range(31)),
    "scope": "held-out only against this PnP policy-expert collection",
    "selection": "existing states in index order, then native scene reset() stream",
    "generation_seed": 7, "generation_candidate_limit_per_task": 1000,
    "generation": "one seed per task, never reseed the candidate stream; no pose perturbation",
    "identity": "SHA256 little-endian float64 state[1:] rounded to 12 decimals; omit simulator time",
    "validation": {
        "wait_steps": 10, "dummy_action": [0.0] * 6 + [-1.0],
        "finite_state_and_observations": True, "success_before_or_during_wait": False,
        "last_step_max_qpos_delta": 0.02, "final_max_abs_qvel": 5.0,
        "reset_deadline_seconds": 120,
    },
    "camera_names": ["agentview", "robot0_eye_in_hand"], "camera_resolution": 256,
}


def file_hash(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def state_hash(state):
    # Time alone must not make a repeated physical state appear held-out.
    state = np.round(np.asarray(state, dtype="<f8")[1:], 12)
    state[state == 0] = 0  # canonicalize negative zero
    return hashlib.sha256(state.tobytes()).hexdigest()


def save_exclusive(path, value):
    with pathlib.Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def deadline(signum, frame):
    raise TimeoutError("native scene reset exceeded fixed 120s deadline")


def validate(env, state):
    if not np.isfinite(state).all():
        return {"valid": False, "reason": "nonfinite_initial_state"}
    obs = env.set_init_state(state)
    if env.check_success():
        return {"valid": False, "reason": "already_successful"}
    delta = 0.0
    for _ in range(10):
        previous = env.sim.data.qpos.copy()
        obs, _, done, _ = env.step(RULES["validation"]["dummy_action"])
        delta = float(np.max(np.abs(env.sim.data.qpos - previous)))
        if done or env.check_success():
            return {"valid": False, "reason": "success_or_done_during_wait"}
    velocity = float(np.max(np.abs(env.sim.data.qvel)))
    finite = all(np.isfinite(obs[k]).all() for k in (
        "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos",
    )) and np.isfinite(env.get_sim_state()).all()
    valid = bool(finite and delta <= 0.02 and velocity <= 5.0)
    return {"valid": valid, "reason": "ok" if valid else "unstable_or_nonfinite",
            "last_qpos_delta": delta, "max_abs_qvel": velocity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--collection-summary", required=True)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Written exclusively before generation or simulation validation, never overwritten.
    save_exclusive(out / "selection_rules.json", RULES)
    summary = json.loads(pathlib.Path(args.collection_summary).read_text())
    assert summary["attempts"] == 31 and len(summary["episodes"]) == 31
    assert [e["attempt"] for e in summary["episodes"]] == list(range(1, 32))
    assert all(e["task_id"] == 2 and e["suite"] == "libero_spatial" for e in summary["episodes"])
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    excluded = {state_hash(x) for x in suite.get_task_init_states(2)[:31]}
    assert len(excluded) == 31
    manifest = {"schema": 1, "rules_sha256": file_hash(out / "selection_rules.json"),
                "collection_summary_sha256": file_hash(args.collection_summary),
                "collector_sha256": file_hash(pathlib.Path(__file__).with_name("collect_pnp_policy_expert.py")),
                "excluded_state_hashes": sorted(excluded), "tasks": []}
    signal.signal(signal.SIGALRM, deadline)
    for task_id in range(10):
        task = suite.get_task(task_id)
        source = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        bddl = pathlib.Path(suite.get_task_bddl_file_path(task_id))
        np.random.seed(7)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(7)
        chosen, audit, seen = [], [], set()
        try:
            existing = suite.get_task_init_states(task_id)
            for index, candidate in enumerate(existing):
                candidate = np.asarray(candidate, dtype=np.float64).copy()
                identity = state_hash(candidate)
                entry = {"source": "existing", "source_index": index, "state_hash": identity}
                if identity in excluded or identity in seen:
                    audit.append(dict(entry, valid=False, reason="collection_overlap_or_duplicate"))
                    continue
                metrics = validate(env, candidate)
                audit.append(dict(entry, **metrics))
                if metrics["valid"]:
                    chosen.append(dict(entry, state=candidate.tolist(), validation=metrics))
                    seen.add(identity)
                if len(chosen) == 50:
                    break
            # Reset RNG only once, before the fixed candidate stream starts.
            np.random.seed(7)
            env.seed(7)
            for index in range(1000):
                if len(chosen) == 50:
                    break
                signal.alarm(120)
                try:
                    env.reset()
                finally:
                    signal.alarm(0)
                candidate = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
                identity = state_hash(candidate)
                entry = {"source": "native_reset", "source_index": index, "state_hash": identity}
                if identity in excluded or identity in seen:
                    audit.append(dict(entry, valid=False, reason="collection_overlap_or_duplicate"))
                    continue
                metrics = validate(env, candidate)
                audit.append(dict(entry, **metrics))
                if metrics["valid"]:
                    chosen.append(dict(entry, state=candidate.tolist(), validation=metrics))
                    seen.add(identity)
            save_exclusive(out / f"task{task_id}_selection_audit.json", audit)
            if len(chosen) != 50:
                raise RuntimeError(f"ERROR_GATE task{task_id}: only {len(chosen)} legal held-out states")
            for order, entry in enumerate(chosen):
                entry["trial_index"] = order
                entry["seed"] = 7
            manifest["tasks"].append({"task_id": task_id, "language": task.language,
                "init_file": str(source), "init_file_sha256": file_hash(source),
                "bddl_sha256": file_hash(bddl), "states": chosen})
            print(f"HELDOUT_TASK {task_id} existing={sum(x['source'] == 'existing' for x in chosen)} generated={sum(x['source'] == 'native_reset' for x in chosen)}", flush=True)
        finally:
            env.close()
    save_exclusive(out / "manifest.json", manifest)
    save_exclusive(out / "manifest.sha256.json", {"sha256": file_hash(out / "manifest.json")})
    print("HELDOUT_FROZEN", file_hash(out / "manifest.json"), flush=True)


if __name__ == "__main__":
    main()
