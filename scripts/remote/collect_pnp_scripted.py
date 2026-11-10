#!/usr/bin/env python3
"""Scripted expert PnP for LIBERO spatial task_id=2.
OSC_POSE actions are normalized [-1,1] mapped to ±0.05 m / ±0.5 rad.
Writes episode_XXXX/{image,wrist_image,state,action}.npy + meta.json
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_ID = 2
TASK_LANG = "pick up the black bowl from table center and place it on the plate"
POS_SCALE = 0.05  # meters at action=1


def make_env(seed: int):
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env, task.language, suite.get_task_init_states(TASK_ID)


def proprio8(obs):
    g = float(np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[0])
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
        np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4),
        np.array([g], dtype=np.float32),
    ])


def pick_bowl_key(obs):
    keys = [k for k in ("akita_black_bowl_1", "akita_black_bowl_2") if f"{k}_pos" in obs]
    if not keys:
        return "akita_black_bowl_1"
    center = np.array([0.0, 0.0], dtype=np.float32)
    best, bestd = keys[0], 1e9
    for k in keys:
        d = np.linalg.norm(np.asarray(obs[f"{k}_pos"], dtype=np.float32)[:2] - center)
        if d < bestd:
            best, bestd = k, d
    return best


def pos_action(eef, target, z=None):
    t = np.asarray(target, dtype=np.float32).copy()
    if z is not None:
        t[2] = z
    delta_m = t - np.asarray(eef, dtype=np.float32)
    # normalize so |1| ~= POS_SCALE meters
    a = np.clip(delta_m / POS_SCALE, -1.0, 1.0)
    return a.astype(np.float32)


def scripted_action(obs, phase, bowl_key, sip):
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    bowl = np.asarray(obs[f"{bowl_key}_pos"], dtype=np.float32)
    plate = np.asarray(obs["plate_1_pos"], dtype=np.float32)
    zero_ori = np.zeros(3, dtype=np.float32)
    open_g, close_g = -1.0, 1.0
    grasp_z = float(bowl[2] + 0.02)  # slightly above body center
    place_z = float(plate[2] + 0.05)

    def pack(pos3, grip):
        return np.concatenate([pos3, zero_ori, [grip]]).astype(np.float32)

    if phase == "hover_bowl":
        a = pos_action(eef, bowl, z=1.05)
        done = np.linalg.norm(eef[:2] - bowl[:2]) < 0.02 and abs(eef[2] - 1.05) < 0.03
        return pack(a, open_g), ("descend_bowl" if done or sip > 80 else phase), (0 if done else sip + 1)

    if phase == "descend_bowl":
        a = pos_action(eef, bowl, z=grasp_z)
        done = eef[2] <= grasp_z + 0.02 or sip > 80
        return pack(a, open_g), ("close" if done else phase), (0 if done else sip + 1)

    if phase == "close":
        a = pos_action(eef, bowl, z=grasp_z) * 0.3
        done = sip > 15
        return pack(a, close_g), ("lift" if done else phase), (0 if done else sip + 1)

    if phase == "lift":
        a = pos_action(eef, bowl, z=1.10)
        done = eef[2] > 1.05 or sip > 60
        return pack(a, close_g), ("hover_plate" if done else phase), (0 if done else sip + 1)

    if phase == "hover_plate":
        a = pos_action(eef, plate, z=1.10)
        done = np.linalg.norm(eef[:2] - plate[:2]) < 0.025 and abs(eef[2] - 1.10) < 0.04
        return pack(a, close_g), ("descend_plate" if done or sip > 100 else phase), (0 if done else sip + 1)

    if phase == "descend_plate":
        a = pos_action(eef, plate, z=place_z)
        done = eef[2] <= place_z + 0.03 or sip > 80
        return pack(a, close_g), ("open" if done else phase), (0 if done else sip + 1)

    if phase == "open":
        a = pos_action(eef, plate, z=place_z) * 0.2
        done = sip > 15
        return pack(a, open_g), ("retreat" if done else phase), (0 if done else sip + 1)

    # retreat
    a = pos_action(eef, plate, z=1.15)
    done = eef[2] > 1.08 or sip > 40
    return pack(a, open_g), ("done" if done else "retreat"), (0 if done else sip + 1)


def rollout(env, init_state, max_steps=500):
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
        if done:
            break
    bowl_key = pick_bowl_key(obs)
    phase, sip = "hover_bowl", 0
    images, wrists, states, actions = [], [], [], []
    success = False
    for t in range(max_steps):
        act, phase, sip = scripted_action(obs, phase, bowl_key, sip)
        images.append(np.asarray(obs["agentview_image"]))
        wrists.append(np.asarray(obs["robot0_eye_in_hand_image"]))
        states.append(proprio8(obs))
        actions.append(act)
        obs, _, done, _ = env.step(act.tolist())
        if env.check_success():
            success = True
            for _ in range(8):
                hold = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
                images.append(np.asarray(obs["agentview_image"]))
                wrists.append(np.asarray(obs["robot0_eye_in_hand_image"]))
                states.append(proprio8(obs))
                actions.append(hold)
                obs, _, done, _ = env.step(hold.tolist())
                if done:
                    break
            break
        if done or phase == "done":
            break
    return {
        "success": bool(success),
        "steps": len(actions),
        "bowl_key": bowl_key,
        "image": np.stack(images, 0) if images else np.zeros((0, 256, 256, 3), np.uint8),
        "wrist_image": np.stack(wrists, 0) if wrists else np.zeros((0, 256, 256, 3), np.uint8),
        "state": np.stack(states, 0).astype(np.float32) if states else np.zeros((0, 8), np.float32),
        "action": np.stack(actions, 0).astype(np.float32) if actions else np.zeros((0, 7), np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env, lang, init_states = make_env(args.seed)
    print(f"COLLECT_START lang={lang!r} n={args.num_episodes} action_space=norm[-1,1]->±{POS_SCALE}m", flush=True)
    summary, n_ok = [], 0
    for i in range(args.num_episodes):
        init = init_states[i % len(init_states)]
        try:
            ep = rollout(env, init, max_steps=args.max_steps)
        except Exception as e:
            print(f"EP {i} ERR {e}", flush=True)
            summary.append({"episode": i, "success": False, "error": str(e)})
            try:
                env.close()
            except Exception:
                pass
            env, lang, init_states = make_env(args.seed + i + 1)
            continue
        ep_dir = out / f"episode_{i:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "image.npy", ep["image"])
        np.save(ep_dir / "wrist_image.npy", ep["wrist_image"])
        np.save(ep_dir / "state.npy", ep["state"])
        np.save(ep_dir / "action.npy", ep["action"])
        meta = {
            "episode": i, "success": ep["success"], "steps": ep["steps"],
            "bowl_key": ep["bowl_key"], "task": TASK_LANG, "task_id": TASK_ID,
            "suite": "libero_spatial", "fps": 10,
            "dataset_ref": "lerobot/libero_spatial_image",
            "action_note": "OSC_POSE normalized [-1,1] -> ±0.05m",
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        summary.append(meta)
        n_ok += int(ep["success"])
        print(f"EP {i} success={ep['success']} steps={ep['steps']} bowl={ep['bowl_key']} ok_so_far={n_ok}/{i+1}", flush=True)
        # reset cleanly for next
        try:
            env.reset()
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            env, lang, init_states = make_env(args.seed + i + 1)
    (out / "summary.json").write_text(json.dumps({"n": args.num_episodes, "successes": n_ok, "episodes": summary}, indent=2))
    print(f"COLLECT_DONE successes={n_ok}/{args.num_episodes} out={out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
