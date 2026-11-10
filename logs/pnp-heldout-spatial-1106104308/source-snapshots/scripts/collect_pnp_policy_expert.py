#!/usr/bin/env python3
"""Collect PnP demos with experiment-A policy as expert (LIBERO spatial task_id=2).
Saves LeRobot-compatible npy episodes. Prefers successful episodes; optionally keeps fails.
"""
from __future__ import annotations
import argparse, collections, dataclasses, json, logging, math, pathlib, sys
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _ws

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
TASK_ID = 2
TASK_LANG = "pick up the black bowl from table center and place it on the plate"


def proprio8(obs):
    g = float(np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[0])
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32).reshape(3),
        np.asarray(obs["robot0_eef_quat"], np.float32).reshape(4),
        np.array([g], np.float32),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-success", type=int, default=30)
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=220)  # libero_spatial horizon-ish
    ap.add_argument("--keep-failures", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = suite.get_task_init_states(TASK_ID)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(args.seed)
    client = _ws.WebsocketClientPolicy(args.host, args.port)

    max_steps = args.max_steps  # spatial default in main.py is 220
    print(
        f"POLICY_COLLECT_START task_id={TASK_ID} lang={task.language!r} "
        f"replan={args.replan_steps} want_success={args.num_success}",
        flush=True,
    )
    summary = []
    n_ok = n_fail = 0
    attempt = 0
    while n_ok < args.num_success and attempt < args.max_attempts:
        attempt += 1
        obs = env.set_init_state(init_states[(attempt - 1) % len(init_states)])
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        action_plan = collections.deque()
        images, wrists, states, actions = [], [], [], []
        success = False
        t = 0
        while t < max_steps:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            wrist_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
            if not action_plan:
                element = {
                    "observation/image": img_r,
                    "observation/wrist_image": wrist_r,
                    "observation/state": proprio8(obs),
                    "prompt": str(task.language),
                }
                chunk = client.infer(element)["actions"]
                assert len(chunk) >= args.replan_steps
                action_plan.extend(chunk[: args.replan_steps])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            images.append(np.asarray(obs["agentview_image"]))
            wrists.append(np.asarray(obs["robot0_eye_in_hand_image"]))
            states.append(proprio8(obs))
            actions.append(action)
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if env.check_success():
                success = True
                break
            if done:
                break
        meta = {
            "attempt": attempt,
            "success": success,
            "steps": len(actions),
            "task": TASK_LANG,
            "task_id": TASK_ID,
            "suite": "libero_spatial",
            "fps": 10,
            "dataset_ref": "lerobot/libero_spatial_image",
            "expert": "pi05_libero_lora/phase1_lora_baseline/29999",
            "replan_steps": args.replan_steps,
        }
        if success or args.keep_failures:
            idx = n_ok if success else (1000 + n_fail)
            ep_dir = out / ("episode_%04d" % idx)
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.save(ep_dir / "image.npy", np.stack(images, 0))
            np.save(ep_dir / "wrist_image.npy", np.stack(wrists, 0))
            np.save(ep_dir / "state.npy", np.stack(states, 0).astype(np.float32))
            np.save(ep_dir / "action.npy", np.stack(actions, 0).astype(np.float32))
            (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            # also save a short mp4 for evidence
            try:
                imageio.mimwrite(str(ep_dir / "replay.mp4"), images[::2], fps=10)
            except Exception as e:
                print("video skip", e, flush=True)
        if success:
            n_ok += 1
        else:
            n_fail += 1
        summary.append(meta)
        print(
            "ATT %d success=%s steps=%d ok=%d/%d fail=%d"
            % (attempt, success, len(actions), n_ok, args.num_success, n_fail),
            flush=True,
        )
        try:
            env.reset()
        except Exception:
            env.close()
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
            env.seed(args.seed + attempt)

    (out / "summary.json").write_text(
        json.dumps({"successes": n_ok, "failures": n_fail, "attempts": attempt, "episodes": summary}, indent=2)
    )
    print("POLICY_COLLECT_DONE ok=%d fail=%d attempts=%d out=%s" % (n_ok, n_fail, attempt, out), flush=True)
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
