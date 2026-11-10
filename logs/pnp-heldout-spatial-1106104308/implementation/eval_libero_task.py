#!/usr/bin/env python3
"""Eval openpi policy server on a single LIBERO task_id (default spatial/2)."""
from __future__ import annotations
import collections, dataclasses, json, logging, pathlib
import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _ws

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    task_id: int = 2
    num_trials: int = 50
    num_steps_wait: int = 10
    max_steps: int = 220
    video_out_path: str = "/root/autodl-tmp/logs/videos_pnp_eval"
    seed: int = 7
    save_video_n: int = 3  # save first N success + first N fail
    tag: str = "eval"
    checkpoint_path: str = ""
    policy_config: str = ""
    init_manifest: str = ""
    protocol_file: str = ""
    results_path: str = ""
    preflight_trials: int = 0


def proprio8(obs):
    g = float(np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[0])
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32).reshape(3),
        np.asarray(obs["robot0_eef_quat"], np.float32).reshape(4),
        np.array([g], np.float32),
    ])


def main(args: Args):
    if args.init_manifest:
        from pnp_heldout_eval import evaluate
        return evaluate(args)
    np.random.seed(args.seed)
    out = pathlib.Path(args.video_out_path)
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(args.seed)
    client = _ws.WebsocketClientPolicy(args.host, args.port)

    logging.info(
        "EVAL_TASK_START suite=%s task_id=%d lang=%r trials=%d replan=%d tag=%s",
        args.task_suite_name, args.task_id, task.language, args.num_trials, args.replan_steps, args.tag,
    )
    successes = 0
    results = []
    n_vid_ok = n_vid_fail = 0
    for ep in tqdm.tqdm(range(args.num_trials)):
        obs = env.set_init_state(init_states[ep % len(init_states)])
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        action_plan = collections.deque()
        frames = []
        success = False
        t = 0
        while t < args.max_steps:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_r = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
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
            frames.append(np.asarray(obs["agentview_image"]))
            obs, _, done, _ = env.step(action.tolist())
            t += 1
            if env.check_success():
                success = True
                break
            if done:
                break
        if success:
            successes += 1
        results.append({"ep": ep, "success": success, "steps": t})
        want_vid = (success and n_vid_ok < args.save_video_n) or ((not success) and n_vid_fail < args.save_video_n)
        if want_vid and frames:
            suffix = "ok" if success else "fail"
            path = out / f"{args.tag}_t{args.task_id}_replan{args.replan_steps}_ep{ep:02d}_{suffix}.mp4"
            try:
                imageio.mimwrite(str(path), frames[::2], fps=10)
            except Exception as e:
                logging.warning("video skip %s", e)
            if success:
                n_vid_ok += 1
            else:
                n_vid_fail += 1
        logging.info("ep=%d success=%s steps=%d rate=%d/%d", ep, success, t, successes, ep + 1)
        try:
            env.reset()
        except Exception:
            env.close()
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
            env.seed(args.seed + ep + 1)

    rate = successes / max(args.num_trials, 1)
    summary = {
        "tag": args.tag,
        "checkpoint_path": args.checkpoint_path,
        "policy_config": args.policy_config,
        "suite": args.task_suite_name,
        "task_id": args.task_id,
        "language": str(task.language),
        "trials": args.num_trials,
        "successes": successes,
        "success_rate": rate,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "state_contract": "eef_pos(3)+eef_quat_xyzw(4)+gripper_qpos[0](1)",
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "resize_size": args.resize_size,
        "init_state_indices": [ep % len(init_states) for ep in range(args.num_trials)],
        "results": results,
    }
    sum_path = out / f"{args.tag}_task{args.task_id}_replan{args.replan_steps}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(
        "EVAL_TASK_DONE tag=%s successes=%d/%d rate=%.4f replan=%d summary=%s"
        % (args.tag, successes, args.num_trials, rate, args.replan_steps, sum_path),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
