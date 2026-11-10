#!/usr/bin/env python3
"""Clean A-policy PnP npy episodes and convert to local LeRobot dataset for openpi.

Cleaning:
- keep success episodes only
- drop non-finite / too-short episodes
- drop near-noop frames: max(|action[:6]|) < NOOP_ABS
- compress long idle runs: consecutive frames with eef_delta < IDLE_EEF and
  action_l2_6d < IDLE_ACT lasting > IDLE_RUN → keep first+last of the run

Images are rotated 180° on convert to match LIBERO/openpi train convention.
"""
from __future__ import annotations
import argparse, json, shutil, pathlib
import numpy as np

NOOP_ABS = 1e-3
IDLE_EEF = 1e-3
IDLE_ACT = 0.05
IDLE_RUN = 8
MIN_LEN = 20
TASK = "pick up the black bowl from table center and place it on the plate"
REPO_ID = "local/pnp_policy_spatial_center"


def eef_delta(state: np.ndarray) -> np.ndarray:
    eef = state[:, :3]
    d = np.linalg.norm(np.diff(eef, axis=0, prepend=eef[:1]), axis=1)
    return d


def clean_episode(state, action, image, wrist):
    n = len(action)
    assert state.shape[0] == n == image.shape[0] == wrist.shape[0]
    keep = np.ones(n, dtype=bool)

    # near-noop absolute
    noop = np.max(np.abs(action[:, :6]), axis=1) < NOOP_ABS
    keep &= ~noop

    # long idle compress: mark middle of long idle runs for drop
    d = eef_delta(state)
    idle = (d < IDLE_EEF) & (np.linalg.norm(action[:, :6], axis=1) < IDLE_ACT)
    i = 0
    while i < n:
        if not idle[i] or not keep[i]:
            i += 1
            continue
        j = i
        while j < n and idle[j] and keep[j]:
            j += 1
        run = j - i
        if run > IDLE_RUN:
            # keep first and last of run; drop middles
            keep[i + 1 : j - 1] = False
        i = j

    idx = np.where(keep)[0]
    return state[idx], action[idx], image[idx], wrist[idx], {
        "before_frames": int(n),
        "after_frames": int(len(idx)),
        "dropped_noop": int(noop.sum()),
        "dropped_idle_mid": int(n - noop.sum() - len(idx)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/autodl-tmp/datasets/pnp_policy_spatial_center")
    ap.add_argument("--clean-out", default="/root/autodl-tmp/datasets/pnp_policy_spatial_center_clean")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--stats-md", default="/root/autodl-tmp/logs/pnp-clean-stats.md")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    clean_out = pathlib.Path(args.clean_out)
    if clean_out.exists():
        shutil.rmtree(clean_out)
    clean_out.mkdir(parents=True)

    eps = sorted(src.glob("episode_*"))
    before_eps = len(eps)
    before_frames = 0
    after_eps = 0
    after_frames = 0
    dropped_eps = []
    per_ep = []

    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        action = np.load(ep / "action.npy").astype(np.float32)
        state = np.load(ep / "state.npy").astype(np.float32)
        image = np.load(ep / "image.npy")
        wrist = np.load(ep / "wrist_image.npy")
        before_frames += len(action)

        reasons = []
        if not meta.get("success", False):
            reasons.append("fail")
        if not (np.isfinite(action).all() and np.isfinite(state).all()):
            reasons.append("nonfinite")
        if image.ndim != 4 or wrist.ndim != 4 or image.shape[1:] != (256, 256, 3):
            reasons.append("bad_frame_shape")

        if reasons:
            dropped_eps.append({"ep": ep.name, "reasons": reasons, "steps": len(action)})
            continue

        st, act, img, wr, cstat = clean_episode(state, action, image, wrist)
        if len(act) < MIN_LEN:
            dropped_eps.append({"ep": ep.name, "reasons": ["too_short_after_clean"], "steps": len(act)})
            continue

        out_ep = clean_out / ("episode_%04d" % after_eps)
        out_ep.mkdir(parents=True)
        np.save(out_ep / "image.npy", img)
        np.save(out_ep / "wrist_image.npy", wr)
        np.save(out_ep / "state.npy", st)
        np.save(out_ep / "action.npy", act)
        meta2 = dict(meta)
        meta2["clean"] = cstat
        meta2["orig_ep"] = ep.name
        (out_ep / "meta.json").write_text(json.dumps(meta2, indent=2))
        after_eps += 1
        after_frames += len(act)
        per_ep.append({"ep": ep.name, **cstat})

    summary = {
        "before_episodes": before_eps,
        "before_frames": before_frames,
        "after_episodes": after_eps,
        "after_frames": after_frames,
        "dropped_episodes": dropped_eps,
        "thresholds": {
            "NOOP_ABS": NOOP_ABS,
            "IDLE_EEF": IDLE_EEF,
            "IDLE_ACT": IDLE_ACT,
            "IDLE_RUN": IDLE_RUN,
            "MIN_LEN": MIN_LEN,
        },
        "per_episode": per_ep,
        "repo_id": args.repo_id,
        "task": TASK,
        "note": "Images flipped 180 on LeRobot convert to match openpi/LIBERO train convention.",
    }
    (clean_out / "summary.json").write_text(json.dumps(summary, indent=2))

    stats_md = pathlib.Path(args.stats_md)
    stats_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PnP clean stats (2026-11-06 Asia/Shanghai)",
        "",
        f"- src: `{src}`",
        f"- clean_out: `{clean_out}`",
        f"- repo_id (LeRobot): `{args.repo_id}`",
        "",
        "## Before → After",
        f"| | episodes | frames |",
        f"|---|---:|---:|",
        f"| before | {before_eps} | {before_frames} |",
        f"| after | {after_eps} | {after_frames} |",
        f"| delta | {after_eps - before_eps} | {after_frames - before_frames} |",
        "",
        "## Filters",
        f"- drop fail / nonfinite / bad shape / after-clean len < {MIN_LEN}",
        f"- drop near-noop frames: max(|action[:6]|) < {NOOP_ABS}",
        f"- compress idle runs > {IDLE_RUN}: eef_delta < {IDLE_EEF} and action_l2_6d < {IDLE_ACT} (keep ends)",
        "",
        f"Dropped episodes: {len(dropped_eps)}",
        "",
    ]
    if dropped_eps:
        lines.append("```json")
        lines.append(json.dumps(dropped_eps, indent=2))
        lines.append("```")
    stats_md.write_text("\n".join(lines) + "\n")
    print("CLEAN_DONE", json.dumps({k: summary[k] for k in ("before_episodes","before_frames","after_episodes","after_frames")}))
    print("STATS_MD", stats_md)

    if args.skip_convert:
        return

    # Convert to LeRobot
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    out_path = HF_LEROBOT_HOME / args.repo_id
    if out_path.exists():
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    for ep in sorted(clean_out.glob("episode_*")):
        img = np.load(ep / "image.npy")
        wrist = np.load(ep / "wrist_image.npy")
        state = np.load(ep / "state.npy").astype(np.float32)
        action = np.load(ep / "action.npy").astype(np.float32)
        # 180° rotate to match openpi train preprocessing
        img_f = np.ascontiguousarray(img[:, ::-1, ::-1])
        wrist_f = np.ascontiguousarray(wrist[:, ::-1, ::-1])
        for t in range(len(action)):
            dataset.add_frame(
                {
                    "image": img_f[t],
                    "wrist_image": wrist_f[t],
                    "state": state[t],
                    "actions": action[t],
                    "task": TASK,
                }
            )
        dataset.save_episode()
        print("SAVED_EP", ep.name, "frames", len(action), flush=True)

    print("CONVERT_DONE", "path", out_path, "episodes", after_eps, "frames", after_frames, flush=True)
    # append convert path to stats
    with stats_md.open("a") as f:
        f.write(f"\n## LeRobot convert\n- path: `{out_path}`\n- flipped_180: true\n")


if __name__ == "__main__":
    main()
