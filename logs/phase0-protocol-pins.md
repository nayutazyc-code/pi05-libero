# Phase 0 protocol & environment pins (acceptance supplement v2)

Status wording: **score met (495/500 ≥ 494); acceptance pending re-review.**

## Source pins (codeload tarballs; remote tree has no .git, pins from install.log / scripts/remote/10-install.sh)
- openpi `215abfb217dbac7d5f1273282331b9b1866c0479` (same as experience-notes candidate)
- LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` (openpi submodule pin)
- aloha `d1dc83afd89ded4379851257fe5d85632d31d5ec`
- lerobot (uv dep) `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`
- `uv.lock` md5 `28032c9ed56f0d2ca81b6b580fedee18`; `examples/libero/requirements.txt` md5 `a0d7c6c20d342c648587d32243a5fce6`
- Runtime: jax 0.5.3 (server venv); robosuite 1.4.1, mujoco 3.2.3 (libero py3.8 venv); GPU RTX 4090 D, driver 580.76.05

## Checkpoint
- `pi05_libero` from official GCS listing, fetched via hf-mirror, every file md5-verified vs GCS: `ckpt-verify.log` → `VERIFY_BAD 0`, `CKPT_VERIFIED_DONE`
- norm_stats `assets/physical-intelligence/libero/norm_stats.json` md5 `03d102a77c0aa343afbf7569c97bbcba`

## Launch args (scripts/remote/20-eval-spatial.sh)
- server: `serve_policy.py --env LIBERO policy:checkpoint --policy.config=pi05_libero --policy.dir=.../pi05_libero`
- client: `examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 50 --args.seed 7 --args.video-out-path /root/autodl-tmp/logs/videos_spatial`
- main.py defaults at pinned commit (verified by grep): replan_steps=5 (action prefix), num_steps_wait=10, resize_size=224, max_steps=220 for libero_spatial, seed=7. Obs/action transforms = unmodified pinned main.py + pi05_libero config.

## Completeness / integrity
- Full log md5 `824c7a42734bd292440ee0dd87eca4e7` (local `logs/eval-spatial-full.log` == remote), 150722 bytes
- 500 scored episodes = 10 tasks × 50, no duplicates/replacements, single run, no rerun
- `Caught exception` count in log: **0** → all 5 failures are timeouts at max_steps=220 (no success), not crashes
- Final lines: `Total success rate: 0.99`, `Total episodes: 500`, `EVAL_DONE` (echo only runs after main.py returns 0 under `set -e`)
- Post-scoring EGL `__del__` exceptions are "Exception ignored" destructor noise at interpreter exit; script still reached EVAL_DONE.
- server.log: 1 websocket handshake EOF = readiness TCP probe in script (`/dev/tcp` open/close), not a client disconnect.

## Failures (task, episode idx as logged)
| task | fails |
|---|---|
| on the cookie box | ep 8 |
| in the top drawer of the wooden cabinet | ep 26, 31 |
| on the wooden cabinet | ep 23, 44 |
Per-task: 7 tasks 50/50; above three 49/50, 48/50, 48/50.

## Video index
Pinned main.py writes `rollout_{task}_{success|failure}.mp4` (overwrites per task/outcome; upstream behaviour, not per-episode). 13 files; md5 list:
- b4559eea98d3ec341146b0d16323e7a1  rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_success.mp4
- 65d64a6fb2308904556178352350daf3  rollout_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_success.mp4
- 9fa6b61b8d3b6fbca0805b3849a301b2  rollout_pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_failure.mp4
- e23f0de8d6a55bcef7430c7a71c417fb  rollout_pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_success.mp4
- 78caea621ff1b39527c68b7464272152  rollout_pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_success.mp4
- cf1804087052b01a5a8cd63415124146  rollout_pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_success.mp4
- 2811f3b310d244f632f7c88fef0682c7  rollout_pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_success.mp4
- cc3410e216aaa5d1f9cdd6e8f5c7ad9d  rollout_pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_failure.mp4
- ae1ce3bc8304835ef0f9c000951e4a90  rollout_pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_success.mp4
- ca29b733ea9fc89020884a055c3d4b82  rollout_pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_success.mp4
- 64e38c3ef29e5407b54e037426633915  rollout_pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_success.mp4
- 393c2b7d84e239783d68be408e352c4a  rollout_pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_failure.mp4
- 5b9a752a22d58f0725273a1045257dc9  rollout_pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_success.mp4

## Environment diffs vs official
- Native uv/conda install instead of Docker (AutoDL has no docker); EGL headless rendering (MUJOCO_GL=egl); PyPI via aliyun mirror; sources via codeload tarballs. No protocol parameter changed.
- Smoke: no separate 10-ep smoke kept; server load + first episodes verified in-run (deviation from plan step 3, recorded).
- Resources: GPU idle now 0 MiB; disk /root/autodl-tmp 40G/150G used; eval wall ~67 min.
