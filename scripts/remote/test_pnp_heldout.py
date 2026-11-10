#!/usr/bin/env python3
"""Exercise scoring, manifest rejection and resume without policy rollouts."""
import contextlib
import copy
import io
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from build_pnp_heldout import file_hash, state_hash
from eval_libero_task import Args
import pnp_heldout_eval as evaluation
from summarize_pnp_heldout import recompute


class HeldoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.bddl = self.root / "scene.bddl"
        self.bddl.write_text("test scene")
        self.manifest = {"excluded_state_hashes": [state_hash([0, i + 10000, 0]) for i in range(31)], "tasks": []}
        for task_id in range(10):
            states = []
            for index in range(50):
                state = [0, task_id * 100 + index, 1]
                states.append({"trial_index": index, "state": state, "state_hash": state_hash(state)})
            self.manifest["tasks"].append({"task_id": task_id, "language": f"task {task_id}", "states": states, "bddl_sha256": file_hash(self.bddl)})
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.protocol = {"manifest": str(self.manifest_path), "manifest_sha256": file_hash(self.manifest_path),
            "groups": {g: {"checkpoint": "test", "config": "test", "replan": 10 if g == "G2" else 5} for g in ("A29999", "G0", "G1", "G2")},
            "group_order": ["A29999", "G0", "G1", "G2"], "eval_scripts_sha256": {},
            "seed": 7, "max_steps": 220, "num_steps_wait": 10, "resize_size": 224, "task_suite_name": "libero_spatial"}
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.args = Args(tag="G0", task_id=1, checkpoint_path="test", policy_config="test", save_video_n=0,
            init_manifest=str(self.manifest_path), protocol_file=str(self.protocol_path),
            results_path=str(self.root / "trials"), video_out_path=str(self.root / "videos"))

    def test_manifest_content_identity_and_rejections(self):
        evaluation.verify_manifest(self.manifest)
        self.assertEqual(state_hash([0, 1, 2]), state_hash([99, 1, 2]))
        for kind in ("duplicate", "leak", "order", "count"):
            manifest = copy.deepcopy(self.manifest)
            states = manifest["tasks"][0]["states"]
            if kind == "duplicate":
                states[1] = copy.deepcopy(states[0])
            elif kind == "leak":
                manifest["excluded_state_hashes"][0] = states[0]["state_hash"]
            elif kind == "order":
                states[0]["trial_index"] = 99
            else:
                states.pop()
            with self.assertRaises(AssertionError, msg=kind):
                evaluation.verify_manifest(manifest)
        self.args.task_id = 10
        with self.assertRaises(AssertionError):
            evaluation.evaluate(self.args)

    def test_error_resume_and_no_duplicate_score(self):
        digest = file_hash(self.protocol_path)
        counts = {"calls": 0, "fail_once": True}
        class Env:
            def __init__(self, **kwargs): self.steps = 0
            def seed(self, seed): pass
            def reset(self): self.steps = 0
            def set_init_state(self, state): return self.obs()
            def obs(self):
                return {"agentview_image": np.zeros((2, 2, 3), np.uint8), "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
                        "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]), "robot0_gripper_qpos": np.zeros(2)}
            def step(self, action):
                self.steps += 1
                return self.obs(), 0, False, {}
            def check_success(self): return self.steps > 10
            def close(self): pass
        class Client:
            def __init__(self, *args): pass
            def get_server_metadata(self): return {"heldout_protocol_sha256": digest, "heldout_group": "G0"}
            def infer(self, obs):
                counts["calls"] += 1
                if counts["fail_once"]:
                    counts["fail_once"] = False
                    raise RuntimeError("injected service error")
                return {"actions": np.zeros((10, 7))}
        suite = types.SimpleNamespace(get_task=lambda i: types.SimpleNamespace(language=f"task {i}"),
            get_task_bddl_file_path=lambda i: str(self.bddl))
        with mock.patch.object(evaluation.benchmark, "get_benchmark_dict", return_value={"libero_spatial": lambda: suite}), \
             mock.patch.object(evaluation, "OffScreenRenderEnv", Env), \
             mock.patch.object(evaluation._ws, "WebsocketClientPolicy", Client), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "injected"):
                evaluation.evaluate(self.args)
            evaluation.evaluate(self.args)
            before = counts["calls"]
            evaluation.evaluate(self.args)
            self.assertEqual(counts["calls"], before)
            events = evaluation.read_events(self.root / "trials/G0/task1")
            self.assertEqual(sum(e["event"] == "result" for e in events), 50)
            self.assertEqual(sum(e["event"] == "error" for e in events), 1)
            report = recompute(self.root / "trials", self.protocol_path, partial=True)
            self.assertEqual(len(report["results"]), 50)
            self.assertEqual(report["extra_attempts"], 1)
            # A changed protocol must reject existing scored events.
            self.protocol["test_revision"] = 2
            self.protocol_path.write_text(json.dumps(self.protocol))
            with self.assertRaisesRegex(AssertionError, "resume protocol mismatch"):
                evaluation.evaluate(self.args)

    def test_independent_2000_trial_recount(self):
        digest = file_hash(self.protocol_path)
        root = self.root / "trials"
        for gi, group in enumerate(self.protocol["group_order"]):
            for task in self.manifest["tasks"]:
                directory = root / group / f"task{task['task_id']}"
                directory.mkdir(parents=True)
                for state in task["states"]:
                    success = state["trial_index"] < 40 + gi
                    event = {"protocol_sha256": digest, "trial_id": f"{group}:libero_spatial:{task['task_id']}:{state['state_hash']}",
                        "group": group, "task_id": task["task_id"], "trial_index": state["trial_index"], "attempt": 1, "stage": "formal"}
                    (directory / f"{state['trial_index']}_start.json").write_text(json.dumps(dict(event, event="start")))
                    (directory / f"{state['trial_index']}_result.json").write_text(json.dumps(dict(event, event="result", success=success,
                        outcome="success" if success else "timeout", steps=1 if success else 220)))
        report = recompute(root, self.protocol_path)
        self.assertEqual(len(report["results"]), 2000)
        self.assertEqual(len(report["rows"]), 40)
        self.assertAlmostEqual(next(r for r in report["rows"] if r["group"] == "G2")["delta_vs_g0_pp"], 4.0)
        file = root / "G0/task0/0_result.json"
        (file.parent / "duplicate_result.json").write_text(file.read_text())
        with self.assertRaises(AssertionError):
            recompute(root, self.protocol_path)


if __name__ == "__main__":
    unittest.main()
