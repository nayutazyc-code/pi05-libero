"""Independent acceptance recount; run with the LIBERO Python environment."""
import collections
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol = json.loads((root / "protocol.json").read_text())
protocol_hash = digest(root / "protocol.json")
manifest = json.loads((root / "heldout/manifest.json").read_text())
assert digest(root / "heldout/manifest.json") == protocol["manifest_sha256"]
groups = ("A29999", "G0", "G1", "G2")
assert protocol["group_order"] == list(groups)
assert dict(protocol["groups"]["G0"], replan=10) == protocol["groups"]["G2"]
assert protocol["groups"]["G0"]["replan"] == 5
excluded = set(manifest["excluded_state_hashes"])
assert len(excluded) == 31
expected = {}
for task_id, task in enumerate(manifest["tasks"]):
    assert task["task_id"] == task_id and len(task["states"]) == 50
    seen = set()
    for index, state in enumerate(task["states"]):
        physical = np.round(np.array(state["state"][1:], dtype="<f8"), 12)
        physical[physical == 0] = 0
        identity = hashlib.sha256(physical.tobytes()).hexdigest()
        assert identity == state["state_hash"] and identity not in seen | excluded
        assert index == state["trial_index"]
        seen.add(identity)
        for group in groups:
            expected[group, task_id, index] = identity
assert len(expected) == 2000
starts, results = {}, {}
for path in sorted((root / "trials").glob("*/task*/*.json")):
    record = json.loads(path.read_text())
    key = record["group"], record["task_id"], record["trial_index"]
    assert record["state_hash"] == expected[key]
    assert record["trial_id"] == "%s:libero_spatial:%d:%s" % (key[0], key[1], expected[key])
    assert record["protocol_sha256"] == protocol_hash and record["stage"] == "formal"
    assert record["seed"] == 7 and record["attempt"] == 1
    assert record["event"] in ("start", "result"), "unexpected execution error"
    target = starts if record["event"] == "start" else results
    assert key not in target, "duplicate event"
    target[key] = record
assert set(starts) == set(results) == set(expected)
successes = collections.Counter()
outcomes = collections.Counter()
for key, record in results.items():
    assert record["time_unix"] >= starts[key]["time_unix"]
    assert record["outcome"] in ("success", "timeout", "policy_failure")
    assert type(record["success"]) is bool and record["success"] == (record["outcome"] == "success")
    assert 1 <= record["steps"] <= 220
    assert record["outcome"] != "timeout" or record["steps"] == 220
    successes[key[:2]] += record["success"]
    outcomes[record["outcome"]] += 1
with (root / "summary/per_task.csv").open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 40 and len({(r["group"], r["task_id"]) for r in rows}) == 40
for row in rows:
    group, task_id = row["group"], int(row["task_id"])
    count = successes[group, task_id]
    assert int(row["successes"]) == count and int(row["trials"]) == 50
    assert abs(float(row["success_rate"]) - count / 50) < 1e-12
    assert abs(float(row["delta_vs_g0_pp"]) - 2 * (count - successes["G0", task_id])) < 1e-10
audit = json.loads((root / "summary/audit.json").read_text())
assert audit["protocol_sha256"] == protocol_hash
assert not audit["missing"] and not audit["execution_errors"] and not audit["interrupted_attempts"]
assert audit["extra_attempts"] == 0
totals = {group: sum(successes[group, task_id] for task_id in range(10)) for group in groups}
report = {"valid_trials": len(results), "start_events": len(starts), "per_task_cells": len(rows),
          "trials_per_group": 500, "successes": totals, "outcomes": dict(outcomes),
          "delta_vs_g0_pp": {group: (count - totals["G0"]) / 5 for group, count in totals.items()},
          "manifest_sha256": protocol["manifest_sha256"], "protocol_sha256": protocol_hash,
          "errors": 0, "duplicates": 0, "missing": 0, "collection_overlap": 0,
          "g0_g2_same_assets": True, "summary_matches_independent_recount": True}
print(json.dumps(report, indent=2))
