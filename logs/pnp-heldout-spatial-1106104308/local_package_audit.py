"""Audit the downloaded evidence without loading policies or running rollouts."""
import collections
import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

from PIL import Image
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(root):
    inventory = read(root / "remote-inventory.json")
    for entry in inventory["files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], path
    for entry in read(root / "source-snapshots.json"):
        path = root / entry["local"]
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], path

    recount = json.loads(subprocess.check_output([
        sys.executable, str(root / "independent_audit.py"), str(root),
    ]))
    assert recount == read(root / "independent-audit.json")
    (root / "local-independent-audit.json").write_text(json.dumps(recount, indent=2) + "\n")
    protocol = read(root / "protocol.json")
    manifest = read(root / "heldout/manifest.json")
    assert digest(root / "heldout/selection_rules.json") == manifest["rules_sha256"]
    assert read(root / "heldout/manifest.sha256.json")["sha256"] == protocol["manifest_sha256"]
    collection = root / "source-snapshots/datasets/pnp_policy_spatial_center/summary.json"
    collector = root / "source-snapshots/scripts/collect_pnp_policy_expert.py"
    assert digest(collection) == manifest["collection_summary_sha256"]
    assert digest(collector) == manifest["collector_sha256"]
    summary = read(collection)
    assert (summary["attempts"], summary["successes"], summary["failures"]) == (31, 30, 1)
    assert [e["attempt"] for e in summary["episodes"]] == list(range(1, 32))
    assert all(e["task_id"] == 2 and e["suite"] == "libero_spatial" for e in summary["episodes"])
    excluded = set(manifest["excluded_state_hashes"])
    used = read(root / "collection-used-initial-states.json")
    assert used["source"] == manifest["tasks"][2]["init_file"]
    assert used["source_sha256"] == manifest["tasks"][2]["init_file_sha256"]
    assert used["indices"] == list(range(31)) and len(used["states"]) == 31
    used_hashes = set()
    for state in used["states"]:
        physical = np.round(np.asarray(state, dtype="<f8")[1:], 12)
        physical[physical == 0] = 0
        used_hashes.add(hashlib.sha256(physical.tobytes()).hexdigest())
    assert len(used_hashes) == 31 and used_hashes == excluded
    rejected = [e for e in read(root / "heldout/task2_selection_audit.json")
                if e["source"] == "existing" and e["source_index"] <= 30]
    assert len(rejected) == 31 and {e["state_hash"] for e in rejected} == excluded
    assert all(not e["valid"] and e["reason"] == "collection_overlap_or_duplicate" for e in rejected)
    sources = {}
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        candidates = read(root / ("heldout/task%d_selection_audit.json" % task_id))
        chosen = {e["state_hash"]: e for e in candidates if e["valid"]}
        for state in task["states"]:
            assert state["seed"] == 7 and all(math.isfinite(x) for x in state["state"])
            metrics = state["validation"]
            assert metrics["valid"] and metrics["reason"] == "ok"
            assert metrics["last_qpos_delta"] <= 0.02 and metrics["max_abs_qvel"] <= 5
            entry = chosen[state["state_hash"]]
            assert all(entry[k] == state[k] for k in ("source", "source_index"))
        sources[str(task_id)] = dict(collections.Counter(s["source"] for s in task["states"]))
        assert sources[str(task_id)] == ({"existing": 19, "native_reset": 31} if task_id == 2 else {"existing": 50})

    scripts = pathlib.Path(__file__).resolve().parents[2] / "scripts/remote"
    for name, expected in protocol["eval_scripts_sha256"].items():
        assert digest(scripts / name) == expected, name
    log_counts = {pattern: len(list(root.glob(pattern))) for pattern in (
        "formal-eval-*.log", "preflight-eval-*.log", "server-*.log")}
    assert list(log_counts.values()) == [40, 8, 4]
    preflight = [read(p) for p in (root / "preflight-trials").glob("*/task*/*.json")]
    assert len(preflight) == 32
    assert all(e["stage"] == "preflight" and e["protocol_sha256"] == digest(root / "protocol.json") for e in preflight)
    preflight_results = [e for e in preflight if e["event"] == "result"]
    assert len({e["trial_id"] for e in preflight_results}) == 16
    assert collections.Counter((e["group"], e["task_id"]) for e in preflight_results) == {
        (g, t): 2 for g in protocol["group_order"] for t in (2, 4)}

    videos = {}
    video_index = ["# 本地代表视频索引", "", "每个已出现的组×任务×结果类别各保留一条；已完整解码。", "",
                   "| 阶段 | 组 | 任务 | 结果 | 视频 |", "|---|---|---:|---|---|"]
    remote_videos = read(root / "video-audit.json")
    for stage, events, directory in (("formal", "trials", "videos"),
                                      ("preflight", "preflight-trials", "preflight-videos")):
        records = [read(p) for p in (root / events).glob("*/task*/*_result.json")]
        wanted = {(e["group"], e["task_id"], e["success"]) for e in records}
        saved = [e for e in records if e["video"]]
        assert len(saved) == len(wanted)
        assert {(e["group"], e["task_id"], e["success"]) for e in saved} == wanted
        expected = {pathlib.Path(e["video"]).name for e in saved}
        assert expected == {p.name for p in (root / directory).glob("*.mp4")}
        prior = {pathlib.Path(e["file"]).name: e for e in remote_videos[stage]["files"]}
        files = []
        for record in sorted(saved, key=lambda e: (e["group"], e["task_id"], e["success"])):
            path = root / directory / pathlib.Path(record["video"]).name
            assert pathlib.Path(record["video"]).parent == pathlib.Path(inventory["root"]) / directory
            decoded = subprocess.check_output([
                "ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path),
                "-map", "0:v:0", "-f", "framemd5", "-",
            ], text=True)
            assert "#dimensions 0: 256x256" in decoded
            frames = sum(bool(line) and not line.startswith("#") for line in decoded.splitlines())
            assert frames > 0 and frames == prior[path.name]["frames_decoded"]
            assert digest(path) == prior[path.name]["sha256"]
            relative = str(path.relative_to(root))
            files.append({"file": relative, "frames_decoded": frames, "sha256": digest(path)})
            video_index.append("| %s | %s | %d | %s | [%s](%s) |" % (
                stage, record["group"], record["task_id"], record["outcome"], path.name, relative))
        videos[stage] = {"videos": len(files), "all_frames_decoded": sum(e["frames_decoded"] for e in files),
                         "outcome_coverage_complete": True, "files": files}

    plots = root.parent / "plots"
    losses = {}
    for path in sorted(plots.glob("pnp*.csv")):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        steps = [int(row["step"]) for row in rows]
        assert steps == sorted(set(steps)), path
        assert all(math.isfinite(float(row["loss"])) for row in rows)
        losses[path.name] = {"points": len(rows), "first_step": steps[0] if steps else None,
                             "last_step": steps[-1] if steps else None, "sha256": digest(path)}
    provenance = read(plots / "pnp-loss-provenance.json")
    for group in ("g0", "g1"):
        entry = losses["pnp_%s_wandb_loss.csv" % group]
        assert (entry["points"], entry["first_step"], entry["last_step"]) == (50, 0, 2450)
        assert entry["points"] == provenance[group]["points"]
    pngs = {}
    for name in ("pnp_g0_wandb_loss_curve.png", "pnp_g1_wandb_loss_curve.png", "pnp_g1_loss_curve.png"):
        path = plots / name
        with Image.open(path) as image:
            image.load()
            assert image.format == "PNG" and min(image.size) > 0
            pngs[name] = {"dimensions": list(image.size), "sha256": digest(path)}
    (root / "local-video-index.md").write_text("\n".join(video_index) + "\n")
    report = {"finished_utc": datetime.now(timezone.utc).isoformat(),
              "remote_files_verified": len(inventory["files"]), "log_counts": log_counts,
              "recount_matches_remote": True, "preflight_trials_excluded": 16,
              "collection_attempts_excluded": 31, "collection_successes": 30, "collection_failures": 1,
              "collection_state_hashes_recomputed_from_original_states": True,
              "heldout_sources_by_task": sources, "frozen_eval_scripts_match": True,
              "videos": videos, "loss_csv": losses, "loss_png": pngs,
              "note": "Empty loss CSVs belong to the earlier OOM attempts; completed G0/G1 wandb curves each have 50 ordered unique steps."}
    (root / "local-package-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files_verified": len(inventory["files"]), "valid_trials": recount["valid_trials"],
                      "videos": {k: {n: v[n] for n in ("videos", "all_frames_decoded")} for k, v in videos.items()},
                      "loss_curves": "G0/G1: 50 unique ascending steps, PNG decoded"}, indent=2))


if __name__ == "__main__":
    audit(pathlib.Path(__file__).resolve().parent)
