#!/usr/bin/env python3
"""Recompute all counts from trial events, never from pre-existing summaries."""
import argparse
import csv
import hashlib
import json
import pathlib


def recompute(root, protocol_path, partial=False):
    protocol_path = pathlib.Path(protocol_path)
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol = json.loads(protocol_path.read_text())
    manifest_path = pathlib.Path(protocol["manifest"])
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    expected = {}
    for group in protocol["group_order"]:
        for task in manifest["tasks"]:
            for state in task["states"]:
                key = f"{group}:libero_spatial:{task['task_id']}:{state['state_hash']}"
                assert key not in expected
                expected[key] = (group, task["task_id"], state["trial_index"])
    assert len(expected) == 2000
    results, starts, terminals, errors = {}, {}, {}, []
    for path in sorted(pathlib.Path(root).glob("*/task*/*.json")):
        event = json.loads(path.read_text())
        assert event["stage"] == "formal", "preflight must never enter formal scores"
        identity = event["trial_id"]
        assert identity in expected and event["protocol_sha256"] == digest, str(path)
        assert (event["group"], event["task_id"], event["trial_index"]) == expected[identity]
        attempt = (identity, event["attempt"])
        assert event["attempt"] >= 1
        if event["event"] == "start":
            assert attempt not in starts
            starts[attempt] = event
        elif event["event"] in ("result", "error"):
            assert attempt not in terminals
            terminals[attempt] = event
            if event["event"] == "error":
                errors.append(event)
                continue
            assert identity not in results, f"duplicate scored identity: {identity}"
            assert isinstance(event["success"], bool)
            assert event["outcome"] in ("success", "timeout", "policy_failure")
            assert event["success"] == (event["outcome"] == "success")
            assert 0 < event["steps"] <= 220
            if event["outcome"] == "timeout":
                assert event["steps"] == 220
            results[identity] = event
        else:
            raise ValueError(f"unknown event: {path}")
    assert set(terminals) <= set(starts), "terminal event lacks start"
    for identity, attempt in starts:
        previous = [n for key, n in starts if key == identity]
        assert sorted(previous) == list(range(1, max(previous) + 1))
        if identity in results:
            assert attempt <= results[identity]["attempt"], "attempt after scored result"
    missing = sorted(set(expected) - set(results))
    if not partial:
        assert not missing, f"missing {len(missing)} valid trials"
    rows = []
    for task_id in range(10):
        base = [e for e in results.values() if e["group"] == "G0" and e["task_id"] == task_id]
        base_rate = sum(e["success"] for e in base) / len(base) if base else None
        for group in protocol["group_order"]:
            records = [e for e in results.values() if e["group"] == group and e["task_id"] == task_id]
            success = sum(e["success"] for e in records)
            rate = success / len(records) if records else None
            rows.append({"group": group, "task_id": task_id, "successes": success,
                         "trials": len(records), "success_rate": rate,
                         "delta_vs_g0_pp": 100 * (rate - base_rate) if rate is not None and base_rate is not None else None})
    return {"protocol_sha256": digest, "rows": rows, "results": results,
            "missing": missing, "execution_errors": errors,
            "interrupted_attempts": [starts[k] for k in set(starts) - set(terminals)],
            "extra_attempts": len(starts) - len(set(key for key, attempt in starts))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    report = recompute(args.root, args.protocol, args.partial)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_task.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    (out / "audit.json").write_text(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2) + "\n")
    rows = report["rows"]
    total = {g: (sum(r["successes"] for r in rows if r["group"] == g), sum(r["trials"] for r in rows if r["group"] == g)) for g in ("A29999", "G0", "G1", "G2")}
    lines = ["# PnP held-out spatial results", "", "Incomplete" if report["missing"] else "Complete: 2000/2000 valid trials", "",
             "G0 is an internal control using some official hyperparameters: batch32 (micro8 × accum4), horizon10, synchronous replan5. Official installed-source evidence: logs/pnp-config-evidence.txt; config.py TrainConfig.batch_size and examples/libero/main.py Args.replan_steps.", "",
             "| Group | Successes / trials | Rate | Δ vs G0 | Difference from G0 |", "|---|---:|---:|---:|---|"]
    notes = {"A29999": "A/29999, no PnP fine-tuning; different data/normalization", "G0": "internal control, micro8 ×4", "G1": "native batch16; different training exposure/RNG", "G2": "same G0 weights/normalization; synchronous replan10 only"}
    bsuccess, bcount = total["G0"]
    for group, (success, count) in total.items():
        rate = f"{100 * success / count:.1f}%" if count else "—"
        delta = f"{100 * (success / count - bsuccess / bcount):+.1f}pp" if count and bcount else "—"
        lines.append(f"| {group} | {success}/{count} | {rate} | {delta} | {notes[group]} |")
    lines += ["", "| Task | Group | Successes / trials | Rate | Δ vs G0 |", "|---:|---|---:|---:|---:|"]
    for row in rows:
        rate = f"{100 * row['success_rate']:.1f}%" if row["success_rate"] is not None else "—"
        delta = f"{row['delta_vs_g0_pp']:+.1f}pp" if row["delta_vs_g0_pp"] is not None else "—"
        lines.append(f"| {row['task_id']} | {row['group']} | {row['successes']}/{row['trials']} | {rate} | {delta} |")
    lines += ["", f"Protocol SHA256: {report['protocol_sha256']}",
              f"Execution errors: {len(report['execution_errors'])}; interrupted attempts: {len(report['interrupted_attempts'])}; extra attempts: {report['extra_attempts']}.", "",
              "Held-out only against all 31 PnP collection attempts; no claim of isolation from A's earlier training data. Task2 includes 31 native-reset generated states, fixed before policy scores. Nonofficial xyz+quaternion+first-gripper state encoding is retained under the prior gate approval. Results are internal comparisons. G0/G1 exposure differs and cannot isolate batch causality; 50 trials/task have sampling uncertainty. G2 is synchronous replanning, not async RTC.", "",
              "Raw events: ../trials/; frozen inputs: ../heldout/manifest.json and ../protocol.json; videos: ../videos/. No prior evaluation results are included."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"RECOMPUTED valid={len(report['results'])} missing={len(report['missing'])}", flush=True)


if __name__ == "__main__":
    main()
