#!/usr/bin/env python3
"""Bind the manifest, exact checkpoint leaves, normalizers and source files."""
import argparse
import hashlib
import json
import pathlib


W = pathlib.Path("/root/autodl-tmp")
SCRIPTS = ["eval_libero_task.py", "pnp_heldout_eval.py", "build_pnp_heldout.py",
           "serve_pnp_heldout.py", "freeze_pnp_heldout.py", "summarize_pnp_heldout.py",
           "103-pnp-heldout-spatial.sh"]


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root):
    root = pathlib.Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    assert files, f"missing assets: {root}"
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]


def check(protocol):
    assert sha(protocol["manifest"]) == protocol["manifest_sha256"]
    for name, digest in protocol["eval_scripts_sha256"].items():
        assert sha(W / "scripts" / name) == digest, name
    for entry in protocol["source_files"]:
        assert sha(entry["path"]) == entry["sha256"], entry["path"]
    checked = set()
    for group in protocol["groups"].values():
        for entry in group["params"] + group["normalization"]:
            if entry["path"] not in checked:
                assert sha(entry["path"]) == entry["sha256"], entry["path"]
                checked.add(entry["path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    if args.verify:
        check(json.loads(out.read_text()))
        print("PROTOCOL_VERIFIED", sha(out), flush=True)
        return
    assert not out.exists(), "frozen protocol must never be overwritten"
    manifest = pathlib.Path(args.manifest).resolve()
    data = json.loads(manifest.read_text())
    assert len(data["tasks"]) == 10
    roots = {
        "A29999": ("pi05_libero_lora", "phase1_lora_baseline/29999"),
        "G0": ("pi05_libero_lora_pnp", "pnp_g0_accum_bs32_mb8/2499"),
        "G1": ("pi05_libero_lora_pnp", "pnp_g1_bs16/2499"),
    }
    groups = {}
    for tag, (cfg, leaf) in roots.items():
        ckpt = W / "openpi/checkpoints" / cfg / leaf
        groups[tag] = {"checkpoint": str(ckpt), "config": cfg, "replan": 5,
                       "params": snapshot(ckpt / "params"),
                       "normalization": snapshot(ckpt / "assets")}
        assert any(pathlib.Path(x["path"]).name == "norm_stats.json" for x in groups[tag]["normalization"])
        print("CHECKPOINT_HASHED", tag, flush=True)
    groups["G2"] = dict(groups["G0"], replan=10)
    sources = [W / "openpi" / x for x in (
        "src/openpi/training/config.py", "src/openpi/policies/policy.py",
        "src/openpi/policies/policy_config.py", "src/openpi/models/pi0.py",
        "src/openpi/policies/libero_policy.py", "examples/libero/main.py",
        "third_party/libero/libero/libero/envs/env_wrapper.py",
    )]
    protocol = {
        "schema": 1, "manifest": str(manifest), "manifest_sha256": sha(manifest),
        "groups": groups, "group_order": ["A29999", "G0", "G1", "G2"],
        "task_order": list(range(10)), "seed": 7, "task_suite_name": "libero_spatial",
        "trials_per_task": 50, "max_steps": 220, "num_steps_wait": 10,
        "resize_size": 224, "render_resolution": 256, "action_horizon": 10,
        "state_contract": "eef_pos(3)+eef_quat_xyzw(4)+gripper_qpos[0](1)",
        "rng": "each request sets jax.random.key(uint32 little-endian SHA256('7:task_id:state_hash:control_step')[:4]); identical across groups/retries",
        "success": "env.check_success() after every control step",
        "timeout": "220 control steps exhausted => task failure; service/walltime exceptions => unscored execution error",
        "trial_wall_deadline_seconds": 600,
        "retries": "no automatic retries; error gate before same-identity retry; completed results immutable",
        "generated_states_disclosure": {str(t["task_id"]): sum(s["source"] == "native_reset" for s in t["states"]) for t in data["tasks"]},
        "eval_scripts_sha256": {name: sha(W / "scripts" / name) for name in SCRIPTS},
        "source_files": [{"path": str(p), "sha256": sha(p)} for p in sources],
        "official_config_evidence": "logs/pnp-config-evidence.txt (installed source snapshot); G0 internal control with some official hyperparameters, not official benchmark",
        "limitations": ["held-out only relative to the 31 PnP collection attempts",
                        "nonofficial 3+4+1 state contract retained by prior gate approval",
                        "G0/G1 training sample exposure and RNG paths differ",
                        "A differs in fine-tuning data and normalization",
                        "G2 synchronous replan only, no asynchronous RTC"],
    }
    with out.open("x") as f:
        json.dump(protocol, f, indent=2)
        f.write("\n")
    print("PROTOCOL_FROZEN", sha(out), flush=True)


if __name__ == "__main__":
    main()
