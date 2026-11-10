#!/usr/bin/env python3
"""Export PnP offline W&B loss without syncing or changing training artifacts."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re

import matplotlib
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = pathlib.Path("/root/autodl-tmp/openpi")
    logs = root.parent / "logs"
    out = logs / "plots"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for tag, exp, logname in [
        ("g0_native", "pnp_g0_bs32", "pnp-g0-train.log"),
        ("g0_mb16_failed", "pnp_g0_accum_bs32_mb16", "pnp-g0-accum-train.log"),
        ("g0", "pnp_g0_accum_bs32_mb8", "pnp-g0-accum-mb8-train.log"),
        ("g1", "pnp_g1_bs16", "pnp-g1-train.log"),
    ]:
        run_id = (root / "checkpoints/pi05_libero_lora_pnp" / exp / "wandb_id.txt").read_text().strip()
        sources = sorted({p.resolve() for p in (root / "wandb").glob(f"*/run-{run_id}.wandb")})
        if len(sources) != 1:
            raise ValueError(f"Expected one W&B source for {tag}: {sources}")
        store = DataStore()
        store.open_for_scan(str(sources[0]))
        values = {}
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField("history"):
                row = {
                    item.key or "/".join(item.nested_key): json.loads(item.value_json)
                    for item in record.history.item
                }
                if "loss" in row:
                    step, loss = int(row["_step"]), float(row["loss"])
                    assert math.isfinite(loss)
                    values[step] = loss
        rows = sorted(values.items())
        logged = {}
        for match in re.finditer(r"Step\s+(\d+):[^\n]*?\bloss=([0-9.eE+-]+)", (logs / logname).read_text(errors="replace")):
            logged[int(match[1])] = float(match[2])
        assert set(values) == set(logged), (tag, "W&B/log step mismatch")
        assert all(abs(loss - logged[step]) <= 5.1e-5 for step, loss in rows)
        with (out / f"pnp_{tag}_wandb_loss.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "loss"])
            writer.writerows(rows)
        if rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(*zip(*rows), linewidth=1.2)
            ax.set(xlabel="Training step", ylabel="Logged mean loss", title=f"PnP {tag.upper()} offline W&B loss (interval=50)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"pnp_{tag}_wandb_loss_curve.png", dpi=150)
            plt.close(fig)
        provenance[tag] = {
            "run_id": run_id, "source": str(sources[0]), "points": len(rows),
            "first": rows[0] if rows else None, "last": rows[-1] if rows else None,
            "log_crosscheck": "passed; rounded log tolerance=0.000051",
            "note": "Logged interval means, every 50 optimizer updates; last 49 updates not logged. Native G0 and micro16 x2: OOM before first loss; G0 replacement: micro8 x4 effective batch32.",
        }
    (out / "pnp-loss-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
