#!/usr/bin/env bash
# Parse step loss from phase1-exp-c.log and write CSV + PNG under logs/plots.
set -euo pipefail
L=${1:-/root/autodl-tmp/logs}
LOG=$L/phase1-exp-c.log
OUTDIR=$L/plots
mkdir -p "$OUTDIR"
/root/autodl-tmp/openpi/.venv/bin/python - <<PY
import re, csv
from pathlib import Path
log = Path("$LOG")
out_csv = Path("$OUTDIR/exp_c_loss.csv")
rows=[]
for line in log.read_text(errors="replace").splitlines():
    m=re.search(r"Step\s+(\d+):\s+.*?\bloss=([0-9.eE+-]+)", line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2))))
# dedupe by step keep last
d={s:l for s,l in rows}
rows=sorted(d.items())
with out_csv.open("w", newline="") as f:
    w=csv.writer(f); w.writerow(["step","loss"]); w.writerows(rows)
print("N_POINTS", len(rows), "CSV", out_csv)
if rows:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs=[r[0] for r in rows]; ys=[r[1] for r in rows]
        plt.figure(figsize=(8,4))
        plt.plot(xs, ys, lw=1.2)
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("Exp C VLM-LoRA-only train loss")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        png=Path("$OUTDIR/exp_c_loss_curve.png")
        plt.savefig(png, dpi=120); print("PNG", png)
    except Exception as e:
        print("PLOT_SKIP", e)
PY
