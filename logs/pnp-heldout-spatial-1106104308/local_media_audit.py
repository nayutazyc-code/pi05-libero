"""Verify transferred videos locally and retain existing training plots."""
import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys

from PIL import Image

root = pathlib.Path(sys.argv[1])
plots = root.parent / "plots"
video_audit = json.loads((root / "video-audit.json").read_text())
videos = []
for stage, details in video_audit.items():
    directory = "videos" if stage == "formal" else "preflight-videos"
    for entry in details["files"]:
        path = root / directory / pathlib.PurePosixPath(entry["file"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                                 "-i", str(path), "-map", "0:v:0", "-threads", "1", "-f", "null", "-",
                                 "-progress", "pipe:1", "-nostats"], capture_output=True, text=True)
        assert result.returncode == 0, (str(path), result.stderr)
        frames = [int(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("frame=")]
        assert frames and frames[-1] == entry["frames_decoded"]
        videos.append({"path": str(path), "frames_decoded": frames[-1], "stage": stage})
csvs, images = [], []
for path in sorted(plots.glob("pnp*loss.csv")):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    assert steps == sorted(set(steps))
    assert all(math.isfinite(float(row["loss"])) for row in rows)
    csvs.append({"file": str(path), "points": len(rows), "strictly_increasing_unique_steps": True,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
provenance = json.loads((plots / "pnp-loss-provenance.json").read_text())
for group in ("g0", "g1"):
    path = plots / ("pnp_%s_wandb_loss.csv" % group)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == list(range(0, 2500, 50))
    assert provenance[group]["points"] == len(rows) == 50
    assert [int(rows[0]["step"]), float(rows[0]["loss"])] == provenance[group]["first"]
    assert [int(rows[-1]["step"]), float(rows[-1]["loss"])] == provenance[group]["last"]
for path in sorted(plots.glob("pnp*loss_curve.png")):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        assert image.width > 0 and image.height > 0
        images.append({"file": str(path), "size": image.size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(images) == 3
report = {"videos_decoded_locally": len(videos), "frames_decoded": sum(v["frames_decoded"] for v in videos),
          "formal_videos": sum(v["stage"] == "formal" for v in videos),
          "preflight_videos": sum(v["stage"] == "preflight" for v in videos),
          "loss_csvs": csvs, "loss_images": images,
          "canonical_g0_g1_points": 50, "canonical_steps": "0..2450 every 50 updates",
          "empty_csv_disclosure": "Historical failed native G0/micro16 runs have header-only CSVs; successful G0 is pnp_g0_wandb_loss.csv.",
          "all_checks_passed": True}
print(json.dumps(report, indent=2))
