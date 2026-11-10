"""Decode every saved representative video, including all frames."""
import collections
import hashlib
import json
import pathlib
import sys

import imageio

root = pathlib.Path(sys.argv[1])
report = {}
for stage, event_dir, video_dir in (("formal", "trials", "videos"),
                                    ("preflight", "preflight-trials", "preflight-videos")):
    records = [json.loads(p.read_text()) for p in (root / event_dir).glob("*/task*/*_result.json")]
    wanted, covered = set(), set()
    paths = []
    for record in records:
        key = record["group"], record["task_id"], record["success"]
        wanted.add(key)
        if record["video"]:
            assert key not in covered
            covered.add(key)
            paths.append(pathlib.Path(record["video"]))
    assert covered == wanted, "missing representative outcome video"
    assert len(set(paths)) == len(paths)
    assert set(paths) == set((root / video_dir).glob("*.mp4")), "orphan or missing video"
    videos = []
    for path in sorted(paths):
        reader = imageio.get_reader(str(path))
        count = 0
        try:
            for frame in reader:
                assert frame.shape == (256, 256, 3)
                count += 1
        finally:
            reader.close()
        assert count > 0
        videos.append({"file": str(path), "frames_decoded": count,
                       "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report[stage] = {"videos": len(videos), "all_frames_decoded": sum(v["frames_decoded"] for v in videos),
                     "outcome_coverage_complete": True, "files": videos}
(root / "video-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({stage: {k: v for k, v in data.items() if k != "files"} for stage, data in report.items()}, indent=2))
