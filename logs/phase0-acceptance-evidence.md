# Phase 0 acceptance evidence supplement (post-reviewer 需改)

- Full log local: `logs/eval-spatial-full.log`
- MD5: `824c7a42734bd292440ee0dd87eca4e7` (matches remote)
- Size: 150722 bytes, 3612 lines
- Videos remote: `/root/autodl-tmp/logs/videos_spatial` (13 files; 3 *failure* + 10 *success* rollouts)

## Parsed episode stats

```
episodes_scored=500 last_eps=500 last_succ=495
successes=495 failures=5
task	n	successes	rate	fail_episodes
pick up the black bowl between the plate and the ramekin and place it on the plate	50	50	100.0%	[]
pick up the black bowl next to the ramekin and place it on the plate	50	50	100.0%	[]
pick up the black bowl from table center and place it on the plate	50	50	100.0%	[]
pick up the black bowl on the cookie box and place it on the plate	50	49	98.0%	[8]
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate	50	48	96.0%	[26, 31]
pick up the black bowl on the ramekin and place it on the plate	50	50	100.0%	[]
pick up the black bowl next to the cookie box and place it on the plate	50	50	100.0%	[]
pick up the black bowl on the stove and place it on the plate	50	50	100.0%	[]
pick up the black bowl next to the plate and place it on the plate	50	50	100.0%	[]
pick up the black bowl on the wooden cabinet and place it on the plate	50	48	96.0%	[23, 44]

FAIL detail:
  task='pick up the black bowl on the cookie box and place it on the plate' episode=8
  task='pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate' episode=26
  task='pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate' episode=31
  task='pick up the black bowl on the wooden cabinet and place it on the plate' episode=23
  task='pick up the black bowl on the wooden cabinet and place it on the plate' episode=44

completeness:
  pick up the black bowl between the plate and the ramekin and: n=50 (expect 50)
  pick up the black bowl next to the ramekin and place it on t: n=50 (expect 50)
  pick up the black bowl from table center and place it on the: n=50 (expect 50)
  pick up the black bowl on the cookie box and place it on the: n=50 (expect 50)
  pick up the black bowl in the top drawer of the wooden cabin: n=50 (expect 50)
  pick up the black bowl on the ramekin and place it on the pl: n=50 (expect 50)
  pick up the black bowl next to the cookie box and place it o: n=50 (expect 50)
  pick up the black bowl on the stove and place it on the plat: n=50 (expect 50)
  pick up the black bowl next to the plate and place it on the: n=50 (expect 50)
  pick up the black bowl on the wooden cabinet and place it on: n=50 (expect 50)
```
