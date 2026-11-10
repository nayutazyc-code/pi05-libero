#!/usr/bin/env python3
"""Use the existing trained policy with reproducible per-request RNG keys."""
import argparse
import hashlib
import json
import logging
import pathlib

import jax
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config


class SeededPolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, obs):
        obs = dict(obs)
        seed = int(obs.pop("heldout_rng_seed"))
        self.policy._rng = jax.random.key(seed)
        return self.policy.infer(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    raw = pathlib.Path(args.protocol).read_bytes()
    protocol = json.loads(raw)
    group = protocol["groups"][args.group]
    cfg = config.get_config(group["config"])
    assert cfg.model.action_horizon == 10
    policy = policy_config.create_trained_policy(cfg, group["checkpoint"])
    websocket_policy_server.WebsocketPolicyServer(
        policy=SeededPolicy(policy), host="127.0.0.1", port=args.port,
        metadata={"heldout_protocol_sha256": hashlib.sha256(raw).hexdigest(),
                  "heldout_group": args.group, "model_metadata": policy.metadata},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
