"""Check accumulated NNX gradients and AdamW update against a full batch on CPU."""
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pnp_grad_accum import accumulate_gradients
from train_pnp_grad_accum import train_step
from openpi.training import utils


class TinyModel(nnx.Module):
    def __init__(self):
        self.kernel = nnx.Param(jnp.array([[0.2, -0.1], [0.3, 0.4]]))

    def train(self):
        pass

    def compute_loss(self, rng, observation, actions, train=True):
        return (observation["x"] @ self.kernel.value - actions) ** 2


def assert_close(left, right):
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for a, b in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


def main():
    model = TinyModel()
    graph, params = nnx.split(model)
    batch = ({"x": jnp.arange(64, dtype=jnp.float32).reshape(32, 2) / 64, "optional": None},
             jnp.ones((32, 2)))
    rng = jax.random.key(42)

    def loss_and_grad(key, observation, actions):
        local = nnx.merge(graph, params)
        return nnx.value_and_grad(lambda m: jnp.mean(m.compute_loss(key, observation, actions)))(local)

    expected_loss, expected_grads = loss_and_grad(rng, *batch)
    for micro in (8, 16, 32):
        actual = jax.jit(lambda b: accumulate_gradients(loss_and_grad, b, rng, micro))(batch)
        assert_close(actual, (expected_loss, expected_grads))

    # Compare the actual training adapter, including clipping, optimizer count and EMA.
    tx = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3))
    state = utils.TrainState(step=jnp.array(0), params=params, model_def=graph, tx=tx,
                             opt_state=tx.init(params), ema_decay=0.9, ema_params=params)
    actual_state, info = jax.jit(lambda s, b: train_step(types.SimpleNamespace(trainable_filter=nnx.Param), rng, s, b))(state, batch)
    updates, expected_opt = tx.update(expected_grads, state.opt_state, params)
    expected_params = optax.apply_updates(params, updates)
    assert_close(actual_state.params, expected_params)
    assert_close(actual_state.opt_state, expected_opt)
    assert_close(actual_state.ema_params, jax.tree.map(lambda a, b: .9 * a + .1 * b, params, expected_params))
    assert_close(info["loss"], expected_loss)
    assert_close(info["grad_norm"], optax.global_norm(expected_grads))
    assert int(actual_state.step) == 1
    try:
        accumulate_gradients(loss_and_grad, batch, rng, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid microbatch accepted")
    print("PASS: micro8/16/32 = full batch; adapter clips/updates/EMA once; invalid size rejected")


if __name__ == "__main__":
    main()
