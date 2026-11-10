"""PnP-only adapter for upstream scripts/train.py; batch_size stays effective."""
import logging
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import wandb

import train as upstream
from pnp_grad_accum import accumulate_gradients


MICROBATCH_SIZE = int(os.environ.get("PNP_MICROBATCH_SIZE", "16"))
_init_wandb = upstream.init_wandb


def init_wandb(config, **kwargs):
    if MICROBATCH_SIZE < 1 or config.batch_size % MICROBATCH_SIZE:
        raise ValueError("Effective batch must be divisible by microbatch size")
    _init_wandb(config, **kwargs)
    metadata = {
        "effective_batch_size": config.batch_size,
        "microbatch_size": MICROBATCH_SIZE,
        "gradient_accumulation_steps": config.batch_size // MICROBATCH_SIZE,
        "gradient_reduction": "mean, then clip once and AdamW once",
        "step_unit": "optimizer updates",
    }
    wandb.config.update(metadata)
    logging.info("PNP_GRAD_ACCUM %s", metadata)


def train_step(config, rng, state, batch):
    def loss_and_grad(micro_rng, observation, actions):
        # Merge inside the loop so NNX mutable graph objects stay within its trace.
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model):
            return jnp.mean(model.compute_loss(micro_rng, observation, actions, train=True))

        return nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(model)

    loss, grads = accumulate_gradients(
        loss_and_grad, batch, jax.random.fold_in(rng, state.step), MICROBATCH_SIZE
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, upstream.optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = upstream.dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )
    if state.ema_decay is not None:
        new_state = upstream.dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(upstream.nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return new_state, {
        "loss": loss,
        "grad_norm": upstream.optax.global_norm(grads),
        "param_norm": upstream.optax.global_norm(kernel_params),
    }


if __name__ == "__main__":
    upstream.init_wandb = init_wandb
    upstream.train_step = train_step
    upstream.main(upstream._config.cli())
