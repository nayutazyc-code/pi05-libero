"""Mean microbatch gradients, before clipping and a single optimizer update."""
import jax


def accumulate_gradients(loss_and_grad, batch, rng, microbatch_size):
    size = batch[1].shape[0]
    if microbatch_size < 1 or size % microbatch_size:
        raise ValueError("Effective batch must be divisible by microbatch size")
    count = size // microbatch_size

    def compute(index):
        microbatch = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index * microbatch_size, microbatch_size), batch
        )
        return loss_and_grad(jax.random.fold_in(rng, index), *microbatch)

    def add(index, total):
        return jax.tree.map(lambda a, b: a + b, total, compute(index))

    total = jax.lax.fori_loop(1, count, add, compute(0))
    return jax.tree.map(lambda x: x / count, total)
