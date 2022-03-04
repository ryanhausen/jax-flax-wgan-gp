import os
from typing import Dict, Union

import comet_ml
import gin
import jax
import jax.image as image
import jax.numpy as jnp
import jax.numpy.fft as fft
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax as opt
from flax.training.train_state import TrainState
from scipy import rand
from tqdm import tqdm

from model import Discriminator, Generator
import data_provider as dp

def config_str_to_dict(config_str: str) -> dict:
    """Converts a Gin.config_str() to a dict for logging with comet.ml"""
    params = {}
    for line in config_str.splitlines():
        if len(line) > 0 and line[0] != "#" and not line.startswith("import"):
            key, val = line.split("=")
            params[key.strip()] = val.strip()
    return params


@gin.configurable(allowlist=["generator_optimizer", "discriminator_optimizer"])
def create_training_state(rng, generator_optimizer, discriminator_optimizer):

    generator = Generator()
    g_params = generator.init(rng, jnp.ones([1, 256, 1]), jnp.ones([1, 256]))["params"]

    discriminator = Discriminator()
    d_params = discriminator.init(rng, jnp.ones([1, 28, 28, 1]))["params"]

    return (
        TrainState.create(apply_fn=generator.apply, params=g_params, tx=generator_optimizer),
        TrainState.create(apply_fn=discriminator.apply, params=d_params, tx=discriminator_optimizer),
    )


def wasserstein_gen_loss(dsc_fake):
    return -dsc_fake.flatten().mean()


def wasserstein_dsc_loss(dsc_fake, dsc_real):
    return (dsc_fake - dsc_real).flatten().mean()


# https://github.com/n2cholas/progan-flax/blob/b40425c63ab1306c3b645a5bd84bc31495846e20/src/training.py#L76
def gradient_penatly_loss(d_params, interp):
    @jax.grad
    def grad_fn(x):
        return Discriminator().apply({"params": d_params}, x).mean()

    gradient_norm = jnp.sqrt((grad_fn(interp) ** 2).sum(axis=(1, 2)))

    return jnp.mean((gradient_norm - 1) ** 2)


def generator_loss_fn(g_params, d_params, batch) -> jnp.float32:
    fake = Generator().apply(
        {"params": g_params}, batch["z"]
    )
    dsc_fake = Discriminator().apply({"params": d_params}, fake)

    loss = wasserstein_gen_loss(dsc_fake)

    return loss


@gin.configurable(allowlist=["lambda_gp"])
def discriminator_loss_fn(
    d_params, g_params, rng, batch, lambda_gp: float = 10
) -> jnp.float32:
    fake = Generator().apply({"params": g_params}, batch["z"]
)
    dsc_fake = Discriminator().apply({"params": d_params}, fake)
    dsc_real = Discriminator().apply({"params": d_params}, batch["real"])

    w_loss = wasserstein_dsc_loss(dsc_fake, dsc_real)

    epsilon = random.uniform(rng, shape=(dsc_fake.shape[0], 1, 1))
    interp = epsilon * batch["real"] + (1 - epsilon) * fake

    gradient_penalty = gradient_penatly_loss(d_params, interp)

    loss = w_loss + lambda_gp * gradient_penalty

    return loss


@jax.jit
def training_step(g_state:TrainState, d_state:TrainState, update_gen:bool, batch:Dict, rng:random.KeyArray):

    real = batch["real"]
    fake = Generator().apply({"params":g_state.params}, batch["z"])

    dsc_real = Discriminator().apply({"params":d_state.params}, real)
    dsc_fake = Discriminator().apply({"params":d_state.params}, fake)

    dsc_grad_fn = jax.value_and_grad(fun=discriminator_loss_fn)
    d_loss, d_grads = dsc_grad_fn(d_state.params, g_state.params, rng, batch)
    d_state = d_state.apply_gradients(grads=d_grads)

    metrics = dict(
        avg_real = dsc_real.mean(),
        avg_fake = dsc_fake.mean(),
        d_loss = d_loss,
    )

    if update_gen:
        gen_grad_fn = jax.value_and_grad(fun=generator_loss_fn)
        g_loss, g_grads = gen_grad_fn(g_state.params, d_state.params, batch)
        g_state = g_state.apply_gradients(grads=g_grads)
        metrics["g_loss"] = g_loss

    return g_state, d_state, metrics


@jax.jit
def test_step(g_state, d_state, batch, rng):
    real = batch["real"]
    fake = Generator().apply({"params":g_state.params}, batch["z"])

    dsc_real = Discriminator().apply({"params":d_state.params}, real)
    dsc_fake = Discriminator().apply({"params":d_state.params}, fake)

    d_loss = discriminator_loss_fn(d_state.params, g_state.params, rng, batch)
    g_loss = generator_loss_fn(g_state.params, d_state.params, batch)

    metrics = dict(
        avg_real = dsc_real.mean(),
        avg_fake = dsc_fake.mean(),
        d_loss = d_loss,
        g_loss = g_loss,
    )

    return metrics

def log_metrics(experiment:comet_ml.Experiment, metrics:Dict, step:int):
    pass

def train_epoch(experiment, g_state, d_state, n_steps_per_gen_update, epoch, data, rng, step):

    for i, batch in tqdm(
        zip(range(data.batches_per_epoch), data),
        total=data.batches_per_epoch,
        desc=f"Epoch {epoch}",
    ):
        do_gen_update = step % n_steps_per_gen_update == 0
        g_state, d_state, metrics = training_step(
            g_state,
            d_state,
            do_gen_update,
            batch,
            rng,
        )

        rng, _ = jax.random.split(rng)

        if i % 100 == 0:
            log_metrics(experiment, metrics, step)

        step += 1

    return g_state, d_state, rng, step


@gin.configurable
def main(num_epochs: int, seed: int, tags=[]):

    experiment = comet_ml.Experiment(
        api_key=os.getenv("COMET_KEY"),
        project_name="fft-gan",
        workspace="ryanhausen",
        auto_metric_logging=False,
        disabled=True,
    )
    experiment.add_tags(tags)
    experiment.log_parameters(config_str_to_dict(gin.config_str(max_line_length=1000)))

    rng = random.PRNGKey(seed)

    training_data, testing_data = dp.get_dataset()

    g_state, d_state = create_training_state(rng)

    step = 1
    for epoch in range(1, num_epochs + 1):
        rng, _ = random.split(rng)

        g_state, d_state, rng, step = train_epoch(
            experiment, g_state, d_state, epoch, training_data, rng, step
        )