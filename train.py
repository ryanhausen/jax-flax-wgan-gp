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

from model import Discriminator, Generator


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

    if update_gen:
        gen_grad_fn = jax.value_and_grad(fun=generator_loss_fn)
        g_loss, g_grads = gen_grad_fn(g_state.params, d_state.params, batch)
        g_state = g_state.apply_gradients(grads=g_grads)

    return g_state, d_state, metrics