# MIT License

# Copyright (c) 2022 Ryan Hausen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

from functools import partial
import jax.numpy as jnp
import jax.image as img
from flax import linen as nn

class Generator(nn.Module):

    @nn.compact
    def __call__(self, x:jnp.ndarray) -> jnp.ndarray:

        batch_size = x.shape[0]

        x = nn.linear.Dense(features=7*7*128)(x)
        x = nn.activation.PReLU(0.2)(x)
        x = jnp.reshape(x, [batch_size, 7, 7, 128])
        x = img.resize(x, [batch_size, 14, 14, 128], method=img.ResizeMethod.LINEAR)
        x = nn.linear.Conv(features=128, kernel_size=(3,3), padding="SAME")(x)
        x = nn.activation.PReLU(0.2)(x)
        x = img.resize(x, [batch_size, 28, 28, 128], method=img.ResizeMethod.LINEAR)
        x = nn.linear.Conv(features=128, kernel_size=(3,3), padding="SAME")(x)
        x = nn.activation.PReLU(0.2)(x)

        x = nn.linear.Conv(features=1, kernel_size=(3, 3), padding="SAME")(x)

        return x

class Discriminator(nn.Module):

    @nn.compact
    def __call__(self, x:jnp.ndarray) -> jnp.ndarray:

        conv = partial(
            nn.linear.Conv,
            kernel_size=(3,3),
            strides=(2, 2),
            padding="SAME",
        )

        x = conv(64)(x)
        x = nn.activation.PReLU()(x)
        x = nn.normalization.LayerNorm()(x)
        x = conv(64)(x)
        x = nn.activation.PReLU()(x)
        x = nn.normalization.LayerNorm()(x)
        x = jnp.reshape(x, [x.shape[0], -1])
        x = nn.linear.Dense(1)(x)

        return x




