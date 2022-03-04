
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

import gin
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from sklearn.datasets import fetch_openml

class DataProvider:
    def __init__(self, data, rng, z_dim, batch_size) -> None:
        self.n = data.shape[0]
        self.rng = rng
        self.z_dim = z_dim
        self.i = 0
        self.data = jnp.reshape(data, [self.n, 28, 28, 1]) / 255 - 0.5
        self.batch_size = batch_size
        self.batches_per_epoch = self.n // batch_size

    def __next__(self,):
        current_batch_idx = self.i * self.batch_size
        real = self.data[
            current_batch_idx : current_batch_idx + self.batch_size, ...
        ]

        z = random.normal(self.rng, [real.shape[0], self.z_dim])
        self.i = (self.i + 1) % self.batches_per_epoch
        self.rng, _ = random.split(self.rng)

        return dict(
            real=real,
            z=z
        )

    def __iter__(self):
        return self


@gin.configurable(allowlist=["z_dim", "batch_size"])
def get_dataset(rng_seed:int, z_dim:int, batch_size:int):
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    return DataProvider(
        X, random.PRNGKey(rng_seed), z_dim, batch_size,
    )
