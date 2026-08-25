# Models

The models module contains the `QBMBase` abstract base class, which holds the visible and hidden units, weights, biases, and random number generator, and the `BQRBM` class which implements the bound-based quantum restricted Boltzmann machine on top of it.
A `BQRBM` is trained with the `train()` method and sampled from with the `sample()` method, using either the simulation or the annealer backend depending on how it was instantiated.
See the [Getting Started](../getting_started.md) page for a complete example, and the [Annealer guide](../guides/annealer.md) for setting up the annealer backend.

::: qbm.models.QBMBase

::: qbm.models.BQRBM
