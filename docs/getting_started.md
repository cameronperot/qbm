# Getting Started

This page walks through installing the `qbm` package and training a BQRBM model from start to
The example uses the simulation backend, which computes the Gibbs state exactly and therefore requires no access to quantum hardware.
For an overview of the theory behind the model, see the [Theory](theory.md) page.

## Installation

The `qbm` package can be installed with
```
pip install qbm
```
It requires Python 3.13 or newer.

## Training a BQRBM

The model is trained on a data set of visible vectors $V_{\text{train}}$, where each row is one training sample and each column corresponds to a visible unit.
The values must be in $\{+1, -1\}$ (spin eigenvalues) or $\{0, 1\}$, with the latter being converted internally.
Here we generate a toy data set of random spin vectors.
```
from qbm.models import BQRBM
from qbm.utils import get_rng

rng = get_rng(42)
V_train = rng.choice([-1, 1], size=(100, 4))
```

A model is initialized by passing the training data, the number of hidden units, and the freeze-out parameters `A_freeze` and `B_freeze`.
One needs to choose whether or not they want to train a model using a simulation or an annealer, and this is done by passing either `simulation_params` or `annealer_params`, with whichever is passed deciding how the samples are generated.
```
model = BQRBM(
    V_train,
    n_hidden=2,
    A_freeze=0.1,
    B_freeze=1.0,
    beta_initial=1.0,
    simulation_params={"beta": 1.0},
    seed=0,
)
```

The model is then trained with mini-batch gradient ascent on the log-likelihood lower bound, where the effective inverse temperature β is treated as a learnable parameter and updated at the end of each epoch.
```
model.train(
    n_epochs=100,
    learning_rate=1e-1,
    learning_rate_beta=1e-1,
    mini_batch_size=10,
    n_samples=10_000,
)
```

Once trained, samples can be generated from the model distribution.
The simulation backend returns a dict with the energies `E`, probabilities `p`, states `states`, and state vectors `state_vectors`, whereas the annealer backend returns a `dimod.SampleSet` object.
```
samples = model.sample(10_000)
```

Finally, a model can be saved to and loaded from disk.
```
model.save("/path/to/model.pkl")
model = BQRBM.load("/path/to/model.pkl")
```

## Next Steps

* The [Annealer](guides/annealer.md) guide explains how to set up and train a model on a D-Wave quantum annealer.
* The [Utilities](guides/utils.md) guide explains how to prepare continuous data for training.
* The [example notebook](example/qbm_example.ipynb) compares the simulation and annealer backends on the same data set.
* The [Theory](theory.md) page provides an overview of the theory behind quantum Boltzmann machines.
* The [API Reference](api/models.md) contains the full API documentation generated from the docstrings.
