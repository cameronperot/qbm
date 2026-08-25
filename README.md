# Quantum Boltzmann Machines

[![CI](https://github.com/cameronperot/qbm/actions/workflows/ci.yml/badge.svg)](https://github.com/cameronperot/qbm/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-3f6ec6)](https://cameronperot.github.io/qbm/)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.13-3776ab)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

The `qbm` Python package is designed for training and analyzing quantum Boltzmann machines (QBMs) using either a simulation or a D-Wave quantum annealer.
A QBM is an energy-based generative model built on a quantum mechanical Hamiltonian rather than a classical energy function, allowing it to capture correlations that a classical model can't express.
The QBM implemented here is the bound-based quantum restricted Boltzmann machine (BQRBM), which has intra-layer restrictions and is trained via maximization of the log-likelihood lower bound, based on the work in *Quantum Boltzman Machine* by Amin et al. [[1]](#1).
This package originated as part of the master's thesis [*Quantum Boltzmann Machines: Applications in Quantitative Finance*](https://arxiv.org/abs/2301.13295) by Cameron Perot [[2]](#2).
An overview of the underlying theory is provided in the [documentation](https://cameronperot.github.io/qbm/theory/).

## Table of Contents
* [Installation](#installation)
* [Quickstart](#quickstart)
* [Usage](#usage)
    * [BQRBM Model](#bqrbm-model)
        * [Initialization](#initialization)
        * [Training](#training)
        * [Sampling](#sampling)
        * [Saving and Loading](#saving-and-loading)
    * [Example](#example)
* [Development](#development)
* [References](#references)

## Installation
The `qbm` package can be installed with
```
pip install qbm
```
The package requires Python 3.13 or later.
The simulation backend runs on any machine with no quantum hardware required, while the annealer backend additionally requires access to a D-Wave quantum annealer via the [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/).

The model is based on the transverse-field Ising Hamiltonian

```math
H = -A \sum_i \sigma_x^{(i)} + B \left( \sum_i h_i \sigma_z^{(i)} + \sum_{i < j} J_{ij} \sigma_z^{(i)} \sigma_z^{(j)} \right)
```

where $A$ and $B$ are the anneal schedule coefficients at the freeze-out point, $h_i$ are the local fields, $J_{ij}$ are the couplings, and $\sigma_x^{(i)}$ and $\sigma_z^{(i)}$ are Pauli operators acting on qubit $i$.

## Quickstart
The following example trains a BQRBM on randomly generated spin vectors using the simulation backend, which computes the Gibbs state exactly and therefore requires no access to quantum hardware.
With `A_freeze > 0` the model has a nonzero transverse field, i.e., the trained model is genuinely quantum.
```
import numpy as np

from qbm.models import BQRBM
from qbm.utils import get_rng

rng = get_rng(42)
V_train = rng.choice([-1, 1], size=(100, 4))

model = BQRBM(
    V_train,
    n_hidden=2,
    A_freeze=0.1,
    B_freeze=1.0,
    beta_initial=1.0,
    simulation_params={"beta": 1.0},
    seed=0,
)
model.train(
    n_epochs=100,
    learning_rate=1e-1,
    learning_rate_beta=1e-1,
    mini_batch_size=10,
    n_samples=10_000,
)
samples = model.sample(10_000)
```
More details on model configuration, as well as a guide for training with a D-Wave quantum annealer, can be found in the [documentation](https://cameronperot.github.io/qbm/).

## Usage

### BQRBM Model
The BQRBM, or bound-based quantum restricted Boltzmann machine, is a quantum Boltzmann machine that has intra-layer restrictions and is trained via maximization of the log-likelihood lower bound.
The model currently only has the ability to train in the specific case where `s_freeze = 1`, i.e., when it reduces to a classical RBM trained with quantum assistance, because estimating the effective inverse temperature is nontrivial for the general case.

All of the arguments to the methods below are further explained in their respective docstrings.

#### Initialization
A BQRBM model can be instantiated as (for example)
```
model = BQRBM(
    V_train,
    n_hidden,
    A_freeze,
    B_freeze,
    beta_initial=1.0,
    simulation_params={"beta": 1.0},
    seed=0,
)
```
The training data `V_train` is an array of visible vectors with shape `(n_samples, n_visible)`, whose values must be in `{+1, -1}` (spin eigenvalues) or `{0, 1}`, with the latter being converted internally.
One needs to choose whether or not they want to train a model using a simulation or an annealer, and this is done by passing either `simulation_params` or `annealer_params`.
Whichever is passed decides how the model is trained, and passing both or neither raises a `ValueError`.
The simulation works by exact computation of the density matrix ρ = e^{-β * H} / Z, whereas the annealer requires the additional keys
* `schedule`: List of `(t, s)` tuples defining the anneal schedule.
* `embedding`: Dict mapping the logical to physical qubits.
* `relative_chain_strength` [optional]: Relative chain strength value.
* `qpu_params` [optional]: Parameters dict to unpack to `DWaveSampler()`, e.g. `{"region": "na-west-1", "solver": "Advantage_system4.1"}`.

A guide for setting up the annealer backend can be found in the [documentation](https://cameronperot.github.io/qbm/guides/annealer/).

#### Training
The model can be trained by running
```
model.train(
    n_epochs=100,
    learning_rate=1e-1,
    learning_rate_beta=1e-1,
    mini_batch_size=10,
    n_samples=10_000,
    callback=None,
)
```
The effective inverse temperature β is treated as a learnable parameter and updated with `learning_rate_beta` at the end of each epoch.
The learning rates can either be floats, or lists/arrays of length `n_epochs` representing the learning rate over the epochs.

#### Sampling
The model can generate samples by running
```
model.sample(
    n_samples,
    answer_mode="raw",
    use_gauge=True,
    binary=False,
)
```
Samples are returned either as a dict with the energies, probabilities, and states (simulation), or as a `dimod.SampleSet` (annealer).

#### Saving and Loading
The model can be saved with
```
model.save("/path/to/model.pkl")
```
and loaded again with
```
model = BQRBM.load("/path/to/model.pkl")
```
Note that one should always use these methods rather than pickling the model directly, because the annealer's qpu and sampler objects do not support standard pickling.

## Example
An example notebook comparing the simulation and annealer backends can be found [here](https://github.com/cameronperot/qbm/blob/master/docs/example/qbm_example.ipynb), and is also rendered in the [documentation](https://cameronperot.github.io/qbm/example/qbm_example/).

## Development
Set up the development environment with
```
uv sync --locked
```

The following commands are available for development:
* Tests: `uv run pytest`
* Lint: `uv run ruff check .`
* Format: `uv run ruff format .`
* Type check: `uv run ty check`
* Pre-commit hooks: `uv run pre-commit run --all-files`
* Documentation: `uv run mkdocs build --strict`
* Add a dependency: `uv add <package>`

## Citation

If you use this package in your research, please cite the original work:

```bibtex
@article{amin2018quantum,
    author = {Amin, Mohammad H. and Andriyash, Evgeny and Rolfe, Jacob and Kulchytskyy, Bohdan and Melko, Roger},
    title = {Quantum {B}oltzmann {M}achine},
    journal = {Physical Review X},
    volume = {8},
    number = {2},
    pages = {021050},
    year = {2018},
    doi = {10.1103/PhysRevX.8.021050},
}
```

As well as the master's thesis this package originated from:

```bibtex
@mastersthesis{perot2022quantum,
    author = {Perot, Cameron},
    title = {Quantum {B}oltzmann {M}achines: Applications in Quantitative {F}inance},
    school = {RWTH Aachen University},
    year = {2022},
    url = {https://arxiv.org/abs/2301.13295},
}
```

## License

This project is licensed under the [MIT License](LICENSE).

# References

<a name="1">[1]</a> Mohammad H. Amin et al. “Quantum Boltzmann Machine”. In: Phys. Rev. X 8 (2 May 2018), p. 021050. doi: 10.1103/PhysRevX.8.021050. url: [https://link.aps.org/doi/10.1103/PhysRevX.8.021050](https://link.aps.org/doi/10.1103/PhysRevX.8.021050).

<a name="2">[2]</a> Cameron Perot. “Quantum Boltzmann Machines: Applications in Quantitative Finance”. Master's Thesis, RWTH Aachen University, 2022. url: [https://arxiv.org/abs/2301.13295](https://arxiv.org/abs/2301.13295).
