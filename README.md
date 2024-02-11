# Quantum Boltzmann Machines
The `qbm` Python package is designed for training and analyzing quantum Boltzmann machines (QBMs) using either a simulation or a D-Wave quantum annealer.
The QBM implemented here is based on the work in *Quantum Boltzman Machine* by Amin et al. [[1]](#1).
This package originated as part of the thesis [*Quantum Boltzmann Machines: Applications in Quantitative Finance*](https://github.com/cameronperot/qbm-quant-finance).

## Table of Contents
* [Installation](#installation)
    * [Conda Environment](#conda-environment)
* [Usage](#usage)
    * [Basic Configuration](#basic-configuration)
    * [BQRBM Model](#bqrbm-model)
        * [Instantiation](#instantiation)
        * [Training](#training)
        * [Sampling](#sampling)
        * [Saving and Loading](#saving-and-loading)
    * [Example](#example)
* [References](#references)

## Installation
The `qbm` package can be installed with
```
pip install qbm
```

### Conda Environment
A predefined conda environment is already configured and ready for installation.
This can be installed by running
```
conda env create -f environment.yml
```

Extra dev dependencies can be installed with
```
conda env update --file environment-dev.yml
```

## Usage

### Basic Configuration
The `qbm` package is mainly configured around the project directory, which can be set with the `QBM_PROJECT_DIR` environment variable.
Once the environment variable is set one can use the `qbm.utils.get_project_dir()` function to get a path object to the directory.

### BQRBM Model
The BQRBM, or bound-based quantum restricted Boltzmann machine, is a quantum Boltzmann machine that has intra-layer restrictions and is trained via maximization of the log-likelihood lower bound.
The model currently only has the ability to train in the specific case where `s_freeze = 1`, i.e., when it reduces to a classical RBM trained with quantum assistance, because estimating the effective inverse temperature is nontrivial for the general case.

All of the arguments to the methods below are further explained in their respective docstrings.

#### Instantiation
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
One needs to choose whether or not they want to train a model using a simulation or an annealer, and this is done by passing either `simulation_params` or `annealer_params`.
Whichever is passed decides how the model is trained.

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

#### Saving and Loading
The model can be saved with
```
model.save("/path/to/model.pkl")
```
and loaded again with
```
model = BQRBM.load("/path/to/model.pkl")
```

## Example
An example notebook can be found [here](example/qbm_example.ipynb)

# References
<a name="1">[1]</a> Mohammad H. Amin et al. “Quantum Boltzmann Machine”. In: Phys. Rev. X 8 (2 May 2018), p. 021050. doi: 10.1103/PhysRevX.8.021050. url: [https://link.aps.org/doi/10.1103/PhysRevX.8.021050](https://link.aps.org/doi/10.1103/PhysRevX.8.021050).
