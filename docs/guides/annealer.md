# Annealer Guide

This page explains how to set up and train a BQRBM on a D-Wave quantum annealer.
Unlike the simulation backend, the annealer backend requires access to D-Wave hardware, which can be obtained through a [D-Wave Leap](https://www.dwavequantum.com/solutions-and-products/cloud-platform/) account.
The annealer generates the samples in the negative phase of the training, and can in principle sample from genuinely quantum distributions by freezing the anneal partway through.

## Freeze-Out Parameters

A BQRBM is mapped onto the annealer at the freeze-out point $s^*$ of the anneal schedule, where the anneal schedule is defined by the energy scales $A(s)$ and $B(s)$ of the annealer.
The values $A(s^*)$ and $B(s^*)$ (in GHz) are passed to the model as `A_freeze` and `B_freeze`.
They can be read off the anneal schedule data of the target QPU, which can be found in the [D-Wave documentation](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specific.html).
```
A_freeze = df_anneal.loc[s_freeze, "A(s) (GHz)"]
B_freeze = df_anneal.loc[s_freeze, "B(s) (GHz)"]
```
The model currently only has the ability to train in the specific case where `s_freeze = 1`, i.e., when it reduces to a classical RBM trained with quantum assistance, because estimating the effective inverse temperature is nontrivial for the general case.

## Annealer Parameters

The annealer is configured via the `annealer_params` dict, which requires the keys
* `schedule`: List of `(t, s)` tuples defining the anneal schedule.
* `embedding`: Dict mapping the logical to physical qubits.
* `relative_chain_strength` [optional]: Relative chain strength value.
* `qpu_params` [optional]: Parameters dict to unpack to `DWaveSampler()`, e.g. `{"region": "na-west-1", "solver": "Advantage_system4.1"}`.

### Anneal Schedule

The `schedule` is passed to the annealer as the `anneal_schedule` parameter, and typically consists of an anneal to the freeze-out point $s^*$ followed by a fast quench to $s = 1$.
For `s_freeze = 1` a simple anneal suffices, e.g.
```
t_r = 20
anneal_schedule = [(0, 0), (t_r, 1)]
```

### Embedding

The BQRBM is a fully connected bipartite graph, which must be embedded into the QPU's hardware graph via minor embedding.
An embedding can be generated with `minorminer.find_embedding`, as documented in the [D-Wave Ocean SDK](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/minorminer.find_embedding.html), and maps each logical qubit onto one or more physical qubits.
Physical qubits representing the same logical qubit form a chain, and the `relative_chain_strength` scales the coupling within the chains relative to the largest learned $h_i$ and $J_{ij}$ values.

### QPU Parameters

The `qpu_params` dict is unpacked to `DWaveSampler()`, and can be used e.g. to select the region and solver.
The allowed $h$ and $J$ ranges are read from the QPU's properties, and the model raises a `ValueError` if training pushes the learned values outside of these ranges.

## Training

Once configured, training with the annealer backend works exactly as with the simulation backend.
Because the annealer is a real-world physical device with an unknown effective temperature, the effective inverse temperature β is treated as a learnable parameter and updated with `learning_rate_beta` at the end of each epoch.
It might be useful to use a larger `learning_rate_beta` in the beginning to help the model find a good temperature, then drop it after a number of epochs, e.g. with the exponential decay schedule provided by `qbm.utils.compute_lr_exp_decay`.
```
model = BQRBM(
    V_train,
    n_hidden,
    A_freeze,
    B_freeze,
    beta_initial=0.5,
    annealer_params=annealer_params,
)
model.train(
    n_epochs=100,
    learning_rate=learning_rates,
    learning_rate_beta=learning_rates_beta,
    mini_batch_size=10,
    n_samples=10_000,
)
```
During sampling the model applies a random gauge transformation to the $h_i$ and $J_{ij}$ values (and reverses it on the returned samples), which is recommended for more robust sample generation and can be disabled with `use_gauge=False`.
A full example comparing the simulation and annealer backends on the same data set can be found in the [example notebook](../example/qbm_example.ipynb).
