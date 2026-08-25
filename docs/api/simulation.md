# Simulation

The simulation module contains the functions used by the simulation backend to construct the transverse Ising Hamiltonian $H$ and compute the Gibbs state density matrix $\rho = e^{-\beta H} / \mathcal{Z}$ exactly over the full $2^{n_{\text{qubits}}}$-dimensional Hilbert space.
Note that the matrices scale exponentially in the number of qubits, so this backend is only feasible for small models.
The Hamiltonian formulation is explained in more detail on the [Theory](../theory.md) page.

::: qbm.simulation
