# Quantum Boltzmann Machines

The `qbm` Python package is designed for training and analyzing quantum Boltzmann machines (QBMs) using either a simulation or a D-Wave quantum annealer.
A QBM is an energy-based generative model built on a quantum mechanical Hamiltonian rather than a classical energy function, allowing it to capture correlations that a classical model can't express.
The QBM implemented here is the bound-based quantum restricted Boltzmann machine (BQRBM), which has intra-layer restrictions and is trained via maximization of the log-likelihood lower bound, based on the work in *Quantum Boltzman Machine* by Amin et al. [[1]](#1).
This package originated as part of the master's thesis [*Quantum Boltzmann Machines: Applications in Quantitative Finance*](https://arxiv.org/abs/2301.13295) by Cameron Perot [[2]](#2).
An overview of the underlying theory is provided in the [documentation](https://cameronperot.github.io/qbm/theory/).

# References
<a name="1">[1]</a> Mohammad H. Amin et al. “Quantum Boltzmann Machine”. In: Phys. Rev. X 8 (2 May 2018), p. 021050. doi: 10.1103/PhysRevX.8.021050. url: [https://link.aps.org/doi/10.1103/PhysRevX.8.021050](https://link.aps.org/doi/10.1103/PhysRevX.8.021050).

<a name="2">[2]</a> Cameron Perot. “Quantum Boltzmann Machines: Applications in Quantitative Finance”. Master's Thesis, RWTH Aachen University, 2022. url: [https://arxiv.org/abs/2301.13295](https://arxiv.org/abs/2301.13295).
