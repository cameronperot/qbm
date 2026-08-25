# Theory

This page provides an overview of the theory behind the quantum Boltzmann machine (QBM), and is based on the work in *Quantum Boltzmann Machine* by Amin et al. [1](#1), as well as *Quantum Boltzmann Machines: Applications in Quantitative Finance* by Cameron Perot [2](#2).
We start with the classical restricted Boltzmann machine (RBM) on which the QBM is based, then introduce the quantum mechanical formulation, and finally explain how a quantum annealer can be used to sample from quantum Boltzmann distributions.
Note that in the quantum sections we use spin eigenvalues $+1$ and $-1$ rather than binary values $0$ and $1$, respectively, in order to maintain consistency with the language of quantum mechanics.

## The Classical Restricted Boltzmann Machine

The restricted Boltzmann machine is an energy-based model defined by the energy function

$$
E(\mathbf{v}, \mathbf{h})
    = -\mathbf{a}^T\mathbf{v} - \mathbf{b}^T\mathbf{h} - \mathbf{v}^T \mathbf{W} \mathbf{h},
$$

where

* $\mathbf{v} \in \{0, 1\}^{n_v}$ represents the visible units, with associated bias vector $\mathbf{a} \in \mathbb{R}^{n_v}$.
* $\mathbf{h} \in \{0, 1\}^{n_h}$ represents the hidden units, with associated bias vector $\mathbf{b} \in \mathbb{R}^{n_h}$.
* $\mathbf{W} \in \mathbb{R}^{n_v \times n_h}$ represents the weights corresponding to the interaction strengths between visible and hidden units.

It is termed *restricted* due to the fact that there are no intralayer connections, i.e., visible units are only connected to hidden units, and vice versa.
The probability to find the system in the configuration $(\mathbf{v}, \mathbf{h})$ is given by the Boltzmann distribution

$$
p(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} e^{-E(\mathbf{v},\mathbf{h})},
$$

with partition function

$$
Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})},
$$

where $\sum_{\mathbf{v},\mathbf{h}}$ denotes the sum over all possible configurations of $\mathbf{v}$ and $\mathbf{h}$.
This partition function is intractable in general, as it requires a sum over an exponential number of configurations.

The imposed restrictions on intralayer connections enable us to write the conditional probabilities of the layers as the product of the individual units' probabilities, e.g.

$$
p(\mathbf{h} | \mathbf{v}) = \prod_{j=1}^{n_h} \sigma\big((2\mathbf{h} - 1) \odot (\mathbf{b} + \mathbf{W}^T\mathbf{v})\big)_j,
$$

where $\sigma(x)$ is the element-wise logistic sigmoid function and $\odot$ denotes element-wise multiplication.

### Optimizing an RBM

Due to the intractability of the partition function, the model cannot be solved exactly in general, thus we resort to other methods to optimize it such as likelihood maximization via gradient ascent.
For data set distribution $p_\text{data}$ and parameters $\theta = (\mathbf{W}, \mathbf{a}, \mathbf{b})$, the log-likelihood is given by

$$
\ell(\theta) = \sum_{\mathbf{v}} p_{\text{data}}(\mathbf{v}) \log p(\mathbf{v}),
$$

with gradients

$$
\begin{align}
    \partial_{w_{ij}} \ell(\theta)
        &= \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}, \\
    \partial_{a_i} \ell(\theta)
        &= \langle v_i \rangle_{\text{data}} - \langle v_i \rangle_{\text{model}}, \\
    \partial_{b_j} \ell(\theta)
        &= \langle h_j \rangle_{\text{data}} - \langle h_j \rangle_{\text{model}}.
\end{align}
$$

The part of the gradient under the data set distribution is referred to as the *positive* phase, and the part under the model distribution is referred to as the *negative* phase.
It is trivial to compute the expectation values in the positive phase, but not so much in the negative phase because $p(\mathbf{v})$ cannot be sampled directly.
In practice the negative phase expectation values are sampled using a Markov chain Monte Carlo method via Gibbs sampling [3](#3), which uses the conditional probabilities $p(\mathbf{h}|\mathbf{v})$ and $p(\mathbf{v}|\mathbf{h})$.
One starts with a visible vector and then samples the hidden units conditioned on the visible units, followed by sampling the visible units conditioned on the hidden units, and so forth until the desired thermalization threshold is reached.

## The Quantum Boltzmann Machine

We start with the $n$-qubit Hamiltonian

$$
H = -\sum_{i=1}^{n} \Gamma_i \sigma_i^x -\sum_{i=1}^{n} b_i \sigma_i^z - \sum_{i=1}^{n}\sum_{j=i+1}^{n} w_{ij} \sigma_i^z \sigma_j^z,
$$

where

$$
\begin{align}
    \sigma_i^x
        &= I^{\otimes i-1} \otimes \sigma_x \otimes I^{\otimes n-i}, \\
    \sigma_i^z
        &= I^{\otimes i-1} \otimes \sigma_z \otimes I^{\otimes n-i},
\end{align}
$$

with $\sigma_x$ and $\sigma_z$ being the Pauli $x$ and $z$ matrices, and $I$ being the $2 \times 2$ identity matrix.
We denote the first $n_v$ qubits as the visible units and the last $n_h$ qubits as the hidden units, thus we have a total of $n_v + n_h = n$ qubits.

The system's distribution is modeled by the density matrix

$$
\rho = \frac{1}{Z} e^{-H},
$$

where $e^{-H} = \sum_{n=0}^{\infty} \frac{1}{n!} (-H)^n$ is the matrix exponential, and $Z = \mathrm{tr}(e^{-H})$ is the partition function.
The marginal probability to measure the visible units in state $|\mathbf{v}\rangle$ is given by

$$
p(\mathbf{v}) = \mathrm{tr}(\Lambda_{\mathbf{v}}\rho),
$$

where $\Lambda_\mathbf{v} = |\mathbf{v}\rangle\langle\mathbf{v}| \otimes I^{\otimes n_h}$ is the projection operator onto the visible state $|\mathbf{v}\rangle$.

### Optimizing a QBM

When optimizing a QBM, it is preferable to maximize the lower bound of the log-likelihood rather than maximizing the log-likelihood itself.
The reason for this is that the partial derivative of the log-likelihood with respect to the parameters has a term which is computationally expensive to compute.
The lower bound of the log-likelihood is given by

$$
\tilde{\ell}(\theta) = \sum_{\mathbf{v}} p_{\text{data}}(\mathbf{v}) \log \mathrm{tr}(\rho_\mathbf{v}),
$$

where we have what is referred to as the *clamped* Hamiltonian, which for a given visible vector $\mathbf{v}$ is

$$
H_\mathbf{v} = \langle\mathbf{v}|H|\mathbf{v}\rangle,
$$

with corresponding clamped density matrix

$$
\rho_\mathbf{v} = \frac{1}{Z_\mathbf{v}} e^{-H_\mathbf{v}},
$$

and $Z_\mathbf{v} = \mathrm{tr}(e^{-H_\mathbf{v}})$.
This is called clamped because the visible qubits are held to the classical state of the visible vector $\mathbf{v}$.

The associated derivatives with respect to the parameters of the lower bound are given by

$$
\begin{align}
    \partial_{w_{ij}} \tilde{\ell}(\theta)
        &= \langle \sigma_i^z \sigma_j^z \rangle_\text{data} - \langle \sigma_i^z \sigma_j^z \rangle_\text{model}, \\
    \partial_{b_i} \tilde{\ell}(\theta)
        &= \langle \sigma_i^z \rangle_\text{data} - \langle \sigma_i^z \rangle_\text{model},
\end{align}
$$

where $\langle \ \cdot \ \rangle_\text{data}$ is the expectation value with respect to the data set, and $\langle \ \cdot \ \rangle_\text{model}$ is the expectation value with respect to the original density matrix.
Just like with the classical RBM, the positive phase is computable in closed form, whereas the negative phase requires samples from the model distribution.

If connections are restricted within the hidden layer, then the hidden unit probabilities are independent in the positive phase and can be computed easily.
Defining the effective hidden field $b_i'(\mathbf{v}) = b_i + (\mathbf{W}^T\mathbf{v})_i$ and $D_i(\mathbf{v}) = \sqrt{\Gamma_i^2 + b_i'(\mathbf{v})^2}$, the positive phase hidden unit expectation values take the form

$$
\langle \sigma_i^z \rangle_\text{data}
    = \sum_\mathbf{v} p_{\text{data}}(\mathbf{v}) \frac{b_i'(\mathbf{v})}{D_i(\mathbf{v})} \tanh\big(D_i(\mathbf{v})\big),
    \quad i \in \mathcal{I}_h,
$$

where $\mathcal{I}_h = \{n_v + 1, \dots, n\}$ represents the hidden qubit indices.
We call a QBM with these intra-layer restrictions which is trained via maximization of the log-likelihood lower bound a bound-based quantum restricted Boltzmann machine, or BQRBM for short.

## Quantum Annealing

Quantum annealing, also known as adiabatic quantum computing, is a branch of quantum computing that is based on the adiabatic theorem, which in the (translated) words of Born and Fock [4](#4):
"A physical system remains in its instantaneous eigenstate if a given perturbation is acting on it slowly enough and if there is a gap between the eigenvalue and the rest of the Hamiltonian's spectrum."
This can be achieved by implementing a Hamiltonian of the form

$$
H(s) = A(s) H_{\text{initial}} + B(s) H_{\text{final}},
$$

where $s \in [0, 1]$.
$H_{\text{initial}}$ is the initial Hamiltonian which describes the system at $s = 0$ and is responsible for introducing quantum fluctuations.
$H_{\text{final}}$ is the final Hamiltonian which describes the system at $s = 1$ and is responsible for encoding the problem defined by the user.

In essence, a quantum annealer starts in the ground state of the initial Hamiltonian, then slowly evolves the system over time so that it remains in the instantaneous ground state.
By the time the annealing process is completed, the Hamiltonian is just that of the problem, and if the system evolved adiabatically, then it should have remained in the instantaneous ground state.
Therefore, when the qubits are measured at the end, they should correspond to a low energy solution of the final Hamiltonian.

### Mapping the QBM to the Annealer

D-Wave quantum annealers implement a time-dependent Hamiltonian of the form

$$
H(s) = A(s) \bigg( -\sum_{i=1}^{n} \sigma_i^x \bigg) + B(s) \bigg( \sum_{i=1}^{n} h_i \sigma_i^z + \sum_{i=1}^{n}\sum_{j=i+1}^{n} J_{ij} \sigma_i^z \sigma_j^z \bigg),
$$

i.e., the final Hamiltonian corresponds to the Ising model described by the $h_i$ and $J_{ij}$ values.

In order to get a quantum annealer to sample from a quantum Boltzmann distribution, one would need to freeze the evolution at some point $s^*$ during the annealing process and then perform the measurements [1](#1).
Because a quantum annealer is a real-world physical device, samples generated with it have an associated temperature called the effective temperature, i.e., the corresponding density operator is of the form

$$
\rho(s, T) = \frac{1}{Z} e^{-\beta H(s)},
$$

where $\beta = 1/kT$ is the effective inverse temperature.
In principle, $\beta$ is an unknown quantity and must be determined in order to effectively use the annealer to generate samples from a quantum Boltzmann distribution.

Comparing the density operator of the QBM to the one above at the freeze-out point $s^*$, we find

$$
\begin{align}
    \Gamma_i
        &= \beta A(s^*), \\
    b_i
        &= -\beta B(s^*) h_i, \\
    w_{ij}
        &= -\beta B(s^*) J_{ij}.
\end{align}
$$

This enables us to map the QBM to the annealer if $\beta$ can be determined to some reasonable degree of accuracy.

Rather than having to choose a value for $\beta$ empirically, there is the possibility to treat it as a learnable parameter, as detailed by Xu and Oates [5](#5).
The method is based on a log-likelihood maximization approach leading to parameter updates of the form

$$
\Delta\hat{\beta} = \eta_{\hat{\beta}}\big(\langle E \rangle_\text{data} - \langle E \rangle_\text{model}\big),
$$

where $\hat{\beta} = 1/k\hat{T}$ is the estimator of the effective inverse temperature, and $\eta_{\hat{\beta}}$ is the associated learning rate.
We must note though, that this approach is only valid for classical Boltzmann distributions.

For the case $s^* = 1$ we have $\lim_{s \to 1} \Gamma_i = 0$, and the QBM reduces to a classical RBM trained with quantum assistance, i.e., using the annealer to generate the samples in the negative phase rather than using Gibbs sampling.
This is the regime in which the `qbm` package currently operates, because estimating the effective inverse temperature is nontrivial for the general case.

# References

<a name="1">[1]</a> Mohammad H. Amin et al. “Quantum Boltzmann Machine”. In: Phys. Rev. X 8 (2 May 2018), p. 021050. doi: 10.1103/PhysRevX.8.021050. url: [https://link.aps.org/doi/10.1103/PhysRevX.8.021050](https://link.aps.org/doi/10.1103/PhysRevX.8.021050).

<a name="2">[2]</a> Cameron Perot. “Quantum Boltzmann Machines: Applications in Quantitative Finance”. Master's Thesis, RWTH Aachen University, 2022. url: [https://arxiv.org/abs/2301.13295](https://arxiv.org/abs/2301.13295).

<a name="3">[3]</a> Geoffrey E. Hinton. “Training Products of Experts by Minimizing Contrastive Divergence”. In: Neural Computation 14 (2002), pp. 1771–1800. doi: 10.1162/089976602760128018.

<a name="4">[4]</a> Max Born and Vladimir Fock. “Beweis des Adiabatensatzes”. In: Zeitschrift für Physik 51 (1928), pp. 165–180. doi: 10.1007/BF01343193.

<a name="5">[5]</a> Junchi Xu and Andrew C. Oates. “Learning the Effective Temperature of a D-Wave Quantum Annealer”. In: 2021 IEEE International Conference on Quantum Computing and Engineering (QCE). 2021, pp. 409–415. doi: 10.1109/QCE52317.2021.00058.
