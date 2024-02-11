from datetime import timedelta
from time import time

import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from qbm.models import QBMBase
from qbm.simulation import compute_H, compute_rho, get_pauli_kron
from qbm.utils import Discretizer, load_artifact, save_artifact


class BQRBM(QBMBase):
    """
    Bound-based Quantum Restricted Boltzmann Machine
    """

    def __init__(
        self,
        V_train,
        n_hidden,
        A_freeze,
        B_freeze,
        beta_initial=1.0,
        beta_range=[0.1, 10],
        annealer_params=None,
        simulation_params=None,
        seed=0,
    ):
        """
        Note: one of either annealer_params or simulation_params must not be None. Whichever
        is not None determines whether the samples are generated by the annealer or the
        simulation, respectively.

        The simulation works by exact computation of ρ(s, T) = e^{-β * H(s)} / Z. H(s) is
        determined by A_freeze and B_freeze.

        :param V_train: Training data set of visible vectors, shape (n_samples, n_visible).
        :param n_hidden: Number of hidden units.
        :param A_freeze: Value of A(s) at the freeze out point. Used for Γ = β * A_freeze.
            Units in GHz.
        :param B_freeze: Value of B(s) at the freeze out point. Used for
            b_i = -β * B_freeze * h_i and w_ij = -β * B_freeze * J_ij. Units in GHz.
        :param beta_initial: Initial value of the effective β. Units in 1/GHz.
        :param beta_range: Range of allowed β values, used for making sure β is not updated
            to an infeasible value (e.g. negative).
        :param annealer_params: Dictionary with keys:
            - "schedule": List of (t, s) tuples defining the anneal schedule.
            - "embedding": Dict mapping the logical to physical qubits.
            - "relative_chain_strength" [optional]: Relative chain strength value.
            - "qpu_params" [optional]: Parameters dict to unpack to DWaveSampler(), e.g.
                {"region": "na-west-1", "solver": "Advantage_system4.1"}
        :param simulation_params: Dictionary with keys:
            - "beta": Effective β that the simulation generates samples at.
        :param seed: Seed for the random number generator. Used for random minibatches, as
            well as the exact sampler.
        """
        # convert from binary to ±1 if necessary
        if set(np.unique(V_train)) == set([0, 1]):
            V_train = self._binary_to_eigen(V_train)
        assert set(np.unique(V_train)) == set([-1, 1])

        self.A_freeze = A_freeze
        self.B_freeze = B_freeze
        self.beta = beta_initial
        self.beta_range = beta_range
        self.beta_history = [beta_initial]
        super().__init__(V_train=V_train, n_hidden=n_hidden, seed=seed)

        # check if requirements met to use annealer or simulation
        if (annealer_params is not None and simulation_params is not None) or (
            annealer_params is None and simulation_params is None
        ):
            raise Exception(
                "You must pass one of either annealer_params or simulation_params, not both"
            )

        elif annealer_params is not None:
            annealer_params_keys = ["schedule", "embedding"]
            for k in annealer_params_keys:
                if k not in annealer_params:
                    raise Exception(
                        f"Missing key in annealer_params. Required keys are {annealer_params_keys}"
                    )

            self.annealer_params = annealer_params
            self._initialize_annealer()

        elif simulation_params is not None:
            simulation_params_keys = ["beta"]
            for k in simulation_params_keys:
                if k not in simulation_params:
                    raise Exception(
                        f"Missing key in simulation_params. Required keys are {simulation_params_keys}"
                    )

            self.simulation_params = simulation_params
            self._pauli_kron = get_pauli_kron(self.n_visible, self.n_hidden)
            self.h_range = np.array(
                self.simulation_params.get("h_range", [-np.inf, np.inf])
            )
            self.J_range = np.array(
                self.simulation_params.get("J_range", [-np.inf, np.inf])
            )

    def sample(self, n_samples, answer_mode="raw", use_gauge=True, binary=False):
        """
        Generate samples using the model, either exact or from the annealer.

        :param n_samples: Number of samples to generate (num_reads param in sample_ising).
        :param answer_mode: "raw" or "histogram".
        :param use_gauge: If True will use a random gauge transformation (recommended for
            more robust sample generation).
        :param binary: If true will convert the state vector values from {+1, -1} to {0, 1}.

        :returns: Dictionary (exact) or Ocean SDK SampleSet object (annealer).
        """
        if hasattr(self, "simulation_params"):
            return self._sample_simulation(n_samples, binary=binary)
        else:
            return self._sample_annealer(
                n_samples, answer_mode=answer_mode, use_gauge=use_gauge, binary=binary
            )

    def train(
        self,
        n_epochs=100,
        learning_rate=1e-1,
        learning_rate_beta=1e-1,
        mini_batch_size=10,
        n_samples=10_000,
        callback=None,
    ):
        """
        Fits the model to the training data.

        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Learning rate. If a list or array, then it will represent the
            learning rate over the epochs, must be of length n_epochs.
        :param learning_rate_beta: Learning rate for the effective temperature. If a list or
            array, then it will represent the learning rate over the epochs, must be of
            length n_epochs.
            Note: It might be useful to use a larger learning_rate_beta in the beginning to
            help the model find a good temperature, then drop it after a number of epochs.
        :param mini_batch_size: Size of the mini-batches.
        :param n_samples: Number of samples to generate after every epoch. Used for
            computing β gradient, as well as the callback.
        :param callback: A function called at the end of each epoch. It takes the arguments
            (model, samples), and returns a dictionary with required keys ["value",
            "print"], where the "print" value is a string to be printed at the end of each
            epoch.
        """
        if isinstance(learning_rate, float):
            learning_rate = [learning_rate] * n_epochs
        assert len(learning_rate) == n_epochs

        if isinstance(learning_rate_beta, float):
            learning_rate_beta = [learning_rate_beta] * n_epochs
        assert len(learning_rate_beta) == n_epochs

        if not hasattr(self, "callback_history"):
            self.callback_history = []

        for epoch in range(1, n_epochs + 1):
            start_time = time()

            # set the effective learning rates
            self.learning_rate = learning_rate[epoch - 1]
            self.learning_rate_beta = learning_rate_beta[epoch - 1]

            # compute and apply gradient updates for each mini batch
            for mini_batch_indices in self._random_mini_batch_indices(mini_batch_size):
                V_pos = self.V_train[mini_batch_indices]
                self._compute_positive_grads(V_pos)
                self._compute_negative_grads(V_pos.shape[0])
                self._apply_grads(self.learning_rate / V_pos.shape[0])
                self._check_h_and_H_ranges()

            # update β
            samples = self.sample(n_samples)
            self._update_beta(samples)
            self._check_h_and_H_ranges()

            # callback function
            if callback is not None:
                callback_output = callback(self, self._get_state_vectors(samples))
                self.callback_history.append(callback_output)

            # print diagnostics
            end_time = time()
            print(
                f"[{type(self).__name__}] epoch {epoch}:",
                f"β = {self.beta:.3f},",
                f"learning rate = {learning_rate[epoch - 1]:.2e},",
                f"β learning rate = {learning_rate_beta[epoch - 1]:.2e},",
                f"epoch duration = {timedelta(seconds=end_time - start_time)}",
            )
            if callback is not None and "print" in callback_output:
                print(callback_output["print"])

    def save(self, file_path, reinitialize_annealer=True):
        """
        Saves the BQRBM model at file_path. Necessary because of pickling issues with the
        qpu and sampler objects.

        :param file_path: Path to save the model to. Must be a Path object or a string with
            ".pkl" file extension.
        :param reinitialize_annealer: If True and has attribute self.annealer_params, then
            will call self._initialize_annealer() after saving.
        """
        if hasattr(self, "annealer_params"):
            self.qpu = None
            self.sampler = None

        save_artifact(self, file_path)

        if hasattr(self, "annealer_params") and reinitialize_annealer:
            self._initialize_annealer()

    @staticmethod
    def load(file_path, initialize_annealer=True):
        """
        Loads the BQRBM model at file_path. Necessary because of pickling issues with the
        qpu and sampler objects.

        :param file_path: Path to the model to load. Must be a Path object or a string with
            ".pkl" file extension.
        :param initialize_annealer: If True and has attribute self.annealer_params, then
            will call self._initialize_annealer().

        :returns: BQRBM instance loaded from the file path.
        """
        model = load_artifact(file_path)
        if hasattr(model, "annealer_params") and initialize_annealer:
            model._initialize_annealer()

        return model

    @property
    def h(self):
        """
        Ising h values. Correspond to b_i = -β * B_freeze * h_i
        """
        return -self.b / (self.beta * self.B_freeze)

    @property
    def J(self):
        """
        Ising J values. Correspond to w_ij = -β * B_freeze * J_ij
        """
        J = np.zeros((self.n_qubits, self.n_qubits))
        J[: self.n_visible, self.n_visible :] = -self.W / (self.beta * self.B_freeze)
        return J

    def _check_h_and_H_ranges(self):
        """
        Raises and exception if h and J values do not fall within h_range and J_range.
        """
        h_satisfied = np.logical_and(
            self.h > self.h_range.min(), self.h < self.h_range.max()
        ).all()
        J_satisfied = np.logical_and(
            self.J > self.J_range.min(), self.J < self.J_range.max()
        ).all()

        if not h_satisfied or not J_satisfied:
            raise Exception("Learned h and J values outside of allowed range")

    def _compute_positive_grads(self, V_pos):
        """
        Computes the gradients for the positive phase, i.e., the expectation values w.r.t.
        the clamped Hamiltonian.

        :param V_pos: Training data set mini-batch of positive vectors, shape
            (mini_batch_size, n_visible).
        """
        b_hidden = self.b[self.n_visible :] + V_pos @ self.W
        D = np.sqrt((self.beta * self.A_freeze) ** 2 + b_hidden**2)
        H_pos = (b_hidden / D) * np.tanh(D)

        self.grads["b_pos"] = np.concatenate((V_pos.mean(axis=0), H_pos.mean(axis=0)))
        self.grads["W_pos"] = V_pos.T @ H_pos / V_pos.shape[0]

    def _compute_negative_grads(self, n_samples):
        """
        Computes the gradients for the negative phase, i.e., the expectation values w.r.t.
        the model distribution.

        :param n_samples: Number of samples to use in the negative phase.
        """
        samples = self.sample(n_samples)
        state_vectors = self._get_state_vectors(samples)

        V_neg = state_vectors[:, : self.n_visible]
        b_hidden = self.b[self.n_visible :] + V_neg @ self.W
        D = np.sqrt((self.beta * self.A_freeze) ** 2 + b_hidden**2)
        H_neg = (b_hidden / D) * np.tanh(D)

        self.grads["b_neg"] = np.concatenate((V_neg.mean(axis=0), H_neg.mean(axis=0)))
        self.grads["W_neg"] = V_neg.T @ H_neg / V_neg.shape[0]

    def _get_state_vectors(self, samples):
        """
        Get the state vectors from the samples (depending on exact or annealer generated).

        :param samples: Return value out of BQRBM.sample().

        :returns: Array of state vectors, shape (n_samples, n_qubits).
        """
        if hasattr(self, "simulation_params"):
            return samples["state_vectors"]
        else:
            return samples.record.sample

    def _initialize_annealer(self):
        """
        Initializes the D-Wave sampler using the fixed embedding provided to the object
        instantiation.
        """
        self.qpu = DWaveSampler(**self.annealer_params.get("qpu_params", {}))
        self.sampler = FixedEmbeddingComposite(self.qpu, self.annealer_params["embedding"])
        self.h_range = np.array(self.qpu.properties["h_range"])
        self.J_range = np.array(self.qpu.properties["j_range"])

    def _initialize_weights_and_biases(self, mu=0, sigma=0.1):
        """
        Initializes the weights and biases.

        :param mu: Mean of the normal distribution of the weights.
        :param sigma: Standard deviation of the normal distribution of the weights.
        """
        self.b = np.zeros(self.n_qubits)
        self.W = self.rng.normal(mu, sigma, (self.n_visible, self.n_hidden))

    def _mean_classical_energy(self, V, H, VW):
        """
        Computes the mean classical energy w.r.t. the weights and biases over the provided
        visible and hidden unit state vectors.

        :param V: Numpy array where the rows are visible units.
        :param H: Numpy array where the rows are hidden units.
        :param VW: V @ W (used to avoid double computation).

        :returns: Mean energy.
        """
        return (
            -(V @ self.b[: self.n_visible]).sum()
            - (H @ self.b[self.n_visible :]).sum()
            - np.einsum("kj,kj", VW, H)
        ) / V.shape[0]

    def _sample_annealer(self, n_samples, answer_mode="raw", use_gauge=True, binary=False):
        """
        Obtain a sample set using the annealer.

        :param n_samples: Number of samples to generate (num_reads param in sample_ising).
        :param answer_mode: "raw" or "histogram".
        :param use_gauge: If True will use a random gauge transformation (recommended for
            more robust sample generation).
        :param binary: If true will convert the state vector values from {-1, +1} to {0, 1}.

        :returns: Ocean SDK SampleSet object.
        """
        # compute the h's and J's
        h = self.h
        J = self.J

        # apply a random gauge
        if use_gauge:
            gauge = self.rng.choice([-1, 1], self.n_qubits)
            h *= gauge
            J *= np.outer(gauge, gauge)

        # compute the chain strength
        chain_strength = self.annealer_params.get("relative_chain_strength")
        if chain_strength is not None:
            chain_strength *= max(np.abs(h).max(), np.abs(J).max())
            chain_strength = min(chain_strength, self.J_range.max())

        # get samples from the annealer
        samples = self.sampler.sample_ising(
            h,
            J,
            num_reads=n_samples,
            anneal_schedule=self.annealer_params.get("schedule"),
            chain_strength=chain_strength,
            answer_mode=answer_mode,
            auto_scale=False,
        )

        # undo the gauge
        if use_gauge:
            samples.record.sample *= gauge

        # convert to binary if specified
        if binary:
            samples.record.sample = self._eigen_to_binary(samples.record.sample)

        return samples

    def _sample_simulation(self, n_samples, binary=False):
        """
        Sample using the exact computed probabilities.

        :param n_samples: Number of samples to generate.
        :param binary: If true will convert the state vector values from {-1, +1} to {0, 1}.

        :returns: Dict with keys:
            - "E": Energies, i.e., the diagonal of the Hamiltonian.
            - "p": Probabilities, i.e., the diagonal of the density matrix.
            - "states": Integer state numbers of sampled states.
            - "state_vectors": Array of sampled state vectors, shape (n_samples, n_qubits).
        """
        # compute the h's and J's
        h = self.h
        J = self.J

        # compute the Hamiltonian and density matrix
        H = compute_H(h, J, self.A_freeze, self.B_freeze, self.n_qubits, self._pauli_kron)
        rho = compute_rho(H, self.simulation_params["beta"], diagonal=(self.A_freeze == 0))

        # sample using the probabilities on the diagonal of rho
        samples = {}
        samples["E"] = np.diag(H).copy()
        samples["p"] = np.diag(rho).copy()
        samples["states"] = self.rng.choice(
            range(2**self.n_qubits), size=n_samples, p=samples["p"]
        )
        samples["state_vectors"] = self._binary_to_eigen(
            np.vstack(
                [Discretizer.int_to_bit_vector(x, self.n_qubits) for x in samples["states"]]
            )
        )

        # convert to binary if specified
        if binary:
            samples["state_vectors"] = self._eigen_to_binary(samples["state_vectors"])

        return samples

    def _update_beta(self, samples):
        """
        Updates the effective β = 1 / kT estimator. Used for scaling the coefficients sent
        to the annealer.

        Note: in its current form, this method only works when s_freeze = 1, i.e., when
        training and sampling from classical Boltzmann distributions.

        :param samples: Samples to use for computing the mean energies w.r.t. the model.
        """
        # compute the train energy
        VW_train = self.V_train @ self.W
        b_eff = self.b[self.n_visible :] + VW_train
        D = np.sqrt((self.beta * self.A_freeze) ** 2 + b_eff**2)
        H_train = (b_eff / D) * np.tanh(D)
        E_train = self._mean_classical_energy(self.V_train, H_train, VW_train)

        # compute the model energy
        state_vectors = self._get_state_vectors(samples)
        V_model = state_vectors[:, : self.n_visible]
        VW_model = V_model @ self.W
        H_model = state_vectors[:, self.n_visible :]
        E_model = self._mean_classical_energy(V_model, H_model, VW_model)

        # update the params
        self.beta = np.clip(
            self.beta + self.learning_rate_beta * (E_train - E_model), *self.beta_range
        )
        self.beta_history.append(self.beta)
