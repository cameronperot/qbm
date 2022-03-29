from abc import ABC, abstractmethod

import numpy as np

from qbm.utils import get_rng


class QBMBase(ABC):
    """
    Abstract base class for Quantum Boltzmann Machines

    Theory based on Quantum Boltzmann Machine by Amin et al.
    https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021050
    """

    def __init__(self, V_train, n_hidden, seed):
        """
        :param V_train: Training data.
        :param n_hidden: Number of hidden units.
        :param seed: Seed for the random number generator.
        """
        self.V_train = V_train
        self.n_visible = V_train.shape[1]
        self.n_hidden = n_hidden
        self.n_qubits = self.n_visible + self.n_hidden
        self.seed = seed
        self.rng = get_rng(self.seed)
        self.grads = {}

        self._initialize_weights_and_biases()

    def _apply_grads(self, learning_rate):
        """
        Applies the gradients from the positive and negative phases using the provided
        learning rate.

        :param learning_rate: Learning rate to scale the gradients with.
        """
        self.b += learning_rate * (self.grads["b_pos"] - self.grads["b_neg"])
        self.W += learning_rate * (self.grads["W_pos"] - self.grads["W_neg"])

    def _binary_to_eigen(self, x):
        """
        Convert bit values {0, 1} to corresponding spin values {+1, -1}.

        :param x: Input array of values {0, 1}.

        :returns: Output array of values {+1, -1}.
        """
        return (1 - 2 * x).astype(np.int8)

    def _eigen_to_binary(self, x):
        """
        Convert spin values {+1, -1} to corresponding bit values {0, 1}.

        :param x: Input array of values {+1, -1}.

        :returns: Output array of values {0, 1}.
        """
        return ((1 - x) / 2).astype(np.int8)

    def _random_mini_batch_indices(self, mini_batch_size):
        """
        Generates random, non-intersecting sets of indices for creating mini-batches of the
        training data.

        :param mini_batch_size: Size of the mini-batches.

        :returns: List of numpy arrays, each array containing the indices corresponding to
            a mini-batch.
        """
        return np.split(
            self.rng.permutation(np.arange(self.V_train.shape[0])),
            np.arange(mini_batch_size, self.V_train.shape[0], mini_batch_size),
        )

    @abstractmethod
    def _compute_positive_grads(self):
        pass

    @abstractmethod
    def _compute_negative_grads(self):
        pass

    @abstractmethod
    def _initialize_weights_and_biases(self):
        pass
