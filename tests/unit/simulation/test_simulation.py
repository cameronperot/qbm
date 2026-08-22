import numpy as np
import pytest

from qbm.simulation import compute_H, compute_rho, get_pauli_kron

n_visible = 2
n_hidden = 2
n_qubits = n_visible + n_hidden


@pytest.fixture
def pauli_kron() -> dict:
    return get_pauli_kron(n_visible, n_hidden)


def test_get_pauli_kron_invalid_n_visible_raises_value_error() -> None:
    with pytest.raises(ValueError, match="n_visible must be positive"):
        get_pauli_kron(0, n_hidden)


def test_get_pauli_kron_invalid_n_hidden_raises_value_error() -> None:
    with pytest.raises(ValueError, match="n_hidden must be positive"):
        get_pauli_kron(n_visible, 0)


def test_compute_H_h_length_mismatch_raises_value_error(
    pauli_kron: dict,
) -> None:
    with pytest.raises(ValueError, match="h has length"):
        compute_H(
            h=np.zeros(n_qubits - 1),
            J=np.zeros((n_qubits, n_qubits)),
            A=0.1,
            B=1.0,
            n_qubits=n_qubits,
            pauli_kron=pauli_kron,
        )


def test_compute_H_J_shape_mismatch_raises_value_error(
    pauli_kron: dict,
) -> None:
    with pytest.raises(ValueError, match="J has shape"):
        compute_H(
            h=np.zeros(n_qubits),
            J=np.zeros((n_qubits - 1, n_qubits)),
            A=0.1,
            B=1.0,
            n_qubits=n_qubits,
            pauli_kron=pauli_kron,
        )


def test_compute_rho_non_square_H_raises_value_error() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        compute_rho(H=np.zeros((2, 3)), beta=1.0)


def test_compute_rho_invalid_beta_raises_value_error() -> None:
    with pytest.raises(ValueError, match="beta must be positive"):
        compute_rho(H=np.zeros((2, 2)), beta=0.0)
