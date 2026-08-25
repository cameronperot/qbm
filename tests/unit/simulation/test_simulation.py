import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix, spmatrix

from qbm.simulation import (
    compute_H,
    compute_rho,
    get_pauli_kron,
    sparse_kron,
)

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


def dense_kron_chain(operators: list[np.ndarray]) -> np.ndarray:
    """Reference Kronecker product built from explicit dense matrices."""
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def basis_state_bits(x: int, n_qubits: int) -> np.ndarray:
    """Bits of basis state index x, MSB first (matching the Kron ordering)."""
    return np.array([(x >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)])


def test_get_pauli_kron_keys_and_shapes() -> None:
    pauli_kron = get_pauli_kron(n_visible, n_hidden)

    expected_keys = (
        {("x", i) for i in range(n_qubits)}
        | {("z_diag", i) for i in range(n_qubits)}
        | {("zz_diag", i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)}
    )
    assert set(pauli_kron.keys()) == expected_keys
    for i in range(n_qubits):
        assert pauli_kron[("x", i)].shape == (2**n_qubits, 2**n_qubits)
        assert pauli_kron[("z_diag", i)].shape == (2**n_qubits,)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            assert pauli_kron[("zz_diag", i, j)].shape == (2**n_qubits,)


@pytest.mark.parametrize(
    ("n_visible", "n_hidden"),
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    ids=["2_qubits", "3_qubits_v1_h2", "3_qubits_v2_h1", "4_qubits"],
)
def test_get_pauli_kron_matches_dense_kron_products(
    n_visible: int, n_hidden: int
) -> None:
    n_qubits = n_visible + n_hidden
    pauli_kron = get_pauli_kron(n_visible, n_hidden)

    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    I2 = np.eye(2)
    for i in range(n_qubits):
        x_i = pauli_kron["x", i]
        z_i = pauli_kron["z_diag", i]
        x_reference = dense_kron_chain([X if k == i else I2 for k in range(n_qubits)])
        z_reference = dense_kron_chain(
            [Z if k == i else I2 for k in range(n_qubits)]
        ).diagonal()

        assert isinstance(x_i, spmatrix)
        assert isinstance(z_i, np.ndarray)
        assert np.allclose(csr_matrix(x_i).toarray(), x_reference)
        assert np.allclose(z_i, z_reference)


def test_get_pauli_kron_zz_diag_equals_elementwise_z_product() -> None:
    pauli_kron = get_pauli_kron(n_visible, n_hidden)

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            zz_ij = pauli_kron["zz_diag", i, j]
            z_i = pauli_kron["z_diag", i]
            z_j = pauli_kron["z_diag", j]
            assert isinstance(zz_ij, np.ndarray)
            assert isinstance(z_i, np.ndarray)
            assert isinstance(z_j, np.ndarray)
            assert np.allclose(zz_ij, z_i * z_j)


@pytest.mark.parametrize(
    "i",
    [0, 1, 2],
    ids=["first", "middle", "last"],
)
def test_sparse_kron_matches_dense_reference(i: int) -> None:
    n_qubits = 3
    A = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))

    result = sparse_kron(i, n_qubits, A)

    reference = dense_kron_chain(
        [
            np.eye(2**i),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.eye(2 ** (n_qubits - i - 1)),
        ]
    )
    assert np.allclose(result.toarray(), reference)


def test_compute_H_A_zero_matches_brute_force_ising_energy(pauli_kron: dict) -> None:
    rng = np.random.RandomState(42)
    h = rng.normal(size=n_qubits)
    h[[1, 3]] = 0
    J = rng.normal(size=(n_qubits, n_qubits))
    J[np.tril_indices(n_qubits)] = 0
    J[0, 2] = 0

    H = compute_H(h=h, J=J, A=0, B=1.3, n_qubits=n_qubits, pauli_kron=pauli_kron)

    # Brute-force Ising energy of every basis state
    energies = np.zeros(2**n_qubits)
    for x in range(2**n_qubits):
        z = 1 - 2 * basis_state_bits(x, n_qubits)
        energies[x] = np.dot(h, z) + sum(
            J[i, j] * z[i] * z[j]
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    assert np.allclose(H, np.diag(1.3 * energies))


def test_compute_H_A_nonzero_matches_dense_reference(pauli_kron: dict) -> None:
    rng = np.random.RandomState(7)
    h = rng.normal(size=n_qubits)
    J = rng.normal(size=(n_qubits, n_qubits))
    J[np.tril_indices(n_qubits)] = 0
    A, B = 0.7, 1.1

    H = compute_H(h=h, J=J, A=A, B=B, n_qubits=n_qubits, pauli_kron=pauli_kron)

    # Dense reference: B * (h·z + z^T J z) diagonal minus A * sum of X operators
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    I2 = np.eye(2)
    diagonal = np.zeros(2**n_qubits)
    for x in range(2**n_qubits):
        z = 1 - 2 * basis_state_bits(x, n_qubits)
        diagonal[x] = np.dot(h, z) + sum(
            J[i, j] * z[i] * z[j]
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    off_diagonal = sum(
        -A * dense_kron_chain([X if k == i else I2 for k in range(n_qubits)])
        for i in range(n_qubits)
    )
    assert np.allclose(H, B * np.diag(diagonal) + off_diagonal)


def test_compute_H_zero_coefficients_skipped_correctly(pauli_kron: dict) -> None:
    rng = np.random.RandomState(3)
    h = rng.normal(size=n_qubits)
    h[[1, 3]] = 0
    J = rng.normal(size=(n_qubits, n_qubits))
    J[np.tril_indices(n_qubits)] = 0
    J[0, 2] = 0
    J[1, 2] = 0
    A, B = 0.5, 1.2

    H = compute_H(h=h, J=J, A=A, B=B, n_qubits=n_qubits, pauli_kron=pauli_kron)

    # Dense reference built the same way, with the zeroed entries simply absent
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    I2 = np.eye(2)
    diagonal = np.zeros(2**n_qubits)
    for x in range(2**n_qubits):
        z = 1 - 2 * basis_state_bits(x, n_qubits)
        diagonal[x] = np.dot(h, z) + sum(
            J[i, j] * z[i] * z[j]
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    off_diagonal = sum(
        -A * dense_kron_chain([X if k == i else I2 for k in range(n_qubits)])
        for i in range(n_qubits)
    )
    assert np.allclose(H, B * np.diag(diagonal) + off_diagonal)


def test_compute_rho_diagonal_and_eigendecomposition_paths_agree() -> None:
    rng = np.random.RandomState(11)
    H = np.diag(rng.normal(size=8))

    rho_diagonal = compute_rho(H, beta=0.8, diagonal=True)
    rho_eigendecomposition = compute_rho(H, beta=0.8, diagonal=False)

    assert np.allclose(rho_diagonal, rho_eigendecomposition)


def test_compute_rho_matches_matrix_exponential() -> None:
    rng = np.random.RandomState(13)
    H = rng.normal(size=(8, 8))
    H = H + H.T

    rho = compute_rho(H, beta=0.9)

    rho_reference = expm(-0.9 * H)
    rho_reference /= np.trace(rho_reference)
    assert np.allclose(rho, rho_reference)
    assert np.isclose(np.trace(rho), 1)


def test_compute_rho_large_beta_concentrates_on_ground_state() -> None:
    energies = np.array([2.0, -1.5, 0.5, 1.0])
    H = np.diag(energies)

    rho = compute_rho(H, beta=100, diagonal=True)

    ground_state_projector = np.diag([0, 1, 0, 0])
    assert np.allclose(rho, ground_state_projector, atol=1e-30)


def test_compute_rho_two_level_system_matches_softmax() -> None:
    H = np.diag([-1.0, 1.0])
    beta = 0.5

    rho = compute_rho(H, beta=beta, diagonal=True)

    p = np.exp(-beta * np.array([-1.0, 1.0]))
    p /= p.sum()
    assert np.allclose(rho.diagonal(), p)
