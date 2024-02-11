import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, identity, kron

# set constants
sparse_X = csr_matrix(([1, 1], ([0, 1], [1, 0])), dtype=np.float64)
sparse_Z = csr_matrix(([1, -1], ([0, 1], [0, 1])), dtype=np.float64)


def get_pauli_kron(n_visible, n_hidden):
    """
    Computes the necessary Pauli Kronecker product (sparse) matrices for a n_visible +
    n_hidden qubit problem. Used as an argument to compute_H, e.g. one would instantiate
    pauli_kron as pauli_kron = get_pauli_kron(n_visible, n_hidden), then pass to compute_H
    when computing the Hamiltonian.

    :param n_visible: Number of visible units.
    :param n_hidden: Number of hidden units.

    :returns: A dictionary of Kronecker product Pauli matrices.
    """
    # set Kronecker product Pauli matrices
    n_qubits = n_visible + n_hidden
    pauli_kron = {}
    for i in range(n_qubits):
        pauli_kron["x", i] = sparse_kron(i, n_qubits, sparse_X)
        pauli_kron["z_diag", i] = sparse_kron(i, n_qubits, sparse_Z).diagonal()
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            pauli_kron["zz_diag", i, j] = pauli_kron["z_diag", i] * pauli_kron["z_diag", j]

    return pauli_kron


def sparse_kron(i, n_qubits, A):
    """
    Compute I_{2^i} ⊗ A ⊗ I_{2^(n_qubits-i-1)}.

    :param i: Index of the "A" matrix.
    :param n_qubits: Total number of qubits.
    :param A: Matrix to tensor with identities.

    :returns: I_{2^i} ⊗ A ⊗ I_{2^(n_qubits-i-1)}.
    """
    if i != 0 and i != n_qubits - 1:
        return kron(kron(identity(2**i), A), identity(2 ** (n_qubits - i - 1)))
    if i == 0:
        return kron(A, identity(2 ** (n_qubits - 1)))
    if i == n_qubits - 1:
        return kron(identity(2 ** (n_qubits - 1)), A)


def compute_H(h, J, A, B, n_qubits, pauli_kron):
    """
    Computes the Hamiltonian of the annealer at relative time s.

    :param h: Linear Ising terms.
    :param J: Quadratic Ising terms.
    :param A: Coefficient of the off-diagonal terms, e.g. A(s).
    :param B: Coefficient of the diagonal terms, e.g. B(s).
    :param n_qubits: Number of qubits.
    :param pauli_kron: Kronecker product Pauli matrices dict.

    :returns: Hamiltonian matrix H.
    """
    # diagonal terms
    H_diag = np.zeros(2**n_qubits)
    for i in range(n_qubits):
        # linear terms
        if h[i] != 0:
            H_diag += (B * h[i]) * pauli_kron["z_diag", i]

        # quadratic terms
        for j in range(i + 1, n_qubits):
            if J[i, j] != 0:
                H_diag += (B * J[i, j]) * pauli_kron["zz_diag", i, j]

    # return just the diagonal if H is a diagonal matrix
    if A == 0:
        return np.diag(H_diag)

    # off-diagonal terms
    H = csr_matrix((2**n_qubits, 2**n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        H -= A * pauli_kron["x", i]

    return (H + diags(H_diag, format="csr")).toarray()


def compute_rho(H, beta, diagonal=False):
    """
    Computes the trace normalized density matrix rho.

    :param H: Hamiltonian matrix.
    :param beta: Inverse temperature beta = 1 / (k_B * T).
    :param diagonal: Flag to indicate whether H is a diagonal matrix or not.

    :return: Density matrix rho.
    """
    # if diagonal then compute directly, else use eigen decomposition
    if diagonal:
        Lambda = H.diagonal()
        exp_beta_Lambda = np.exp(-beta * (Lambda - Lambda.min()))
        return np.diag(exp_beta_Lambda / exp_beta_Lambda.sum())
    else:
        Lambda, S = eigh(H)
        exp_beta_Lambda = np.exp(-beta * (Lambda - Lambda.min()))
        return (S * (exp_beta_Lambda / exp_beta_Lambda.sum())) @ S.T
