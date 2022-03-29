import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from qbm.models import BQRBM
from qbm.utils import Discretizer, get_rng

n_visible = 8
n_hidden = 4
n_samples = 100
n_qubits = n_visible + n_hidden
learning_rate = 1e-3


def mock_initialize_annealer(model):
    setattr(model, "qpu", None)
    setattr(model, "h_range", np.array([-4, 4]))
    setattr(model, "J_range", np.array([-1, 1]))


@pytest.fixture
def model_simulation(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    model = BQRBM(
        V_train=V_train,
        n_hidden=n_hidden,
        A_freeze=0.1,
        B_freeze=1.1,
        beta_initial=1.5,
        beta_range=[0.1, 10],
        simulation_params={"beta": 1.5},
        seed=0,
    )

    return model


@pytest.fixture
def model_annealer(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    model = BQRBM(
        V_train=V_train,
        n_hidden=n_hidden,
        A_freeze=0.1,
        B_freeze=1.1,
        beta_initial=1.5,
        beta_range=[0.1, 10],
        annealer_params={"embedding": {1: [1], 2: [2]}, "schedule": [(0, 0), (20, 1)]},
        seed=0,
    )

    return model


def test_init_simultor(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    A_freeze = 0.1
    B_freeze = 1.1
    beta_initial = 0.5
    beta_range = [0.1, 10]
    simulation_params = {"beta": 1.0}
    seed = 0

    model = BQRBM(
        V_train=V_train,
        n_hidden=n_hidden,
        A_freeze=A_freeze,
        B_freeze=B_freeze,
        beta_initial=beta_initial,
        beta_range=beta_range,
        simulation_params=simulation_params,
        seed=seed,
    )

    assert (model.V_train == 1 - 2 * V_train).all()
    assert model.n_hidden == n_hidden
    assert model.n_visible == n_visible
    assert model.n_qubits == n_qubits
    assert model.seed == seed
    assert model.simulation_params == simulation_params
    assert model.A_freeze == A_freeze
    assert model.B_freeze == B_freeze
    assert model.beta == beta_initial
    assert model.beta_history == [beta_initial]
    assert model.beta_range == beta_range
    assert hasattr(model, "simulation_params")
    assert not hasattr(model, "annealer_params")


def test_init_simultor_bad_params(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    A_freeze = 0.1
    B_freeze = 1.1
    beta_initial = 0.5
    beta_range = [0.1, 10]
    simulation_params = {}
    seed = 0

    with pytest.raises(Exception):
        model = BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
            beta_initial=beta_initial,
            beta_range=beta_range,
            simulation_params=simulation_params,
            seed=seed,
        )


def test_init_simultor_annealer_both_fail(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    A_freeze = 0.1
    B_freeze = 1.1
    annealer_params = {"embedding": {1: [1], 2: [2]}, "schedule": [(0, 0), (20, 1)]}
    simulation_params = {"beta": 1.0}

    with pytest.raises(Exception):
        model = BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
            annealer_params=annealer_params,
            simulation_params=simulation_params,
        )


def test_init_simultor_annealer_none_fail(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    A_freeze = 0.1
    B_freeze = 1.1

    with pytest.raises(Exception):
        model = BQRBM(
            V_train=V_train, n_hidden=n_hidden, A_freeze=A_freeze, B_freeze=B_freeze,
        )


def test_init_annealer(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    A_freeze = 0.1
    B_freeze = 1.1
    beta_initial = 0.5
    beta_range = [0.1, 10]
    annealer_params = {"embedding": {1: [1], 2: [2]}, "schedule": [(0, 0), (20, 1)]}
    seed = 0

    model = BQRBM(
        V_train=V_train,
        n_hidden=n_hidden,
        A_freeze=A_freeze,
        B_freeze=B_freeze,
        beta_initial=beta_initial,
        beta_range=beta_range,
        annealer_params=annealer_params,
        seed=seed,
    )

    assert (model.V_train == 1 - 2 * V_train).all()
    assert model.n_hidden == n_hidden
    assert model.n_visible == n_visible
    assert model.n_qubits == n_qubits
    assert model.seed == seed
    assert model.annealer_params == annealer_params
    assert model.A_freeze == A_freeze
    assert model.B_freeze == B_freeze
    assert model.beta == beta_initial
    assert model.beta_history == [beta_initial]
    assert model.beta_range == beta_range
    assert not hasattr(model, "simulation_params")
    assert hasattr(model, "annealer_params")


def test_init_annealer_bad_params(monkeypatch):
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer)

    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)
    V_train = discretizer.df_to_bit_array(df)

    A_freeze = 0.1
    B_freeze = 1.1
    beta_initial = 0.5
    beta_range = [0.1, 10]
    annealer_params = {}
    seed = 0

    with pytest.raises(Exception):
        model = BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
            beta_initial=beta_initial,
            beta_range=beta_range,
            annealer_params=annealer_params,
            seed=seed,
        )


def test_sample_annealer(monkeypatch, model_annealer):
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    monkeypatch.setattr(
        "qbm.models.BQRBM._sample_annealer",
        lambda self, n_samples, answer_mode, use_gauge, binary: "test",
    )

    model = model_annealer

    assert model.sample(10) == "test"


def test_sample_simulation(monkeypatch, model_simulation):
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    monkeypatch.setattr(
        "qbm.models.BQRBM._sample_simulation", lambda self, n_samples, binary: "test",
    )

    model = model_simulation

    assert model.sample(10) == "test"


def test__mean_classical_energy(model_simulation):
    rng = get_rng(0)
    V = rng.rand(n_samples, n_visible)
    H = rng.rand(n_samples, n_hidden)
    W = rng.rand(n_visible, n_hidden)

    E = 0
    for k in range(n_samples):
        E += (
            -V[k] @ model_simulation.b[:n_visible]
            - H[k] @ model_simulation.b[n_visible:]
            - V[k] @ W @ H[k]
        )
    E /= n_samples

    E_model_simulation = model_simulation._mean_classical_energy(V, H, V @ W)

    assert np.isclose(E, E_model_simulation)


def test__compute_positive_grads(model_simulation):
    rng = get_rng(0)
    V_pos = rng.rand(n_samples, n_visible)
    Γ = model_simulation.beta * model_simulation.A_freeze
    b_eff = model_simulation.b[n_visible:] + V_pos @ model_simulation.W
    D = np.sqrt(Γ ** 2 + b_eff ** 2)
    H_pos = (b_eff / D) * np.tanh(D)

    grads = {
        "b_pos": np.zeros(model_simulation.b.shape),
        "W_pos": np.zeros(model_simulation.W.shape),
    }
    for k in range(n_samples):
        grads["b_pos"] += np.concatenate((V_pos[k] / n_samples, H_pos[k] / n_samples))
        grads["W_pos"] += np.outer(V_pos[k], H_pos[k]) / n_samples

    model_simulation._compute_positive_grads(V_pos)

    assert b_eff.shape == (n_samples, n_hidden)
    assert D.shape == (n_samples, n_hidden)
    assert H_pos.shape == (n_samples, n_hidden)
    for grad_name, grad in grads.items():
        assert np.isclose(grad, model_simulation.grads[grad_name]).all()


def test__compute_negative_grads(monkeypatch, model_simulation):
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    V_neg = state_vectors[:, :n_visible]
    Γ = model_simulation.beta * model_simulation.A_freeze
    b_eff = model_simulation.b[n_visible:] + V_neg @ model_simulation.W
    D = np.sqrt(Γ ** 2 + b_eff ** 2)
    H_neg = (b_eff / D) * np.tanh(D)
    monkeypatch.setattr(
        "qbm.models.BQRBM.sample", lambda self, n_samples: {"state_vectors": state_vectors}
    )

    grads = {
        "b_neg": np.zeros(model_simulation.b.shape),
        "W_neg": np.zeros(model_simulation.W.shape),
    }
    for k in range(n_samples):
        grads["b_neg"] += np.concatenate((V_neg[k] / n_samples, H_neg[k] / n_samples))
        grads["W_neg"] += np.outer(V_neg[k], H_neg[k]) / n_samples

    model_simulation._compute_negative_grads(n_samples)

    for grad_name, grad in grads.items():
        assert np.isclose(grad, model_simulation.grads[grad_name]).all()


def test__update_beta(monkeypatch, model_simulation):
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    monkeypatch.setattr("qbm.models.BQRBM.sample", lambda self, n_samples: state_vectors)

    beta = model_simulation.beta
    setattr(model_simulation, "learning_rate", learning_rate)

    V_train = model_simulation.V_train
    VW_train = V_train @ model_simulation.W
    b_eff = model_simulation.b[n_visible:] + VW_train
    D = np.sqrt((model_simulation.beta * model_simulation.A_freeze) ** 2 + b_eff ** 2)
    H_train = (b_eff / D) * np.tanh(D)
    E_train = model_simulation._mean_classical_energy(V_train, H_train, VW_train)

    V_model_simulation = state_vectors[:, :n_visible]
    H_model_simulation = state_vectors[:, n_visible:]
    E_model_simulation = model_simulation._mean_classical_energy(
        V_model_simulation, H_model_simulation, V_model_simulation @ model_simulation.W
    )

    Δbeta = learning_rate * (E_train - E_model_simulation)

    model_simulation.learning_rate_beta = learning_rate
    model_simulation._update_beta({"state_vectors": state_vectors})

    assert model_simulation.beta == np.clip(beta + Δbeta, *model_simulation.beta_range)
