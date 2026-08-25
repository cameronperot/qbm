from pathlib import Path
from typing import Any

import dimod
import numpy as np
import pytest

from qbm.models import BQRBM
from qbm.simulation import compute_H, compute_rho
from qbm.utils import get_rng

n_visible = 8
n_hidden = 4
n_samples = 100
n_qubits = n_visible + n_hidden
learning_rate = 1e-3


def mock_initialize_annealer(model: Any) -> None:
    model.qpu = None
    model.h_range = np.array([-4, 4])
    model.J_range = np.array([-1, 1])


def test_init_simulation(monkeypatch: Any, V_train: np.ndarray) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

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


def test_init_simulation_bad_params(monkeypatch: Any, V_train: np.ndarray) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

    A_freeze = 0.1
    B_freeze = 1.1
    beta_initial = 0.5
    beta_range = [0.1, 10]
    simulation_params = {}
    seed = 0

    with pytest.raises(ValueError, match="Missing key in simulation_params"):
        BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
            beta_initial=beta_initial,
            beta_range=beta_range,
            simulation_params=simulation_params,
            seed=seed,
        )


def test_init_simulation_annealer_both_fail(
    monkeypatch: Any, V_train: np.ndarray
) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

    A_freeze = 0.1
    B_freeze = 1.1
    annealer_params = {"embedding": {1: [1], 2: [2]}, "schedule": [(0, 0), (20, 1)]}
    simulation_params = {"beta": 1.0}

    with pytest.raises(
        ValueError, match="one of either annealer_params or simulation_params"
    ):
        BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
            annealer_params=annealer_params,
            simulation_params=simulation_params,
        )


def test_init_simulation_annealer_none_fail(
    monkeypatch: Any, V_train: np.ndarray
) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

    A_freeze = 0.1
    B_freeze = 1.1

    with pytest.raises(
        ValueError, match="one of either annealer_params or simulation_params"
    ):
        BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
        )


def test_init_annealer(monkeypatch: Any, V_train: np.ndarray) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

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


def test_init_annealer_bad_params(monkeypatch: Any, V_train: np.ndarray) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

    A_freeze = 0.1
    B_freeze = 1.1
    beta_initial = 0.5
    beta_range = [0.1, 10]
    annealer_params = {}
    seed = 0

    with pytest.raises(ValueError, match="Missing key in annealer_params"):
        BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=A_freeze,
            B_freeze=B_freeze,
            beta_initial=beta_initial,
            beta_range=beta_range,
            annealer_params=annealer_params,
            seed=seed,
        )


def test_sample_annealer_dispatches_to_annealer(model_annealer: BQRBM) -> None:
    rng = np.random.RandomState(14)
    model_annealer.b = rng.normal(size=n_qubits)
    model_annealer.W = rng.normal(size=(n_visible, n_hidden))
    raw_samples = rng.choice([-1, 1], size=(5, n_qubits)).astype(np.int8)
    sampler = FakeSampler(raw_samples, model_annealer.h)
    model_annealer.sampler = sampler

    samples = model_annealer.sample(5)

    assert isinstance(samples, dimod.SampleSet)
    assert np.array_equal(samples.record.sample, raw_samples)


def test_sample_simulation_dispatches_to_simulation() -> None:
    model = make_small_simulation_model()

    samples = model.sample(20)

    assert isinstance(samples, dict)
    assert set(samples.keys()) == {"E", "p", "states", "state_vectors"}
    assert samples["states"].shape == (20,)
    assert samples["state_vectors"].shape == (20, model.n_qubits)


def test__mean_classical_energy(model_simulation: BQRBM) -> None:
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


def test__compute_positive_grads(model_simulation: BQRBM) -> None:
    rng = get_rng(0)
    V_pos = rng.rand(n_samples, n_visible)
    Γ = model_simulation.beta * model_simulation.A_freeze
    b_eff = model_simulation.b[n_visible:] + V_pos @ model_simulation.W
    D = np.sqrt(Γ**2 + b_eff**2)
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


def test__compute_negative_grads(monkeypatch: Any, model_simulation: BQRBM) -> None:
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    V_neg = state_vectors[:, :n_visible]
    Γ = model_simulation.beta * model_simulation.A_freeze
    b_eff = model_simulation.b[n_visible:] + V_neg @ model_simulation.W
    D = np.sqrt(Γ**2 + b_eff**2)
    H_neg = (b_eff / D) * np.tanh(D)
    monkeypatch.setattr(
        "qbm.models.BQRBM.sample",
        lambda self, n_samples: {"state_vectors": state_vectors},
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


def test__update_beta(monkeypatch: Any, model_simulation: BQRBM) -> None:
    rng = get_rng(0)
    state_vectors = rng.rand(n_samples, n_qubits)
    monkeypatch.setattr(
        "qbm.models.BQRBM.sample", lambda self, n_samples: state_vectors
    )

    beta = model_simulation.beta
    model_simulation.learning_rate = learning_rate

    V_train = model_simulation.V_train
    VW_train = V_train @ model_simulation.W
    b_eff = model_simulation.b[n_visible:] + VW_train
    D = np.sqrt((model_simulation.beta * model_simulation.A_freeze) ** 2 + b_eff**2)
    H_train = (b_eff / D) * np.tanh(D)
    E_train = model_simulation._mean_classical_energy(V_train, H_train, VW_train)

    V_model_simulation = state_vectors[:, :n_visible]
    H_model_simulation = state_vectors[:, n_visible:]
    E_model_simulation = model_simulation._mean_classical_energy(
        V_model_simulation, H_model_simulation, V_model_simulation @ model_simulation.W
    )

    Δbeta = learning_rate * (E_train - E_model_simulation)

    model_simulation.learning_rate_beta = learning_rate
    samples: Any = {"state_vectors": state_vectors}
    model_simulation._update_beta(samples)

    assert model_simulation.beta == np.clip(
        beta + Δbeta, model_simulation.beta_range[0], model_simulation.beta_range[1]
    )


def test_init_invalid_V_train_raises_value_error() -> None:
    rng = get_rng(0)
    V_train = rng.choice([0, 2], size=(n_samples, n_visible))

    with pytest.raises(ValueError, match="must be in"):
        BQRBM(
            V_train=V_train,
            n_hidden=n_hidden,
            A_freeze=0.1,
            B_freeze=1.1,
            simulation_params={"beta": 1.0},
        )


def test_train_learning_rate_length_mismatch_raises_value_error(
    model_simulation: BQRBM,
) -> None:
    with pytest.raises(ValueError, match="learning_rate has length"):
        model_simulation.train(n_epochs=2, learning_rate=[learning_rate])


def test_train_learning_rate_beta_length_mismatch_raises_value_error(
    model_simulation: BQRBM,
) -> None:
    with pytest.raises(ValueError, match="learning_rate_beta has length"):
        model_simulation.train(n_epochs=2, learning_rate_beta=[learning_rate])


def test__check_h_and_H_ranges_h_out_of_range_raises_value_error(
    model_simulation: BQRBM,
) -> None:
    model_simulation.h_range = np.array([1, 2])

    with pytest.raises(ValueError, match="outside of allowed range"):
        model_simulation._check_h_and_H_ranges()


def test__check_h_and_H_ranges_J_out_of_range_raises_value_error(
    model_simulation: BQRBM,
) -> None:
    model_simulation.J_range = np.array([1, 2])

    with pytest.raises(ValueError, match="outside of allowed range"):
        model_simulation._check_h_and_H_ranges()


def make_small_simulation_model(A_freeze: float = 0.1, seed: int = 0) -> BQRBM:
    rng = get_rng(seed)
    V_train = rng.choice([0, 1], size=(20, 4))

    return BQRBM(
        V_train=V_train,
        n_hidden=2,
        A_freeze=A_freeze,
        B_freeze=1.1,
        beta_initial=1.5,
        beta_range=[0.1, 10],
        simulation_params={"beta": 1.5},
        seed=seed,
    )


class FakeSampler:
    """
    Fake annealer sampler which records sample_ising calls and returns samples
    transformed by whatever gauge it detects in the received h.
    """

    def __init__(self, raw_samples: np.ndarray, h_base: np.ndarray) -> None:
        self.raw_samples = raw_samples
        self.h_base = h_base
        self.calls: list[dict[str, Any]] = []

    def sample_ising(self, h: np.ndarray, J: np.ndarray, **kwargs: Any) -> Any:
        self.calls.append({"h": h.copy(), "J": J.copy(), "kwargs": kwargs})
        gauge = np.round(h / self.h_base).astype(np.int8)
        return dimod.SampleSet.from_samples(
            self.raw_samples * gauge,
            vartype=dimod.SPIN,
            energy=np.zeros(self.raw_samples.shape[0]),
        )


def test_h_property_returns_scaled_negative_b(model_simulation: BQRBM) -> None:
    rng = np.random.RandomState(5)
    model_simulation.b = rng.normal(size=n_qubits)

    assert np.allclose(
        model_simulation.h, -model_simulation.b / (1.5 * model_simulation.B_freeze)
    )


def test_J_property_returns_scaled_negative_W_block(model_simulation: BQRBM) -> None:
    rng = np.random.RandomState(6)
    model_simulation.W = rng.normal(size=(n_visible, n_hidden))

    J = model_simulation.J

    expected = np.zeros((n_qubits, n_qubits))
    expected[:n_visible, n_visible:] = -model_simulation.W / (
        1.5 * model_simulation.B_freeze
    )
    assert np.allclose(J, expected)


def test__check_h_and_H_ranges_in_range_passes(model_simulation: BQRBM) -> None:
    model_simulation.h_range = np.array([-10, 10])
    model_simulation.J_range = np.array([-10, 10])

    assert model_simulation._check_h_and_H_ranges() is None


def test__get_state_vectors_sample_set_returns_record_sample(
    model_simulation: BQRBM,
) -> None:
    raw_samples = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    samples = dimod.SampleSet.from_samples(
        raw_samples, vartype=dimod.SPIN, energy=[0.0, 0.0]
    )

    state_vectors = model_simulation._get_state_vectors(samples)

    assert np.array_equal(state_vectors, raw_samples)


def test__get_state_vectors_dict_returns_state_vectors(
    model_simulation: BQRBM,
) -> None:
    state_vectors = np.array([[1, -1], [-1, 1]])
    samples: Any = {"state_vectors": state_vectors}

    assert np.array_equal(model_simulation._get_state_vectors(samples), state_vectors)


def test__sample_simulation_returns_correct_shapes_and_probabilities() -> None:
    model = make_small_simulation_model()
    n_samples = 100

    samples = model._sample_simulation(n_samples)

    assert set(samples.keys()) == {"E", "p", "states", "state_vectors"}
    assert samples["E"].shape == (2**model.n_qubits,)
    assert samples["p"].shape == (2**model.n_qubits,)
    assert samples["states"].shape == (n_samples,)
    assert samples["state_vectors"].shape == (n_samples, model.n_qubits)
    assert np.isclose(samples["p"].sum(), 1)
    assert set(np.unique(samples["state_vectors"])) == {-1, 1}


def test__sample_simulation_probabilities_match_exact_rho() -> None:
    model = make_small_simulation_model()

    samples = model._sample_simulation(100)

    H = compute_H(
        model.h,
        model.J,
        model.A_freeze,
        model.B_freeze,
        model.n_qubits,
        model._pauli_kron,
    )
    rho = compute_rho(H, model.simulation_params["beta"], diagonal=False)
    assert np.allclose(samples["E"], np.diag(H))
    assert np.allclose(samples["p"], np.diag(rho))


def test__sample_simulation_A_freeze_zero_uses_diagonal_rho() -> None:
    model = make_small_simulation_model(A_freeze=0)

    samples = model._sample_simulation(100)

    H = compute_H(
        model.h, model.J, 0, model.B_freeze, model.n_qubits, model._pauli_kron
    )
    rho = compute_rho(H, model.simulation_params["beta"], diagonal=True)
    assert np.allclose(samples["p"], np.diag(rho))
    assert np.allclose(samples["E"], np.diag(H))


def test__sample_simulation_binary_true_returns_bit_values() -> None:
    model = make_small_simulation_model()

    samples = model._sample_simulation(100, binary=True)

    assert set(np.unique(samples["state_vectors"])) == {0, 1}


def test__sample_simulation_same_seed_same_samples() -> None:
    model_1 = make_small_simulation_model(seed=3)
    model_2 = make_small_simulation_model(seed=3)

    samples_1 = model_1._sample_simulation(100)
    samples_2 = model_2._sample_simulation(100)

    assert np.array_equal(samples_1["states"], samples_2["states"])
    assert np.array_equal(samples_1["state_vectors"], samples_2["state_vectors"])


@pytest.mark.parametrize(
    "learning_rate_form",
    [1e-3, [1e-3, 2e-3]],
    ids=["float", "list"],
)
def test_train_updates_parameters_and_records_history(
    learning_rate_form: float | list[float],
) -> None:
    model = make_small_simulation_model()
    b_initial = model.b.copy()
    W_initial = model.W.copy()
    beta_initial = model.beta
    callback_outputs = [
        {"value": 1, "print": "epoch 1"},
        {"value": 2, "print": "epoch 2"},
    ]

    def callback(model: BQRBM, samples: np.ndarray) -> dict[str, Any]:
        return callback_outputs[len(model.beta_history) - 2]

    model.train(
        n_epochs=2,
        learning_rate=learning_rate_form,
        learning_rate_beta=1e-3,
        mini_batch_size=10,
        n_samples=50,
        callback=callback,
    )

    assert len(model.beta_history) == 3
    assert model.beta_history[0] == beta_initial
    assert model.callback_history == callback_outputs
    assert all(0.1 <= beta <= 10 for beta in model.beta_history)
    assert not np.allclose(model.b, b_initial)
    assert not np.allclose(model.W, W_initial)


def test_train_beta_clipped_to_beta_range() -> None:
    model = make_small_simulation_model()
    model.beta_range = [1.5, 1.5]

    model.train(
        n_epochs=2,
        learning_rate=1e-3,
        learning_rate_beta=1e3,
        mini_batch_size=10,
        n_samples=50,
    )

    assert all(beta == 1.5 for beta in model.beta_history)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    model = make_small_simulation_model()
    file_path = tmp_path / "model.pkl"

    model.save(file_path)
    loaded_model = BQRBM.load(file_path)

    assert np.allclose(loaded_model.b, model.b)
    assert np.allclose(loaded_model.W, model.W)
    assert loaded_model.beta == model.beta
    samples_original = model.sample(10)
    samples_loaded = loaded_model.sample(10)
    assert isinstance(samples_original, dict)
    assert isinstance(samples_loaded, dict)
    assert np.array_equal(samples_original["states"], samples_loaded["states"])
    assert np.array_equal(
        samples_original["state_vectors"], samples_loaded["state_vectors"]
    )


def test__sample_annealer_no_gauge_passes_unchanged_h_J_and_kwargs(
    model_annealer: BQRBM,
) -> None:
    rng = np.random.RandomState(8)
    model_annealer.b = rng.normal(size=n_qubits)
    model_annealer.W = rng.normal(size=(n_visible, n_hidden))
    h_base = model_annealer.h
    J_base = model_annealer.J
    raw_samples = rng.choice([-1, 1], size=(7, n_qubits)).astype(np.int8)
    sampler = FakeSampler(raw_samples, h_base)
    model_annealer.sampler = sampler

    samples = model_annealer._sample_annealer(7, use_gauge=False)

    assert len(sampler.calls) == 1
    call = sampler.calls[0]
    assert np.allclose(call["h"], h_base)
    assert np.allclose(call["J"], J_base)
    assert call["kwargs"]["num_reads"] == 7
    assert call["kwargs"]["answer_mode"] == "raw"
    assert call["kwargs"]["auto_scale"] is False
    assert call["kwargs"]["chain_strength"] is None
    assert (
        call["kwargs"]["anneal_schedule"] == model_annealer.annealer_params["schedule"]
    )
    assert np.array_equal(samples.record.sample, raw_samples)


def test__sample_annealer_gauge_transformed_and_undone(model_annealer: BQRBM) -> None:
    rng = np.random.RandomState(9)
    model_annealer.b = rng.normal(size=n_qubits)
    model_annealer.W = rng.normal(size=(n_visible, n_hidden))
    h_base = model_annealer.h
    J_base = model_annealer.J
    raw_samples = rng.choice([-1, 1], size=(7, n_qubits)).astype(np.int8)
    sampler = FakeSampler(raw_samples, h_base)
    model_annealer.sampler = sampler

    samples = model_annealer._sample_annealer(7, use_gauge=True)

    call = sampler.calls[0]
    gauge = np.round(call["h"] / h_base).astype(np.int64)
    assert set(np.unique(gauge)) <= {-1, 1}
    assert np.allclose(call["h"], h_base * gauge)
    assert np.allclose(call["J"], J_base * np.outer(gauge, gauge))
    # The fake returned raw_samples * gauge; the undo must recover raw_samples
    assert np.array_equal(samples.record.sample, raw_samples)


def test__sample_annealer_relative_chain_strength_scaled(
    model_annealer: BQRBM,
) -> None:
    rng = np.random.RandomState(10)
    model_annealer.b = rng.normal(size=n_qubits)
    model_annealer.W = rng.normal(size=(n_visible, n_hidden))
    model_annealer.annealer_params = {
        **model_annealer.annealer_params,
        "relative_chain_strength": 0.5,
    }
    h_base = model_annealer.h
    J_base = model_annealer.J
    raw_samples = rng.choice([-1, 1], size=(5, n_qubits)).astype(np.int8)
    sampler = FakeSampler(raw_samples, h_base)
    model_annealer.sampler = sampler

    model_annealer._sample_annealer(5)

    expected_chain_strength = 0.5 * max(np.abs(h_base).max(), np.abs(J_base).max())
    assert sampler.calls[0]["kwargs"]["chain_strength"] == pytest.approx(
        expected_chain_strength
    )


def test__sample_annealer_chain_strength_capped_at_J_range_max(
    model_annealer: BQRBM,
) -> None:
    rng = np.random.RandomState(11)
    model_annealer.b = rng.normal(size=n_qubits)
    model_annealer.W = rng.normal(size=(n_visible, n_hidden))
    model_annealer.annealer_params = {
        **model_annealer.annealer_params,
        "relative_chain_strength": 100,
    }
    raw_samples = rng.choice([-1, 1], size=(5, n_qubits)).astype(np.int8)
    sampler = FakeSampler(raw_samples, model_annealer.h)
    model_annealer.sampler = sampler

    model_annealer._sample_annealer(5)

    assert sampler.calls[0]["kwargs"]["chain_strength"] == model_annealer.J_range.max()


def test__sample_annealer_binary_true_converts_sample_values(
    model_annealer: BQRBM,
) -> None:
    rng = np.random.RandomState(12)
    model_annealer.b = rng.normal(size=n_qubits)
    model_annealer.W = rng.normal(size=(n_visible, n_hidden))
    raw_samples = rng.choice([-1, 1], size=(5, n_qubits)).astype(np.int8)
    sampler = FakeSampler(raw_samples, model_annealer.h)
    model_annealer.sampler = sampler

    samples = model_annealer._sample_annealer(5, use_gauge=False, binary=True)

    assert np.array_equal(
        samples.record.sample, ((1 - raw_samples) / 2).astype(np.int8)
    )
    assert set(np.unique(samples.record.sample)) == {0, 1}


def test__sample_annealer_uninitialized_sampler_raises_runtime_error(
    model_annealer: BQRBM,
) -> None:
    model_annealer.sampler = None

    with pytest.raises(RuntimeError, match="Annealer sampler is not initialized"):
        model_annealer._sample_annealer(5)


def test_save_annealer_model_without_reinitialization(
    monkeypatch: Any, model_annealer: BQRBM, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )
    file_path = tmp_path / "model.pkl"

    model_annealer.save(file_path, reinitialize_annealer=False)

    assert model_annealer.qpu is None
    assert model_annealer.sampler is None
    loaded_model = BQRBM.load(file_path, initialize_annealer=False)
    assert loaded_model.annealer_params == model_annealer.annealer_params


def test_save_annealer_model_reinitializes_annealer(
    monkeypatch: Any, model_annealer: BQRBM, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )
    model_annealer.h_range = np.array([-99, 99])
    file_path = tmp_path / "model.pkl"

    model_annealer.save(file_path, reinitialize_annealer=True)

    assert np.array_equal(model_annealer.h_range, np.array([-4, 4]))


def test_load_annealer_model_reinitializes_annealer(
    monkeypatch: Any, model_annealer: BQRBM, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )
    file_path = tmp_path / "model.pkl"
    model_annealer.save(file_path, reinitialize_annealer=False)

    loaded_model = BQRBM.load(file_path, initialize_annealer=True)

    assert np.array_equal(loaded_model.h_range, np.array([-4, 4]))


def test_train_second_call_accumulates_callback_history() -> None:
    model = make_small_simulation_model()

    model.train(
        n_epochs=1,
        learning_rate=1e-3,
        mini_batch_size=10,
        n_samples=50,
        callback=lambda model, samples: {"value": 1},
    )
    model.train(
        n_epochs=1,
        learning_rate=1e-3,
        mini_batch_size=10,
        n_samples=50,
        callback=lambda model, samples: {"value": 2},
    )

    assert model.callback_history == [{"value": 1}, {"value": 2}]
