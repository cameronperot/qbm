from typing import Any

import numpy as np

from qbm.models import BQRBM


def test__apply_grads_updates_b_and_W_by_scaled_difference(
    model_simulation: BQRBM,
) -> None:
    rng = np.random.RandomState(42)
    b_initial = model_simulation.b.copy()
    W_initial = model_simulation.W.copy()
    learning_rate = 0.25
    model_simulation.grads = {
        "b_pos": rng.normal(size=b_initial.shape),
        "b_neg": rng.normal(size=b_initial.shape),
        "W_pos": rng.normal(size=W_initial.shape),
        "W_neg": rng.normal(size=W_initial.shape),
    }
    grads = model_simulation.grads

    model_simulation._apply_grads(learning_rate)

    assert np.allclose(
        model_simulation.b,
        b_initial + learning_rate * (grads["b_pos"] - grads["b_neg"]),
    )
    assert np.allclose(
        model_simulation.W,
        W_initial + learning_rate * (grads["W_pos"] - grads["W_neg"]),
    )


def test__binary_to_eigen_known_values(model_simulation: BQRBM) -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    result = model_simulation._binary_to_eigen(x)

    assert np.array_equal(result, np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]))
    assert result.dtype == np.int8


def test__eigen_to_binary_known_values(model_simulation: BQRBM) -> None:
    x = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

    result = model_simulation._eigen_to_binary(x)

    assert np.array_equal(result, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    assert result.dtype == np.int8


def test__binary_to_eigen_and_back_round_trip(model_simulation: BQRBM) -> None:
    rng = np.random.RandomState(7)
    x = rng.randint(0, 2, size=(10, model_simulation.n_qubits))

    round_trip = model_simulation._eigen_to_binary(model_simulation._binary_to_eigen(x))

    assert np.array_equal(round_trip, x.astype(np.int8))


def test__random_mini_batch_indices_partitions_V_train(
    model_simulation: BQRBM,
) -> None:
    n_train = model_simulation.V_train.shape[0]
    mini_batch_size = 300

    batches = model_simulation._random_mini_batch_indices(mini_batch_size)

    assert [batch.shape[0] for batch in batches] == [300, 300, 300, 100]
    concatenated = np.concatenate(batches)
    assert sorted(concatenated.tolist()) == list(range(n_train))


def test__random_mini_batch_indices_same_seed_same_permutation(
    monkeypatch: Any, V_train: np.ndarray
) -> None:
    monkeypatch.setattr("qbm.models.BQRBM._initialize_annealer", lambda model: None)
    kwargs: dict[str, Any] = {
        "V_train": V_train,
        "n_hidden": 2,
        "A_freeze": 0.1,
        "B_freeze": 1.1,
        "simulation_params": {"beta": 1.5},
    }
    model_1 = BQRBM(seed=0, **kwargs)
    model_2 = BQRBM(seed=0, **kwargs)
    model_3 = BQRBM(seed=1, **kwargs)

    batches_1 = model_1._random_mini_batch_indices(10)
    batches_2 = model_2._random_mini_batch_indices(10)
    batches_3 = model_3._random_mini_batch_indices(10)

    assert all(
        np.array_equal(batch_1, batch_2)
        for batch_1, batch_2 in zip(batches_1, batches_2, strict=True)
    )
    assert any(
        not np.array_equal(batch_1, batch_3)
        for batch_1, batch_3 in zip(batches_1, batches_3, strict=True)
    )


def test__random_mini_batch_indices_divisible_batch_size(
    model_simulation: BQRBM,
) -> None:
    batches = model_simulation._random_mini_batch_indices(500)

    assert [batch.shape[0] for batch in batches] == [500, 500]
