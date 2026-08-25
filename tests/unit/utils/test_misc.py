from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qbm.utils import (
    compute_df_ensemble_stats,
    compute_df_stats,
    compute_kl_divergence,
    compute_lower_tail_concentration,
    compute_lr_exp_decay,
    compute_upper_tail_concentration,
    filter_df_on_values,
    get_project_dir,
    get_rng,
    load_artifact,
    save_artifact,
)


@pytest.fixture
def df() -> pd.DataFrame:
    n_rows = 100
    return pd.DataFrame(
        {
            "a": np.linspace(0, 1, n_rows),
            "b": np.linspace(-1, 1, n_rows),
            "c": np.concatenate(
                [np.zeros(round(n_rows / 2)), np.ones(round(n_rows / 2))]
            ),
            "d": np.concatenate(
                [np.zeros(round(n_rows / 3)), np.ones(round(2 * n_rows / 3))]
            ),
        }
    )


def test_compute_df_ensemble_stats_identical_dfs_returns_input_values() -> None:
    df_input = pd.DataFrame(np.arange(6).reshape((3, 2))).astype(np.float64)

    ensemble_stats = compute_df_ensemble_stats([df_input, df_input])

    assert ensemble_stats["means"].equals(df_input)
    assert ensemble_stats["medians"].equals(df_input)
    assert ensemble_stats["stds"].equals(pd.DataFrame(np.zeros((3, 2))))


def test_compute_df_stats_known_values_returns_concrete_statistics() -> None:
    df_input = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})

    stats = compute_df_stats(df_input)

    # std with pandas default ddof=1: sqrt((2.25 + 0.25 + 0.25 + 2.25) / 3)
    expected = pd.DataFrame(
        {
            "a": [1.0, 4.0, 2.5, 2.5, np.sqrt(5 / 3)],
            "b": [10.0, 40.0, 25.0, 25.0, 10.0 * np.sqrt(5 / 3)],
        },
        index=pd.Index(["min", "max", "mean", "median", "std"]),
    )
    pd.testing.assert_frame_equal(stats, expected)


@pytest.mark.parametrize(
    ("z", "expected"),
    [(0.6, 2 / 3), (0.95, 1.0), (0.25, 0.0)],
    ids=["partial_overlap", "all_below_threshold", "no_joint_observations"],
)
def test_compute_lower_tail_concentration_known_values(
    z: float, expected: float
) -> None:
    # U and V chosen so hand-counted overlaps give non-trivial fractions
    U = np.array([0.9, 0.2, 0.5, 0.3, 0.7])
    V = np.array([0.4, 0.8, 0.1, 0.6, 0.9])

    assert compute_lower_tail_concentration(z, U, V) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("z", "expected"),
    [(0.5, 0.5), (0.15, 0.8), (0.85, 0.0)],
    ids=["partial_overlap", "most_above_threshold", "no_joint_observations"],
)
def test_compute_upper_tail_concentration_known_values(
    z: float, expected: float
) -> None:
    U = np.array([0.9, 0.2, 0.5, 0.3, 0.7])
    V = np.array([0.4, 0.8, 0.1, 0.6, 0.9])

    assert compute_upper_tail_concentration(z, U, V) == pytest.approx(expected)


def test_compute_tail_concentration_array_z_consistent_with_scalar_z() -> None:
    z = np.linspace(0.1, 0.9, 17)
    U = np.linspace(0, 0.9, 100)
    V = np.linspace(0, 1, 100)

    ltc = compute_lower_tail_concentration(z, U, V)
    utc = compute_upper_tail_concentration(z, U, V)

    ltc_reference = np.array([compute_lower_tail_concentration(z_i, U, V) for z_i in z])
    utc_reference = np.array([compute_upper_tail_concentration(z_i, U, V) for z_i in z])

    assert np.allclose(ltc, ltc_reference, equal_nan=True)
    assert np.allclose(utc, utc_reference, equal_nan=True)


def test_filter_df_on_values_drop_filter_columns_False(df: pd.DataFrame) -> None:
    column_values = {"c": 0, "d": 1}

    df_filtered = filter_df_on_values(df, column_values, drop_filter_columns=False)

    assert "c" in df_filtered.columns
    assert "d" in df_filtered.columns
    assert (df_filtered["c"] == column_values["c"]).all()
    assert (df_filtered["d"] == column_values["d"]).all()
    assert df_filtered.shape[0] == round(df.shape[0] * (1 / 2 - 1 / 3))


def test_filter_df_on_values_drop_filter_columns_True(df: pd.DataFrame) -> None:
    column_values = {"c": 0, "d": 1}

    df_filtered = filter_df_on_values(df, column_values, drop_filter_columns=True)

    assert "c" not in df_filtered.columns
    assert "d" not in df_filtered.columns
    assert df_filtered.shape[0] == round(df.shape[0] * (1 / 2 - 1 / 3))


def test_get_project_dir_env_not_set_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QBM_PROJECT_DIR", raising=False)

    with pytest.raises(RuntimeError, match="QBM_PROJECT_DIR env var not set"):
        get_project_dir()


def test_get_project_dir_nonexistent_path_raises_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QBM_PROJECT_DIR", str(tmp_path / "does_not_exist"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_project_dir()


def test_get_project_dir_existing_path_returns_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QBM_PROJECT_DIR", str(tmp_path))

    assert get_project_dir() == tmp_path


def test_get_rng_same_seed_returns_identical_draw_sequences() -> None:
    rng_1 = get_rng(42)
    rng_2 = get_rng(42)

    assert np.array_equal(rng_1.rand(10), rng_2.rand(10))
    assert np.array_equal(rng_1.normal(size=5), rng_2.normal(size=5))


def test_get_rng_different_seeds_return_different_draw_sequences() -> None:
    assert not np.array_equal(get_rng(1).rand(10), get_rng(2).rand(10))


def test_get_rng_seed_none_returns_usable_generator() -> None:
    rng = get_rng()

    assert isinstance(rng, np.random.RandomState)
    assert rng.rand(3).shape == (3,)


def test_compute_kl_divergence_disjoint_supports_known_value() -> None:
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    assert compute_kl_divergence(p_data, q_data) == pytest.approx(-0.2549111859834866)


def test_compute_kl_divergence_identical_distributions_returns_zero() -> None:
    p_data = np.linspace(-10, 10, 1000)

    assert compute_kl_divergence(p_data, p_data) == 0


def test_compute_kl_divergence_relative_smooth_known_value() -> None:
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    result = compute_kl_divergence(
        p_data, q_data, epsilon_smooth=1e-3, relative_smooth=True
    )

    assert result == pytest.approx(5.796398238419606)


def test_compute_kl_divergence_smooth_known_value() -> None:
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    result = compute_kl_divergence(p_data, q_data, epsilon_smooth=1e-3)

    assert result == pytest.approx(2.7651487251655382)


def test_load_artifact_nonexistent_file_raises_file_not_found(tmp_path: Path) -> None:
    file_path = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_artifact(file_path)


def test_load_artifact_invalid_extension_raises_value_error(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("data")

    with pytest.raises(ValueError, match="unsupported extension"):
        load_artifact(file_path)


def test_save_and_load_artifact_json_round_trip(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.json"
    artifact = {"a": 1, "b": [1, 2, 3]}

    save_artifact(artifact, file_path)

    assert load_artifact(file_path) == artifact


def test_save_and_load_artifact_pickle_round_trip(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.pkl"
    artifact = {"a": 1, "b": np.array([1.0, 2.0])}

    save_artifact(artifact, file_path)

    loaded_artifact = load_artifact(file_path)
    assert loaded_artifact["a"] == artifact["a"]
    assert np.array_equal(loaded_artifact["b"], artifact["b"])


def test_save_and_load_artifact_str_path_round_trip(tmp_path: Path) -> None:
    file_path = str(tmp_path / "artifact.json")
    artifact = {"a": 1, "b": 2}

    save_artifact(artifact, file_path)

    assert load_artifact(file_path) == artifact


def test_save_artifact_parent_dir_does_not_exist_creates_parents(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "new_dir" / "artifact.json"
    artifact = {"a": 1}

    save_artifact(artifact, file_path)

    assert file_path.exists()
    assert load_artifact(file_path) == artifact


def test_save_artifact_invalid_suffix_raises_value_error(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.invalid"

    with pytest.raises(ValueError, match="unsupported extension"):
        save_artifact({"a": 1}, file_path)


@pytest.mark.parametrize(
    "epoch, decay_epoch, period",
    [(0, 5, 10), (5, 5, 10), (6, 5, 10)],
    ids=["before_decay", "at_decay_epoch", "after_decay"],
)
def test_compute_lr_exp_decay(epoch: int, decay_epoch: int, period: int) -> None:
    lr_factor = compute_lr_exp_decay(epoch, decay_epoch, period)

    if epoch <= decay_epoch:
        assert lr_factor == 1
    else:
        assert lr_factor == 2 ** ((decay_epoch - epoch) / period)


@pytest.mark.parametrize(
    "epochs",
    [[1, 2, 3], (1, 2, 3)],
    ids=["list", "tuple"],
)
def test_compute_lr_exp_decay_sequence(epochs: list[int] | tuple[int, ...]) -> None:
    lr_factors = compute_lr_exp_decay(epochs, decay_epoch=2, period=3)

    assert np.allclose(lr_factors, [1, 1, 2 ** (-1 / 3)])


def test_compute_df_ensemble_stats_empty_dfs_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dfs must not be empty"):
        compute_df_ensemble_stats([])


def test_compute_kl_divergence_empty_p_data_raises_value_error() -> None:
    with pytest.raises(ValueError, match="p_data must not be empty"):
        compute_kl_divergence(p_data=np.array([]), q_data=np.array([1.0, 2.0]))


def test_compute_kl_divergence_empty_q_data_raises_value_error() -> None:
    with pytest.raises(ValueError, match="q_data must not be empty"):
        compute_kl_divergence(p_data=np.array([1.0, 2.0]), q_data=np.array([]))


def test_compute_kl_divergence_invalid_n_bins_raises_value_error() -> None:
    with pytest.raises(ValueError, match="n_bins must be positive"):
        compute_kl_divergence(
            p_data=np.array([1.0, 2.0]), q_data=np.array([1.0, 2.0]), n_bins=0
        )
