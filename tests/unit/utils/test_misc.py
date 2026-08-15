import json
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

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


def test_compute_df_ensemble_stats(monkeypatch: Any) -> None:
    df = pd.DataFrame(np.arange(6).reshape((3, 2))).astype(np.float64)

    ensemble_stats = compute_df_ensemble_stats([df, df])

    assert ensemble_stats["means"].equals(df)
    assert ensemble_stats["medians"].equals(df)
    assert ensemble_stats["stds"].equals(pd.DataFrame(np.zeros((3, 2))))


def test_compute_df_stats(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.min", lambda self: 1)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.max", lambda self: 2)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.mean", lambda self: 3)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.median", lambda self: 4)
    monkeypatch.setattr("qbm.utils.misc.pd.DataFrame.std", lambda self: 5)

    df = pd.DataFrame(np.ones((2, 3)))
    expected_df = pd.DataFrame.from_dict(
        {"min": 1, "max": 2, "mean": 3, "median": 4, "std": 5}, orient="index"
    )

    stats = compute_df_stats(df)

    assert stats.equals(expected_df)


@patch("qbm.utils.misc.np.logical_and")
@patch("qbm.utils.misc.np.sum")
def test_compute_lower_tail_concentration(mock_sum: Any, mock_logical_and: Any) -> None:
    mock_sum.return_value = 123
    mock_logical_and.return_value = "test_logical_and"

    z = 0.5
    U = np.linspace(0, 0.9, 100)
    V = np.linspace(0, 1, 100)

    ltc = compute_lower_tail_concentration(z, U, V)

    assert (mock_logical_and.call_args[0][0] == (z >= U)).all()
    assert (mock_logical_and.call_args[0][1] == (z >= V)).all()
    assert mock_sum.call_args_list[0][0][0] == "test_logical_and"
    assert (mock_sum.call_args_list[1][0][0] == (z >= U)).all()
    assert ltc == 1


@patch("qbm.utils.misc.np.logical_and")
@patch("qbm.utils.misc.np.sum")
def test_compute_upper_tail_concentration(mock_sum: Any, mock_logical_and: Any) -> None:
    mock_sum.return_value = 123
    mock_logical_and.return_value = "test_logical_and"

    z = 0.5
    U = np.linspace(0, 0.9, 100)
    V = np.linspace(0, 1, 100)

    utc = compute_upper_tail_concentration(z, U, V)

    assert (mock_logical_and.call_args[0][0] == (1 - z < U)).all()
    assert (mock_logical_and.call_args[0][1] == (1 - z < V)).all()
    assert mock_sum.call_args_list[0][0][0] == "test_logical_and"
    assert (mock_sum.call_args_list[1][0][0] == (1 - z < U)).all()
    assert utc == 1


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


def test_get_project_dir_env_not_set(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.os.getenv", lambda x: None)

    with pytest.raises(Exception, match="QBM_PROJECT_DIR env var not set"):
        get_project_dir()


def test_get_project_dir_path_does_not_exist(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.os.getenv", lambda x: "/test/path")
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: False)

    with pytest.raises(Exception, match="does not exist"):
        get_project_dir()


def test_get_project_dir_success(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.os.getenv", lambda x: "/test/path")
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    project_dir = get_project_dir()
    assert project_dir == Path("/test/path")


@patch("qbm.utils.misc.SeedSequence")
@patch("qbm.utils.misc.MT19937")
@patch("qbm.utils.misc.RandomState")
def test_get_rng(
    mock_RandomState: Any, mock_MT19937: Any, mock_SeedSequence: Any
) -> None:
    mock_MT19937.return_value = "test_MT19937"
    mock_SeedSequence.return_value = "test_SeedSequence"
    mock_RandomState.return_value = "test_RandomState"

    seed = 42
    rng = get_rng(seed)

    mock_SeedSequence.assert_called_with(seed)
    mock_MT19937.assert_called_with("test_SeedSequence")
    mock_RandomState.assert_called_with("test_MT19937")
    assert rng == "test_RandomState"


def test_compute_kl_divergence() -> None:
    n_bins = 32
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    hist_data, bin_edges = np.histogram(p_data, bins=n_bins)
    hist_samples, _ = np.histogram(q_data, bins=bin_edges)
    p = hist_data / p_data.shape[0]
    q = hist_samples / q_data.shape[0]

    support = np.logical_and(p > 0, q > 0)
    p = p[support]
    q = q[support]

    assert compute_kl_divergence(p_data, q_data) == np.sum(p * np.log(p / q))


def test_compute_kl_divergence_zero() -> None:
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-10, 10, 1000)

    assert compute_kl_divergence(p_data, q_data) == 0


def test_compute_kl_divergence_nonzero() -> None:
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    assert compute_kl_divergence(p_data, q_data) != 0


def test_compute_kl_divergence_relative_smooth() -> None:
    n_bins = 32
    epsilon_smooth = 1e-3
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    hist_data, bin_edges = np.histogram(p_data, bins=n_bins)
    hist_samples, _ = np.histogram(q_data, bins=bin_edges)
    p = hist_data / p_data.shape[0]
    q = hist_samples / q_data.shape[0]

    smooth_mask = np.logical_and(p > 0, q == 0)
    q[smooth_mask] = epsilon_smooth * p[smooth_mask]
    q[np.logical_not(smooth_mask)] -= q[smooth_mask].sum() / (
        len(q) - smooth_mask.sum()
    )

    support = np.logical_and(p > 0, q > 0)
    p = p[support]
    q = q[support]

    assert compute_kl_divergence(
        p_data, q_data, epsilon_smooth=epsilon_smooth, relative_smooth=True
    ) == np.sum(p * np.log(p / q))


def test_compute_kl_divergence_smooth() -> None:
    n_bins = 32
    epsilon_smooth = 1e-3
    p_data = np.linspace(-10, 10, 1000)
    q_data = np.linspace(-1, 1, 1000)

    hist_data, bin_edges = np.histogram(p_data, bins=n_bins)
    hist_samples, _ = np.histogram(q_data, bins=bin_edges)
    p = hist_data / p_data.shape[0]
    q = hist_samples / q_data.shape[0]

    smooth_mask = np.logical_and(p > 0, q == 0)
    q[smooth_mask] = epsilon_smooth
    q[np.logical_not(smooth_mask)] -= q[smooth_mask].sum() / (
        len(q) - smooth_mask.sum()
    )

    support = np.logical_and(p > 0, q > 0)
    p = p[support]
    q = q[support]

    result = compute_kl_divergence(p_data, q_data, epsilon_smooth=epsilon_smooth)
    assert result == np.sum(p * np.log(p / q))


def test_load_artifact_invalid_file_path(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: False)

    file_path = Path("/test/path/file")

    with pytest.raises(Exception, match="does not exist"):
        load_artifact(file_path)


def test_load_artifact_invalid_file_extension(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file")

    with pytest.raises(Exception, match="does not exist"):
        load_artifact(file_path)


def test_load_artifact_json_success(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.json")
    artifact = {"a": 1, "b": 2}
    json_ = json.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=json_)) as mock_file:
        loaded_artifact = load_artifact(file_path)

        assert loaded_artifact == artifact
        mock_file.assert_called_with(file_path)


def test_load_artifact_pickle_success(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.pkl")
    artifact = {"a": 1, "b": 2}
    pickle_ = pickle.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=pickle_)) as mock_file:
        loaded_artifact = load_artifact(file_path)

        assert loaded_artifact == artifact
        mock_file.assert_called_with(file_path, "rb")


def test_load_artifact_json_success_str(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = "/test/path/file.json"
    artifact = {"a": 1, "b": 2}
    json_ = json.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=json_)) as mock_file:
        loaded_artifact = load_artifact(file_path)

        assert loaded_artifact == artifact
        mock_file.assert_called_with(Path(file_path))


def test_load_artifact_pickle_success_str(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = "/test/path/file.pkl"
    artifact = {"a": 1, "b": 2}
    pickle_ = pickle.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=pickle_)) as mock_file:
        loaded_artifact = load_artifact(file_path)

        assert loaded_artifact == artifact
        mock_file.assert_called_with(Path(file_path), "rb")


@pytest.mark.parametrize(
    "epoch, decay_epoch, period", [(0, 5, 10), (5, 5, 10), (6, 5, 10)]
)
def test_compute_lr_exp_decay(epoch: int, decay_epoch: int, period: int) -> None:
    lr_factor = compute_lr_exp_decay(epoch, decay_epoch, period)

    if epoch <= decay_epoch:
        assert lr_factor == 1
    else:
        assert lr_factor == 2 ** ((decay_epoch - epoch) / period)


@patch("qbm.utils.misc.Path.mkdir")
def test_save_artifact_parent_does_not_exist(mock_mkdir: Any, monkeypatch: Any) -> None:
    mock_mkdir.return_value = None
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: False)

    file_path = Path("/test/path/file.json")
    artifact = {"a": 1, "b": 2}
    json_ = json.dumps(artifact)

    with patch("builtins.open", mock_open(read_data=json_)):
        save_artifact(artifact, file_path)
        mock_mkdir.assert_called_with(parents=True)


def test_save_artifact_invalid_suffix(monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.invalid")
    artifact = {"a": 1, "b": 2}

    with pytest.raises(Exception, match="Invalid file extension"):
        save_artifact(artifact, file_path)


@patch("qbm.utils.misc.json.dump")
def test_save_artifact_json_success(mock_dump: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.json")
    artifact = {"a": 1, "b": 2}

    with patch("builtins.open") as mock_file:
        save_artifact(artifact, file_path)

        mock_file.assert_called_with(file_path, "w")
        mock_dump.assert_called()


@patch("qbm.utils.misc.pickle.dump")
def test_save_artifact_pickle_success(mock_dump: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = Path("/test/path/file.pkl")
    artifact = {"a": 1, "b": 2}

    with patch("builtins.open") as mock_file:
        save_artifact(artifact, file_path)

        mock_file.assert_called_with(file_path, "wb")
        mock_dump.assert_called()


@patch("qbm.utils.misc.json.dump")
def test_save_artifact_json_success_str(mock_dump: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = "/test/path/file.json"
    artifact = {"a": 1, "b": 2}

    with patch("builtins.open") as mock_file:
        save_artifact(artifact, file_path)

        mock_file.assert_called_with(Path(file_path), "w")
        mock_dump.assert_called()


@patch("qbm.utils.misc.pickle.dump")
def test_save_artifact_pickle_success_str_path(
    mock_dump: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr("qbm.utils.misc.Path.exists", lambda self: True)

    file_path = "/test/path/file.pkl"
    artifact = {"a": 1, "b": 2}

    with patch("builtins.open") as mock_file:
        save_artifact(artifact, file_path)

        mock_file.assert_called_with(Path(file_path), "wb")
        mock_dump.assert_called()
