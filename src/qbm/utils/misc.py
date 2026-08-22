from __future__ import annotations

import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import MT19937, RandomState, SeedSequence


def compute_df_ensemble_stats(
    dfs: Sequence[pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Computes the means, medians, and standard deviations column/row-wise over the input
    list of dataframes.

    Args:
        dfs: List of dataframes with identical row/column names.

    Returns:
        Dictionary of dataframes with the means, medians, and standard deviations.

    Raises:
        ValueError: If dfs is empty.
        TypeError: If any of the computed statistics is not a DataFrame.
    """
    if len(dfs) == 0:
        raise ValueError("dfs must not be empty")
    df = pd.concat(dfs)
    means = df.groupby(df.index).mean()
    medians = df.groupby(df.index).median()
    stds = df.groupby(df.index).std()
    if not isinstance(means, pd.DataFrame):
        raise TypeError("Grouped means is not a DataFrame")
    if not isinstance(medians, pd.DataFrame):
        raise TypeError("Grouped medians is not a DataFrame")
    if not isinstance(stds, pd.DataFrame):
        raise TypeError("Grouped stds is not a DataFrame")

    return {"means": means, "medians": medians, "stds": stds}


def compute_df_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the min, max, mean, median, and standard deviation of the columns in the
    dataframe.

    Args:
        df: Dataframe.

    Returns:
        Dataframe of the statistics.
    """
    return pd.DataFrame.from_dict(
        {
            "min": df.min(),
            "max": df.max(),
            "mean": df.mean(),
            "median": df.median(),
            "std": df.std(),
        },
        orient="index",
    )


def filter_df_on_values(
    df: pd.DataFrame,
    column_values: Mapping[Any, Any],
    drop_filter_columns: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of the dataframe filtered conditionally on provided
    column values.

    Args:
        df: Dataframe to filter.
        column_values: Dictionary where the keys are column names, and the
            values are values on which to filter the dataframe.
        drop_filter_columns: If True returns a copy of the dataframe with
            the filtered columns dropped.

    Returns:
        A dataframe filtered conditionally on the provided column values.
    """
    df = df.copy()
    for column, value in column_values.items():
        df = df.loc[df[column] == value]

    if drop_filter_columns:
        df.drop(column_values.keys(), axis=1, inplace=True)

    return df


def get_project_dir() -> Path:
    """
    Gets the project directory path from the environment and checks if it is valid.

    Returns:
        Path object of the project directory.

    Raises:
        RuntimeError: If the QBM_PROJECT_DIR env var is not set.
        FileNotFoundError: If the path does not exist.
    """
    dir_path = os.getenv("QBM_PROJECT_DIR")
    if dir_path is None:
        raise RuntimeError("QBM_PROJECT_DIR env var not set")

    dir_path = Path(dir_path)
    if dir_path.exists():
        return dir_path
    else:
        raise FileNotFoundError(f"Path '{dir_path}' does not exist")


def get_rng(seed: int | None = None) -> RandomState:
    """
    Creates a random number generator with the specified seed value.

    Args:
        seed: Seed value for the rng.

    Returns:
        Numpy RandomState object.
    """
    return RandomState(MT19937(SeedSequence(seed)))


def compute_kl_divergence(
    p_data: np.ndarray,
    q_data: np.ndarray,
    n_bins: int = 32,
    epsilon_smooth: float | None = None,
    relative_smooth: bool = False,
) -> float:
    """
    Computes the D_KL(p_data || q_data).

    Note:
        this is a crude approximation of the KL divergence.

    Args:
        p_data: Array of data values to compute the p distribution from.
        q_data: Array of data values to compute the q distribution from.
        n_bins: Number of bins to use in histograms.
        epsilon_smooth: Value to use with q distribution smoothing.
        relative_smooth: Whether or not the smoothed values are relative to the p
            distribution.

    Returns:
        D_KL(p || q).

    Raises:
        ValueError: If p_data or q_data is empty, if n_bins is not positive, or if
            either distribution does not sum to 1.
    """
    if p_data.shape[0] == 0:
        raise ValueError("p_data must not be empty")
    if q_data.shape[0] == 0:
        raise ValueError("q_data must not be empty")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive (got {n_bins})")
    hist_data, bin_edges = np.histogram(p_data, bins=n_bins)
    hist_samples, _ = np.histogram(q_data, bins=bin_edges)

    p = hist_data / p_data.shape[0]
    q = hist_samples / q_data.shape[0]

    if epsilon_smooth is not None:
        smooth_mask = np.logical_and(p > 0, q == 0)
        not_smooth_mask = np.logical_not(smooth_mask)
        q[smooth_mask] = epsilon_smooth

        if relative_smooth:
            q[smooth_mask] *= p[smooth_mask]

        q[not_smooth_mask] -= q[smooth_mask].sum() / not_smooth_mask.sum()

    if not np.isclose(p.sum(), 1, atol=1e-3):
        raise ValueError(f"p distribution does not sum to 1 (sums to {p.sum()})")
    if not np.isclose(q.sum(), 1, atol=1e-3):
        raise ValueError(f"q distribution does not sum to 1 (sums to {q.sum()})")

    support = np.logical_and(p > 0, q > 0)
    p = p[support]
    q = q[support]

    return (p * np.log(p / q)).sum()


def load_artifact(file_path: str | Path) -> Any:
    """
    Loads a pickle or json artifact (depending on the file extension).

    Args:
        file_path: Path of the file to load.

    Returns:
        Loaded python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has an unsupported file extension.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    if file_path.suffix not in (".json", ".pkl"):
        raise ValueError(
            f"File {file_path} has an unsupported extension "
            f"'{file_path.suffix}' (must be '.json' or '.pkl')"
        )

    if file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)


def compute_lr_exp_decay(
    epoch: float | Sequence[float] | np.ndarray,
    decay_epoch: float,
    period: float,
    base: float = 2.0,
) -> float | np.ndarray:
    """
    Exponential decay function for use in learning rate scheduling. It is relative, so
    one must multiply the base learning rate by the output of this function.

    Args:
        epoch: Current epoch (scalar or array of epochs).
        decay_epoch: Epoch at which to begin the decay.
        period: Decay period.
        base: Base number of the exponential decay.

    Returns:
        The learning rate scaling factor (scalar or array, matching the input).
    """
    epoch_array = np.asarray(epoch)
    return base ** (np.minimum(decay_epoch - epoch_array, 0) / period)


def save_artifact(artifact: Any, file_path: str | Path) -> None:
    """
    Saves a pickle or json artifact (depending on the file extension).

    Args:
        artifact: Python object to save.
        file_path: Path of the file to save.

    Raises:
        ValueError: If the file has an unsupported file extension.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    if file_path.suffix not in (".json", ".pkl"):
        raise ValueError(
            f"File {file_path} has an unsupported extension "
            f"'{file_path.suffix}' (must be '.json' or '.pkl')"
        )

    if file_path.suffix == ".json":
        with open(file_path, "w") as f:
            json.dump(artifact, f, indent=4)
    elif file_path.suffix == ".pkl":
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)


def compute_lower_tail_concentration(
    z: float | np.ndarray, U: np.ndarray, V: np.ndarray
) -> float | np.ndarray:
    """
    Lower tail concentration function defined as:
    L(z) = P(U <= z | V <= z) = P(U <= z, V <= z) / P(U <= z)
    References:
        - https://freakonometrics.hypotheses.org/2435
        - https://openacttexts.github.io/Loss-Data-Analytics/C-DependenceModel
            (section 14.5.4.3)
        - https://www.casact.org/sites/default/files/old/studynotes_venter_tails_of_copulas.pdf
            (section 3)

    Args:
        z: Tail dependence parameter (scalar or array of parameters).
        U: Input array for first variable (e.g. X.rank() / (len(X) + 1)).
        V: Input array for second variable (e.g. Y.rank() / (len(Y) + 1)).

    Returns:
        Lower tail concentration function (scalar or array, one value per z).
    """
    z_expanded = np.asarray(z)[..., np.newaxis]
    return np.sum(np.logical_and(z_expanded >= U, z_expanded >= V), axis=-1) / np.sum(
        z_expanded >= U, axis=-1
    )


def compute_upper_tail_concentration(
    z: float | np.ndarray, U: np.ndarray, V: np.ndarray
) -> float | np.ndarray:
    """
    Upper tail concentration function defined as:
    R(z) = P(U > z | V > z) = P(U > z, V > z) / P(U > z)
    References:
        - https://freakonometrics.hypotheses.org/2435
        - https://openacttexts.github.io/Loss-Data-Analytics/C-DependenceModel
            (section 14.5.4.3)
        - https://www.casact.org/sites/default/files/old/studynotes_venter_tails_of_copulas.pdf
            (section 3)

    Args:
        z: Tail dependence parameter (scalar or array of parameters).
        U: Input array for first variable (e.g. X.rank() / (len(X) + 1)).
        V: Input array for second variable (e.g. Y.rank() / (len(Y) + 1)).

    Returns:
        Upper tail concentration function (scalar or array, one value per z).
    """
    z_expanded = np.asarray(z)[..., np.newaxis]
    return np.sum(np.logical_and(z_expanded < U, z_expanded < V), axis=-1) / np.sum(
        z_expanded < U, axis=-1
    )
