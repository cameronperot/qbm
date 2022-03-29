import json
import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch, mock_open

from qbm.utils import PowerTransformer


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "a": np.arange(1024),
            "b": np.linspace(0, 10, 1024),
            "c": np.linspace(-10, 10, 1024),
        }
    )

    return df


def test_PowerTransformer_init_default(df):
    transformer = PowerTransformer(df)

    assert transformer.power == 0.5
    assert transformer.threshold == 1
    assert set(transformer.columns) == set(df.columns)
    for column in df.columns:
        assert df[column].mean() == transformer.μ[column]
        assert df[column].std() == transformer.σ[column]


def test_PowerTransformer_init_kwargs(df):
    power = 0.1
    threshold = 1.1
    columns = ["a", "b"]

    transformer = PowerTransformer(df, power=power, threshold=threshold, columns=columns)

    assert transformer.power == power
    assert transformer.threshold == threshold
    assert set(transformer.columns) == set(columns)
    for column in df.columns:
        assert df[column].mean() == transformer.μ[column]
        assert df[column].std() == transformer.σ[column]


def test_PowerTransformer_transform_inplace(df):
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)
    transformer.transform(df, inplace=True)

    assert df_transformed.equals(df)


def test_PowerTransformer_transform_all_columns(df):
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)

    for column in df.columns:
        x = df[column]
        x_standardized = (x - x.mean()) / x.std()
        assert (np.diff(df_transformed[column]) >= 0).all()
        assert np.isclose(
            df_transformed[column][np.abs(x_standardized) <= threshold],
            x[np.abs(x_standardized) <= threshold],
        ).all()
        assert np.logical_not(
            np.isclose(
                df_transformed[column][np.abs(x_standardized) > threshold],
                x[np.abs(x_standardized) > threshold],
            )
        ).all()


def test_PowerTransformer_transform_subset_columns(df):
    power = 0.5
    threshold = 1
    columns = ["a", "b"]

    transformer = PowerTransformer(df, power=power, threshold=threshold, columns=columns)
    df_transformed = transformer.transform(df)

    for column in columns:
        x = df[column]
        x_standardized = (x - x.mean()) / x.std()
        assert (np.diff(df_transformed[column]) >= 0).all()
        assert np.isclose(
            df_transformed[column][np.abs(x_standardized) <= threshold],
            x[np.abs(x_standardized) <= threshold],
        ).all()
        assert np.logical_not(
            np.isclose(
                df_transformed[column][np.abs(x_standardized) > threshold],
                x[np.abs(x_standardized) > threshold],
            )
        ).all()

    for column in set(df.columns) - set(columns):
        assert (df_transformed[column] == df[column]).all()


def test_PowerTransformer_inverse_transform_inplace(df):
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)
    df_inverse_transformed = transformer.inverse_transform(df_transformed)
    transformer.inverse_transform(df_transformed, inplace=True)

    assert df_inverse_transformed.equals(df_transformed)


def test_PowerTransformer_inverse_transform_all_columns(df):
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)
    df_inverse_transformed = transformer.inverse_transform(df_transformed)

    for column in df.columns:
        assert np.isclose(df_inverse_transformed[column], df[column]).all()


def test_PowerTransformer_inverse_transform_subset_columns(df):
    power = 0.5
    threshold = 1
    columns = ["a", "b"]

    transformer = PowerTransformer(df, power=power, threshold=threshold, columns=columns)
    df_transformed = transformer.transform(df)
    df_inverse_transformed = transformer.inverse_transform(df_transformed)

    for column in columns:
        assert np.isclose(df_inverse_transformed[column], df[column]).all()

    for column in set(df.columns) - set(columns):
        assert (df_inverse_transformed[column] == df[column]).all()
