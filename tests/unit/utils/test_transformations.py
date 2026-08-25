import numpy as np
import pandas as pd
import pytest

from qbm.utils import PowerTransformer


@pytest.fixture
def df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "a": np.arange(1024),
            "b": np.linspace(0, 10, 1024),
            "c": np.linspace(-10, 10, 1024),
        }
    )

    return df


def test_PowerTransformer_init_default(df: pd.DataFrame) -> None:
    transformer = PowerTransformer(df)

    assert transformer.power == 0.5
    assert transformer.threshold == 1
    assert set(transformer.columns) == set(df.columns)
    for column in df.columns:
        assert df[column].mean() == transformer.μ[column]
        assert df[column].std() == transformer.σ[column]


def test_PowerTransformer_init_kwargs(df: pd.DataFrame) -> None:
    power = 0.1
    threshold = 1.1
    columns = ["a", "b"]

    transformer = PowerTransformer(
        df, power=power, threshold=threshold, columns=columns
    )

    assert transformer.power == power
    assert transformer.threshold == threshold
    assert set(transformer.columns) == set(columns)
    for column in df.columns:
        assert df[column].mean() == transformer.μ[column]
        assert df[column].std() == transformer.σ[column]


def test_PowerTransformer_transform_inplace(df: pd.DataFrame) -> None:
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)
    df_inplace = transformer.transform(df, inplace=True)

    assert df_transformed.equals(df)
    assert df_inplace is df


def test_PowerTransformer_transform_all_columns(df: pd.DataFrame) -> None:
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


def test_PowerTransformer_transform_subset_columns(df: pd.DataFrame) -> None:
    power = 0.5
    threshold = 1
    columns = ["a", "b"]

    transformer = PowerTransformer(
        df, power=power, threshold=threshold, columns=columns
    )
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


def test_PowerTransformer_inverse_transform_inplace(df: pd.DataFrame) -> None:
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)
    df_inverse_transformed = transformer.inverse_transform(df_transformed)
    df_inplace = transformer.inverse_transform(df_transformed, inplace=True)

    assert df_inverse_transformed.equals(df_transformed)
    assert df_inplace is df_transformed


def test_PowerTransformer_inverse_transform_all_columns(df: pd.DataFrame) -> None:
    power = 0.5
    threshold = 1

    transformer = PowerTransformer(df, power=power, threshold=threshold)
    df_transformed = transformer.transform(df)
    df_inverse_transformed = transformer.inverse_transform(df_transformed)

    for column in df.columns:
        assert np.isclose(df_inverse_transformed[column], df[column]).all()


def test_PowerTransformer_inverse_transform_subset_columns(
    df: pd.DataFrame,
) -> None:
    power = 0.5
    threshold = 1
    columns = ["a", "b"]

    transformer = PowerTransformer(
        df, power=power, threshold=threshold, columns=columns
    )
    df_transformed = transformer.transform(df)
    df_inverse_transformed = transformer.inverse_transform(df_transformed)

    for column in columns:
        assert np.isclose(df_inverse_transformed[column], df[column]).all()

    for column in set(df.columns) - set(columns):
        assert (df_inverse_transformed[column] == df[column]).all()


def test_PowerTransformer_init_invalid_power_raises_value_error(
    df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="power must be < 1"):
        PowerTransformer(df, power=1.0)


def test_PowerTransformer_init_nonpositive_power_raises_value_error(
    df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="power must be > 0"):
        PowerTransformer(df, power=0.0)


def test_PowerTransformer_init_invalid_threshold_raises_value_error(
    df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        PowerTransformer(df, threshold=0.5)


def test_PowerTransformer_init_unknown_columns_raises_value_error(
    df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="not a subset"):
        PowerTransformer(df, columns=["a", "d"])


def test_PowerTransformer_transform_missing_column_raises_value_error(
    df: pd.DataFrame,
) -> None:
    transformer = PowerTransformer(df)

    with pytest.raises(ValueError, match="missing configured columns"):
        transformer.transform(df.drop(columns=["a"]))


def test_PowerTransformer_inverse_transform_missing_column_raises_value_error(
    df: pd.DataFrame,
) -> None:
    transformer = PowerTransformer(df)

    with pytest.raises(ValueError, match="missing configured columns"):
        transformer.inverse_transform(df.drop(columns=["a"]))
