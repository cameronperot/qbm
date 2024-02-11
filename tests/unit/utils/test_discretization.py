import numpy as np
import pandas as pd
import pytest

from qbm.utils import Discretizer

max_bits = 12


@pytest.fixture
def df_and_discretization_params(request):
    n_bits = request.param
    df = pd.DataFrame(
        {
            "a": np.arange(2**n_bits),
            "b": np.linspace(0, 10, 2**n_bits),
            "c": np.linspace(-10, 10, 2**n_bits),
            "a_bit": np.ones(2**n_bits),
            "b_bit": np.zeros(2**n_bits),
            "c_bit": np.ones(2**n_bits),
        }
    )

    return df, n_bits


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_bit_vector_to_int(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    for i in range(2**n_bits):
        bit_vector = discretizer.int_to_bit_vector(i, n_bits)

        assert bit_vector == [int(x) for x in bin(i)[2:].zfill(n_bits)]


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_bit_vector_to_string(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    for i in range(2**n_bits):
        bit_vector = discretizer.int_to_bit_vector(i, n_bits)
        bit_string = discretizer.bit_vector_to_string(bit_vector)

        assert bit_string == bin(i)[2:].zfill(n_bits)


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_int_to_bit_vector(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    for i in range(2**n_bits):
        bit_vector = discretizer.int_to_bit_vector(i, n_bits)
        i_recovered = discretizer.bit_vector_to_int(bit_vector)

        assert i == i_recovered


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_discretize(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    for column in df.columns:
        if column.endswith("_bit"):
            continue

        x_discrete = discretizer.discretize(df[column], **discretizer.params[column])

        assert x_discrete.min() == 0
        assert x_discrete.max() == 2**n_bits - 1
        assert (x_discrete == np.arange(2**n_bits)).all()


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_discretizatio_params(df_and_discretization_params):
    df, n_bits = df_and_discretization_params

    epsilon = {}
    for i, column in enumerate(df.columns):
        epsilon[column] = {"min": -i / 10, "max": i / 10}
    discretizer = Discretizer(df, n_bits, epsilon=epsilon)

    for column in df.columns:
        if column.endswith("_bit"):
            continue

        assert discretizer.params[column]["n_bits"] == n_bits
        assert (
            discretizer.params[column]["x_min"] == df[column].min() - epsilon[column]["min"]
        )
        assert (
            discretizer.params[column]["x_max"] == df[column].max() + epsilon[column]["max"]
        )


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_discretize_df(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    df_discrete = discretizer.discretize_df(df)

    for column in df.columns:
        if column.endswith("_bit"):
            continue

        x_discrete = discretizer.discretize(df[column], **discretizer.params[column])

        assert (x_discrete == df_discrete[column]).all()


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_df_to_bit_array(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    bit_array = discretizer.df_to_bit_array(df)

    assert bit_array.shape == (df.shape[0], discretizer.n_bits_total)
    assert set(np.unique(bit_array)) == set([0, 1])


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_bit_array_to_df(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    bit_array = discretizer.df_to_bit_array(df)
    df_recovered = discretizer.bit_array_to_df(bit_array)

    assert (np.isclose(df_recovered, df)).all()


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_undiscretize(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    for column in df.columns:
        if column.endswith("_bit"):
            continue

        x_discrete = discretizer.discretize(df[column], **discretizer.params[column])
        x_recovered = discretizer.undiscretize(x_discrete, **discretizer.params[column])

        assert np.isclose(x_recovered, df[column]).all()


@pytest.mark.parametrize("df_and_discretization_params", range(1, max_bits), indirect=True)
def test_undiscretize_df(df_and_discretization_params):
    df, n_bits = df_and_discretization_params
    discretizer = Discretizer(df, n_bits)

    df_discrete = discretizer.discretize_df(df)
    df_recovered = discretizer.undiscretize_df(df_discrete)

    assert np.isclose(df_recovered, df).all()
