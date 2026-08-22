from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd


class ColumnParams(TypedDict):
    """Per-column discretization parameters."""

    n_bits: int
    x_min: NotRequired[float]
    x_max: NotRequired[float]


class Discretizer:
    """
    Discretizes dataframe columns into bit representations and converts them back.
    Columns whose names end in "_bit" are treated as single bits and are not scaled.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_bits: int,
        epsilon: Mapping[str, Mapping[str, float]] = {},
    ) -> None:
        """
        Initializes the discretizer.

        Args:
            df: Dataframe of numerical values.
            n_bits: Number of bits to discretize to.
            epsilon: Optional dictionary of min/max offset values.
        """
        self.columns = df.columns
        self.n_bits = n_bits
        self.epsilon = epsilon
        self.params: dict[str, ColumnParams] = {}
        self.split_indices = []
        self.n_bits_total = 0

        for i, column in enumerate(self.columns):
            if column.endswith("_bit"):
                self.params[column] = {"n_bits": 1}
            else:
                self.params[column] = {
                    "n_bits": self.n_bits,
                    "x_min": df[column].min(),
                    "x_max": df[column].max(),
                }
                if column in self.epsilon:
                    self.params[column]["x_min"] -= self.epsilon[column]["min"]
                    self.params[column]["x_max"] += self.epsilon[column]["max"]

            # Update the split indices
            if i < len(self.columns) - 1:
                self.split_indices.append(
                    self.params[column]["n_bits"]
                    if i == 0
                    else self.split_indices[i - 1] + self.params[column]["n_bits"]
                )

            self.n_bits_total += self.params[column]["n_bits"]

    @staticmethod
    def bit_vector_to_int(bit_vector: Sequence[int] | np.ndarray) -> int:
        """
        Converts a bit vector to its integer representation.

        Args:
            bit_vector: Input bit vector.

        Returns:
            Integer representation of the input bit vector.
        """
        return int("".join(str(x) for x in bit_vector), 2)

    @staticmethod
    def bit_vector_to_string(bit_vector: Sequence[int]) -> str:
        """
        Converts a bit vector to a bit string.

        Args:
            bit_vector: Input bit vector.

        Returns:
            Bit string of the input bit vector.
        """
        return "".join(str(x) for x in bit_vector)

    @staticmethod
    def int_to_bit_vector(x: int, n_bits: int) -> list[int]:
        """
        Converts the integer x to an n_bits-bit bit vector.

        Args:
            x: Integer value which to convert.
            n_bits: Length of the bit vector.

        Returns:
            Bit vector of length n_bits.
        """
        return [1 if i == "1" else 0 for i in bin(x)[2:].zfill(n_bits)]

    @staticmethod
    @np.vectorize
    def discretize(x: float, n_bits: int, x_min: float, x_max: float) -> int:
        """
        Convert the value x into its n_bits-bit integer representation.

        Args:
            x: Float value to convert.
            n_bits: Number of bits to discretize to.
            x_min: Minimum value for scaling.
            x_max: Maximum value for scaling.

        Returns:
            An integer representation of x.
        """
        scaling_factor = (2**n_bits - 1) / (x_max - x_min)

        x = round((x - x_min) * scaling_factor)
        assert x >= 0 and x <= 2**n_bits - 1
        assert isinstance(x, int)
        return x

    @staticmethod
    @np.vectorize
    def undiscretize(x: float, n_bits: int, x_min: float, x_max: float) -> float:
        """
        Convert the value x into a float from its n_bits-bit integer representation.

        Args:
            x: Int value to convert.
            n_bits: Number of bits to discretize to.
            x_min: Minimum value for scaling.
            x_max: Maximum value for scaling.

        Returns:
            A float representation of x.
        """
        scaling_factor = (2**n_bits - 1) / (x_max - x_min)

        assert x < 2**n_bits
        return x / scaling_factor + x_min

    def discretize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all columns of a dataframe to bit representation.

        Args:
            df: Dataframe which to convert.

        Returns:
            A discretized version of df.
        """
        df_discretized = df.copy()
        for column in df.columns:
            if column.endswith("_bit"):
                df_discretized[column] = df[column].astype(np.int8)
            else:
                df_discretized[column] = self.discretize(
                    df[column], **self.params[column]
                )

        return df_discretized

    def undiscretize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all columns of a dataframe to floats from bit representation.

        Args:
            df: Dataframe which to convert.

        Returns:
            An undiscretized version of df.
        """
        df_undiscretized = df.copy()
        for column in df.columns:
            if column.endswith("_bit"):
                df_undiscretized[column] = df[column].astype(np.int8)
            else:
                df_undiscretized[column] = self.undiscretize(
                    df[column], **self.params[column]
                )

        return df_undiscretized

    def df_to_bit_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts a dataframe of floats to a bit array.

        Args:
            df: Dataframe which to convert.

        Returns:
            Array of bits of shape (df.shape[0], self.n_bits_total).
        """
        assert set(self.columns) == set(df.columns)

        df = self.discretize_df(df)
        bit_array = np.hstack(
            [
                np.vstack(
                    [
                        self.int_to_bit_vector(x, self.params[column]["n_bits"])
                        for x in df[column]
                    ]
                )
                for column in self.columns
            ]
        )

        return bit_array

    def bit_array_to_df(self, bit_array: np.ndarray) -> pd.DataFrame:
        """
        Converts a bit array to a dataframe of floats.

        Args:
            bit_array: Bit array which to convert.

        Returns:
            Dataframe of shape (bit_array.shape[0], len(self.columns)).
        """
        assert len(bit_array[0]) == self.n_bits_total

        rows = [
            [
                self.bit_vector_to_int(x)
                for x in np.array_split(bit_vector, self.split_indices)
            ]
            for bit_vector in bit_array
        ]
        df = self.undiscretize_df(pd.DataFrame(rows, columns=self.columns))

        return df
