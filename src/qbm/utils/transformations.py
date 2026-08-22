from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


class PowerTransformer:
    """
    Transforms data points that lie beyond the provided threshold by taking their
    power (<1) to scale them closer to the mean.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        threshold: float = 1.0,
        power: float = 0.5,
        columns: Sequence[Hashable] | None = None,
    ) -> None:
        """
        Initializes the transformer.

        Args:
            df: Dataframe of data to scale.
            threshold: Number of standard deviations from the mean beyond which to
                begin the scaling (applied to both tails).
            power: Power at which to scale the outlier.
            columns: Optional list of columns to apply the transformation to. If no
                columns are provided, then all columns are transformed.
        Raises:
            ValueError: If power >= 1, if threshold < 1, if power <= 0, or if
                columns is not a subset of df.columns.
        """
        if power >= 1:
            raise ValueError(f"power must be < 1 (got {power})")
        if power <= 0:
            raise ValueError(f"power must be > 0 (got {power})")
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1 (got {threshold})")

        if columns is None:
            self.columns = df.columns
        else:
            if not set(columns).issubset(df.columns):
                raise ValueError(
                    f"columns {list(columns)} are not a subset of "
                    f"df.columns {list(df.columns)}"
                )
            self.columns = columns
        self.power = power
        self.threshold = threshold
        self.offset = threshold - threshold**power

        self.μ = {}
        self.σ = {}
        for column in df.columns:
            self.μ[column] = df[column].mean()
            self.σ[column] = df[column].std()

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Transforms the data to the scaled space.

        Args:
            df: Dataframe to scale.
            inplace: If True then it operates on the same dataframe, if False then
                it creates a copy.

        Returns:
            Dataframe of transformed data.

        Raises:
            ValueError: If a configured column is missing from df.
        """
        if not set(self.columns).issubset(df.columns):
            raise ValueError(
                f"df is missing configured columns "
                f"{list(set(self.columns) - set(df.columns))}"
            )
        if not inplace:
            df = df.copy()

        for column in self.columns:
            μ = self.μ[column]
            σ = self.σ[column]
            x = (df[column] - μ) / σ
            mask = np.abs(x) > self.threshold
            x[mask] = ((np.abs(x) ** self.power + self.offset) * np.sign(x))[mask]
            df[column] = x * σ + μ

        return df

    def inverse_transform(
        self, df: pd.DataFrame, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Transforms the data back from the scaled space.

        Args:
            df: Dataframe to scale.
            inplace: If True then it operates on the same dataframe, if False then
                it creates a copy.

        Returns:
            Dataframe of untransformed data.

        Raises:
            ValueError: If a configured column is missing from df.
        """
        if not set(self.columns).issubset(df.columns):
            raise ValueError(
                f"df is missing configured columns "
                f"{list(set(self.columns) - set(df.columns))}"
            )
        if not inplace:
            df = df.copy()

        for column in self.columns:
            μ = self.μ[column]
            σ = self.σ[column]
            x = (df[column] - μ) / σ
            mask = np.abs(x) > self.threshold
            x[mask] = ((np.abs(x) - self.offset) ** (1 / self.power) * np.sign(x))[mask]
            df[column] = x * σ + μ

        return df
