from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


class PowerTransformer:
    """
    Transforms data points that lie beyond the provided threshold by a taking their
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
        :param df: Dataframe of data to scale.
        :param threshold: Number of standard deviations above the mean at which to begin
            the scaling.
        :param power: Power at which to scale the outlier.
        :param columns: Optional list of columns to apply the transformation to. If no
            columns are provided, then all columns are transformed.
        """
        assert power < 1
        assert threshold >= 1

        if columns is None:
            self.columns = df.columns
        else:
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

        :param df: Dataframe to scale.
        :param inplace: If True then it operates on the same dataframe, if False then
            it creates a copy.

        :returns: Dataframe of transformed data.
        """
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

        :param df: Dataframe to scale.
        :param inplace: If True then it operates on the same dataframe, if False then
            it creates a copy.

        :returns: Dataframe of untransformed data.
        """
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
