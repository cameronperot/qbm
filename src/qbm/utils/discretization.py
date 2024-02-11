import numpy as np
import pandas as pd


class Discretizer:
    def __init__(self, df, n_bits, epsilon={}):
        """
        :param df: Dataframe of numerical values.
        :param n_bits: Number of bits to discretize to.
        :param epsilon: Optional dictionary of min/max offset values.
        """
        self.columns = df.columns
        self.n_bits = n_bits
        self.epsilon = epsilon
        self.params = {}
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

            # update the split indices
            if i < len(self.columns) - 1:
                self.split_indices.append(
                    self.params[column]["n_bits"]
                    if i == 0
                    else self.split_indices[i - 1] + self.params[column]["n_bits"]
                )

            self.n_bits_total += self.params[column]["n_bits"]

    @staticmethod
    def bit_vector_to_int(bit_vector):
        """
        Converts a bit vector to a bit string.

        :param bit_vector: Input bit vector.

        :returns: Bit string of the input bit vector.
        """
        return int("".join(str(x) for x in bit_vector), 2)

    @staticmethod
    def bit_vector_to_string(bit_vector):
        """
        Converts a bit vector to a bit string.

        :param bit_vector: Input bit vector.

        :returns: Bit string of the input bit vector.
        """
        return "".join(str(x) for x in bit_vector)

    @staticmethod
    def int_to_bit_vector(x, n_bits):
        """
        Converts the integer x to an n_bits-bit bit vector.

        :param x: Integer value which to convert.
        :param n_bits: Length of the bit vector.

        :returns: Bit vector of length n_bits.
        """
        return [1 if i == "1" else 0 for i in bin(x)[2:].zfill(n_bits)]

    @staticmethod
    @np.vectorize
    def discretize(x, n_bits, x_min, x_max):
        """
        Convert the value x into a n-bit bit string.

        :param x: Float value to convert.
        :param n_bits: Length of the bit string.
        :param x_min: Minimum value for scaling.
        :param x_max: Maximum value for scaling.

        :returns: A bit string representation of x.
        """
        scaling_factor = (2**n_bits - 1) / (x_max - x_min)

        x = round((x - x_min) * scaling_factor)
        assert x >= 0 and x <= 2**n_bits - 1
        return x

    @staticmethod
    @np.vectorize
    def undiscretize(x, n_bits, x_min, x_max):
        """
        Convert the value x into a float from a n-bit bit string.

        :param x: Int value to convert.
        :param n_bits: Length of the bit string.
        :param x_min: Minimum value for scaling.
        :param x_max: Maximum value for scaling.

        :returns: A float representation of x.
        """
        scaling_factor = (2**n_bits - 1) / (x_max - x_min)

        assert x < 2**n_bits
        return x / scaling_factor + x_min

    def discretize_df(self, df):
        """
        Convert all columns of a dataframe to bit representation.

        :param df: Dataframe which to convert.

        :returns: A discretized version of df.
        """
        df_discretized = df.copy()
        for column in df.columns:
            if column.endswith("_bit"):
                df_discretized[column] = df[column].astype(np.int8)
            else:
                df_discretized[column] = self.discretize(df[column], **self.params[column])

        return df_discretized

    def undiscretize_df(self, df):
        """
        Convert all columns of a dataframe to floats from bit representation.

        :param df: Dataframe which to convert.

        :returns: An undiscretized version of df_discretized.
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

    def df_to_bit_array(self, df):
        """
        Converts a dataframe of floats to a bit array.

        :param df: Dataframe which to convert.

        :param returns: Array of bits of shape (df.shape[0], self.n_bits_total).
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

    def bit_array_to_df(self, bit_array):
        """
        Converts bit array a dataframe of floats.

        :param bit_array: Bit array which to convert.

        :param returns: Dataframe of shape (bit_array.shape[0], len(self.columns)).
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
