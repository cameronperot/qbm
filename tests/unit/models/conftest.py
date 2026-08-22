from typing import Any

import numpy as np
import pandas as pd
import pytest

from qbm.models import BQRBM
from qbm.utils import Discretizer, get_rng

n_visible = 8
n_hidden = 4


def mock_initialize_annealer(model: Any) -> None:
    model.qpu = None
    model.h_range = np.array([-4, 4])
    model.J_range = np.array([-1, 1])


@pytest.fixture
def V_train() -> np.ndarray:
    rng = get_rng(0)
    df = pd.DataFrame.from_dict({"x": rng.normal(0, 1, 1000)})
    discretizer = Discretizer(df, n_bits=n_visible)

    return discretizer.df_to_bit_array(df)


@pytest.fixture
def model_simulation(monkeypatch: Any, V_train: np.ndarray) -> BQRBM:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

    return BQRBM(
        V_train=V_train,
        n_hidden=n_hidden,
        A_freeze=0.1,
        B_freeze=1.1,
        beta_initial=1.5,
        beta_range=[0.1, 10],
        simulation_params={"beta": 1.5},
        seed=0,
    )


@pytest.fixture
def model_annealer(monkeypatch: Any, V_train: np.ndarray) -> BQRBM:
    monkeypatch.setattr(
        "qbm.models.BQRBM._initialize_annealer", mock_initialize_annealer
    )

    return BQRBM(
        V_train=V_train,
        n_hidden=n_hidden,
        A_freeze=0.1,
        B_freeze=1.1,
        beta_initial=1.5,
        beta_range=[0.1, 10],
        annealer_params={"embedding": {1: [1], 2: [2]}, "schedule": [(0, 0), (20, 1)]},
        seed=0,
    )
