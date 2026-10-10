"""The published noise-probe number has to match the script."""
from __future__ import annotations

import pytest

from scripts.probe_residual_noise import measure


def test_one_percent_white_noise_crosses_the_residual_cut():
    row = measure(seed=0, noise_frac=0.01)
    assert row["clean"] == pytest.approx(0.015978941319174836)
    assert row["noisy"] == pytest.approx(1.1347296382361)
    assert f"{row['noisy']:.6e}" == "1.134730e+00"
    assert row["clean"] < row["cut"] < row["noisy"]
