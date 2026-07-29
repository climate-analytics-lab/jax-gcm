"""Focused tests for the independent cloud-diagnostic evaluator."""

import math

import numpy as np
import pytest

from diagnose_cloud_cover import masked_rmse, resolved_config


def test_masked_rmse_uses_only_qc_passed_pairs():
    rmse = masked_rmse(
        np.array([0.1, 0.9, 0.3]), np.array([0.2, 0.1, 0.5]), np.array([True, False, True])
    )
    assert rmse == pytest.approx(math.sqrt(0.025))


def test_diagnostic_config_requires_cloud_observation_files(tmp_path):
    with pytest.raises(ValueError, match="cldrad"):
        resolved_config({"atm": "atm.nc"}, tmp_path)
