"""Focused tests for the independent cloud-diagnostic evaluator."""

import math

import numpy as np
import pytest

from cloud_operators import cloudc_plus_cloudstr_raw
from diagnose_cloud_cover import goodness_of_fit, resolved_config


def test_goodness_of_fit_uses_only_qc_passed_pairs():
    metrics = goodness_of_fit(
        np.array([0.1, 0.9, 0.3]), np.array([0.2, 0.1, 0.5]), np.array([True, False, True])
    )
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.025))
    assert metrics["mae"] == pytest.approx(0.15)
    assert metrics["bias"] == pytest.approx(-0.15)


def test_diagnostic_config_requires_cloud_observation_files(tmp_path):
    with pytest.raises(ValueError, match="cldrad"):
        resolved_config({"atm": "atm.nc"}, tmp_path)


def test_raw_cloud_sum_does_not_impose_an_overlap_assumption():
    result = cloudc_plus_cloudstr_raw({"cloudc": np.array([0.8]), "cloudstr": np.array([0.7])})
    np.testing.assert_allclose(result, [1.5])
