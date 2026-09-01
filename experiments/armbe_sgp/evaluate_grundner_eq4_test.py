"""Tests for the published Grundner EQ4 implementation."""

import numpy as np

from evaluate_grundner_eq4 import grundner_eq4


def test_grundner_eq4_gate_and_bounds():
    prediction = grundner_eq4(
        relative_humidity=np.array([0.8, 2.0, 0.8]),
        temperature=np.array([270.0, 270.0, 270.0]),
        qc=np.array([0.0, 1.0e-5, 1.0e-5]),
        qi=np.array([0.0, 0.0, 0.0]),
        rh_gradient_height=np.zeros(3),
    )
    assert prediction[0] == 0.0
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_rh_constraint_removes_low_rh_upturn():
    arguments = {
        "relative_humidity": np.array([0.1, 0.8]),
        "temperature": np.array([257.06, 280.0]),
        "qc": np.full(2, 1.0e-5),
        "qi": np.zeros(2),
        "rh_gradient_height": np.zeros(2),
    }
    unconstrained = grundner_eq4(**arguments, enforce_rh_monotonicity=False)
    constrained = grundner_eq4(**arguments, enforce_rh_monotonicity=True)
    assert constrained[0] <= unconstrained[0]
    assert constrained[1] == unconstrained[1]
