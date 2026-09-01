"""Tests for conservative ECHAM L47 remapping."""

import numpy as np

from build_echam_l47_pilot import remap_profile


def test_remap_profile_conserves_condensate_mass():
    height = np.arange(15.0, 615.0, 30.0)
    pressure = 100000.0 - 12.0 * height
    density = np.full(height.shape, 1.2)
    pressure_half = np.linspace(92000.0, 100000.0, 5)
    qc = np.full(height.shape, 1.0e-4)
    qi = np.full(height.shape, 2.0e-5)
    result = remap_profile(
        pressure=pressure,
        density=density,
        height=height,
        temperature=np.full(height.shape, 280.0),
        relative_humidity=np.full(height.shape, 0.7),
        specific_humidity=np.full(height.shape, 0.005),
        qc=qc,
        qi=qi,
        cloud_fraction=np.full(height.shape, 0.4),
        source_valid=np.ones(height.shape, dtype=bool),
        pressure_half=pressure_half,
    )
    np.testing.assert_allclose(result["source_qc_mass"], result["remapped_qc_mass"])
    np.testing.assert_allclose(result["source_qi_mass"], result["remapped_qi_mass"])
    assert np.all((result["coverage_fraction"] >= 0.0) & (result["coverage_fraction"] <= 1.01))
