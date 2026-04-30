"""Tests for jcm/physics/forcing/speedy_forcing.py — specifically the
CO2 absorptivity mapping introduced in #285. The legacy SPEEDY behavior
was `ablco2 = ablco2_ref * exp(0.005 * (model_year + tyear - co2_year_ref))`
under `ForcingParameters.increase_co2=True`. After the refactor `ablco2` is
a function of `forcing.co2_vmr`; these tests pin the new linear-in-ratio
mapping and verify it stays in the same ballpark as the legacy formula
across the historical CO2 trajectory.
"""

import unittest

import jax.numpy as jnp

from jcm.physics.forcing.speedy_forcing import (
    ablco2_from_co2_vmr,
    CO2_VMR_REF_PPMV,
)
from jcm.physics.speedy.physics_data import ablco2_ref


# Legacy SPEEDY formula reproduced verbatim for regression comparison.
DEL_CO2_LEGACY = 0.005   # absorptivity growth rate per year
LEGACY_REF_YEAR = 1950   # ForcingParameters.co2_year_ref default


def _legacy_ablco2(year: float) -> float:
    """Evaluate the pre-#285 absorptivity formula at a given calendar year."""
    return float(ablco2_ref * jnp.exp(DEL_CO2_LEGACY * (year - LEGACY_REF_YEAR)))


def _co2_legacy_equivalent(year: float) -> float:
    """Return the CO2 ppmv that, under the new linear mapping, reproduces
    SPEEDY's legacy ``ablco2_ref * exp(0.005 * (year - 1950))`` exactly.

    This is the trajectory a user upgrading from `increase_co2=True` should
    feed into `forcing.co2_vmr` to match their previous runs bit-for-bit.
    """
    return float(CO2_VMR_REF_PPMV * jnp.exp(DEL_CO2_LEGACY * (year - LEGACY_REF_YEAR)))


class TestAblco2FromCO2VMR(unittest.TestCase):

    def test_reference_co2_reproduces_baseline(self):
        """At 360 ppmv (the reference) we should recover ablco2_ref exactly,
        matching SPEEDY's pre-`increase_co2` (legacy default-off) behavior.
        """
        ablco2 = ablco2_from_co2_vmr(jnp.asarray(CO2_VMR_REF_PPMV))
        self.assertAlmostEqual(float(ablco2), ablco2_ref, places=8)

    def test_linear_in_co2(self):
        """`ablco2` should scale linearly with `co2_vmr`."""
        a1 = float(ablco2_from_co2_vmr(jnp.asarray(360.0)))
        a2 = float(ablco2_from_co2_vmr(jnp.asarray(720.0)))
        self.assertAlmostEqual(a2 / a1, 2.0, places=6)

    def test_reproduces_legacy_with_matched_co2_trajectory(self):
        """Backward-compat anchor: feeding the new system the CO2 trajectory
        produced by the legacy growth rate (`360 * exp(0.005 * dyears)`)
        recovers the legacy `ablco2` formula exactly. Users upgrading from
        `increase_co2=True` should use `_co2_legacy_equivalent` to reproduce
        old runs.
        """
        for year in (1950, 1990, 2020, 2050, 2100):
            new = float(ablco2_from_co2_vmr(jnp.asarray(_co2_legacy_equivalent(year))))
            legacy = _legacy_ablco2(year)
            self.assertAlmostEqual(
                new, legacy, places=4,
                msg=(f"ablco2 at year {year}: new={new:.6f} legacy={legacy:.6f}. "
                     "Linear-in-ratio mapping should reproduce the legacy formula "
                     "exactly when fed the legacy growth-equivalent CO2 trajectory. "
                     "If this drifts, the conversion in speedy_forcing.py changed "
                     "and the test needs to update with it."),
            )


if __name__ == "__main__":
    unittest.main()
