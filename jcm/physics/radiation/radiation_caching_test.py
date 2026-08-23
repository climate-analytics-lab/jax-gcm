"""Tests for the shared radiation sub-stepping cache (#671).

Radiation is solved every ``radiation_interval`` but applied every step, so
the cached shortwave must be rescaled to the current sun. These cover the
rescaling contract itself; the per-scheme wiring is exercised by each
scheme's own integration tests.
"""
import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.radiation import (
    _CACHED_SW_FIELDS,
    cached_radiation_tendency,
    rescale_cached_radiation,
)
from jcm.physics.radiation.radiation_types import RadiationData

NCOLS, NLEV = 4, 3


def _solved_at(mu0):
    """Build a RadiationData as if solved at ``mu0``, with fluxes ~ mu0."""
    mu0 = jnp.asarray(mu0, dtype=jnp.float32)
    rad = RadiationData.zeros((NCOLS,), NLEV)
    sw = {name: getattr(rad, name) + 100.0 * mu0 for name in _CACHED_SW_FIELDS}
    return rad.copy(
        cos_zenith=mu0,
        cos_zenith_at_compute=mu0,
        lw_flux_up=jnp.full((NLEV + 1, NCOLS), 240.0),
        lw_heating_rate=jnp.full((NLEV, NCOLS), -1.5e-5),
        noa_frac_toa_sw_up=jnp.full((NCOLS,), 0.02),
        **sw,
    )


class ZenithRescalingTest(unittest.TestCase):
    def test_shortwave_scales_with_the_zenith_ratio(self):
        rad = _solved_at([0.8] * NCOLS)
        out = rescale_cached_radiation(rad, jnp.full((NCOLS,), 0.4))
        for name in _CACHED_SW_FIELDS:
            np.testing.assert_allclose(
                np.asarray(getattr(out, name)),
                np.asarray(getattr(rad, name)) * 0.5,
                rtol=1e-6, err_msg=name,
            )

    def test_same_sun_is_the_identity(self):
        # A cached step at the compute-step geometry must reproduce it
        # exactly, so enabling the rescale cannot perturb an interval of 1.
        rad = _solved_at([0.9, 0.5, 0.1, 0.02])
        out = rescale_cached_radiation(rad, rad.cos_zenith_at_compute)
        for name in _CACHED_SW_FIELDS:
            np.testing.assert_allclose(
                np.asarray(getattr(out, name)),
                np.asarray(getattr(rad, name)),
                rtol=1e-6, err_msg=name,
            )

    def test_longwave_is_untouched(self):
        rad = _solved_at([0.8] * NCOLS)
        out = rescale_cached_radiation(rad, jnp.full((NCOLS,), 0.2))
        np.testing.assert_array_equal(out.lw_flux_up, rad.lw_flux_up)
        np.testing.assert_array_equal(
            out.lw_heating_rate, rad.lw_heating_rate,
        )

    def test_aerosol_free_fractions_are_not_rescaled(self):
        # They are flux RATIOS, so they are already zenith-independent;
        # scaling them would corrupt the ERFari diagnostic.
        rad = _solved_at([0.8] * NCOLS)
        out = rescale_cached_radiation(rad, jnp.full((NCOLS,), 0.2))
        np.testing.assert_array_equal(
            out.noa_frac_toa_sw_up, rad.noa_frac_toa_sw_up,
        )

    def test_night_gives_zero_rather_than_negative_flux(self):
        # cos_zenith goes negative below the horizon; an unguarded ratio
        # would flip the sign of every cached shortwave flux.
        rad = _solved_at([0.8] * NCOLS)
        out = rescale_cached_radiation(rad, jnp.full((NCOLS,), -0.35))
        for name in _CACHED_SW_FIELDS:
            self.assertTrue(
                bool(jnp.all(getattr(out, name) == 0.0)), msg=name,
            )

    def test_a_column_dark_at_compute_time_stays_dark_and_finite(self):
        # mu0_at_compute ~ 0 makes the ratio singular. The cached fluxes are
        # zero anyway, so no factor can recover a solve that never ran: hold
        # zero rather than divide. This is the residual error the interval
        # bounds, and the reason not to push radiation_interval much past 2 h.
        rad = _solved_at([0.0] * NCOLS)
        out = rescale_cached_radiation(rad, jnp.full((NCOLS,), 0.6))
        for name in _CACHED_SW_FIELDS:
            got = np.asarray(getattr(out, name))
            self.assertTrue(np.all(np.isfinite(got)), msg=name)
            np.testing.assert_allclose(got, 0.0, atol=0.0, err_msg=name)

    def test_current_cos_zenith_is_refreshed_for_downstream_consumers(self):
        # JAM oxidant photolysis reads radiation.cos_zenith as the CURRENT
        # sun (aerosol/jam/chemistry/oxidants.py). Before the split it was
        # frozen for the whole radiation interval.
        rad = _solved_at([0.8] * NCOLS)
        now = jnp.full((NCOLS,), 0.25)
        out = rescale_cached_radiation(rad, now)
        np.testing.assert_allclose(np.asarray(out.cos_zenith), 0.25)
        np.testing.assert_allclose(
            np.asarray(out.cos_zenith_at_compute), 0.8,
        )

    def test_rescaled_heating_reaches_the_replayed_tendency(self):
        # The whole point: the tendency actually applied to the model must
        # carry the rescaled shortwave, not the compute-step value.
        rad = _solved_at([0.8] * NCOLS)
        out = rescale_cached_radiation(rad, jnp.full((NCOLS,), 0.4))
        tend = cached_radiation_tendency(out, (NLEV, NCOLS))
        np.testing.assert_allclose(
            np.asarray(tend.temperature),
            np.asarray(rad.sw_heating_rate * 0.5 + rad.lw_heating_rate),
            rtol=1e-6,
        )

    def test_per_column_ratios_are_independent(self):
        # The terminator runs through the grid, so neighbouring columns get
        # very different ratios in the same call.
        rad = _solved_at([1.0, 1.0, 1.0, 1.0])
        out = rescale_cached_radiation(
            rad, jnp.asarray([1.0, 0.5, 0.0, -0.5], dtype=jnp.float32),
        )
        np.testing.assert_allclose(
            np.asarray(out.surface_sw_down), [100.0, 50.0, 0.0, 0.0],
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
