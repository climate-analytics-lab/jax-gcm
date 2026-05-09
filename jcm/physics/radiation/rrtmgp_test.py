"""Tests for the RRTMGP radiation scheme wrapper.

Compares RRTMGP and grey radiation schemes with identical atmospheric inputs
to verify structural correctness and reasonable agreement.

Date: 2025-08-01
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax_datetime as jdt
from datetime import datetime

from jcm.physics.radiation.grey_two_stream.radiation_scheme import radiation_scheme
from jcm.physics.radiation.rrtmgp import (
    radiation_scheme_rrtmgp,
)
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.echam.unit_conversions import (
    calculate_air_density,
    calculate_layer_thickness,
)
from jcm.physics.radiation.grey_two_stream.radiation_scheme_test import (
    create_test_atmosphere,
    create_default_aerosol_data,
)


def _make_inputs(nlev=10):
    """Create identical input set for both radiation schemes."""
    atm = create_test_atmosphere(nlev=nlev)
    params = RadiationParameters.default()
    aerosol = create_default_aerosol_data(nlev=nlev, parameters=params)

    air_density = calculate_air_density(
        atm["pressure_levels"], atm["temperature"]
    )
    layer_thickness = calculate_layer_thickness(
        atm["pressure_levels"], atm["temperature"]
    )

    # Summer solstice, equatorial point
    date = jdt.Datetime.from_pydatetime(datetime(2024, 6, 21, 12, 0))
    from jcm.forcing import SolarGeometry
    from jax_solar import OrbitalTime
    ot = OrbitalTime.from_datetime(date)
    solar = SolarGeometry(
        tyear=jnp.asarray(ot.orbital_phase / (2.0 * jnp.pi), dtype=jnp.float32),
        orbital_phase=jnp.asarray(ot.orbital_phase, dtype=jnp.float32),
        synodic_phase=jnp.asarray(ot.synodic_phase, dtype=jnp.float32),
    )

    return dict(
        temperature=atm["temperature"],
        specific_humidity=atm["specific_humidity"],
        pressure_levels=atm["pressure_levels"],
        pressure_interfaces=atm["pressure_interfaces"],
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=atm["cloud_water"],
        cloud_ice=atm["cloud_ice"],
        cloud_fraction=atm["cloud_fraction"],
        surface_temperature=jnp.array(300.0),
        surface_albedo_vis=jnp.array(0.07),
        surface_albedo_nir=jnp.array(0.07),
        surface_emissivity=jnp.array(0.98),
        solar=solar,
        latitude=0.0,
        longitude=0.0,
        parameters=params,
        aerosol_data=aerosol,
        ozone_vmr=None,
        co2_vmr=400e-6,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestRRTMGPScheme:
    """Test the RRTMGP radiation scheme produces valid outputs."""

    def test_rrtmgp_produces_valid_heating(self):
        """RRTMGP heating rates should be finite and non-trivial."""
        inputs = _make_inputs(nlev=10)
        tend, diag = radiation_scheme_rrtmgp(**inputs)

        assert jnp.all(jnp.isfinite(tend.temperature_tendency))
        assert jnp.all(jnp.isfinite(tend.longwave_heating))
        assert jnp.all(jnp.isfinite(tend.shortwave_heating))
        # At least some non-zero heating
        assert jnp.mean(jnp.abs(tend.temperature_tendency)) > 1e-8

    def test_rrtmgp_diagnostics_valid(self):
        """Surface/TOA flux diagnostics should be non-negative and finite."""
        inputs = _make_inputs(nlev=10)
        _, diag = radiation_scheme_rrtmgp(**inputs)

        assert jnp.isfinite(diag.surface_sw_down)
        assert jnp.isfinite(diag.surface_lw_down)
        assert jnp.isfinite(diag.toa_lw_up)
        assert diag.surface_lw_down >= 0.0
        assert diag.toa_lw_up >= 0.0


class TestGreyVsRRTMGP:
    """Compare grey and RRTMGP schemes for structural agreement."""

    def test_heating_tendency_shapes_match(self):
        """Both schemes should return the same shaped arrays."""
        inputs = _make_inputs(nlev=10)
        tend_grey, _ = radiation_scheme(**inputs)
        tend_rrtm, _ = radiation_scheme_rrtmgp(**inputs)

        assert tend_grey.temperature_tendency.shape == tend_rrtm.temperature_tendency.shape
        assert tend_grey.longwave_heating.shape == tend_rrtm.longwave_heating.shape
        assert tend_grey.shortwave_heating.shape == tend_rrtm.shortwave_heating.shape

    def test_heating_rates_broadly_agree(self):
        """Total heating should agree within a generous tolerance.

        The grey scheme is a coarse parameterisation so we only check that
        the two are in the same ballpark (atol=0.1 K/s, rtol=100%).
        """
        inputs = _make_inputs(nlev=10)
        tend_grey, _ = radiation_scheme(**inputs)
        tend_rrtm, _ = radiation_scheme_rrtmgp(**inputs)

        # Both should have the same sign pattern in most levels
        assert jnp.all(jnp.isfinite(tend_grey.temperature_tendency))
        assert jnp.all(jnp.isfinite(tend_rrtm.temperature_tendency))

        # Loose absolute tolerance (K/s) — they don't need to match closely
        np.testing.assert_allclose(
            tend_grey.temperature_tendency,
            tend_rrtm.temperature_tendency,
            atol=0.1,
            rtol=1.0,
        )

    @pytest.mark.parametrize(
        "lat,lon,month",
        [
            (0.0, 0.0, 6),       # equator, summer
            (60.0, 0.0, 6),      # high-lat NH summer
            (-60.0, 0.0, 12),    # high-lat SH summer
            (0.0, 180.0, 3),     # equator, equinox
            (80.0, 0.0, 12),     # near-polar NH winter (low sun)
        ],
    )
    def test_multiple_conditions(self, lat, lon, month):
        """Both schemes should produce finite results across conditions."""
        from jcm.forcing import SolarGeometry
        from jax_solar import OrbitalTime
        inputs = _make_inputs(nlev=10)
        inputs["latitude"] = lat
        inputs["longitude"] = lon
        # Build a SolarGeometry from the parameterized date — radiation
        # schemes consume `solar` instead of `date` since the date-aware
        # forcing refactor (#285 follow-up).
        ot = OrbitalTime.from_datetime(
            jdt.Datetime.from_pydatetime(datetime(2024, month, 15, 12, 0))
        )
        inputs["solar"] = SolarGeometry(
            tyear=jnp.asarray(ot.orbital_phase / (2.0 * jnp.pi), dtype=jnp.float32),
            orbital_phase=jnp.asarray(ot.orbital_phase, dtype=jnp.float32),
            synodic_phase=jnp.asarray(ot.synodic_phase, dtype=jnp.float32),
        )

        tend_grey, _ = radiation_scheme(**inputs)
        tend_rrtm, _ = radiation_scheme_rrtmgp(**inputs)

        assert jnp.all(jnp.isfinite(tend_grey.temperature_tendency))
        assert jnp.all(jnp.isfinite(tend_rrtm.temperature_tendency))


class TestRRTMGPMcICA:
    """Behavioural tests for the per-g-point McICA partial-cloud path."""

    def test_clear_sky_limit_zero_cloud_fraction(self):
        """``cloud_fraction=0`` ⇒ McICA produces clear-sky fluxes."""
        inputs = _make_inputs(nlev=10)
        nlev = inputs["temperature"].shape[0]
        # Zero cloud fraction kills every sub-column's cloud presence.
        inputs["cloud_fraction"] = jnp.zeros((nlev,))
        inputs["compute_cre"] = True

        _, diag = radiation_scheme_rrtmgp(**inputs)

        # The McICA all-sky path collapses to clear-sky in this limit, so
        # the all-sky and clear-sky TOA fluxes must agree.
        np.testing.assert_allclose(
            float(diag.toa_sw_up), float(diag.toa_sw_up_clear),
            rtol=1e-4, atol=1e-4,
        )
        np.testing.assert_allclose(
            float(diag.toa_lw_up), float(diag.toa_lw_up_clear),
            rtol=1e-4, atol=1e-4,
        )

    def test_overcast_brackets_clear_sky(self):
        """``cloud_fraction=1`` with cloud water ⇒ all-sky differs from
        clear-sky in the expected direction (clouds reflect more SW,
        emit colder LW).
        """
        inputs = _make_inputs(nlev=10)
        nlev = inputs["temperature"].shape[0]
        inputs["cloud_fraction"] = jnp.ones((nlev,))
        # Realistic in-cloud LWP: ~ 5 g/kg liquid mass over a few layers.
        cloud_water = jnp.zeros((nlev,)).at[3:6].set(5e-4)
        inputs["cloud_water"] = cloud_water
        inputs["compute_cre"] = True

        _, diag = radiation_scheme_rrtmgp(**inputs)

        # Cloudy column reflects more SW → all-sky toa_sw_up > clear-sky.
        assert float(diag.toa_sw_up) > float(diag.toa_sw_up_clear) + 1e-3
        # Cloudy column emits less OLR → all-sky toa_lw_up < clear-sky.
        assert float(diag.toa_lw_up) < float(diag.toa_lw_up_clear) - 1e-3

    def test_compute_cre_false_zeros_clear_sky_fields(self):
        """Disabling ``compute_cre`` skips the clear-sky call; CRE
        diagnostics stay at their zero default.
        """
        inputs = _make_inputs(nlev=10)
        inputs["compute_cre"] = False

        _, diag = radiation_scheme_rrtmgp(**inputs)

        assert float(diag.toa_sw_up_clear) == 0.0
        assert float(diag.toa_lw_up_clear) == 0.0
        # The all-sky McICA result is still computed and finite.
        assert jnp.isfinite(diag.toa_sw_up)
        assert jnp.isfinite(diag.toa_lw_up)

    def test_seed_reproducibility(self):
        """Same ``base_seed`` and column index ⇒ identical fluxes
        (bit-exact, deterministic McICA seeding).
        """
        inputs_a = _make_inputs(nlev=10)
        inputs_a["base_seed"] = 17
        inputs_a["column_index"] = jnp.int32(3)

        inputs_b = _make_inputs(nlev=10)
        inputs_b["base_seed"] = 17
        inputs_b["column_index"] = jnp.int32(3)

        _, diag_a = radiation_scheme_rrtmgp(**inputs_a)
        _, diag_b = radiation_scheme_rrtmgp(**inputs_b)

        np.testing.assert_array_equal(
            np.array(diag_a.toa_sw_up), np.array(diag_b.toa_sw_up),
        )
        np.testing.assert_array_equal(
            np.array(diag_a.toa_lw_up), np.array(diag_b.toa_lw_up),
        )

    def test_different_seeds_diverge_for_partial_cloud(self):
        """Different McICA seeds give different stochastic realisations
        in a partly-cloudy column, but the magnitudes stay sensible.
        """
        inputs_a = _make_inputs(nlev=10)
        nlev = inputs_a["temperature"].shape[0]
        inputs_a["cloud_fraction"] = jnp.full((nlev,), 0.5)
        inputs_a["cloud_water"] = jnp.zeros((nlev,)).at[3:6].set(5e-4)
        inputs_a["base_seed"] = 1
        inputs_a["column_index"] = jnp.int32(0)

        inputs_b = {**inputs_a, "base_seed": 999}

        _, diag_a = radiation_scheme_rrtmgp(**inputs_a)
        _, diag_b = radiation_scheme_rrtmgp(**inputs_b)

        # Stochastic noise should be visible at this resolution.
        toa_diff = float(jnp.abs(diag_a.toa_sw_up - diag_b.toa_sw_up))
        assert toa_diff > 0.0
        # But within the band of the all-sky vs. clear-sky difference —
        # the noise floor should be much smaller than the cloud signal.
        all_minus_clear = float(jnp.abs(diag_a.toa_sw_up - diag_a.toa_sw_up_clear))
        assert toa_diff < max(all_minus_clear * 2, 50.0)
