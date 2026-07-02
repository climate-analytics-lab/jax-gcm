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

class TestRRTMGPTermCacheCoords:
    """The composable term caches per-column lat/lon at ``cache_coords``."""

    def test_cache_coords_sets_per_column_latlon(self):
        from dinosaur.sigma_coordinates import SigmaCoordinates
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
        from jcm.utils import get_coords

        coords = get_coords(
            SigmaCoordinates.equidistant(8), spectral_truncation=21,
        )
        term = RRTMGPRadiation()
        assert not term._coords_cached
        term.cache_coords(coords)
        assert term._coords_cached
        nlon, nlat = coords.horizontal.nodal_shape
        ncols = nlon * nlat
        # One latitude / longitude per column, in degrees.
        lats = term._lats.get_value()
        lons = term._lons.get_value()
        assert lats.shape == (ncols,)
        assert lons.shape == (ncols,)
        assert float(jnp.max(jnp.abs(lats))) <= 90.0 + 1e-3


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


class TestRRTMGPThinCloudInflation:
    """A thin but resolved cloud carrying large grid-mean condensate must not
    NaN the radiation (the in-cloud-water inflation / cloud-optical-depth
    runaway that crashed RRTMGP+1M). Guarded by the in-cloud condensate cap
    (``_MAX_IN_CLOUD_CONDENSATE``).
    """

    def test_finite_for_thin_cloud_high_condensate(self):
        inputs = _make_inputs(nlev=10)
        nlev = inputs["temperature"].shape[0]
        # cf ~ 0.02 (resolved, not clear) with large grid-mean condensate so
        # the in-cloud value (grid_mean / cf) is ~0.25-0.5 kg/kg.
        inputs["cloud_fraction"] = jnp.zeros((nlev,)).at[3:6].set(0.02)
        inputs["cloud_water"] = jnp.zeros((nlev,)).at[3:6].set(5e-3)
        inputs["cloud_ice"] = jnp.zeros((nlev,)).at[4:7].set(8e-3)
        inputs["compute_cre"] = True

        tend, diag = radiation_scheme_rrtmgp(**inputs)
        assert jnp.all(jnp.isfinite(tend.temperature_tendency))
        assert jnp.isfinite(diag.toa_sw_up)
        assert jnp.isfinite(diag.toa_lw_up)
        # The thin cloud is still radiatively active (not silently dropped).
        assert float(diag.toa_sw_up) > float(diag.toa_sw_up_clear) - 1e-3


class TestRRTMGPRadiationQuickWins:
    """Regression pins for the radiation-glue fixes (fix-plan PR 3).

    Each test fails on the pre-fix code:
      - TOA insolation was ∝ µ0² (radiation_flux, already ×µ0, passed as
        the normal-incidence ``irrad`` which the library multiplies by µ0
        again) — pinned by ``test_toa_insolation_is_single_cosine`` at a
        µ0 ≈ 0.5 point where the bug halves the insolation.
      - The pressure halo was edge-filled, halving the boundary layers'
        centered-difference Δp (2× heating) — pinned on the halo helper.
      - The hardcoded ``sfc_alb=0.07`` / ``sfc_emis=0.98`` ignored the
        surface scheme — pinned by asserting the per-column values shape
        the SW/LW fluxes.
      - Condensate rode into gas optics as vapour via ``q_t`` while
        ``q_c`` was zeroed — pinned by comparing the clear-sky (CRE)
        fluxes of a cloudy and a cloud-free column.
    """

    @staticmethod
    def _cloud_free(inputs):
        nlev = inputs["temperature"].shape[0]
        inputs["cloud_water"] = jnp.zeros((nlev,))
        inputs["cloud_ice"] = jnp.zeros((nlev,))
        inputs["cloud_fraction"] = jnp.zeros((nlev,))
        return inputs

    def test_toa_insolation_is_single_cosine(self):
        from jax_solar import direct_solar_irradiance
        inputs = self._cloud_free(_make_inputs(nlev=10))
        # June solstice, local noon at lon 0: subsolar latitude +23.4°, so
        # -36.6° puts the sun 60° from zenith → µ0 ≈ 0.5, where the old
        # µ0² insolation is HALF the correct value.
        inputs["latitude"] = -36.6
        _, diag = radiation_scheme_rrtmgp(**inputs)

        mu0 = float(jnp.reshape(diag.cos_zenith, (-1,))[0])
        assert 0.3 < mu0 < 0.7, f"geometry sanity: mu0={mu0}"
        irrad = float(direct_solar_irradiance(
            inputs["solar"].orbital_phase,
            inputs["parameters"].solar_constant,
        ))
        expected = irrad * mu0
        actual = float(diag.toa_sw_down)
        assert abs(actual - expected) / expected < 0.02, (
            f"TOA SW down {actual:.1f} vs irrad*mu0 {expected:.1f} "
            f"(the mu0^2 bug gives ~{expected * mu0:.1f})"
        )

    def test_pressure_halo_encodes_true_boundary_thickness(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.radiation.rrtmgp import _to_3d_pressure_halo

        # The production hybrid L47 grid — the case that broke both naive
        # halo constructions: edge fill halves a uniform-grid boundary Δp,
        # and linear extrapolation (2p[0]−p[1]) is +75 % at the surface and
        # NEGATIVE at the log-spaced top.
        vertical = get_echam_levels(47)
        ph = jnp.asarray(vertical.a_boundaries) + jnp.asarray(
            vertical.b_boundaries
        ) * 101325.0
        pf = 0.5 * (ph[:-1] + ph[1:])
        # Surface-first, as prepare_rrtmgp_data hands it to the halo helper.
        pf_sf, ph_sf = pf[::-1], ph[::-1]
        dp_bottom = ph_sf[0] - ph_sf[1]
        dp_top = ph_sf[-2] - ph_sf[-1]

        out = _to_3d_pressure_halo(pf_sf, dp_bottom, dp_top, 47, 1)[0, 0]
        # The library's centered difference 0.5*(p[k-1]-p[k+1]) must
        # reproduce the model's TRUE half-level layer thickness at both
        # boundaries (surface ≈ 780 Pa, top ≈ 1.99 Pa on this grid).
        assert float(0.5 * (out[0] - out[2])) == pytest.approx(
            float(dp_bottom), rel=1e-5
        )
        assert float(0.5 * (out[-3] - out[-1])) == pytest.approx(
            float(dp_top), rel=1e-5
        )
        # And the top halo stays strictly positive (log-pressure safety).
        assert float(out[-1]) > 0.0

    def test_surface_albedo_reaches_sw_solver(self):
        dark = self._cloud_free(_make_inputs(nlev=10))
        bright = self._cloud_free(_make_inputs(nlev=10))
        bright["surface_albedo_vis"] = jnp.array(0.7)
        bright["surface_albedo_nir"] = jnp.array(0.7)

        _, diag_dark = radiation_scheme_rrtmgp(**dark)
        _, diag_bright = radiation_scheme_rrtmgp(**bright)

        # Reflected SW at the surface matches the prescribed albedo (vis ==
        # nir here, so the broadband blend is exact).
        ratio = float(diag_bright.surface_sw_up) / float(
            diag_bright.surface_sw_down
        )
        assert ratio == pytest.approx(0.7, abs=0.02)
        # And the extra reflection reaches the TOA budget (ice-albedo
        # feedback pathway). Hardcoded 0.07 gave identical fluxes for both.
        assert float(diag_bright.toa_sw_up) > 2.0 * float(diag_dark.toa_sw_up)

    def test_surface_emissivity_reaches_lw_solver(self):
        grey_sfc = self._cloud_free(_make_inputs(nlev=10))
        black_sfc = self._cloud_free(_make_inputs(nlev=10))
        grey_sfc["surface_emissivity"] = jnp.array(0.5)
        black_sfc["surface_emissivity"] = jnp.array(1.0)

        _, diag_grey = radiation_scheme_rrtmgp(**grey_sfc)
        _, diag_black = radiation_scheme_rrtmgp(**black_sfc)

        # Upwelling surface LW = eps*B(T_s) + (1-eps)*LW_down. Pin the full
        # linear relation for both emissivities (the hardcoded-0.98 code
        # gave the same upwelling flux regardless of the passed value; the
        # margin between the two runs is only ~22 W/m² because this humid
        # 300 K atmosphere has LW_down close to sigma*T^4).
        sigma_t4 = 5.670374e-8 * 300.0**4
        for eps_val, diag in ((0.5, diag_grey), (1.0, diag_black)):
            expected = (
                eps_val * sigma_t4
                + (1.0 - eps_val) * float(diag.surface_lw_down)
            )
            assert float(diag.surface_lw_up) == pytest.approx(
                expected, rel=0.01
            ), f"eps={eps_val}"
        assert float(diag_black.surface_lw_up) > float(
            diag_grey.surface_lw_up
        )

    def test_clear_sky_cre_sees_vapor_only(self):
        cloudy = _make_inputs(nlev=10)
        nlev = cloudy["temperature"].shape[0]
        cloudy["cloud_water"] = jnp.zeros((nlev,)).at[3:6].set(2e-3)
        cloudy["cloud_ice"] = jnp.zeros((nlev,)).at[2:4].set(5e-4)
        cloudy["cloud_fraction"] = jnp.zeros((nlev,)).at[2:6].set(0.5)
        cloudy["compute_cre"] = True

        clear = self._cloud_free(_make_inputs(nlev=10))
        clear["compute_cre"] = True

        _, diag_cloudy = radiation_scheme_rrtmgp(**cloudy)
        _, diag_clear = radiation_scheme_rrtmgp(**clear)

        # The clear-sky (CRE) branch must see the SAME atmosphere for both
        # columns: vapour only. Before the q_t fix the cloudy column's
        # condensate was counted as extra water vapour in gas optics, so
        # its "clear-sky" OLR was biased low relative to the truly clear
        # column.
        assert float(diag_cloudy.toa_lw_up_clear) == pytest.approx(
            float(diag_clear.toa_lw_up_clear), rel=1e-4
        )
        assert float(diag_cloudy.toa_sw_up_clear) == pytest.approx(
            float(diag_clear.toa_sw_up_clear), rel=1e-4
        )
