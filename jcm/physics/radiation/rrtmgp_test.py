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
from jcm.physics.radiation.grey_two_stream.radiation_scheme_test import (
    create_test_atmosphere,
    create_default_aerosol_data,
    calculate_air_density,
    calculate_layer_thickness,
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


class TestRRTMGPEffectiveRadii:
    """Effective-radius handling in the RRTMGP input prep (finding 2.36).

    The ice fallback must be ECHAM's Moss/Foot power law on the in-cloud
    IWC in g/m3 — thin cirrus gets small crystals — and microphysical
    radii from the clouds carry (2M preffl/preffi) must override the
    fallbacks where provided.
    """

    K_CIRRUS = 1  # TOA-first index of the cirrus layer

    def _make_state(self, nlev=8, dz=5000.0, iwc_gm3=1e-4):
        """In-cloud-condensate RadiationState with one thin cirrus layer."""
        import jcm.constants as c
        from jcm.physics.radiation.grey_two_stream.radiation_scheme import (
            prepare_radiation_state,
        )

        pressure_levels = jnp.linspace(10000.0, 90000.0, nlev)  # TOA-first
        pressure_interfaces = jnp.linspace(5000.0, 95000.0, nlev + 1)
        temperature = jnp.full(nlev, 250.0)
        air_density = pressure_levels / (c.rd * temperature)
        layer_thickness = jnp.full(nlev, dz)
        # In-cloud mixing ratio giving exactly ``iwc_gm3`` of in-cloud ice
        # (the rrtmgp caller hands prepare_radiation_state in-cloud values).
        cloud_ice = jnp.zeros(nlev).at[self.K_CIRRUS].set(
            iwc_gm3 * 1e-3 / air_density[self.K_CIRRUS]
        )
        state = prepare_radiation_state(
            temperature=temperature,
            specific_humidity=jnp.full(nlev, 1e-4),
            pressure_levels=pressure_levels,
            pressure_interfaces=pressure_interfaces,
            layer_thickness=layer_thickness,
            air_density=air_density,
            cloud_water=jnp.zeros(nlev),
            cloud_ice=cloud_ice,
            cloud_fraction=jnp.zeros(nlev).at[self.K_CIRRUS].set(0.3),
            cos_zenith=jnp.array(0.5),
        )
        return state, layer_thickness

    @staticmethod
    def _r_eff_um(rrtmgp_input, key, nlev):
        """Interior r_eff profile (um), flipped back to TOA-first."""
        interior = rrtmgp_input[key][0, 0, 1:-1]
        assert interior.shape == (nlev,)
        return interior[::-1] * 1e6

    def test_thin_cirrus_gets_small_crystals(self):
        """IWC = 1e-4 g/m3 must give r_eff_ice ~ 11.4 um through the prep.

        The previous fabricated formula (T-ramp x clip(path-ratio*1e4))
        yielded ~40-160 um here, saturating the LUT edge for thin cirrus.
        """
        from jcm.physics.radiation.rrtmgp import prepare_rrtmgp_data

        nlev = 8
        state, layer_thickness = self._make_state(nlev=nlev, iwc_gm3=1e-4)
        out = prepare_rrtmgp_data(
            state, layer_thickness, jnp.array(1.0), jnp.array(290.0),
        )
        r_ice = self._r_eff_um(out, "cloud_r_eff_ice", nlev)
        expected = 83.8 * 1e-4 ** 0.216  # ~11.46 um (ECHAM Moss/Foot)
        assert float(r_ice[self.K_CIRRUS]) < 15.0
        assert np.isclose(float(r_ice[self.K_CIRRUS]), expected, rtol=1e-4)

        # Denser cirrus: 0.01 g/m3 -> ~31 um
        state, layer_thickness = self._make_state(nlev=nlev, iwc_gm3=1e-2)
        out = prepare_rrtmgp_data(
            state, layer_thickness, jnp.array(1.0), jnp.array(290.0),
        )
        r_ice = self._r_eff_um(out, "cloud_r_eff_ice", nlev)
        assert np.isclose(
            float(r_ice[self.K_CIRRUS]), 83.8 * 1e-2 ** 0.216, rtol=1e-4,
        )

    def test_provided_microphysical_radii_override_fallback(self):
        """Clouds-carry radii (> 0) win; zeros keep the diagnostic fallback."""
        from jcm.physics.radiation.cloud_optics import (
            effective_radius_liquid,
        )
        from jcm.physics.radiation.rrtmgp import prepare_rrtmgp_data

        nlev = 8
        k = self.K_CIRRUS
        state, layer_thickness = self._make_state(nlev=nlev, iwc_gm3=1e-4)
        r_eff_liq_um = jnp.zeros(nlev).at[k + 2].set(9.5)
        r_eff_ice_um = jnp.zeros(nlev).at[k].set(25.0)
        out = prepare_rrtmgp_data(
            state, layer_thickness, jnp.array(1.0), jnp.array(290.0),
            r_eff_liq_um=r_eff_liq_um, r_eff_ice_um=r_eff_ice_um,
        )
        r_liq = self._r_eff_um(out, "cloud_r_eff_liq", nlev)
        r_ice = self._r_eff_um(out, "cloud_r_eff_ice", nlev)
        # Provided values pass through (um)
        assert np.isclose(float(r_ice[k]), 25.0, rtol=1e-5)
        assert np.isclose(float(r_liq[k + 2]), 9.5, rtol=1e-5)
        # Unprovided levels fall back to the diagnostics
        fallback_liq = float(effective_radius_liquid(jnp.array(1.0), 0.5))
        assert np.isclose(float(r_liq[k]), fallback_liq, rtol=1e-5)
        assert np.isclose(
            float(r_ice[k + 1]), 83.8, rtol=1e-4,
        )  # zero-IWC guard value


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


class TestRRTMGPGreenhouseGases:
    """Prescribed GHG profiles must reach the gas optics and warm the column.

    Covers the ``vmr_fields`` plumbing for O3 / CH4 / N2O (CO2 rides along
    in every test via ``_make_inputs``) and the scalar ``cdnc_factor``
    normalisation branch.
    """

    def test_added_ghgs_reduce_olr(self):
        nlev = 10
        base = _make_inputs(nlev=nlev)
        base["compute_cre"] = False
        # Scalar (ndim == 0) cdnc factor exercises the normalisation branch.
        base["aerosol_data"] = base["aerosol_data"].copy(
            cdnc_factor=jnp.float32(1.0),
        )
        _, diag_base = radiation_scheme_rrtmgp(**base)

        enhanced = dict(base)
        # 4x CO2 + realistic CH4 / N2O + a stratosphere-weighted O3 profile.
        enhanced["co2_vmr"] = 1600e-6
        enhanced["ch4_vmr"] = jnp.array(1.8e-6)
        enhanced["n2o_vmr"] = jnp.array(320e-9)
        enhanced["ozone_vmr"] = jnp.geomspace(8e-6, 3e-8, nlev)  # TOA-first
        _, diag_ghg = radiation_scheme_rrtmgp(**enhanced)

        olr_base = float(diag_base.toa_lw_up)
        olr_ghg = float(diag_ghg.toa_lw_up)
        assert np.isfinite(olr_base) and np.isfinite(olr_ghg)
        # Greenhouse effect: more absorbers -> less outgoing longwave.
        assert olr_ghg < olr_base, (
            f"adding 4xCO2+CH4+N2O+O3 must reduce OLR "
            f"(base {olr_base:.2f}, ghg {olr_ghg:.2f} W/m2)"
        )
        # The reduction should be a few W/m2, not a rounding artefact.
        assert olr_base - olr_ghg > 1.0


class TestColumnVectorHelper:
    def test_column_vector_reshapes_vmapped_scalars(self):
        from jcm.physics.radiation.rrtmgp import _column_vector_rrtmgp

        vals = jnp.arange(6.0).reshape(6, 1)
        out = _column_vector_rrtmgp(vals, 6)
        assert out.shape == (6,)
        assert jnp.allclose(out, jnp.arange(6.0))


class TestRRTMGPTermComputeAndCache:
    """Term-level ``__call__``: full compute, sub-step caching, carry wiring.

    Drives ``RRTMGPRadiation`` exactly the way ``ComposablePhysics`` does —
    a column-vectorised ``PhysicsState`` plus the shared diagnostics dict —
    with ``radiation_interval = 2 x dt``, so the first call must run the
    full scheme and the second call must replay the cached heating rates
    (while still bumping the radiation step counter).
    """

    NLEV = 8
    NCOLS = 2
    DT = 1800.0

    def _term_and_inputs(self):
        import jcm.constants as c
        from flax import nnx
        from jcm.forcing import ForcingData
        from jcm.physics.aerosol.aerosol_types import AerosolData
        from jcm.physics.chemistry.simple_chemistry import ChemistryData
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics.radiation.radiation_types import RadiationData
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics_interface import PhysicsState

        nlev, ncols = self.NLEV, self.NCOLS

        # Recompute every 2nd step: interval = 2 x dt.
        params = RadiationParameters.default(radiation_interval=2 * self.DT)
        term = RRTMGPRadiation(params=params, compute_cre=False)
        # Per-column lat/lon normally cached from the model coords; two
        # columns (equator, mid-latitude) are enough here.
        term._lats = nnx.Variable(jnp.array([0.0, 45.0]))
        term._lons = nnx.Variable(jnp.array([0.0, 90.0]))

        # A plausible TOA-first clear-sky column, broadcast to 2 columns.
        col = lambda profile: jnp.broadcast_to(  # noqa: E731
            jnp.asarray(profile)[:, None], (len(profile), ncols),
        )
        p_full = jnp.linspace(2e3, 9.5e4, nlev)
        p_half = jnp.linspace(1e3, 1.0e5, nlev + 1)
        T = jnp.linspace(220.0, 288.0, nlev)
        rho = p_full / (c.rd * T)
        dz = (p_half[1:] - p_half[:-1]) / (rho * c.grav)

        state = PhysicsState.zeros(
            (nlev, ncols),
            temperature=col(T),
            specific_humidity=col(jnp.geomspace(1e-6, 8e-3, nlev)),
            normalized_surface_pressure=jnp.ones((ncols,)),
        )
        diagnostics = {
            "_dt_seconds": self.DT,
            "pressure_full": col(p_full),
            "pressure_half": col(p_half),
            "layer_thickness": col(dz),
            "air_density": col(rho),
            "radiation": RadiationData.zeros((ncols,), nlev).copy(
                surface_albedo_vis=jnp.full((ncols,), 0.07),
                surface_albedo_nir=jnp.full((ncols,), 0.07),
                surface_emissivity=jnp.full((ncols,), 0.98),
            ),
            "surface": SurfaceData.zeros((ncols,), nlev).copy(
                surface_temperature=jnp.full((ncols,), 288.0),
            ),
            "chemistry": ChemistryData.zeros((ncols,), nlev),
            "aerosol": AerosolData.zeros((ncols,), nlev),
            "clouds": CloudData.zeros((ncols,), nlev),
        }
        forcing = ForcingData.zeros((ncols,))
        return term, state, diagnostics, forcing

    def test_compute_then_cache_cycle(self):
        term, state, diagnostics, forcing = self._term_and_inputs()

        # --- Step 0: radiation step counter 0 -> full compute.
        tend1, diag1 = term(state, diagnostics, forcing, None)
        rad1 = diag1["radiation"]
        assert int(rad1.step) == 1, "step counter must advance on compute"
        assert tend1.temperature.shape == (self.NLEV, self.NCOLS)
        assert bool(jnp.all(jnp.isfinite(tend1.temperature)))
        # Clear-sky OLR from a 288 K surface must be physically sized.
        olr = np.asarray(rad1.toa_lw_up)
        assert olr.shape == (self.NCOLS,)
        assert np.all(olr > 100.0) and np.all(olr < 400.0), f"OLR {olr}"
        # The tendency the term reports is the total heating rate.
        np.testing.assert_allclose(
            np.asarray(tend1.temperature),
            np.asarray(rad1.sw_heating_rate + rad1.lw_heating_rate),
            rtol=1e-5, atol=1e-10,
        )
        # CRE mirror onto the clouds carry.
        np.testing.assert_array_equal(
            np.asarray(diag1["clouds"].toa_lw_up_all),
            np.asarray(rad1.toa_lw_up),
        )
        np.testing.assert_array_equal(
            np.asarray(diag1["clouds"].toa_sw_up_all),
            np.asarray(rad1.toa_sw_up),
        )

        # --- Step 1: interval = 2 steps -> cached replay. Perturb the
        # atmosphere to prove the output comes from the cache, not a
        # recompute.
        hot_state = state.copy(temperature=state.temperature + 10.0)
        tend2, diag2 = term(hot_state, diag1, forcing, None)
        rad2 = diag2["radiation"]
        assert int(rad2.step) == 2, "step counter must advance on cached steps"
        np.testing.assert_array_equal(
            np.asarray(rad2.toa_lw_up), np.asarray(rad1.toa_lw_up),
        )
        np.testing.assert_array_equal(
            np.asarray(rad2.lw_heating_rate),
            np.asarray(rad1.lw_heating_rate),
        )
        np.testing.assert_allclose(
            np.asarray(tend2.temperature),
            np.asarray(rad1.sw_heating_rate + rad1.lw_heating_rate),
            rtol=1e-5, atol=1e-10,
        )


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


class TestRRTMGPVerticalOrientation:
    """Per-level inputs must reach the solver in the library's frame.

    jcm physics columns are TOA-first while the jax-rrtmgp library is
    surface-first; temperature/pressure/clouds flip in
    ``prepare_rrtmgp_data`` but the per-band aerosol and the gas-VMR
    profiles previously skipped the flip. The signature failure was
    surface-concentrated aerosol tau acting at the model top: spurious
    top-level LW cooling from JAM's dust/BC LW bands grew a two-grid
    oscillation at 1 Pa that NaN'd coupled JAM runs by day ~10, and
    MACv2-SP's SW tau heated the top of every RRTMGP run. These tests
    pin locality (aerosol acts at the levels that carry it) and the
    ozone profile's orientation.
    """

    def _lw_heating(self, inputs):
        _, diag = radiation_scheme_rrtmgp(**inputs)
        return np.asarray(diag.lw_heating_rate)

    def test_low_level_lw_aerosol_acts_low_not_at_top(self):
        from jcm.physics.aerosol.aerosol_types import AerosolData

        nlev = 20
        inputs = _make_inputs(nlev=nlev)
        inputs["compute_cre"] = False

        clean = AerosolData.zeros((), nlev, n_bnd_sw=14, n_bnd_lw=16)
        # Dust-like LW aerosol confined to the three SURFACE layers
        # (TOA-first indices -3:). With the historical orientation bug
        # these landed at the library top and cooled the TOA layer ~100x.
        tau = jnp.zeros((16, nlev)).at[:, -3:].set(0.05)
        ssa = jnp.zeros((16, nlev)).at[:, -3:].set(0.5)
        asy = jnp.zeros((16, nlev)).at[:, -3:].set(0.3)
        dusty = clean.copy(
            aod_lw_per_band=tau, ssa_lw_per_band=ssa, asy_lw_per_band=asy,
        )

        base = self._lw_heating({**inputs, "aerosol_data": clean})
        pert = self._lw_heating({**inputs, "aerosol_data": dusty})
        diff = pert - base

        top_change = abs(diff[0])
        low_change = np.abs(diff[-3:]).max()
        # The aerosol must act where it is: significant response in the
        # loaded surface layers, and the TOA layer essentially untouched
        # (it holds no aerosol and only sees the tiny OLR perturbation).
        assert low_change > 1e-7, f"no low-level LW response ({low_change})"
        assert top_change < 0.1 * low_change, (
            f"top-layer LW heating changed by {top_change} vs low-level "
            f"{low_change} — surface aerosol is acting at the model top "
            "(vertical orientation regression)"
        )

    def test_ozone_profile_orientation_reaches_gas_optics(self):
        nlev = 20
        inputs = _make_inputs(nlev=nlev)
        inputs["compute_cre"] = False

        pf = np.asarray(inputs["pressure_levels"])
        # Stratospheric ozone: 8 ppm bump centred at 20 hPa (upper part
        # of this TOA-first column), near-zero in the troposphere.
        o3 = jnp.asarray(8.0e-6 * np.exp(
            -((np.log(pf) - np.log(2000.0)) / 1.0) ** 2
        ))

        _, d_correct = radiation_scheme_rrtmgp(**{**inputs, "ozone_vmr": o3})
        _, d_flipped = radiation_scheme_rrtmgp(
            **{**inputs, "ozone_vmr": o3[::-1]}
        )
        sw_c = np.asarray(d_correct.sw_heating_rate)
        sw_f = np.asarray(d_flipped.sw_heating_rate)
        assert sw_c.max() > 1e-7, "daytime column expected (noon equator)"
        # Orientation must matter at all (guards against the profile being
        # silently discarded)...
        assert np.abs(sw_c - sw_f).max() > 1e-7
        # ...and the physically-oriented profile must put the ozone SW
        # heating in the upper half of the column (TOA-first indices).
        upper = sw_c[: nlev // 2].max()
        lower = sw_c[nlev // 2:].max()
        assert upper > lower, (
            f"SW heating peak below mid-column (upper {upper}, lower "
            f"{lower}) — ozone profile entering gas optics upside down"
        )


class TestRRTMGPAerosolFreeAlternate(TestRRTMGPTermComputeAndCache):
    """Alternating aerosol-on/aerosol-off single solve (jax-gcm#583).

    Inherits the column harness from ``TestRRTMGPTermComputeAndCache`` and
    re-drives it with ``aerosol_free_alternate=True``. The contract:
    on even radiation calls the solve carries aerosol and refreshes the
    all-sky TOA slots (``*noa`` held); on odd radiation calls it drops the
    aerosol optics and refreshes the ``*noa`` slots (all-sky held).
    """

    def _alternating_term_and_inputs(self):
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        term, state, diagnostics, forcing = self._term_and_inputs()
        alt = RRTMGPRadiation(
            params=term.params.get_value(),
            compute_cre=False,
            aerosol_free_diagnostics=True,
            aerosol_free_alternate=True,
        )
        alt._lats, alt._lons = term._lats, term._lons

        # A visibly dusty column — with zero aerosol the on/off solves would
        # be identical and the test could not tell the branches apart.
        aer = diagnostics["aerosol"]
        nlev, ncols = self.NLEV, self.NCOLS
        diagnostics = {**diagnostics, "aerosol": aer.copy(
            aod_profile=jnp.full((nlev, ncols), 0.05),
            ssa_profile=jnp.full((nlev, ncols), 0.9),
            asy_profile=jnp.full((nlev, ncols), 0.7),
            aod_total=jnp.full((ncols,), 0.4),
            aod_sw_per_band=jnp.full((14, nlev, ncols), 0.05),
            ssa_sw_per_band=jnp.full((14, nlev, ncols), 0.9),
            asy_sw_per_band=jnp.full((14, nlev, ncols), 0.7),
            aod_lw_per_band=jnp.full((16, nlev, ncols), 0.02),
            ssa_lw_per_band=jnp.full((16, nlev, ncols), 0.5),
            asy_lw_per_band=jnp.full((16, nlev, ncols), 0.5),
        )}
        return alt, state, diagnostics, forcing

    def test_alternate_requires_the_diagnostic_to_be_on(self):
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation
        with pytest.raises(ValueError, match="aerosol_free_diagnostics"):
            RRTMGPRadiation(aerosol_free_alternate=True)

    def test_slots_alternate_between_aerosol_on_and_off(self):
        term, state, diagnostics, forcing = self._alternating_term_and_inputs()

        # --- radiation call 0 (even): aerosol ON.
        _, d0 = term(state, diagnostics, forcing, None)
        r0 = d0["radiation"]
        assert np.all(np.asarray(r0.toa_lw_up) > 100.0), "all-sky must be fresh"
        # *noa still holds its initial (zero) value — no aerosol-off solve ran.
        np.testing.assert_array_equal(np.asarray(r0.toa_lw_up_noa), 0.0)
        np.testing.assert_array_equal(np.asarray(r0.toa_sw_up_noa), 0.0)

        # --- call 1: sub-step, cached replay (interval = 2 x dt).
        _, d1 = term(state, d0, forcing, None)

        # --- radiation call 2 (odd): aerosol OFF.
        _, d2 = term(state, d1, forcing, None)
        r2 = d2["radiation"]
        # All-sky is HELD from call 0 — this step's solve had no aerosol.
        np.testing.assert_array_equal(
            np.asarray(r2.toa_lw_up), np.asarray(r0.toa_lw_up),
        )
        np.testing.assert_array_equal(
            np.asarray(r2.toa_sw_up), np.asarray(r0.toa_sw_up),
        )
        # *noa is now populated by the aerosol-free solve.
        noa = np.asarray(r2.toa_lw_up_noa)
        assert np.all(noa > 100.0), f"aerosol-free OLR not populated: {noa}"
        # ...and it differs from the all-sky value, i.e. the aerosol really
        # was removed rather than the same solve being copied across.
        assert not np.allclose(noa, np.asarray(r0.toa_lw_up)), (
            "aerosol-off OLR identical to aerosol-on — optics were not zeroed"
        )

    def test_model_feels_aerosol_free_heating_on_odd_calls(self):
        """The whole point of the approximation: heating alternates too.

        On the aerosol-off call the tendency handed back to the model is the
        aerosol-free one, so the direct effect enters with a 50 % duty cycle.
        """
        term, state, diagnostics, forcing = self._alternating_term_and_inputs()
        t0, d0 = term(state, diagnostics, forcing, None)      # aerosol ON
        _, d1 = term(state, d0, forcing, None)                # cached
        t2, d2 = term(state, d1, forcing, None)               # aerosol OFF

        assert not np.allclose(
            np.asarray(t0.temperature), np.asarray(t2.temperature),
        ), "heating identical on/off — the alternation is not reaching the model"
