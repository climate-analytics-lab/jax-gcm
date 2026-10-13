"""Tests for ``jcm.rce`` — single-column radiative-convective equilibrium.

The fast tests cover the *machinery* — the fixed-RH humidity closure (kg/kg,
the canonical convention), the steady-insolation helper, and the physics-package
composition. The slow tests run short RCE integrations end-to-end (grey and
RRTMGP) and assert the column stays finite, approaches a steady atmospheric
state (``dT/dt → 0``; note the TOA flux need *not* vanish in a fixed-SST RCE —
the imbalance is the implied ocean heat flux), keeps a convectively bounded
lapse rate, and produces order-of-magnitude-reasonable OLR.
"""

import functools
import unittest

import numpy as np
import jax.numpy as jnp
import pytest
from dinosaur.sigma_coordinates import SigmaCoordinates

import jcm.constants as c
from jcm.forcing import SolarGeometry
from jcm.rce import (
    _STRATOSPHERE_Q_FLOOR,
    AerosolFree,
    _pressure_centers,
    closure_saturation_specific_humidity,
    fixed_rh_closure,
    rce_column,
    rce_initial_state,
    rce_physics,
    run_rce,
    steady_insolation,
)
from jcm.physics.clouds.sundqvist import CloudParameters
from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.single_column_model import SingleColumnModel


class TestFixedRhClosure(unittest.TestCase):
    """The fixed-RH humidity closure (uniform troposphere, dry stratosphere)."""

    def setUp(self):
        self.vertical = SigmaCoordinates.equidistant(20)
        # rce_initial_state builds a correctly top-first ordered column
        # (index 0 = model top, index -1 = surface).
        self.ic = rce_initial_state(self.vertical, sst=300.0, relative_humidity=0.7)

    def test_closure_sets_uniform_tropospheric_rh_in_kg_per_kg(self):
        """Closure holds RH = env value through the troposphere (kg/kg)."""
        rh = 0.7
        closure = fixed_rh_closure(rh, self.vertical)
        out = closure(self.ic, forcing=None)

        ps = float(self.ic.normalized_surface_pressure) * c.p0
        pfull = _pressure_centers(self.vertical, jnp.asarray(ps))
        qsat = closure_saturation_specific_humidity(pfull, self.ic.temperature)
        # Tropospheric levels (p ≥ 100 hPa) sit at the uniform environmental RH —
        # no surface-to-top taper that would dry the convecting layer.
        trop = np.asarray(pfull) >= 1.0e4
        rh_diag = np.asarray(out.specific_humidity) / np.asarray(qsat)
        self.assertTrue(np.allclose(rh_diag[trop], rh, atol=1e-4))
        # Surface (index -1) value is a realistic several g/kg expressed in kg/kg
        # (~0.005–0.03); a ~1000x reading would mean the closure emitted g/kg.
        self.assertGreater(float(out.specific_humidity[-1]), 0.005)
        self.assertLess(float(out.specific_humidity[-1]), 0.03)

    def test_stratosphere_is_tapered_dry_and_floored(self):
        """RH tapers off in the stratosphere; q stays finite and ≥ trace floor."""
        # A grid that reaches the near-vacuum top, so the hard floor is exercised.
        vertical = SigmaCoordinates.equidistant(60)
        ic = rce_initial_state(vertical, sst=300.0, relative_humidity=0.7)
        out = fixed_rh_closure(0.7, vertical)(ic, forcing=None)
        pfull = np.asarray(_pressure_centers(vertical, jnp.asarray(c.p0)))

        self.assertTrue(jnp.all(jnp.isfinite(out.specific_humidity)))
        self.assertTrue(jnp.all(out.specific_humidity >= _STRATOSPHERE_Q_FLOOR))
        # Above the taper window (p ≤ 20 hPa) RH is forced to zero, so q clamps
        # to the trace floor — no spurious stratospheric moisture for RRTMGP.
        strat = pfull <= 2.0e3
        self.assertTrue(np.any(strat))
        self.assertTrue(np.allclose(
            np.asarray(out.specific_humidity)[strat], _STRATOSPHERE_Q_FLOOR))

    def test_humidity_tracks_temperature(self):
        """Warmer columns hold more vapour at the surface (q slaved to T)."""
        closure = fixed_rh_closure(0.7, self.vertical)
        warm = closure(self.ic.copy(temperature=self.ic.temperature + 5.0), forcing=None)
        cool = closure(self.ic, forcing=None)
        self.assertGreater(
            float(warm.specific_humidity[-1]), float(cool.specific_humidity[-1]),
        )


class TestSteadyInsolation(unittest.TestCase):
    """The fixed-``SolarGeometry`` helper."""

    def test_returns_constant_solar_geometry(self):
        solar = steady_insolation(day_of_year_fraction=0.22, time_of_day_fraction=0.5)
        self.assertIsInstance(solar, SolarGeometry)
        self.assertAlmostEqual(float(solar.tyear), 0.22, places=5)
        self.assertAlmostEqual(float(solar.orbital_phase), 2 * np.pi * 0.22, places=4)
        self.assertAlmostEqual(float(solar.synodic_phase), 2 * np.pi * 0.5, places=4)


class TestRcePhysicsComposition(unittest.TestCase):
    """``rce_physics`` composes the minimal radiative-convective stack directly."""

    def test_default_is_minimal_radiative_convective(self):
        physics = rce_physics()
        self.assertEqual(
            [t.category for t in physics.terms],
            ["prepare", "forcing", "clear_sky", "radiation", "convection"],
        )
        names = {t.category: t.name for t in physics.terms}
        self.assertEqual(names["radiation"], "rrtmgp_radiation")
        self.assertEqual(names["convection"], "betts_miller_convection")

    def test_accepts_custom_radiation_and_convection_terms(self):
        from jcm.physics.convection.tiedtke_nordeng import TiedtkeConvection
        from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation

        physics = rce_physics(
            radiation=GreyTwoStreamRadiation(), convection=TiedtkeConvection(),
        )
        names = {t.category: t.name for t in physics.terms}
        self.assertEqual(names["radiation"], "grey_two_stream_radiation")
        self.assertEqual(names["convection"], "tiedtke_convection")

    def test_aerosol_free_replaces_only_the_aerosol_term(self):
        """``AerosolFree`` swaps MACv2-SP out and publishes clean-air aerosol.

        Zero optical depth everywhere and the clean-air Twomey factor
        (``cdnc_factor = 1``), leaving the rest of the stack -- clouds
        included -- in place.
        """
        from jcm.physics.echam.testing import idealized_echam_physics

        full = idealized_echam_physics()
        physics = full.replace("aerosol", AerosolFree())
        self.assertEqual(
            [t.category for t in physics.terms],
            [t.category for t in full.terms],
        )
        names = {t.category: t.name for t in physics.terms}
        self.assertEqual(names["aerosol"], "aerosol_free")

        term = AerosolFree()
        nlev, ncols = 5, 3
        state = rce_initial_state(
            SigmaCoordinates.equidistant(nlev), sst=300.0,
        )
        state = state.copy(
            temperature=jnp.broadcast_to(state.temperature[:, None], (nlev, ncols)),
        )
        tend, diags = term(state, {"clouds": "kept"}, None, None)
        self.assertEqual(diags["clouds"], "kept")
        aerosol = diags["aerosol"]
        self.assertEqual(float(jnp.max(jnp.abs(aerosol.aod_profile))), 0.0)
        np.testing.assert_allclose(np.asarray(aerosol.cdnc_factor), 1.0)
        self.assertEqual(float(jnp.max(jnp.abs(tend.temperature))), 0.0)


class TestRceColumnConstruction(unittest.TestCase):
    """``rce_column`` wiring of the SCM, forcing, free-evolution and closure."""

    def test_builds_scm_with_fixed_sst_and_free_temperature(self):
        scm = rce_column(sst=302.0, relative_humidity=0.7, vertical=SigmaCoordinates.equidistant(8),
                         radiation=GreyTwoStreamRadiation())
        self.assertIsInstance(scm, SingleColumnModel)
        self.assertEqual(scm.free_evolve, ("temperature",))
        self.assertIsNotNone(scm.state_closure)
        self.assertAlmostEqual(float(scm.forcing.sea_surface_temperature[0, 0]), 302.0)

    def test_default_betts_miller_rhbm_sits_below_environmental_rh(self):
        # convection=None builds the default Betts-Miller from the column knobs.
        # rhbm is decoupled from the environmental/closure RH and defaults to
        # relative_humidity − 0.1 so deep (precipitating) convection fires; a
        # degenerate rhbm == relative_humidity would zero the scheme out.
        scm = rce_column(relative_humidity=0.65, tau_convection=5400.0,
                         vertical=SigmaCoordinates.equidistant(8), radiation=GreyTwoStreamRadiation())
        params = [t for t in scm.physics.terms if t.category == "convection"][0].params.get_value()
        self.assertAlmostEqual(float(params.rhbm), 0.55, places=5)
        self.assertLess(float(params.rhbm), 0.65)
        self.assertAlmostEqual(float(params.tau_bm), 5400.0, places=3)

    def test_explicit_convective_rh_is_tracked(self):
        scm = rce_column(relative_humidity=0.8, convective_rh=0.6,
                         vertical=SigmaCoordinates.equidistant(8), radiation=GreyTwoStreamRadiation())
        params = [t for t in scm.physics.terms if t.category == "convection"][0].params.get_value()
        self.assertAlmostEqual(float(params.rhbm), 0.6, places=5)

    def test_degenerate_convective_rh_is_rejected(self):
        # convective_rh >= relative_humidity is a silent no-op (the bug this PR
        # fixes): the default Betts-Miller stays in its non-precipitating branch
        # and produces zero tendency. Reject it at construction.
        with self.assertRaisesRegex(ValueError, "convective_rh"):
            rce_column(relative_humidity=0.7, convective_rh=0.7,
                       vertical=SigmaCoordinates.equidistant(8),
                       radiation=GreyTwoStreamRadiation())

    def test_initial_state_can_trigger_the_echam_cubase_walk(self):
        """The seeded tropical column must be one Tiedtke can convect in.

        ECHAM's ``cubase`` drops a column the moment the dry-lifted parcel
        is not buoyant, so a sounding running at ``lapse_rate`` down to the
        surface never reaches its own LCL and gets NO convection — which is
        how the release-validation SCM lost its whole convective aerosol
        pathway (#773) while every convection unit test stayed green.
        """
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            ConvectionParameters, find_cloud_base,
        )
        from jcm.physics.echam.echam_levels import get_echam_levels
        vertical = get_echam_levels(47)
        ic = rce_initial_state(vertical, sst=302.0, relative_humidity=0.8)
        pfull = _pressure_centers(vertical, jnp.asarray(c.p0))
        ph = (np.asarray(vertical.a_boundaries)
              + np.asarray(vertical.b_boundaries) * float(c.p0))
        dp = jnp.asarray(np.abs(np.diff(ph)))
        rho = pfull / (c.rd * ic.temperature)
        dz = dp / (rho * c.grav)
        # A modest sub-grid excess (zlift = 0.5 K). The half-level walk
        # tests the parcel against ``cuini``'s interface environment, whose
        # moist-adiabatic interpolation from the level above is cooler than
        # the full levels in a conditionally unstable layer; with ECHAM's
        # maximum 1 K excess even the unmixed sounding just reaches its LCL,
        # so the control below would not discriminate.
        cfg = ConvectionParameters.default(cu_thvsig=0.5)
        _cb, found = find_cloud_base(ic.temperature, ic.specific_humidity,
                                     pfull, cfg, None, dz,
                                     pressure_half=jnp.asarray(ph))
        self.assertTrue(bool(found), "no cloud base in the seeded RCE column")
        # ...because the sub-cloud layer is well mixed. Compare the
        # potential-temperature spread through it against the unmixed
        # profile rather than an absolute threshold: the seeded height is a
        # scale-height estimate, so the mixed layer is near-neutral rather
        # than exactly isentropic in the true (p, T) coordinates.
        unmixed = rce_initial_state(vertical, sst=302.0, relative_humidity=0.8,
                                    mixed_layer_top_m=0.0)
        exner = (float(c.p0) / np.asarray(pfull)) ** (float(c.rd) / float(c.cpd))
        z = np.asarray(ic.geopotential) / c.grav
        ml = z < 700.0
        self.assertTrue(np.any(ml))
        spread = float(np.ptp(np.asarray(ic.temperature)[ml] * exner[ml]))
        spread_unmixed = float(
            np.ptp(np.asarray(unmixed.temperature)[ml] * exner[ml]))
        self.assertLess(spread, 0.5 * spread_unmixed)
        _cb2, found_unmixed = find_cloud_base(
            unmixed.temperature, unmixed.specific_humidity, pfull, cfg, None, dz,
            pressure_half=jnp.asarray(ph),
        )
        self.assertFalse(bool(found_unmixed),
                         "the unmixed profile should not trigger cubase")

    def test_interactive_humidity_frees_q_and_drops_closure(self):
        scm = rce_column(relative_humidity=0.7, vertical=SigmaCoordinates.equidistant(8),
                         radiation=GreyTwoStreamRadiation(), interactive_humidity=True)
        self.assertIn("specific_humidity", scm.free_evolve)
        self.assertIn("temperature", scm.free_evolve)
        self.assertIsNone(scm.state_closure)


@functools.lru_cache(maxsize=1)
def _grey_rce_rollout(sst=300.0, relative_humidity=0.7, n_days=50.0):
    """One shared 50-day grey RCE rollout for the slow grey-RCE classes.

    The equilibration, lapse-rate, and radiative-convective-balance
    assertions all interrogate the *same* physical configuration, so they
    share a single cached integration (the pattern
    ``scm_boundary_layer_cases_test.py`` uses) instead of running four
    independent 40–50-day rollouts. PR CI runs the slow suite in a single
    process, so the cache is fully effective there.
    """
    vertical = SigmaCoordinates.equidistant(20)
    scm = rce_column(
        sst=sst, relative_humidity=relative_humidity, lat_deg=0.0,
        radiation=GreyTwoStreamRadiation(
            params=RadiationParameters.default(solar_constant=420.0),
        ),
        vertical=vertical, dt_seconds=1800.0,  # convective_rh defaults to RH−0.1
    )
    ic = rce_initial_state(vertical, sst=sst, relative_humidity=relative_humidity)
    return vertical, scm, run_rce(scm, ic, n_days=n_days)


@pytest.mark.slow
class TestRceIntegrationGrey(unittest.TestCase):
    """End-to-end grey-radiation RCE: physical troposphere, bounded lapse rate.

    Grey radiation on a cheap sigma grid keeps this fast; the RRTMGP Case-1
    configuration is exercised separately in :class:`TestRceIntegrationRrtmgp`.
    Equilibration itself is pinned (with tighter thresholds) by
    :class:`TestRceRadiativeConvectiveBalance` on the same cached rollout.
    """

    def test_troposphere_physical_and_lapse_rate_bounded(self):
        vertical, _, preds = _grey_rce_rollout()
        T = np.asarray(preds.relaxed_states["temperature"][-1])
        # Exclude the single thin top layer (grey radiation over-warms it —
        # a known artefact of the scheme at the model top, not a framework bug).
        trop = T[2:]
        self.assertTrue(np.all(trop > 150.0))
        self.assertTrue(np.all(trop < 360.0))

        # Lapse rate in the lower/mid troposphere must not exceed the dry
        # adiabat (~10 K/km) — i.e. convection has removed the super-adiabatic
        # layers the radiative-equilibrium profile would otherwise have.
        sigma = 0.5 * (np.asarray(vertical.boundaries)[:-1]
                       + np.asarray(vertical.boundaries)[1:])
        z = -7.6e3 * np.log(np.maximum(sigma, 1e-4))  # approx height
        lower = sigma > 0.4  # lower troposphere
        dT = np.diff(T[lower])
        dz = np.diff(z[lower])
        lapse = -dT / dz  # K/m
        self.assertTrue(np.all(lapse < 0.011),
                        f"super-adiabatic lapse rate: max {np.max(lapse) * 1000:.1f} K/km")


@pytest.mark.slow
class TestRceRadiativeConvectiveBalance(unittest.TestCase):
    """The defining RCE check: convection *balances* radiation through the column.

    The other slow tests assert the column equilibrates (``dT/dt → 0``) and stays
    physical — but a fixed-SST column reaches those targets in pure *radiative*
    equilibrium with convection doing nothing (the regression this guards: an
    rhbm tied to the environmental RH puts Betts-Miller in its non-precipitating
    branch). This test instead asserts convection is genuinely active and that
    convective heating cancels radiative cooling layer-by-layer in the convecting
    troposphere — the homebrew RCE result (issue #523): both terms ~O(0.1–1)
    K/day, opposed, summing to a near-zero residual.

    Grey radiation on a cheap sigma grid keeps a ~50-day integration fast; the
    physics of the convective trigger is identical to the RRTMGP case.
    """

    def _run(self, sst=300.0, relative_humidity=0.7, n_days=50.0):
        return _grey_rce_rollout(sst, relative_humidity, n_days)

    @staticmethod
    def _heating_rates(preds):
        """Convective and radiative heating [K/day] at the final step, ``(nlev,)``.

        Radiation and convection are the only terms touching temperature in the
        RCE stack, so the convective contribution is the total physics tendency
        minus the radiative heating reported in the diagnostics dict.
        """
        rad = preds.physics_data["radiation"]
        rad_h = (np.asarray(rad.sw_heating_rate)[..., 0]
                 + np.asarray(rad.lw_heating_rate)[..., 0]) * 86400.0
        total = np.asarray(preds.tendencies.temperature) * 86400.0
        conv_h = total - rad_h
        return conv_h[-1], rad_h[-1], total[-1]

    def test_convection_is_active_and_balances_radiation(self):
        _, _, preds = self._run()
        conv, rad, net = self._heating_rates(preds)
        self.assertTrue(np.all(np.isfinite(conv)))

        # 1) Convection is genuinely doing work — precip > 0 and substantial
        #    heating somewhere in the column (this is exactly zero under the bug).
        precip = np.asarray(
            preds.physics_data["betts_miller_precip"]
        ).reshape(len(preds.times), -1)[-1, -1]
        self.assertGreater(float(precip) * 86400.0, 0.05,
                           "no precipitation — convection never fired")
        self.assertGreater(np.max(np.abs(conv)), 0.1,
                           "convective heating is negligible — convection silent")

        # 2) Radiative-convective balance in the convecting layer: where
        #    convection is active it cancels the radiative cooling, so the net
        #    residual is small compared to either large opposing term.
        active = np.abs(conv) > 0.2 * np.max(np.abs(conv))
        self.assertGreaterEqual(int(np.sum(active)), 3)
        rms_conv = np.sqrt(np.mean(conv[active] ** 2))
        rms_rad = np.sqrt(np.mean(rad[active] ** 2))
        rms_net = np.sqrt(np.mean(net[active] ** 2))
        self.assertGreater(rms_rad, 0.5 * rms_conv)   # comparable magnitudes
        self.assertLess(rms_net, 0.3 * rms_conv)      # they cancel → balance

        # 3) Layer-by-layer the two terms are near mirror images (the figure in
        #    the homebrew notebook): convective heating ≈ −radiative heating.
        corr = np.corrcoef(conv[active], -rad[active])[0, 1]
        self.assertGreater(corr, 0.9)

    def test_settles_toward_equilibrium(self):
        _, _, preds = self._run()
        dT = preds.tendencies.temperature
        rms0 = float(jnp.sqrt(jnp.mean(dT[0] ** 2)) * 86400.0)
        rms_end = float(jnp.sqrt(jnp.mean(dT[-1] ** 2)) * 86400.0)
        self.assertLess(rms_end, 0.3 * rms0)
        self.assertLess(rms_end, 0.2)  # K/day, near steady


@pytest.mark.slow
class TestRceIntegrationRrtmgp(unittest.TestCase):
    """RRTMGP + Betts-Miller fixed-RH RCE on echam-47 — issue #523 Case 1.

    With specific humidity in its canonical kg/kg units, RRTMGP runs cleanly and
    the column reaches a radiative-convective balance with an Earth-like OLR and
    a near-surface RH that matches the prescribed value.
    """

    def test_case1_equilibrates_with_reasonable_olr_and_rh(self):
        scm = rce_column(
            sst=300.0, relative_humidity=0.7, solar_constant=728.4, lat_deg=42.55,
            nlev=47, dt_seconds=1200.0,  # RRTMGP is the default radiation
        )
        ic = rce_initial_state(scm.vertical, sst=300.0, relative_humidity=0.7)
        preds = run_rce(scm, ic, n_days=20.0)

        T = preds.relaxed_states["temperature"]
        dT = preds.tendencies.temperature
        rad = preds.physics_data["radiation"]

        self.assertTrue(bool(jnp.all(jnp.isfinite(T[-1]))))

        rms0 = float(jnp.sqrt(jnp.mean(dT[0] ** 2)) * 86400.0)
        rms_end = float(jnp.sqrt(jnp.mean(dT[-1] ** 2)) * 86400.0)
        self.assertLess(rms_end, 0.3 * rms0)  # settling toward equilibrium

        # Convection must actually be running (not a silent radiative-only
        # equilibrium): the column precipitates. With the default convective_rh
        # (= relative_humidity − 0.1 = 0.6) Betts-Miller stays in its deep,
        # precipitating branch.
        precip = np.asarray(
            preds.physics_data["betts_miller_precip"]
        ).reshape(len(preds.times), -1)[-1, -1]
        self.assertGreater(float(precip) * 86400.0, 0.05)

        # Outgoing longwave should sit in the broad terrestrial range.
        olr = float(rad.toa_lw_up[-1].reshape(-1)[0])
        self.assertGreater(olr, 150.0)
        self.assertLess(olr, 320.0)

        # The fixed-RH closure holds the diagnosed near-surface RH at the
        # requested value — a direct check that kg/kg units are correct
        # end-to-end (a unit error would put this near saturation or ~0).
        rh = np.asarray(preds.physics_data["relative_humidity"])
        rh_surface = float(rh[-1].reshape(T.shape[1], -1)[-1, 0])
        self.assertAlmostEqual(rh_surface, 0.70, delta=0.1)


@pytest.mark.slow
class TestRceWholeModelTiedtke(unittest.TestCase):
    """RCE on the *full* ECHAM term stack, with RRTMGP radiation.

    Unlike the minimal radiative-convective ``rce_physics`` stack, this drives
    the complete ECHAM column (``echam_physics()``: surface turbulent fluxes,
    TTE-TKE vertical diffusion, the Sundqvist cover, Tiedtke-Nordeng
    convection, the 1-moment microphysics and RRTMGP radiation) as a genuine
    single-column integration of the whole model. The configuration is in
    ``docs/source/design/rce_testbed.md``: SST 300 K, 0°N/0°E, 47 levels,
    ``dt`` 900 s, a prescribed uniform 5 m/s wind (the surface evaporation
    needs a wind), prognostic humidity, no large-scale forcing (an RCE column
    has no mean vertical motion), 80 days. The solar constant is 420 W/m²,
    which at this fixed sun delivers 431 W/m² at the top of the atmosphere, 5 %
    above RCEMIP's 409.6; it is left at the value the testbed was built with.
    The CRE diagnostic is off (``radiation_compute_cre=False``): it adds a
    clear-sky solve and nothing else, and every number below is identical with
    it on.

    The column is **aerosol-free** (``AerosolFree`` replaces MACv2-SP). The
    MACv2-SP plumes are a geographic climatology, and this column at 0°N/0°E
    sits in the Central African biomass-burning plume (AOD 0.33 at 550 nm,
    SSA 0.87, Ångström 2), which on its own absorbs ~100 W/m² of shortwave in
    the lower troposphere. An RCE test means the idealised clear-air column.

    The assertions are on the **time mean** over days 40-80: a single-column
    mass-flux scheme in RCE has an intrinsic high-frequency convective cycle,
    but the time-mean column must be a physical radiative-convective
    equilibrium whose scatter stays bounded and whose convection never dies
    out. The integration runs once per class (``setUpClass``) and the tests
    read its series.

    **Where the bounds come from.** Each is the extreme over seven
    trajectories (the one this test runs, and six differing from it only by
    1e-4 K of initial temperature noise) and over the 40-day windows days
    40-80 and 80-120 (the unperturbed trajectory also over 120-160 and
    160-200), plus a margin of at least three times the across-trajectory
    range of the window mean, rounded outward. Measured on dev 40701518 with
    the interior stability of ``echam_physics()`` (ECHAM's moist,
    cloud-weighted buoyancy, :doc:`/science/vertical_diffusion`; jax 0.10.2,
    jax-rrtmgp 0.5.0, float32, CPU; with the land tile merged, dev f1f0df1e,
    this trajectory gives P / E 0.965 and TOA net 38.5 W/m²); the table and
    the measurements are in the design page.

    ============================  ==================  =====================
    quantity                      measured extreme    pinned
    ============================  ==================  =====================
    P / E                         0.958 .. 1.001      > 0.93
    Tiedtke steps (ktype > 0)     0.978 .. 0.984      > 0.90
    time-mean convective P        0.68 .. 0.79 mm/d   > 0
    column water drift            -0.0006 .. 0.0505   |.| < 0.1 mm/d
    water budget residual / E     < 3.5e-5            < 1e-3
    rms of the mean heating       0.0107 .. 0.0528    < 0.1 K/day
    largest per-level std         6.9 .. 7.4          < 10 K/day
    model-top T (days 40-80 min)  160.06 .. 160.13    > 155 K
    TOA net (SW dn - SW up - OLR) 37.3 .. 41.0        30 .. 50 W/m²
    TOA SW albedo                 0.469 .. 0.476      0.44 .. 0.50
    lowest-level cover (mean)     0                   < 0.01
    ============================  ==================  =====================

    The near-surface state (air 1.1 K below the SST, 19.7-20.0 g/kg) is
    pinned to a band around the measurement, ``297.5 .. 300.0`` K and
    ``18 .. 22`` g/kg. The largest per-level scatter of the heating is in the
    mixed-phase deck at 495 hPa (263 K).

    **What the column is, and is not.** It is in balance, though still
    adjusting in days 40-80: P/E is 0.96 there, 0.99 in days 80-120 and
    0.998-1.001 afterwards, as P and E fall together from 1.15 and 1.20 mm/d to
    1.04, the water drift goes 0.048, 0.011, -0.001, 0.003 mm/d over the four
    windows of the 200-day run, the TOA net is 38.4, 39.0, 38.3, 41.0 W/m² and
    the lowest level stays clear. It is overcast: the maximum-random total
    cloud cover is 1.0 in every step, from a deck between 237 and 626 hPa
    whose layer covers are 0.97-1.0 at 237-302 and 375-538 hPa, 0.83 at 337
    hPa (falling to 0.52 by day 160), 0.92 at 581 and 0.66 at 626 hPa (days
    40-80; a layer at 208 hPa, cover 0.54, appears in days 160-200). The 1 Pa
    top layer cools from its 200 K start and holds 160.1 K from about day 40,
    at the lower edge of RRTMGP's temperature tables, where jax-rrtmgp 0.5.0
    extends them linearly as ECHAM's RRTMG does; it is a property of a model
    top without dynamical or sponge heating and is pinned only to stay finite
    and above 155 K.

    The column water budget IS pinned: every term's ledger is conservative in
    the host's own layer mass except Tiedtke's, which creates the water its
    precipitation-flux floor removes (ECHAM behaviour, #912) and publishes it
    as ``convection.precip_floor_source``, so over the averaging window the
    change of column water equals evaporation minus precipitation plus that
    source.
    """

    @classmethod
    def setUpClass(cls):
        from jcm.physics.echam.echam_terms import echam_physics

        nlev = 47
        # Every bound below was measured with ECHAM's own T63 cover constants,
        # so the testbed states them rather than reading the shipped defaults,
        # which are jcm's calibrated set (jcm/physics/clouds/
        # echam_cloud_defaults.py). With the calibrated set the same column
        # holds cloud at its lowest level (time-mean cover 0.275 against the
        # < 0.01 pinned) and rains 0.926 of what it evaporates against the
        # > 0.93 pinned (docs/source/design/rce_testbed.md).
        physics = echam_physics(
            radiation=RadiationParameters.default(solar_constant=420.0),
            clouds=CloudParameters.default(
                truncation=63, crt=0.75, crs=0.975, nex=2.0, csatsc=0.7,
                cinv=0.25),
            radiation_compute_cre=False,
        ).replace("aerosol", AerosolFree())
        scm = rce_column(
            sst=300.0, relative_humidity=0.7, lat_deg=0.0, nlev=nlev,
            dt_seconds=900.0, physics=physics, interactive_humidity=True,
        )
        # Surface evaporation (the moisture source) needs a non-zero near-surface
        # wind; rce_initial_state seeds zero wind, so set a light background flow.
        ic = rce_initial_state(scm.vertical, sst=300.0, relative_humidity=0.7).copy(
            u_wind=jnp.full(nlev, 5.0),
        )
        # 80 days = 40-day spin-up + the 40-day averaging window below (about
        # 90 s on CPU). The column is still adjusting in days 40-80 (P/E 0.96
        # against 1.00 by day 120); the pins hold the extreme over four windows
        # of the 200-day run plus a margin (class docstring).
        preds = run_rce(scm, ic, n_days=80.0)

        cls.spd = int(round(86400.0 / 900.0))
        cls.window = slice(-40 * cls.spd, None)
        cls.T = np.asarray(preds.relaxed_states["temperature"])
        cls.q = np.asarray(preds.relaxed_states["specific_humidity"])
        cls.tot = np.asarray(preds.tendencies.temperature) * 86400.0  # K/day

        # The vertical axis of every series is the physics frame's: pick the
        # surface and the model top by pressure, not by array position.
        vertical = scm.coords.vertical
        ps = float(np.asarray(ic.normalized_surface_pressure) * c.p0)
        pfull = np.asarray(_pressure_centers(vertical, jnp.asarray(ps)))
        cls.i_surface = int(np.argmax(pfull))
        cls.i_top = int(np.argmin(pfull))

        def column(x):
            return np.asarray(x).reshape(len(preds.times), -1)[:, 0]

        def profile(x):
            return np.asarray(x).reshape(len(preds.times), nlev, -1)[:, :, 0]

        cls.precip_conv = column(preds.physics_data["convection"].precip_conv)
        rain = column(preds.physics_data["clouds"].precip_rain)
        snow = column(preds.physics_data["clouds"].precip_snow)
        cls.total_precip = cls.precip_conv + rain + snow
        cls.evap = column(preds.physics_data["surface"].effective_evaporation)
        cls.floor_source = column(preds.physics_data["convection"].precip_floor_source)
        cls.convecting = column(preds.physics_data["convection"].ktype) > 0
        cls.cloud_fraction = profile(preds.physics_data["clouds"].cloud_fraction)
        radiation = preds.physics_data["radiation"]
        cls.sw_down = column(radiation.toa_sw_down)
        cls.sw_up = column(radiation.toa_sw_up)
        cls.olr = column(radiation.toa_lw_up)

        mass = np.diff(np.asarray(vertical.a_boundaries)
                       + np.asarray(vertical.b_boundaries) * ps) / c.grav
        qc = np.asarray(preds.tracer_states["qc"])
        qi = np.asarray(preds.tracer_states["qi"])
        cls.column_water = ((cls.q + qc + qi) * mass).sum(axis=-1)

    def test_whole_model_column_reaches_physical_time_mean_rce(self):
        T, q, tot, spd, window = self.T, self.q, self.tot, self.spd, self.window

        # Stays finite over the whole integration (no blow-up).
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertTrue(np.all(np.isfinite(q)))

        # The 1 Pa top layer sits at the lower edge of RRTMGP's temperature
        # tables, finite and well above the runaway the tables' mirroring
        # caused before jax-rrtmgp 0.5.0 (a layer below 160 K cooling faster).
        self.assertGreater(float(T[window][:, self.i_top].min()), 155.0)

        # Time-mean radiative-convective equilibrium over the last 40 days: the
        # mean total heating tends to zero even though instantaneous physics
        # fluctuates.
        mean_tend = tot[window].mean(axis=0)
        self.assertLess(float(np.sqrt(np.mean(mean_tend ** 2))), 0.1)  # K/day

        # Near-surface state: the air a degree or so below the SST (an upward
        # sensible heat flux) at about 80 % relative humidity.
        t_surface = float(T[window][:, self.i_surface].mean())
        q_surface = float(q[window][:, self.i_surface].mean()) * 1e3
        self.assertGreater(t_surface, 297.5)
        self.assertLess(t_surface, 300.0)
        self.assertGreater(q_surface, 18.0)  # g/kg
        self.assertLess(q_surface, 22.0)

        # The precipitation is finite and the column precipitates.
        precip, total = self.precip_conv, self.total_precip
        self.assertTrue(np.all(np.isfinite(precip)))
        self.assertTrue(np.all(np.isfinite(total)))
        self.assertGreater(float(total[window].mean()), 0.0)

        # The high-frequency flicker is bounded: the largest per-level
        # temporal standard deviation of the total heating over the window.
        # It sits in the mixed-phase cloud deck at 495 hPa (263 K), whose
        # heating scatters by 6.9-7.4 K/day between trajectories and windows.
        max_temporal_std = float(np.max(tot[window].std(axis=0)))
        self.assertLess(max_temporal_std, 10.0)  # K/day

        # Column water budget over the window: Δ(column water)/Δt =
        # E − P + (the Tiedtke floor source, #912) on the host's own layer
        # mass. Measured residual below 3.5e-5 of E, which is float32
        # round-off on a sum of ~1 mm/d terms.
        evap, water = self.evap, self.column_water
        dwater_dt = (water[-1] - water[-40 * spd - 1]) / (40 * spd * 900.0)
        residual = float(evap[window].mean() - total[window].mean()
                         + self.floor_source[window].mean() - dwater_dt)
        self.assertLess(abs(residual), 1e-3 * float(evap[window].mean()))

    def test_precipitation_substantially_balances_evaporation(self):
        # The RCE balance this testbed exists to pin: over the window the
        # column rains what it evaporates (P/E 0.958-1.001 across trajectories
        # and windows), its water is steady, and its convection is alive.
        window, spd = self.window, self.spd
        self.assertGreater(float(self.total_precip[window].mean()),
                           0.93 * float(self.evap[window].mean()))

        # Steady column water: the drift over the window is the difference of
        # the two ends of a 40-day series, in mm/d (kg/m²/s × 86400).
        water = self.column_water
        drift = (water[-1] - water[-40 * spd - 1]) / (40 * spd * 900.0) * 86400.0
        self.assertLess(abs(float(drift)), 0.1)

        # Convection is alive: Tiedtke triggers in 0.978-0.984 of the steps of
        # every measured window, and its time-mean precipitation is 0.68-0.79
        # mm/d, about 0.68 of the total. The extinction this guards against drives
        # the equilibrium convective precipitation to exactly zero, which the
        # strict positivity catches; the step fraction catches the column
        # that convects only occasionally.
        self.assertGreater(float(self.precip_conv[window].mean()), 0.0)
        self.assertGreater(float(self.convecting[window].mean()), 0.90)

    def test_toa_balance_and_lowest_level_stays_clear(self):
        window = self.window

        # Top-of-atmosphere balance. In a fixed-SST RCE the net flux need not
        # vanish: it is the implied ocean heat flux, +37 to +41 W/m² here.
        sw_down = float(self.sw_down[window].mean())
        sw_up = float(self.sw_up[window].mean())
        toa_net = sw_down - sw_up - float(self.olr[window].mean())
        self.assertGreater(toa_net, 30.0)
        self.assertLess(toa_net, 50.0)
        albedo = sw_up / sw_down
        self.assertGreater(albedo, 0.44)
        self.assertLess(albedo, 0.50)

        # No fog: the lowest model level holds no cloud cover in the time
        # mean (0 across trajectories and windows). A fogged level has a
        # mean cover of 1.
        lowest_cover = float(self.cloud_fraction[window][:, self.i_surface].mean())
        self.assertLess(lowest_cover, 0.01)
