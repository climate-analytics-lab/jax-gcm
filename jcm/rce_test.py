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
from jcm.physics.clouds.sundqvist import saturation_specific_humidity
from jcm.rce import (
    _STRATOSPHERE_Q_FLOOR,
    _pressure_centers,
    fixed_rh_closure,
    rce_column,
    rce_initial_state,
    rce_physics,
    run_rce,
    steady_insolation,
)
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
        qsat = saturation_specific_humidity(pfull, self.ic.temperature)
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
    """RCE on the *full* ECHAM physics stack with Tiedtke convection.

    Unlike the minimal radiative-convective ``rce_physics`` stack, this drives
    the complete ``echam_physics()`` column — surface turbulent fluxes, TTE-TKE
    vertical diffusion, 1-moment microphysics, Sundqvist clouds, Tiedtke-Nordeng
    convection and radiation — as a genuine single-column integration of the
    whole model. Humidity is prognostic (the surface evaporation supplies it; the
    fixed-RH closure is incompatible with the model's own moisture physics).

    The assertions are on the **time mean**: a single-column mass-flux scheme in
    RCE has an intrinsic high-frequency convective cycle (a residual cloud-base
    flicker remains — fully removing it needs the half-level flux re-stagger of
    ``cuasc``/``cudtdq``, tracked separately), but the time-mean column must be a
    physical radiative-convective equilibrium with continuously active
    convection. This is the regression guard for the closure fix that anchors the
    cloud-base mass flux to the surface moisture supply (ECHAM ``zmfub``) so
    convection runs continuously instead of switching fully on/off.
    """

    def test_whole_model_column_reaches_physical_time_mean_rce(self):
        from jcm.physics.echam.echam_terms import echam_physics

        nlev = 47
        physics = echam_physics(
            radiation_scheme="grey",
            radiation=RadiationParameters.default(solar_constant=420.0),
        )
        scm = rce_column(
            sst=300.0, relative_humidity=0.7, lat_deg=0.0, nlev=nlev,
            dt_seconds=900.0, physics=physics, interactive_humidity=True,
        )
        # Surface evaporation (the moisture source) needs a non-zero near-surface
        # wind; rce_initial_state seeds zero wind, so set a light background flow.
        ic = rce_initial_state(scm.vertical, sst=300.0, relative_humidity=0.7).copy(
            u_wind=jnp.full(nlev, 5.0),
        )
        # 80 days = 40-day spin-up + the 40-day averaging window below.
        # Measured convergence of the windowed rms mean tendency (K/day):
        # days 10-50: 0.243, 20-50: 0.223, 30-60: 0.143, 40-80: 0.099 —
        # the column approaches its time-mean equilibrium slowly, so
        # shortening the spin-up materially erodes the 0.2 K/day margin.
        # Keep the full 80 days (~73 s wall-clock; the margin is worth it).
        preds = run_rce(scm, ic, n_days=80.0)

        T = np.asarray(preds.relaxed_states["temperature"])
        q = np.asarray(preds.relaxed_states["specific_humidity"])
        tot = np.asarray(preds.tendencies.temperature) * 86400.0  # K/day

        # Stays finite over the whole integration (no blow-up).
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertTrue(np.all(np.isfinite(q)))

        # Time-mean radiative-convective equilibrium over the last 40 days: the
        # mean total heating tends to zero even though instantaneous convection
        # fluctuates.
        spd = int(round(86400.0 / 900.0))
        mean_tend = tot[-40 * spd:].mean(axis=0)
        self.assertLess(float(np.sqrt(np.mean(mean_tend ** 2))), 0.2)  # K/day

        # Physical near-surface state: air temperature within a few K of the SST,
        # and a realistic surface specific humidity (several to ~25 g/kg).
        self.assertGreater(float(T[-1, -1]), 294.0)
        self.assertLess(float(T[-1, -1]), 306.0)
        self.assertGreater(float(q[-1, -1]) * 1e3, 5.0)
        self.assertLess(float(q[-1, -1]) * 1e3, 30.0)

        # Convection is genuinely (and near-continuously) active — the closure
        # fix keeps it on rather than flickering fully off. Time-mean convective
        # precip over the last 40 days is positive.
        precip = np.asarray(
            preds.physics_data["convection"].precip_conv
        ).reshape(len(preds.times), -1)[:, 0]
        self.assertGreater(float(precip[-40 * spd:].mean()), 0.0)

        # The high-frequency convective flicker is bounded. The cloud-base
        # mass-flux closure fix (anchoring to the surface moisture supply +
        # keeping convection active while it is supplied) roughly halves the
        # per-level temporal scatter of the total heating (≈14 → ≈7 K/day);
        # the bare-CAPE on/off closure exceeds this bound. (A residual flicker
        # remains pending the half-level flux re-stagger — see the class
        # docstring — so this is an upper bound, not a "smooth" assertion.)
        max_temporal_std = float(np.max(tot[-40 * spd:].std(axis=0)))
        self.assertLess(max_temporal_std, 10.0)  # K/day
