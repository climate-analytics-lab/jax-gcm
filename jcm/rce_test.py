"""Tests for ``jcm.rce`` — single-column radiative-convective equilibrium.

The fast tests cover the *machinery* — the fixed-RH humidity closure (kg/kg,
the canonical convention), the steady-insolation helper, and the physics-package
composition. The slow tests run short RCE integrations end-to-end (grey and
RRTMGP) and assert the column stays finite, approaches a steady atmospheric
state (``dT/dt → 0``; note the TOA flux need *not* vanish in a fixed-SST RCE —
the imbalance is the implied ocean heat flux), keeps a convectively bounded
lapse rate, and produces order-of-magnitude-reasonable OLR.
"""

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
from jcm.single_column_model import SingleColumnModel


class TestFixedRhClosure(unittest.TestCase):
    """The Manabe-Wetherald fixed-RH humidity closure."""

    def setUp(self):
        self.vertical = SigmaCoordinates.equidistant(20)
        # rce_initial_state builds a correctly top-first ordered column
        # (index 0 = model top, index -1 = surface).
        self.ic = rce_initial_state(self.vertical, sst=300.0, relative_humidity=0.7)

    def test_closure_sets_rh_qsat_in_kg_per_kg(self):
        """Closure output equals ``rh(σ)·qsat`` in kg/kg at the resolved levels."""
        rh = 0.7
        closure = fixed_rh_closure(rh, self.vertical)
        out = closure(self.ic, forcing=None)

        ps = float(self.ic.normalized_surface_pressure) * c.p0
        pfull = _pressure_centers(self.vertical, jnp.asarray(ps))
        sigma = pfull / ps
        rh_profile = rh * jnp.clip((sigma - 0.02) / 0.98, 0.0, 1.0)
        expected = jnp.maximum(
            rh_profile * saturation_specific_humidity(pfull, self.ic.temperature),
            _STRATOSPHERE_Q_FLOOR,
        )
        self.assertTrue(jnp.allclose(out.specific_humidity, expected, rtol=1e-5))
        # Surface (index -1) value is a realistic several g/kg expressed in kg/kg
        # (~0.005–0.03); a ~1000x reading would mean the closure emitted g/kg.
        self.assertGreater(float(out.specific_humidity[-1]), 0.005)
        self.assertLess(float(out.specific_humidity[-1]), 0.03)

    def test_stratosphere_is_dry_floor(self):
        """The Manabe-Wetherald taper drives the top to the trace floor (no NaN)."""
        closure = fixed_rh_closure(0.7, self.vertical)
        out = closure(self.ic, forcing=None)
        # Top (index 0) is dry: RH taper × tiny cold-point qsat falls below the
        # floor, so it clamps to the trace stratospheric value.
        self.assertAlmostEqual(
            float(out.specific_humidity[0]), _STRATOSPHERE_Q_FLOOR, places=9,
        )
        self.assertTrue(jnp.all(jnp.isfinite(out.specific_humidity)))
        self.assertTrue(jnp.all(out.specific_humidity >= _STRATOSPHERE_Q_FLOOR))

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
                         radiation_scheme="grey")
        self.assertIsInstance(scm, SingleColumnModel)
        self.assertEqual(scm.free_evolve, ("temperature",))
        self.assertIsNotNone(scm.state_closure)
        self.assertAlmostEqual(float(scm.forcing.sea_surface_temperature[0, 0]), 302.0)

    def test_betts_miller_params_from_rce_column(self):
        scm = rce_column(relative_humidity=0.65, tau_convection=5400.0,
                         vertical=SigmaCoordinates.equidistant(8), radiation_scheme="grey")
        bm = [t for t in scm.physics.terms if t.category == "convection"][0]
        params = bm.params.get_value()
        self.assertAlmostEqual(float(params.rhbm), 0.65, places=5)
        self.assertAlmostEqual(float(params.tau_bm), 5400.0, places=3)

    def test_unknown_radiation_scheme_raises(self):
        with self.assertRaises(ValueError):
            rce_column(radiation_scheme="nope", vertical=SigmaCoordinates.equidistant(8))

    def test_unknown_convection_raises(self):
        with self.assertRaises(ValueError):
            rce_column(radiation_scheme="grey", convection="nope",
                       vertical=SigmaCoordinates.equidistant(8))

    def test_interactive_humidity_frees_q_and_drops_closure(self):
        scm = rce_column(relative_humidity=0.7, vertical=SigmaCoordinates.equidistant(8),
                         radiation_scheme="grey", interactive_humidity=True)
        self.assertIn("specific_humidity", scm.free_evolve)
        self.assertIn("temperature", scm.free_evolve)
        self.assertIsNone(scm.state_closure)


@pytest.mark.slow
class TestRceIntegrationGrey(unittest.TestCase):
    """End-to-end grey-radiation RCE: finiteness, equilibration, lapse rate.

    Grey radiation on a cheap sigma grid keeps this fast; the RRTMGP Case-1
    configuration is exercised separately in :class:`TestRceIntegrationRrtmgp`.
    """

    def _run(self, sst=300.0, n_days=40.0):
        vertical = SigmaCoordinates.equidistant(20)
        scm = rce_column(
            sst=sst, relative_humidity=0.7, solar_constant=420.0, lat_deg=0.0,
            vertical=vertical, radiation_scheme="grey", dt_seconds=1800.0,
        )
        ic = rce_initial_state(vertical, sst=sst, relative_humidity=0.7)
        return scm, run_rce(scm, ic, n_days=n_days)

    def test_runs_finite_and_equilibrates(self):
        _, preds = self._run()
        T = preds.relaxed_states["temperature"]
        dT = preds.tendencies.temperature

        self.assertTrue(bool(jnp.all(jnp.isfinite(T[-1]))))

        rms0 = float(jnp.sqrt(jnp.mean(dT[0] ** 2)) * 86400.0)
        rms_end = float(jnp.sqrt(jnp.mean(dT[-1] ** 2)) * 86400.0)
        # The column should be settling: the final RMS heating rate is well
        # below the initial transient and small in absolute terms.
        self.assertLess(rms_end, 0.5 * rms0)
        self.assertLess(rms_end, 0.3)  # K/day

    def test_troposphere_physical_and_lapse_rate_bounded(self):
        _, preds = self._run()
        T = np.asarray(preds.relaxed_states["temperature"][-1])
        # Exclude the single thin top layer (grey radiation over-warms it —
        # a known artefact of the scheme at the model top, not a framework bug).
        trop = T[2:]
        self.assertTrue(np.all(trop > 150.0))
        self.assertTrue(np.all(trop < 360.0))

        # Lapse rate in the lower/mid troposphere must not exceed the dry
        # adiabat (~10 K/km) — i.e. convection has removed the super-adiabatic
        # layers the radiative-equilibrium profile would otherwise have.
        vertical = SigmaCoordinates.equidistant(20)
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
class TestRceIntegrationRrtmgp(unittest.TestCase):
    """RRTMGP + Betts-Miller fixed-RH RCE on echam-47 — issue #523 Case 1.

    With specific humidity in its canonical kg/kg units, RRTMGP runs cleanly and
    the column reaches a radiative-convective balance with an Earth-like OLR and
    a near-surface RH that matches the prescribed value.
    """

    def test_case1_equilibrates_with_reasonable_olr_and_rh(self):
        scm = rce_column(
            sst=300.0, relative_humidity=0.7, solar_constant=728.4, lat_deg=42.55,
            nlev=47, radiation_scheme="rrtmgp", dt_seconds=1200.0,
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
