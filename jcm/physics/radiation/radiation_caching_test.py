"""Tests for the shared radiation sub-stepping cache.

Radiation is solved every ``radiation_interval`` but applied every step. A
solve is stored as shortwave TRANSMISSIVITIES (ECHAM ``trsol``/``trsof``) at a
floored, interval-midpoint zenith, and every step rebuilds the shortwave as
``trans * S(t) * max(mu0(t), 0)``. These cover that contract itself; the
per-scheme wiring is exercised by each scheme's own integration tests.
"""
import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax_solar import OrbitalTime, direct_solar_irradiance, get_solar_sin_altitude

from jcm.forcing import SolarGeometry
from jcm.physics.radiation import (
    _CACHED_SW_FIELDS,
    SURFACE_OPTICS_KEY,
    apply_insolation,
    cached_radiation_tendency,
    echam_solve_cos_zenith,
    hold_land_albedo,
    solve_solar_geometry,
    surface_optics_bands_for_solve,
    surface_optics_for_solve,
    sw_solve_to_transmissivity,
)
from jcm.physics.radiation.radiation_types import RadiationData, RadiationParameters
from jcm.physics.surface.echam.surface_types import SurfaceData

NCOLS, NLEV = 4, 3
S0 = 1361.0


def _solar(synodic_phase, orbital_phase=0.3):
    z = jnp.zeros((), jnp.float32)
    return SolarGeometry(
        tyear=z + orbital_phase / (2 * np.pi),
        orbital_phase=z + orbital_phase,
        synodic_phase=jnp.asarray(synodic_phase, jnp.float32),
    )


def _solved(inc):
    """Build the RadiationData a solve with incoming beam ``inc`` would leave it.

    Every shortwave quantity is a fixed fraction of the beam (distinct per
    field, so a swapped field is caught) and the transmissivities are derived
    by the production function.
    """
    inc = jnp.asarray(inc, dtype=jnp.float32)
    rad = RadiationData.zeros((NCOLS,), NLEV)
    frac = {n: 0.1 + 0.05 * i for i, n in enumerate(_CACHED_SW_FIELDS)}
    sw = {n: getattr(rad, n) + frac[n] * inc for n in _CACHED_SW_FIELDS}
    rad = rad.copy(
        toa_sw_down=inc,
        lw_flux_up=jnp.full((NLEV + 1, NCOLS), 240.0),
        lw_heating_rate=jnp.full((NLEV, NCOLS), -1.5e-5),
        noa_frac_toa_sw_up=jnp.full((NCOLS,), 0.02),
        **sw,
    )
    return sw_solve_to_transmissivity(rad), frac


def _now(solar, lon, lat):
    """Return the independent expectation S(t) * max(mu0(t), 0), straight from jax_solar."""
    ot = OrbitalTime(solar.orbital_phase, solar.synodic_phase)
    mu = np.maximum(np.asarray(get_solar_sin_altitude(ot, lon, lat)), 0.0)
    return float(direct_solar_irradiance(solar.orbital_phase, S0)) * mu


LON = jnp.asarray([0.0, 90.0, 180.0, 270.0])
LAT = jnp.asarray([0.0, 30.0, -30.0, 60.0])


class TransmissivityContractTest(unittest.TestCase):
    def test_every_step_rebuilds_shortwave_from_trans_and_the_sun_now(self):
        rad, frac = _solved([300.0, 600.0, 900.0, 1200.0])
        solar = _solar(1.1)
        out = apply_insolation(rad, solar, LON, LAT, S0)
        inc = _now(solar, LON, LAT)
        np.testing.assert_allclose(np.asarray(out.toa_sw_down), inc, rtol=1e-5)
        for name in _CACHED_SW_FIELDS:
            trans = np.asarray(getattr(rad, f"{name}_trans"))
            np.testing.assert_allclose(
                np.asarray(getattr(out, name)), trans * inc,
                rtol=1e-5, atol=1e-6, err_msg=name)

    def test_the_solve_sun_reproduces_the_solved_fluxes(self):
        # Applying the sun the solve saw returns the solve's own fluxes: the
        # transmissivity is exactly flux / beam.
        solar = _solar(1.1)
        inc = _now(solar, LON, LAT)
        rad, frac = _solved(jnp.maximum(inc, 1.0))
        out = apply_insolation(rad, solar, LON, LAT, S0)
        lit = inc > 1.0
        for name in _CACHED_SW_FIELDS:
            np.testing.assert_allclose(
                np.asarray(getattr(out, name))[..., lit],
                np.asarray(getattr(rad, name))[..., lit],
                rtol=1e-5, err_msg=name)

    def test_night_is_exactly_zero_and_never_negative(self):
        rad, _ = _solved([300.0] * NCOLS)
        solar = _solar(0.0)                 # midnight at lon 0, noon at 180
        out = apply_insolation(rad, solar, LON, LAT, S0)
        inc = np.asarray(out.toa_sw_down)
        dark = inc == 0.0
        self.assertTrue(dark.any() and (~dark).any())
        for name in _CACHED_SW_FIELDS:
            got = np.asarray(getattr(out, name))
            self.assertTrue(np.all(got >= 0.0), msg=name)
            self.assertTrue(np.all(got[..., dark] == 0.0), msg=name)

    def test_a_column_dark_at_the_solve_is_sunlit_later(self):
        # The solve is made at the floored zenith, so a column that is dark
        # when radiation runs carries a finite transmissivity and receives
        # shortwave as soon as the sun reaches it.
        mu_solve = echam_solve_cos_zenith(jnp.zeros(NCOLS))
        np.testing.assert_allclose(np.asarray(mu_solve), 0.1)
        beam = S0 * mu_solve
        rad, frac = _solved(beam)
        for name in _CACHED_SW_FIELDS:
            self.assertTrue(
                bool(jnp.all(getattr(rad, f"{name}_trans") > 0.0)), msg=name)
        out = apply_insolation(rad, _solar(jnp.pi), LON, LAT, S0)   # sun up at lon 0
        lit = np.asarray(out.toa_sw_down) > 0
        self.assertTrue(lit.any())
        self.assertTrue(np.all(np.asarray(out.surface_sw_down)[lit] > 0.0))
        self.assertTrue(np.all(np.asarray(out.sw_heating_rate)[:, lit] > 0.0))

    def test_applying_does_not_compound(self):
        # The transmissivity is never overwritten, so repeating a step, or
        # marching the sun and returning, cannot drift the result.
        rad, _ = _solved([500.0] * NCOLS)
        solar = _solar(2.0)
        first = apply_insolation(rad, solar, LON, LAT, S0)
        cur = rad
        for ph in (0.3, 4.0, 5.5, 1.0, 2.0):
            cur = apply_insolation(cur, _solar(ph), LON, LAT, S0)
        again = apply_insolation(cur, solar, LON, LAT, S0)
        for name in _CACHED_SW_FIELDS + ("toa_sw_down",):
            np.testing.assert_allclose(
                np.asarray(getattr(again, name)),
                np.asarray(getattr(first, name)), rtol=1e-6, err_msg=name)

    def test_longwave_and_aerosol_free_fractions_are_untouched(self):
        rad, _ = _solved([500.0] * NCOLS)
        out = apply_insolation(rad, _solar(2.0), LON, LAT, S0)
        np.testing.assert_array_equal(out.lw_flux_up, rad.lw_flux_up)
        np.testing.assert_array_equal(out.lw_heating_rate, rad.lw_heating_rate)
        np.testing.assert_array_equal(
            out.noa_frac_toa_sw_up, rad.noa_frac_toa_sw_up)

    def test_cos_zenith_is_the_current_clipped_sun(self):
        rad, _ = _solved([500.0] * NCOLS)
        solar = _solar(2.0)
        out = apply_insolation(rad, solar, LON, LAT, S0)
        ot = OrbitalTime(solar.orbital_phase, solar.synodic_phase)
        np.testing.assert_allclose(
            np.asarray(out.cos_zenith),
            np.maximum(np.asarray(get_solar_sin_altitude(ot, LON, LAT)), 0.0),
            atol=1e-6)

    def test_heating_reaches_the_replayed_tendency(self):
        rad, _ = _solved([500.0] * NCOLS)
        out = apply_insolation(rad, _solar(2.0), LON, LAT, S0)
        tend = cached_radiation_tendency(out, (NLEV, NCOLS))
        np.testing.assert_allclose(
            np.asarray(tend.temperature),
            np.asarray(out.sw_heating_rate + out.lw_heating_rate), rtol=1e-6)

    def test_transmissivity_gradient_is_finite_including_a_zero_beam(self):
        rad, _ = _solved([500.0, 0.0, 300.0, 0.0])

        def f(inc):
            r, _ = _solved(inc)
            return jnp.sum(apply_insolation(r, _solar(2.0), LON, LAT, S0)
                           .surface_sw_down)

        g = jax.grad(f)(jnp.asarray([500.0, 0.0, 300.0, 0.0]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


class CachedFieldCoverageTest(unittest.TestCase):
    def test_every_shortwave_field_is_cached_as_a_transmissivity(self):
        # A shortwave field left off the cache list would keep a stale value
        # between solves. Every SW flux/heating field of RadiationData is
        # rebuilt from a ``_trans`` partner, except ``toa_sw_down``, which IS
        # the incoming beam; the ``noa_frac_*`` ratios are sun-independent.
        names = {f.name for f in dataclasses.fields(RadiationData)}
        sw_flux_like = {
            n for n in names
            if ("sw" in n.split("_")) and not n.startswith("noa_frac")
            and not n.endswith("_trans")
        } - {"toa_sw_down"}
        self.assertEqual(sw_flux_like, set(_CACHED_SW_FIELDS))
        for n in _CACHED_SW_FIELDS:
            self.assertIn(f"{n}_trans", names)

    def test_trans_fields_are_excluded_from_output(self):
        from jcm.physics.composable_physics import ComposablePhysics
        for n in _CACHED_SW_FIELDS:
            self.assertIn(
                f"radiation.{n}_trans", ComposablePhysics._EXCLUDED_OUTPUT_KEYS)


class SolveZenithTest(unittest.TestCase):
    def test_floor_and_curvature_match_echam(self):
        # mo_radiation.f90::pre_radiation: rae/(sqrt(mu^2 + rae(rae+2)) - mu),
        # then MAX(., 0.1).
        mu = np.array([-0.5, 0.0, 0.05, 0.1, 0.5, 1.0])
        rae = 0.1277e-2
        curved = rae / (np.sqrt(mu**2 + rae * (rae + 2)) - mu)
        np.testing.assert_allclose(
            np.asarray(echam_solve_cos_zenith(jnp.asarray(mu))),
            np.maximum(curved, 0.1), rtol=1e-5)
        self.assertTrue(np.all(np.asarray(echam_solve_cos_zenith(
            jnp.asarray(mu))) >= 0.1))

    def test_gradient_is_finite_everywhere(self):
        g = jax.vmap(jax.grad(echam_solve_cos_zenith))(
            jnp.linspace(-1.0, 1.0, 21))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_solve_is_at_the_interval_midpoint(self):
        # ECHAM radiation_time: 0.5 * (trigrad - delta_time) after the step.
        params = RadiationParameters.default(radiation_interval=7200.0)
        solar = _solar(1.0)
        out = solve_solar_geometry(solar, 720.0, params)
        shift = 0.5 * (7200.0 - 720.0)
        np.testing.assert_allclose(
            float(out.synodic_phase - solar.synodic_phase),
            2 * np.pi * shift / 86400.0, rtol=1e-4)
        every = solve_solar_geometry(
            solar, 720.0, RadiationParameters.default(radiation_interval=0.0))
        np.testing.assert_allclose(
            float(every.synodic_phase), float(solar.synodic_phase), rtol=1e-6)


class MonthlyInsolationTest(unittest.TestCase):
    """31 days at T63, the production solve cadence: the mean insolation is exact.

    Drives the real insolation path (``solve_solar_geometry``,
    ``echam_solve_cos_zenith``, ``sw_solve_to_transmissivity``,
    ``apply_insolation``) on the T63 Gaussian grid with 12-minute steps and a
    2-hour solve, with a stand-in column physics (the shortwave a fixed 0.7 of
    the beam it saw). The time mean of ``toa_sw_down`` must be the exact
    insolation integral, with no longitude structure from the solve cadence.
    """

    NLON, NLAT, DT, EVERY, DAYS = 192, 96, 720.0, 10, 31

    def test_monthly_mean_matches_the_exact_integral(self):
        lon = np.arange(self.NLON) * 360.0 / self.NLON
        x, w = np.polynomial.legendre.leggauss(self.NLAT)
        lat = np.degrees(np.arcsin(x))
        w = w / w.sum()
        LON2, LAT2 = np.meshgrid(lon, lat)
        lons = jnp.asarray(LON2.ravel(), jnp.float32)
        lats = jnp.asarray(LAT2.ravel(), jnp.float32)
        ncols = lons.size
        params = RadiationParameters.default(
            radiation_interval=self.EVERY * self.DT)
        year = 366.0 * 86400.0

        def solar_at(t):
            return SolarGeometry(
                tyear=t / year, orbital_phase=2 * jnp.pi * t / year,
                synodic_phase=2 * jnp.pi * jnp.mod(t, 86400.0) / 86400.0)

        @jax.jit
        def solve_step(t):
            sol = solve_solar_geometry(solar_at(t), self.DT, params)
            ot = OrbitalTime(sol.orbital_phase, sol.synodic_phase)
            mu = echam_solve_cos_zenith(get_solar_sin_altitude(ot, lons, lats))
            beam = direct_solar_irradiance(sol.orbital_phase, S0) * mu
            rad = RadiationData.zeros((ncols,), 1).copy(
                toa_sw_down=beam, surface_sw_down=0.7 * beam)
            return sw_solve_to_transmissivity(rad)

        @jax.jit
        def apply_step(rad, t):
            return apply_insolation(rad, solar_at(t), lons, lats, S0)

        @jax.jit
        def exact(t):
            ot = OrbitalTime(2 * jnp.pi * t / year,
                             2 * jnp.pi * jnp.mod(t, 86400.0) / 86400.0)
            return direct_solar_irradiance(ot.orbital_phase, S0) * jnp.maximum(
                get_solar_sin_altitude(ot, lons, lats), 0.0)

        nstep = int(self.DAYS * 86400 / self.DT)
        toa = np.zeros(ncols)
        sfc = np.zeros(ncols)
        rad = RadiationData.zeros((ncols,), 1)
        for k in range(nstep):
            t = jnp.float32(k * self.DT)
            if k % self.EVERY == 0:
                rad = solve_step(t).copy(
                    sw_flux_up=rad.sw_flux_up, sw_flux_down=rad.sw_flux_down,
                    sw_heating_rate=rad.sw_heating_rate)
            rad = apply_step(rad, t)
            toa += np.asarray(rad.toa_sw_down)
            sfc += np.asarray(rad.surface_sw_down)
        toa /= nstep
        sfc /= nstep
        # Reference: the continuous integral, sampled every 2 minutes.
        ref = np.zeros(ncols)
        fine = np.arange(0.0, self.DAYS * 86400.0, 120.0)
        for t in fine:
            ref += np.asarray(exact(jnp.float32(t)))
        ref /= fine.size

        gm = lambda f: float((f.reshape(self.NLAT, self.NLON) * w[:, None]).sum()
                             / self.NLON)
        self.assertLess(abs(gm(toa) - gm(ref)), 0.1)
        self.assertLess(abs(gm(sfc) - 0.7 * gm(ref)), 0.07)
        eq = np.abs(lat) < 10.0
        band = (toa - ref).reshape(self.NLAT, self.NLON)[eq].mean(0)
        self.assertLess(float(band.std()), 0.05)
        self.assertLess(float(np.abs(band).max()), 0.2)


class HeldLandAlbedoTest(unittest.TestCase):
    """``hold_land_albedo``: the land albedo of the last solve rides the surface carry (#979)."""

    def _diagnostics(self, held, current):
        return {
            "surface": SurfaceData.zeros((NCOLS,), NLEV).copy(
                land_albedo_at_solve=jnp.full((NCOLS,), held)),
            SURFACE_OPTICS_KEY: {"land_albedo": jnp.full((NCOLS,), current)},
        }

    def test_a_solve_takes_this_steps_albedo(self):
        out = hold_land_albedo(self._diagnostics(0.7, 0.4), jnp.asarray(True))
        np.testing.assert_allclose(out["surface"].land_albedo_at_solve, 0.4)

    def test_a_replay_keeps_the_solves_albedo(self):
        out = hold_land_albedo(self._diagnostics(0.7, 0.4), jnp.asarray(False))
        np.testing.assert_allclose(out["surface"].land_albedo_at_solve, 0.7)

    def test_nothing_to_hold_is_a_no_op(self):
        no_optics = {"surface": SurfaceData.zeros((NCOLS,), NLEV)}
        self.assertIs(hold_land_albedo(no_optics, jnp.asarray(True)), no_optics)
        no_land = {**self._diagnostics(0.7, 0.4), SURFACE_OPTICS_KEY: {"albedo_vis": 0.1}}
        self.assertIs(hold_land_albedo(no_land, jnp.asarray(True)), no_land)
        self.assertEqual(hold_land_albedo({}, jnp.asarray(True)), {})


class SurfaceOpticsBandsForSolveTest(unittest.TestCase):
    """jax-gcm#959: the per-band direct/diffuse quartet a solve should use."""

    def test_quartet_present_on_surface_optics_key_is_returned_directly(self):
        diagnostics = {
            SURFACE_OPTICS_KEY: {
                "albedo_vis_dir": jnp.full((NCOLS,), 0.3),
                "albedo_vis_dif": jnp.full((NCOLS,), 0.07),
                "albedo_nir_dir": jnp.full((NCOLS,), 0.25),
                "albedo_nir_dif": jnp.full((NCOLS,), 0.07),
            },
        }
        vis_dir, vis_dif, nir_dir, nir_dif = surface_optics_bands_for_solve(
            diagnostics, NCOLS)
        np.testing.assert_allclose(vis_dir, 0.3)
        np.testing.assert_allclose(vis_dif, 0.07)
        np.testing.assert_allclose(nir_dir, 0.25)
        np.testing.assert_allclose(nir_dif, 0.07)

    def test_no_quartet_falls_back_to_the_broadband_blend_not_to_zero(self):
        """Regression guard: ``RadiationData`` always carries the quartet
        LEAVES (zero-initialised like every other field), so a carry that
        sets only the broadband albedo via ``.copy(surface_albedo_vis=...)``
        must NOT read back as an all-zero direct/diffuse albedo -- it must
        fall through to the BLENDED broadband value (``0.46*vis+0.54*nir``),
        not the raw vis/nir pair (a Codex-review catch: with vis != nir,
        feeding the raw pair through the per-band split gives a per-band
        albedo no caller predating #959 asked for).
        """
        diagnostics = {
            "radiation": RadiationData.zeros((NCOLS,), NLEV).copy(
                surface_albedo_vis=jnp.full((NCOLS,), 0.3),
                surface_albedo_nir=jnp.full((NCOLS,), 0.2),
            ),
        }
        vis, nir, _ = surface_optics_for_solve(diagnostics, NCOLS)
        np.testing.assert_allclose(vis, 0.3)
        np.testing.assert_allclose(nir, 0.2)

        vis_dir, vis_dif, nir_dir, nir_dif = surface_optics_bands_for_solve(
            diagnostics, NCOLS)
        broadband = 0.46 * 0.3 + 0.54 * 0.2
        np.testing.assert_allclose(vis_dir, broadband)
        np.testing.assert_allclose(vis_dif, broadband)
        np.testing.assert_allclose(nir_dir, broadband)
        np.testing.assert_allclose(nir_dif, broadband)

    def test_surface_optics_key_without_a_quartet_also_falls_back(self):
        """A boundary-condition term predating #959 publishes only the
        broadband keys on ``SURFACE_OPTICS_KEY``; the bands helper must
        still fall back to their BLEND (not the raw pair) rather than
        KeyError or read a stale carry.
        """
        diagnostics = {
            SURFACE_OPTICS_KEY: {
                "albedo_vis": jnp.full((NCOLS,), 0.4),
                "albedo_nir": jnp.full((NCOLS,), 0.1),
                "emissivity": jnp.full((NCOLS,), 0.98),
            },
        }
        vis_dir, vis_dif, nir_dir, nir_dif = surface_optics_bands_for_solve(
            diagnostics, NCOLS)
        broadband = 0.46 * 0.4 + 0.54 * 0.1
        np.testing.assert_allclose(vis_dir, broadband)
        np.testing.assert_allclose(vis_dif, broadband)
        np.testing.assert_allclose(nir_dir, broadband)
        np.testing.assert_allclose(nir_dif, broadband)


if __name__ == "__main__":
    unittest.main()