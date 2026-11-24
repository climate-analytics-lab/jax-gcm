"""Tests for the pySES omega (pressure vertical velocity, Dp/Dt) provider.

The provider replicates pySES's own T-prognostic CAM-SE hydrostatic
continuity integral (``cam_se.explicit_terms.init_common_variables`` /
``eval_temperature_vertical_advection_term``), stopping one step short of
that function's ``/ (rho * cp)`` conversion to an adiabatic-heating
tendency. See ``PysesCamSEDycore._omega_from_gll`` for the formula and its
provenance (jax-gcm#698).
"""

import os
import unittest

os.environ.setdefault("PYSES_BACKEND", "jax")
os.environ.setdefault("PYSES_USE_CPU", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from jcm.dycore.pyses.dycore import PysesCamSEDycore
from jcm.dycore.pyses.coords import (
    DEFAULT_MODEL_TOP_FRACTION,
    full_echam_hybrid,
)
from jcm.dycore.pyses.pyses_dycore_test import T63_TERRAIN  # noqa: F401

import jcm.constants as c

pytestmark = pytest.mark.requires_extra("pyses")

# A purely meridional, zonally-symmetric flow v = V0*cos(lat), u = 0 has
# closed-form horizontal divergence div(V) = -2*V0*sin(lat)/a (a standard
# spherical-divergence identity) -- locally nonzero (a Hadley-type
# overturning pattern) but globally zero net divergence, as any smooth
# wind field on a closed sphere must be. With flat (horizontally uniform)
# surface pressure the pressure-advection term u.grad(p) vanishes and the
# divergence is level-independent, so the hydrostatic continuity integral
# collapses to the closed form checked below.
V0_SI = 30.0    # m/s
PS_SI = 97000.0  # Pa, flat


def _closed_form_omega(lat, p_mid, p_top, v0=V0_SI):
    """Exact continuum omega(lat, level) for the analytic flow above.

    omega = -integral_{p_top}^{p_mid} div(V) dp' + v.grad(p)
          = -(-2*v0*sin(lat)/a) * (p_mid - p_top) + 0
          =  (2*v0*sin(lat)/a) * (p_mid - p_top)

    Independent of any model's discretization -- this is the "continuity
    integral computed independently" the tests below check the provider
    against. ``lat`` and ``p_mid`` broadcast against each other.
    """
    a = float(c.rearth)
    return (2.0 * v0 * np.sin(lat) / a) * (p_mid - p_top)


def _dycore(nx=3, terrain_file=None, **kwargs):
    return PysesCamSEDycore(
        nx=nx, npt=4, dt_seconds=900.0, terrain_file=terrain_file,
        physics_dtype=jnp.float32, compute_omega=True, **kwargs)


class OmegaProviderTest(unittest.TestCase):
    """Declaration, the resting-state zero, and the analytic agreement checks."""

    @classmethod
    def setUpClass(cls):
        cls.dycore = _dycore(terrain_file=T63_TERRAIN)
        cls.state = cls.dycore.initial_state(None)

    def test_declaration_and_default_off(self):
        self.assertEqual(self.dycore.physics_field_names(), ("omega",))
        off = PysesCamSEDycore(nx=3, npt=4, dt_seconds=900.0,
                               terrain_file=T63_TERRAIN,
                               physics_dtype=jnp.float32)
        self.assertEqual(off.physics_field_names(), ())
        self.assertEqual(off.physics_fields(None, None), {})

    def test_resting_state_gives_exactly_zero(self):
        # USSA rest: u = v = 0, so every term of omega carries a wind
        # factor (the mass-flux divergence, or the u.grad(p) dot product)
        # and omega must be identically zero (not merely small).
        physics_state = self.dycore.to_physics_state(self.state)
        out = self.dycore.physics_fields(self.state, physics_state)
        self.assertEqual(set(out), {"omega"})
        omega = np.asarray(out["omega"])
        nlev = self.dycore.nlev
        self.assertEqual(omega.shape, (nlev, 1, self.dycore.colmap.num_cols))
        self.assertEqual(omega.dtype, np.float32)
        np.testing.assert_array_equal(omega, 0.0)

    @staticmethod
    def _uniform_pressure_profile(dycore, ps_si):
        """``(d_pressure_1d, p_mid_1d, p_top)`` from this core's own (a, b) tables.

        Flat (horizontally uniform) ``ps`` makes the moist layer pressure
        thickness -- and hence the mid-level pressure -- a pure function of
        level, shared by every GLL point.
        """
        a_pa, b = full_echam_hybrid(dycore.nlev, DEFAULT_MODEL_TOP_FRACTION)
        p_top = float(a_pa[0])
        dp_1d = np.diff(a_pa) + np.diff(b) * ps_si
        p_int_1d = np.concatenate([[p_top], p_top + np.cumsum(dp_1d)])
        p_mid_1d = 0.5 * (p_int_1d[:-1] + p_int_1d[1:])
        return dp_1d, p_mid_1d, p_top

    @staticmethod
    def _analytic_gll_fields(dycore, dp_1d):
        """GLL ``(d_pressure, wind)`` for the module's analytic divergent flow."""
        coords = np.asarray(dycore.h_grid["physical_coords"])
        lat = coords[..., 0]              # pySES convention: [..., 0] = lat
        nlev = dycore.nlev

        def bcast_level(x1d, dtype=jnp.float64):
            return jnp.asarray(
                np.broadcast_to(x1d, lat.shape + (nlev,)), dtype)

        d_pressure = bcast_level(dp_1d)
        v = V0_SI * np.cos(lat)
        zeros = jnp.zeros(lat.shape + (nlev,), jnp.float64)
        v_field = bcast_level(np.broadcast_to(v[..., None], lat.shape + (nlev,)))
        wind = jnp.stack([zeros, v_field], axis=-1)
        return lat, d_pressure, wind

    def test_analytic_divergent_flow_matches_continuity_integral(self):
        # Fresh, small, FLAT (terrain_file=None) dycore: flat ps keeps the
        # analytic closed form exact (no horizontal pressure-gradient term).
        dycore = _dycore(nx=8)
        dp_1d, p_mid_1d, p_top = self._uniform_pressure_profile(dycore, PS_SI)
        lat, d_pressure, wind = self._analytic_gll_fields(dycore, dp_1d)

        got = np.asarray(dycore._omega_from_gll(d_pressure, wind))

        want = _closed_form_omega(
            lat[..., None], p_mid_1d[None, None, None, :], p_top)
        want = np.broadcast_to(want, got.shape)

        scale = np.abs(want).max()
        self.assertGreater(scale, 0.0)
        # Unlike the frontogenesis provider's analytic test (a quadratic
        # function of a raw per-element gradient), the pressure-weighted
        # divergence here is summed (cumsum) over levels, which averages
        # out most of the discrete SE gradient's element-corner error:
        # measured at ne8 it is ~0.15% max / ~0.01% mean, two orders of
        # magnitude tighter than frontogenesis's ~8% max / ~1% mean. Keep
        # some headroom over the measured values rather than pin them
        # exactly.
        err_max = np.abs(got - want).max() / scale
        err_mean = np.abs(got - want).mean() / scale
        self.assertLess(err_max, 0.01, msg=f"max rel error {err_max:.3%}")
        self.assertLess(err_mean, 0.002, msg=f"mean rel error {err_mean:.3%}")

    def test_dinosaur_and_pyses_agree_on_a_shared_analytic_column(self):
        """Both backends' omega providers, on the same analytic flow.

        Dinosaur carries divergence as an independent prognostic spectral
        field (not reconstructed from u, v), so its own test
        (``jcm.dycore.physics_fields_test.DinosaurOmegaProviderTest``) can
        set a spatially uniform divergence directly. pySES carries u, v
        explicitly, so this test instead sets the SAME analytic flow this
        module uses elsewhere (v = V0*cos(lat), locally divergent but
        globally zero-net, as any smooth wind field on a closed sphere must
        be) and lets each backend's own discretized machinery diagnose
        omega from it independently, on the (near-)identical ECHAM/ICON L47
        hybrid table both backends build from
        (``jcm.physics.echam.echam_levels.get_echam_levels`` -- pySES's
        ``full_echam_hybrid`` only replaces the singular top interface).
        """
        from dinosaur.hybrid_coordinates import HybridCoordinates
        from dinosaur.primitive_equations import State
        from dinosaur.scales import units

        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        # --- pySES side ---
        pyses_dycore = _dycore(nx=8)
        dp_1d, p_mid_1d, p_top = self._uniform_pressure_profile(
            pyses_dycore, PS_SI)
        lat_gll, d_pressure, wind = self._analytic_gll_fields(
            pyses_dycore, dp_1d)
        omega_pyses_gll = np.asarray(
            pyses_dycore._omega_from_gll(d_pressure, wind))

        # Target a mid-troposphere level (away from the thin, finite-top
        # layers where pySES's ~1 Pa top-interface fix and dinosaur's exact
        # p_top=0 diverge) and a column at 45 N (away from the poles and
        # from the equator, where sin(lat)=0 makes the comparison trivial).
        level = pyses_dycore.nlev - 25     # counts from the top; mid-troposphere
        target_lat = np.radians(45.0)
        col_idx = np.argmin(np.abs(
            np.asarray(pyses_dycore.colmap.latitudes) - target_lat))
        omega_pyses_col = np.asarray(
            pyses_dycore.colmap.gather_3d(jnp.asarray(omega_pyses_gll))
        )[level, 0, col_idx]
        got_lat = float(pyses_dycore.colmap.latitudes[col_idx])

        # --- dinosaur side: the same analytic flow, same hybrid table ---
        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        dino_dycore = DinosaurDycore(
            coords=coords, terrain=TerrainData.aquaplanet(coords),
            dt_seconds=600.0, compute_omega=True)
        self.assertEqual(dino_dycore.physics_field_names(), ("omega",))

        hor = coords.horizontal
        nlev = coords.vertical.layers
        nlon, nlat = hor.nodal_shape
        lats = np.asarray(hor.latitudes)
        specs = dino_dycore.physics_specs
        divergence = -2.0 * V0_SI * np.sin(lats) / float(c.rearth)
        divergence_field = np.broadcast_to(
            divergence[None, None, :], (nlev, nlon, nlat))
        d0 = specs.nondimensionalize(1.0 / units.second)
        zeros = np.zeros((nlev, nlon, nlat))
        lsp = np.log(float(specs.nondimensionalize(PS_SI * units.pascal)))
        state = State(
            vorticity=hor.to_modal(zeros),
            divergence=hor.to_modal(divergence_field * float(d0)),
            temperature_variation=hor.to_modal(zeros),
            log_surface_pressure=hor.to_modal(lsp * np.ones((1, nlon, nlat))),
            tracers={}, sim_time=0.0,
        )
        self.assertIsInstance(coords.vertical, HybridCoordinates)
        omega_dino = np.asarray(dino_dycore._compute_omega(state))

        dino_lat_idx = int(np.argmin(np.abs(lats - target_lat)))
        omega_dino_col = omega_dino[level, 0, dino_lat_idx]
        dino_lat = float(lats[dino_lat_idx])

        # Reference: the exact continuum value at each backend's own
        # matched latitude/level (the two differ slightly, since neither
        # grid lands exactly on 45 N).
        p_mid = p_mid_1d[level]
        want_pyses = _closed_form_omega(got_lat, p_mid, p_top)
        want_dino = _closed_form_omega(dino_lat, p_mid, p_top)
        self.assertGreater(abs(want_pyses), 0.0)

        # Each backend's own discretization error against the continuum
        # truth (spectral for dinosaur, SE-local for pySES): measured
        # ~0.05% (pySES, ne8) and ~0.01% (dinosaur, T21) -- both tight,
        # since this analytic field is smooth and well resolved at either
        # resolution. Keep headroom over the measured values.
        self.assertLess(abs(omega_pyses_col - want_pyses) / abs(want_pyses),
                        0.02)
        self.assertLess(abs(omega_dino_col - want_dino) / abs(want_dino),
                        0.02)
        # The two backends against EACH OTHER: the combined tolerance the
        # staggering/discretization difference allows (two independent
        # numerical schemes on two different grids, not a bit-identity
        # check) -- measured ~2.7%, dominated by the two grids' nearest
        # columns to 45 N landing at slightly different latitudes (45.5
        # vs 47.1 degrees) rather than by either discretization error.
        self.assertLess(
            abs(omega_pyses_col - omega_dino_col) / abs(want_pyses), 0.08,
            msg=f"pyses={omega_pyses_col:.4g} dinosaur={omega_dino_col:.4g} "
                f"(want~{want_pyses:.4g} Pa/s)")


class OmegaConvectionCompositionTest(unittest.TestCase):
    """TiedtkeConvection composes and runs on pySES with cubasmc active."""

    @pytest.mark.slow
    def test_model_end_to_end_with_cubasmc_active(self):
        from jcm.dycore.pyses import build_forcing
        from jcm.dycore.pyses.pyses_dycore_test import T63_FORCING
        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics

        dycore = _dycore(terrain_file=T63_TERRAIN)
        model = Model(
            dycore=dycore,
            # cu_lmfmid defaults True (ECHAM's own default): Model resolves
            # the dycore-field contract by switching PysesCamSEDycore's
            # omega provider on automatically, so cubasmc runs here with no
            # extra configuration (jax-gcm#698).
            physics=idealized_echam_physics(),
        )
        self.assertTrue(dycore.compute_omega)
        self.assertIn("omega", model.physics.required_dycore_fields())
        forcing = build_forcing(T63_FORCING, dycore)
        dt_days = dycore.dt_seconds / 86400.0
        model.run(forcing=forcing, save_interval=dt_days, total_time=2 * dt_days)
        ps_end = dycore.to_physics_state(model._final_dycore_state)
        temp = np.asarray(ps_end.temperature)
        self.assertTrue(np.isfinite(temp).all())


if __name__ == "__main__":
    unittest.main()
