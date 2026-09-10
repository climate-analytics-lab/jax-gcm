"""Tests for the pySES frontogenesis physics-fields provider.

The provider mirrors CAM-SE's ``compute_frontogenesis``: per-level GLL
spherical gradients, the quadratic form
``F = -grad(theta) . (grad(u_vec) grad(theta))``, DSS, pg2 average.
"""

import unittest

import numpy as np
import jax.numpy as jnp

from jcm.dycore.pyses.dycore import PysesCamSEDycore
from jcm.dycore.pyses.pyses_dycore_test import T63_TERRAIN  # noqa: F401

import jcm.constants as c


def _dycore(nx=3, **kwargs):
    return PysesCamSEDycore(
        nx=nx, npt=4, dt_seconds=900.0, terrain_file=T63_TERRAIN,
        physics_dtype=jnp.float32, compute_frontogenesis=True, **kwargs)


class FrontogenesisProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dycore = _dycore()
        cls.state = cls.dycore.initial_state(None)

    def test_declaration_and_default_off(self):
        self.assertEqual(self.dycore.physics_field_names(),
                         ("frontogenesis",))
        off = PysesCamSEDycore(nx=3, npt=4, dt_seconds=900.0,
                               terrain_file=T63_TERRAIN,
                               physics_dtype=jnp.float32)
        self.assertEqual(off.physics_field_names(), ())
        self.assertEqual(off.physics_fields(None, None), {})

    def test_resting_state_gives_exactly_zero(self):
        # USSA rest: u = v = 0, so every term of F carries a wind gradient
        # factor and F must be identically zero (not merely small).
        physics_state = self.dycore.to_physics_state(self.state)
        out = self.dycore.physics_fields(self.state, physics_state)
        self.assertEqual(set(out), {"frontogenesis"})
        f = np.asarray(out["frontogenesis"])
        nlev = self.dycore.nlev
        self.assertEqual(f.shape, (nlev, 1, self.dycore.colmap.num_cols))
        self.assertEqual(f.dtype, np.float32)
        np.testing.assert_array_equal(f, 0.0)

    @staticmethod
    def _gll_harness(nx=8, nlev=3):
        """Lightweight stand-in exposing exactly what the helper reads.

        Building a full ne8 dycore (terrain interp, vertical grid) is slow;
        ``_frontogenesis_from_gll`` only touches ``h_grid``, ``_rearth`` and
        ``colmap``, so a namespace with those is a faithful harness at a
        resolution where the analytic comparison is meaningful (ne3's 30-deg
        elements leave tens-of-percent max-norm discretization error).
        """
        import types

        from pyses.mesh_generation.element_local_metric import (
            init_quasi_uniform_grid_elem_local,
        )

        from jcm.dycore.pyses.physics_grid import FVPhysicsGrid

        h_grid, dims = init_quasi_uniform_grid_elem_local(
            nx, 4, calc_smooth_tensor=True)
        ns = types.SimpleNamespace(
            h_grid=h_grid,
            _rearth=float(c.rearth),
            colmap=FVPhysicsGrid(h_grid, dims),
        )
        ns.nlev = nlev
        return ns

    def _analytic_gll_fields(self, harness, u0=30.0, theta_amp=10.0):
        """Analytic (u, v, theta) at the GLL points, plus the exact F.

        u = u0 cos(lat), v = 0, theta' = A cos(lat) sin(lon):
        the only surviving term is the cross one,
        F = -theta_x theta_y u_y = -A^2 u0 cos(lon) sin(lon) sin^2(lat)/a^3
        — exactly the term that catches a zonal/meridional component-order
        bug in the gradient contraction.
        """
        coords = np.asarray(harness.h_grid["physical_coords"])
        # pySES convention: physical_coords[..., 0] is LATITUDE,
        # [..., 1] is longitude (verified empirically against
        # horizontal_gradient of sin(lat)/sin(lon)).
        lat = coords[..., 0]
        lon = coords[..., 1]
        nlev = harness.nlev
        a = float(c.rearth)

        def stack(f2d):
            return jnp.asarray(
                np.repeat(f2d[..., None], nlev, axis=-1), jnp.float64)

        u = stack(u0 * np.cos(lat))
        v = stack(np.zeros_like(lat))
        theta = stack(theta_amp * np.cos(lat) * np.sin(lon))
        f_exact = (-theta_amp ** 2 * u0
                   * np.cos(lon) * np.sin(lon) * np.sin(lat) ** 2 / a ** 3)
        return u, v, theta, f_exact

    def test_analytic_deformation_matches(self):
        harness = self._gll_harness(nx=8)
        u, v, theta, f_exact_gll = self._analytic_gll_fields(harness)
        got = np.asarray(PysesCamSEDycore._frontogenesis_from_gll(
            harness, u, v, theta))

        # Exact F averaged to pg2 the same way (constant per level, so
        # compare against the pg2 average of the exact GLL field).
        f_gll = jnp.asarray(
            np.repeat(f_exact_gll[..., None], harness.nlev, axis=-1))
        want = np.asarray(harness.colmap.gather_3d(f_gll))

        scale = np.abs(want).max()
        self.assertGreater(scale, 0.0)
        # The max-norm error concentrates at element corners, where the
        # DSS'd derivative-of-interpolant differs from the pointwise-exact
        # field at O(h^2) (21.6% at ne3 -> 5.6% at ne8); the bulk of the
        # field agrees far more tightly, so bound both norms.
        err_max = np.abs(got - want).max() / scale
        err_mean = np.abs(got - want).mean() / scale
        self.assertLess(err_max, 0.08, msg=f"max rel error {err_max:.3%}")
        self.assertLess(err_mean, 0.01,
                        msg=f"mean rel error {err_mean:.3%}")

    def test_quadratic_scaling_in_theta_is_exact(self):
        harness = self._gll_harness(nx=4)
        u, v, theta1, _ = self._analytic_gll_fields(harness, theta_amp=5.0)
        _, _, theta2, _ = self._analytic_gll_fields(harness, theta_amp=10.0)
        f1 = np.asarray(PysesCamSEDycore._frontogenesis_from_gll(
            harness, u, v, theta1))
        f2 = np.asarray(PysesCamSEDycore._frontogenesis_from_gll(
            harness, u, v, theta2))
        # theta enters only through its gradient, and F is exactly
        # quadratic in it: doubling the perturbation quadruples F.
        np.testing.assert_allclose(f2, 4.0 * f1, rtol=1e-10)

    def test_model_end_to_end_with_frontal_gw(self):
        from jcm.dycore.pyses import build_forcing
        from jcm.dycore.pyses.pyses_dycore_test import T63_FORCING
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        from jcm.physics.convection.tiedtke_nordeng import ConvectionParameters

        dycore = _dycore()
        model = Model(
            dycore=dycore,
            physics=echam_physics(
                radiation_scheme="grey", gw_scheme="frontal",
                # pySES computes omega internally but exposes no provider
                # for it (#698), so ECHAM's ``lmfmid`` mid-level convection
                # trigger cannot run on this backend. Turning it off with
                # the reference's own namelist switch is the documented
                # escape hatch; the Model contract check would otherwise
                # (correctly) refuse to build. Exercised here so the hatch
                # stays working.
                convection=ConvectionParameters.default(cu_lmfmid=False),
            ),
        )
        self.assertEqual(model.physics.required_dycore_fields(),
                         ("frontogenesis",))
        forcing = build_forcing(T63_FORCING, dycore)
        dt_days = dycore.dt_seconds / 86400.0
        model.run(forcing=forcing, save_interval=dt_days,
                  total_time=2 * dt_days)
        ps_end = dycore.to_physics_state(model._final_dycore_state)
        temp = np.asarray(ps_end.temperature)
        self.assertTrue(np.isfinite(temp).all())


if __name__ == "__main__":
    unittest.main()
