"""Tests for the pg2 finite-volume physics grid (GLL ↔ column bridge).

Skipped automatically when the optional ``pyses`` dependency is missing.
Run on CPU: ``JAX_PLATFORMS=cpu pytest jcm/dycore/pyses -q``.
"""

import os
import unittest

# pyses freezes its backend at first import — select jax/CPU before pytest
# (or anything else) can import it.
os.environ.setdefault("PYSES_BACKEND", "jax")
os.environ.setdefault("PYSES_USE_CPU", "1")

import numpy as np
import pytest

pytest.importorskip("pyses")

import jax.numpy as jnp

from jcm.dycore.pyses.physics_grid import FVPhysicsGrid


class TestFVPhysicsGrid(unittest.TestCase):
    """pg2 remap identities on an ne3 cubed sphere."""

    @classmethod
    def setUpClass(cls):
        from pyses.mesh_generation.element_local_metric import (
            init_quasi_uniform_grid_elem_local,
        )

        cls.nx, cls.npt, cls.nlev = 3, 4, 5
        cls.h_grid, cls.dims = init_quasi_uniform_grid_elem_local(
            cls.nx, cls.npt, calc_smooth_tensor=True,
        )
        cls.grid = FVPhysicsGrid(cls.h_grid, cls.dims, nf=2)

    def test_num_cols_is_nelem_times_four(self):
        self.assertEqual(self.grid.num_cols, int(self.dims["num_elem"]) * 4)

    def test_cell_centres_are_valid_coordinates(self):
        lat, lon = self.grid.latitudes, self.grid.longitudes
        self.assertEqual(lat.shape, (self.grid.num_cols,))
        self.assertTrue(np.all(np.abs(lat) < np.pi / 2))
        self.assertTrue(np.all((lon >= 0.0) & (lon < 2 * np.pi)))
        # Seam safety: every cell centre must lie inside its own element's
        # angular neighbourhood — compare against the element's Cartesian
        # node mean, which is wrap-free by construction.
        gll = np.asarray(self.h_grid["physical_coords"])
        glat, glon = gll[..., 0], gll[..., 1]
        exyz = np.stack([np.cos(glat) * np.cos(glon),
                         np.cos(glat) * np.sin(glon),
                         np.sin(glat)], -1).mean(axis=(1, 2))
        exyz /= np.linalg.norm(exyz, axis=-1, keepdims=True)
        cxyz = np.stack([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon),
                         np.sin(lat)], -1)
        # cos(angle between cell centre and its element centre)
        cos_sep = np.einsum(
            "ck,ck->c",
            cxyz,
            np.repeat(exyz, 4, axis=0),
        )
        # ne3 elements span ~30 deg; centres must be well within one element.
        self.assertGreater(float(cos_sep.min()), np.cos(np.radians(25.0)))

    def test_gather_of_constant_is_exact(self):
        shape = np.asarray(self.h_grid["physical_coords"]).shape[:3]
        const = jnp.full(shape + (self.nlev,), 3.75, dtype=jnp.float64)
        cols = self.grid.gather_3d(const)
        self.assertEqual(cols.shape, (self.nlev, 1, self.grid.num_cols))
        np.testing.assert_allclose(np.asarray(cols), 3.75, rtol=0, atol=1e-13)

    def test_scatter_then_gather_is_identity(self):
        """The Hannah et al. R1 identity: FV -> GLL -> FV is exact."""
        rng = np.random.default_rng(0)
        cols = jnp.asarray(
            rng.normal(size=(self.nlev, 1, self.grid.num_cols)))
        back = self.grid.gather_3d(self.grid.scatter_3d(cols))
        np.testing.assert_allclose(
            np.asarray(back), np.asarray(cols), rtol=0, atol=1e-10)

    def test_scatter_casts_to_float64(self):
        cols = jnp.zeros((self.nlev, 1, self.grid.num_cols), dtype=jnp.float32)
        self.assertEqual(self.grid.scatter_3d(cols).dtype, jnp.float64)
        cols2 = jnp.zeros((1, self.grid.num_cols), dtype=jnp.float32)
        self.assertEqual(self.grid.scatter_2d(cols2).dtype, jnp.float64)

    def test_dss_idempotent_and_preserves_continuous_fields(self):
        # A smooth global function evaluated at the (shared) GLL nodes is C0
        # by construction: DSS must leave it (essentially) unchanged.
        gll = np.asarray(self.h_grid["physical_coords"])
        smooth = jnp.asarray(np.sin(gll[..., 0]) * np.cos(gll[..., 1]))
        once = self.grid.dss(smooth)
        np.testing.assert_allclose(np.asarray(once), np.asarray(smooth),
                                   rtol=0, atol=1e-12)
        # And on a *discontinuous* field, DSS is a projection: P(P(x)) = P(x).
        rng = np.random.default_rng(1)
        rough = jnp.asarray(rng.normal(size=gll.shape[:3] + (self.nlev,)))
        p1 = self.grid.dss(rough)
        p2 = self.grid.dss(p1)
        np.testing.assert_allclose(np.asarray(p2), np.asarray(p1),
                                   rtol=0, atol=1e-12)

    def test_gather_2d_shape(self):
        shape = np.asarray(self.h_grid["physical_coords"]).shape[:3]
        out = self.grid.gather_2d(jnp.ones(shape))
        self.assertEqual(out.shape, (1, self.grid.num_cols))


if __name__ == "__main__":
    unittest.main()
