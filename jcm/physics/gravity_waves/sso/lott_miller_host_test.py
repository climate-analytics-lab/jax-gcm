"""LottMillerSso must run on whatever horizontal layout the host uses.

The drag is a per-column scheme vmapped over columns, so a whole
``(nlev, nlon, nlat)`` grid and a column-vectorized ``(nlev, ncols)`` block
holding the same columns must produce the same tendency. The term previously
unpacked ``nlev, ncols = state.temperature.shape``, which restricted it to the
column-vectorized layout.

The pressure and height diagnostics are built directly here rather than via
MoistAirColumnState, so these tests exercise LottMillerSso alone.
"""

import unittest

import numpy as np

from jcm import constants as pc
from jcm.physics.gravity_waves.sso.lott_miller import LottMillerSso
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
from jcm.utils import get_coords

NLEV = 8


def _setup():
    coords = get_coords(np.linspace(0.0, 1.0, NLEV + 1), spectral_truncation=21)
    # Real orography: a flat world produces identically zero drag, and every
    # comparison below would then pass without testing anything.
    from importlib import resources
    bc = resources.files("jcm.data.bc.t30.clim")
    terrain = TerrainData.from_file(bc / "terrain.nc", coords=coords)
    return coords, terrain


def _fields(nlon, nlat):
    """Build a sheared, stratified atmosphere that launches real drag."""
    rng = np.random.default_rng(0)
    lev = np.linspace(0.1, 1.0, NLEV).reshape(NLEV, 1, 1)
    shape = (NLEV, nlon, nlat)
    return {
        "temperature": (200.0 + 90.0 * lev
                        + rng.normal(0, 1.0, shape)).astype("f4"),
        "u_wind": rng.normal(8.0, 4.0, shape).astype("f4"),
        "v_wind": rng.normal(0.0, 4.0, shape).astype("f4"),
        "geopotential": np.broadcast_to(
            pc.grav * 16000.0 * (1.0 - lev), shape).astype("f4"),
    }


def _diagnostics(sigma_half, sigma_full, geopotential, surface_pressure):
    """Pressure and height diagnostics in the layout the caller supplies."""
    vshape = (-1,) + (1,) * surface_pressure.ndim
    return {
        "_dt_seconds": 1800.0,
        "pressure_full": (sigma_full.reshape(vshape)
                          * surface_pressure[np.newaxis]),
        "pressure_half": (sigma_half.reshape(vshape)
                          * surface_pressure[np.newaxis]),
        "height_full": geopotential / pc.grav,
    }


def _run(as_grid):
    coords, terrain = _setup()
    nlon, nlat = coords.horizontal.nodal_shape
    f = _fields(nlon, nlat)

    def lay(x):
        return x if as_grid else x.reshape(x.shape[0], -1)

    ps = np.full((nlon, nlat), 1.0e5, dtype="f4")
    ps = ps if as_grid else ps.reshape(-1)

    sigma_full = np.asarray(coords.vertical.centers, dtype="f4")
    sigma_half = np.asarray(coords.vertical.boundaries, dtype="f4")

    state = PhysicsState(
        u_wind=lay(f["u_wind"]), v_wind=lay(f["v_wind"]),
        temperature=lay(f["temperature"]),
        specific_humidity=lay(np.full_like(f["temperature"], 3.0)),
        geopotential=lay(f["geopotential"]),
        normalized_surface_pressure=ps,
    )
    term = LottMillerSso()
    term.cache_coords(coords)
    tend, _ = term(
        state,
        _diagnostics(sigma_half, sigma_full, lay(f["geopotential"]), ps),
        None, terrain)
    return tend, (nlon, nlat)


class TestHostLayouts(unittest.TestCase):
    def test_grid_host_returns_grid_shaped_tendencies(self):
        # The layout that used to raise.
        tend, (nlon, nlat) = _run(as_grid=True)
        self.assertEqual(tend.u_wind.shape, (NLEV, nlon, nlat))
        self.assertEqual(tend.v_wind.shape, (NLEV, nlon, nlat))
        self.assertEqual(tend.temperature.shape, (NLEV, nlon, nlat))

    def test_column_host_returns_column_shaped_tendencies(self):
        tend, (nlon, nlat) = _run(as_grid=False)
        self.assertEqual(tend.u_wind.shape, (NLEV, nlon * nlat))

    def test_the_drag_is_actually_non_zero(self):
        # Guards every comparison here: with flat orography the scheme returns
        # zeros and the layouts would agree without proving anything.
        tend, _ = _run(as_grid=True)
        self.assertGreater(float(np.abs(np.asarray(tend.u_wind)).max()), 0.0)

    def test_layouts_agree_column_for_column(self):
        grid, (nlon, nlat) = _run(as_grid=True)
        flat, _ = _run(as_grid=False)
        for name in ("u_wind", "v_wind", "temperature"):
            g = np.asarray(getattr(grid, name)).reshape(NLEV, nlon * nlat)
            f = np.asarray(getattr(flat, name))
            np.testing.assert_allclose(g, f, rtol=1e-5, atol=1e-8,
                                       err_msg=name)

    def test_tendencies_are_finite(self):
        tend, _ = _run(as_grid=True)
        for name in ("u_wind", "v_wind", "temperature"):
            self.assertTrue(
                np.isfinite(np.asarray(getattr(tend, name))).all(), name)


if __name__ == "__main__":
    unittest.main()
