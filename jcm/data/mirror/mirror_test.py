"""Unit tests for the mirror builders' pure-math pieces.

The builders themselves read multi-GB Glade sources; these tests cover
the grid/statistics machinery on synthetic inputs.
"""

import unittest

import numpy as np

from jcm.data.mirror.bundles import (conservative_to_gaussian,
                                     gaussian_latlon, interp_to)
from jcm.data.mirror.sso import finalize
from jcm.data.mirror.registry import build_registry, write_registry


class SsoFinalizeTest(unittest.TestCase):
    def _acc(self, **over):
        n = np.full(4, 10.0)
        acc = {"n": n, "land": n.copy(), "sh": np.zeros(4),
               "sh2": np.zeros(4), "shx2": np.zeros(4),
               "shy2": np.zeros(4), "shxy": np.zeros(4),
               "pic": np.zeros(4), "val": np.zeros(4)}
        acc.update(over)
        return acc

    def test_isotropic_slope_has_zero_anisotropy(self):
        # equal x/y gradient variance, no correlation -> gamma = 1 (round)
        acc = self._acc(shx2=np.full(4, 10.0), shy2=np.full(4, 10.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 1.0)
        np.testing.assert_allclose(out["orosig"], 1.0)  # sqrt(K+L=1)

    def test_pure_xslope_is_fully_anisotropic(self):
        # gradient variance only in x -> gamma = 0, theta = 0 deg
        acc = self._acc(shx2=np.full(4, 20.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orogam"], 0.0)
        np.testing.assert_allclose(out["orothe"], 0.0)
        np.testing.assert_allclose(out["orosig"], np.sqrt(2.0))

    def test_ocean_cells_are_zeroed(self):
        acc = self._acc(land=np.array([10.0, 0.0, 10.0, 0.0]),
                        sh=np.full(4, 50.0), shx2=np.full(4, 4.0))
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], [5.0, 0.0, 5.0, 0.0])
        np.testing.assert_allclose(out["lsm"], [1.0, 0.0, 1.0, 0.0])

    def test_variance_from_sufficient_statistics(self):
        h = np.array([1.0, 3.0, 5.0, 7.0])
        acc = self._acc(n=np.full(4, 2.0), land=np.full(4, 2.0),
                        sh=2 * h, sh2=2 * h ** 2 + 2.0)
        # per-cell: two samples h +/- 1 -> mean h, std 1
        out = finalize(acc)
        np.testing.assert_allclose(out["orog"], h)
        np.testing.assert_allclose(out["orostd"], 1.0)


class RegridTest(unittest.TestCase):
    def test_gaussian_latlon_shapes(self):
        lats, lons = gaussian_latlon(96)
        self.assertEqual((lats.size, lons.size), (96, 192))
        self.assertTrue(np.all(np.diff(lats) > 0))
        self.assertAlmostEqual(lons[0], 0.0)

    def test_conservative_preserves_constant_field(self):
        src_lats = np.linspace(-89.75, 89.75, 360)
        src_lons = np.arange(720) * 0.5
        lats, lons = gaussian_latlon(32)
        out = conservative_to_gaussian(np.full((360, 720), 3.5),
                                       src_lats, src_lons, lats, lons)
        np.testing.assert_allclose(out, 3.5)

    def test_conservative_preserves_global_integral(self):
        rng = np.random.default_rng(0)
        src_lats = np.linspace(-89.75, 89.75, 360)
        src_lons = np.arange(720) * 0.5
        field = rng.random((360, 720))
        lats, lons = gaussian_latlon(24)
        out = conservative_to_gaussian(field, src_lats, src_lons, lats, lons)
        w_src = np.cos(np.deg2rad(src_lats))[:, None]
        # target cell weights: sum of source weights landing in each cell
        src_int = (field * w_src).sum()
        lat_edges = np.concatenate([[-90], 0.5 * (lats[1:] + lats[:-1]),
                                    [90]])
        # integral back over target using source-weight-consistent areas
        lat_bin = np.searchsorted(lat_edges, src_lats) - 1
        w_tgt = np.zeros(lats.size)
        np.add.at(w_tgt, lat_bin, w_src[:, 0] * src_lons.size)
        tgt_int = (out.mean(axis=1) * w_tgt).sum()
        self.assertAlmostEqual(tgt_int / src_int, 1.0, places=6)

    def test_interp_to_wraps_longitude(self):
        import xarray as xr
        src = xr.DataArray(
            np.tile(np.sin(np.deg2rad(np.arange(360))), (91, 1)),
            dims=("lat", "lon"),
            coords={"lat": np.linspace(-90, 90, 91),
                    "lon": np.arange(360.0)})
        out = interp_to(src, np.array([0.0]), np.array([359.5]))
        expected = 0.5 * (np.sin(np.deg2rad(359)) + np.sin(0.0))
        self.assertAlmostEqual(out.values.item(), expected, places=6)


class RegistryTest(unittest.TestCase):
    def test_registry_hashes_files(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "sub").mkdir()
            (Path(d) / "sub" / "a.nc").write_bytes(b"hello")
            path = write_registry(d)
            reg = json.loads(Path(path).read_text())
            self.assertIn("sub/a.nc", reg["files"])
            self.assertEqual(reg["files"]["sub/a.nc"]["size"], 5)
            # registry.json itself is excluded
            reg2 = build_registry(d)
            self.assertNotIn("registry.json", reg2["files"])


if __name__ == "__main__":
    unittest.main()
