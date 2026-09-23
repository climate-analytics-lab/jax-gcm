"""Unit tests for the shared host-side regridding module."""

import unittest

import numpy as np

from jcm.data.regridding import (build_regridder, conservative_to_gaussian,
                                 fill_nearest, gaussian_latlon, interp_to,
                                 nearest_index, unit_sphere_vectors)


class GridHelpersTest(unittest.TestCase):
    def test_gaussian_latlon_shapes(self):
        lats, lons = gaussian_latlon(96)
        self.assertEqual((lats.size, lons.size), (96, 192))
        self.assertTrue(np.all(np.diff(lats) > 0))
        self.assertAlmostEqual(lons[0], 0.0)

    def test_unit_sphere_vectors_are_unit(self):
        v = unit_sphere_vectors(np.array([0.0, 45.0, -90.0]),
                                np.array([0.0, 90.0, 180.0]))
        np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0)

    def test_nearest_index_identity_on_same_points(self):
        lat = np.array([10.0, -20.0, 55.0])
        lon = np.array([0.0, 120.0, 250.0])
        np.testing.assert_array_equal(nearest_index(lat, lon, lat, lon),
                                      [0, 1, 2])


class ConservativeTest(unittest.TestCase):
    def test_preserves_constant_field(self):
        src_lats = np.linspace(-89.75, 89.75, 360)
        src_lons = np.arange(720) * 0.5
        lats, lons = gaussian_latlon(32)
        out = conservative_to_gaussian(np.full((360, 720), 3.5),
                                       src_lats, src_lons, lats, lons)
        np.testing.assert_allclose(out, 3.5)

    def test_preserves_global_integral(self):
        rng = np.random.default_rng(0)
        src_lats = np.linspace(-89.75, 89.75, 360)
        src_lons = np.arange(720) * 0.5
        field = rng.random((360, 720))
        lats, lons = gaussian_latlon(24)
        out = conservative_to_gaussian(field, src_lats, src_lons, lats, lons)
        w_src = np.cos(np.deg2rad(src_lats))[:, None]
        src_int = (field * w_src).sum()
        # weight each target cell by the source area it received
        glat, glon = np.meshgrid(src_lats, src_lons, indexing="ij")
        rg = build_regridder(glon.ravel(), glat.ravel(),
                             np.cos(np.deg2rad(glat)).ravel(),
                             lons, lats, dst_in_degrees=True)
        w_tgt = rg._covered_area.reshape(lons.size, lats.size).T
        tgt_int = (out * w_tgt).sum()
        self.assertAlmostEqual(tgt_int / src_int, 1.0, places=10)

    def test_rectilinear_axes_match_flattened_mesh(self):
        # 1-D lon/lat axes with a 2-D area (#533) must build the identical
        # operator as the pre-flattened mesh, for both (lon, lat) and
        # (lat, lon) area layouts — and apply directly to native
        # unflattened fields, e.g. a (time, lat, lon) input4MIPs series.
        rng = np.random.default_rng(3)
        src_lats = np.linspace(-85.0, 85.0, 18)
        src_lons = np.arange(36) * 10.0
        mlon, mlat = np.meshgrid(src_lons, src_lats, indexing="ij")
        area = np.cos(np.deg2rad(mlat))                  # (nlon, nlat)
        lats, lons = gaussian_latlon(8)
        ref = build_regridder(mlon.ravel(), mlat.ravel(), area.ravel(),
                              lons, lats, dst_in_degrees=True)
        field = rng.random((3, 36, 18))                  # (time, lon, lat)
        expect = ref(field.reshape(3, -1))
        for rect_area, f in ((area, field),
                             (area.T, np.swapaxes(field, -1, -2))):
            rg = build_regridder(src_lons, src_lats, rect_area,
                                 lons, lats, dst_in_degrees=True)
            np.testing.assert_allclose(rg._matrix.toarray(),
                                       ref._matrix.toarray())
            # Both trailing layouts are recognized by shape.
            np.testing.assert_allclose(rg(field), expect)
            np.testing.assert_allclose(rg(np.swapaxes(field, -1, -2)), expect)
            np.testing.assert_allclose(rg(f), expect)
        with self.assertRaisesRegex(ValueError, "trailing spatial"):
            rg(field[:, :7, :5])

    def test_rectilinear_ambiguous_area_shape_raises(self):
        src_lats = np.linspace(-85.0, 85.0, 18)
        src_lons = np.arange(36) * 10.0
        lats, lons = gaussian_latlon(8)
        with self.assertRaisesRegex(ValueError, "src_area shape"):
            build_regridder(src_lons, src_lats, np.ones((7, 5)),
                            lons, lats, dst_in_degrees=True)

    def test_matches_bruteforce_binning(self):
        # independent reference: loop-based nearest-center area-weighted mean
        rng = np.random.default_rng(1)
        src_lats = np.linspace(-85.0, 85.0, 40)
        src_lons = np.arange(80) * 4.5
        field = rng.random((40, 80))
        lats, lons = gaussian_latlon(8)
        out = conservative_to_gaussian(field, src_lats, src_lons, lats, lons)
        ref_num = np.zeros((lats.size, lons.size))
        ref_den = np.zeros((lats.size, lons.size))
        for j, la in enumerate(src_lats):
            i_lat = np.argmin(np.abs(lats - la))
            w = np.cos(np.deg2rad(la))
            for i, lo in enumerate(src_lons):
                d = np.abs(lons - lo)
                i_lon = np.argmin(np.minimum(d, 360.0 - d))
                ref_num[i_lat, i_lon] += w * field[j, i]
                ref_den[i_lat, i_lon] += w
        ref = np.where(ref_den > 0, ref_num / np.maximum(ref_den, 1e-30), 0.0)
        np.testing.assert_allclose(out, ref, rtol=1e-12)


class BilinearTest(unittest.TestCase):
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

    def test_fill_nearest_takes_closest_valid_value(self):
        lats = np.array([0.0, 10.0])
        lons = np.array([0.0, 1.0, 30.0])
        field = np.array([[[1.0, np.nan, 3.0],
                           [4.0, 5.0, 6.0]]])
        out = fill_nearest(field, lats, lons)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out[0, 0, 1], 1.0)     # nearest valid is (0, 0)
        self.assertEqual(out[0, 1, 2], 6.0)     # untouched cells identical


if __name__ == "__main__":
    unittest.main()


class ConservativeOverlapTest(unittest.TestCase):
    """The exact-overlap remap the dust products are coarsened/refined with."""

    def _field(self, lats, lons):
        la, lo = np.meshgrid(np.deg2rad(lats), np.deg2rad(lons), indexing="ij")
        return 1.0 + np.cos(la) ** 2 * np.cos(3 * lo) + 0.5 * np.sin(la)

    def test_gaussian_bounds_reproduce_the_quadrature_weights(self):
        from jcm.data.regridding import latitude_bounds
        lats, _ = gaussian_latlon(48)
        edges = np.sin(np.deg2rad(latitude_bounds(lats)))
        np.testing.assert_allclose(
            np.diff(edges), np.polynomial.legendre.leggauss(48)[1], atol=1e-12)
        self.assertTrue(np.all((edges[:-1] < np.sin(np.deg2rad(lats)))
                               & (np.sin(np.deg2rad(lats)) < edges[1:])))

    def test_constant_is_preserved_coarsening_and_refining(self):
        from jcm.data.regridding import conservative_overlap
        fine, coarse = gaussian_latlon(192), gaussian_latlon(160)
        for (sl, so), (dl, do) in ((fine, coarse), (coarse, fine)):
            out = conservative_overlap(np.full((2, sl.size, so.size), 2.5),
                                       sl, so, dl, do)
            self.assertEqual(out.shape, (2, dl.size, do.size))
            np.testing.assert_allclose(out, 2.5, rtol=1e-12)

    def test_global_integral_is_conserved(self):
        from jcm.data.regridding import conservative_overlap
        weights = lambda n: np.polynomial.legendre.leggauss(n)[1]  # noqa: E731
        for src_n, dst_n in ((192, 160), (96, 192), (384, 192)):
            (sl, so), (dl, do) = gaussian_latlon(src_n), gaussian_latlon(dst_n)
            f = self._field(sl, so)
            out = conservative_overlap(f, sl, so, dl, do)
            total_src = (weights(src_n)[:, None] * f).sum() / so.size
            total_dst = (weights(dst_n)[:, None] * out).sum() / do.size
            self.assertAlmostEqual(total_src, total_dst, places=12)

    def test_identity_on_the_same_grid(self):
        from jcm.data.regridding import conservative_overlap
        lats, lons = gaussian_latlon(96)
        f = self._field(lats, lons)
        np.testing.assert_allclose(
            conservative_overlap(f, lats, lons, lats, lons), f, atol=1e-12)

    def test_missing_source_is_renormalised_not_diluted(self):
        from jcm.data.regridding import conservative_overlap
        (sl, so), (dl, do) = gaussian_latlon(192), gaussian_latlon(96)
        f = np.full((sl.size, so.size), np.nan)
        f[100:120, 40:80] = 0.03              # a valid patch, NaN elsewhere
        out = conservative_overlap(f, sl, so, dl, do)
        finite = np.isfinite(out)
        self.assertTrue(finite.any() and (~finite).any())
        np.testing.assert_allclose(out[finite], 0.03, rtol=1e-12)
