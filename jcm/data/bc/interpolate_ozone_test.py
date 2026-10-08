"""Tests for the field-generic log-pressure interpolation helper.

``vertical_interp_log_p`` is public and documented as applying to any field
given on pressure levels, not just ozone (#830), so the contract under test is
"axis 1 is the pressure axis, every other axis is carried through untouched,
whatever the rank". The 4-D case additionally pins bit-identity against a
direct per-column ``np.interp`` loop, which is what the ozone climatology
builder used before the helper was generalised — the generalisation must not
move a single bit of the shipped boundary-condition files.
"""

import unittest

import numpy as np

from jcm.data.bc.interpolate_ozone import vertical_interp_log_p


def _plev_ascending(n=11):
    """Source pressures, surface-last (ascending), spanning 1 hPa to 1000 hPa."""
    return np.logspace(np.log10(100.0), np.log10(100000.0), n)


def _plev_target(n=7):
    """Target pressures strictly inside the source range (no extrapolation)."""
    return np.logspace(np.log10(200.0), np.log10(98000.0), n)


def _reference_loop(source, plev_source, plev_target):
    """Per-column ``np.interp``, written without any rank assumption.

    Deliberately independent of the implementation: it flattens by hand and
    interpolates one column at a time, which is the definition the helper has
    to meet at every rank.
    """
    log_src = np.log(plev_source)
    if log_src[0] > log_src[-1]:
        log_src = log_src[::-1]
        source = np.flip(source, axis=1)
    moved = np.moveaxis(source, 1, -1)
    flat = moved.reshape(-1, plev_source.size)
    out = np.stack(
        [np.interp(np.log(plev_target), log_src, col) for col in flat]
    ).astype(source.dtype)
    return np.moveaxis(
        out.reshape(*moved.shape[:-1], plev_target.size), -1, 1
    )


class VerticalInterpRankTest(unittest.TestCase):
    """Any number of leading/trailing axes; only axis 1 is the pressure axis."""

    def setUp(self):
        self.rng = np.random.default_rng(20830)
        self.psrc = _plev_ascending()
        self.ptgt = _plev_target()

    def _check_rank(self, shape):
        field = self.rng.random(shape)
        out = vertical_interp_log_p(field, self.psrc, self.ptgt)
        expected_shape = shape[:1] + (self.ptgt.size,) + shape[2:]
        self.assertEqual(out.shape, expected_shape)
        np.testing.assert_array_equal(
            out, _reference_loop(field, self.psrc, self.ptgt)
        )

    def test_3d_time_pressure_column(self):
        # (time, pressure, column) — the shape the old 4-D unpacking rejected.
        self._check_rank((3, 11, 6))

    def test_4d_time_pressure_lat_lon(self):
        # The ozone-file shape, i.e. the one that already worked.
        self._check_rank((3, 11, 5, 4))

    def test_5d_ensemble_pressure_time_lat_lon(self):
        # An extra leading ensemble axis: two axes before and three after the
        # reshape, so it exercises the moveaxis/flatten round trip properly.
        self._check_rank((2, 11, 3, 5, 4))

    def test_2d_minimum_rank(self):
        self._check_rank((4, 11))

    def test_trailing_axes_are_not_transposed(self):
        # A field constant in pressure but varying across the trailing axes
        # comes back with those axes in the same order — catches a moveaxis
        # or reshape that silently permutes lat/lon.
        horiz = self.rng.random((5, 4))
        field = np.broadcast_to(horiz, (3, 11, 5, 4)).copy()
        out = vertical_interp_log_p(field, self.psrc, self.ptgt)
        for k in range(self.ptgt.size):
            np.testing.assert_allclose(out[0, k], horiz)


class VerticalInterpBitIdentityTest(unittest.TestCase):
    """The 4-D result must be bit-identical to the pre-#830 implementation."""

    def _old_implementation(self, o3_source, plev_source, plev_target):
        """Interpolate as the pre-#830 implementation did, 4-D only."""
        log_src = np.log(plev_source)
        log_tgt = np.log(plev_target)
        if log_src[0] > log_src[-1]:
            log_src = log_src[::-1]
            o3_source = o3_source[:, ::-1]
        out = np.empty(
            (o3_source.shape[0], plev_target.size, *o3_source.shape[2:]),
            dtype=o3_source.dtype,
        )
        ntime, _, nlat, nlon = o3_source.shape
        for t in range(ntime):
            for j in range(nlat):
                for i in range(nlon):
                    out[t, :, j, i] = np.interp(
                        log_tgt, log_src, o3_source[t, :, j, i]
                    )
        return out

    def _check_identical(self, plev_source, dtype):
        rng = np.random.default_rng(830)
        field = rng.random((3, 11, 5, 4)).astype(dtype)
        ptgt = _plev_target()
        new = vertical_interp_log_p(field, plev_source, ptgt)
        old = self._old_implementation(field, plev_source, ptgt)
        self.assertEqual(new.dtype, old.dtype)
        np.testing.assert_array_equal(new, old)

    def test_bit_identical_ascending_float64(self):
        self._check_identical(_plev_ascending(), np.float64)

    def test_bit_identical_ascending_float32(self):
        self._check_identical(_plev_ascending(), np.float32)

    def test_bit_identical_descending_float32(self):
        # The ozone files are top-first, so the descending branch is the one
        # the shipped T63L47 file actually took.
        self._check_identical(_plev_ascending()[::-1].copy(), np.float32)


class VerticalInterpPressureOrderTest(unittest.TestCase):
    """Descending source pressure is flipped, not trusted to np.interp."""

    def test_descending_matches_ascending(self):
        rng = np.random.default_rng(7)
        psrc = _plev_ascending()
        ptgt = _plev_target()
        field = rng.random((2, 11, 3))
        from_ascending = vertical_interp_log_p(field, psrc, ptgt)
        # Same physical field, written top-first instead of surface-first.
        from_descending = vertical_interp_log_p(
            np.flip(field, axis=1), psrc[::-1].copy(), ptgt
        )
        np.testing.assert_array_equal(from_ascending, from_descending)

    def test_descending_recovers_exact_nodes(self):
        # Interpolating onto the source levels themselves must return the
        # source values — the sharpest check that the flip pairs pressures
        # with the right levels rather than reversing the profile.
        psrc = _plev_ascending()
        profile = np.arange(11.0)
        field = profile.reshape(1, 11, 1)
        out = vertical_interp_log_p(
            np.flip(field, axis=1), psrc[::-1].copy(), psrc
        )
        np.testing.assert_allclose(out[0, :, 0], profile)

    def test_non_monotonic_source_pressure_raises(self):
        psrc = _plev_ascending()
        psrc[5], psrc[6] = psrc[6], psrc[5]
        with self.assertRaisesRegex(ValueError, "monotonic"):
            vertical_interp_log_p(np.zeros((2, 11, 3)), psrc, _plev_target())


class VerticalInterpValidationTest(unittest.TestCase):
    """Shape errors name the shapes rather than failing deep inside numpy."""

    def test_level_count_mismatch_raises_naming_shapes(self):
        with self.assertRaisesRegex(ValueError, r"11 levels.*axis 1 has 9"):
            vertical_interp_log_p(
                np.zeros((3, 9, 4)), _plev_ascending(), _plev_target()
            )

    def test_one_dimensional_source_raises(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            vertical_interp_log_p(
                np.zeros(11), _plev_ascending(), _plev_target()
            )

    def test_two_dimensional_plev_source_raises(self):
        with self.assertRaisesRegex(ValueError, "1-D"):
            vertical_interp_log_p(
                np.zeros((3, 11, 4)),
                _plev_ascending().reshape(11, 1),
                _plev_target(),
            )


class VerticalInterpDtypeTest(unittest.TestCase):
    """The input dtype survives, so the writer controls the file's precision."""

    def test_dtype_preserved(self):
        psrc, ptgt = _plev_ascending(), _plev_target()
        for dtype in (np.float32, np.float64):
            # The subtest label must be a plain string: xdist ships subtest
            # reports through execnet, which cannot serialise a numpy type
            # object, so passing ``dtype=np.float32`` fails under -n but
            # passes serially.
            with self.subTest(dtype=np.dtype(dtype).name):
                field = np.ones((2, 11, 3), dtype=dtype)
                out = vertical_interp_log_p(field, psrc, ptgt)
                self.assertEqual(out.dtype, dtype)

    def test_interpolation_is_linear_in_log_pressure(self):
        # A profile linear in ln(p) must be reproduced exactly at any target
        # level — the property that makes log-pressure the right space.
        psrc = _plev_ascending()
        profile = 3.0 * np.log(psrc) - 1.0
        out = vertical_interp_log_p(
            profile.reshape(1, psrc.size, 1), psrc, _plev_target()
        )
        np.testing.assert_allclose(
            out[0, :, 0], 3.0 * np.log(_plev_target()) - 1.0
        )



def _write_source(path, plev_pa, value_fn):
    """Write a 12-month (time, plev, lat, lon) ozone source on ``plev_pa``."""
    import xarray as xr

    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 180.0])
    o3 = np.broadcast_to(
        value_fn(plev_pa)[None, :, None, None],
        (12, plev_pa.size, lat.size, lon.size)).astype(np.float32)
    ds = xr.Dataset(
        {"O3": (("time", "plev", "lat", "lon"), o3)},
        coords={"time": np.arange(1, 13), "plev": plev_pa,
                "lat": lat, "lon": lon})
    ds.to_netcdf(path)


class OzoneTopClampGuardTest(unittest.TestCase):
    """A source that stops below the model lid is refused, not clamped.

    ``np.interp`` holds the source's top value above its top level, so a CAM6
    climatology ending near 4 hPa wrote its 7 ppmv stratospheric value into
    every L47 level up to the 1 Pa lid (#1060): ~40x the mesospheric ozone,
    which held the model top some 50 K too warm.
    """

    def test_a_source_below_the_lid_is_refused(self):
        import tempfile
        from pathlib import Path

        from jcm.data.bc.interpolate_ozone import interpolate_ozone

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.nc"
            # 400 Pa (4 hPa) to 1000 hPa, like the CAM6 L32 climatology.
            _write_source(src, np.logspace(np.log10(400.0), 5.0, 20),
                          lambda p: 7e-6 * np.ones_like(p))
            with self.assertRaisesRegex(ValueError, "above it"):
                interpolate_ozone(src, Path(tmp) / "out.nc", 47)
            self.assertFalse((Path(tmp) / "out.nc").exists())

    def test_the_clamp_can_be_asked_for(self):
        import tempfile
        from pathlib import Path

        import xarray as xr

        from jcm.data.bc.interpolate_ozone import interpolate_ozone

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.nc"
            _write_source(src, np.logspace(np.log10(400.0), 5.0, 20),
                          lambda p: 7e-6 * np.ones_like(p))
            interpolate_ozone(src, Path(tmp) / "out.nc", 47,
                              allow_top_clamp=True)
            with xr.open_dataset(Path(tmp) / "out.nc") as out:
                np.testing.assert_allclose(out.O3.values[0, 0, 0, 0], 7e-6,
                                           rtol=1e-6)

    def test_a_source_reaching_the_lid_keeps_its_mesospheric_decline(self):
        import tempfile
        from pathlib import Path

        import xarray as xr

        from jcm.data.bc.interpolate_ozone import interpolate_ozone

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.nc"
            # FZJ-like range (0.01 Pa to 1000 hPa); ozone linear in ln p.
            _write_source(src, np.logspace(-2.0, 5.0, 40),
                          lambda p: 1e-7 * (1.0 + np.log(p / 0.01)))
            interpolate_ozone(src, Path(tmp) / "out.nc", 47)
            with xr.open_dataset(Path(tmp) / "out.nc") as out:
                top = out.O3.values[0, :3, 0, 0]
                p = out.level_pressure_pa.values[:3]
            np.testing.assert_allclose(top, 1e-7 * (1.0 + np.log(p / 0.01)),
                                       rtol=1e-5)
            self.assertTrue(np.all(np.diff(top) > 0))


class PackagedOzoneTest(unittest.TestCase):
    """The packaged T63L47 climatology declines through the mesosphere.

    ``forcing.ozone_file: auto`` resolves ``jcm/data/bc/t63/ozone.nc`` for
    every T63L47 preset that does not name a bundle (the 1M/2M/RRTMGP
    members); its top levels must carry the mesospheric decline, as the
    mirror's ``ozone_pd`` bundle does, not a clamped stratospheric value.
    """

    def test_top_levels_decline_to_the_lid(self):
        from importlib import resources

        import xarray as xr

        path = resources.files("jcm") / "data" / "bc" / "t63" / "ozone.nc"
        with xr.open_dataset(str(path)) as ds:
            w = np.cos(np.deg2rad(ds.lat))
            profile = (ds.O3.mean(("time", "lon")).weighted(w).mean("lat")
                       .values)
            p = ds.level_pressure_pa.values
        # Increasing downward from the lid to the stratospheric maximum.
        imax = int(np.argmax(profile))
        self.assertGreater(p[imax], 300.0)
        self.assertTrue(np.all(np.diff(profile[:imax + 1]) > 0))
        # Mesospheric values at the lid: well under 1 ppmv at 1 Pa (the
        # clamped CAM6 file held 7.4 ppmv there).
        self.assertLess(profile[0], 0.5e-6)
        self.assertGreater(profile[0], 0.05e-6)


class ZonalMeanClimatologyTest(unittest.TestCase):
    """``zonal_mean_climatology`` builds the packaged copy from the bundle."""

    def test_zonal_mean_is_broadcast_and_profile_kept(self):
        import xarray as xr

        from jcm.data.mirror.ozone import zonal_mean_climatology

        rng = np.random.default_rng(0)
        o3 = rng.uniform(1e-7, 8e-6, (12, 5, 3, 4)).astype(np.float32)
        ds = xr.Dataset(
            {"O3": (("time", "level", "lat", "lon"), o3, {"units": "mole mole-1"}),
             "level_pressure_pa": (("level",), np.arange(5.0))},
            coords={"time": np.arange(1, 13), "level": np.arange(5),
                    "lat": [-30.0, 0.0, 30.0], "lon": [0.0, 90.0, 180.0, 270.0]},
            attrs={"history": "h."})
        out = zonal_mean_climatology(ds, "src")
        self.assertEqual(out.O3.dims, ds.O3.dims)
        self.assertEqual(out.O3.dtype, ds.O3.dtype)
        np.testing.assert_allclose(out.O3.values, np.broadcast_to(
            o3.mean(axis=-1, keepdims=True), o3.shape), rtol=1e-6)
        self.assertEqual(out.attrs["source_file"], "src")
        self.assertEqual(out.O3.attrs["units"], "mole mole-1")
        np.testing.assert_array_equal(out.level_pressure_pa.values,
                                      ds.level_pressure_pa.values)


if __name__ == "__main__":
    unittest.main()
