"""Tests for :mod:`jcm.ozone_climatology`.

Covers the file loader's contract (variable name, dimension layout,
horizontal+vertical grid match) and the column-flatten convention.

Vertical interpolation lives offline in ``jcm.data.bc.interpolate_ozone``
— the online loader only consumes a pre-interpolated file with shape
``(time, level, lat, lon)`` matching the model's hybrid grid. This test
file builds a synthetic such file so it runs in milliseconds without
loading the real 58 MB CMIP6 source.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.ozone_climatology import OzoneClimatology


def _write_pre_interpolated_ozone(
    path: Path, nlon: int, nlat: int, nlev: int,
) -> None:
    """Write a synthetic pre-interpolated ozone file.

    Matches the format produced by ``jcm.data.bc.interpolate_ozone``:
    ``(time=12, level=nlev, lat, lon)`` mole/mole. O3 varies by level
    (peaks mid-stratosphere) and lon so the column-mapping test is
    sensitive to ordering.
    """
    lat = np.linspace(-88.0, 88.0, nlat).astype(np.float64)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False).astype(np.float64)
    o3 = np.zeros((12, nlev, nlat, nlon), dtype=np.float32)
    peak_lev = nlev // 4   # mid-stratosphere
    for k in range(nlev):
        decay = np.exp(-((k - peak_lev) / 5.0) ** 2)
        for j in range(nlat):
            for i in range(nlon):
                o3[:, k, j, i] = (
                    8.0e-6 * decay
                    * (1.0 + 0.1 * np.cos(np.deg2rad(lat[j])))
                    * (1.0 + 0.01 * i / nlon)
                )
    ds = xr.Dataset(
        {"O3": (("time", "level", "lat", "lon"), o3,
                {"units": "mole mole-1"})},
        coords={
            "time": np.arange(12),
            "level": np.arange(nlev, dtype=np.int32),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)


class TestOzoneClimatology(unittest.TestCase):

    def test_from_file_shape_and_ppmv_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 8, 4, 16
            _write_pre_interpolated_ozone(path, nlon, nlat, nlev)
            clim = OzoneClimatology.from_file(
                path, nlon=nlon, nlat=nlat, nlev=nlev,
            )

        # ``from_file`` returns a 12-month ``TimeSeries`` (WRAP_YEAR
        # mode) so the seasonal cycle rides through
        # ``ForcingData.select(date)``. ``.values`` carries the data.
        self.assertEqual(clim.o3_ppmv.values.shape, (12, nlev, nlon * nlat))
        # File peak 8e-6 mole/mole → ~8 ppmv after the *1e6.
        self.assertAlmostEqual(float(clim.o3_ppmv.values.max()), 8.8, delta=1.0)
        self.assertTrue(clim.is_loaded())

    def test_horizontal_grid_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            _write_pre_interpolated_ozone(path, nlon=8, nlat=4, nlev=16)
            with self.assertRaises(ValueError):
                OzoneClimatology.from_file(path, nlon=16, nlat=8, nlev=16)

    def test_vertical_grid_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            _write_pre_interpolated_ozone(path, nlon=8, nlat=4, nlev=16)
            with self.assertRaises(ValueError):
                OzoneClimatology.from_file(path, nlon=8, nlat=4, nlev=47)

    def test_missing_variable_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.nc"
            xr.Dataset({"foo": (("x",), np.zeros(3))}).to_netcdf(path)
            with self.assertRaises(ValueError):
                OzoneClimatology.from_file(path, nlon=1, nlat=1, nlev=1)

    def test_empty_sentinel(self):
        clim = OzoneClimatology.empty()
        self.assertFalse(clim.is_loaded())

    def test_single_column_grid_is_loaded(self):
        """A legitimate ``(nlev, 1)`` SCM climatology must NOT look empty.

        Regression for codex P2 review on PR #484: the previous
        ``shape[1] > 1`` check treated single-column SCM forcing as
        unloaded, silently falling back to the analytical profile.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            _write_pre_interpolated_ozone(path, nlon=1, nlat=1, nlev=8)
            clim = OzoneClimatology.from_file(path, nlon=1, nlat=1, nlev=8)
        self.assertTrue(clim.is_loaded())
        self.assertEqual(clim.o3_ppmv.values.shape, (12, 8, 1))

    def test_column_ordering_matches_reshape_convention(self):
        """``OzoneClimatology`` must flatten ``(nlat, nlon)`` to the same
        column order as :func:`jcm.physics.composable_physics._reshape_state_to_columns`
        (lon-major, lat-minor — i.e. ``col = i_lon * nlat + i_lat``)
        — checked on every one of the 12 monthly slices.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 8, 4, 4
            _write_pre_interpolated_ozone(path, nlon, nlat, nlev)
            clim = OzoneClimatology.from_file(
                path, nlon=nlon, nlat=nlat, nlev=nlev,
            )

            ds = xr.open_dataset(path, decode_times=False)
            all_months = ds.O3.values  # (12, nlev, nlat, nlon)
            lon_major = np.transpose(all_months, (0, 1, 3, 2)) * 1e6
            expected = lon_major.reshape(12, nlev, nlon * nlat)

            np.testing.assert_allclose(
                np.asarray(clim.o3_ppmv.values), expected, rtol=1e-5,
            )


    def test_transient_file_routes_to_by_date_alignment(self):
        """A multi-year monthly file (``ntime != 12``) must use
        ``BY_DATE`` alignment so the same fraction-of-year doesn't
        sample the same source month every year. Otherwise
        ``WRAP_YEAR`` would silently corrupt SSP / historical runs.
        """
        import datetime as _dt

        from jcm.forcing import BY_DATE

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3_transient.nc"
            nlon, nlat, nlev = 4, 2, 4
            ntime = 24   # 2 years of monthly data
            o3 = np.zeros((ntime, nlev, nlat, nlon), dtype=np.float32)
            for t in range(ntime):
                o3[t] = float(t) * 1e-7
            # CF-encoded time axis the loader can decode.
            base = _dt.datetime(2024, 1, 15)
            time_values = np.array(
                [(base + _dt.timedelta(days=30 * i)).timestamp()
                 for i in range(ntime)],
            ) - _dt.datetime(1970, 1, 1).timestamp()
            ds = xr.Dataset(
                {"O3": (("time", "level", "lat", "lon"), o3,
                        {"units": "mole mole-1"})},
                coords={
                    "time": ("time", time_values / 86400.0,
                             {"units": "days since 1970-01-01",
                              "calendar": "standard"}),
                    "level": np.arange(nlev, dtype=np.int32),
                    "lat": np.linspace(-88, 88, nlat).astype(np.float64),
                    "lon": np.linspace(0, 360, nlon,
                                       endpoint=False).astype(np.float64),
                },
            )
            ds.to_netcdf(path)

            clim = OzoneClimatology.from_file(
                path, nlon=nlon, nlat=nlat, nlev=nlev,
            )

        self.assertEqual(clim.o3_ppmv.values.shape, (ntime, nlev, nlon * nlat))
        self.assertEqual(int(clim.o3_ppmv.align_mode), BY_DATE)

    def test_lat_mismatch_raises_value_error(self):
        """Same shape but flipped latitude axis must fail loudly,
        not silently wire ozone into the wrong latitudes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 8, 4, 4
            _write_pre_interpolated_ozone(path, nlon, nlat, nlev)

            # File built ascending (-88..88) — model expects descending.
            model_lat_descending = np.linspace(88.0, -88.0, nlat)
            with self.assertRaisesRegex(ValueError, "latitudes don't match"):
                OzoneClimatology.from_file(
                    path, nlon=nlon, nlat=nlat, nlev=nlev,
                    lat_deg=model_lat_descending,
                )

    def test_lon_mismatch_raises_value_error(self):
        """Shifted longitude grid must fail loudly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 8, 4, 4
            _write_pre_interpolated_ozone(path, nlon, nlat, nlev)

            # File built [0, 360) — model expects [-180, 180).
            shifted_lon = np.linspace(-180.0, 180.0, nlon, endpoint=False)
            with self.assertRaisesRegex(ValueError, "longitudes don't match"):
                OzoneClimatology.from_file(
                    path, nlon=nlon, nlat=nlat, nlev=nlev,
                    lon_deg=shifted_lon,
                )

    def test_matching_coords_pass_validation(self):
        """When file lat/lon match the model exactly, the load succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 8, 4, 4
            _write_pre_interpolated_ozone(path, nlon, nlat, nlev)

            model_lat = np.linspace(-88.0, 88.0, nlat)
            model_lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
            clim = OzoneClimatology.from_file(
                path, nlon=nlon, nlat=nlat, nlev=nlev,
                lat_deg=model_lat, lon_deg=model_lon,
            )
        self.assertTrue(clim.is_loaded())

    def test_monthly_seasonal_cycle_rides_through_select(self):
        """``ForcingData.select(date)`` must slice the 12-month ozone
        climatology to the date's month (WRAP_YEAR mode), so different
        dates produce different post-select ``o3_ppmv`` profiles.
        """
        import jax.numpy as jnp
        import jax_datetime as jdt
        from datetime import datetime

        from jcm.date import DateData
        from jcm.forcing import default_forcing

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 4, 2, 4
            # Hand-write a file whose monthly anomalies are large so we
            # can detect that the slicer picks the right month.
            o3 = np.zeros((12, nlev, nlat, nlon), dtype=np.float32)
            for m in range(12):
                o3[m] = float(m) * 1e-6   # 0..11 ppmv per month after *1e6
            ds = xr.Dataset(
                {"O3": (("time", "level", "lat", "lon"), o3,
                        {"units": "mole mole-1"})},
                coords={
                    "time": np.arange(12),
                    "level": np.arange(nlev, dtype=np.int32),
                    "lat": np.linspace(-88, 88, nlat).astype(np.float64),
                    "lon": np.linspace(0, 360, nlon,
                                       endpoint=False).astype(np.float64),
                },
            )
            ds.to_netcdf(path)

            clim = OzoneClimatology.from_file(
                path, nlon=nlon, nlat=nlat, nlev=nlev,
            )

        forcing = default_forcing(
            type("G", (), {
                "nodal_shape": (nlon, nlat),
                "latitudes": jnp.linspace(-jnp.pi / 2, jnp.pi / 2, nlat),
                "longitudes": jnp.linspace(0, 2 * jnp.pi, nlon),
            })(),
        )
        forcing = forcing.copy(ozone_climatology=clim)

        def _date(month_zero_based: int) -> DateData:
            dt = jdt.Datetime.from_pydatetime(
                datetime(2026, month_zero_based + 1, 15),
            )
            return DateData.zeros(dt=dt)

        # WRAP_YEAR splits the year evenly into 12 bins; mid-January
        # lands in bin 0, mid-July in bin 6.
        jan = forcing.select(_date(0))
        jul = forcing.select(_date(6))

        self.assertAlmostEqual(float(jan.ozone_climatology.o3_ppmv.max()), 0.0)
        self.assertAlmostEqual(float(jul.ozone_climatology.o3_ppmv.max()), 6.0)


if __name__ == "__main__":
    unittest.main()
