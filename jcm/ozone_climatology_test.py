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

import jax.numpy as jnp
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

        self.assertEqual(clim.o3_ppmv.shape, (nlev, nlon * nlat))
        # File peak 8e-6 mole/mole → ~8 ppmv after the *1e6.
        self.assertAlmostEqual(float(clim.o3_ppmv.max()), 8.8, delta=1.0)
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

    def test_column_ordering_matches_reshape_convention(self):
        """``OzoneClimatology`` must flatten ``(nlat, nlon)`` to the same
        column order as :func:`jcm.physics.composable_physics._reshape_state_to_columns`
        (lon-major, lat-minor — i.e. ``col = i_lon * nlat + i_lat``).
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o3.nc"
            nlon, nlat, nlev = 8, 4, 4
            _write_pre_interpolated_ozone(path, nlon, nlat, nlev)
            clim = OzoneClimatology.from_file(
                path, nlon=nlon, nlat=nlat, nlev=nlev,
            )

            ds = xr.open_dataset(path, decode_times=False)
            ann_mean = ds.O3.mean(dim="time").values  # (nlev, nlat, nlon)
            ann_mean_lon_major = np.transpose(ann_mean, (0, 2, 1)) * 1e6
            expected = ann_mean_lon_major.reshape(nlev, nlon * nlat)

            np.testing.assert_allclose(
                np.asarray(clim.o3_ppmv), expected, rtol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
