"""Tests for column_lat_lon: legacy meshgrid equivalence + scattered grids."""

import unittest
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jcm.physics.coords_util import column_lat_lon


@dataclass(frozen=True)
class _Separable:
    nodal_shape: tuple
    latitudes: jnp.ndarray
    longitudes: jnp.ndarray


@dataclass(frozen=True)
class _Scattered(_Separable):
    column_latitudes: jnp.ndarray = None
    column_longitudes: jnp.ndarray = None


class ColumnLatLonTest(unittest.TestCase):
    def test_separable_matches_legacy_meshgrid(self):
        # Bit-identical to the meshgrid(lat, lon).reshape(-1) convention the
        # shipped schemes used inline (lon-major flattening).
        lat = jnp.linspace(-1.4, 1.4, 5)
        lon = jnp.linspace(0.0, 6.0, 8)
        grid = _Separable((8, 5), lat, lon)
        got_lat, got_lon = column_lat_lon(grid)
        ref_lat, ref_lon = jnp.meshgrid(lat, lon)
        np.testing.assert_array_equal(np.asarray(got_lat),
                                      np.asarray(ref_lat.reshape(-1)))
        np.testing.assert_array_equal(np.asarray(got_lon),
                                      np.asarray(ref_lon.reshape(-1)))

    def test_scattered_columns_take_full_arrays(self):
        # A (1, ncol) scattered grid: the separable pair cannot represent the
        # columns (longitudes has length 1); column_* arrays win.
        ncol = 12
        rng = np.random.default_rng(3)
        col_lat = jnp.asarray(rng.uniform(-1.5, 1.5, ncol))
        col_lon = jnp.asarray(rng.uniform(0.0, 2 * np.pi, ncol))
        grid = _Scattered(
            nodal_shape=(1, ncol), latitudes=col_lat,
            longitudes=col_lon[:1],
            column_latitudes=col_lat, column_longitudes=col_lon)
        got_lat, got_lon = column_lat_lon(grid)
        np.testing.assert_array_equal(np.asarray(got_lat), np.asarray(col_lat))
        np.testing.assert_array_equal(np.asarray(got_lon), np.asarray(col_lon))
        # Genuinely per-column: the longitudes are not all one value.
        self.assertGreater(float(jnp.std(got_lon)), 0.1)

    def test_full_shape_arrays_flatten(self):
        # Fake-cubed-sphere style: lat/lon already shaped like nodal_shape.
        lat = jnp.arange(24.0).reshape(2, 3, 4) / 24.0
        lon = jnp.arange(24.0).reshape(2, 3, 4) / 4.0
        grid = _Separable((2, 3, 4), lat, lon)
        got_lat, got_lon = column_lat_lon(grid)
        np.testing.assert_array_equal(np.asarray(got_lat),
                                      np.asarray(lat.reshape(-1)))
        np.testing.assert_array_equal(np.asarray(got_lon),
                                      np.asarray(lon.reshape(-1)))


if __name__ == "__main__":
    unittest.main()
