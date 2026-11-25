"""Tests for the built-in ECHAM/ICON hybrid level tables."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.echam.echam_levels import (
    get_echam_levels,
    hybrid_levels_from_table,
    is_builtin_table,
    levels_from_level_table,
    read_vct_file,
)


class EchamLevelsTest(unittest.TestCase):
    def _check_table(self, nlev, p_top_hpa_max):
        hc = get_echam_levels(nlev)
        a = np.asarray(hc.a_boundaries)
        b = np.asarray(hc.b_boundaries)
        self.assertEqual(a.shape, (nlev + 1,))
        self.assertEqual(b.shape, (nlev + 1,))
        # Half-pressures strictly increase from TOA to a 1013.25 hPa surface.
        ph = a + b * 101325.0
        self.assertTrue(np.all(np.diff(ph) > 0.0))
        self.assertAlmostEqual(float(ph[-1]), 101325.0)
        self.assertLessEqual(float(ph[0]) / 100.0, p_top_hpa_max)
        # b spans [0, 1] exactly at the ends.
        self.assertEqual(float(b[0]), 0.0)
        self.assertEqual(float(b[-1]), 1.0)

    def test_l47_table(self):
        self._check_table(47, p_top_hpa_max=0.01)

    def test_l95_ma_table(self):
        # ECHAM6 middle-atmosphere grid: lid at ~0.01 hPa, first full
        # level below ~0.02 hPa half-pressure.
        self._check_table(95, p_top_hpa_max=1e-6)
        hc = get_echam_levels(95)
        ph1_hpa = float(np.asarray(hc.a_boundaries)[1]) / 100.0
        self.assertLess(ph1_hpa, 0.03)
        self.assertGreater(ph1_hpa, 0.01)

    def test_unsupported_count_raises(self):
        with self.assertRaises(ValueError):
            get_echam_levels(63)

    def test_l40_now_unsupported(self):
        # The old L40 "grid" was the bottom 40 rows of L47 — a 274 hPa model
        # top, not a real grid (issue #680 item 1). It is gone; 40 now raises
        # the unknown-count error like any other unsupported count.
        with self.assertRaises(ValueError) as ctx:
            get_echam_levels(40)
        self.assertIn("47, 95", str(ctx.exception))

    def test_truncated_table_guard_fires(self):
        # A table whose top interface still sits at a high pressure (a bottom
        # slice of a longer vct) is rejected by the construction-time guard.
        import jax.numpy as jnp

        from jcm.physics.echam.echam_levels import _checked_hybrid
        # 3-level toy with a 274 hPa "top" (a[0]=27400 Pa) and b spanning [0,1].
        a = jnp.array([27400.0, 10000.0, 2000.0, 0.0])
        b = jnp.array([0.0, 0.3, 0.6, 1.0])
        with self.assertRaisesRegex(ValueError, "truncated vct"):
            _checked_hybrid(a, b)

    def test_full_depth_table_passes_guard(self):
        import jax.numpy as jnp

        from jcm.physics.echam.echam_levels import _checked_hybrid
        a = jnp.array([0.0, 5000.0, 2000.0, 0.0])
        b = jnp.array([0.0, 0.3, 0.6, 1.0])
        hc = _checked_hybrid(a, b)
        self.assertEqual(np.asarray(hc.a_boundaries).shape, (4,))


def _l95_subset(indices=(0, 10, 20, 30, 40, 55, 70, 80, 90, 95)):
    """Return a thinned table made of L95's own interfaces (float64, TOA first)."""
    full = get_echam_levels(95)
    idx = list(indices)
    return (np.asarray(full.a_boundaries, np.float64)[idx],
            np.asarray(full.b_boundaries, np.float64)[idx])


class UserLevelTableTest(unittest.TestCase):
    """``hybrid_levels_from_table`` / ``read_vct_file`` / ``levels_from_level_table``."""

    def _builtin_arrays(self, nlev):
        hc = get_echam_levels(nlev)
        return (np.asarray(hc.a_boundaries, np.float64),
                np.asarray(hc.b_boundaries, np.float64))

    def test_inline_table_equal_to_builtin_builds_identical_coordinate(self):
        for nlev in (47, 95):
            a, b = self._builtin_arrays(nlev)
            built_in = get_echam_levels(nlev)
            user = levels_from_level_table({"a": a.tolist(), "b": b.tolist()},
                                           layers=nlev)
            np.testing.assert_array_equal(np.asarray(user.a_boundaries),
                                          np.asarray(built_in.a_boundaries))
            np.testing.assert_array_equal(np.asarray(user.b_boundaries),
                                          np.asarray(built_in.b_boundaries))
            self.assertEqual(user.layers, nlev)
            self.assertTrue(is_builtin_table(user))

    def test_file_table_equal_to_builtin_builds_identical_coordinate(self):
        import tempfile
        from pathlib import Path

        import xarray as xr

        a, b = self._builtin_arrays(47)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vct.nc"
            xr.Dataset({"vct_a": ("nvclev", a), "vct_b": ("nvclev", b)}).to_netcdf(path)
            from_file = levels_from_level_table(str(path), layers=47)
            np.testing.assert_array_equal(
                np.asarray(from_file.a_boundaries),
                np.asarray(get_echam_levels(47).a_boundaries))
            np.testing.assert_array_equal(
                np.asarray(from_file.b_boundaries),
                np.asarray(get_echam_levels(47).b_boundaries))

    def test_file_without_vct_variables_raises(self):
        import tempfile
        from pathlib import Path

        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_vct.nc"
            xr.Dataset({"hyam": ("lev", np.zeros(3))}).to_netcdf(path)
            with self.assertRaisesRegex(ValueError, "no vct_a, vct_b"):
                read_vct_file(path)

    def test_thinned_table_is_not_builtin_whatever_its_count(self):
        a, b = _l95_subset()
        levels = hybrid_levels_from_table(a, b)
        self.assertEqual(levels.layers, 9)
        self.assertFalse(is_builtin_table(levels))
        # A different table that happens to have 47 layers is still foreign.
        a47, b47 = self._builtin_arrays(47)
        b47 = b47.copy()
        b47[-2] -= 1e-3
        self.assertFalse(is_builtin_table(hybrid_levels_from_table(a47, b47)))

    def test_pyses_regularised_top_is_still_the_builtin_table(self):
        """The pySES table is the built-in one with a finite top interface."""
        from dinosaur.hybrid_coordinates import HybridCoordinates

        from jcm.dycore.pyses.coords import full_echam_hybrid
        from jcm.ozone_climatology import _check_level_axis
        for nlev in (47, 95):
            a, b = full_echam_hybrid(nlev)
            pyses = HybridCoordinates(a_boundaries=jnp.asarray(a, jnp.float32),
                                      b_boundaries=jnp.asarray(b, jnp.float32))
            self.assertGreater(float(pyses.a_boundaries[0]), 0.0)
            self.assertTrue(is_builtin_table(pyses))
        # ...so a bundle on the built-in L47 levels is accepted for it.
        import xarray as xr
        from importlib import resources
        path = resources.files("jcm") / "data" / "bc" / "t63" / "ozone.nc"
        a, b = full_echam_hybrid(47)
        pyses47 = HybridCoordinates(a_boundaries=jnp.asarray(a, jnp.float32),
                                    b_boundaries=jnp.asarray(b, jnp.float32))
        with xr.open_dataset(path) as ds:
            _check_level_axis(ds, str(path), pyses47, required=True)

    def test_layers_must_equal_table_length(self):
        a, b = _l95_subset()
        with self.assertRaisesRegex(ValueError, "grid.layers=47 .* 9 layers"):
            levels_from_level_table({"a": a, "b": b}, layers=47)
        # null/absent layers take the count from the table.
        self.assertEqual(levels_from_level_table({"a": a, "b": b}).layers, 9)

    def test_malformed_tables_raise(self):
        a, b = _l95_subset()
        cases = {
            "unequal length": (a, b[:-1], "a has 10 entries but b has 9"),
            "too short": (a[:2], b[:2], "at least 3"),
            "not finite": (np.where(np.arange(a.size) == 3, np.nan, a), b, "finite"),
            "surface first": (a[::-1], b[::-1], "SURFACE-FIRST"),
            "truncated bottom": (a[:-1], b[:-1], r"a\[-1\]=0"),
            "normalised a": (a / 101325.0 + 1.0, b, r"a\[-1\]=0"),
            "negative b": (a, np.where(np.arange(b.size) == 1, -0.1, b), "within"),
            "b decreasing": (a, np.where(np.arange(b.size) == 6, b[7] + 0.1, b),
                             "non-decreasing"),
            "pressure not increasing": (np.where(np.arange(a.size) == 3, a[4] + 5.0, a),
                                        b, "does not increase strictly"),
            "truncated top": (*_l95_subset((70, 80, 90, 95)), "truncated vct"),
        }
        for name, (aa, bb, message) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, message):
                    hybrid_levels_from_table(aa, bb)

    def test_2d_table_and_unknown_inline_keys_raise(self):
        a, b = _l95_subset()
        with self.assertRaisesRegex(ValueError, "1-D"):
            hybrid_levels_from_table(a[None], b[None])
        with self.assertRaisesRegex(ValueError, r"\{a: \[\.\.\.\], b: \[\.\.\.\]\}"):
            levels_from_level_table({"a": a, "bb": b})
        with self.assertRaisesRegex(ValueError, "netCDF"):
            levels_from_level_table(47)


if __name__ == "__main__":
    unittest.main()
