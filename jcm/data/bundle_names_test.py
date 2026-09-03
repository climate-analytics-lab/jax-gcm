"""Unit tests for the pure ``bundle_is_published`` availability predicate.

The module is import-free (loaded by file path from ``tools/benchmark.py``), so
the predicate is tested directly here rather than only through the runner.
"""

import unittest

from jcm.data import bundle_names


class TestBundleIsPublished(unittest.TestCase):
    def test_unpublished_horizontal_grid_never_published(self):
        # A grid outside PUBLISHED_GRIDS has no bundle for ANY key, regardless
        # of level count or vertical family.
        for key in bundle_names.EMISSION_AUTO_BUNDLES:
            self.assertFalse(
                bundle_names.bundle_is_published(key, "t42", 47, "hybrid"))

    def test_level_free_keys_ignore_level_and_vertical(self):
        # emissions/dms/dust are purely horizontal: on a published grid they
        # resolve at any layer count AND any vertical (incl. sigma).
        for key in ("emissions_file", "dms_file", "dust_file"):
            for nlev in (8, 47):
                for vertical in ("hybrid", "sigma"):
                    self.assertTrue(
                        bundle_names.bundle_is_published(
                            key, "t63", nlev, vertical),
                        (key, nlev, vertical))

    def test_oxidants_require_published_level_and_hybrid(self):
        # The level-dependent oxidants bundle needs a published layer count AND
        # a hybrid vertical.
        self.assertTrue(bundle_names.bundle_is_published(
            "oxidants_file", "t63", 47, "hybrid"))
        # Unpublished layer count nulls even on hybrid (t63_l8 gap).
        self.assertFalse(bundle_names.bundle_is_published(
            "oxidants_file", "t63", 8, "hybrid"))
        # Published (token, nlev) but SIGMA vertical nulls — the round-14 hole:
        # the bundle is on hybrid-level pressures.
        self.assertFalse(bundle_names.bundle_is_published(
            "oxidants_file", "t63", 47, "sigma"))

    def test_vertical_defaults_to_hybrid(self):
        # Level-free callers (and hybrid grids) need not pass ``vertical``.
        self.assertTrue(bundle_names.bundle_is_published(
            "oxidants_file", "t63", 47))
        self.assertTrue(bundle_names.bundle_is_published(
            "dms_file", "t63", 8))


if __name__ == "__main__":
    unittest.main()
