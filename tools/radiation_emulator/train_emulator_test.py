"""Tests for the radiation-emulator trainer.

These target the parts that fail silently rather than loudly: a split that
leaks correlated columns into validation (reporting skill the emulator does
not have), and a heating-rate loss dominated by the near-vacuum model top.
"""

import os
import pathlib
import sys
import unittest

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from train_emulator import (  # noqa: E402
    mass_weights,
    uniform_weights,
    solar_group_ids,
    split_by_group,
)


class _FakeDataset:
    """Minimal stand-in exposing the variables solar_group_ids reads."""

    def __init__(self, orbital, synodic):
        import xarray as xr

        self._vars = {
            "orbital_phase": xr.DataArray(np.asarray(orbital)),
            "synodic_phase": xr.DataArray(np.asarray(synodic)),
        }

    def __getitem__(self, name):
        return self._vars[name]


class SolarGroupTest(unittest.TestCase):
    """Columns from one snapshot must be recognised as one group."""

    def test_columns_sharing_a_geometry_form_one_group(self):
        ds = _FakeDataset([1.0, 1.0, 1.0, 2.0, 2.0], [0.5, 0.5, 0.5, 0.25, 0.25])
        ids = solar_group_ids(ds)
        self.assertEqual(ids.shape, (5,))
        self.assertEqual(len(np.unique(ids)), 2)
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[3])

    def test_geometry_randomised_per_column_gives_one_group_each(self):
        rng = np.random.default_rng(0)
        ds = _FakeDataset(rng.random(50), rng.random(50))
        self.assertEqual(len(np.unique(solar_group_ids(ds))), 50)


class SplitTest(unittest.TestCase):
    """The split must hold groups together and still fill every partition."""

    def test_no_group_is_split_across_partitions(self):
        group_ids = np.repeat(np.arange(20), 100)
        splits = split_by_group(group_ids, (0.8, 0.1, 0.1), seed=0)
        seen = {}
        for k, idx in enumerate(splits):
            for g in np.unique(group_ids[idx]):
                self.assertNotIn(g, seen, f"group {g} in two partitions")
                seen[g] = k
        self.assertEqual(sum(len(s) for s in splits), len(group_ids))

    def test_partition_sizes_track_the_requested_fractions(self):
        group_ids = np.repeat(np.arange(100), 50)
        splits = split_by_group(group_ids, (0.8, 0.1, 0.1), seed=1)
        n = len(group_ids)
        for got, want in zip(splits, (0.8, 0.1, 0.1)):
            self.assertAlmostEqual(len(got) / n, want, delta=0.03)

    def test_one_dominant_group_still_leaves_val_and_test_populated(self):
        # The failure this guards: taking groups in random order let a run of
        # small groups fill the training quota, then the one huge group
        # arrived while training was still emptiest and swallowed everything.
        group_ids = np.concatenate([
            np.zeros(600, dtype=int), np.repeat(np.arange(1, 6), 20),
        ])
        splits = split_by_group(group_ids, (0.8, 0.1, 0.1), seed=0)
        for k, idx in enumerate(splits):
            self.assertGreater(len(idx), 0, f"partition {k} is empty")

    def test_too_few_groups_is_a_clear_error_not_a_silent_empty_split(self):
        with self.assertRaises(ValueError) as ctx:
            split_by_group(np.zeros(100, dtype=int), (0.8, 0.1, 0.1), seed=0)
        self.assertIn("empty partition", str(ctx.exception))

    def test_split_is_reproducible_for_a_given_seed(self):
        group_ids = np.repeat(np.arange(30), 10)
        a = split_by_group(group_ids, (0.8, 0.1, 0.1), seed=7)
        b = split_by_group(group_ids, (0.8, 0.1, 0.1), seed=7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


class WeightingTest(unittest.TestCase):
    """The two weightings answer different questions; both are reported."""

    def test_uniform_weights_do_not_discount_the_thin_top_layer(self):
        # The failure that motivated training on uniform weights: mass
        # weighting gave the ~2 Pa top layer ~1e-5 of the loss, the emulator
        # reached 130 K/day there, and the GCM NaN'd in under five days.
        p_half = np.array([[0.0, 2.0, 5e4, 1.0e5]])
        mass = np.asarray(mass_weights(p_half))[0]
        uniform = np.asarray(uniform_weights(p_half))[0]
        self.assertLess(mass[0], 1e-4)
        np.testing.assert_allclose(uniform, 1.0 / 3.0)
        np.testing.assert_allclose(uniform.sum(), 1.0)

    def test_uniform_weights_match_the_layer_count(self):
        p_half = np.zeros((5, 48))
        self.assertEqual(np.asarray(uniform_weights(p_half)).shape, (5, 47))


class MassWeightTest(unittest.TestCase):
    """Mass weighting stays available as the energy-error lens."""

    def test_weights_sum_to_one_per_column(self):
        p_half = np.array([[1.0, 10.0, 1e4, 1e5], [2.0, 20.0, 2e4, 9e4]])
        w = np.asarray(mass_weights(p_half))
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-6)

    def test_thin_top_layer_gets_negligible_weight(self):
        # A T63L47 column: ~1 Pa top layer against a ~1000 hPa surface.
        p_half = np.array([[1.0, 5.0, 5e4, 1.0e5]])
        w = np.asarray(mass_weights(p_half))[0]
        self.assertLess(w[0], 1e-4)
        self.assertGreater(w[-1], 0.4)


if __name__ == "__main__":
    unittest.main()
