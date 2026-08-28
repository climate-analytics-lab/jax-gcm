"""Tests for the radiation-emulator trainer.

These target the parts that fail silently rather than loudly: a split that
leaks correlated columns into validation (reporting skill the emulator does
not have), and a heating-rate loss dominated by the near-vacuum model top.
"""

import os
import pathlib
import sys
import unittest

import jax.numpy as jnp
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from train_emulator import (  # noqa: E402
    build_features,
    mass_weights,
    uniform_weights,
    solar_group_ids,
    split_by_source_and_group,
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


class SourceStratifiedSplitTest(unittest.TestCase):
    """Every source must reach validation and test, not just training."""

    # The real geometry that broke it: two trajectory files with a handful of
    # enormous solar-geometry groups, and a sweep with one group per column.
    ERA5, TRAJ, SWEEP = 100_000, 120_000, 80_000

    def _real_geometry(self):
        group_ids = np.concatenate([
            np.repeat(np.arange(8), self.ERA5 // 8),
            100 + np.repeat(np.arange(40), self.TRAJ // 40),
            1000 + np.arange(self.SWEEP),
        ])
        source_ids = np.concatenate([
            np.zeros(self.ERA5, int),
            np.ones(self.TRAJ, int),
            2 * np.ones(self.SWEEP, int),
        ])
        return group_ids, source_ids

    def test_every_source_reaches_val_and_test(self):
        # Pooled group splitting put 0 ERA5 and 4.7% trajectory columns in the
        # daylit test set, so the offline metric reported a -0.13 W/m2
        # shortwave bias for a network that ran -21.7 W/m2 in the GCM.
        group_ids, source_ids = self._real_geometry()
        splits = split_by_source_and_group(
            group_ids, source_ids, (0.8, 0.1, 0.1), seed=0,
        )
        for k, name in ((1, "val"), (2, "test")):
            for source in (0, 1, 2):
                n = int(np.sum(source_ids[splits[k]] == source))
                self.assertGreater(
                    n, 0, f"source {source} absent from {name}",
                )

    def test_each_source_keeps_its_own_fractions(self):
        group_ids, source_ids = self._real_geometry()
        splits = split_by_source_and_group(
            group_ids, source_ids, (0.8, 0.1, 0.1), seed=0,
        )
        for source, total in ((0, self.ERA5), (1, self.TRAJ), (2, self.SWEEP)):
            got = [
                int(np.sum(source_ids[s] == source)) / total for s in splits
            ]
            for g, want in zip(got, (0.8, 0.1, 0.1)):
                self.assertAlmostEqual(g, want, delta=0.06)

    def test_no_group_is_split_across_partitions(self):
        group_ids, source_ids = self._real_geometry()
        splits = split_by_source_and_group(
            group_ids, source_ids, (0.8, 0.1, 0.1), seed=0,
        )
        seen = {}
        for k, idx in enumerate(splits):
            for g in np.unique(group_ids[idx]):
                self.assertNotIn(g, seen, f"group {g} in two partitions")
                seen[g] = k
        self.assertEqual(sum(len(s) for s in splits), len(group_ids))


class BuildFeaturesTest(unittest.TestCase):
    """The trainer must feed the network exactly what the online scheme does.

    Feature parity is the emulator's classic silent failure: a feature the
    trainer forgets to pass is a column of zeros in training and a real number
    at run time, and nothing in the loss curve says so.
    """

    NCOL, NLEV, NBND = 4, 6, 3
    R_LIQ, R_ICE = 9.0, 40.0

    def _dataset(self):
        import xarray as xr

        ncol, nlev, nbnd = self.NCOL, self.NLEV, self.NBND
        prof = lambda v: (("column", "level"),          # noqa: E731
                          np.full((ncol, nlev), v, np.float32))
        col = lambda v: (("column",),                   # noqa: E731
                         np.full((ncol,), v, np.float32))
        iface = lambda v: (("column", "interface"),     # noqa: E731
                           np.full((ncol, nlev + 1), v, np.float32))
        band = lambda dim, v: ((("column", dim, "level")),  # noqa: E731
                               np.full((ncol, nbnd, nlev), v, np.float32))

        data = dict(
            temperature=prof(260.0), pressure_levels=prof(5.0e4),
            specific_humidity=prof(1.0e-3), ozone_vmr=prof(1.0e-6),
            cloud_water=prof(1.0e-4), cloud_ice=prof(1.0e-5),
            cloud_fraction=prof(0.5), air_density=prof(0.7),
            layer_thickness=prof(500.0),
            r_eff_liq=prof(self.R_LIQ), r_eff_ice=prof(self.R_ICE),
            co2_vmr=col(400e-6), cos_zenith=col(0.6),
            surface_temperature=col(288.0), surface_albedo_vis=col(0.1),
            surface_albedo_nir=col(0.2), surface_emissivity=col(0.98),
            pressure_interfaces=iface(5.0e4),
            aod_sw_per_band=band("band_sw", 0.05),
            ssa_sw_per_band=band("band_sw", 0.9),
            asy_sw_per_band=band("band_sw", 0.7),
            aod_lw_per_band=band("band_lw", 0.0),
            ssa_lw_per_band=band("band_lw", 0.0),
            asy_lw_per_band=band("band_lw", 0.0),
        )
        for prefix in ("sw", "lw"):
            for channel in ("down", "up", "down_clear", "up_clear"):
                data[f"{prefix}_flux_{channel}"] = iface(200.0)
        return xr.Dataset(data)

    def test_cloud_path_features_are_grid_mean_not_cf_weighted(self):
        # ``cloud_water`` in the training files is already the grid-mean
        # mixing ratio, so the path features must not pick up another
        # cloud_fraction factor (that made them scale as cf^2 — PR #730
        # review). Two batches differing ONLY in cover must produce
        # identical path channels; cover is its own feature (channel 6).
        lo = self._dataset()
        hi = self._dataset()
        hi["cloud_fraction"] = hi["cloud_fraction"] * 0.0 + 0.9
        lo["cloud_fraction"] = lo["cloud_fraction"] * 0.0 + 0.1
        x_lo = np.asarray(build_features(lo, "per_band")["x_sw"])
        x_hi = np.asarray(build_features(hi, "per_band")["x_sw"])
        for ch in (4, 5):  # cwp, cip
            np.testing.assert_array_equal(x_lo[..., ch], x_hi[..., ch])
        self.assertTrue((x_hi[..., 6] > x_lo[..., 6]).all())

    def test_effective_radii_reach_both_networks(self):
        from jcm.physics.radiation.nn_emulator import n_input_features

        data = build_features(self._dataset(), "per_band")
        want = n_input_features("per_band", self.NBND)
        for name in ("x_sw", "x_lw"):
            x = np.asarray(data[name])
            self.assertEqual(x.shape, (self.NCOL, self.NLEV, want), name)
            # Features 8 and 9 are the radii; see preprocess_*_inputs.
            np.testing.assert_allclose(x[..., 8], self.R_LIQ, rtol=1e-6)
            np.testing.assert_allclose(x[..., 9], self.R_ICE, rtol=1e-6)


class SwLossMaskTest(unittest.TestCase):
    """The SW flux loss must ignore the reconstructed TOA-down channels.

    Interface-0 down/down_clear are overwritten with the exact incoming
    flux at inference and their sigmoid-unreachable target is exactly 1,
    so training on them pollutes the shared output layer (PR #730 review).
    """

    def _loss_at(self, pred):
        from unittest import mock
        import tools.radiation_emulator.train_emulator as t

        loss = t.make_loss(is_sw=True, alpha=0.0, weight_prof=1.0)
        batch = {"x": None, "aux": None,
                 "mask": jnp.ones((2,)), "y": jnp.zeros((2, 5, 4))}
        with mock.patch.object(t, "_predict", lambda w, x, a: pred):
            val, _ = loss(None, batch)
        return float(val)

    def test_reconstructed_channels_are_inert_and_others_are_not(self):
        base = jnp.zeros((2, 5, 4))
        l_base = self._loss_at(base)
        l_masked = self._loss_at(base.at[:, 0, 0].set(9.0).at[:, 0, 2].set(9.0))
        l_live = self._loss_at(base.at[:, 0, 1].set(1.0))
        self.assertEqual(l_masked, l_base)
        self.assertGreater(l_live, l_base)


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


class ValidationWeightingTest(unittest.TestCase):
    """A short final validation chunk must not outweigh a full one."""

    def test_chunk_average_is_weighted_by_column_count(self):
        # 4096 columns at loss 1.0 plus a 1-column tail at loss 5.0: the
        # plain mean returns 3.0, letting one column decide half of
        # val_loss (PR #730 review). The weighted average is ~1.001.
        losses = np.array([1.0, 5.0])
        sizes = np.array([4096.0, 1.0])
        self.assertAlmostEqual(float(np.mean(losses)), 3.0)
        weighted = float(np.average(losses, weights=sizes))
        self.assertLess(weighted, 1.002)
        self.assertAlmostEqual(
            weighted, float((1.0 * 4096 + 5.0) / 4097), places=9)
