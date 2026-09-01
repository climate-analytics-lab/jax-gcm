"""Tests for the radiation-emulator trainer.

These target the parts that fail silently rather than loudly: a split that
leaks correlated columns into validation (reporting skill the emulator does
not have), and a heating-rate loss dominated by the near-vacuum model top.
"""

import os
import pathlib
import sys
import unittest

import jax
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
    hybrid_objective,
    band_metrics,
    flux_to_heating_rate,
    SECONDS_PER_DAY,
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
        # scale/p_half are part of the batch contract that _loss_terms reads;
        # alpha=0 drops the heating term but the numerators are still traced.
        batch = {"x": None, "aux": None,
                 "mask": jnp.ones((2,)), "y": jnp.zeros((2, 5, 4)),
                 "scale": jnp.ones((2,)),
                 "p_half": jnp.asarray([[1.0, 100.0, 5e3, 5e4, 1e5]] * 2)}
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


class BandMetricsSWBoundaryTest(unittest.TestCase):
    """SW metrics must score the reconstructed profile, as deployed."""

    @staticmethod
    def _data(nlev=3, toa=1000.0):
        # TOA-first interfaces with a near-vacuum top layer, as at T63L47.
        p_half = np.array([[1.0, 100.0, 5e4, 1.0e5]])
        down = np.array([toa, 900.0, 800.0, 700.0])
        up = np.full(nlev + 1, 100.0)
        labels = np.stack([down, up, down, up], axis=-1)[None]
        return dict(sw_scale=np.array([toa]), sw_labels=labels,
                    lit=np.array([True]), pressure_interfaces=p_half)

    def test_unreachable_toa_down_is_not_charged_as_error(self):
        # A sigmoid cannot emit exactly 1, so the raw output undershoots the
        # incoming flux; the deployed path overwrites that interface, and an
        # evaluator that does not would charge 20 W/m^2 across a 99 Pa layer.
        data = self._data()
        pred_norm = data["sw_labels"] / 1000.0
        pred_norm[0, 0, 0] = 0.98          # 980 vs 1000 W/m^2 at TOA down
        m = band_metrics(pred_norm, data, np.array([0]), True)
        self.assertLess(m["heating_rmse_worst_level"], 1e-6)

        # What the unreconstructed comparison would have scored instead --
        # score() ranks --sweep candidates on exactly this number.
        raw = np.asarray(jax.vmap(flux_to_heating_rate)(
            jnp.asarray(pred_norm[..., 0] * 1000.0),
            jnp.asarray(pred_norm[..., 1] * 1000.0),
            data["pressure_interfaces"])) * SECONDS_PER_DAY
        true = np.asarray(jax.vmap(flux_to_heating_rate)(
            jnp.asarray(data["sw_labels"][..., 0]),
            jnp.asarray(data["sw_labels"][..., 1]),
            data["pressure_interfaces"])) * SECONDS_PER_DAY
        self.assertGreater(np.abs(raw - true).max(), 100.0)

    def test_genuine_interior_error_is_still_scored(self):
        # The override must not be a blanket excuse: an error one interface
        # down is real and must survive.
        data = self._data()
        pred_norm = data["sw_labels"] / 1000.0
        pred_norm[0, 1, 0] = 0.80          # 800 vs 900 W/m^2 at interface 1
        m = band_metrics(pred_norm, data, np.array([0]), True)
        self.assertGreater(m["heating_rmse_worst_level"], 1.0)

    def test_longwave_is_unaffected(self):
        # LW has no normalising incoming flux and no boundary override.
        data = self._data()
        data = dict(lw_scale=data["sw_scale"], lw_labels=data["sw_labels"],
                    pressure_interfaces=data["pressure_interfaces"])
        pred_norm = data["lw_labels"] / 1000.0
        pred_norm[0, 0, 0] = 0.98
        m = band_metrics(pred_norm, data, np.array([0]), False)
        self.assertGreater(m["heating_rmse_worst_level"], 100.0)


class ObjectiveAggregationTest(unittest.TestCase):
    """The validation objective must not depend on how the split is chunked.

    The heating term is an RMSE and ``sqrt`` is nonlinear, so averaging
    per-chunk objectives -- by any per-chunk weight -- is not the split-wide
    objective. ``_loss_terms`` returns the raw numerators and
    ``hybrid_objective`` roots once over the accumulated totals, which removes
    the dependence on chunk boundaries (PR #730 review).
    """

    def test_hybrid_objective_roots_the_pooled_sums_not_per_chunk(self):
        # Two equal chunks with heating SSEs 0 and 8 over n = 2 each: the true
        # split-wide heating RMSE is sqrt((0 + 8) / 4) = sqrt(2) ~ 1.414. The
        # old sample-weighted average of per-chunk RMSEs gives
        # (2*sqrt(0/2) + 2*sqrt(8/2)) / 4 = 1.0 -- the value this fix rejects.
        alpha, n_elem = 1.0, 4          # flux term zeroed out via flux_sse = 0
        pooled = hybrid_objective(alpha, hr_sse=8.0, flux_sse=0.0,
                                  n=4.0, n_elem=n_elem)
        self.assertAlmostEqual(pooled, np.sqrt(2.0), places=6)
        per_chunk = (2 * np.sqrt(0.0 / 2) + 2 * np.sqrt(8.0 / 2)) / 4
        self.assertAlmostEqual(per_chunk, 1.0, places=6)
        self.assertNotAlmostEqual(pooled, per_chunk, places=2)

    def _aggregate(self, chunks, is_sw, alpha):
        # _predict is mocked to echo the batch's stored prediction, so
        # splitting the rows into chunks is a pure partition of the same data.
        from unittest import mock
        import tools.radiation_emulator.train_emulator as t

        with mock.patch.object(t, "_predict", lambda w, x, a: x):
            terms = [t._loss_terms(is_sw, 1.0, None, b) for b in chunks]
        hr = sum(float(x[0]) for x in terms)
        fl = sum(float(x[1]) for x in terms)
        n = sum(float(x[2]) for x in terms)
        return hybrid_objective(alpha, hr, fl, n, int(terms[0][3]))

    def test_loss_terms_aggregate_identically_across_partitions(self):
        # A longwave split of 6 columns: evaluated whole vs split into an
        # unequal 1 + 5 must give the identical objective, whereas the previous
        # per-chunk-averaged val_loss did not.
        rng = np.random.default_rng(0)
        ncol, nlev = 6, 4
        pred = jnp.asarray(rng.uniform(0.2, 0.8, (ncol, nlev + 1, 4)))
        y = jnp.asarray(rng.uniform(0.2, 0.8, (ncol, nlev + 1, 4)))
        p_half = jnp.asarray(np.sort(
            rng.uniform(1.0, 1.0e5, (ncol, nlev + 1)), axis=1))
        scale = jnp.asarray(rng.uniform(200.0, 500.0, (ncol,)))
        mask = jnp.ones((ncol,))

        def batch(sl):
            return {"x": pred[sl], "aux": None, "y": y[sl],
                    "scale": scale[sl], "p_half": p_half[sl], "mask": mask[sl]}

        whole = self._aggregate([batch(slice(0, 6))], False, alpha=1e-3)
        split = self._aggregate(
            [batch(slice(0, 1)), batch(slice(1, 6))], False, alpha=1e-3)
        self.assertAlmostEqual(whole, split, places=6)


if __name__ == "__main__":
    unittest.main()
