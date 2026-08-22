"""Tests for the RRTMGP emulator training-data generator.

These guard the two things a training set can get silently wrong: labels
that are the right shape but physically inconsistent (clear-sky brighter
than all-sky at the surface), and labels that carry more McICA sampling
noise than the generator claims (seed averaging not actually averaging
independent draws).

Every RRTMGP-driving test shares one column count and level count so the
XLA compilation (~20 s) is paid once for the module.
"""

import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from generate_training_data import (  # noqa: E402
    LABEL_FIELDS,
    PROFILE_FIELDS,
    _clip_to_bounds,
    _finalize_batch,
    _latin_hypercube,
    _orient_toa_first,
    _per_band_optics,
    _solar_geometry_for_cos_zenith,
    band_counts,
    build_dataset,
    expand_state_files,
    generate,
    label_batch,
    label_quality_mask,
    make_labeller,
    perturbation_sweep,
)

NCOL = 8
NLEV = 10

_BANDS = None
_LABELLER = None


def _bands():
    global _BANDS
    if _BANDS is None:
        _BANDS = band_counts()
    return _BANDS


def _labeller():
    global _LABELLER
    if _LABELLER is None:
        _LABELLER = make_labeller(0)
    return _LABELLER


def _sweep_batch(seed=0, n_columns=NCOL):
    n_bnd_sw, n_bnd_lw, sw_centers, lw_centers = _bands()
    rng = np.random.default_rng(seed)
    raw = perturbation_sweep(
        n_columns, NLEV, rng, n_bnd_sw, n_bnd_lw, sw_centers, lw_centers,
    )
    return _finalize_batch(raw, n_bnd_sw, n_bnd_lw)


def _cloudy_sunlit_batch(cloud_fraction=0.6):
    """Build a sweep batch with a thick low cloud deck under a high sun.

    The clear-vs-all-sky and seed-averaging tests both need columns that
    are certainly cloudy and certainly lit; the LHS sweep only guarantees
    that in the aggregate. The fraction stays PARTIAL because an overcast
    (f = 1) deck makes every McICA sub-column identical, which would zero
    the sampling noise the averaging test is about.
    """
    batch = _sweep_batch(seed=1)
    ncol = batch["temperature"].shape[0]
    slab = np.zeros((ncol, NLEV))
    slab[:, -3:] = 1.0
    batch["cloud_fraction"] = slab * cloud_fraction
    batch["cloud_water"] = batch["cloud_fraction"] * 5.0e-4
    batch["cloud_ice"] = batch["cloud_fraction"] * 1.0e-4
    rng = np.random.default_rng(7)
    (batch["latitude"], batch["longitude"], batch["orbital_phase"],
     batch["synodic_phase"], _) = _solar_geometry_for_cos_zenith(
        rng, np.full(ncol, 0.8),
    )
    return batch


class ColumnSourceTest(unittest.TestCase):
    """Source-side sampling and layout, no RRTMGP required."""

    def test_latin_hypercube_covers_every_stratum(self):
        rng = np.random.default_rng(0)
        u = _latin_hypercube(rng, 50, 4)
        self.assertEqual(u.shape, (50, 4))
        for d in range(4):
            # One sample per 1/n stratum on every axis is the property
            # that makes LHS cheaper than an outer product.
            strata = np.floor(u[:, d] * 50).astype(int)
            self.assertEqual(len(np.unique(strata)), 50)

    def test_sweep_spans_the_requested_physical_ranges(self):
        batch = _sweep_batch(seed=2, n_columns=512)
        self.assertGreater(batch["surface_albedo_vis"].max(), 0.6)
        self.assertLess(batch["surface_albedo_vis"].min(), 0.15)
        self.assertGreater(batch["surface_temperature"].max(), 310.0)
        self.assertLess(batch["surface_temperature"].min(), 230.0)
        self.assertGreater(batch["cloud_fraction"].max(), 0.95)
        self.assertEqual(batch["cloud_fraction"].min(), 0.0)
        aod = batch["aod_sw_per_band"].sum(axis=2).max(axis=1)
        self.assertGreater(aod.max() / aod.min(), 100.0)

    def test_solar_geometry_hits_the_target_cos_zenith(self):
        rng = np.random.default_rng(0)
        target = np.linspace(0.05, 1.0, 64)
        lat, _, _, _, mu = _solar_geometry_for_cos_zenith(rng, target)
        np.testing.assert_allclose(mu, target, atol=2e-3)
        # Latitude must not be a deterministic function of mu0, or the
        # emulator can read the answer off the coordinate.
        self.assertLess(abs(np.corrcoef(lat, mu)[0, 1]), 0.95)

    def test_orientation_uses_each_pressure_arrays_own_order(self):
        # JCM output can carry surface-first full levels alongside
        # TOA-first interfaces; both must land TOA-first.
        batch = {
            "pressure_levels": np.array([[1000.0, 500.0, 100.0]]),
            "temperature": np.array([[290.0, 260.0, 230.0]]),
            "pressure_interfaces": np.array([[50.0, 300.0, 700.0, 1013.0]]),
        }
        out = _orient_toa_first(batch, aux={})
        np.testing.assert_array_equal(
            out["pressure_levels"], [[100.0, 500.0, 1000.0]])
        np.testing.assert_array_equal(
            out["temperature"], [[230.0, 260.0, 290.0]])
        np.testing.assert_array_equal(
            out["pressure_interfaces"], [[50.0, 300.0, 700.0, 1013.0]])


class SweepCoverageTest(unittest.TestCase):
    """The sweep must cover the states the coupled model actually visits.

    A coupled run failed because it did not: the sweep's grid stopped at
    100 Pa while the model top is 1 Pa, and its single tropospheric lapse
    rate floored at 190 K piled every top-level column on the floor. The
    trajectory source covered 242-252 K there, so the training distribution
    was bimodal with a hole at 205-240 -- exactly where the live model sat.
    """

    def _sweep(self, n=2000):
        return perturbation_sweep(
            n, NLEV, np.random.default_rng(0), 14, 16,
            np.linspace(300.0, 4000.0, 14), np.linspace(4000.0, 50000.0, 16))

    def test_pressure_grid_reaches_the_model_top(self):
        p = self._sweep(64)["pressure_levels"][0]
        self.assertLessEqual(p[0], 1.0)
        self.assertGreater(p[-1], 1.0e5)
        self.assertTrue(np.all(np.diff(p) > 0), "must stay TOA-first")

    def test_model_top_temperature_is_spread_not_piled(self):
        t_top = self._sweep()["temperature"][:, 0]
        self.assertGreater(t_top.std(), 10.0)
        # No single value may claim a large share: that is the floor bug.
        _, counts = np.unique(np.round(t_top, 1), return_counts=True)
        self.assertLess(counts.max() / t_top.size, 0.05)

    def test_model_top_brackets_the_coupled_model_range(self):
        # Measured from a T63L47 ERA5-initialised run: 204.5 to 245.7 K.
        t_top = self._sweep()["temperature"][:, 0]
        self.assertLess(t_top.min(), 204.5)
        self.assertGreater(t_top.max(), 245.7)

    def test_profile_has_a_tropopause_minimum_and_warm_stratopause(self):
        b = self._sweep(512)
        t, p = b["temperature"], b["pressure_levels"][0]
        k_strat = int(np.argmin(np.abs(p - 1.0e2)))
        k_trop = int(np.argmin(np.abs(p - 2.0e4)))
        # Coldest point aloft is the tropopause, not the surface or the
        # stratopause -- the shape ozone heating produces.
        self.assertLess(t[:, k_trop].mean(), t[:, -1].mean())
        self.assertLess(t[:, k_trop].mean(), t[:, k_strat].mean())


class LabelTest(unittest.TestCase):
    """RRTMGP-driven label properties."""

    def test_tiny_generation_produces_finite_labels(self):
        batch, labels, quality = generate(
            "perturbation", n_columns=NCOL, nlev=NLEV, n_seeds=2,
            base_seed=0, batch_size=NCOL, rng_seed=0,
        )
        # The synthetic sweep is constructed in-bounds, so nothing should
        # need rejecting; a non-zero count here means the sweep drifted.
        self.assertEqual(quality["n_rejected"], 0)
        for name in LABEL_FIELDS:
            arr = labels[name]
            self.assertEqual(arr.shape, (NCOL, NLEV + 1), name)
            self.assertTrue(np.all(np.isfinite(arr)), name)
            self.assertTrue(np.all(arr >= 0.0), name)
        for name in PROFILE_FIELDS:
            self.assertEqual(batch[name].shape, (NCOL, NLEV), name)
        self.assertEqual(labels["cos_zenith"].shape, (NCOL,))
        # A clear-sky solve must have run: compute_cre=True is what
        # populates these, and the withheld default is exactly zero.
        self.assertGreater(labels["lw_flux_up_clear"].max(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.nc")
            build_dataset(batch, labels, {"source": "perturbation"}).to_netcdf(
                path)
            self.assertTrue(os.path.exists(path))

    def test_clear_sky_surface_sw_exceeds_all_sky_when_cloudy(self):
        batch = _cloudy_sunlit_batch()
        labels, _ = label_batch(batch, n_seeds=4, labeller=_labeller())
        self.assertGreater(labels["total_cloud_cover"].min(), 0.5)
        clear = labels["sw_flux_down_clear"][:, -1]
        all_sky = labels["sw_flux_down"][:, -1]
        self.assertTrue(np.all(clear >= all_sky - 1e-3),
                        f"clear {clear} vs all-sky {all_sky}")
        # And the cloud must actually do something, or the comparison is
        # vacuous.
        self.assertGreater(np.mean(clear - all_sky), 1.0)

    def test_seed_averaging_reduces_label_variance(self):
        batch = _cloudy_sunlit_batch()
        labeller = _labeller()

        def toa_sw_up_mean(n_seeds, block):
            # Disjoint seed blocks so each estimate is independent.
            draws = [
                np.asarray(labeller(batch, block * n_seeds + s)["sw_flux_up"])
                for s in range(n_seeds)
            ]
            return np.mean(draws, axis=0)[:, 0]

        spread_1 = np.std([toa_sw_up_mean(1, b) for b in range(4)], axis=0)
        spread_4 = np.std([toa_sw_up_mean(4, b) for b in range(4)], axis=0)
        self.assertGreater(spread_1.mean(), 1.0,
                           "single-draw McICA noise should be O(W/m2)")
        self.assertLess(spread_4.mean(), 0.75 * spread_1.mean())

    def test_identical_seeds_reproduce_bit_for_bit(self):
        batch = _sweep_batch(seed=3)
        first, _ = label_batch(batch, n_seeds=2, labeller=_labeller())
        second, _ = label_batch(batch, n_seeds=2, labeller=_labeller())
        for name in LABEL_FIELDS:
            np.testing.assert_array_equal(first[name], second[name], name)


class InputSanitisationTest(unittest.TestCase):
    """Guards against out-of-range inputs reaching RRTMGP.

    A 550 nm SSA a whisker above 1 (which ECHAM's aerosol diagnostic does
    produce) flips the sign of the MACv2-SP per-band denominator and yields a
    ~1e21 far-infrared SSA, which drove the LW solver to >1000 W/m2 OLR.
    """

    def test_ssa_above_one_does_not_explode_the_per_band_scaling(self):
        _, _, sw_centers, lw_centers = _bands()
        aod = np.full((2, NLEV), 0.05)
        ssa = np.full((2, NLEV), 1.0 + 3.0e-4)
        asy = np.full((2, NLEV), 0.7)
        angstrom = np.full((2,), 2.0)
        stats = {}
        for centers in (sw_centers, lw_centers):
            _, ssa_band, _ = _per_band_optics(
                aod, ssa, asy, angstrom, centers, stats)
            self.assertTrue(np.all(ssa_band >= 0.0))
            self.assertTrue(np.all(ssa_band <= 1.0),
                            f"max per-band SSA {ssa_band.max():g}")
        self.assertEqual(stats["ssa550"][0], 2 * NLEV * 2)

    def test_finalize_clips_emissivity_above_one(self):
        n_sw, n_lw, _, _ = _bands()
        batch = perturbation_sweep(
            2, NLEV, np.random.default_rng(0), n_sw, n_lw, *_bands()[2:])
        batch["surface_emissivity"] = np.array([1.9, 0.98])
        stats = {}
        out = _finalize_batch(batch, n_sw, n_lw, stats)
        np.testing.assert_allclose(out["surface_emissivity"], [1.0, 0.98])
        self.assertEqual(stats["surface_emissivity"][0], 1)


class StateFileExpansionTest(unittest.TestCase):
    """A model run writes one file per chunk, so a trajectory spans files."""

    def test_expands_globs_and_lists_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ["out_day10.nc", "out_day20.nc", "other.nc"]
            for name in names:
                pathlib.Path(tmp, name).touch()
            self.assertEqual(
                expand_state_files(os.path.join(tmp, "out_day*.nc")),
                [os.path.join(tmp, "out_day10.nc"),
                 os.path.join(tmp, "out_day20.nc")])
            self.assertEqual(
                expand_state_files(f"{tmp}/other.nc, {tmp}/out_day10.nc"),
                [os.path.join(tmp, "other.nc"),
                 os.path.join(tmp, "out_day10.nc")])

    def test_no_state_file_still_drives_one_pass(self):
        self.assertEqual(expand_state_files(None), [None])

    def test_unmatched_pattern_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            expand_state_files("/nonexistent/path/*.nc")


class LabelQualityTest(unittest.TestCase):
    """The last line of defence: reject labels no correct solve can produce."""

    def _good(self):
        batch = {
            "temperature": np.full((3, NLEV), 250.0),
            "surface_temperature": np.full((3,), 280.0),
        }
        labels = {name: np.full((3, NLEV + 1), 100.0)
                  for name in LABEL_FIELDS}
        for name in ("sw_flux_down", "sw_flux_down_clear"):
            labels[name] = np.full((3, NLEV + 1), 300.0)
        return batch, labels

    def test_clean_labels_are_all_kept(self):
        keep, reasons = label_quality_mask(*self._good())
        self.assertTrue(keep.all())
        self.assertEqual(reasons, {})

    def test_rejects_longwave_above_black_body(self):
        batch, labels = self._good()
        # 1089 W/m2 OLR from a 280 K column: the real failure that motivated
        # this check.
        labels["lw_flux_up"][1, 0] = 1089.0
        keep, reasons = label_quality_mask(batch, labels)
        np.testing.assert_array_equal(keep, [True, False, True])
        self.assertEqual(reasons, {"longwave above black body": 1})

    def test_rejects_nonfinite_and_negative_and_overbright_shortwave(self):
        batch, labels = self._good()
        labels["lw_flux_down"][0, 3] = np.nan
        labels["sw_flux_up"][1, 2] = -5.0
        labels["sw_flux_up_clear"][2, 4] = 400.0     # exceeds its down flux
        keep, reasons = label_quality_mask(batch, labels)
        self.assertFalse(keep.any())
        self.assertEqual(set(reasons), {"non-finite flux", "negative flux",
                                        "SW up exceeds SW down"})

    def test_clip_to_bounds_leaves_in_range_arrays_untouched(self):
        stats = {}
        arrays = {"cloud_fraction": np.array([0.0, 0.5, 1.0])}
        out = _clip_to_bounds(arrays, stats)
        np.testing.assert_array_equal(out["cloud_fraction"],
                                      arrays["cloud_fraction"])
        self.assertEqual(stats, {})


if __name__ == "__main__":
    unittest.main()
