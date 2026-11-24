"""Tests for the time-mean safety fix (jax-gcm#1079) in aerocom_cmor.py.

These build the kind of per-step sequence ``CloudsatCosp`` would emit under
``output_averages`` (some night steps, some fully-attenuated-layer steps),
average them exactly as ``jcm.model.integrate`` does (a plain arithmetic
mean of every field, companions included), and check that running the
resulting one-frame Dataset through ``convert()`` recovers the
CFMIP/CMIP-defined conditional mean -- not the plain time mean. No jcosp
dependency: this only exercises the post-processing math, with synthetic
numbers standing in for jcosp's outputs.
"""

import contextlib
import io
import pathlib
import tempfile
import unittest

import numpy as np
import xarray as xr

from tools.aerocom_cmor import convert


def _capture_stderr():
    """Build a fresh ``redirect_stderr`` context manager over a new buffer."""
    return contextlib.redirect_stderr(io.StringIO())


def _mean_frame(steps: dict[str, np.ndarray]) -> xr.Dataset:
    """Plain time mean of each per-step field, as a single (1, 1) frame.

    ``steps[name]`` is a 1-D array of per-step values; the returned
    Dataset holds ``mean(steps[name])`` on a trivial (lat=1, lon=1) grid,
    exactly what ``output_averages`` would have produced for a single
    gridpoint.
    """
    return xr.Dataset({
        name: xr.DataArray(np.full((1, 1), float(np.mean(vals))),
                           dims=("lat", "lon"))
        for name, vals in steps.items()
    })


def _write_and_read(ds: xr.Dataset, var: str) -> float:
    with tempfile.TemporaryDirectory() as td:
        written, _ = convert(ds, "JCM-t", "all_2000", "2010", "monthly",
                             pathlib.Path(td))
        fname = [f for f in written if f"_{var}_" in f]
        assert len(fname) == 1, (var, written)
        got = xr.open_dataset(pathlib.Path(td) / fname[0])
        return float(got[var].values[0, 0])


class CalipsoValidWeightedMeanTest(unittest.TestCase):
    """cllcalipso/.../cltcalipso: GOCCP excludes undefined profiles and
    POOLS valid-subcolumn counts rather than averaging per-step ratios.

    The term stores the field pre-multiplied by that step's valid
    fraction (``cllcalipso_n``, "nsublayer/ncolumns" -- see
    ``cosp_cloudsat.py::_calipso``), so these fixtures build the PRODUCT
    directly: ``cllcalipso`` = (fraction of valid subcolumns that are
    cloudy) x (fraction of subcolumns that are valid at all).
    """

    def test_recovers_valid_only_mean(self):
        # 4 steps, binary validity (every subcolumn in a step is either
        # all valid or all attenuated): in-layer cover 0.6 and 0.2 on the
        # two valid steps, the two attenuated steps contribute 0 to both
        # series (the term already zeros those via _defined()).
        cover = [0.6, 0.0, 0.2, 0.0]
        valid = [1.0, 0.0, 1.0, 0.0]
        ds = _mean_frame({"cllcalipso": cover, "cllcalipso_n": valid})
        got = _write_and_read(ds, "cllcalipso")
        # Valid-only mean of (0.6, 0.2) = 0.4, in percent.
        self.assertAlmostEqual(got, 40.0, places=5)

    def test_plain_mean_would_have_been_wrong(self):
        """Sanity check that the naive (pre-fix) mean differs materially."""
        cover = [0.6, 0.0, 0.2, 0.0]
        valid = [1.0, 0.0, 1.0, 0.0]
        plain_mean_pct = 100.0 * np.mean(cover)   # 20.0
        ds = _mean_frame({"cllcalipso": cover, "cllcalipso_n": valid})
        got = _write_and_read(ds, "cllcalipso")
        self.assertNotAlmostEqual(got, plain_mean_pct, places=3)
        self.assertAlmostEqual(got, 40.0, places=5)

    def test_pools_valid_subcolumn_counts_not_per_step_ratios(self):
        """Regression test for a Codex finding on this PR.

        Two steps, both "valid" (nsublayer > 0) but with DIFFERENT
        fractions of subcolumns actually usable: step 1 has an in-layer
        cover of 0.5 among its valid subcolumns but only 40% of
        subcolumns were valid at all; step 2 has cover 1.0 with 100%
        valid. GOCCP's pooled estimate is sum(frac*n)/sum(n) = (0.5*0.4 +
        1.0*1.0)/(0.4 + 1.0) = 6/7 (~85.7%), NOT the naive equal-weight
        average of the two ratios (0.5 + 1.0)/2 = 75% a bare 0/1 validity
        flag (treating both steps as "equally valid") would have given.
        """
        weighted_cover = [0.5 * 0.4, 1.0 * 1.0]  # the term's pre-weighted field
        valid_frac = [0.4, 1.0]
        ds = _mean_frame({
            "cllcalipso": weighted_cover, "cllcalipso_n": valid_frac})
        got = _write_and_read(ds, "cllcalipso")
        self.assertAlmostEqual(got, 100.0 * 6.0 / 7.0, places=5)
        naive_binary_pct = 100.0 * 0.75
        self.assertNotAlmostEqual(got, naive_binary_pct, places=3)

    def test_missing_companion_falls_back_with_warning(self):
        cover = [0.6, 0.0, 0.2, 0.0]
        ds = _mean_frame({"cllcalipso": cover})  # no _n companion
        with _capture_stderr() as err:
            got = _write_and_read(ds, "cllcalipso")
        self.assertIn("jax-gcm#1079", err.getvalue())
        # Falls back to the plain (biased) mean, in percent.
        self.assertAlmostEqual(got, 100.0 * np.mean(cover), places=5)

    def test_all_undefined_is_nan_not_zero(self):
        """A layer that is NEVER valid must report missing, not 0 cover."""
        cover = [0.0, 0.0]
        valid = [0.0, 0.0]
        ds = _mean_frame({"cllcalipso": cover, "cllcalipso_n": valid})
        got = _write_and_read(ds, "cllcalipso")
        self.assertTrue(np.isnan(got))


class ModisSunlitDaytimeMeanTest(unittest.TestCase):
    """cltmodis/clwmodis/climodis/cltisccp: CFMIP daytime means."""

    def test_recovers_daytime_mean(self):
        # 4 steps: 2 night (field forced to 0 by the term's night masking),
        # 2 day with cover 0.5 and 0.3.
        clt = [0.0, 0.5, 0.0, 0.3]
        sunlit = [0.0, 1.0, 0.0, 1.0]
        ds = _mean_frame({"cltmodis": clt, "cosp_sunlit": sunlit})
        got = _write_and_read(ds, "cltmodis")
        # Daytime-only mean of (0.5, 0.3) = 0.4, in percent.
        self.assertAlmostEqual(got, 40.0, places=5)

    def test_plain_mean_is_biased_by_sunlit_fraction(self):
        clt = [0.0, 0.5, 0.0, 0.3]
        sunlit = [0.0, 1.0, 0.0, 1.0]
        plain_mean_pct = 100.0 * np.mean(clt)  # 20.0 (half the daytime value)
        ds = _mean_frame({"cltmodis": clt, "cosp_sunlit": sunlit})
        got = _write_and_read(ds, "cltmodis")
        self.assertNotAlmostEqual(got, plain_mean_pct, places=3)

    def test_histogram_channel_recovers_daytime_mean(self):
        """The joint histograms get the same correction, per channel.

        ``clmodis`` needs all 7x7 = 49 ``clmodis.<i>`` channels present
        before ``_collect_histograms`` reassembles it (same requirement
        ``cosp_cloudsat_test.HistogramCmorTest`` exercises); each channel
        here follows the same night/day-with-two-different-values pattern
        as the scalar test above, with a distinct per-channel magnitude.
        """
        n_bins = 49
        sunlit = [0.0, 1.0, 0.0, 1.0]
        rng = np.random.default_rng(0)
        day_a = rng.uniform(0.01, 0.1, size=n_bins)
        day_b = rng.uniform(0.01, 0.1, size=n_bins)
        steps = {"cosp_sunlit": sunlit}
        for i in range(n_bins):
            steps[f"clmodis.{i}"] = [0.0, day_a[i], 0.0, day_b[i]]
        ds = _mean_frame(steps)
        with tempfile.TemporaryDirectory() as td:
            written, _ = convert(ds, "JCM-t", "all_2000", "2010", "monthly",
                                 pathlib.Path(td))
            fname = [f for f in written if "_clmodis_" in f]
            self.assertEqual(len(fname), 1)
            got = xr.open_dataset(pathlib.Path(td) / fname[0])
        expected_pct = 100.0 * (day_a + day_b) / 2.0  # daytime-only mean
        np.testing.assert_allclose(
            got["clmodis"].values.reshape(-1), expected_pct, atol=1e-5)

    def test_missing_sunlit_companion_falls_back_with_warning(self):
        clt = [0.0, 0.5, 0.0, 0.3]
        ds = _mean_frame({"cltmodis": clt})  # no cosp_sunlit companion
        with _capture_stderr() as err:
            got = _write_and_read(ds, "cltmodis")
        self.assertIn("jax-gcm#1079", err.getvalue())
        self.assertAlmostEqual(got, 100.0 * np.mean(clt), places=5)


class ModisInCloudRatioTest(unittest.TestCase):
    """reffclwmodis/tauwmodis/lwpmodis (+ice): CMIP in-cloud means.

    ``cosp_sunlit`` is included in every "new format" fixture below even
    though these tests are not about day/night: the term only ever emits
    the pre-weighted ``reffclwmodis``/etc. together with ``cosp_sunlit``
    (both gated on ``enable_modis or enable_isccp``), so a real post-fix
    file always has it, and the fallback tests below rely on exactly that.
    """

    def test_recovers_in_cloud_mean(self):
        # 4 steps: liquid cloud present (cf=0.4) on 2 steps with Reff 1e-5 m
        # and 2e-5 m; absent (cf=0, weighted product also 0) the other 2.
        clwmodis = [0.0, 0.4, 0.0, 0.4]
        reff_weighted = [0.0, 1e-5 * 0.4, 0.0, 2e-5 * 0.4]
        ds = _mean_frame({
            "reffclwmodis": reff_weighted, "clwmodis": clwmodis,
            "cosp_sunlit": [1.0, 1.0, 1.0, 1.0]})
        got = _write_and_read(ds, "reffclwmodis")
        # In-cloud mean of (1e-5, 2e-5) = 1.5e-5 m (units pass through as-is).
        self.assertAlmostEqual(got, 1.5e-5, places=10)

    def test_plain_mean_is_presence_weighted_not_in_cloud(self):
        clwmodis = [0.0, 0.4, 0.0, 0.4]
        reff_weighted = [0.0, 1e-5 * 0.4, 0.0, 2e-5 * 0.4]
        plain_mean = np.mean(reff_weighted)  # 3e-6, not the in-cloud 1.5e-5
        ds = _mean_frame({
            "reffclwmodis": reff_weighted, "clwmodis": clwmodis,
            "cosp_sunlit": [1.0, 1.0, 1.0, 1.0]})
        got = _write_and_read(ds, "reffclwmodis")
        self.assertNotAlmostEqual(got, plain_mean, places=8)

    def test_night_steps_cancel_without_a_separate_sunlit_division(self):
        """The ratio is self-correcting for night.

        Adding more all-zero (night) steps to both series must not move
        the recovered value, even though ``cosp_sunlit`` here is NOT all
        ones -- its presence only gates whether the ratio is applied at
        all (see ``test_weight_present_without_sunlit_marker_is_not_
        divided`` below); the ratio itself needs no sunlit division.
        """
        clwmodis = [0.0, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0]
        reff_weighted = [0.0, 1e-5 * 0.4, 0.0, 2e-5 * 0.4, 0.0, 0.0, 0.0]
        sunlit = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ds = _mean_frame({
            "reffclwmodis": reff_weighted, "clwmodis": clwmodis,
            "cosp_sunlit": sunlit})
        got = _write_and_read(ds, "reffclwmodis")
        self.assertAlmostEqual(got, 1.5e-5, places=10)

    def test_weight_present_without_sunlit_marker_is_not_divided(self):
        """Regression test for a Codex finding on this PR.

        A pre-jax-gcm#1079 file has ``clwmodis`` (it always existed) but
        NOT ``cosp_sunlit`` (brand new) even though ``reffclwmodis`` there
        is still the OLD, unweighted, bare in-cloud mean. ``clwmodis``'s
        mere presence must NOT be read as "this is the new weighted
        format" -- that would divide a historical mean by cloud fraction
        and inflate it, silently corrupting old data instead of falling
        back.
        """
        reff_bare = [4e-6, 8e-6]  # NOT pre-weighted -- the old format.
        clwmodis = [0.4, 0.5]
        ds = _mean_frame({
            "reffclwmodis": reff_bare, "clwmodis": clwmodis})
        with _capture_stderr() as err:
            got = _write_and_read(ds, "reffclwmodis")
        self.assertIn("jax-gcm#1079", err.getvalue())
        self.assertIn("cosp_sunlit", err.getvalue())
        # Passed through unmodified -- NOT divided by clwmodis.
        self.assertAlmostEqual(got, np.mean(reff_bare), places=10)

    def test_missing_weight_companion_falls_back_with_warning(self):
        """``cosp_sunlit`` present but the weight itself absent.

        Should not happen from the term, but the code must not crash.
        """
        reff_weighted = [0.0, 4e-6, 0.0, 8e-6]
        ds = _mean_frame({
            "reffclwmodis": reff_weighted,
            "cosp_sunlit": [0.0, 1.0, 0.0, 1.0]})  # no clwmodis
        with _capture_stderr() as err:
            got = _write_and_read(ds, "reffclwmodis")
        self.assertIn("jax-gcm#1079", err.getvalue())
        self.assertAlmostEqual(got, np.mean(reff_weighted), places=10)

    def test_never_cloudy_is_nan_not_zero(self):
        ds = _mean_frame({
            "reffclwmodis": [0.0, 0.0], "clwmodis": [0.0, 0.0],
            "cosp_sunlit": [1.0, 1.0]})
        got = _write_and_read(ds, "reffclwmodis")
        self.assertTrue(np.isnan(got))


class CompanionSourcesNotReportedAsUnmappedTest(unittest.TestCase):
    """The #1079 companions are consumed, not a coverage gap."""

    def test_companions_absent_from_skipped(self):
        ds = _mean_frame({
            "cltcalipso": [0.5], "cltcalipso_n": [1.0],
            "cltmodis": [0.3], "cosp_sunlit": [1.0],
            "reffclwmodis": [4e-6], "clwmodis": [0.4],
        })
        with tempfile.TemporaryDirectory() as td:
            _, skipped = convert(ds, "JCM-t", "all_2000", "2010", "monthly",
                                 pathlib.Path(td))
        for companion in ("cltcalipso_n", "cosp_sunlit"):
            self.assertNotIn(companion, skipped)


if __name__ == "__main__":
    unittest.main()
