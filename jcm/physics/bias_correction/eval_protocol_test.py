"""Tests for the evaluation reference-year protocol.

Every reported number in the paper, poster, deck and docx comes from one
protocol: a 450-day free run from 1 January 2001, first 90 days discarded,
scored against a 2001-only ERA5 reference with 2002 as the out-of-sample spot
check. When the evaluation was generalised so a held-out period could be
scored, the reference span started being derived from the run length, which
silently turned the canonical reference into a 2001-2002 average. Nothing
failed, because a cached reference file masked it.

These tests pin the rule so that cannot recur.
"""

import unittest

import numpy as np
import xarray as xr

from jcm.physics.bias_correction.eval_protocol import (
    CANONICAL_REF2_YEAR,
    CANONICAL_REF_YEARS,
    LAST_FULL_YEAR,
    SeasonMeans,
    chunk_spans,
    is_canonical,
    period_suffix,
    reference_label,
    resolve_reference,
    scoring_blocks,
    too_short_to_score,
)


class TestCanonicalProtocol(unittest.TestCase):
    """The default must reproduce every published number, exactly."""

    def test_default_uses_a_2001_only_reference(self):
        refs, ref2, canonical = resolve_reference(2001, 450.0)
        self.assertTrue(canonical)
        self.assertEqual(refs, (2001, 2001))
        self.assertEqual(ref2, 2002)

    def test_the_naive_derivation_would_have_differed(self):
        # Guards the guard. A 450-day run really does spill into 2002, so
        # deriving the span from the run length gives (2001, 2002). If that
        # ever stops being true the pin above has quietly become a no-op and
        # this suite would pass for the wrong reason.
        derived, _, _ = resolve_reference(2001, 450.0, ref_years_env=None,
                                          last_full_year=LAST_FULL_YEAR)
        naive_hi = min(2001 + int((450.0 - 1) // 365), LAST_FULL_YEAR)
        self.assertEqual(naive_hi, 2002)
        self.assertNotEqual(derived, (2001, naive_hi))

    def test_canonical_constants_match_what_is_published(self):
        self.assertEqual(CANONICAL_REF_YEARS, (2001, 2001))
        self.assertEqual(CANONICAL_REF2_YEAR, 2002)

    def test_canonical_period_takes_no_filename_suffix(self):
        # eval_out/ holds the canonical products under unsuffixed names; a
        # suffix here would orphan every one of them.
        self.assertEqual(period_suffix(2001, 450.0), "")

    def test_a_different_period_is_suffixed(self):
        self.assertEqual(period_suffix(2016, 2645.0), "_2016_2645d")
        self.assertEqual(period_suffix(2001, 900.0), "_2001_900d")

    def test_a_different_period_is_not_canonical(self):
        for year, days in ((2001, 900.0), (2016, 450.0), (2002, 450.0)):
            self.assertFalse(is_canonical(year, days),
                             f"{year}/{days} claimed canonical")

    def test_int_days_still_recognised_as_canonical(self):
        # M4_DAYS parses to float, but a caller passing 450 must not silently
        # fall through to the derived branch.
        self.assertTrue(is_canonical(2001, 450))


class TestHoldoutProtocol(unittest.TestCase):
    """The 2016-2022 holdout, which does want a multi-year reference."""

    def test_holdout_spans_the_scored_years(self):
        refs, ref2, canonical = resolve_reference(2016, 2645.0)
        self.assertFalse(canonical)
        self.assertEqual(refs, (2016, 2022))
        # The store ends in 2023, so no year is left for a spot check.
        self.assertIsNone(ref2)

    def test_the_2023_stub_is_never_averaged_into_a_reference(self):
        # Without the clamp the upper bound would be 2023, whose ten days
        # would bias the reference toward early January.
        refs, _, _ = resolve_reference(2016, 2645.0)
        self.assertLessEqual(refs[1], LAST_FULL_YEAR)

    def test_explicit_override_wins(self):
        refs, ref2, _ = resolve_reference(2016, 2645.0,
                                          ref_years_env="2016-2020",
                                          ref2_year_env="2021")
        self.assertEqual(refs, (2016, 2020))
        self.assertEqual(ref2, 2021)

    def test_a_single_year_override_parses(self):
        refs, _, _ = resolve_reference(2016, 2645.0, ref_years_env="2018")
        self.assertEqual(refs, (2018, 2018))

    def test_spot_check_can_be_disabled(self):
        _, ref2, _ = resolve_reference(2001, 450.0, ref2_year_env="none")
        self.assertIsNone(ref2)

    def test_override_beats_the_canonical_pin(self):
        # An explicit request must win even on the canonical period, so the
        # 2001-2002 reference remains reachable deliberately.
        refs, _, canonical = resolve_reference(2001, 450.0,
                                               ref_years_env="2001-2002")
        self.assertTrue(canonical)
        self.assertEqual(refs, (2001, 2002))


class TestReferenceLabel(unittest.TestCase):
    def test_single_year_and_span_render_differently(self):
        self.assertEqual(reference_label((2001, 2001)), "2001")
        self.assertEqual(reference_label((2016, 2022)), "2016-2022")


class TestScoringBlocks(unittest.TestCase):
    """Cutting one long run into windows that are comparable to each other."""

    def test_the_holdout_length_yields_no_blocks(self):
        # One block is the run itself and measures no spread, so a 7-year run
        # asked for 7-year blocks must return nothing rather than a single
        # window that would print a spread of zero.
        self.assertEqual(scoring_blocks(2645.0), ())

    def test_three_times_the_holdout_gives_three_blocks(self):
        blocks = scoring_blocks(90.0 + 21 * 365)
        self.assertEqual(len(blocks), 3)

    def test_the_first_block_is_exactly_the_holdout_window(self):
        # This is what makes a long run a check on the short one: block 0
        # covers the same days the 2645-day protocol scored, so it should
        # reproduce that number for the same term.
        self.assertEqual(scoring_blocks(90.0 + 21 * 365)[0], (90.0, 2645.0))

    def test_blocks_are_contiguous_and_equal(self):
        blocks = scoring_blocks(90.0 + 21 * 365)
        spans = {hi - lo for lo, hi in blocks}
        self.assertEqual(spans, {7 * 365.0})
        for (_, hi), (lo, _) in zip(blocks, blocks[1:]):
            self.assertEqual(hi, lo)

    def test_a_short_remainder_is_left_unscored(self):
        # A trailing stub folded into the last block would give that block a
        # different seasonal composition from the others, which is exactly the
        # comparison the blocks exist to support.
        blocks = scoring_blocks(90.0 + 17 * 365)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[-1][1], 90.0 + 14 * 365)

    def test_block_length_is_configurable(self):
        blocks = scoring_blocks(90.0 + 21 * 365, block_years=3)
        self.assertEqual(len(blocks), 7)
        self.assertEqual(blocks[0], (90.0, 90.0 + 3 * 365))


class TestTooShortToScore(unittest.TestCase):
    """A run with no winter in it is a mistake, not a result."""

    def test_the_canonical_protocol_is_scoreable(self):
        # 360 scored days from 1 April reaches every day of the year once.
        self.assertIsNone(too_short_to_score(450.0))

    def test_the_holdout_and_a_long_run_are_scoreable(self):
        self.assertIsNone(too_short_to_score(2645.0))
        self.assertIsNone(too_short_to_score(90.0 + 21 * 365))

    def test_a_run_with_no_full_year_is_refused(self):
        msg = too_short_to_score(100.0)
        self.assertIsNotNone(msg)
        # The message has to say what to do about it, since this fires before
        # the model runs and the alternative is a NaN column after it.
        self.assertIn("M4_DAYS", msg)

    def test_the_spin_up_drop_counts_against_the_span(self):
        # 400 days is more than a year, but not after 90 are discarded.
        self.assertIsNotNone(too_short_to_score(400.0))
        self.assertIsNone(too_short_to_score(400.0, drop_days=0.0))


class TestChunkSpans(unittest.TestCase):
    """Splitting a run into segments must not change what is integrated."""

    def test_no_chunking_requested_means_one_segment(self):
        self.assertEqual(chunk_spans(450.0, 0, 5.0), (450.0,))
        self.assertEqual(chunk_spans(450.0, None, 5.0), (450.0,))

    def test_segments_sum_to_the_full_run(self):
        for total in (2645.0, 7755.0, 3650.0):
            spans = chunk_spans(total, 1825.0, 5.0)
            self.assertAlmostEqual(sum(spans), total, places=9)

    def test_segments_are_equal_to_within_one_save_interval(self):
        # Each distinct segment length costs a compile of the integrator, so
        # spans differing by more than one save interval would be a waste.
        spans = chunk_spans(7755.0, 1825.0, 5.0)
        self.assertLessEqual(max(spans) - min(spans), 5.0)

    def test_every_segment_is_a_whole_number_of_save_intervals(self):
        # The model computes outer_steps as int(total_time / save_interval), so
        # a fractional segment silently drops its tail.
        for span in chunk_spans(7755.0, 1000.0, 5.0):
            self.assertAlmostEqual(span % 5.0, 0.0, places=9)

    def test_a_run_length_that_would_truncate_is_refused(self):
        with self.assertRaises(ValueError):
            chunk_spans(452.0, 0, 5.0)


def _fake_run(days, nlev=2, nlon=4, nlat=3, offset=0.0):
    """Build a dataset shaped like ModelPredictions.to_xarray, contents known."""
    days = np.asarray(days, dtype=float)
    shape = (len(days), nlev, nlon, nlat)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape)
    values = base + days.reshape(-1, 1, 1, 1) + offset
    dims = ("time", "level", "lon", "lat")
    return xr.Dataset(
        {"temperature": (dims, values),
         "specific_humidity": (dims, values * 0.001)},
        coords={"time": days, "level": np.arange(nlev),
                "lon": np.arange(nlon) * 90.0,
                "lat": np.linspace(-60, 60, nlat)},
    )


class TestSeasonMeans(unittest.TestCase):
    """The reduction a segmented run depends on."""

    def setUp(self):
        # 5-day samples, the cadence the evaluation saves at, over a span
        # that splits into two equal blocks after a 90-day spin-up drop.
        self.days = np.arange(1, 165) * 5.0
        self.ds = _fake_run(self.days)
        self.windows = (("", 90.0, 820.0), ("_blk0", 90.0, 455.0),
                        ("_blk1", 455.0, 820.0))

    def test_segmented_matches_single_segment(self):
        # The claim that makes long runs possible at all.
        whole = SeasonMeans(self.windows)
        whole.add(self.ds, self.days)

        pieces = SeasonMeans(self.windows)
        for lo, hi in ((0, 50), (50, 51), (51, 164)):
            part = self.ds.isel(time=slice(lo, hi))
            pieces.add(part, self.days[lo:hi])

        for season in ("annual", "djf", "jja"):
            np.testing.assert_allclose(
                pieces.mean("", season, "temperature").values,
                whole.mean("", season, "temperature").values, rtol=1e-12,
                err_msg=f"{season} mean depends on the segmentation")
            self.assertEqual(pieces.samples("", season),
                             whole.samples("", season))

    def test_spin_up_is_excluded(self):
        acc = SeasonMeans((("", 90.0, 820.0),))
        acc.add(self.ds, self.days)
        kept = int(((self.days >= 90.0) & (self.days < 820.0)).sum())
        self.assertEqual(acc.samples(""), kept)
        self.assertLess(kept, len(self.days))

    def test_blocks_partition_the_scored_span(self):
        acc = SeasonMeans(self.windows)
        acc.add(self.ds, self.days)
        self.assertEqual(acc.samples("_blk0") + acc.samples("_blk1"),
                         acc.samples(""))

    def test_blocks_of_equal_length_see_equal_sample_counts(self):
        acc = SeasonMeans(self.windows)
        acc.add(self.ds, self.days)
        for season in ("annual", "djf", "jja"):
            self.assertEqual(acc.samples("_blk0", season),
                             acc.samples("_blk1", season),
                             f"blocks disagree on {season} sample count")

    def test_a_block_mean_differs_from_the_full_window(self):
        # Guards the guard: the synthetic field trends with time, so identical
        # block and full-window means would mean the windows are not being
        # applied at all.
        acc = SeasonMeans(self.windows)
        acc.add(self.ds, self.days)
        self.assertFalse(np.allclose(acc.mean("_blk0", "annual", "temperature"),
                                     acc.mean("", "annual", "temperature")))

    def test_humidity_is_accumulated_annually_only(self):
        acc = SeasonMeans((("", 90.0, 820.0),))
        acc.add(self.ds, self.days)
        acc.mean("", "annual", "specific_humidity")
        with self.assertRaises(KeyError):
            acc.mean("", "djf", "specific_humidity")

    def test_an_empty_window_raises_rather_than_returning_nan(self):
        acc = SeasonMeans((("_late", 5000.0, 6000.0),))
        acc.add(self.ds, self.days)
        self.assertEqual(acc.samples("_late"), 0)
        with self.assertRaises(KeyError):
            acc.mean("_late", "annual", "temperature")

    def test_djf_wraps_the_year_boundary(self):
        acc = SeasonMeans((("", 0.0, 820.0),))
        acc.add(self.ds, self.days)
        inside = self.days < 820.0
        doy = self.days[inside] % 365.0
        expected = int(((doy >= 334) | (doy < 59)).sum())
        self.assertEqual(acc.samples("", "djf"), expected)
        self.assertGreater(expected, 0)


if __name__ == "__main__":
    unittest.main()
