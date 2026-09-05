"""Tests for multi-year offline dataset assembly (synthetic cache, no model)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jcm.physics.bias_correction.era5_cache import (
    STATE_FIELDS,
    _era5_path,
    _manifest_path,
    _save_atomic,
    _states_path,
    cache_meta,
)
from jcm.physics.bias_correction.era5_data import (
    VARS_3D,
    model_clock_seconds,
)
from jcm.physics.bias_correction.multiyear import (
    climatology_starts,
    offline_dataset,
    parse_year_span,
    rollout_starts,
    year_pairs,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# Small but structurally faithful: 8 levels (the feature vector is 4*nlev) on a
# tiny horizontal grid, with the real 5-slots-per-save cadence.
NLEV, NLON, NLAT = 8, 4, 3
N_FRAMES, N_SLOTS = 12, 60


def _write_year(root, year, meta):
    """Write a cached year whose values encode their own frame/slot index.

    Encoding the index lets a test assert *which* frame was paired with
    *which* target, which is the only property that actually matters here.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    slot_shape = (N_SLOTS, NLEV, NLON, NLAT)
    era5 = {}
    for i, v in enumerate(VARS_3D):
        a = np.arange(N_SLOTS, dtype=np.float32)[:, None, None, None]
        era5[f"era5__{v}"] = np.broadcast_to(
            a + 1000.0 * i, slot_shape).astype(np.float32)
    _save_atomic(_era5_path(root, year),
                 time_seconds=model_clock_seconds(year, N_SLOTS, 6), **era5)

    frame_shape = (N_FRAMES, NLEV, NLON, NLAT)
    states = {}
    for name in STATE_FIELDS:
        a = np.arange(N_FRAMES, dtype=np.float32)[:, None, None, None]
        states[f"state__{name}"] = np.broadcast_to(
            a, frame_shape).astype(np.float32)
    _save_atomic(_states_path(root, year), **states)
    _manifest_path(root, year).write_text(json.dumps(meta, indent=2))


class TestYearPairs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.meta = cache_meta(
            get_speedy_coords(layers=8, spectral_truncation=21),
            cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
            time_step_minutes=30.0, spinup_days=10.0)
        _write_year(self.root, 2001, self.meta)

    def tearDown(self):
        self._tmp.cleanup()

    def test_row_count_is_kept_frames_times_columns(self):
        # 12 frames want slots 5,10,...,60; slot 60 does not exist, so 11
        # survive. With spin-up disabled, stride 1 keeps all 11.
        feats, targets = year_pairs(self.root, 2001, meta=self.meta,
                                    frame_stride=1, spinup_days=0.0)
        self.assertEqual(feats.shape, (11 * NLON * NLAT, 4 * NLEV))
        self.assertEqual(targets.shape, feats.shape)

    def test_stride_subsamples_frames(self):
        feats, _ = year_pairs(self.root, 2001, meta=self.meta, frame_stride=4,
                              spinup_days=0.0)
        # 11 usable frames, every 4th -> frames 0, 4, 8.
        self.assertEqual(feats.shape[0], 3 * NLON * NLAT)

    def test_spinup_frames_are_dropped(self):
        # Each cached year is its own nudged run from bootstrap, so its
        # opening frames are a convergence transient the free-running term
        # never sees. 2.5 days at 1.25 d/frame drops the first two.
        feats, _ = year_pairs(self.root, 2001, meta=self.meta,
                              frame_stride=1, spinup_days=2.5)
        self.assertEqual(feats.shape[0], (11 - 2) * NLON * NLAT)

    def test_spinup_defaults_to_the_cached_value(self):
        # The cache records spinup_days=10; at 1.25 d/frame that is 8 frames,
        # leaving 3 of the 11 usable ones.
        feats, _ = year_pairs(self.root, 2001, meta=self.meta, frame_stride=1)
        self.assertEqual(feats.shape[0], (11 - 8) * NLON * NLAT)

    def test_rejects_a_spinup_that_consumes_the_year(self):
        with self.assertRaises(ValueError):
            year_pairs(self.root, 2001, meta=self.meta, frame_stride=1,
                       spinup_days=999.0)

    def test_features_come_from_the_state_not_the_target(self):
        # Frame values are 0..11, ERA5 values are 0..59 (+1000*field). If the
        # two were ever swapped the features would carry the target's range,
        # which a shape check alone would not notice.
        feats, _ = year_pairs(self.root, 2001, meta=self.meta, frame_stride=1)
        self.assertLessEqual(float(feats.max()), float(N_FRAMES))

    def test_targets_are_finite_and_nonzero(self):
        # State and target differ by construction, so a temp/humidity nudge
        # must produce a non-zero tendency somewhere.
        _, targets = year_pairs(self.root, 2001, meta=self.meta,
                                frame_stride=1)
        self.assertTrue(np.isfinite(targets).all())
        self.assertGreater(float(np.abs(targets).max()), 0.0)


class TestParseYearSpan(unittest.TestCase):
    def test_range_and_list(self):
        self.assertEqual(parse_year_span("1995-1998"), [1995, 1996, 1997, 1998])
        self.assertEqual(parse_year_span("2001,2003"), [2001, 2003])
        self.assertEqual(parse_year_span("2001-2002,2005"),
                         [2001, 2002, 2005])

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            parse_year_span("2005-2001")
        with self.assertRaises(ValueError):
            parse_year_span("")


class TestClimatologyStarts(unittest.TestCase):
    """Start selection across years, and the window-fits-the-year guard."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.meta = cache_meta(
            get_speedy_coords(layers=8, spectral_truncation=21),
            cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
            time_step_minutes=30.0, spinup_days=2.5)
        self.years = [2001, 2002, 2003]
        for year in self.years:
            _write_year(self.root, year, self.meta)

    def tearDown(self):
        self._tmp.cleanup()

    def test_pool_cycles_years_before_advancing_the_day(self):
        # The point of the multi-year pool: one seasonal window seen in many
        # years, not many seasons of one year. A pool the size of the span
        # must therefore hit every year at the SAME frame.
        starts = climatology_starts(self.root, self.years, meta=self.meta,
                                    n_starts=3, win_slots=5)
        self.assertEqual([r["year"] for r in starts], [2001, 2002, 2003])
        self.assertEqual({r["frame"] for r in starts}, {2})

    def test_larger_pool_then_steps_the_day_forward(self):
        starts = climatology_starts(self.root, self.years, meta=self.meta,
                                    n_starts=6, win_slots=5)
        self.assertEqual([r["year"] for r in starts],
                         [2001, 2002, 2003, 2001, 2002, 2003])
        self.assertEqual([r["frame"] for r in starts], [2, 2, 2, 3, 3, 3])

    def test_days_puts_one_window_at_each_listed_day(self):
        # The whole point: the default pool cannot reach a second season
        # because the frame advances 1.25 days per year-cycle. An explicit day
        # list must place a window at each day, still cycling years within a
        # day so every season keeps many realisations.
        # day = (frame + 1) * save_days, so 3.75 -> 2, 6.25 -> 4, 8.75 -> 6.
        starts = climatology_starts(self.root, self.years, meta=self.meta,
                                    n_starts=9, win_slots=5,
                                    days=[3.75, 6.25, 8.75])
        # Returned in cycle order: all years of the first day, then the next.
        self.assertEqual([r["frame"] for r in starts],
                         [2, 2, 2, 4, 4, 4, 6, 6, 6])
        self.assertEqual([r["year"] for r in starts],
                         [2001, 2002, 2003, 2001, 2002, 2003,
                          2001, 2002, 2003])
        # Each day appears once per year: many realisations, one window.
        for frame in (2, 4, 6):
            self.assertEqual(sum(1 for r in starts if r["frame"] == frame), 3)

    def test_repeating_a_day_doubles_its_share_of_the_pool(self):
        # Weighting is by sampling frequency, not a loss multiplier: the driver
        # runs one start per update and Adam divides the step by sqrt(v), so
        # scaling one update's loss scales gradient and second moment together
        # and barely moves the step. Appearing twice per epoch really is 2x.
        starts = climatology_starts(self.root, self.years, meta=self.meta,
                                    n_starts=9, win_slots=5,
                                    days=[3.75, 3.75, 6.25])
        counts = {f: sum(1 for r in starts if r["frame"] == f)
                  for f in {r["frame"] for r in starts}}
        self.assertEqual(counts[2], 6)   # day 3.75 listed twice
        self.assertEqual(counts[4], 3)   # day 6.25 listed once

    def test_repeated_day_copies_are_spread_not_adjacent(self):
        # A repeated day is how a season gets weighted, and the two copies must
        # land at their own slots in the cycle. Keying the sort on
        # (year, frame) collapsed them, so the driver -- which walks the pool
        # round-robin, one start per update -- saw the identical start on two
        # consecutive updates instead of once per half-epoch.
        starts = climatology_starts(self.root, self.years, meta=self.meta,
                                    n_starts=6, win_slots=5,
                                    days=[3.75, 3.75])
        pairs = [(r["year"], r["frame"]) for r in starts]
        self.assertEqual(pairs, [(2001, 2), (2002, 2), (2003, 2),
                                 (2001, 2), (2002, 2), (2003, 2)])
        # Adjacent entries must differ, which is what "spread" means here.
        self.assertTrue(all(pairs[i] != pairs[i + 1]
                            for i in range(len(pairs) - 1)))

    def test_day_inside_the_spinup_raises(self):
        # Silently clamping into the spinup would train on states the nudged
        # run had not settled, which is exactly the class of quiet error the
        # cache metadata checks exist to prevent.
        with self.assertRaises(ValueError):
            climatology_starts(self.root, self.years, meta=self.meta,
                               n_starts=3, win_slots=5, days=[1.25])

    def test_days_none_is_the_shipped_behaviour(self):
        # Backward compatibility: every result to date came from this path.
        a = climatology_starts(self.root, self.years, meta=self.meta,
                               n_starts=6, win_slots=5)
        b = climatology_starts(self.root, self.years, meta=self.meta,
                               n_starts=6, win_slots=5, days=None)
        self.assertEqual([r["frame"] for r in a], [r["frame"] for r in b])
        self.assertEqual([r["year"] for r in a], [r["year"] for r in b])

    def test_sim_time_encodes_day_of_year_not_the_year(self):
        # The model never learns which calendar year a start came from: the
        # ocean forcing is a repeating climatology and insolation depends only
        # on fraction-of-year, so identical frames must give identical
        # sim_time across years.
        starts = climatology_starts(self.root, self.years, meta=self.meta,
                                    n_starts=3, win_slots=5)
        self.assertEqual(len({r["sim_time"] for r in starts}), 1)
        self.assertAlmostEqual(starts[0]["sim_time"], 3 * 1.25 * 86400.0)

    def test_window_running_past_the_year_end_raises(self):
        # Windows are required to fit inside their own year, which is what
        # keeps the cache read per-year. Silently truncating the mean would
        # weight the start of the window twice.
        with self.assertRaises(ValueError) as cm:
            climatology_starts(self.root, self.years, meta=self.meta,
                               n_starts=1, win_slots=N_SLOTS)
        self.assertIn("past the end of the year", str(cm.exception))

    def test_lead_targets_only_when_requested(self):
        plain = climatology_starts(self.root, self.years, meta=self.meta,
                                   n_starts=1, win_slots=5)
        self.assertNotIn("leadT", plain[0])
        combined = climatology_starts(self.root, self.years, meta=self.meta,
                                      n_starts=1, win_slots=5, lead_slots=2)
        self.assertIn("leadT", combined[0])
        self.assertIn("leadq", combined[0])


class TestRolloutStarts(unittest.TestCase):
    """Curriculum starts: one cache pass serving every stage lead."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.meta = cache_meta(
            get_speedy_coords(layers=8, spectral_truncation=21),
            cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
            time_step_minutes=30.0, spinup_days=2.5)
        self.years = [2001, 2002, 2003]
        for year in self.years:
            _write_year(self.root, year, self.meta)

    def tearDown(self):
        self._tmp.cleanup()

    def test_every_stage_lead_is_collected_in_one_pass(self):
        starts = rollout_starts(self.root, self.years, meta=self.meta,
                                n_starts=3, lead_slots=[1, 2, 4],
                                n_frames=N_FRAMES)
        self.assertEqual(len(starts), 3)
        for r in starts:
            self.assertEqual(sorted(r["targets"]), [1, 2, 4])

    def test_targets_are_the_slot_the_lead_points_at(self):
        # ERA5 values encode their slot index, so a lead of L from start slot
        # s0 must carry the value s0 + L. This is the check that would catch a
        # lead applied in model steps instead of ERA5 slots.
        starts = rollout_starts(self.root, self.years, meta=self.meta,
                                n_starts=1, lead_slots=[1, 4],
                                n_frames=N_FRAMES)
        s0 = int((2 + 1) * 1.25 * 4)          # frame 2 -> slot 15
        # _write_year encodes value = slot + 1000 * index-in-VARS_3D, and
        # temperature is index 2.
        offset = 1000.0 * VARS_3D.index("temperature")
        for lead in (1, 4):
            got = float(np.asarray(starts[0]["targets"][lead][0]).flat[0])
            self.assertEqual(got - offset, float(s0 + lead))

    def test_starts_spread_over_seasons_not_just_years(self):
        # A term trained on one day of year has never seen the rest of the
        # calendar, and the 450-day evaluation free run goes non-finite. Every
        # start must differ in day of year as well as in year.
        starts = rollout_starts(self.root, self.years, meta=self.meta,
                                n_starts=3, lead_slots=[1], n_frames=N_FRAMES)
        self.assertEqual([r["year"] for r in starts], [2001, 2002, 2003])
        frames = [r["frame"] for r in starts]
        self.assertEqual(len(set(frames)), 3, "starts share a day of year")
        self.assertEqual(len(set(r["sim_time"] for r in starts)), 3)

    def test_lead_past_the_year_end_raises(self):
        with self.assertRaises(ValueError) as cm:
            rollout_starts(self.root, self.years, meta=self.meta, n_starts=1,
                           lead_slots=[N_SLOTS], n_frames=N_FRAMES)
        self.assertIn("past the end of the year", str(cm.exception))


class TestOfflineDataset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.meta = cache_meta(
            get_speedy_coords(layers=8, spectral_truncation=21),
            cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
            time_step_minutes=30.0, spinup_days=10.0)
        for year in (2001, 2002, 2003):
            _write_year(self.root, year, self.meta)

    def tearDown(self):
        self._tmp.cleanup()

    def test_concatenates_every_year(self):
        # spinup_days=0 so this measures concatenation alone; the spin-up drop
        # has its own tests.
        feats, targets = offline_dataset(self.root, [2001, 2002, 2003],
                                         meta=self.meta, frame_stride=1,
                                         spinup_days=0.0, verbose=False)
        self.assertEqual(feats.shape[0], 3 * 11 * NLON * NLAT)
        self.assertEqual(targets.shape[0], feats.shape[0])

    def test_spinup_is_dropped_from_every_year_not_just_the_first(self):
        # Each year is an independent run, so each has its own transient.
        feats, _ = offline_dataset(self.root, [2001, 2002, 2003],
                                   meta=self.meta, frame_stride=1,
                                   spinup_days=2.5, verbose=False)
        self.assertEqual(feats.shape[0], 3 * (11 - 2) * NLON * NLAT)

    def test_refuses_a_year_cached_with_other_settings(self):
        # Blending products built under different nudging strengths would mix
        # incompatible labels, so the mismatch must surface here.
        _write_year(self.root, 2004, dict(self.meta, tau_seconds=43200.0))
        with self.assertRaises(ValueError):
            offline_dataset(self.root, [2001, 2004], meta=self.meta,
                            frame_stride=1, verbose=False)


if __name__ == "__main__":
    unittest.main()
