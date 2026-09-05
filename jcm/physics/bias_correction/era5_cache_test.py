"""Tests for the per-year ERA5 cache (no model run, no cloud access).

The expensive half of `build_year` is a full nudged model year, which is a
server job rather than a unit test. What is tested here is the part that can
silently corrupt a multi-hour build: completion marking, resumability, and
the refusal to mix products built with different settings.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jcm.physics.bias_correction.era5_cache import (
    CACHE_VERSION,
    DAYS_PER_MODEL_YEAR,
    STATE_FIELDS,
    _era5_path,
    _manifest_path,
    _save_atomic,
    _states_path,
    cache_meta,
    is_cached,
    load_year,
    state_slot_indices,
)
from jcm.physics.bias_correction.era5_data import (
    check_year_is_complete,
    model_clock_seconds,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords

NT, NLEV, NLON, NLAT = 4, 8, 64, 32
VARS_3D = ("u_wind", "v_wind", "temperature", "specific_humidity")


def _fake_cached_year(root, year, meta, *, write_manifest=True):
    """Lay down a year the way `build_year` does, without running a model."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    shape = (NT, NLEV, NLON, NLAT)
    fields = {v: np.full(shape, float(i), dtype=np.float32)
              for i, v in enumerate(VARS_3D)}
    # A realistic clock: seconds since 1970 for a 2001 start, ~9.8e8. Using a
    # small arange here would hide the precision requirement the cache relies
    # on (see test_clock_survives_the_round_trip_exactly).
    seconds = model_clock_seconds(year, NT, 6)
    _save_atomic(_era5_path(root, year), time_seconds=seconds,
                 **{f"era5__{k}": v for k, v in fields.items()})
    _save_atomic(_states_path(root, year),
                 **{f"state__{n}": np.full(shape, 1.0, dtype=np.float32)
                    for n in STATE_FIELDS})
    if write_manifest:
        _manifest_path(root, year).write_text(json.dumps(meta, indent=2))
    return fields, seconds


class TestCacheMeta(unittest.TestCase):
    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)

    def test_records_the_grid_and_the_run_settings(self):
        meta = cache_meta(self.coords, cadence_hours=6, save_days=1.25,
                          tau_seconds=21600.0, time_step_minutes=30.0,
                          spinup_days=10.0)
        self.assertEqual(meta["cache_version"], CACHE_VERSION)
        self.assertEqual(meta["layers"], 8)
        self.assertEqual((meta["nlon"], meta["nlat"]),
                         tuple(int(x) for x in self.coords.horizontal.nodal_shape))
        self.assertEqual(meta["cadence_hours"], 6)

    def test_a_different_grid_produces_different_metadata(self):
        # The guard that stops a T31 product being read by a T63 run, which
        # previously surfaced as all-NaN scores rather than an error.
        t31 = cache_meta(get_speedy_coords(layers=8, spectral_truncation=31),
                         cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
                         time_step_minutes=30.0, spinup_days=10.0)
        t21 = cache_meta(self.coords, cadence_hours=6, save_days=1.25,
                         tau_seconds=21600.0, time_step_minutes=30.0,
                         spinup_days=10.0)
        self.assertNotEqual(t31, t21)


class TestCompletionAndResume(unittest.TestCase):
    """A multi-hour build must be safe to interrupt and restart."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.meta = cache_meta(
            get_speedy_coords(layers=8, spectral_truncation=21),
            cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
            time_step_minutes=30.0, spinup_days=10.0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_complete_year_is_reported_cached(self):
        _fake_cached_year(self.root, 2001, self.meta)
        self.assertTrue(is_cached(self.root, 2001))
        self.assertTrue(is_cached(self.root, 2001, self.meta))

    def test_missing_year_is_not_cached(self):
        self.assertFalse(is_cached(self.root, 1999))

    def test_arrays_without_a_manifest_are_not_cached(self):
        # The interrupted-build case: the heavy files landed but the run died
        # before finishing. Treating that as done would train on a truncated
        # year, so the manifest is written last and is the only marker.
        _fake_cached_year(self.root, 2001, self.meta, write_manifest=False)
        self.assertTrue(_era5_path(self.root, 2001).exists())
        self.assertFalse(is_cached(self.root, 2001))

    def test_settings_change_invalidates_the_year(self):
        _fake_cached_year(self.root, 2001, self.meta)
        other = dict(self.meta, tau_seconds=43200.0)
        self.assertFalse(is_cached(self.root, 2001, other))

    def test_atomic_write_leaves_no_temporary_behind(self):
        _fake_cached_year(self.root, 2001, self.meta)
        self.assertEqual(list(self.root.glob("*.tmp.npz")), [])


class TestShortYearGuard(unittest.TestCase):
    """A source year that stops early must fail loudly, not cache quietly."""

    def test_a_full_year_passes(self):
        # 6-hourly over 365 model days is 1460 slots.
        check_year_is_complete(1460, DAYS_PER_MODEL_YEAR, 6, 2001)
        check_year_is_complete(365, DAYS_PER_MODEL_YEAR, 24, 2001)   # daily

    def test_the_real_2023_stub_raises(self):
        # The case that actually occurred: the WeatherBench2 store ends
        # 2023-01-10, so a full-year request returns 40 six-hourly slots. It
        # was cached as a complete year before this guard existed, with 355 of
        # its days nudged toward one frozen January snapshot.
        with self.assertRaises(ValueError) as cm:
            check_year_is_complete(40, DAYS_PER_MODEL_YEAR, 6, 2023)
        msg = str(cm.exception)
        self.assertIn("40", msg)
        self.assertIn("1460", msg)
        self.assertIn("2023", msg)

    def test_a_partial_request_of_a_full_year_is_fine(self):
        # Asking for fewer days than a year is legitimate; the guard compares
        # against what was REQUESTED, not against a whole year.
        check_year_is_complete(20, 5, 6, 2001)
        with self.assertRaises(ValueError):
            check_year_is_complete(12, 5, 6, 2001)

    def test_a_short_year_cannot_pair_enough_frames_to_train(self):
        # Belt and braces: even if a short year reached disk, the pairing
        # yields only a handful of frames, all inside spin-up. state_slot_
        # indices caps at the available slots rather than clamping.
        idx = state_slot_indices(292, 1.25, 6, 40)
        self.assertEqual(len(idx), 7)
        self.assertTrue((idx < 40).all())


class TestStateSlotPairing(unittest.TestCase):
    """The frame-to-target pairing, i.e. where the label lag came from."""

    def test_frame_zero_is_one_save_in_not_zero(self):
        # The first save lands at t = save_interval. Pairing frame 0 with slot
        # 0 would lag every label by one save interval.
        idx = state_slot_indices(292, 1.25, 6, 1460)
        self.assertEqual(idx[0], 5)

    def test_unpaired_tail_frames_are_dropped_not_clipped(self):
        # A 292-frame year needs slots 5..1460, but only 0..1459 exist. The
        # last frame must be dropped; clipping would pair it with a stale
        # target and look perfectly healthy.
        idx = state_slot_indices(292, 1.25, 6, 1460)
        self.assertEqual(len(idx), 291)
        self.assertEqual(idx[-1], 1455)
        self.assertTrue((idx < 1460).all())

    def test_spacing_is_one_save_interval(self):
        idx = state_slot_indices(292, 1.25, 6, 1460)
        np.testing.assert_array_equal(np.diff(idx), 5)

    def test_daily_saves_pair_one_slot_per_day(self):
        # The offline configuration: daily saves against a daily-sampled target.
        idx = state_slot_indices(10, 1.0, 24, 10)
        np.testing.assert_array_equal(idx, np.arange(1, 10))

    def test_rejects_a_cadence_that_cannot_pair(self):
        # 1.25 d saves against 24 h ERA5 is 1.25 slots per save: no integer
        # pairing exists, so refuse rather than round into a drifting offset.
        with self.assertRaises(ValueError):
            state_slot_indices(10, 1.25, 24, 100)


class TestLoadYear(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.meta = cache_meta(
            get_speedy_coords(layers=8, spectral_truncation=21),
            cadence_hours=6, save_days=1.25, tau_seconds=21600.0,
            time_step_minutes=30.0, spinup_days=10.0)
        self.fields, self.seconds = _fake_cached_year(self.root, 2001,
                                                      self.meta)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trips_fields_and_clock(self):
        fields, seconds, states = load_year(self.root, 2001, meta=self.meta)
        np.testing.assert_array_equal(seconds, self.seconds)
        for name in VARS_3D:
            np.testing.assert_array_equal(fields[name], self.fields[name])
        self.assertEqual(set(states), set(STATE_FIELDS))

    def test_clock_survives_the_round_trip_exactly(self):
        # Fields are stored as float32 to halve the cache, but the clock must
        # not be: seconds since 1970 are ~1e9, where float32 resolves only to
        # ~64 s, which is enough to send a BY_DATE lookup to the wrong 6-hourly
        # slot. Exact equality is the guard against a future blanket cast.
        _, seconds, _ = load_year(self.root, 2001)
        self.assertEqual(seconds.dtype, np.float64)
        np.testing.assert_array_equal(seconds, self.seconds)
        self.assertNotEqual(float(np.float32(self.seconds[1])),
                            self.seconds[1],
                            "test clock is too small to exercise the precision "
                            "requirement it exists to document")

    def test_can_skip_the_states(self):
        # The offline stage wants only the targets; skipping the states avoids
        # reading a few hundred MB per year it will not use.
        _, _, states = load_year(self.root, 2001, with_states=False)
        self.assertIsNone(states)

    def test_rejects_a_year_built_with_other_settings(self):
        wrong = dict(self.meta, cadence_hours=24)
        with self.assertRaises(ValueError) as cm:
            load_year(self.root, 2001, meta=wrong)
        self.assertIn("cadence_hours", str(cm.exception))

    def test_missing_manifest_raises_rather_than_returning_partial_data(self):
        _manifest_path(self.root, 2001).unlink()
        with self.assertRaises(FileNotFoundError):
            load_year(self.root, 2001)


if __name__ == "__main__":
    unittest.main()
