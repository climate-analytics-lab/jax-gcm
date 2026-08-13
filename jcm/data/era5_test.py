"""Tests for the WeatherBench2 ERA5 source (#610).

Everything except the ``@pytest.mark.slow`` live test runs on synthetic
data — no network.
"""
import unittest

import numpy as np
import pytest
import xarray as xr

from jcm.data import era5


class TestStoreSelection(unittest.TestCase):
    def test_smallest_oversampling_store_wins(self):
        self.assertIn("64x32", era5.select_store(48))
        self.assertIn("240x121", era5.select_store(96))
        self.assertIn("240x121", era5.select_store(192))
        self.assertIn("360x181", era5.select_store(320))

    def test_finer_than_any_store_takes_finest(self):
        self.assertIn("360x181", era5.select_store(1024))


class TestInterpLogP(unittest.TestCase):
    def test_matches_per_column_interp_and_clamps(self):
        rng = np.random.default_rng(0)
        p_src = np.array([5e3, 1e4, 5e4, 8.5e4, 1e5])
        field = rng.normal(size=(2, 5, 3, 4))
        # Targets include values outside the source range (clamp).
        p_tgt = rng.uniform(1e3, 1.1e5, size=(2, 6, 3, 4))
        got = era5._interp_log_p(field, p_src, p_tgt)
        for t in range(2):
            for i in range(3):
                for j in range(4):
                    want = np.interp(np.log(p_tgt[t, :, i, j]),
                                     np.log(p_src), field[t, :, i, j])
                    np.testing.assert_allclose(got[t, :, i, j], want,
                                               rtol=1e-12)

    def test_linear_in_log_p_field_is_exact(self):
        p_src = np.array([1e4, 2e4, 4e4, 8e4])
        field = np.log(p_src)[None, :, None, None] * np.ones((1, 4, 2, 2))
        p_tgt = np.full((1, 3, 2, 2), 3e4)
        got = era5._interp_log_p(field, p_src, p_tgt)
        np.testing.assert_allclose(got, np.log(3e4), rtol=1e-12)


class TestModelLevelPressures(unittest.TestCase):
    def test_hybrid_and_sigma(self):
        from types import SimpleNamespace
        ps = np.full((2, 3, 4), 1.0e5)
        hyb = SimpleNamespace(a_centers=np.array([100.0, 0.0]),
                              b_centers=np.array([0.0, 0.9]))
        p = era5._model_level_pressures(hyb, ps)
        self.assertEqual(p.shape, (2, 2, 3, 4))
        np.testing.assert_allclose(p[:, 0], 100.0)
        np.testing.assert_allclose(p[:, 1], 9.0e4)
        sig = SimpleNamespace(centers=np.array([0.1, 0.5, 1.0]))
        p = era5._model_level_pressures(sig, ps)
        np.testing.assert_allclose(p[:, 1], 5.0e4)


def _synthetic_wb2(nlev=4, nlat=19, nlon=36, ntime=2):
    """Build a tiny ERA5-like dataset in the renamed post-_open_store layout."""
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    level = np.array([100.0, 500.0, 850.0, 1000.0])[:nlev]  # hPa
    time = np.array([np.datetime64("2000-01-01") + np.timedelta64(6 * k, "h")
                     for k in range(ntime)])
    shape = (ntime, nlev, nlat, nlon)
    # T linear in log-p so the vertical interpolation is exact.
    T = np.broadcast_to(
        (250.0 + 10.0 * np.log(level * 100.0 / 5.0e4))[None, :, None, None],
        shape)
    ds = xr.Dataset(
        {
            "u": (("time", "level", "lat", "lon"), np.full(shape, 7.0)),
            "v": (("time", "level", "lat", "lon"), np.zeros(shape)),
            "T": (("time", "level", "lat", "lon"), T.copy()),
            "sp": (("time", "lat", "lon"),
                   np.full((ntime, nlat, nlon), 1.0e5)),
        },
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
    )
    return ds


class TestToModelGrid(unittest.TestCase):
    def _coords(self):
        from dinosaur.sigma_coordinates import SigmaCoordinates

        from jcm.utils import get_coords
        return get_coords(SigmaCoordinates.equidistant(8),
                          spectral_truncation=21)

    def test_shapes_layout_and_vertical_exactness(self):
        coords = self._coords()
        nlon, nlat = coords.horizontal.nodal_shape
        out = era5._to_model_grid(_synthetic_wb2(), coords, ("u", "v", "T"))
        self.assertEqual(out["T"].dims, ("time", "lev", "lon", "lat"))
        self.assertEqual(out["T"].shape, (2, 8, nlon, nlat))
        self.assertEqual(out["sp"].shape, (2, nlon, nlat))
        self.assertTrue(np.isfinite(out["T"].values).all())
        # T is linear in log-p: interior model levels must reproduce it.
        sigma = np.asarray(coords.vertical.centers)
        p_model = sigma * 1.0e5
        want = 250.0 + 10.0 * np.log(p_model / 5.0e4)
        # Levels inside the source range [100 hPa, 1000 hPa] are exact;
        # levels outside clamp.
        inside = (p_model >= 1.0e4) & (p_model <= 1.0e5)
        got = out["T"].values[0, :, 0, 0]
        np.testing.assert_allclose(got[inside], want[inside], rtol=1e-5)
        self.assertTrue(np.all(np.isfinite(got[~inside])))

    def test_nudging_target_wiring(self):
        from jcm.forcing import BY_DATE, TimeSeries
        coords = self._coords()
        ds = era5._to_model_grid(_synthetic_wb2(), coords, ("u", "v", "T"))
        from jcm.nudging import NudgingTarget
        target = NudgingTarget.from_dataset(ds)
        self.assertIsInstance(target.u_wind, TimeSeries)
        self.assertEqual(int(target.u_wind.align_mode), BY_DATE)
        self.assertEqual(target.u_wind.values.shape[:2], (2, 8))


def _fake_coords(nlon=64, nlat=32, centers=None):
    from types import SimpleNamespace
    return SimpleNamespace(
        horizontal=SimpleNamespace(
            nodal_shape=(nlon, nlat),
            latitudes=np.linspace(-1.5, 1.5, nlat),
            longitudes=np.linspace(0.0, 6.2, nlon)),
        vertical=SimpleNamespace(
            centers=np.linspace(0.1, 1.0, 8) if centers is None
            else centers))


class TestCacheKey(unittest.TestCase):
    def test_key_distinguishes_grid_window_freq(self):
        keys = {
            era5._window_key(c, s, e, f, v)
            for c in (_fake_coords(64, 32), _fake_coords(192, 96))
            for s, e in (("2000-01-01", "2000-02-01"),
                         ("2000-01-01", "2000-03-01"))
            for f in ("6h", "1d")
            for v in (("u", "v", "T"), ("u", "v", "T", "q", "z"))
        }
        self.assertEqual(len(keys), 16)

    def test_key_distinguishes_same_shape_different_levels(self):
        # Same dims, different sigma definitions must not share a cache
        # entry (Codex P2 on #611).
        a = era5._window_key(_fake_coords(), "2000-01-01", "2000-02-01",
                             "6h", ("u",))
        b = era5._window_key(
            _fake_coords(centers=np.linspace(0.05, 0.95, 8)),
            "2000-01-01", "2000-02-01", "6h", ("u",))
        self.assertNotEqual(a, b)


class TestRunnerWiring(unittest.TestCase):
    def test_nudging_config_masks_top_and_pbl(self):
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.physics.echam.echam_levels import get_echam_levels
        vertical = get_echam_levels(47)
        cfg = OmegaConf.create({"tau_hours": 6.0, "pbl_levels": 2,
                                "min_pressure_hpa": 60.0})
        inv_tau, nlev = runners._nudging_inv_tau(cfg, vertical)
        self.assertEqual(nlev, 47)
        p_ref = (np.asarray(vertical.a_centers)
                 + np.asarray(vertical.b_centers) * 101325.0)
        self.assertTrue(np.all(inv_tau[p_ref < 6000.0] == 0.0))
        self.assertTrue(np.all(inv_tau[-2:] == 0.0))
        interior = (p_ref >= 6000.0)
        interior[-2:] = False
        np.testing.assert_allclose(inv_tau[interior], 1.0 / 21600.0)

    def test_maybe_add_nudging_appends_term(self):
        from omegaconf import OmegaConf

        from jcm import runners
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        cfg = OmegaConf.create(
            {"nudging": {"enabled": True, "tau_hours": 6.0}})
        physics = runners.maybe_add_nudging(speedy_physics(), cfg, coords)
        self.assertIn("nudging", [t.category for t in physics.terms])
        # Disabled → untouched.
        physics = runners.maybe_add_nudging(
            speedy_physics(), OmegaConf.create({}), coords)
        self.assertNotIn("nudging", [t.category for t in physics.terms])

    def test_config_presets_compose(self):
        from jcm.runners_test import _compose
        cfg = _compose(["nudging=era5", "init=era5",
                        "run.start_date=2010-01-01"])
        self.assertTrue(cfg.nudging.enabled)
        self.assertEqual(cfg.nudging.source, "era5")
        self.assertEqual(cfg.init.kind, "era5")
        self.assertIsNone(cfg.init.date)


@pytest.mark.slow
class TestLiveWb2(unittest.TestCase):
    """One tiny real pull from the public WB2 store (network required)."""

    def test_nudging_target_from_cloud(self):
        from dinosaur.sigma_coordinates import SigmaCoordinates

        from jcm.utils import get_coords
        coords = get_coords(SigmaCoordinates.equidistant(8),
                            spectral_truncation=21)
        try:
            target = era5.nudging_target(coords, "2010-01-01", "2010-01-01",
                                         cache=False)
        except Exception as e:  # noqa: BLE001 — no network on this node
            self.skipTest(f"WB2 store unreachable: {e}")
        u = np.asarray(target.u_wind.values)
        self.assertEqual(u.shape[1:], (8, *coords.horizontal.nodal_shape))
        self.assertTrue(np.isfinite(u).all())
        self.assertLess(np.abs(u).max(), 150.0)


if __name__ == "__main__":
    unittest.main()
