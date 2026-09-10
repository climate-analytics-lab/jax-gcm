"""Unit tests for ``jcm.analysis`` — the xarray post-processing layer."""

from __future__ import annotations

import numpy as np
import xarray as xr

import jcm.constants as c
from jcm.analysis import (
    area_weights,
    column_burden,
    column_integral,
    global_mean,
    layer_pressure_thickness,
)


def _gauss_lats(n: int) -> np.ndarray:
    """Gauss-Legendre latitudes [deg] (the dinosaur/jcm output-grid case)."""
    nodes, _ = np.polynomial.legendre.leggauss(n)
    return np.rad2deg(np.arcsin(nodes))


def test_area_weights_recovers_exact_gauss_legendre_weights():
    # Any latitude ordering must recover the leggauss weights (reordered to
    # match), not the cos(lat) approximation.
    n = 16
    nodes, gauss_w = np.polynomial.legendre.leggauss(n)
    lats = np.rad2deg(np.arcsin(nodes))
    # Shuffle to prove the reorder-to-data-order logic.
    perm = np.array([3, 0, 15, 7, 1, 9, 2, 14, 4, 8, 5, 13, 6, 10, 11, 12])
    lats_shuffled = lats[perm]
    w = area_weights(lats_shuffled)
    np.testing.assert_allclose(np.asarray(w), gauss_w[perm], atol=1e-12)
    # And it is genuinely different from the cos(lat) fallback.
    assert not np.allclose(np.asarray(w),
                           np.cos(np.deg2rad(lats_shuffled)), atol=1e-3)


def test_area_weights_falls_back_to_cosine_for_uniform_grid():
    lats = np.linspace(-87.0, 87.0, 24)      # not Gauss-Legendre nodes
    w = area_weights(lats)
    np.testing.assert_allclose(np.asarray(w), np.cos(np.deg2rad(lats)))


def test_area_weights_reads_lat_from_xarray():
    lats = _gauss_lats(8)
    ds = xr.Dataset(coords={"lat": lats})
    nodes, gauss_w = np.polynomial.legendre.leggauss(8)
    np.testing.assert_allclose(np.asarray(area_weights(ds)), gauss_w,
                               atol=1e-12)


def test_global_mean_of_constant_is_the_constant_both_branches():
    const = 3.5
    # Gauss-Legendre grid -> exact-weight branch.
    gl = xr.DataArray(np.full((8, 4), const),
                      dims=("lat", "lon"),
                      coords={"lat": _gauss_lats(8),
                              "lon": np.linspace(0, 360, 4, endpoint=False)})
    assert float(global_mean(gl)) == const
    # Uniform grid -> cos(lat) fallback branch.
    uni = xr.DataArray(np.full((10, 4), const),
                       dims=("lat", "lon"),
                       coords={"lat": np.linspace(-80, 80, 10),
                               "lon": np.linspace(0, 360, 4, endpoint=False)})
    np.testing.assert_allclose(float(global_mean(uni)), const)


def test_global_mean_without_lat_is_unweighted():
    # No lat coordinate -> plain unweighted mean over the horizontal dims.
    da = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]),
                      dims=("x", "y"))
    np.testing.assert_allclose(float(global_mean(da)), 2.5)


def test_global_mean_keeps_nonhorizontal_dims():
    da = xr.DataArray(
        np.arange(2 * 3 * 8 * 4, dtype=float).reshape(2, 3, 8, 4),
        dims=("time", "level", "lat", "lon"),
        coords={"lat": _gauss_lats(8),
                "lon": np.linspace(0, 360, 4, endpoint=False)})
    gm = global_mean(da)
    assert gm.dims == ("time", "level")
    assert gm.shape == (2, 3)


def _surface_first_half_pressures(dp_level: np.ndarray) -> np.ndarray:
    # Interface profile whose -diff along level_i equals dp_level.
    return np.concatenate([[1000.0], 1000.0 - np.cumsum(dp_level)])


def test_layer_pressure_thickness_prefers_pressure_thickness():
    dp_level = np.array([100.0, 150.0, 250.0])
    ph = _surface_first_half_pressures(dp_level)
    ds = xr.Dataset({
        "pressure_thickness": (("time", "level"), dp_level[None]),
        "pressure_half": (("time", "level_i"), ph[None]),
    })
    out = layer_pressure_thickness(ds)
    # The (length-1) time axis is preserved, not squeezed to t=0.
    assert out.dims == ("time", "level")
    np.testing.assert_allclose(np.asarray(out), dp_level[None])


def test_layer_pressure_thickness_falls_back_to_pressure_half_diff():
    dp_level = np.array([100.0, 150.0, 250.0])
    ph = _surface_first_half_pressures(dp_level)
    ds = xr.Dataset({"pressure_half": (("time", "level_i"), ph[None])})
    out = layer_pressure_thickness(ds)
    assert out.dims == ("time", "level")
    np.testing.assert_allclose(np.asarray(out), dp_level[None])


def test_layer_pressure_thickness_preserves_time_when_ps_evolves():
    # Two timesteps with genuinely different layer thicknesses (ps evolved):
    # the time dim must survive so each timestep gets its own Δp, not a frozen
    # t=0 profile. Covers the pressure_thickness branch.
    dp_t0 = np.array([100.0, 150.0, 250.0])
    dp_t1 = np.array([110.0, 140.0, 260.0])
    ds = xr.Dataset({
        "pressure_thickness": (("time", "level"), np.stack([dp_t0, dp_t1])),
    })
    out = layer_pressure_thickness(ds)
    assert out.dims == ("time", "level")
    np.testing.assert_allclose(np.asarray(out), np.stack([dp_t0, dp_t1]))


def test_layer_pressure_thickness_time_varying_pressure_half_fallback():
    # Same, via the pressure_half differencing fallback.
    dp_t0 = np.array([100.0, 150.0, 250.0])
    dp_t1 = np.array([110.0, 140.0, 260.0])
    ph = np.stack([_surface_first_half_pressures(dp_t0),
                   _surface_first_half_pressures(dp_t1)])
    ds = xr.Dataset({"pressure_half": (("time", "level_i"), ph)})
    out = layer_pressure_thickness(ds)
    assert out.dims == ("time", "level")
    # No stale level coordinate is attached after differencing the interfaces.
    assert "level" not in out.coords
    np.testing.assert_allclose(np.asarray(out), np.stack([dp_t0, dp_t1]))


def test_column_burden_is_time_varying_when_ps_evolves():
    # The whole point: with ps evolving, the burden of a FIXED mixing ratio
    # must differ per timestep (dp tracks ps) and equal the hand-computed
    # per-timestep q·dp/g — not a single frozen-t=0 value repeated.
    dp_t0 = np.array([100.0, 150.0, 250.0])
    dp_t1 = np.array([110.0, 140.0, 260.0])
    ph = np.stack([_surface_first_half_pressures(dp_t0),
                   _surface_first_half_pressures(dp_t1)])
    q = np.array([1e-3, 2e-3, 3e-3])
    ds = xr.Dataset({
        "so4": (("time", "level"), np.stack([q, q])),
        "pressure_half": (("time", "level_i"), ph),
    })
    burden = column_burden(ds, "so4")
    assert burden.dims == ("time",)
    expected = np.array([(q * dp_t0).sum(), (q * dp_t1).sum()]) / c.grav
    np.testing.assert_allclose(np.asarray(burden), expected)
    # Genuinely time-varying — a frozen-t=0 dp would make these equal.
    assert float(burden.isel(time=0)) != float(burden.isel(time=1))


def test_column_integral_matches_hand_computation():
    q = xr.DataArray(np.array([2.0, 4.0, 6.0]), dims=("level",))
    dp = xr.DataArray(np.array([100.0, 200.0, 300.0]), dims=("level",))
    expected = (2.0 * 100.0 + 4.0 * 200.0 + 6.0 * 300.0) / c.grav
    np.testing.assert_allclose(float(column_integral(q, dp)), expected)


def test_column_burden_end_to_end():
    dp_level = np.array([100.0, 150.0, 250.0])
    ph = _surface_first_half_pressures(dp_level)
    q = np.array([1e-3, 2e-3, 3e-3])
    ds = xr.Dataset({
        "so4": (("time", "level"), q[None]),
        "pressure_half": (("time", "level_i"), ph[None]),
    })
    burden = column_burden(ds, "so4")
    expected = (q * dp_level).sum() / c.grav
    np.testing.assert_allclose(float(burden.isel(time=0)), expected)
