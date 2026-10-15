"""Unit tests for ``jcm.analysis`` — the xarray post-processing layer."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import jcm.constants as c
from jcm.analysis import (
    area_weights,
    column_burden,
    column_integral,
    global_mean,
    layer_pressure_thickness,
    to_pressure_levels,
    total_cloud_cover,
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
    np.testing.assert_allclose(
        float(global_mean(gl)), const, rtol=1e-14, atol=0.0)
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


def test_global_mean_keeps_plev_too():
    # #258: a to_pressure_levels() output's 'plev' axis must be treated like
    # 'level' -- not averaged away as if it were horizontal.
    da = xr.DataArray(
        np.arange(2 * 3 * 8 * 4, dtype=float).reshape(2, 3, 8, 4),
        dims=("time", "plev", "lat", "lon"),
        coords={"lat": _gauss_lats(8),
                "lon": np.linspace(0, 360, 4, endpoint=False)})
    gm = global_mean(da)
    assert gm.dims == ("time", "plev")
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


def _cover(profile, **kwargs):
    """Total cover of a bare 1-D profile, as a float."""
    da = xr.DataArray(np.asarray(profile, dtype=float), dims=("level",))
    return float(total_cloud_cover(da, **kwargs))


def test_total_cloud_cover_single_layer_is_that_layer():
    np.testing.assert_allclose(_cover([0.37]), 0.37)


def test_total_cloud_cover_adjacent_equal_layers_overlap_maximally():
    # Maximum overlap of a vertically contiguous cloud: two (or five) stacked
    # half-covered layers are ONE half-covered cloud, not 0.75 or 0.97.
    np.testing.assert_allclose(_cover([0.5, 0.5]), 0.5)
    np.testing.assert_allclose(_cover([0.5] * 5), 0.5)


def test_total_cloud_cover_adjacent_unequal_layers_take_the_maximum():
    np.testing.assert_allclose(_cover([0.3, 0.7, 0.4]), 0.7)


def test_total_cloud_cover_separated_clouds_overlap_randomly():
    # A fully clear layer between two decks breaks the maximum-overlap chain,
    # so the two combine randomly: 1 - (1-a)(1-b).
    a, b = 0.4, 0.6
    np.testing.assert_allclose(_cover([a, 0.0, b]), 1.0 - (1 - a) * (1 - b))


def test_total_cloud_cover_is_between_column_max_and_random_overlap():
    # The bracketing that motivates using this definition for the gate: the
    # column max is a lower bound and random overlap an upper bound, and a
    # profile with both contiguous and separated cloud is strictly inside.
    profile = np.array([0.6, 0.5, 0.0, 0.3, 0.2, 0.0, 0.45])
    maxrandom = _cover(profile)
    assert profile.max() < maxrandom < 1.0 - np.prod(1.0 - profile)


def test_total_cloud_cover_is_orientation_independent():
    # The clear-sky product is symmetric in the vertical, so surface-first and
    # TOA-first profiles score the same and no orientation guard is needed —
    # which is what makes pre-#710 files safe to score with this.
    profile = np.array([0.1, 0.0, 0.85, 0.3, 0.3, 0.0, 0.55, 0.2])
    np.testing.assert_allclose(_cover(profile), _cover(profile[::-1]),
                               rtol=1e-12)
    # The agreement is "to rounding", not exact: the ``max(1 - c_{k-1}, zepsec)``
    # floor is applied in loop order, so it caps a different denominator in the
    # reversed column. A cover within zepsec of 1 is where that shows, and it
    # shows at O(zepsec) — far below anything a gate or a climatology reads.
    near_one = np.array([0.2, 1.0 - 5e-13])
    assert abs(_cover(near_one) - _cover(near_one[::-1])) < 1e-11


def test_total_cloud_cover_clips_out_of_range_values():
    # Saturated/negative excursions in saved output must not produce a cover
    # outside [0, 1] or a negative "clear" fraction.
    np.testing.assert_allclose(_cover([-0.2, 0.4, 1.3]), 1.0)
    np.testing.assert_allclose(_cover([-0.5, -1e-9]), 0.0)


def test_total_cloud_cover_handles_a_fully_cloudy_layer():
    # c = 1 makes the Fortran's 1/(1-c) denominator singular; the zepsec floor
    # holds it at 1e-12 while the matching numerator is exactly zero, so the
    # answer is 1 wherever the overcast layer sits in the column (including
    # the ends).
    for profile in ([1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.3, 0.2, 1.0]):
        np.testing.assert_allclose(_cover(profile), 1.0)
    assert np.isfinite(_cover([1.0, 1.0, 1.0]))


def test_total_cloud_cover_preserves_time_and_horizontal_dims():
    rng = np.random.default_rng(0)
    cf = xr.DataArray(
        rng.uniform(0.0, 1.0, size=(2, 5, 8, 4)),
        dims=("time", "level", "lat", "lon"),
        coords={"level": np.linspace(1.0, 0.0, 5),
                "lat": _gauss_lats(8),
                "lon": np.linspace(0, 360, 4, endpoint=False)})
    cover = total_cloud_cover(cf)
    assert cover.dims == ("time", "lat", "lon")
    assert cover.shape == (2, 8, 4)
    # The reduced axis' coordinate is gone; the surviving ones are intact.
    assert "level" not in cover.coords
    np.testing.assert_allclose(np.asarray(cover["lat"]), _gauss_lats(8))
    # Per column it agrees with the 1-D reference, and it is a valid fraction.
    np.testing.assert_allclose(float(cover.isel(time=1, lat=3, lon=2)),
                               _cover(cf.isel(time=1, lat=3, lon=2).values))
    assert np.all(np.asarray(cover) >= 0.0)
    assert np.all(np.asarray(cover) <= 1.0)


def test_total_cloud_cover_accepts_a_named_vertical_dim():
    # Interface-named or otherwise non-default vertical axes are addressable.
    cf = xr.DataArray(np.array([[0.5, 0.0], [0.25, 0.25]]),
                      dims=("col", "lev"))
    np.testing.assert_allclose(np.asarray(total_cloud_cover(cf, dim="lev")),
                               [0.5, 0.25])


def test_total_cloud_cover_rejects_a_missing_vertical_dim():
    cf = xr.DataArray(np.array([0.5, 0.25]), dims=("lat",))
    with pytest.raises(ValueError, match="not a dimension"):
        total_cloud_cover(cf)


def test_total_cloud_cover_handles_float32_input():
    # Production runs integrate in float32 (``physics_dtype``), so saved
    # cloud_fraction is float32 and this is the realistic input. An overcast
    # layer must still score 1 rather than NaN (the Fortran's ``zxsec`` would
    # round to exactly 1.0 here and turn its guarded 0/1e-12 into 0/0), and
    # the float32 column must score what its float64 widening scores.
    profile = np.array([0.5, 1.0, 0.2], dtype=np.float32)
    cover = total_cloud_cover(xr.DataArray(profile, dims=("level",)))
    assert np.isfinite(float(cover))
    np.testing.assert_allclose(float(cover), 1.0)
    # And a float32 column with no overcast layer matches the float64 answer.
    partly = np.array([0.5, 0.0, 0.2], dtype=np.float32)
    np.testing.assert_allclose(
        float(total_cloud_cover(xr.DataArray(partly, dims=("level",)))),
        _cover(partly.astype(np.float64)), rtol=1e-7)


def test_total_cloud_cover_rejects_an_empty_vertical_axis():
    cf = xr.DataArray(np.zeros((0, 3)), dims=("level", "col"))
    with pytest.raises(ValueError, match="no levels"):
        total_cloud_cover(cf)


def test_total_cloud_cover_carries_cf_attributes():
    cover = total_cloud_cover(xr.DataArray(np.array([0.3, 0.4]),
                                           dims=("level",)))
    assert cover.name == "total_cloud_cover"
    assert cover.attrs["standard_name"] == "cloud_area_fraction"
    assert cover.attrs["units"] == "1"


def test_total_cloud_cover_stays_lazy_on_a_dask_array():
    # The release-validation gate hands this an open_mfdataset (dask) array
    # spanning a whole year, so the reduction must build a graph rather than
    # pull every level of every step into memory. Laziness is the assertion;
    # exact agreement with the numpy path is the correctness check.
    import dask.array as dask_array

    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, size=(4, 6, 8, 4))
    cf = xr.DataArray(values, dims=("time", "level", "lat", "lon"))
    lazy = xr.DataArray(dask_array.from_array(values, chunks=(2, 3, 4, 4)),
                        dims=cf.dims)

    cover = total_cloud_cover(lazy)
    assert isinstance(cover.data, dask_array.Array)
    # Bit-identical, not merely close: the two paths run the same arithmetic.
    np.testing.assert_array_equal(np.asarray(cover.compute()),
                                  np.asarray(total_cloud_cover(cf)))


def test_total_cloud_cover_propagates_nan_and_leaves_the_input_alone():
    # A missing level must not be silently skipped — a column with one is
    # unscoreable, and reporting the cover of the remaining levels would hide
    # a corrupt file from the gate's NaN scan. And the [0, 1] clip must not
    # write back into the caller's Dataset.
    values = np.array([[0.3, 0.5], [np.nan, 0.5], [1.4, -0.2]])
    cf = xr.DataArray(values.copy(), dims=("level", "col"))

    cover = np.asarray(total_cloud_cover(cf))
    assert np.isnan(cover[0])
    np.testing.assert_allclose(cover[1], 0.5)
    # The caller's array still holds its NaN and its out-of-range values.
    np.testing.assert_array_equal(np.asarray(cf), values)


# ---------------------------------------------------------------------------
# to_pressure_levels (#258)


def _analytic_sigma_dataset(surface_pa):
    """Build a column whose field is EXACTLY linear in ln(pressure).

    ``surface_pa`` may be a scalar (one column) or an array (a ``col``
    dimension of columns, each with its own surface pressure, e.g. from
    orography) -- a linear-in-ln(p) field makes log-pressure linear
    interpolation exact everywhere, not merely at the model's own levels,
    which is what isolates "is the interpolation itself correct" from "does
    it merely reproduce its own input nodes".
    """
    sigma = np.array([0.99, 0.8, 0.6, 0.4, 0.2, 0.05])    # surface-first
    surface_pa = np.asarray(surface_pa, dtype=np.float64)
    pressure_full = sigma[:, None] * surface_pa[None, :] \
        if surface_pa.ndim else sigma * surface_pa
    a0, a1 = 250.0, 10.0                                  # T = a0 + a1*ln(p)
    temperature = a0 + a1 * np.log(pressure_full)
    dims = ("level", "col") if surface_pa.ndim else ("level",)
    # A field with no level dimension must pass through untouched.
    surface_pressure = (("col",), surface_pa) if surface_pa.ndim else ((), surface_pa)
    ds = xr.Dataset(
        {"temperature": (dims, temperature),
         "pressure_full": (dims, pressure_full),
         "surface_pressure": surface_pressure},
        coords={"level": sigma},
    )
    return ds, sigma, pressure_full, a0, a1


def test_round_trips_exactly_at_the_models_own_pressures():
    ds, sigma, pressure_full, a0, a1 = _analytic_sigma_dataset(100_000.0)
    out = to_pressure_levels(ds, pressure_full / 100.0)   # same nodes, hPa
    np.testing.assert_array_equal(out["temperature"].values,
                                  ds["temperature"].values)
    # plev carries the requested order and the model's own Pa units.
    np.testing.assert_allclose(out["plev"].values, pressure_full)
    assert out["plev"].attrs["units"] == "Pa"
    assert out["plev"].attrs["positive"] == "down"
    assert out["plev"].attrs["standard_name"] == "air_pressure"


def test_round_trips_exactly_at_the_models_own_pressures_in_float32():
    # Codex review finding: a NATIVELY float32 pressure_full (jcm writes
    # most level fields float32) must have its log taken at float64
    # BEFORE comparing against the float64 target built from levels_hpa --
    # promoting the result of an already-float32 np.log afterwards cannot
    # recover the precision lost taking it. Built directly in float32
    # (not cast down from an exact float64 original) so the test cannot
    # pass merely because rounding happened to go the convenient way.
    pressure_full = np.array([100_000.0, 50_000.0, 10_000.0], dtype=np.float32)
    temperature = np.array([300.0, 260.0, 220.0], dtype=np.float32)
    ds = xr.Dataset(
        {"temperature": ("level", temperature),
         "pressure_full": ("level", pressure_full)},
        coords={"level": [0, 1, 2]})

    out = to_pressure_levels(ds, [1000.0, 500.0, 100.0])
    assert out["temperature"].dtype == np.float32
    assert not np.any(np.isnan(out["temperature"].values))
    np.testing.assert_array_equal(out["temperature"].values, temperature)


def test_interpolates_to_a_known_value_between_model_levels():
    ds, sigma, pressure_full, a0, a1 = _analytic_sigma_dataset(100_000.0)
    midpoints_pa = np.exp(
        (np.log(pressure_full[:-1]) + np.log(pressure_full[1:])) / 2)
    out = to_pressure_levels(ds, midpoints_pa / 100.0)
    want = a0 + a1 * np.log(midpoints_pa)
    # Linear-in-ln(p) is exactly reproduced by linear interpolation in
    # ln(p), anywhere between two nodes, to floating-point precision.
    np.testing.assert_allclose(out["temperature"].values, want,
                               rtol=0.0, atol=1e-10)


def test_masks_below_the_surface_and_above_the_model_lid():
    ds, sigma, pressure_full, a0, a1 = _analytic_sigma_dataset(100_000.0)
    # 2000 hPa is below the (1000 hPa) surface; 1 hPa is above the (50 hPa)
    # model lid; 500 hPa is a genuine in-range target between them.
    out = to_pressure_levels(ds, [2000.0, 500.0, 1.0])
    got = out["temperature"].values
    assert np.isnan(got[0])
    assert np.isnan(got[2])
    assert np.isfinite(got[1])
    np.testing.assert_allclose(got[1], a0 + a1 * np.log(50_000.0))


def test_masking_is_per_column_with_varying_surface_pressure():
    # Three columns with different (orography-driven) surface pressures, so
    # each has its own actual max pressure (surface * sigma[0] = 0.99):
    # 990, 891, 792 hPa. 850 hPa is in-range for the first two but below
    # the third's own column range.
    ds, sigma, pressure_full, a0, a1 = _analytic_sigma_dataset(
        [100_000.0, 90_000.0, 80_000.0])
    out = to_pressure_levels(ds, [850.0])
    got = out["temperature"].isel(plev=0).values
    assert np.isfinite(got[0]) and np.isfinite(got[1])
    assert np.isnan(got[2])
    # The analytic field is linear in ln(actual pressure), independent of
    # which column's sigma/surface-pressure produced that pressure.
    np.testing.assert_allclose(got[0], a0 + a1 * np.log(85_000.0))
    np.testing.assert_allclose(got[1], a0 + a1 * np.log(85_000.0))


def test_fields_without_a_level_dimension_pass_through_unchanged():
    ds, *_ = _analytic_sigma_dataset(100_000.0)
    out = to_pressure_levels(ds, [1000.0, 500.0])
    xr.testing.assert_identical(out["surface_pressure"], ds["surface_pressure"])
    # pressure_full/pressure_half describe the SIGMA grid interpolated
    # from, not the new plev grid, so they are left as they were too.
    xr.testing.assert_identical(out["pressure_full"], ds["pressure_full"])


def test_requires_pressure_full_and_a_nonempty_level_list():
    ds, *_ = _analytic_sigma_dataset(100_000.0)
    with pytest.raises(ValueError, match="pressure_full"):
        to_pressure_levels(ds.drop_vars("pressure_full"), [1000.0])
    with pytest.raises(ValueError, match="non-empty"):
        to_pressure_levels(ds, [])


def test_preserves_the_requested_level_order_unsorted():
    ds, *_ = _analytic_sigma_dataset(100_000.0)
    out = to_pressure_levels(ds, [500.0, 1000.0, 10.0])
    np.testing.assert_array_equal(out["plev"].values, [50_000.0, 100_000.0, 1_000.0])


def test_dataset_path_on_a_saved_sigma_file(tmp_path):
    # The Python API on an actual round-tripped netCDF, not just an
    # in-memory Dataset -- the "existing output Dataset" half of #258.
    ds, sigma, pressure_full, a0, a1 = _analytic_sigma_dataset(100_000.0)
    path = tmp_path / "sigma.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        # 900 hPa (not 1000): the column's own top level is only 990 hPa
        # (sigma[0] = 0.99 * the 1000 hPa surface_pa), so 900 is the
        # in-range point closest to it.
        out = to_pressure_levels(loaded.load(), [900.0, 500.0, 100.0])
    want = a0 + a1 * np.log(np.array([90_000.0, 50_000.0, 10_000.0]))
    np.testing.assert_allclose(out["temperature"].values, want, rtol=1e-10)


def test_stays_lazy_on_a_dask_backed_dataset():
    # Codex review finding: a run opened lazily (xr.open_mfdataset(...), as
    # the release-validation gate does for total_cloud_cover) must convert
    # without apply_ufunc's default dask="forbidden" forcing every chunk
    # into memory first. Laziness is the assertion; exact agreement with
    # the eager (NumPy) path is the correctness check.
    import dask.array as dask_array

    ds, sigma, pressure_full, a0, a1 = _analytic_sigma_dataset(
        [100_000.0, 90_000.0, 80_000.0])
    lazy = ds.copy()
    for name in ("temperature", "pressure_full"):
        # The 'level' (core) dimension must stay a single chunk; chunk
        # over 'col' only.
        lazy[name] = lazy[name].copy(
            data=dask_array.from_array(ds[name].values, chunks=(-1, 1)))

    levels = [850.0, 500.0, 200.0]
    out = to_pressure_levels(lazy, levels)
    assert isinstance(out["temperature"].data, dask_array.Array)
    np.testing.assert_array_equal(
        np.asarray(out["temperature"].compute()),
        np.asarray(to_pressure_levels(ds, levels)["temperature"]))
