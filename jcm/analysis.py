"""xarray/post-processing layer for model OUTPUT.

The one home for area weights, global means, layer pressure thicknesses and
column burdens computed on saved jcm netCDF output. Issue #640 found each of
these independently reimplemented at least four times
(``tools/jam_burden_report.py``, ``tools/release_validation/health.py``,
``tools/validate_era5_bundle.py`` and inline in
:func:`jcm.runners.run_chunked`); this module is where they now live once.
:func:`total_cloud_cover` joins them for the same reason: overlap is a
definition, not an incidental reduction, so the release-validation gate and any
analysis notebook must read the same one
(``docs/source/design/cloud_cover_gate.md``). It is **not** the saved
``radiation.total_cloud_cover`` diagnostic despite the matching name: that one
is the in-model McICA sub-column cover, sampled under the flux solve's own
overlap rule from a differently-preprocessed cloud fraction, and the design doc
above sets out how far apart the two run in practice. Nor is it the saved
``clouds.total_cloud_cover``, which is the same overlap applied to the
*instantaneous* fraction inside the model and time-averaged afterwards;
this function applied to a saved mean profile is the offline approximation to
it.

Everything here takes labelled xarray in and returns xarray (or a Python
``float``) out, using numpy internally. It operates on *saved* output — never
on device arrays inside a jitted physics step. The in-model, device-array
column integral lives separately in
:func:`jcm.physics.diagnostics.aerocom._column_integral`, because physics runs
on JAX arrays inside ``jit`` (it integrates over ``pressure_half`` interfaces
in the physics-internal frame). The two are deliberately kept apart —
cross-reference, do not merge. The one piece of in-model code this module does
call is the maximum-random overlap recurrence behind :func:`total_cloud_cover`
(:func:`jcm.physics.clouds.cloud_overlap.max_random_cover`), which is
array-type-agnostic and is handed numpy here: the cover the physics step saves
and the cover scored here must be the same arithmetic.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import jcm.constants as c
from jcm.physics.clouds.cloud_overlap import max_random_cover

#: Dimensions that are never horizontal. The horizontal dims of a field are
#: everything *else* — this is the convention promoted from
#: ``tools/jam_burden_report.py::_horizontal_dims``. ``plev`` (#258,
#: :func:`to_pressure_levels`) is the pressure-surface counterpart of
#: ``level``/``level_i`` and belongs here for the same reason: without it,
#: ``global_mean`` of a ``plev``-dimensioned field would average over
#: pressure as well as latitude/longitude, silently collapsing a vertical
#: profile into a single number instead of reducing only the horizontal.
_NON_HORIZONTAL = ("time", "level", "level_i", "plev", "mode")


def _horizontal_dims(da: xr.DataArray) -> list[str]:
    """Horizontal dims of ``da`` — all dims except time/level/level_i/mode."""
    return [d for d in da.dims if d not in _NON_HORIZONTAL]


def _lat_degrees(source) -> np.ndarray:
    """Latitudes [deg] from an xarray object (its ``lat`` coord) or an array."""
    if isinstance(source, (xr.Dataset, xr.DataArray)):
        return np.asarray(source["lat"].values, dtype=float)
    return np.asarray(source, dtype=float)


def area_weights(lat) -> xr.DataArray:
    """Horizontal-mean area weights for a latitude set.

    Accepts an :class:`xarray.Dataset`/:class:`xarray.DataArray` (reads its
    ``lat`` coordinate, in degrees) or a 1-D array of latitudes.

    When the latitudes are Gauss-Legendre quadrature nodes — matched by
    comparing ``sin(lat)`` against ``numpy.polynomial.legendre.leggauss``'s
    nodes after argsort, ``atol=1e-6`` — the **exact quadrature weights** are
    returned, reordered to the data's latitude order. That matters for
    conservation diagnostics: a transport residual whose quadrature integral
    should cancel then actually reads zero, whereas ``cos(lat)`` is only the
    leading approximation of those weights and biases meridionally structured
    fields. dinosaur output grids are Gauss-Legendre, so jcm output takes this
    branch. Any other grid falls back to ``cos(lat)`` weights.

    The returned DataArray carries dim ``("lat",)`` but no ``lat`` coordinate,
    so :meth:`xarray.DataArray.weighted` broadcasts it by dimension name
    without coordinate-value alignment (which float32 vs float64 latitude
    labels could otherwise disturb).
    """
    lat_deg = _lat_degrees(lat)
    sin_lat = np.sin(np.deg2rad(lat_deg))
    nodes, gauss_w = np.polynomial.legendre.leggauss(lat_deg.size)
    order = np.argsort(sin_lat)
    if np.allclose(sin_lat[order], nodes, atol=1e-6):
        weights = np.empty_like(gauss_w)
        weights[order] = gauss_w
    else:
        weights = np.cos(np.deg2rad(lat_deg))
    return xr.DataArray(weights, dims=("lat",))


def global_mean(da: xr.DataArray, weights: xr.DataArray | None = None,
                skipna: bool | None = None) -> xr.DataArray:
    """Area-weighted mean of ``da`` over its horizontal dims.

    Horizontal dims are all dims except time/level/level_i/mode (see
    :func:`_horizontal_dims`); time/level/... are kept intact. ``weights=None``
    computes :func:`area_weights` from ``da``'s ``lat`` coordinate when one
    exists, otherwise takes an unweighted mean. Uses
    :meth:`xarray.DataArray.weighted`.

    ``skipna`` is passed through to the reduction. xarray's default skips
    NaN cells and renormalises by the weights of the finite ones, which is
    right for a masked field but hides a partial blow-up; pass
    ``skipna=False`` where one NaN cell must make the mean NaN (a regression
    band, a health gate).
    """
    dims = _horizontal_dims(da)
    if weights is None and "lat" in da.coords:
        weights = area_weights(da)
    if weights is not None and "lat" in dims:
        return da.weighted(weights).mean(dims, skipna=skipna)
    return da.mean(dims, skipna=skipna)


def layer_pressure_thickness(ds: xr.Dataset) -> xr.DataArray:
    """Per-layer Δp [Pa] aligned with the 3-D fields' ``level`` orientation.

    The ``time`` dimension is **preserved** when present: Δp scales with the
    surface pressure, which evolves over the trajectory, so a single frozen
    profile would misweight every timestep but the first. Downstream,
    :func:`column_integral` / :func:`column_burden` then broadcast ``q·dp`` by
    dimension name and integrate the *correct* per-timestep layer masses.
    (The ``tools/jam_burden_report.py`` predecessor took ``isel(time=0)`` — a
    tolerated frozen-``t=0`` approximation in that CLI; as a public library
    function that would be a defect, so it is not carried over. Consumers that
    want a single number still reduce over time *after* the mass-weighted
    product, which is where the reduction belongs.)

    Prefer the model's own ``pressure_thickness`` diagnostic when present: it
    is written directly on the ``level`` axis, already aligned with the tracer
    fields, so there is no interface/mid-level differencing to get wrong. Take
    ``abs`` only to be sign-robust — it is emitted positive.

    Fall back to differencing ``pressure_half`` for post-#710 files written
    before ``pressure_thickness`` existed. Both output vertical axes run
    surface-first (#710), so differencing along ``level_i`` lands the result
    already aligned with the ``level`` axis — no orientation guard needed. The
    result carries the interface field's non-vertical coordinates (``time``,
    horizontal) but *no* level coordinate: after differencing, the ``level_i``
    interface values no longer describe the mid-layers, so attaching them (even
    renamed) would be a stale label.

    This targets current output only. Trajectories written before #710 stored
    interfaces TOA-first under a ``level_i`` bare index (dinosaur) or a
    ``level_interface`` dim (pyses); they are not supported here, and the
    convention change is called out in the release notes rather than
    compensated for at read time.
    """
    if "pressure_thickness" in ds:
        return np.abs(ds["pressure_thickness"])

    ph = ds["pressure_half"]
    axis = list(ph.dims).index("level_i")
    dp = np.abs(np.diff(np.asarray(ph.values), axis=axis))
    dims = tuple("level" if d == "level_i" else d for d in ph.dims)
    # Keep every coordinate that does NOT live on the interface axis (time,
    # horizontal); drop the level_i coordinate — its interface values do not
    # apply to the differenced mid-layers.
    coords = {name: coord for name, coord in ph.coords.items()
              if "level_i" not in coord.dims}
    return xr.DataArray(dp, dims=dims, coords=coords)


def column_integral(q: xr.DataArray, dp: xr.DataArray) -> xr.DataArray:
    """Mass-weighted column integral ``(q*dp).sum('level')/g`` over levels.

    For a mixing ratio ``q`` [kg/kg] and a layer pressure thickness ``dp``
    [Pa] this returns the column burden [kg/m²]. ``g`` is the live
    :mod:`jcm.constants` singleton (honours :func:`jcm.constants.set_constants`
    overrides — do not hardcode a literal).
    """
    return (q * dp).sum("level") / c.grav


def column_burden(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Column burden [kg/m²] of ``ds[var]`` using the file's own layer Δp."""
    return column_integral(ds[var], layer_pressure_thickness(ds))


def _interp_log_pressure(field, ln_pressure, target_ln_pressure, xp=np):
    """Batched, broadcasting log-pressure linear interpolation (#258).

    ``field``/``ln_pressure`` carry the vertical axis **last**
    (``(..., nlev)``, any leading batch shape -- the convention
    :func:`xarray.apply_ufunc` hands a gufunc with ``vectorize=False``, which
    is how :func:`to_pressure_levels` calls this); ``target_ln_pressure`` is
    ``(nplev,)``. Returns ``(..., nplev)``: ``field`` linearly interpolated in
    ln(pressure) onto ``target_ln_pressure``, ``NaN`` wherever a target lies
    outside that column's own ``[min, max]`` ln(pressure) range (below the
    surface, or above the model lid -- extrapolating past the ends of the
    actual profile is not a physically meaningful "pressure-level field"
    value).

    No assumption is made about ``ln_pressure``'s storage order (ascending,
    the model's own surface-first descending, or anything else): it is
    sorted internally, per column, before interpolating, so the result does
    not depend on it.

    ``xp`` is the array module -- plain ``numpy`` for the saved-output path
    below, or ``jax.numpy`` (inside ``jax.jit``/``jax.grad``, no Python
    loop over the batch shape) for a differentiable use. This is the same
    array-module-agnostic pattern :func:`total_cloud_cover` uses for
    :func:`jcm.physics.clouds.cloud_overlap.max_random_cover`.

    ``ln_pressure`` is promoted to ``target_ln_pressure``'s dtype before any
    comparison, whatever ``field``'s/``ln_pressure``'s own storage dtype:
    ``target_ln_pressure`` is always float64 (it comes from the exact,
    user-given ``levels_hpa``), and comparing it against a float32
    ``ln_pressure`` (jcm output is written float32 for most fields) rounds
    differently enough that an exact endpoint -- a target pressure equal to
    the column's own top or bottom level -- can land a hair outside the
    float32 range and get masked as NaN that a float64 comparison would not.
    The returned array is cast back to ``field``'s own dtype, so this
    function is dtype-preserving overall.
    """
    out_dtype = field.dtype
    ln_pressure = ln_pressure.astype(target_ln_pressure.dtype)
    nlev = ln_pressure.shape[-1]
    order = xp.argsort(ln_pressure, axis=-1)
    ln_p = xp.take_along_axis(ln_pressure, order, axis=-1)
    values = xp.take_along_axis(field, order, axis=-1)

    # `idx_hi[..., j]` counts how many levels' ln(pressure) are <= target j,
    # i.e. the index just past target j's lower bracket; clipped so the
    # bracket [i_lo, i_lo+1] always exists, including for a target outside
    # the column's range (masked separately below rather than by clipping
    # the index, which would otherwise silently extrapolate flat).
    cmp = ln_p[..., None, :] <= target_ln_pressure[:, None]   # (..., nplev, nlev)
    idx_hi = xp.sum(cmp, axis=-1)
    i_lo = xp.clip(idx_hi - 1, 0, nlev - 2)
    i_hi = i_lo + 1

    p_lo = xp.take_along_axis(ln_p, i_lo, axis=-1)
    p_hi = xp.take_along_axis(ln_p, i_hi, axis=-1)
    f_lo = xp.take_along_axis(values, i_lo, axis=-1)
    f_hi = xp.take_along_axis(values, i_hi, axis=-1)
    fraction = (target_ln_pressure - p_lo) / (p_hi - p_lo)
    interpolated = f_lo + fraction * (f_hi - f_lo)

    p_min = ln_p[..., :1]
    p_max = ln_p[..., -1:]
    out_of_range = (target_ln_pressure < p_min) | (target_ln_pressure > p_max)
    nan = xp.asarray(np.nan, dtype=interpolated.dtype)
    return xp.where(out_of_range, nan, interpolated).astype(out_dtype)


def _nearest_level_value(field, ln_pressure, target_ln_pressure, fill_value,
                         xp=np):
    """Nearest-model-level (in ln(pressure)) selection for categorical fields.

    :func:`to_pressure_levels` calls this instead of
    :func:`_interp_log_pressure` for an integer or boolean ``field``: a
    categorical diagnostic (a flag, a land/ocean/ice tile index) has no
    meaningful value "40% of the way between" two model levels the way a
    continuous physical field does, so it is NOT interpolated -- it takes
    the value of whichever model level's ln(pressure) is closest to each
    target, exactly, with no blending. Same batching convention, same
    column-order independence and the same out-of-range test as
    :func:`_interp_log_pressure` (below the surface or above the model
    lid); there, that range masks the result ``NaN``, which neither an
    int nor a bool array can hold, so it is ``fill_value`` here instead
    (:func:`to_pressure_levels` documents the mask in the variable's
    attrs, since the array itself cannot carry it).
    """
    order = xp.argsort(ln_pressure, axis=-1)
    ln_p = xp.take_along_axis(ln_pressure, order, axis=-1)
    values = xp.take_along_axis(field, order, axis=-1)

    distance = xp.abs(ln_p[..., None, :] - target_ln_pressure[:, None])
    nearest = xp.argmin(distance, axis=-1)
    selected = xp.take_along_axis(values, nearest, axis=-1)

    p_min = ln_p[..., :1]
    p_max = ln_p[..., -1:]
    out_of_range = (target_ln_pressure < p_min) | (target_ln_pressure > p_max)
    fill = xp.asarray(fill_value, dtype=selected.dtype)
    return xp.where(out_of_range, fill, selected)


def validate_pressure_levels(levels_hpa) -> np.ndarray:
    """Validate ``run.pressure_levels``/``to_pressure_levels``'s ``levels_hpa``.

    Shared by every entry point that takes this list -- :func:`to_pressure_levels`
    itself, and the config-time check ``jcm.runners`` makes before any
    integration starts -- so a typo (a negative or repeated pressure, a
    stray ``NaN``) is refused the same way and at the earliest point
    regardless of which door it came in by.

    Returns the list as a 1-D float64 array (hPa, in the given order --
    never re-sorted or de-duplicated for the caller).

    Raises
    ------
    ValueError
        If ``levels_hpa`` is not a non-empty 1-D sequence, contains a
        non-finite value (``NaN``/``inf``), a non-positive pressure (a
        pressure is a magnitude; log-pressure interpolation is undefined
        for a vacuum or negative one), or repeats a pressure (an
        unambiguous caller error -- the repeat could only ever read back
        the same column values twice).

    """
    levels_hpa = np.asarray(levels_hpa, dtype=np.float64)
    if levels_hpa.ndim != 1 or levels_hpa.size == 0:
        raise ValueError(
            "levels_hpa must be a non-empty 1-D list of pressures in hPa; "
            f"got shape {levels_hpa.shape}.")
    if not np.all(np.isfinite(levels_hpa)):
        raise ValueError(
            "levels_hpa must be finite (no NaN/inf); got "
            f"{levels_hpa.tolist()!r}.")
    if np.any(levels_hpa <= 0):
        raise ValueError(
            "levels_hpa must be positive pressures in hPa; got "
            f"{levels_hpa.tolist()!r}.")
    if np.unique(levels_hpa).size != levels_hpa.size:
        raise ValueError(
            "levels_hpa must not repeat a pressure; got "
            f"{levels_hpa.tolist()!r}.")
    return levels_hpa


#: ``level``-dimensioned variables that describe the *sigma* grid itself
#: rather than a physical field sampled on it -- interpolating them onto
#: ``plev`` would be meaningless (``pressure_full`` would just read back
#: close to the target pressures it was interpolated against) or actively
#: wrong (``pressure_thickness`` is a sigma-layer thickness, not a
#: per-pressure-level quantity). Left as they were, describing the sigma
#: grid :func:`to_pressure_levels` interpolated *from*. Height on pressure
#: surfaces (``height_full``, e.g. "Z500") is a standard, useful diagnostic
#: and is deliberately *not* in this set -- it gets interpolated like any
#: other field.
_SIGMA_GRID_VARS = frozenset({"pressure_full", "pressure_thickness"})


def to_pressure_levels(ds: xr.Dataset, levels_hpa) -> xr.Dataset:
    """Interpolate every ``level``-dimensioned field onto pressure surfaces.

    Log-pressure linear interpolation (:func:`_interp_log_pressure`) from the
    model's own ``pressure_full`` [Pa] -- which varies with both column and
    saved time, so it is read fresh per time/column rather than assumed
    static -- onto the fixed list ``levels_hpa`` [hPa]. A point whose target
    pressure falls below that column's surface or above the model lid is
    ``NaN``; both ends are masked, not only the surface, because
    extrapolating past either end of the actual profile would not describe a
    real pressure-level field (#258 asks for sub-surface masking; masking
    above the lid too is the same reasoning applied consistently -- see the
    PR description for the tradeoff).

    An integer or boolean ``level``-dimensioned field -- a categorical
    diagnostic such as a surface-tile index or a convective-type flag, the
    same class ``ModelPredictions.to_xarray`` already excludes from
    interval-mean output because it has no defined mean -- is likewise not
    linearly interpolated: it takes the value of whichever model level's
    ln(pressure) is closest to the target (:func:`_nearest_level_value`),
    with its dtype preserved. An out-of-range cell there is filled rather
    than ``NaN``, which neither dtype can hold: ``False`` for a boolean
    field, ``-1`` for a signed integer one, or that dtype's maximum
    representable value for an UNSIGNED integer one (``-1`` is not
    representable there; NumPy 2 raises rather than silently wrapping it).
    The variable's ``comment`` attribute states the exact value used.

    This is a pure function of ``ds`` -- the option that applies it at save
    time (``run.pressure_levels``, see :ref:`monthly-means-cli`'s neighbour
    in :doc:`running_at_scale`) calls this same function on the dataset about
    to be written, so a saved run and a later call on its netCDF agree
    exactly. Sigma (``level``) output is untouched by calling this: it
    returns a new :class:`xarray.Dataset` built from ``ds``, the same
    pattern as every other function in this module.

    Parameters
    ----------
    ds : xarray.Dataset
        Must carry ``pressure_full`` [Pa] on the same ``level`` dimension as
        the fields being converted (the model's layer pressures, surface-
        first per ``docs/source/design/output_vertical_conventions.md`` --
        but see "No assumption" above, the actual order does not matter
        here).
    levels_hpa : sequence of float
        Target pressures in hPa, in whatever order the caller wants them to
        appear on the new ``plev`` axis (not re-sorted). Converted to Pa
        before interpolating, so the new ``plev`` coordinate is in the same
        units as ``pressure_full`` -- not hPa -- matching every other
        pressure variable already in a jcm output file.

    Returns
    -------
    xarray.Dataset
        Every variable that had a ``level`` dimension now has ``plev``
        instead (others, including ``pressure_full``/``pressure_half``
        themselves, pass through unchanged — they describe the *sigma*
        grid this was interpolated from, not the new one). A floating
        variable is linearly interpolated; an integer/boolean one takes
        its nearest model level instead, with an out-of-range cell filled
        rather than ``NaN`` and a ``comment`` attribute saying so. ``plev``
        carries CF attributes (``standard_name="air_pressure"``,
        ``units="Pa"``, ``positive="down"``, ``axis="Z"``).

    """
    if "pressure_full" not in ds:
        raise ValueError(
            "to_pressure_levels needs ds['pressure_full'] [Pa] -- the "
            "model's own column pressures -- to interpolate from.")
    if "level" not in ds["pressure_full"].dims:
        raise ValueError(
            "ds['pressure_full'] has no 'level' dimension to interpolate "
            f"over (dims: {ds['pressure_full'].dims}).")
    levels_hpa = validate_pressure_levels(levels_hpa)
    target_pa = levels_hpa * 100.0
    target_ln_pa = np.log(target_pa)

    # float64, BEFORE the log -- jcm writes most level fields
    # (``pressure_full`` included) float32, and ``np.log`` of a float32
    # array rounds to float32 precision at the point it is taken;
    # promoting the already-rounded result afterwards (as
    # ``_interp_log_pressure`` still does, for a caller that hands it a
    # float32 ``ln_pressure`` directly) cannot recover precision already
    # lost. Comparing that rounded value against the float64 target below
    # can otherwise mask an exact profile endpoint as a hair out of range.
    #
    # How that cast is done depends on what is backing ``pressure_full``:
    #
    # - A plain NumPy or **Dask** array: ``.astype(np.float64)`` is exact
    #   and, for Dask, stays LAZY -- the cast becomes one more graph node,
    #   not a trigger to materialise the whole array. A saved run opened
    #   with ``xr.open_mfdataset(...)`` must stay lazy through this
    #   conversion the same way it does through ``total_cloud_cover``.
    # - A live **JAX** array -- the call this option makes right before
    #   ``ds.to_netcdf(...)`` (``run.pressure_levels``) sees exactly that,
    #   ``predictions.to_xarray()`` not having gone through a save/load
    #   round trip -- is different: with the process-wide
    #   ``jax_enable_x64`` default (off throughout jcm), JAX's own
    #   ``astype`` silently keeps it float32, so the precision is lost
    #   regardless. ``.values`` forces a host NumPy copy first (unlike
    #   Dask, a JAX array has no useful "lazy" state to lose by doing so,
    #   and this is exactly the device-to-host transfer ``ds.to_netcdf``
    #   needs in a moment anyway), and NumPy's own ``astype`` on that real
    #   host array is unrestricted.
    import jax

    pressure_full = ds["pressure_full"]
    if isinstance(pressure_full.data, jax.Array):
        pressure_full = pressure_full.copy(
            data=np.asarray(pressure_full.values, dtype=np.float64))
        ln_pressure_full = np.log(pressure_full)
    else:
        ln_pressure_full = np.log(pressure_full.astype(np.float64))
    nplev = target_pa.size

    def _interp(field_values, ln_p_values):
        return _interp_log_pressure(field_values, ln_p_values, target_ln_pa)

    data_vars = {}
    for name, var in ds.data_vars.items():
        if "level" not in var.dims or name in _SIGMA_GRID_VARS:
            data_vars[name] = var
            continue
        # Categorical (int/bool) diagnostics -- a land/ocean/ice tile
        # index, a convective-type flag -- are kept in instantaneous
        # output precisely because they have no defined mean
        # (``ModelPredictions.to_xarray``'s ``omitted_interval_mean_
        # variables``); the same is true of a *blend* of two levels, so
        # these take the nearest model level instead of being linearly
        # interpolated (:func:`_nearest_level_value`), and an
        # out-of-range cell -- which ``_interp_log_pressure`` marks NaN --
        # is filled with ``fill_value`` (neither dtype can hold NaN),
        # documented on the variable rather than carried as a separate
        # mask array.
        if var.dtype.kind in "iub":
            # -1 is not representable in an UNSIGNED dtype (jcm's own
            # land/ocean/ice tile indices and bit-flag diagnostics are
            # typically uint8/uint32): NumPy 2 raises OverflowError
            # casting it rather than silently wrapping, so the sentinel
            # has to be chosen per dtype, not just per kind.
            if var.dtype.kind == "b":
                fill_value = False
            elif var.dtype.kind == "u":
                fill_value = np.iinfo(var.dtype).max
            else:
                fill_value = -1

            def _nearest(field_values, ln_p_values, fill_value=fill_value):
                return _nearest_level_value(
                    field_values, ln_p_values, target_ln_pa, fill_value)

            kernel, extra_attrs = _nearest, {
                "comment": (
                    "Categorical field: value of the nearest model level "
                    "in ln(pressure), not interpolated. Cells below the "
                    f"surface or above the model lid are filled with "
                    f"{fill_value!r}.")}
        else:
            kernel, extra_attrs = _interp, {}
        # ``dask="parallelized"`` (plus the output's new chunk size, which
        # ``apply_ufunc`` cannot infer since ``plev`` replaces ``level``
        # with a different length): a saved run opened lazily, e.g.
        # ``xr.open_mfdataset(...)`` as the release-validation gate does for
        # ``total_cloud_cover``, converts without first loading every
        # chunk into memory -- both kernels run per dask block exactly as
        # they do on a plain NumPy array, since a block IS one.
        interpolated = xr.apply_ufunc(
            kernel, var, ln_pressure_full,
            input_core_dims=[["level"], ["level"]],
            output_core_dims=[["plev"]],
            exclude_dims={"level"},
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"plev": nplev}},
            output_dtypes=[var.dtype],
        )
        interpolated.attrs = {**dict(var.attrs), **extra_attrs}
        data_vars[name] = interpolated

    out = xr.Dataset(data_vars, attrs=dict(ds.attrs))
    out = out.assign_coords(plev=("plev", target_pa))
    out["plev"].attrs = {
        "standard_name": "air_pressure",
        "long_name": "pressure",
        "units": "Pa",
        "positive": "down",
        "axis": "Z",
    }
    return out


def total_cloud_cover(cloud_fraction: xr.DataArray,
                      dim: str = "level") -> xr.DataArray:
    r"""Total cloud cover [1] under ECHAM's maximum-random overlap.

    This is ECHAM's own ``aclcov``: vertically contiguous cloud is treated as
    maximally overlapped and cloud separated by clear air as randomly
    overlapped. The clear-sky fraction accumulates over the column as

    .. math::

        C_\mathrm{clear} = (1 - c_0)\;
            \prod_{k=1}^{n-1}
            \frac{1 - \max(c_k, c_{k-1})}{\max(1 - c_{k-1}, \epsilon)},
        \qquad \mathrm{aclcov} = 1 - C_\mathrm{clear},

    with the product running over ``k = 1 … n-1`` — the Fortran's ``DO 923``
    loop, ``jk = 2, klev`` in ECHAM6 and ``jk = jks+1, klev`` in ICON, where
    ``jks`` is the first active level and is 1 for a full column. This is a
    transcription of
    ``mo_cloud.f90`` section "10.2 Total cloud cover" (ICON
    ``atm_phy_echam/mo_cloud.f90`` lines 1165-1182; the same loop is ECHAM6
    ``mo_cloud.f90`` lines 1359-1383, whose ``paclcov`` is a time accumulation
    of it and whose ``aclcov_na`` is this instantaneous value), with the
    Fortran's denominator ``1 - min(c_{k-1}, 1 - epsilon)`` written as the
    equal ``max(1 - c_{k-1}, epsilon)``, which does not collapse to 0/0 in
    float32. The one addition to the Fortran is the ``[0, 1]`` clip of the
    input: saved output can carry small out-of-range excursions that the
    in-model ``paclc`` never has, and an unclipped ``c > 1`` would make the
    numerator negative and the "clear-sky fraction" meaningless.

    The recurrence is :func:`jcm.physics.clouds.cloud_overlap.max_random_cover`,
    the one the physics step accumulates into ``clouds.total_cloud_cover``
    online (instantaneous cover, time-averaged under ``output_averages``, as
    ECHAM's ``paclcov``). This function is its offline counterpart for output
    that does not carry that field: a cover of a *saved time-mean* profile,
    which differs from the time mean of the instantaneous cover (the product
    is non-linear in ``cloud_fraction``; see
    ``docs/source/design/cloud_cover_gate.md``).

    The product is built **level by level with lazy xarray slices**, so a
    dask-backed array (anything opened with
    :func:`xarray.open_mfdataset`, as the release-validation gate does) stays
    lazy and is evaluated chunk by chunk. Nothing here calls ``.values`` or
    otherwise forces the whole window into memory, and on a numpy-backed array
    the same loop holds one horizontal slice at a time rather than
    materialising every pair factor of a year of output at once.

    **Why this quantity.** Two cheaper reductions of a cloud-fraction profile
    bracket it but neither is the cover a climate model reports:

    * the **column maximum**, ``cloud_fraction.max(dim)``, is only a *lower*
      bound — it assumes every layer overlaps maximally, so two half-covered
      decks in different parts of the column read 0.5 rather than 0.75;
    * **random overlap**, ``1 - prod(1 - c_k)``, is the *upper* bound — it
      ignores that a physically continuous cloud spans several model layers
      and so double-counts its edges.

    Maximum-random sits between them, is deterministic, is computable from any
    saved output (only ``cloud_fraction`` is needed), is the construction
    ECHAM6 itself uses for ``aclcov`` (Stevens et al. 2013,
    doi:10.1002/jame.20015), and is a *total* cover — the basis the
    satellite climatologies are quoted on. On two archived T63 L47
    ECHAM+RRTMGP year runs, settled window, area-weighted, the three
    definitions gave for the 2M / 1M members: column max 0.553 / 0.563, this
    function 0.701 / 0.691, random overlap 0.847 / 0.823. For scale, the GEWEX
    Cloud Assessment puts the observed global cloud amount at 0.68 +/- 0.03
    for optical depth > 0.1 (0.56 to 0.74 across detection thresholds;
    Stubenrauch et al. 2013, doi:10.1175/BAMS-D-12-00117.1). Those runs
    predate #690/#707, so they calibrate the definitions against each other
    and are not validation numbers for current code — the full table, its
    provenance and its caveats are in
    ``docs/source/design/cloud_cover_gate.md``.

    **Orientation.** The result does not depend on which end of ``dim`` is the
    surface. Cancelling the denominators leaves the clear-sky product as the
    adjacent-pair factors ``1 - max(c_k, c_{k-1})`` divided by the *interior*
    levels' ``1 - c_k``, and both sets are invariant under reversing the axis —
    so no surface-first/TOA-first guard is needed, and pre-#710 files (whose
    vertical conventions are described in
    ``docs/source/design/output_vertical_conventions.md``) score the same as
    current ones. The two orders agree to rounding, differing by at most
    ``O(zepsec)``: the ``max(1 - c_{k-1}, zepsec)`` guard is applied in loop
    order, so it caps a *different* denominator in the reversed column and a
    profile holding cover within ``1e-12`` of 1 (say ``[0.2, 1 - 5e-13]``) can
    differ in that last digit. The cancellation argument itself is exact.

    Parameters
    ----------
    cloud_fraction : xarray.DataArray
        Grid-mean cloud fraction, any dimensionality, with a vertical axis
        named ``dim``. May be dask-backed; the result then is too. Values are
        clipped to ``[0, 1]`` before use, out of place — the caller's array is
        never modified. ``NaN`` propagates: a column with a missing level
        scores ``NaN`` rather than a cover computed from the rest of it.
    dim : str, optional
        Name of the vertical dimension to reduce over. Default ``"level"``.

    Returns
    -------
    xarray.DataArray
        Total cloud cover in ``[0, 1]``, named ``total_cloud_cover`` and
        carrying CF attributes (``standard_name`` ``cloud_area_fraction``),
        with ``dim`` (and any coordinate defined on it) removed and every other
        dimension and coordinate preserved.

    Notes
    -----
    The overlap product is **non-linear** in ``cloud_fraction``, so a time or
    area mean must be taken of this function's output, never of its input.

    """
    if dim not in cloud_fraction.dims:
        raise ValueError(
            f"{dim!r} is not a dimension of the cloud fraction "
            f"(dims: {cloud_fraction.dims})")
    if cloud_fraction.sizes[dim] == 0:
        raise ValueError(
            f"the cloud fraction has no levels along {dim!r}; a column with "
            "no layers has no cover to compute")

    # Drop the coordinates defined on the reduced axis (``level`` itself, and
    # any auxiliary coordinate that varies with it) *before* slicing, so the
    # per-level slices below carry no conflicting scalar ``level`` coordinate
    # and align positionally. The rest of the coordinates ride through.
    c = cloud_fraction.drop_vars(
        [name for name, coord in cloud_fraction.coords.items()
         if dim in coord.dims])

    def level(k):
        """Level ``k`` of the cloud fraction, clipped, as float64.

        float64 so the product of up to ~100 factors accumulates in double
        precision whatever precision the file was saved in. Per level rather
        than on the whole array so the numpy path's peak stays at a couple of
        horizontal slices; ``astype`` and ``clip`` are both out of place, so
        the caller's array is never modified, and on a dask array both are
        lazy, so nothing is loaded here at all.
        """
        return c.isel({dim: k}).astype(np.float64).clip(0.0, 1.0)

    # The recurrence is the in-model one (the module that holds it explains
    # the denominator and why every factor lies in [0, 1]), so the cover the
    # physics step accumulates and the cover scored here cannot drift apart.
    cover = max_random_cover(level, c.sizes[dim], np)
    cover.attrs = {
        "standard_name": "cloud_area_fraction",
        "long_name": "total cloud cover (maximum-random overlap)",
        "units": "1",
    }
    return cover.rename("total_cloud_cover")
