"""xarray/post-processing layer for model OUTPUT.

The one home for area weights, global means, layer pressure thicknesses and
column burdens computed on saved jcm netCDF output. Issue #640 found each of
these independently reimplemented at least four times
(``tools/jam_burden_report.py``, ``tools/release_validation/health.py``,
``tools/validate_era5_bundle.py`` and inline in
:func:`jcm.runners.run_chunked`); this module is where they now live once.

Everything here takes labelled xarray in and returns xarray (or a Python
``float``) out, using numpy internally. It operates on *saved* output — never
on device arrays inside a jitted physics step. The in-model, device-array
column integral lives separately in
:func:`jcm.physics.diagnostics.aerocom._column_integral`, because physics runs
on JAX arrays inside ``jit`` (it integrates over ``pressure_half`` interfaces
in the physics-internal frame). The two are deliberately kept apart —
cross-reference, do not merge.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import jcm.constants as c

#: Dimensions that are never horizontal. The horizontal dims of a field are
#: everything *else* — this is the convention promoted from
#: ``tools/jam_burden_report.py::_horizontal_dims``.
_NON_HORIZONTAL = ("time", "level", "level_i", "mode")


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


def global_mean(da: xr.DataArray, weights: xr.DataArray | None = None
                ) -> xr.DataArray:
    """Area-weighted mean of ``da`` over its horizontal dims.

    Horizontal dims are all dims except time/level/level_i/mode (see
    :func:`_horizontal_dims`); time/level/... are kept intact. ``weights=None``
    computes :func:`area_weights` from ``da``'s ``lat`` coordinate when one
    exists, otherwise takes an unweighted mean. Uses
    :meth:`xarray.DataArray.weighted`.
    """
    dims = _horizontal_dims(da)
    if weights is None and "lat" in da.coords:
        weights = area_weights(da)
    if weights is not None and "lat" in dims:
        return da.weighted(weights).mean(dims)
    return da.mean(dims)


def layer_pressure_thickness(ds: xr.Dataset) -> xr.DataArray:
    """Per-layer Δp [Pa] aligned with the 3-D fields' ``level`` orientation.

    Prefer the model's own ``pressure_thickness`` diagnostic when present: it
    is written directly on the ``level`` axis, already aligned with the tracer
    fields, so there is no interface/mid-level differencing to get wrong. Take
    ``abs`` only to be sign-robust — it is emitted positive.

    Fall back to differencing ``pressure_half`` for post-#710 files written
    before ``pressure_thickness`` existed. Both output vertical axes run
    surface-first (#710), so differencing along ``level_i`` lands the result
    already aligned with the ``level`` axis — no orientation guard needed.

    This targets current output only. Trajectories written before #710 stored
    interfaces TOA-first under a ``level_i`` bare index (dinosaur) or a
    ``level_interface`` dim (pyses); they are not supported here, and the
    convention change is called out in the release notes rather than
    compensated for at read time.
    """
    if "pressure_thickness" in ds:
        dp = ds["pressure_thickness"]
        if "time" in dp.dims:
            dp = dp.isel(time=0)
        return np.abs(dp)

    ph = ds["pressure_half"]
    if "time" in ph.dims:
        ph = ph.isel(time=0)
    axis = list(ph.dims).index("level_i")
    dp = np.abs(np.diff(np.asarray(ph.values), axis=axis))
    dims = tuple("level" if d == "level_i" else d for d in ph.dims)
    return xr.DataArray(dp, dims=dims)


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
