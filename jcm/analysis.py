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
(``docs/source/design/cloud_cover_gate.md``).

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


#: ECHAM's ``zepsec`` security epsilon (``mo_cloud.f90``, "Security
#: parameters": ``zepsec = 1.0e-12``). The overlap denominator uses
#: ``zxsec = 1 - zepsec`` so a cell with cover exactly 1 divides by 1e-12
#: rather than by zero; its numerator is zero there, so the factor is zero.
_ZEPSEC = 1.0e-12


def total_cloud_cover(cloud_fraction: xr.DataArray,
                      dim: str = "level") -> xr.DataArray:
    r"""Total cloud cover [1] under ECHAM's maximum-random overlap.

    This is ECHAM's own ``aclcov``: vertically contiguous cloud is treated as
    maximally overlapped and cloud separated by clear air as randomly
    overlapped. The clear-sky fraction accumulates over the column as

    .. math::

        C_\mathrm{clear} = (1 - c_0)\;
            \prod_{k=1}^{n-1}
            \frac{1 - \max(c_k, c_{k-1})}{1 - \min(c_{k-1}, 1-\epsilon)},
        \qquad \mathrm{aclcov} = 1 - C_\mathrm{clear},

    a transcription of ``mo_cloud.f90`` section "10.2 Total cloud cover"
    (ICON ``atm_phy_echam/mo_cloud.f90`` lines 1165-1182; the same loop is
    ECHAM6 ``mo_cloud.f90`` lines 1359-1381). The running product is evaluated
    level by level exactly as the Fortran's ``DO 923`` loop does, which also
    keeps peak memory at one horizontal slice rather than materialising every
    pair factor of a full year of output at once.

    **Why this quantity.** Two cheaper reductions of a cloud-fraction profile
    bracket it but neither is the cover a climate model reports:

    * the **column maximum**, ``cloud_fraction.max(dim)``, is only a *lower*
      bound — it assumes every layer overlaps maximally, so two half-covered
      decks in different parts of the column read 0.5 rather than 0.75;
    * **random overlap**, ``1 - prod(1 - c_k)``, is the *upper* bound — it
      ignores that a physically continuous cloud spans several model layers
      and so double-counts its edges.

    Maximum-random sits between them, is deterministic, is computable from any
    saved output (only ``cloud_fraction`` is needed), and is the definition the
    reference model and the satellite products are quoted on. On two July-2026
    T63 L47 ECHAM+RRTMGP year runs (last chunk, area-weighted, pre-#690 code,
    so indicative magnitudes only) the three definitions gave, for the 2M / 1M
    members: column max 0.546 / 0.559, this function 0.682 / 0.665, random
    overlap 0.835 / 0.798. The middle pair is the one that lands on ECHAM6's
    climatological total cover (~0.62-0.65) and near the satellite estimates
    (ISCCP/MODIS ~0.66-0.67, CALIPSO-GOCCP ~0.70).

    **Orientation.** The result does not depend on which end of ``dim`` is the
    surface. Cancelling the denominators leaves the clear-sky product as the
    adjacent-pair factors ``1 - max(c_k, c_{k-1})`` divided by the *interior*
    levels' ``1 - c_k``, and both sets are invariant under reversing the axis —
    so no surface-first/TOA-first guard is needed, and pre-#710 files (whose
    vertical conventions are described in
    ``docs/source/design/output_vertical_conventions.md``) score identically to
    current ones. Only floating-point rounding distinguishes the two orders.

    Parameters
    ----------
    cloud_fraction : xarray.DataArray
        Grid-mean cloud fraction, any dimensionality, with a vertical axis
        named ``dim``. Values are clipped to ``[0, 1]`` before use.
    dim : str, optional
        Name of the vertical dimension to reduce over. Default ``"level"``.

    Returns
    -------
    xarray.DataArray
        Total cloud cover, with ``dim`` (and any coordinate defined on it)
        removed and every other dimension and coordinate preserved.

    Notes
    -----
    The overlap product is **non-linear** in ``cloud_fraction``, so a time or
    area mean must be taken of this function's output, never of its input.

    """
    if dim not in cloud_fraction.dims:
        raise ValueError(
            f"{dim!r} is not a dimension of the cloud fraction "
            f"(dims: {cloud_fraction.dims})")

    axis = list(cloud_fraction.dims).index(dim)
    # float64 throughout, in one allocation. ``zxsec`` is not representable in
    # float32 (it rounds to 1.0, turning an overcast layer's guarded 0/1e-12
    # into 0/0), so the working copy has to be double precision; and it has to
    # be a genuine copy, since the in-place clip must not reach back into the
    # caller's Dataset the way ``np.asarray`` on float64 input would allow.
    c = np.array(cloud_fraction.values, dtype=np.float64)
    np.clip(c, 0.0, 1.0, out=c)
    c = np.moveaxis(c, axis, 0)

    zxsec = 1.0 - _ZEPSEC
    clear = 1.0 - c[0]
    for k in range(1, c.shape[0]):
        clear = clear * ((1.0 - np.maximum(c[k], c[k - 1]))
                         / (1.0 - np.minimum(c[k - 1], zxsec)))

    dims = tuple(d for d in cloud_fraction.dims if d != dim)
    # Drop coordinates defined on the reduced axis (``level`` itself, and any
    # auxiliary coordinate that varies with it); keep the rest.
    coords = {name: coord for name, coord in cloud_fraction.coords.items()
              if dim not in coord.dims}
    return xr.DataArray(1.0 - clear, dims=dims, coords=coords,
                        name="total_cloud_cover")
