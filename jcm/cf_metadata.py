"""CF-Convention metadata for JCM output, and the one vertical convention.

Every JCM output file uses a **single vertical direction: surface-first.**
Index 0 of both vertical axes — ``level`` (layer mid-levels, length ``nlev``)
and ``level_i`` (layer interfaces, length ``nlev+1``) — is the one nearest the
ground, and the sigma coordinate decreases with index towards the model top.
The physics-internal frame and the ECHAM/HAMMOZ input tables are the opposite
way round (TOA-first); this module is the single place where output is turned
into the file convention, so the two frames cannot drift apart per-variable.

That mattered: before this module the full-level axis was flipped on output
while the interface axis was left TOA-first, so ``diff(pressure_half)`` and any
``level``-dimensioned field ran in opposite directions in the same file, with
nothing in the metadata to say so (#710). Pairing them — a heating rate from
the saved fluxes against the saved temperature, a mass-weighted burden — came
out vertically reversed and plausible-looking.

What makes the file self-describing now:

* both vertical axes are real coordinate variables carrying ``positive`` and
  ``axis``, so the direction is *stated* rather than inferred by comparing
  pressures at each end;
* the hybrid ``(a, b)`` tables travel with the file as coordinate variables in
  the file's own (surface-first) order, so ``p = a + b·p_s`` is reproducible
  from the file alone, and CF ``formula_terms`` names them. The parametric
  ``standard_name`` (``atmosphere_hybrid_sigma_pressure_coordinate``) is stamped
  only when those ``formula_terms`` can be emitted — CF-1.11 §4.3.3 requires the
  two together, so a dynamics-only file (no ``surface_pressure`` to reference)
  gets a plain units+positive vertical axis, which is still CF-conformant,
  rather than a parametric coordinate a checker would reject.

``positive`` describes the direction in which the coordinate *values* increase
(CF-1.11 §4.3), not the storage order — a sigma coordinate is ``positive =
"down"`` whichever way it is stored. The storage order is readable from the
coordinate values themselves (they descend from ~1 to ~0), and is spelled out
in ``long_name`` for a human skimming ``ncdump -h``.

``positive`` is deliberately *not* set on ``pressure_full`` / ``pressure_half``
/ ``height_*``: CF defines it for vertical coordinate variables, and those are
data variables. Their orientation is the orientation of the axis they are on,
which is now the same axis convention as everything else in the file.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: Name of the full-level (layer mid-level) vertical dimension.
LEVEL_DIM = "level"
#: Name of the interface (layer boundary) vertical dimension. Both dycore
#: backends use this name; pyses previously wrote ``level_interface``, which
#: forced readers to try both (#710).
LEVEL_INTERFACE_DIM = "level_i"

#: Names of the hybrid-coefficient coordinate variables written alongside the
#: two vertical axes, in the file's surface-first order.
HYBRID_A_FULL = "hybrid_a_full"
HYBRID_B_FULL = "hybrid_b_full"
HYBRID_A_HALF = "hybrid_a_half"
HYBRID_B_HALF = "hybrid_b_half"

_SIGMA_COMMON_ATTRS = {
    "units": "1",
    "positive": "down",
    "axis": "Z",
}

#: The parametric ``standard_name`` a hybrid/sigma vertical axis carries — but
#: only when it is accompanied by ``formula_terms``. CF-1.11 §4.3.3 requires a
#: parametric vertical coordinate to name its ``formula_terms``, so a checker
#: rejects this ``standard_name`` on an axis that cannot (a dynamics-only file
#: has no ``surface_pressure`` to reference). Applied conditionally in
#: :func:`apply_cf_attributes` rather than living in ``_SIGMA_COMMON_ATTRS``.
#: Pure-sigma grids are the ``a = 0`` special case of the hybrid formula, so one
#: standard_name covers both coordinate families the model supports.
_PARAMETRIC_SIGMA_STANDARD_NAME = "atmosphere_hybrid_sigma_pressure_coordinate"

#: Attributes for variables JCM writes that have a CF standard name. This table
#: deliberately covers only the vertical-coordinate neighbourhood and the core
#: prognostics, and is applied LAST (in ``ModelPredictions._trajectory_dataset``)
#: so these curated names win over both the per-physics units CSVs and the
#: per-term declarations. Physics diagnostics (the radiation fluxes among them)
#: now declare their own units next to the code that computes them via
#: :attr:`jcm.physics.physics_term.PhysicsTerm.output_attrs` (#740); a variable
#: absent from every source simply carries whatever attrs it already had.
_VARIABLE_ATTRS: dict[str, dict[str, str]] = {
    "pressure_full": {
        "standard_name": "air_pressure",
        "units": "Pa",
        "long_name": "air pressure at layer mid-level",
    },
    "pressure_half": {
        "standard_name": "air_pressure",
        "units": "Pa",
        "long_name": "air pressure at layer interface",
    },
    "height_full": {
        "standard_name": "geopotential_height",
        "units": "m",
        "long_name": "geopotential height at layer mid-level",
    },
    "height_half": {
        "standard_name": "geopotential_height",
        "units": "m",
        "long_name": "geopotential height at layer interface",
    },
    "surface_pressure": {
        "standard_name": "surface_air_pressure",
        "units": "Pa",
        "long_name": "surface air pressure",
    },
    "temperature": {
        "standard_name": "air_temperature",
        "units": "K",
        "long_name": "air temperature",
    },
    "specific_humidity": {
        "standard_name": "specific_humidity",
        "units": "kg kg-1",
        "long_name": "specific humidity",
    },
    "u_wind": {
        "standard_name": "eastward_wind",
        "units": "m s-1",
        "long_name": "eastward wind",
    },
    "v_wind": {
        "standard_name": "northward_wind",
        "units": "m s-1",
        "long_name": "northward wind",
    },
}

#: Attributes for the non-vertical coordinate axes.
_COORD_ATTRS: dict[str, dict[str, str]] = {
    "lon": {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
        "long_name": "longitude",
    },
    "lat": {
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
        "long_name": "latitude",
    },
    # ``units`` for time is owned by xarray's datetime encoding — setting it
    # here would collide with the encoding on write. Applied only to a
    # datetime64 axis; see ``_time_attrs``.
    "time": {"standard_name": "time", "axis": "T", "long_name": "time"},
    # Axes with no CF standard name; a long_name is all CF asks for.
    "mode": {"long_name": "aerosol mode"},
    "sw_band": {"long_name": "shortwave spectral band index"},
    "lw_band": {"long_name": "longwave spectral band index"},
    "longitudinal_mode": {"long_name": "zonal wavenumber"},
    "total_wavenumber": {"long_name": "total wavenumber"},
    "surface": {"long_name": "singleton surface axis"},
    "realization": {"long_name": "ensemble realization"},
    "sample": {"long_name": "sample index"},
}


def hybrid_boundaries(vertical) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a_boundaries [Pa], b_boundaries)`` for a dinosaur vertical.

    TOA-first, as the coordinate objects store them. ``SigmaCoordinates`` is
    the ``a = 0`` case, which keeps one code path for both families.
    """
    if hasattr(vertical, "a_boundaries"):
        a = np.asarray(vertical.a_boundaries, dtype=float)
        b = np.asarray(vertical.b_boundaries, dtype=float)
    else:
        b = np.asarray(vertical.boundaries, dtype=float)
        a = np.zeros_like(b)
    return a, b


def orient_surface_first(ds):
    """Reverse both vertical axes of ``ds`` so index 0 is the surface.

    Applied to datasets built in the physics-internal (TOA-first) frame. Both
    axes are flipped together — flipping only ``level`` is exactly the #710
    bug. Axes absent from ``ds`` are skipped, so this is safe on a
    dynamics-only dataset with no interface fields.
    """
    flip = {
        dim: slice(None, None, -1)
        for dim in (LEVEL_DIM, LEVEL_INTERFACE_DIM)
        if dim in ds.dims
    }
    return ds.isel(**flip) if flip else ds


def orient_top_first(ds):
    """Reverse both vertical axes of ``ds`` so index 0 is the model top.

    The reader-side inverse of :func:`orient_surface_first`: it converts a
    file-convention (surface-first) Dataset back into the top-first
    physics-internal frame that :class:`~jcm.physics_interface.PhysicsState`
    and the dycore expect. The two functions are the *same* involution — both
    just reverse the vertical axes — but carrying two names documents at each
    call site which direction the conversion is going. Axes absent from ``ds``
    are skipped, so this is safe on a dataset with no interface fields.
    """
    flip = {
        dim: slice(None, None, -1)
        for dim in (LEVEL_DIM, LEVEL_INTERFACE_DIM)
        if dim in ds.dims
    }
    return ds.isel(**flip) if flip else ds


def attach_vertical_coordinates(ds, a_boundaries_pa, b_boundaries, p0: float):
    """Attach surface-first sigma coordinates and the hybrid ``(a, b)`` tables.

    ``a_boundaries_pa`` / ``b_boundaries`` are the TOA-first interface tables
    (as :func:`hybrid_boundaries` returns them); they are reversed here to the
    file's surface-first order. Nominal sigma is ``a/p0 + b``, evaluated at
    mid-levels for ``level`` and at interfaces for ``level_i`` — the *same*
    table for both axes, which is what guarantees they cannot disagree.

    ``level`` is overwritten rather than left as the dycore wrote it so that
    the two axes provably come from one source.
    """
    a_half = np.asarray(a_boundaries_pa, dtype=float)[::-1]
    b_half = np.asarray(b_boundaries, dtype=float)[::-1]
    # Mid-level coefficients are the interface mean, in the same (now
    # surface-first) order.
    a_full = 0.5 * (a_half[:-1] + a_half[1:])
    b_full = 0.5 * (b_half[:-1] + b_half[1:])

    if LEVEL_DIM in ds.dims and ds.sizes[LEVEL_DIM] == a_full.size:
        ds = ds.assign_coords({
            LEVEL_DIM: (LEVEL_DIM, a_full / p0 + b_full),
            HYBRID_A_FULL: (LEVEL_DIM, a_full),
            HYBRID_B_FULL: (LEVEL_DIM, b_full),
        })
    if LEVEL_INTERFACE_DIM in ds.dims and ds.sizes[LEVEL_INTERFACE_DIM] == a_half.size:
        ds = ds.assign_coords({
            LEVEL_INTERFACE_DIM: (LEVEL_INTERFACE_DIM, a_half / p0 + b_half),
            HYBRID_A_HALF: (LEVEL_INTERFACE_DIM, a_half),
            HYBRID_B_HALF: (LEVEL_INTERFACE_DIM, b_half),
        })
    return ds


def _time_attrs(time_coord) -> dict[str, str]:
    """Attributes for the ``time`` axis, honest about what it actually is.

    CF requires a variable with ``standard_name = "time"`` to carry
    reference-time ``units`` ("days since ..."). xarray supplies those from its
    encoding only for a **datetime64** coordinate; a bare numeric elapsed-days
    axis would get the standard name with no decodable units, which is worse
    than no claim at all — the file would announce CF conformance a reader
    cannot honour. So a numeric axis is labelled as elapsed time and is *not*
    claimed to be a CF time coordinate.

    Backends should emit datetime64 (``PysesCamSEDycore.to_xarray`` and
    ``ModelPredictions._trajectory_dataset`` both do); this is the guard for
    anything that does not.
    """
    if np.issubdtype(np.asarray(time_coord.values).dtype, np.datetime64):
        return dict(_COORD_ATTRS["time"])
    return {"axis": "T", "units": "d",
            "long_name": "elapsed simulation time"}


def apply_cf_attributes(ds):
    """Set CF ``standard_name``/``units``/``axis``/``positive`` where known.

    Existing attributes are overwritten for the names listed in this module —
    the per-physics units tables are attached first and are less specific
    (e.g. they carry no ``standard_name``). Variables not listed are untouched.
    """
    for name, attrs in _COORD_ATTRS.items():
        if name in ds.coords:
            ds[name].attrs.update(
                _time_attrs(ds[name]) if name == "time" else attrs)

    for name, attrs in _VARIABLE_ATTRS.items():
        if name in ds.variables:
            ds[name].attrs.update(attrs)

    for a_name, unit, long_name in (
        (HYBRID_A_FULL, "Pa", "hybrid A coefficient at layer mid-level"),
        (HYBRID_A_HALF, "Pa", "hybrid A coefficient at layer interface"),
        (HYBRID_B_FULL, "1", "hybrid B coefficient at layer mid-level"),
        (HYBRID_B_HALF, "1", "hybrid B coefficient at layer interface"),
    ):
        if a_name in ds.coords:
            ds[a_name].attrs.update({"units": unit, "long_name": long_name})

    # ``formula_terms`` may only name variables that are actually in the file;
    # ``surface_pressure`` is a physics diagnostic, so a dynamics-only dataset
    # gets the rest of the vertical metadata without a dangling reference.
    has_ps = "surface_pressure" in ds.variables
    for dim, a_name, b_name, where in (
        (LEVEL_DIM, HYBRID_A_FULL, HYBRID_B_FULL, "layer mid-level"),
        (LEVEL_INTERFACE_DIM, HYBRID_A_HALF, HYBRID_B_HALF, "layer interface"),
    ):
        if dim not in ds.coords:
            continue
        attrs: dict[str, Any] = dict(_SIGMA_COMMON_ATTRS)
        attrs["long_name"] = (
            f"nominal sigma (a/p0 + b) at {where}, surface-first "
            "(index 0 is the surface)"
        )
        # The parametric ``standard_name`` is only CF-legal when the axis also
        # carries ``formula_terms`` (§4.3.3); emit them together or neither.
        # An axis with only units + positive is still a valid CF vertical
        # coordinate, so a dynamics-only file stays conformant.
        if has_ps and a_name in ds.coords and b_name in ds.coords:
            attrs["formula_terms"] = (
                f"ap: {a_name} b: {b_name} ps: surface_pressure"
            )
            attrs["standard_name"] = _PARAMETRIC_SIGMA_STANDARD_NAME
        ds[dim].attrs.update(attrs)

    ds.attrs.setdefault("Conventions", "CF-1.11")
    return ds


def finalize_output(
    ds,
    *,
    vertical=None,
    a_boundaries_pa=None,
    b_boundaries=None,
    p0: float | None = None,
    flip_vertical: bool = True,
    extra_attrs: Mapping[str, str] | None = None,
):
    """Turn a freshly-built trajectory Dataset into the output convention.

    The single entry point every backend's ``to_xarray`` goes through: orient
    both vertical axes surface-first (``flip_vertical=False`` for a backend
    that already built them that way), attach the sigma / hybrid coordinates,
    and stamp CF attributes.

    Pass either ``vertical`` (a dinosaur vertical coordinate object) or an
    explicit ``a_boundaries_pa``/``b_boundaries`` pair, both TOA-first.
    """
    if p0 is None:
        from jcm import constants as c
        p0 = float(c.p0)
    if flip_vertical:
        ds = orient_surface_first(ds)
    if vertical is not None:
        a_boundaries_pa, b_boundaries = hybrid_boundaries(vertical)
    if a_boundaries_pa is not None:
        ds = attach_vertical_coordinates(ds, a_boundaries_pa, b_boundaries, p0)
    ds = apply_cf_attributes(ds)
    if extra_attrs:
        ds.attrs.update(extra_attrs)
    return ds
