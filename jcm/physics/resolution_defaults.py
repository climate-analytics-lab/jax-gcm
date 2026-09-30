"""Resolution-dependent defaults for tunable physics parameters.

A reference model often sets some tunable parameters per grid: ECHAM6.3, for
example, gives its cloud cover a different critical relative humidity at T63
than at T127 (``mo_echam_cloud_params.f90::sucloud``). Those values are
*defaults*. They stay ordinary differentiable leaves of the scheme's
``Parameters`` object, and a value the user sets always wins over them.

The mechanism, for any scheme:

* The scheme holds a table of its reference values per spectral truncation,
  ``{truncation: {field: value}}``.
* :func:`resolution_defaults` turns the table and the run's truncation into
  one ``{field: value}`` mapping. At a tabulated truncation it returns the
  table's values exactly. Between two tabulated truncations it interpolates
  linearly in the truncation number, except for the fields named in
  ``nearest`` (integers and switches in the reference), which take the value
  of the nearer tabulated truncation, the finer one on a tie. Outside the
  tabulated range it holds the end value and warns once, naming the grid. A
  grid with no spectral truncation takes the ``fallback`` row, with a
  warning.
* The scheme's ``Parameters.default(truncation=...)`` builds its defaults from
  that mapping; ``echam_physics(coords=...)`` and the Hydra runner pass the
  run's truncation at physics construction, so the parameter pytree a user
  holds after construction is final. :func:`default_parameters` builds any
  ``Parameters`` class this way, and falls back to ``default()`` for a class
  without resolution defaults.

Interpolated values between the reference's own truncations are jcm's
choice, not the reference's, and are untuned.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Collection, Mapping

import numpy as np

__all__ = [
    "check_defaults_grid",
    "default_parameters",
    "defaults_flag_kwargs",
    "has_resolution_defaults",
    "resolution_defaults",
    "spectral_truncation",
]

#: ``(table name, truncation)`` pairs already warned about, so that each
#: out-of-range grid is reported once per process rather than once per term.
_WARNED: set[tuple[str, int | None]] = set()


def spectral_truncation(coords) -> int | None:
    """Return the triangular truncation of ``coords``, or ``None``.

    ``utils.get_coords`` builds a truncation-``T`` grid with
    ``max_wavenumber = T``, so ``longitude_wavenumbers = T + 1``. A grid
    without spectral wavenumbers (the pySES cubed sphere) returns ``None``.
    """
    horizontal = getattr(coords, "horizontal", None)
    wavenumbers = getattr(horizontal, "longitude_wavenumbers", None)
    return None if wavenumbers is None else int(wavenumbers) - 1


def _warn_once(table_name: str, truncation: int | None, message: str) -> None:
    key = (table_name, truncation)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, UserWarning, stacklevel=3)


def resolution_defaults(
    table: Mapping[int, Mapping[str, Any]],
    truncation: int | None,
    *,
    nearest: Collection[str] = (),
    fallback: int,
    table_name: str,
) -> dict[str, Any]:
    """Return the defaults for ``truncation`` from a per-truncation table.

    Args:
        table: ``{truncation: {field: value}}``; every row has the same fields.
        truncation: the run's spectral truncation, or ``None`` for a grid that
            is not spectral.
        nearest: fields that are integers or switches in the reference. They
            are never interpolated: between two tabulated truncations they
            take the nearer one's value, and the finer one's on a tie.
        fallback: the tabulated truncation whose row a non-spectral grid gets.
        table_name: names the table in warnings.

    Returns:
        ``{field: value}`` for every field of the table.

    """
    keys = sorted(table)
    if truncation is None:
        _warn_once(table_name, None, (
            f"{table_name}: the grid has no spectral truncation; using the "
            f"T{fallback} defaults. Set the parameters explicitly to choose "
            "others."))
        return dict(table[fallback])
    if truncation in table:
        return dict(table[truncation])
    if truncation < keys[0] or truncation > keys[-1]:
        end = keys[0] if truncation < keys[0] else keys[-1]
        _warn_once(table_name, truncation, (
            f"{table_name}: T{truncation} is outside the tabulated range "
            f"T{keys[0]}-T{keys[-1]}; holding the T{end} defaults. Set the "
            "parameters explicitly to choose others."))
        return dict(table[end])
    upper = next(k for k in keys if k > truncation)
    lower = max(k for k in keys if k < truncation)
    weight = (truncation - lower) / (upper - lower)
    near = upper if (truncation - lower) >= (upper - truncation) else lower
    out = {}
    for field, low_value in table[lower].items():
        if field in nearest:
            out[field] = table[near][field]
        else:
            out[field] = low_value + weight * (table[upper][field] - low_value)
    return out


def has_resolution_defaults(params_cls) -> bool:
    """Whether ``params_cls.default`` accepts a ``truncation`` keyword."""
    default = getattr(params_cls, "default", None)
    if default is None:
        return False
    try:
        return "truncation" in inspect.signature(default).parameters
    except (TypeError, ValueError):
        return False


def default_parameters(params_cls, truncation: int | None = 63):
    """``params_cls.default(truncation=...)``, or ``default()`` without one.

    The one call the factory and the runner use to build a scheme's defaults
    for the run's grid, so that a scheme gains resolution defaults by adding a
    ``truncation`` keyword to its ``default`` and nothing else.
    """
    if has_resolution_defaults(params_cls):
        return params_cls.default(truncation=truncation)
    return params_cls.default()


def defaults_flag_kwargs(term_cls, are_defaults: bool) -> dict[str, bool]:
    """``{"params_are_defaults": are_defaults}`` if ``term_cls`` takes it.

    A term that holds resolution-dependent parameters accepts a
    ``params_are_defaults`` keyword, so that its ``cache_coords`` can tell the
    defaults the factory or runner built for a grid (which it checks against
    the model grid) from an object the user supplied (which it leaves alone).
    Terms without resolution defaults do not take it and get ``{}``.
    """
    try:
        accepted = inspect.signature(term_cls.__init__).parameters
    except (TypeError, ValueError):
        return {}
    if "params_are_defaults" in accepted:
        return {"params_are_defaults": bool(are_defaults)}
    return {}


def _describe(truncation: int | None) -> str:
    return "a non-spectral grid" if truncation is None else f"T{truncation}"


def check_defaults_grid(owner: str, built_for: int | None, coords) -> None:
    """Warn if resolution defaults were built for another grid than ``coords``.

    Called from a term's ``cache_coords`` for parameters the factory or the
    runner built as defaults (never for an object the user supplied). It only
    warns: the parameter pytree was fixed at construction and is not
    re-resolved. A single-column grid has no horizontal resolution, so any
    defaults suit it and nothing is checked.

    Args:
        owner: names the term in the warning.
        built_for: the truncation the defaults were built for (``None`` for a
            non-spectral grid).
        coords: the model grid.

    """
    nodal_shape = getattr(getattr(coords, "horizontal", None),
                          "nodal_shape", None)
    if nodal_shape is not None and int(np.prod(nodal_shape)) == 1:
        return
    grid = spectral_truncation(coords)
    if built_for == grid:
        return
    warnings.warn(
        f"{owner}: its parameter defaults were built for "
        f"{_describe(built_for)} but the model grid is {_describe(grid)}. "
        "Build the physics with the grid (echam_physics(coords=...); the "
        "Hydra runner does this) to get the grid's defaults, or pass the "
        "parameters explicitly.",
        UserWarning, stacklevel=3)
