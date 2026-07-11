"""Coordinate adapters for the pyses CAM-SE backend.

jcm physics packages read a small attribute surface off ``coords``:
``coords.nodal_shape`` (``(nlev, *horizontal_shape)``),
``coords.vertical`` (a dinosaur vertical-coordinate object — here
:class:`~dinosaur.hybrid_coordinates.HybridCoordinates`, which
:class:`jcm.physics.echam.echam_coords.EchamCoords` unpacks into the hybrid
``(a, b)`` tables), and ``coords.horizontal.{nodal_shape, latitudes,
longitudes}``. This module provides minimal frozen-dataclass adapters that
expose exactly that surface over the pg2 physics-column layout.

Horizontal layout — ``(1, ncol)`` and its one compromise
--------------------------------------------------------
The protocol permits any horizontal shape, but
``ComposablePhysics(vectorize_columns=True)`` (the ECHAM stack) requires
exactly **two** horizontal dims, so the scattered pg2 columns are exposed as
a degenerate 2-D grid ``(1, ncol)``. 3-D physics fields are then
``(nlev, 1, ncol)``, flattening losslessly to ``(nlev, ncol)`` columns.

The shipped physics terms cache per-column lat/lon with the *separable-grid*
recipe ``jnp.meshgrid(latitudes, longitudes).reshape(-1)`` (see e.g.
``grey_two_stream.radiation_scheme.cache_coords``). On a scattered column set
no separable ``(latitudes, longitudes)`` pair reproduces both coordinates, so
this adapter follows the developer prototype's choice:

* ``latitudes`` carries the **full per-column** latitudes ``(ncol,)`` —
  ``meshgrid(lat, lon)`` with a length-1 ``longitudes`` yields ``(1, ncol)``
  arrays whose flattened latitude is exact per column;
* ``longitudes`` is a **single reference longitude** ``(1,)`` (the first
  column's) — every column therefore sees the *same* longitude in
  meshgrid-cached schemes (solar zenith / MACv2-SP plume placement). The
  diurnal cycle is correct in time but loses its longitudinal phase.

This is an accepted, documented limitation of coupling the shipped
meshgrid-based ``cache_coords`` to an unstructured column set; fixing it
properly needs a small physics-side change (accept pre-flattened per-column
lat/lon when ``nodal_shape[0] == 1``) and is flagged as an open issue for the
ne30 production configuration. Code that needs the true per-column
coordinates (terrain/forcing interpolation, output regridding, tests) must
use :attr:`SEHorizontalGrid.column_latitudes` /
:attr:`SEHorizontalGrid.column_longitudes` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from dinosaur.hybrid_coordinates import HybridCoordinates

from jcm.physics.echam.echam_levels import get_echam_levels


# Reference surface pressure (Pa) shared by the hybrid table normalisation
# and pyses's ``reference_surface_mass``.
P0_PA = 1.0e5

# Fraction of the first finite ``a`` interface used to replace the singular
# a[0] = 0 model top of the raw ECHAM/ICON L47 table (~1 Pa top).
DEFAULT_MODEL_TOP_FRACTION = 0.5


def full_echam_hybrid(nlev: int = 47,
                      top_fraction: float = DEFAULT_MODEL_TOP_FRACTION):
    """Full ICON/ECHAM hybrid table with a finite (non-singular) model top.

    The raw L47 table's top interface is ``a = b = 0`` (p_top = 0): singular
    for the analytic initial state's pressure→height inversion and stiff for
    the explicit CAM-SE core. We keep **all** ``nlev`` layers and replace
    only that top interface with ``top_fraction * a[1]`` (~1 Pa for L47),
    which leaves the ``a`` boundaries strictly increasing at the top
    (``a[0] < a[1]``) and the top pure-pressure (``b[0] = 0`` unchanged).

    A ~1 Pa top layer carries ~1 Pa of mass; on an explicit core it needs the
    native ``nu_top`` Laplacian sponge (see ``PysesCamSEDycore``'s
    ``n_sponge`` / ``nu_top`` arguments) to keep vertically propagating waves
    from reflecting and accumulating at the lid — the CAM-SE analogue of
    ECHAM's ``lmidatm`` ``mo_upper_sponge``.

    Args:
        nlev: Number of model layers (must exist in the ECHAM/ICON tables;
            47 is the production grid).
        top_fraction: ``a[0] -> top_fraction * a[1]`` replacement factor,
            in ``(0, 1)``.

    Returns:
        ``(a_boundaries_pa, b_boundaries)`` as float64 numpy arrays of length
        ``nlev + 1``, ordered **top-first** (index 0 = model top, index -1 =
        surface where ``b = 1``) — the shared ordering of the ECHAM tables,
        the pyses vertical grid, and jcm's physics-internal frame.

    """
    if not 0.0 < top_fraction < 1.0:
        raise ValueError(f"top_fraction must be in (0, 1); got {top_fraction}")
    hc = get_echam_levels(nlev)
    a = np.asarray(hc.a_boundaries, dtype=np.float64).copy()
    b = np.asarray(hc.b_boundaries, dtype=np.float64).copy()
    a[0] = top_fraction * a[1]
    return a, b


@dataclass(frozen=True)
class SEHorizontalGrid:
    """Horizontal-grid adapter over the pg2 physics columns.

    Attributes:
        nodal_shape: ``(1, ncol)`` — the degenerate 2-D layout physics sees.
        latitudes: Per-column latitudes (radians), shape ``(ncol,)``.
            Broadcasts against the trailing ``(1, ncol)`` axes of 3-D fields
            and meshgrids to exact per-column values.
        longitudes: Single reference longitude (radians), shape ``(1,)`` —
            see the module docstring for why per-column longitudes cannot be
            represented in the separable convention.
        column_latitudes: Full per-column latitudes (radians), ``(ncol,)``.
        column_longitudes: Full per-column longitudes (radians), ``(ncol,)``
            in ``[0, 2π)``. Use these (not ``longitudes``) for any per-column
            geometry outside the shipped physics cache path.

    """

    nodal_shape: tuple
    latitudes: jnp.ndarray
    longitudes: jnp.ndarray
    column_latitudes: jnp.ndarray = field(repr=False, default=None)
    column_longitudes: jnp.ndarray = field(repr=False, default=None)

    @classmethod
    def from_columns(cls, lat_rad, lon_rad, dtype=jnp.float32):
        """Build the adapter from per-column latitude/longitude arrays (radians)."""
        lat = jnp.asarray(np.asarray(lat_rad).reshape(-1), dtype=dtype)
        lon = jnp.asarray(np.asarray(lon_rad).reshape(-1), dtype=dtype)
        return cls(
            nodal_shape=(1, int(lat.shape[0])),
            latitudes=lat,
            longitudes=lon[:1],
            column_latitudes=lat,
            column_longitudes=lon,
        )

    def to_modal(self, *args, **kwargs):
        raise NotImplementedError(
            "The spectral-element column grid has no spherical-harmonic "
            "basis; modal transforms are a dinosaur-backend concept."
        )

    def to_nodal(self, *args, **kwargs):
        raise NotImplementedError(
            "The spectral-element column grid has no spherical-harmonic "
            "basis; modal transforms are a dinosaur-backend concept."
        )


@dataclass(frozen=True)
class EchamSECoords:
    """Coordinate-system adapter consumed by jcm physics packages.

    ``vertical`` is a dinosaur :class:`HybridCoordinates` (``a_boundaries``
    in Pa, top-first) so :meth:`EchamCoords.from_coordinate_system` picks up
    the exact hybrid tables the dynamics integrates on; physics-side pressure
    reconstruction ``p = a + b * Ps`` is therefore consistent with the
    dycore's ``d_mass`` column to float32 round-off.
    """

    horizontal: SEHorizontalGrid
    vertical: HybridCoordinates

    @property
    def nlev(self) -> int:
        return int(np.asarray(self.vertical.b_boundaries).shape[0]) - 1

    @property
    def nodal_shape(self):
        """``(nlev, 1, ncol)`` — level count plus the horizontal layout."""
        return (self.nlev,) + tuple(self.horizontal.nodal_shape)


def make_echam_se_coords(lat_rad, lon_rad, a_boundaries_pa, b_boundaries,
                         dtype=jnp.float32) -> EchamSECoords:
    """Assemble the physics coords adapter from columns + hybrid tables.

    The hybrid tables are stored float32 to match the physics working
    precision (the dynamics keeps its own float64 copy inside the pyses
    ``v_grid``).
    """
    hybrid = HybridCoordinates(
        a_boundaries=jnp.asarray(np.asarray(a_boundaries_pa), dtype=dtype),
        b_boundaries=jnp.asarray(np.asarray(b_boundaries), dtype=dtype),
    )
    return EchamSECoords(
        horizontal=SEHorizontalGrid.from_columns(lat_rad, lon_rad, dtype=dtype),
        vertical=hybrid,
    )
