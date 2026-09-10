"""A first-class coordinate system for single-column configurations.

The single-column model previously built its coordinates as a
``SimpleNamespace`` duck-typing the parts of dinosaur's
``CoordinateSystem`` that physics packages happen to read. That worked,
but it had the classic stub problems: the contract lived in a comment
rather than a type, an attribute the stub lacked surfaced as a bare
``AttributeError`` deep inside a physics term, and there was nothing to
test against.

``ColumnCoordinates`` is the explicit version. It carries a real
vertical coordinate object (``SigmaCoordinates`` or
``HybridCoordinates``) over an ``(nlon, nlat) = (1, 1)`` horizontal
"grid" pinned at one geographic location, and implements exactly the
surface the physics packages consume:

- ``coords.vertical`` — the vertical coordinate object, unchanged;
- ``coords.nodal_shape`` — ``(nlev, 1, 1)``, matching the grid model's
  ``(nlev, nlon, nlat)`` convention that e.g. ``EchamTermBase.cache_coords``
  assumes;
- ``coords.horizontal.nodal_shape`` — ``(1, 1)``;
- ``coords.horizontal.latitudes`` / ``longitudes`` — 1-element arrays in
  **radians** (the convention of dinosaur's grids);
- ``coords.horizontal.nodal_axes`` — ``(longitudes, sin(latitudes))``,
  the pair ``utils`` and ``predictions`` unpack.

Spectral attributes (``longitude_wavenumbers`` etc.) are deliberately
absent — a single column has no spectral truncation — but they fail
with an explanatory error instead of a bare ``AttributeError``, so a
spectral-only physics package (e.g. speedy's horizontal diffusion)
reports *why* it cannot run in column mode.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

__all__ = ["ColumnCoordinates", "ColumnHorizontalGrid"]


_SPECTRAL_ATTRS = frozenset({
    "longitude_wavenumbers",
    "total_wavenumbers",
    "modal_shape",
    "modal_axes",
    "mask",
})


@dataclasses.dataclass(frozen=True)
class ColumnHorizontalGrid:
    """The ``(1, 1)`` horizontal "grid" of a single column.

    Attributes:
        latitudes: 1-element array, radians.
        longitudes: 1-element array, radians.

    """

    latitudes: jnp.ndarray
    longitudes: jnp.ndarray

    @classmethod
    def at_degrees(cls, lat_deg: float, lon_deg: float) -> "ColumnHorizontalGrid":
        return cls(
            latitudes=jnp.asarray([float(np.deg2rad(lat_deg))]),
            longitudes=jnp.asarray([float(np.deg2rad(lon_deg))]),
        )

    @property
    def nodal_shape(self) -> tuple[int, int]:
        return (1, 1)

    @property
    def nodal_axes(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(longitudes, sin(latitudes))`` — dinosaur's nodal-axes convention."""
        return (self.longitudes, jnp.sin(self.latitudes))

    def __getattr__(self, name: str):
        """Turn spectral-attribute access into an explanatory error."""
        if name in _SPECTRAL_ATTRS:
            raise AttributeError(
                f"ColumnHorizontalGrid has no spectral attribute {name!r}: a "
                "single column has no spectral truncation. The physics "
                "package requesting it cannot run in single-column mode."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclasses.dataclass(frozen=True)
class ColumnCoordinates:
    """Coordinate system for one atmospheric column at a fixed location.

    A drop-in for the places the SCM hands coordinates to physics: it
    implements the consumed subset of dinosaur's ``CoordinateSystem``
    with the same shapes and conventions as the full grid model, so a
    ``PhysicsTerm`` cannot tell the difference (and a term that needs
    what a column cannot provide gets an explanatory error).

    Attributes:
        vertical: A ``SigmaCoordinates``/``HybridCoordinates`` instance.
        horizontal: The single-point horizontal grid.

    """

    vertical: object
    horizontal: ColumnHorizontalGrid

    @classmethod
    def at_location(
        cls, vertical, lat_deg: float, lon_deg: float
    ) -> "ColumnCoordinates":
        """Build column coordinates at ``(lat_deg, lon_deg)`` in degrees."""
        return cls(
            vertical=vertical,
            horizontal=ColumnHorizontalGrid.at_degrees(lat_deg, lon_deg),
        )

    @property
    def nlev(self) -> int:
        return _vertical_nlev(self.vertical)

    @property
    def nodal_shape(self) -> tuple[int, int, int]:
        """``(nlev, nlon, nlat)`` — the grid model's convention."""
        return (self.nlev, 1, 1)


def _vertical_nlev(vertical) -> int:
    """Layer count of a Sigma/Hybrid vertical coordinate object."""
    if hasattr(vertical, "centers"):
        return int(np.asarray(vertical.centers).shape[0])
    if hasattr(vertical, "a_boundaries"):
        return int(np.asarray(vertical.a_boundaries).shape[0]) - 1
    raise TypeError(
        f"Unsupported vertical coordinate type {type(vertical).__name__!r}; "
        "expected SigmaCoordinates or HybridCoordinates."
    )
