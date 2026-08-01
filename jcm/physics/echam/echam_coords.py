"""ECHAM-specific coordinate system data.

This module defines the EchamCoords struct that caches precomputed
coordinate data needed by ECHAM physics parameterizations, following
the same pattern as SpeedyCoords for SPEEDY physics.

The coordinates are stored in the hybrid (a, b) form so that the
hybrid-coordinate pressure relation

    p(k, col) = a(k) + b(k) * P_s(col)

can be evaluated exactly at every column. Pure sigma coordinates are a
special case with a = 0 and b = sigma.
"""

import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem


@tree_math.struct
class EchamCoords:
    """ECHAM-specific coordinate system data.

    All fields are constant during a simulation.

    Attributes:
        nodal_shape: Grid dimensions (nlev, nlon, nlat)
        a_full: Hybrid 'a' coefficient at level centers (Pa) [nlev]
        b_full: Hybrid 'b' coefficient at level centers [-] [nlev]
        a_half: Hybrid 'a' coefficient at half levels (Pa) [nlev+1]
        b_half: Hybrid 'b' coefficient at half levels [-] [nlev+1]
        lat: Latitude in radians
        lon: Longitude in radians

    """

    nodal_shape: tuple
    a_full: jnp.ndarray
    b_full: jnp.ndarray
    a_half: jnp.ndarray
    b_half: jnp.ndarray
    lat: jnp.ndarray
    lon: jnp.ndarray

    def xarray_additional_coords(self):
        """Return additional xarray coordinates for ECHAM physics fields."""
        return {}

    def calculate_pressure_full(self, surface_pressure_pa: jnp.ndarray) -> jnp.ndarray:
        """Compute pressure at full (center) levels.

        p(k, col) = a_full(k) + b_full(k) * P_s(col)

        Args:
            surface_pressure_pa: Surface pressure in Pascal [ncols] or [nlon, nlat]

        Returns:
            Pressure at each full level [nlev, ncols] or [nlev, nlon, nlat]

        """
        sp = surface_pressure_pa
        if sp.ndim == 1:
            return self.a_full[:, None] + self.b_full[:, None] * sp[None, :]
        return (
            self.a_full[:, None, None]
            + self.b_full[:, None, None] * sp[None, :, :]
        )

    def calculate_pressure_half(self, surface_pressure_pa: jnp.ndarray) -> jnp.ndarray:
        """Compute pressure at half (interface) levels.

        Args:
            surface_pressure_pa: Surface pressure in Pascal [ncols] or [nlon, nlat]

        Returns:
            Pressure at each half level [nlev+1, ncols] or [nlev+1, nlon, nlat]

        """
        sp = surface_pressure_pa
        if sp.ndim == 1:
            return self.a_half[:, None] + self.b_half[:, None] * sp[None, :]
        return (
            self.a_half[:, None, None]
            + self.b_half[:, None, None] * sp[None, :, :]
        )

    @classmethod
    def from_coordinate_system(cls, coords: CoordinateSystem):
        """Create EchamCoords from a dinosaur CoordinateSystem.

        Handles both SigmaCoordinates (stored as a=0, b=sigma) and
        HybridCoordinates (stored with native a, b coefficients).

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object

        Returns:
            EchamCoords struct containing coordinate data

        """
        from dinosaur.hybrid_coordinates import HybridCoordinates

        vertical = coords.vertical
        if isinstance(vertical, HybridCoordinates):
            # ICON HybridCoordinates store `a_boundaries` in Pa (ICON native
            # convention); use directly.
            a_half = jnp.asarray(vertical.a_boundaries)
            b_half = jnp.asarray(vertical.b_boundaries)
        else:
            # SigmaCoordinates: a = 0, b = sigma
            sigma_boundaries = jnp.asarray(vertical.boundaries)
            a_half = jnp.zeros_like(sigma_boundaries)
            b_half = sigma_boundaries

        # Full (center) levels: linear average of adjacent half-level a, b
        a_full = 0.5 * (a_half[:-1] + a_half[1:])
        b_full = 0.5 * (b_half[:-1] + b_half[1:])

        return cls(
            nodal_shape=coords.nodal_shape,
            a_full=a_full,
            b_full=b_full,
            a_half=a_half,
            b_half=b_half,
            lat=jnp.asarray(coords.horizontal.latitudes),
            lon=jnp.asarray(coords.horizontal.longitudes),
        )
