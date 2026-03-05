"""ICON-specific coordinate system data.

This module defines the IconCoords struct that caches precomputed
coordinate data needed by ICON physics parameterizations, following
the same pattern as SpeedyCoords for SPEEDY physics.
"""

import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem


@tree_math.struct
class IconCoords:
    """ICON-specific coordinate system data.

    This struct caches precomputed coordinate data needed by ICON physics.
    All fields are constant during a simulation.

    Attributes:
        nodal_shape: Grid dimensions (nlev, nlon, nlat)
        fsg: Sigma coordinates at level centers (nlev,)
        hsg: Sigma coordinates at half levels / boundaries (nlev+1,)
        lat: Latitude in radians
        lon: Longitude in radians

    """

    nodal_shape: tuple
    fsg: jnp.ndarray
    hsg: jnp.ndarray
    lat: jnp.ndarray
    lon: jnp.ndarray

    @classmethod
    def from_coordinate_system(cls, coords: CoordinateSystem):
        """Create IconCoords from a dinosaur CoordinateSystem.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object

        Returns:
            IconCoords struct containing coordinate data

        """
        return cls(
            nodal_shape=coords.nodal_shape,
            fsg=jnp.asarray(coords.vertical.centers),
            hsg=jnp.asarray(coords.vertical.boundaries),
            lat=jnp.asarray(coords.horizontal.latitudes),
            lon=jnp.asarray(coords.horizontal.longitudes),
        )
