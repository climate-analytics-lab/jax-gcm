"""TerrainData struct for boundary conditions that vary per simulation.

Date: 2026-01-26
"""
import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
from jcm.constants import grav
from jcm.utils import VALID_NODAL_SHAPES, VALID_TRUNCATIONS, validate_ds, spectral_truncation


def get_terrain(orography: jnp.ndarray = None, fmask: jnp.ndarray = None, nodal_shape=None,
                terrain_file=None, fmask_threshold=0.1, grid: HorizontalGridTypes = None):
    """Get the orography data for the model grid. If fmask and/or orography are provided, use them directly
    (defaulting the other to zeros if only one is provided). If terrain_file is provided, load both from file.
    Otherwise, default both to zeros with shape nodal_shape.

    Args:
        orography: Orography height (m) (ix, il). If None but fmask is provided, defaults to zeros (flat).
        fmask: Fractional land-sea mask (ix, il). If None but orography is provided, defaults to zeros (all ocean).
        nodal_shape: Shape of the nodal grid (ix, il). Used when neither fmask, orography, nor terrain_file are provided.
        terrain_file: Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
        target_resolution: Spectral truncation to interpolate the terrain data to, default None (no interpolation).
        fmask_threshold: Threshold for rounding fmask values that are close to 0 or 1.

    Returns:
        Orography height (m) (ix, il)
        Land-sea mask (ix, il)

    """
    # the spectral resolution is total wavenumbers - 2
    target_resolution = grid.total_wavenumbers - 2 if grid is not None else None

    if fmask is None and orography is None:
        if terrain_file is None:
            # if only nodal shape is provided, return zeros of that shape
            if nodal_shape is None:
                raise ValueError("Must provide at least one of: fmask, orography, terrain_file, or nodal_shape.")
            return jnp.zeros(nodal_shape), jnp.zeros(nodal_shape)

        # if only terrain file is provided, set orography and fmask from terrain file
        import xarray as xr
        from jcm.data.bc.interpolate import upsample_terrain_ds
        ds = xr.open_dataset(terrain_file)
        validate_ds(ds, expected_structure={"lsm": ("lon", "lat"), "orog": ("lon", "lat")})
        if target_resolution is not None:
            if target_resolution not in VALID_TRUNCATIONS:
                raise ValueError(f"Invalid target resolution: {target_resolution}. Must be one of: {VALID_TRUNCATIONS}.")
            ds = upsample_terrain_ds(ds, grid=grid)
        elif orography.shape not in VALID_NODAL_SHAPES:
            raise ValueError(f"Invalid terrain data shape: {orography.shape}. Must be one of: {VALID_NODAL_SHAPES}.")
        
        # set orography and fmask after upsampling happens
        orography, fmask = jnp.asarray(ds['orog']), jnp.asarray(ds['lsm'])
        
    elif fmask is None:
        # If orography provided but fmask not, default fmask to any orography > 0
        fmask = (orography > 0.0).astype(float)

    elif orography is None:
        # If fmask provided but orography not, default orography to zeros (flat)
        orography = jnp.zeros_like(fmask)

    # Set values close to 0 or 1 to exactly 0 or 1
    fmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

    return orography, fmask


@tree_math.struct
class TerrainData:
    """Boundary conditions that vary per simulation.

    Attributes:
        orog: Orography height (m), shape (ix, il)
        phis0: Spectrally truncated surface geopotential, shape (ix, il)
        fmask: Fractional land-sea mask, shape (ix, il)
        lfluxland: Whether to compute land surface fluxes (bool)

    """

    orog: jnp.ndarray
    phis0: jnp.ndarray
    fmask: jnp.ndarray
    lfluxland: jnp.bool_

    def copy(self, orog=None, fmask=None, phis0=None, lfluxland=None):
        """Copy an instance of TerrainData"""
        return TerrainData(
            orog=orog if orog is not None else self.orog,
            phis0=phis0 if phis0 is not None else self.phis0,
            fmask=fmask if fmask is not None else self.fmask,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland
        )

    @classmethod
    def from_coords(cls, coords: CoordinateSystem, orography=None, fmask=None, lfluxland=None,
                    terrain_file=None, interpolate=False):
        """Initialize TerrainData from a dinosaur CoordinateSystem.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object.
            orography (optional): Orography height (m), shape (ix, il). If None, defaults to zeros.
            fmask (optional): Fractional land-sea mask, shape (ix, il). If None, defaults to zeros (all ocean).
            lfluxland (optional): Whether to compute land surface fluxes (defaults to False if not provided).
            terrain_file (optional): Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            interpolate (optional): Whether to interpolate the terrain data (default False).

        Returns:
            TerrainData object

        """
        # Orography and surface geopotential
        orog, fmask = get_terrain(
            fmask=fmask,
            orography=orography,
            nodal_shape=coords.horizontal.nodal_shape,
            terrain_file=terrain_file,
            grid=coords.horizontal if interpolate else None
        )

        # if the user did not specify lfluxland, and fmask is > 0 anywhere (i.e. there is some land), 
        # set lfluxland to True, otherwise set to False if not specified
        if jnp.sum(fmask) > 0 and lfluxland is None:
            lfluxland = True
        elif lfluxland is None:
            lfluxland = False

        phi0 = grav * orog
        phis0 = spectral_truncation(coords.horizontal, phi0)

        return cls(orog=orog, phis0=phis0, fmask=fmask, lfluxland=jnp.bool_(lfluxland))

    @classmethod
    def from_file(cls, terrain_file, coords: CoordinateSystem, lfluxland=True):
        """Initialize TerrainData from a given terrain file containing orog and lsm.

        Args:
            terrain_file: Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            coords: dinosaur.coordinate_systems.CoordinateSystem object.
            target_resolution (optional): Spectral truncation to interpolate the terrain data to, default None (no interpolation).
            lfluxland (optional): Whether to compute land surface fluxes (default True).

        Returns:
            TerrainData object

        """
        orography, fmask = get_terrain(terrain_file=terrain_file, grid=coords.horizontal)

        # Validate that terrain matches coords
        if orography.shape != coords.horizontal.nodal_shape:
            raise ValueError(
                f"Terrain shape {orography.shape} does not match coords horizontal shape {coords.horizontal.nodal_shape}"
            )

        phi0 = grav * orography
        phis0 = spectral_truncation(coords.horizontal, phi0)

        return cls(orog=orography, phis0=phis0, fmask=fmask, lfluxland=jnp.bool_(lfluxland))

    @classmethod
    def aquaplanet(cls, coords: CoordinateSystem):
        """Initialize an aquaplanet TerrainData (flat, all ocean, no land fluxes).

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object.

        Returns:
            TerrainData object with all zeros for orography and fmask.

        """
        nodal_shape = coords.horizontal.nodal_shape
        return cls(
            orog=jnp.zeros(nodal_shape),
            phis0=jnp.zeros(nodal_shape),
            fmask=jnp.zeros(nodal_shape),
            lfluxland=jnp.bool_(False)
        )

    @classmethod
    def single_column(cls, orog=0., fmask=0., phis0=None, lfluxland=False):
        """Initialize a TerrainData instance for a single column model.

        Args:
            orog (optional): Orography height in meters (default 0).
            fmask (optional): Fractional land-sea mask (default 0, all ocean).
            phis0 (optional): Spectrally truncated surface geopotential (default grav * orog).
            lfluxland (optional): Whether to compute land surface fluxes (default False).

        Returns:
            TerrainData object

        """
        # Letting user specify phis0 allows for the case of pulling one column from full terrain,
        # where phis0 != grav * orog due to spectral truncation.
        if phis0 is None:
            phis0 = grav * orog

        return cls(
            orog=jnp.array([[orog]]),
            phis0=jnp.array([[phis0]]),
            fmask=jnp.array([[fmask]]),
            lfluxland=jnp.bool_(lfluxland)
        )
