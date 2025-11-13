"""
Date: 2/1/2024
For storing all variables related to the model's grid space.
"""
import jax.numpy as jnp
import tree_math
from jcm.constants import p0, grav, cp
from jcm.utils import SIGMA_LAYER_BOUNDARIES, TRUNCATION_FOR_NODAL_SHAPE, VALID_NODAL_SHAPES, VALID_TRUNCATIONS, get_coords, spectral_truncation, validate_ds
from jcm.data.bc.interpolate import upsample_terrain_ds
from dinosaur.coordinate_systems import CoordinateSystem
from typing import Tuple

def get_terrain(orography: jnp.ndarray=None, fmask: jnp.ndarray=None, nodal_shape=None,
                terrain_file=None, target_resolution=None, fmask_threshold=0.1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get the orography data for the model grid. If fmask and/or orography are provided, use them directly
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
    if fmask is None and orography is None:
        if terrain_file is None:
            if nodal_shape is None:
                raise ValueError("Must provide at least one of: fmask, orography, terrain_file, or nodal_shape.")
            return jnp.zeros(nodal_shape), jnp.zeros(nodal_shape)
        
        import xarray as xr
        ds = xr.open_dataset(terrain_file)
        validate_ds(ds, expected_structure={"lsm": ("lon", "lat"), "orog": ("lon", "lat")})
        orography, fmask = jnp.asarray(ds['orog']), jnp.asarray(ds['lsm'])
        if target_resolution is not None:
            if target_resolution not in VALID_TRUNCATIONS:
                raise ValueError(f"Invalid target resolution: {target_resolution}. Must be one of: {VALID_TRUNCATIONS}.")
            ds = upsample_terrain_ds(ds, target_resolution=target_resolution)
        elif orography.shape not in VALID_NODAL_SHAPES:
            raise ValueError(f"Invalid terrain data shape: {orography.shape}. Must be one of: {VALID_NODAL_SHAPES}.")

    elif fmask is None:
        # If orography provided but fmask not, default fmask to any orography > 0
        fmask = (orography > 0.0).astype(jnp.float32)

    elif orography is None:
        # If fmask provided but orography not, default orography to zeros (flat)
        orography = jnp.zeros_like(fmask)

    # Set values close to 0 or 1 to exactly 0 or 1
    fmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

    return orography, fmask

def _initialize_vertical(kx):
    # Definition of model levels
    # Layer thicknesses and full (u,v,T) levels
    if kx not in SIGMA_LAYER_BOUNDARIES:
        raise ValueError(f"Invalid number of vertical levels: {kx}")
    hsg = SIGMA_LAYER_BOUNDARIES[kx]
    fsg = (hsg[1:] + hsg[:-1])/2.
    dhs = jnp.diff(hsg)
    sigl = jnp.log(fsg)

    # 1.2 Functions of sigma and latitude (from initialize_physics in speedy.F90)
    grdsig = grav/(dhs*p0)
    grdscp = grdsig/cp

    # Weights for vertical interpolation at half-levels(1,kx) and surface
    # Note that for phys.par. half-lev(k) is between full-lev k and k+1
    # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
    # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
    wvi = jnp.zeros((kx, 2))
    wvi = wvi.at[:-1, 0].set(1./jnp.diff(sigl))
    wvi = wvi.at[:-1, 1].set((jnp.log(hsg[1:-1])-sigl[:-1])*wvi[:-1, 0])
    wvi = wvi.at[-1, 1].set((jnp.log(0.99)-sigl[-1])*wvi[-2,0])

    return hsg, fsg, dhs, sigl, grdsig, grdscp, wvi

@tree_math.struct
class Geometry:
    nodal_shape: tuple[int, int, int] # (kx, ix, il)

    orog: jnp.ndarray # orography height (m), shape (ix, il)
    phis0: jnp.ndarray # spectrally truncated surface geopotential
    fmask: jnp.ndarray # fractional land-sea mask (ix, il)

    radang: jnp.ndarray # latitude in radians
    sia: jnp.ndarray # sin of latitude
    coa: jnp.ndarray # cos of latitude

    hsg: jnp.ndarray # sigma layer boundaries
    fsg: jnp.ndarray # sigma layer midpoints
    dhs: jnp.ndarray # sigma layer thicknesses
    sigl: jnp.ndarray # log of sigma layer midpoints

    grdsig: jnp.ndarray # g/(d_sigma p0): to convert fluxes of u,v,q into d(u,v,q)/dt
    grdscp: jnp.ndarray # g/(d_sigma p0 c_p): to convert energy fluxes into dT/dt
    wvi: jnp.ndarray # Weights for vertical interpolation

    @classmethod
    def from_coords(cls, coords: CoordinateSystem, orography=None, fmask=None, terrain_file=None, interpolate=False, truncation_number=None):
        """
        Initializes all of the speedy model geometry variables from a dinosaur CoordinateSystem.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object.
            orography (optional): Orography height (m), shape (ix, il). If None, defaults to zeros.
            fmask (optional): Fractional land-sea mask, shape (ix, il). If None, defaults to zeros (all ocean).
            terrain_file (optional): Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            interpolate (optional): Whether to interpolate the terrain data (default False).
            truncation_number (optional): Spectral truncation number for surface geopotential. If None, inferred from coords.

        Returns:
            Geometry object
        """
        # Orography and surface geopotential
        orog, fmask = get_terrain(fmask=fmask, orography=orography, nodal_shape=coords.horizontal.nodal_shape,
                                  terrain_file=terrain_file, target_resolution=coords.horizontal.total_wavenumbers-2 if interpolate else None)
        phi0 = grav * orog
        phis0 = spectral_truncation(coords.horizontal, phi0, truncation_number=truncation_number)

        # Horizontal functions of latitude (from south to north)
        radang = coords.horizontal.latitudes
        sia, coa = jnp.sin(radang), jnp.cos(radang)

        # Vertical functions of sigma
        kx = coords.nodal_shape[0]
        hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = _initialize_vertical(kx)

        return cls(nodal_shape=coords.nodal_shape,
                   orog=orog, phis0=phis0, fmask=fmask,
                   radang=radang, sia=sia, coa=coa,
                   hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
                   grdsig=grdsig, grdscp=grdscp, wvi=wvi)
    
    @classmethod
    def from_spectral_truncation(cls, spectral_truncation, num_levels=8, **kwargs):
        """
        Initializes all of the speedy model geometry variables from spectral truncation (legacy code from speedy.f90).

        Args:
            spectral_truncation: Spectral truncation number for horizontal resolution.
            num_levels (optional): Number of vertical levels `kx` (default 8).
            orography (optional): Orography height (m), shape (ix, il). If None, defaults to zeros.
            fmask (optional): Fractional land-sea mask, shape (ix, il). If None, defaults to zeros (all ocean).
            terrain_file (optional): Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            interpolate (optional): Whether to interpolate the terrain data (default False).
            truncation_number (optional): Spectral truncation number for surface geopotential. If None, inferred from spectral_truncation.

        Returns:
            Geometry object
        """
        return cls.from_coords(coords=get_coords(layers=num_levels, spectral_truncation=spectral_truncation), **kwargs)

    @classmethod
    def from_grid_shape(cls, nodal_shape, **kwargs):
        """
        Initializes all of the speedy model geometry variables from grid dimensions (legacy code from speedy.f90).

        Args:
            nodal_shape: Shape of the nodal grid `(ix,il)`.
            num_levels (optional): Number of vertical levels `kx` (default 8).
            orography (optional): Orography height (m), shape (ix, il). If None, defaults to zeros.
            fmask (optional): Fractional land-sea mask, shape (ix, il). If None, defaults to zeros (all ocean).
            terrain_file (optional): Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            interpolate (optional): Whether to interpolate the terrain data (default False).
            truncation_number (optional): Spectral truncation number for surface geopotential. If None, inferred from nodal_shape.

        Returns:
            Geometry object
        """
        if nodal_shape not in VALID_NODAL_SHAPES:
            raise ValueError(f"Invalid nodal shape: {nodal_shape}. Must be one of: {VALID_NODAL_SHAPES}.")
        return cls.from_spectral_truncation(TRUNCATION_FOR_NODAL_SHAPE[nodal_shape], **kwargs)
    
    @classmethod
    def from_terrain_file(cls, terrain_file, target_resolution=None, num_levels=8, truncation_number=None):
        """
        Initializes all of the speedy model geometry variables from a given terrain file containing orog and lsm.
        
        Args:
            terrain_file: Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            target_resolution (optional): Spectral truncation to interpolate the terrain data to, default None (no interpolation).
            num_levels (optional): Number of vertical levels `kx` (default 8).
            truncation_number (optional): Spectral truncation number for surface geopotential. If None, inferred from nodal_shape.
        
        Returns:
            Geometry object
        """
        orography, fmask = get_terrain(terrain_file=terrain_file, target_resolution=target_resolution)
        return cls.from_grid_shape(
            nodal_shape=orography.shape,
            num_levels=num_levels,
            orography=orography,
            fmask=fmask,
            truncation_number=truncation_number
        )

    @classmethod
    def single_column_geometry(cls, radang=0., orog=0., fmask=0., phis0=None, num_levels=8):
        """
        Initializes a Geometry instance for a single column model.

        Args:
            radang (optional): Latitude of the single column in radians (default 0).
            orog (optional): Orography height in meters (default 0).
            fmask (optional): Fractional land-sea mask (default 0, all ocean).
            phis0 (optional): Spectrally truncated surface geopotential (default grav * orog).
            num_levels (optional): Number of vertical levels (default 8).

        Returns:
            Geometry object
        """
        sia, coa = jnp.sin(radang), jnp.cos(radang)

        # Letting user specify phis0 allows for the case of pulling one column from a full geometry,
        # where phis0 =/= grav * orog due to spectral truncation.
        if phis0 is None:
            phis0 = grav * orog

        # Vertical functions of sigma
        hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = _initialize_vertical(num_levels)

        return cls(nodal_shape=(num_levels, 1, 1),
                   orog=jnp.array([[orog]]), phis0=jnp.array([[phis0]]), fmask=jnp.array([[fmask]]),
                   radang=jnp.array([[radang]]), sia=jnp.array([[sia]]), coa=jnp.array([[coa]]),
                   hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
                   grdsig=grdsig, grdscp=grdscp, wvi=wvi)

def coords_from_geometry(geometry: Geometry) -> CoordinateSystem:
    """
    Extracts a dinosaur CoordinateSystem from a Geometry object.

    Args:
        geometry: Geometry object.

    Returns:
        Compatible CoordinateSystem object.
    """
    return get_coords(
        layers=geometry.nodal_shape[0],
        spectral_truncation=TRUNCATION_FOR_NODAL_SHAPE[geometry.nodal_shape[1:]]
    )