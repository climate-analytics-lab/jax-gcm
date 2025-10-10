"""
Date: 2/1/2024
For storing all variables related to the model's grid space.
"""
import jax.numpy as jnp
from matplotlib.pyplot import grid
import tree_math
from jcm.constants import p0, grav, cp
from jcm.utils import spectral_truncation
import dinosaur
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.primitive_equations import PrimitiveEquationsSpecs
from dinosaur.scales import SI_SCALE

sigma_layer_boundaries = {
    5: jnp.array([0.0, 0.15, 0.35, 0.65, 0.9, 1.0]),
    7: jnp.array([0.02, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    8: jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
}

truncation_for_nodal_shape = {
    (64, 32): 21,
    (96, 48): 31,
    (128, 64): 42,
    (256, 128): 85,
    (320, 160): 106,
    (360, 180): 119,
    (512, 256): 170,
    (640, 320): 213,
    (1024, 512): 340,
    (1280, 640): 425,
}

def get_coords(layers=8, horizontal_resolution=31) -> CoordinateSystem:
    """
    Returns a CoordinateSystem object for the given number of layers and horizontal resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, or 425).
    """
    try:
        horizontal_grid = getattr(dinosaur.spherical_harmonic.Grid, f'T{horizontal_resolution}')
    except AttributeError:
        raise ValueError(f"Invalid horizontal resolution: {horizontal_resolution}. Must be one of: 21, 31, 42, 85, 106, 119, 170, 213, 340, or 425.")
    if layers not in sigma_layer_boundaries:
        raise ValueError(f"Invalid number of layers: {layers}. Must be one of: {list(sigma_layer_boundaries.keys())}")

    physics_specs = PrimitiveEquationsSpecs.from_si(scale=SI_SCALE)

    return CoordinateSystem(
        horizontal=horizontal_grid(radius=physics_specs.radius),
        vertical=dinosaur.sigma_coordinates.SigmaCoordinates(sigma_layer_boundaries[layers])
    )

def _initialize_vertical(kx):
    # Definition of model levels
    # Layer thicknesses and full (u,v,T) levels
    if kx not in sigma_layer_boundaries:
        raise ValueError(f"Invalid number of vertical levels: {kx}")
    hsg = sigma_layer_boundaries[kx]
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
    def from_coords(cls, coords: CoordinateSystem=None, orography=None, truncation_number=None):
        """
        Initializes all of the speedy model geometry variables from a dinosaur CoordinateSystem.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object

        Returns:
            Geometry object
        """
        # Orography and surface geopotential
        orog = jnp.zeros(coords.horizontal.nodal_shape)
        phis0 = jnp.zeros(coords.horizontal.nodal_shape)

        # Horizontal functions of latitude (from south to north)
        radang = coords.horizontal.latitudes
        sia, coa = jnp.sin(radang), jnp.cos(radang)
        
        # Vertical functions of sigma
        kx = coords.nodal_shape[0]
        hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = _initialize_vertical(kx)

        # Orography and surface geopotential
        orography = jnp.zeros(coords.horizontal.nodal_shape) if orography is None else orography
        phi0 = grav * orography
        phis0 = spectral_truncation(coords.horizontal, phi0, truncation_number=truncation_number)

        return cls(nodal_shape=coords.nodal_shape,
                   orog=orog, phis0=phis0,
                   radang=radang, sia=sia, coa=coa,
                   hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
                   grdsig=grdsig, grdscp=grdscp, wvi=wvi)

    @classmethod
    def from_grid_shape(cls, nodal_shape=None, node_levels=None, orography=None, truncation_number=None):
        """
        Initializes all of the speedy model geometry variables from grid dimensions (legacy code from speedy.f90).

        Args:
            nodal_shape: Shape of the nodal grid `(ix,il)`
            node_levels: Number of vertical levels `kx`

        Returns:
            Geometry object
        """
        # Orography and surface geopotential
        orog = jnp.zeros(nodal_shape)
        phis0 = jnp.zeros(nodal_shape)

        # Horizontal functions of latitude (from south to north)
        il = nodal_shape[1]
        iy = (il + 1)//2
        j = jnp.arange(1, iy + 1)
        sia_half = jnp.cos(jnp.pi * (j - 0.25) / (il + 0.5))
        coa_half = jnp.sqrt(1.0 - sia_half ** 2.0)
        sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
        coa = jnp.concatenate((coa_half, coa_half[::-1]), axis=0).ravel()
        radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)

        # Vertical functions of sigma
        kx = node_levels
        hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = _initialize_vertical(kx)

        # Orography and surface geopotential
        grid = get_coords(horizontal_resolution=truncation_for_nodal_shape[nodal_shape])
        orography = jnp.zeros(nodal_shape) if orography is None else orography
        phi0 = grav * orography
        phis0 = spectral_truncation(grid, phi0, truncation_number=truncation_number)
        
        return cls(nodal_shape=(node_levels,) + nodal_shape,
                   orog=orog, phis0=phis0,
                   radang=radang, sia=sia, coa=coa,
                   hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
                   grdsig=grdsig, grdscp=grdscp, wvi=wvi)
