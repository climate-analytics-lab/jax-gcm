"""
Date: 2/1/2024
For storing all variables related to the model's grid space.
"""
import jax.numpy as jnp
import tree_math
from jcm.constants import p0, grav, cp
from dinosaur.coordinate_systems import CoordinateSystem
from dinosaur.vertical_interpolation import HybridCoordinates
from typing import Optional
from jcm.vertical.icon_levels import HybridLevels, ICONLevels

sigma_layer_boundaries = {
    5: jnp.array([0.0, 0.15, 0.35, 0.65, 0.9, 1.0]),
    7: jnp.array([0.02, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    8: jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
}

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
    
    # Coordinate arrays for xarray conversion and physics
    latitudes: Optional[jnp.ndarray] = None
    longitudes: Optional[jnp.ndarray] = None
    
    # Hybrid coordinate support - store only the essential arrays
    hybrid_a_boundaries: Optional[jnp.ndarray] = None
    hybrid_b_boundaries: Optional[jnp.ndarray] = None
    
    @property
    def is_hybrid(self) -> bool:
        """Check if geometry uses hybrid coordinates."""
        return self.hybrid_a_boundaries is not None
    
    @property
    def nlevels(self) -> int:
        """Number of vertical levels."""
        return self.nodal_shape[0]
    
    @property
    def nlon(self) -> int:
        """Number of longitude points."""
        return self.nodal_shape[1]
    
    @property
    def nlat(self) -> int:
        """Number of latitude points."""
        return self.nodal_shape[2]
    
    def get_pressure_levels(self, surface_pressure: jnp.ndarray) -> jnp.ndarray:
        """Calculate pressure at level centers.
        
        Args:
            surface_pressure: Surface pressure field (Pa)
            
        Returns:
            Pressure at level centers (nlevels, *surface_pressure.shape)
        """
        if self.is_hybrid:
            # p = a + b * p_surface
            a_centers = (self.hybrid_a_boundaries[:-1] + self.hybrid_a_boundaries[1:]) / 2
            b_centers = (self.hybrid_b_boundaries[:-1] + self.hybrid_b_boundaries[1:]) / 2
            return a_centers[:, None] + b_centers[:, None] * surface_pressure[None, :]
        else:
            # Pure sigma coordinates: p = σ * p_surface
            return self.fsg[:, None] * surface_pressure[None, :] * p0
    
    def get_pressure_interfaces(self, surface_pressure: jnp.ndarray) -> jnp.ndarray:
        """Calculate pressure at level interfaces.
        
        Args:
            surface_pressure: Surface pressure field (Pa)
            
        Returns:
            Pressure at interfaces (nlevels+1, *surface_pressure.shape)
        """
        if self.is_hybrid:
            # p = a + b * p_surface
            return self.hybrid_a_boundaries[:, None] + self.hybrid_b_boundaries[:, None] * surface_pressure[None, :]
        else:
            # Pure sigma coordinates: p = σ * p_surface
            return self.hsg[:, None] * surface_pressure[None, :] * p0
    
    def get_layer_thickness(self, surface_pressure: jnp.ndarray) -> jnp.ndarray:
        """Calculate layer thickness in pressure coordinates.
        
        Args:
            surface_pressure: Surface pressure field (Pa)
            
        Returns:
            Layer thickness (dp) at each level (nlevels, *surface_pressure.shape)
        """
        pressure_interfaces = self.get_pressure_interfaces(surface_pressure)
        return jnp.diff(pressure_interfaces, axis=0)
    
    def get_coordinate_info(self) -> dict:
        """Get summary of coordinate system information."""
        info = {
            'nodal_shape': self.nodal_shape,
            'coordinate_type': 'hybrid' if self.is_hybrid else 'sigma',
            'nlevels': self.nlevels,
            'nlon': self.nlon,
            'nlat': self.nlat,
        }
        
        # Basic coordinate info without storing full coordinate system
        info['horizontal_grid'] = f"Grid({self.nlon}x{self.nlat})"
        info['vertical_grid'] = f"{'Hybrid' if self.is_hybrid else 'Sigma'}Coordinates({self.nlevels})"
        
        if self.is_hybrid:
            info['hybrid_levels'] = self.nlevels
            info['top_pressure'] = float(self.hybrid_a_boundaries[0])
            info['surface_sigma'] = float(self.hybrid_b_boundaries[-1])
        
        return info

    @classmethod
    def _get_horizontal_coords(cls, coords: CoordinateSystem):
        """Extract horizontal coordinate information from CoordinateSystem."""
        radang = coords.horizontal.latitudes
        sia, coa = jnp.sin(radang), jnp.cos(radang)
        return radang, sia, coa
    
    @classmethod
    def _get_horizontal_coords_from_shape(cls, nodal_shape):
        """Calculate horizontal coordinates from grid shape (legacy)."""
        il = nodal_shape[1]
        iy = (il + 1)//2
        j = jnp.arange(1, iy + 1)
        sia_half = jnp.cos(jnp.pi * (j - 0.25) / (il + 0.5))
        coa_half = jnp.sqrt(1.0 - sia_half ** 2.0)
        sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
        coa = jnp.concatenate((coa_half, coa_half[::-1]), axis=0).ravel()
        radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)
        return radang, sia, coa
    
    @classmethod
    def _get_hybrid_vertical_coords(cls, nlevels: int):
        """Get vertical coordinates for hybrid system."""
        hybrid_levels = ICONLevels.get_levels(nlevels)
        
        # Create approximate sigma levels for backward compatibility
        hsg = hybrid_levels.b_boundaries
        fsg = hybrid_levels.b_centers
        dhs = jnp.diff(hsg)
        
        # Create pseudo-sigma logarithm (avoiding log(0))
        fsg_safe = jnp.where(fsg > 1e-10, fsg, 1e-10)
        sigl = jnp.log(fsg_safe)
        
        # Create dummy conversion factors (overridden by hybrid calculations)
        grdsig = jnp.ones_like(dhs) * grav / p0
        grdscp = grdsig / cp
        wvi = jnp.zeros((nlevels, 2))
        
        return hsg, fsg, dhs, sigl, grdsig, grdscp, wvi, hybrid_levels

    @classmethod
    def from_coords(cls, coords: CoordinateSystem=None, hybrid: bool = False):
        """
        Initialize geometry from a dinosaur CoordinateSystem.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object
            hybrid: If True, interpret as hybrid coordinates; if False, use sigma

        Returns:
            Geometry object
        """
        # Horizontal coordinates
        radang, sia, coa = cls._get_horizontal_coords(coords)
        
        # Vertical coordinates
        kx = coords.nodal_shape[0]
        if hybrid:
            hsg, fsg, dhs, sigl, grdsig, grdscp, wvi, hybrid_levels = cls._get_hybrid_vertical_coords(kx)
            hybrid_a_boundaries = hybrid_levels.a_boundaries
            hybrid_b_boundaries = hybrid_levels.b_boundaries
        else:
            hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = _initialize_vertical(kx)
            hybrid_a_boundaries = None
            hybrid_b_boundaries = None

        return cls(
            nodal_shape=coords.nodal_shape,
            radang=radang, sia=sia, coa=coa,
            hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
            grdsig=grdsig, grdscp=grdscp, wvi=wvi,
            latitudes=coords.horizontal.latitudes,
            longitudes=coords.horizontal.longitudes,
            hybrid_a_boundaries=hybrid_a_boundaries,
            hybrid_b_boundaries=hybrid_b_boundaries
        )

    @classmethod
    def from_grid_shape(cls, nodal_shape=None, node_levels=None, hybrid: bool = False):
        """
        Initialize geometry from grid dimensions (legacy code from speedy.f90).

        Args:
            nodal_shape: Shape of the nodal grid `(ix,il)`
            node_levels: Number of vertical levels `kx`
            hybrid: If True, use hybrid coordinates; if False, use sigma

        Returns:
            Geometry object
        """
        # Horizontal coordinates
        radang, sia, coa = cls._get_horizontal_coords_from_shape(nodal_shape)

        # Vertical coordinates
        if hybrid:
            hsg, fsg, dhs, sigl, grdsig, grdscp, wvi, hybrid_levels = cls._get_hybrid_vertical_coords(node_levels)
            hybrid_a_boundaries = hybrid_levels.a_boundaries
            hybrid_b_boundaries = hybrid_levels.b_boundaries
        else:
            hsg, fsg, dhs, sigl, grdsig, grdscp, wvi = _initialize_vertical(node_levels)
            hybrid_a_boundaries = None
            hybrid_b_boundaries = None
        
        # For from_grid_shape, we need to calculate longitude array
        ix = nodal_shape[0]
        longitudes = jnp.linspace(0, 2 * jnp.pi * (ix - 1) / ix, ix)
        
        return cls(
            nodal_shape=(node_levels,) + nodal_shape,
            radang=radang, sia=sia, coa=coa,
            hsg=hsg, fsg=fsg, dhs=dhs, sigl=sigl,
            grdsig=grdsig, grdscp=grdscp, wvi=wvi,
            latitudes=radang,
            longitudes=longitudes,
            hybrid_a_boundaries=hybrid_a_boundaries,
            hybrid_b_boundaries=hybrid_b_boundaries
        )
    
    @classmethod
    def for_icon_physics(cls, horizontal_resolution: int = 85, nlevels: int = 47):
        """
        Create geometry optimized for ICON physics with hybrid coordinates.
        
        Args:
            horizontal_resolution: T-resolution (21, 31, 42, 85, 106, 119, 170, 213, 340, 425)
            nlevels: Number of vertical levels (must be available in ICON tables)
            
        Returns:
            Geometry object with ICON hybrid coordinates and appropriate horizontal grid
        """
        # Import here to avoid circular imports
        from jcm.model import get_coords_hybrid
        
        # Create coordinate system with hybrid coordinates
        coords = get_coords_hybrid(nlevels, horizontal_resolution)
        
        # Create geometry with hybrid coordinates
        return cls.from_hybrid_coords(coords, nlevels)