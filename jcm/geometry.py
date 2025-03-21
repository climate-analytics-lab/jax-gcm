'''
Date: 2/1/2024
For storing all variables related to the model's grid space.
'''
import jax.numpy as jnp
import tree_math
import jcm.physical_constants as pc
from dinosaur.coordinate_systems import CoordinateSystem

sigma_layer_boundaries = {
    5: jnp.array([0.0, 0.15, 0.35, 0.65, 0.9, 1.0]),
    7: jnp.array([0.02, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    8: jnp.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
}

@tree_math.struct
class Geometry:
    hsg: jnp.ndarray
    dhs: jnp.ndarray
    fsg: jnp.ndarray
    sigl: jnp.ndarray
    radang: jnp.ndarray
    sia: jnp.ndarray
    coa: jnp.ndarray

    # Functions of sigma and latitude (initial. in INPHYS in F90)
    grdsig: jnp.ndarray # g/(d_sigma p0): to convert fluxes of u,v,q into d(u,v,q)/dt
    grdscp: jnp.ndarray # g/(d_sigma p0 c_p): to convert energy fluxes into dT/dt
    wvi: jnp.ndarray # Weights for vertical interpolation

    def copy(self, hsg=None, dhs=None, fsg=None, sigl=None, radang=None, sia=None, coa=None, grdsig=None, grdscp=None, wvi=None):
        return Geometry(
            hsg if hsg is not None else self.hsg,
            dhs if dhs is not None else self.dhs,
            fsg if fsg is not None else self.fsg,
            sigl if sigl is not None else self.sigl,
            radang if radang is not None else self.radang,
            sia if sia is not None else self.sia,
            coa if coa is not None else self.coa,
            grdsig if grdsig is not None else self.grdsig,
            grdscp if grdscp is not None else self.grdscp,
            wvi if wvi is not None else self.wvi
        )

    def _initialize_physics(self):
        # 1.2 Functions of sigma and latitude (from initialize_physics in speedy.F90)
        grdsig = pc.grav/(self.dhs*pc.p0)
        grdscp = grdsig/pc.cp

        # Weights for vertical interpolation at half-levels(1,kx) and surface
        # Note that for phys.par. half-lev(k) is between full-lev k and k+1
        # Fhalf(k) = Ffull(k)+WVI(K,2)*(Ffull(k+1)-Ffull(k))
        # Fsurf = Ffull(kx)+WVI(kx,2)*(Ffull(kx)-Ffull(kx-1))
        wvi = self.wvi
        wvi = wvi.at[:-1, 0].set(1./(self.sigl[1:]-self.sigl[:-1]))
        wvi = wvi.at[:-1, 1].set((jnp.log(self.hsg[1:-1])-self.sigl[:-1])*wvi[:-1, 0])
        wvi = wvi.at[-1, 1].set((jnp.log(0.99)-self.sigl[-1])*wvi[-2,0])

        return self.copy(grdsig=grdsig, grdscp=grdscp, wvi=wvi)
        
    
    # Initializes all of the model geometry variables.
    @classmethod 
    def from_coords(self, coords: CoordinateSystem=None):
        kx = len(coords.vertical.boundaries)-1
        
        # Definition of model levels
        # Layer thicknesses and full (u,v,T) levels
        # FIXME: hsg, fsg, dhs should be coords.vertical.boundaries, centers, layer_thickness respectively, but there is some issue with coords being jitted
        if kx not in sigma_layer_boundaries:
            raise ValueError(f"Invalid number of vertical levels: {kx}")
        hsg = sigma_layer_boundaries[kx]
        fsg = 0.5 * (hsg[1:] + hsg[:-1])
        dhs = hsg[1:] - hsg[:-1]

        sigl = jnp.log(fsg) # Moved here from physical_constants

        # Horizontal functions
        # Latitudes and functions of latitude
        radang = coords.horizontal.latitudes
        sia = jnp.sin(radang)
        coa = jnp.cos(radang)

        return Geometry(
            hsg=hsg,
            dhs=dhs,
            fsg=fsg,
            sigl=sigl,
            radang=radang,
            sia=sia,
            coa=coa,
            grdsig=jnp.zeros(kx),
            grdscp=jnp.zeros(kx),
            wvi=jnp.zeros((kx, 2))
        )._initialize_physics()
    
        # Initializes all of the model geometry variables.
    @classmethod 
    def initialize_geometry(self, nodal_shape=None, node_levels=None):
        kx, il = node_levels, nodal_shape[1]

        # Definition of model levels
        # Layer thicknesses and full (u,v,T) levels
        if kx not in sigma_layer_boundaries:
            raise ValueError(f"Invalid number of vertical levels: {kx}")
        hsg = sigma_layer_boundaries[kx]
        fsg = 0.5 * (hsg[1:] + hsg[:-1])
        dhs = hsg[1:] - hsg[:-1]

        sigl = jnp.log(fsg) # Moved here from physical_constants

        # Horizontal functions
        # Latitudes and functions of latitude
        # NB: J=1 is Southernmost point!
        iy = (il + 1)//2
        j = jnp.arange(1, iy + 1)
        sia_half = jnp.cos(jnp.pi * (j - 0.25) / (il + 0.5))
        coa_half = jnp.sqrt(1.0 - sia_half ** 2.0)
        sia = jnp.concatenate((-sia_half, sia_half[::-1]), axis=0).ravel()
        coa = jnp.concatenate((coa_half, coa_half[::-1]), axis=0).ravel()
        radang = jnp.concatenate((-jnp.arcsin(sia_half), jnp.arcsin(sia_half)[::-1]), axis=0)

        return Geometry(
            hsg=hsg,
            dhs=dhs,
            fsg=fsg,
            sigl=sigl,
            radang=radang,
            sia=sia,
            coa=coa,
            grdsig=jnp.zeros(kx),
            grdscp=jnp.zeros(kx),
            wvi=jnp.zeros((kx, 2))
        )._initialize_physics()