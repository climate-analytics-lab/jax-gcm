import jax.numpy as jnp
import tree_math
from params import ix, il, kx

# Radiation and cloud constants
# Albedo values
albsea = 0.07  # Albedo over sea
albice = 0.60  # Albedo over sea ice (for ice fraction = 1)
albsn  = 0.60  # Albedo over snow (for snow cover = 1)

# Longwave parameters
epslw  = 0.05  # Fraction of blackbody spectrum absorbed/emitted by PBL only
emisfc = 0.98  # Longwave surface emissivity

# Placeholder for CO2-related variable
ablco2_ref = None  # To be initialized elsewhere

@tree_math.struct
class ModRadConData:
    # Time-invariant fields (arrays) - #FIXME: since this is time invariant, should it be intiailizd/held somewhere else?
    # fband = energy fraction emitted in each LW band = f(T)
    fband: jnp.ndarray 

    # Radiative properties of the surface (updated in fordate)
    # Albedo and snow cover arrays
    alb_l: jnp.ndarray  # Daily-mean albedo over land (bare-land + snow)
    alb_s: jnp.ndarray  # Daily-mean albedo over sea (open sea + sea ice)
    albsfc: jnp.ndarray # Combined surface albedo (land + sea)
    snowc: jnp.ndarray  # Effective snow cover (fraction)

    # Transmissivity and blackbody radiation (updated in radsw/radlw)
    tau2: jnp.ndarray    # Transmissivity of atmospheric layers
    st4a: jnp.ndarray     # Blackbody emission from full and half atmospheric levels
    stratc: jnp.ndarray      # Stratospheric correction term
    flux: jnp.ndarray         # Radiative flux in different spectral bands

    def __init__(self, nodal_shape, node_levels) -> None:
        self.fband = jnp.zeros((301,4))
        self.albl = jnp.zeros((nodal_shape))
        self.alb_s = jnp.zeros((nodal_shape))
        self.albsfc = jnp.zeros((nodal_shape))
        self.snowc = jnp.zeros((nodal_shape))
        self.tau2 = jnp.zeros((nodal_shape,node_levels,4))
        self.st4a = jnp.zeros((nodal_shape,node_levels,2))
        self.stratc = jnp.zeros((nodal_shape,2))
        self.flux = jnp.zeros((nodal_shape,4))


    def copy(self, fband=None, alb_l=None, alb_s=None, albsfc=None, snowc=None, tau2=None, st4a=None, stratc=None, flux=None):
        return ModRadConData(
            fband=fband if fband is not None else self.fband,
            alb_l=alb_l if alb_l is not None else self.alb_l,
            alb_s=alb_s if alb_s is not None else self.alb_s,
            albsfc=albsfc if albsfc is not None else self.albsfc,
            snowc=snowc if snowc is not None else self.snowc,
            tau2=tau2 if tau2 is not None else self.tau2,
            st4a=st4a if st4a is not None else self.st4a,
            stratc=stratc if stratc is not None else self.stratc,
            flux=flux if flux is not None else self.flux)
