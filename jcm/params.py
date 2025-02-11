'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''
import tree_math
import jax.numpy as jnp
from jax import tree_util 

@tree_math.struct
class ConvectionParameters:
    psmin = jnp.array(0.8) # Minimum (normalised) surface pressure for the occurrence of convection
    trcnv = jnp.array(6.0) # Time of relaxation (in hours) towards reference state
    rhil = jnp.array(0.7) # Relative humidity threshold in intermeduate layers for secondary mass flux
    rhbl = jnp.array(0.9) # Relative humidity threshold in the boundary layer
    entmax = jnp.array(0.5) # Maximum entrainment as a fraction of cloud-base mass flux
    smf = jnp.array(0.8) # Ratio between secondary and primary mass flux at cloud-base

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class CondensationParameters:
    trlsc = 4.0   # Relaxation time (in hours) for specific humidity
    rhlsc = 0.9   # Maximum relative humidity threshold (at sigma=1)
    drhlsc = 0.1  # Vertical range of relative humidity threshold
    rhblsc = 0.95 # Relative humidity threshold for boundary layer
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ShortwaveRadiationParameters:
    # parameters for `get_zonal_average_fields`

    albcl   = 0.43  # Cloud albedo (for cloud cover = 1)
    albcls  = 0.50  # Stratiform cloud albedo (for st. cloud cover = 1)
    
    # Shortwave absorptivities (for dp = 10^5 Pa)
    absdry =  0.033 # Absorptivity of dry air (visible band)
    absaer =  0.033 # Absorptivity of aerosols (visible band)
    abswv1 =  0.022 # Absorptivity of water vapour
    abswv2 = 15.000 # Absorptivity of water vapour
    abscl1 =  0.015 # Absorptivity of clouds (visible band, maximum value)
    abscl2 =  0.15  # Absorptivity of clouds

    # Longwave absorptivities (for dp = 10^5 Pa)
    ablwin = 0.3   # Absorptivity of air in "window" band
    ablwv1 = 0.7   # Absorptivity of water vapour in H2O band 1 (weak) (for dq = 1 g/kg)
    ablwv2 = 50.0  # Absorptivity of water vapour in H2O band 2 (strong) (for dq = 1 g/kg)
    ablcl1 = 12.0  # Absorptivity of "thick" clouds in window band (below cloud top)
    ablcl2 = 0.6   # Absorptivity of "thin" upper clouds in window and H2O bands

    # parameters for `clouds`
    
    rhcl1   = 0.30  # Relative humidity threshold corresponding to cloud cover = 0
    rhcl2   = 1.00  # Relative humidity correponding to cloud cover = 1
    qacl    = 0.20  # Specific humidity threshold for cloud cover
    wpcl    = 0.2   # Cloud cover weight for the square-root of precipitation (for p = 1 mm/day)
    pmaxcl  = 10.0  # Maximum value of precipitation (mm/day) contributing to cloud cover
    clsmax  = 0.60  # Maximum stratiform cloud cover
    clsminl = 0.15  # Minimum stratiform cloud cover over land (for RH = 1)
    gse_s0  = 0.25  # Gradient of dry static energy corresponding to stratiform cloud cover = 0
    gse_s1  = 0.40  # Gradient of dry static energy corresponding to stratiform cloud cover = 1

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ModRadConParameters:
    # Albedo values
    albsea = 0.07  # Albedo over sea
    albice = 0.60  # Albedo over sea ice (for ice fraction = 1)
    albsn  = 0.60  # Albedo over snow (for snow cover = 1)

    # Longwave parameters
    epslw  = 0.05  # Fraction of blackbody spectrum absorbed/emitted by PBL only
    emisfc = 0.98  # Longwave surface emissivity
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SurfaceFluxParameters:
    fwind0 = 0.95 # Ratio of near-sfc wind to lowest-level wind

    # Weight for near-sfc temperature extrapolation (0-1) :
    # 1 : linear extrapolation from two lowest levels
    # 0 : constant potential temperature ( = lowest level)
    ftemp0 = 1.0

    # Weight for near-sfc specific humidity extrapolation (0-1) :
    # 1 : extrap. with constant relative hum. ( = lowest level)
    # 0 : constant specific hum. ( = lowest level)
    fhum0 = 0.0

    cdl = 2.4e-3   # Drag coefficient for momentum over land
    cds = 1.0e-3   # Drag coefficient for momentum over sea
    chl = 1.2e-3   # Heat exchange coefficient over land
    chs = 0.9e-3   # Heat exchange coefficient over sea
    vgust = 5.0    # Wind speed for sub-grid-scale gusts
    ctday = 1.0e-2 # Daily-cycle correction (dTskin/dSSRad)
    dtheta = 3.0   # Potential temp. gradient for stability correction
    fstab = 0.67   # Amplitude of stability correction (fraction)
    clambda = 7.0  # Heat conductivity in skin-to-root soil layer
    clambsn = 7.0  # Heat conductivity in soil for snow cover = 1

    lscasym = True   # true : use an asymmetric stability coefficient
    lskineb = True   # true : redefine skin temp. from energy balance

    hdrag = 2000.0 # Height scale for orographic correction

    def isnan(self):
        self.lscasym = 0
        self.lskineb = 0
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class VerticalDiffusionParameters:
    trshc = 6.0  # Relaxation time (in hours) for shallow convection
    trvdi = 24.0  # Relaxation time (in hours) for moisture diffusion
    trvds = 6.0  # Relaxation time (in hours) for super-adiabatic conditions
    redshc = 0.5  # Reduction factor of shallow convection in areas of deep convection
    rhgrad = 0.5  # Maximum gradient of relative humidity (d_RH/d_sigma)
    segrad = 0.1  # Minimum gradient of dry static energy (d_DSE/d_phi)

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class LandModelParameters:
    sd2sc = 60.0 # Snow depth (mm water) corresponding to snow cover = 1
    # Soil moisture parameters
    swcap = 0.30 # Soil wetness at field capacity (volume fraction)
    swwil = 0.17 # Soil wetness at wilting point  (volume fraction)
    thrsh = 0.1 # Threshold for land-sea mask definition (i.e. minimum fraction of either land or sea)
    # Model parameters (default values)
    depth_soil = 1.0 # Soil layer depth (m)
    depth_lice = 5.0 # Land-ice depth (m)
    tdland  = 40.0 # Dissipation time (days) for land-surface temp. anomalies
    flandmin = 1.0/3.0 # Minimum fraction of land for the definition of anomalies
    hcapl  = depth_soil*2.50e+6 # Heat capacities per m^2 (depth*heat_cap/m^3)
    hcapli = depth_lice*1.93e+6

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class Parameters:
    convection: ConvectionParameters
    condensation: CondensationParameters
    shortwave_radiation: ShortwaveRadiationParameters
    mod_radcon: ModRadConParameters
    surface_flux: SurfaceFluxParameters
    vertical_diffusion: VerticalDiffusionParameters
    land_model: LandModelParameters

    @classmethod
    def init(self):
        return Parameters(
            convection = ConvectionParameters(),
            condensation = CondensationParameters(),
            shortwave_radiation = ShortwaveRadiationParameters(),
            mod_radcon = ModRadConParameters(),
            surface_flux = SurfaceFluxParameters(),
            vertical_diffusion = VerticalDiffusionParameters(),
            land_model = LandModelParameters(),
        )
