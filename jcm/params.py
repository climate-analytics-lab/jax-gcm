'''
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
'''
import tree_math
import jax.numpy as jnp
from jax import tree_util 

@tree_math.struct
class ConvectionParameters:
    psmin: jnp.ndarray # Minimum (normalised) surface pressure for the occurrence of convection
    trcnv: jnp.ndarray # Time of relaxation (in hours) towards reference state
    rhil: jnp.ndarray # Relative humidity threshold in intermeduate layers for secondary mass flux
    rhbl: jnp.ndarray # Relative humidity threshold in the boundary layer
    entmax: jnp.ndarray # Maximum entrainment as a fraction of cloud-base mass flux
    smf: jnp.ndarray # Ratio between secondary and primary mass flux at cloud-base

    @classmethod
    def default(self):
        return ConvectionParameters(
            psmin = jnp.array(0.8),
            trcnv = jnp.array(6.0),
            rhil = jnp.array(0.7),
            rhbl = jnp.array(0.9),
            entmax = jnp.array(0.5),
            smf = jnp.array(0.8)
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class CondensationParameters:
    trlsc: jnp.ndarray   # Relaxation time (in hours) for specific humidity
    rhlsc: jnp.ndarray  # Maximum relative humidity threshold (at sigma=1)
    drhlsc: jnp.ndarray  # Vertical range of relative humidity threshold
    rhblsc: jnp.ndarray # Relative humidity threshold for boundary layer
    
    @classmethod
    def default(self):
        return CondensationParameters(
            trlsc = jnp.array(4.0),
            rhlsc = jnp.array(0.9),
            drhlsc = jnp.array(0.1),
            rhblsc = jnp.array(0.95)
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ShortwaveRadiationParameters:
    # parameters for `get_zonal_average_fields`

    albcl:  jnp.ndarray  # Cloud albedo (for cloud cover = 1)
    albcls: jnp.ndarray # Stratiform cloud albedo (for st. cloud cover = 1)
    
    # Shortwave absorptivities (for dp = 10^5 Pa)
    absdry: jnp.ndarray # Absorptivity of dry air (visible band)
    absaer: jnp.ndarray # Absorptivity of aerosols (visible band)
    abswv1: jnp.ndarray # Absorptivity of water vapour
    abswv2: jnp.ndarray # Absorptivity of water vapour
    abscl1: jnp.ndarray # Absorptivity of clouds (visible band, maximum value)
    abscl2: jnp.ndarray # Absorptivity of clouds

    # Longwave absorptivities (for dp = 10^5 Pa)
    ablwin: jnp.ndarray   # Absorptivity of air in "window" band
    ablwv1: jnp.ndarray   # Absorptivity of water vapour in H2O band 1 (weak) (for dq = 1 g/kg)
    ablwv2: jnp.ndarray # Absorptivity of water vapour in H2O band 2 (strong) (for dq = 1 g/kg)
    ablcl1: jnp.ndarray # Absorptivity of "thick" clouds in window band (below cloud top)
    ablcl2: jnp.ndarray   # Absorptivity of "thin" upper clouds in window and H2O bands

    # parameters for `clouds`
    
    rhcl1: jnp.ndarray  # Relative humidity threshold corresponding to cloud cover = 0
    rhcl2: jnp.ndarray  # Relative humidity correponding to cloud cover = 1
    qacl: jnp.ndarray  # Specific humidity threshold for cloud cover
    wpcl: jnp.ndarray   # Cloud cover weight for the square-root of precipitation (for p = 1 mm/day)
    pmaxcl: jnp.ndarray  # Maximum value of precipitation (mm/day) contributing to cloud cover
    clsmax: jnp.ndarray  # Maximum stratiform cloud cover
    clsminl: jnp.ndarray  # Minimum stratiform cloud cover over land (for RH = 1)
    gse_s0: jnp.ndarray # Gradient of dry static energy corresponding to stratiform cloud cover = 0
    gse_s1: jnp.ndarray  # Gradient of dry static energy corresponding to stratiform cloud cover = 1

    @classmethod 
    def default(self):
        return ShortwaveRadiationParameters(
            albcl = jnp.array(0.43),
            albcls = jnp.array(0.50),
            absdry = jnp.array(0.033),
            absaer = jnp.array(0.033),
            abswv1 = jnp.array(0.022),
            abswv2 = jnp.array(15.000),
            abscl1 = jnp.array(0.015),
            abscl2 = jnp.array(0.15),
            ablwin = jnp.array(0.3),
            ablwv1 = jnp.array(0.7),
            ablwv2 = jnp.array(50.0),
            ablcl1 = jnp.array(12.0),
            ablcl2 = jnp.array(0.6),
            rhcl1 = jnp.array(0.30),
            rhcl2 = jnp.array(1.00),
            qacl = jnp.array(0.20),
            wpcl = jnp.array(0.2),
            pmaxcl = jnp.array(10.0),
            clsmax = jnp.array(0.60),
            clsminl = jnp.array(0.15),
            gse_s0 = jnp.array(0.25),
            gse_s1 = jnp.array(0.40)
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ModRadConParameters:
    # Albedo values
    albsea: jnp.ndarray  # Albedo over sea
    albice: jnp.ndarray  # Albedo over sea ice (for ice fraction = 1)
    albsn: jnp.ndarray # Albedo over snow (for snow cover = 1)

    # Longwave parameters
    epslw: jnp.ndarray  # Fraction of blackbody spectrum absorbed/emitted by PBL only
    emisfc: jnp.ndarray  # Longwave surface emissivity
    
    @classmethod
    def default(self):
        return ModRadConParameters(
            albsea = jnp.array(0.07),
            albice = jnp.array(0.60),
            albsn = jnp.array(0.60),
            epslw = jnp.array(0.05),
            emisfc = jnp.array(0.98)
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SurfaceFluxParameters:
    fwind0: jnp.ndarray # Ratio of near-sfc wind to lowest-level wind

    # Weight for near-sfc temperature extrapolation (0-1) :
    # 1 : linear extrapolation from two lowest levels
    # 0 : constant potential temperature ( = lowest level)
    ftemp0: jnp.ndarray

    # Weight for near-sfc specific humidity extrapolation (0-1) :
    # 1 : extrap. with constant relative hum. ( = lowest level)
    # 0 : constant specific hum. ( = lowest level)
    fhum0: jnp.ndarray

    cdl: jnp.ndarray   # Drag coefficient for momentum over land
    cds: jnp.ndarray   # Drag coefficient for momentum over sea
    chl: jnp.ndarray  # Heat exchange coefficient over land
    chs: jnp.ndarray   # Heat exchange coefficient over sea
    vgust: jnp.ndarray   # Wind speed for sub-grid-scale gusts
    ctday: jnp.ndarray # Daily-cycle correction (dTskin/dSSRad)
    dtheta: jnp.ndarray   # Potential temp. gradient for stability correction
    fstab: jnp.ndarray   # Amplitude of stability correction (fraction)
    clambda: jnp.ndarray  # Heat conductivity in skin-to-root soil layer
    clambsn: jnp.ndarray # Heat conductivity in soil for snow cover = 1

    lscasym: jnp.bool   # true : use an asymmetric stability coefficient
    lskineb: jnp.bool   # true : redefine skin temp. from energy balance

    hdrag: jnp.ndarray # Height scale for orographic correction

    @classmethod 
    def default(self):
        return SurfaceFluxParameters(
            fwind0 = jnp.array(0.95),
            ftemp0 = jnp.array(1.0),
            fhum0 = jnp.array(0.0),
            cdl = jnp.array(2.4e-3),
            cds = jnp.array(1.0e-3),
            chl = jnp.array(1.2e-3),
            chs = jnp.array(0.9e-3),
            vgust = jnp.array(5.0),
            ctday = jnp.array(1.0e-2),
            dtheta = jnp.array(3.0),
            fstab = jnp.array(0.67),
            clambda = jnp.array(7.0),
            clambsn = jnp.array(7.0),
            lscasym = True,
            lskineb = True,
            hdrag = jnp.array(2000.0)
        )
    
    def isnan(self):
        self.lscasym = 0
        self.lskineb = 0
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class VerticalDiffusionParameters:
    trshc: jnp.ndarray  # Relaxation time (in hours) for shallow convection
    trvdi: jnp.ndarray  # Relaxation time (in hours) for moisture diffusion
    trvds: jnp.ndarray  # Relaxation time (in hours) for super-adiabatic conditions
    redshc: jnp.ndarray  # Reduction factor of shallow convection in areas of deep convection
    rhgrad: jnp.ndarray  # Maximum gradient of relative humidity (d_RH/d_sigma)
    segrad: jnp.ndarray  # Minimum gradient of dry static energy (d_DSE/d_phi)

    @classmethod
    def default(self):
        return VerticalDiffusionParameters(
            trshc = jnp.array(6.0),
            trvdi = jnp.array(24.0),
            trvds = jnp.array(6.0),
            redshc = jnp.array(0.5),
            rhgrad = jnp.array(0.5),
            segrad = jnp.array(0.1)
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class LandModelParameters:
    sd2sc: jnp.ndarray # Snow depth (mm water) corresponding to snow cover = 1
    # Soil moisture parameters
    swcap: jnp.ndarray # Soil wetness at field capacity (volume fraction)
    swwil: jnp.ndarray # Soil wetness at wilting point  (volume fraction)
    thrsh: jnp.ndarray# Threshold for land-sea mask definition (i.e. minimum fraction of either land or sea)
    # Model parameters (default values)
    depth_soil: jnp.ndarray # Soil layer depth (m)
    depth_lice: jnp.ndarray # Land-ice depth (m)
    tdland: jnp.ndarray # Dissipation time (days) for land-surface temp. anomalies
    flandmin: jnp.ndarray # Minimum fraction of land for the definition of anomalies
    hcapl: jnp.ndarray # Heat capacities per m^2 (depth*heat_cap/m^3)
    hcapli: jnp.ndarray

    @classmethod 
    def default(self):
        return LandModelParameters(
            sd2sc = jnp.array(60.0),
            swcap = jnp.array(0.30),
            swwil = jnp.array(0.17),
            thrsh = jnp.array(0.1),
            depth_soil = jnp.array(1.0),
            depth_lice = jnp.array(5.0),
            tdland = jnp.array(40.0),
            flandmin = jnp.array(1.0/3.0),
            hcapl = jnp.array(1.0*2.50e+6),
            hcapli = jnp.array(5.0*1.93e+6)
        )

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
    def default(self):
        return Parameters(
            convection = ConvectionParameters.default(),
            condensation = CondensationParameters.default(),
            shortwave_radiation = ShortwaveRadiationParameters.default(),
            mod_radcon = ModRadConParameters.default(),
            surface_flux = SurfaceFluxParameters.default(),
            vertical_diffusion = VerticalDiffusionParameters.default(),
            land_model = LandModelParameters.default(),
        )
    
    def isnan(self):
        return Parameters(
            convection=self.convection.isnan(),
            condensation=self.condensation.isnan(),
            shortwave_radiation=self.shortwave_radiation.isnan(),
            mod_radcon = self.mod_radcon.isnan(),
            surface_flux = self.surface_flux.isnan(),
            vertical_diffusion = self.vertical_diffusion.isnan(),
            land_model = self.land_model.isnan(),
        )
    
    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))

    
