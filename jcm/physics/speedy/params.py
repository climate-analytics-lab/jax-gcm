"""
Date: 1/25/2024.
For storing variables used by multiple physics schemes.
"""
import tree_math
import jax.numpy as jnp
from jax import tree_util

@tree_math.struct
class ConvectionParameters:
    """Stores parameters for the convection scheme.

    Attributes:
        psmin: Minimum (normalised) surface pressure for the occurrence of convection
        trcnv: Time of relaxation (in hours) towards reference state
        rhil: Relative humidity threshold in intermeduate layers for secondary mass flux
        rhbl: Relative humidity threshold in the boundary layer
        entmax: Maximum entrainment as a fraction of cloud-base mass flux
        smf: Ratio between secondary and primary mass flux at cloud-base
    """
    psmin: jnp.ndarray  
    trcnv: jnp.ndarray  
    rhil: jnp.ndarray 
    rhbl: jnp.ndarray  
    entmax: jnp.ndarray  
    smf: jnp.ndarray  

    @classmethod
    def default(cls):
        return cls(
            psmin = jnp.array(0.8),
            trcnv = jnp.array(6.0),
            rhil = jnp.array(0.7),
            rhbl = jnp.array(0.9),
            entmax = jnp.array(0.5),
            smf = jnp.array(0.8)
        )

    def isnan(self):
        """Checks if any parameter is NaN."""
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ForcingParameters:
    """Stores parameters for external model forcings.

    Attributes:
        increase_co2: Minimum (normalised) surface pressure for the occurrence of convection
        co2_year_ref: Time of relaxation (in hours) towards reference state
    """
    increase_co2: jnp.bool 
    co2_year_ref: jnp.int32  

    @classmethod
    def default(cls):
        return cls(
            increase_co2 = True,
            co2_year_ref = 1950,
        )

    def isnan(self):
        """Checks if any parameter is NaN."""
        self.increase_co2 = 0
        self.co2_year_ref = 0
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class CondensationParameters:
    """Stores parameters for the large-scale condensation scheme.

    Attributes:
        trlsc: Relaxation time (in hours) for specific humidity.
        rhlsc: Maximum relative humidity threshold (at sigma=1).
        drhlsc: Vertical range of the relative humidity threshold.
        rhblsc: Relative humidity threshold for the boundary layer.
    """
    trlsc: jnp.ndarray   
    rhlsc: jnp.ndarray  
    drhlsc: jnp.ndarray  
    rhblsc: jnp.ndarray 

    @classmethod
    def default(cls):
        return cls(
            trlsc = jnp.array(4.0),
            rhlsc = jnp.array(0.9),
            drhlsc = jnp.array(0.1),
            rhblsc = jnp.array(0.95)
        )

    def isnan(self):
        """Checks if any parameter is NaN."""
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ShortwaveRadiationParameters:
    """Stores parameters for radiation and cloud schemes (get_zonal_average_fields).

    Attributes:
        albcl: Cloud albedo (for cloud cover = 1).
        albcls: Stratiform cloud albedo (for stratiform cloud cover = 1).
        absdry: Absorptivity of dry air (visible band).
        absaer: Absorptivity of aerosols (visible band).
        abswv1: Absorptivity of water vapour (weak band).
        abswv2: Absorptivity of water vapour (strong band).
        abscl1: Absorptivity of clouds (visible band, maximum value).
        abscl2: Absorptivity of clouds.
        ablwin: Absorptivity of air in the "window" band.
        ablwv1: Absorptivity of water vapour in H2O band 1 (weak).
        ablwv2: Absorptivity of water vapour in H2O band 2 (strong).
        ablcl1: Absorptivity of "thick" clouds in window band (below cloud top).
        ablcl2: Absorptivity of "thin" upper clouds in window and H2O bands.
        rhcl1: Relative humidity threshold for cloud cover = 0.
        rhcl2: Relative humidity threshold for cloud cover = 1.
        qacl: Specific humidity threshold for cloud cover.
        wpcl: Cloud cover weight for the square-root of precipitation.
        pmaxcl: Maximum precipitation (mm/day) contributing to cloud cover.
        clsmax: Maximum stratiform cloud cover.
        clsminl: Minimum stratiform cloud cover over land (for RH = 1).
        gse_s0: Gradient of dry static energy for stratiform cloud cover = 0.
        gse_s1: Gradient of dry static energy for stratiform cloud cover = 1.
    """
    # Cloud albedo
    albcl:  jnp.ndarray 
    albcls: jnp.ndarray 
    # Shortwave absorptivities (for dp = 10^5 Pa)
    absdry: jnp.ndarray 
    absaer: jnp.ndarray 
    abswv1: jnp.ndarray 
    abswv2: jnp.ndarray 
    abscl1: jnp.ndarray 
    abscl2: jnp.ndarray
    # Longwave absorptivities (for dp = 10^5 Pa)
    ablwin: jnp.ndarray 
    ablwv1: jnp.ndarray 
    ablwv2: jnp.ndarray 
    ablcl1: jnp.ndarray 
    ablcl2: jnp.ndarray 
    # parameters for `clouds`
    rhcl1: jnp.ndarray  
    rhcl2: jnp.ndarray  
    qacl: jnp.ndarray  
    wpcl: jnp.ndarray   
    pmaxcl: jnp.ndarray  
    clsmax: jnp.ndarray  
    clsminl: jnp.ndarray  
    gse_s0: jnp.ndarray 
    gse_s1: jnp.ndarray  

    @classmethod
    def default(cls):
        return cls(
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
        """Checks if any parameter is NaN."""
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class ModRadConParameters:
    """Stores parameters for modifying radiation and convection.

    Attributes:
        albsea: Albedo over sea.
        albice: Albedo over sea ice (for ice fraction = 1).
        albsn: Albedo over snow (for snow cover = 1).
        epslw: Fraction of blackbody spectrum absorbed/emitted by PBL only.
        emisfc: Longwave surface emissivity.
    """
    # Albedo values
    albsea: jnp.ndarray  
    albice: jnp.ndarray  
    albsn: jnp.ndarray 
    # Longwave parameters
    epslw: jnp.ndarray  
    emisfc: jnp.ndarray  

    @classmethod
    def default(cls):
        return cls(
            albsea = jnp.array(0.07),
            albice = jnp.array(0.60),
            albsn = jnp.array(0.60),
            epslw = jnp.array(0.05),
            emisfc = jnp.array(0.98)
        )

    def isnan(self):
        """Checks if any parameter is NaN."""
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SurfaceFluxParameters:
    """Stores parameters for the surface flux scheme.

    Attributes:
        fwind0: Ratio of near-surface wind to lowest-level wind.
        ftemp0: Weight for near-surface temperature extrapolation (0-1). If 1, linear extrapolation from 
        two lowest levels. If 0, constant potential temperature ( = lowest level)
        fhum0: Weight for near-surface specific humidity extrapolation (0-1). If 1, extrap. with constant 
        relative hum. ( = lowest level). If 0, constant specific hum. ( = lowest level). 
        cdl: Drag coefficient for momentum over land.
        cds: Drag coefficient for momentum over sea.
        chl: Heat exchange coefficient over land.
        chs: Heat exchange coefficient over sea.
        vgust: Wind speed for sub-grid-scale gusts.
        ctday: Daily-cycle correction (dTskin/dSSRad).
        dtheta: Potential temperature gradient for stability correction.
        fstab: Amplitude of stability correction (fraction).
        clambda: Heat conductivity in skin-to-root soil layer.
        clambsn: Heat conductivity in soil for snow cover = 1.
        lscasym: If True, use an asymmetric stability coefficient.
        lskineb: If True, redefine skin temp. from energy balance.
        hdrag: Height scale for orographic correction.
    """
    fwind0: jnp.ndarray 
    ftemp0: jnp.ndarray
    fhum0: jnp.ndarray
    cdl: jnp.ndarray   
    cds: jnp.ndarray   
    chl: jnp.ndarray  
    chs: jnp.ndarray  
    vgust: jnp.ndarray   
    ctday: jnp.ndarray 
    dtheta: jnp.ndarray  
    fstab: jnp.ndarray  
    clambda: jnp.ndarray  
    clambsn: jnp.ndarray
    lscasym: jnp.bool   
    lskineb: jnp.bool   
    hdrag: jnp.ndarray 

    @classmethod
    def default(cls):
        return cls(
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
        """Checks if any parameter is NaN."""
        self.lscasym = 0
        self.lskineb = 0
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class VerticalDiffusionParameters:
    """Stores parameters for the vertical diffusion scheme.

    Attributes:
        trshc: Relaxation time (in hours) for shallow convection.
        trvdi: Relaxation time (in hours) for moisture diffusion.
        trvds: Relaxation time (in hours) for super-adiabatic conditions.
        redshc: Reduction factor of shallow convection in deep convection areas.
        rhgrad: Maximum gradient of relative humidity (d_RH/d_sigma).
        segrad: Minimum gradient of dry static energy (d_DSE/d_phi).
    """
    trshc: jnp.ndarray
    trvdi: jnp.ndarray
    trvds: jnp.ndarray
    redshc: jnp.ndarray
    rhgrad: jnp.ndarray
    segrad: jnp.ndarray
    
    @classmethod
    def default(cls):
        return cls(
            trshc = jnp.array(6.0),
            trvdi = jnp.array(24.0),
            trvds = jnp.array(6.0),
            redshc = jnp.array(0.5),
            rhgrad = jnp.array(0.5),
            segrad = jnp.array(0.1)
        )

    def isnan(self):
        """Checks if any parameter is NaN."""
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class LandModelParameters:
    """Stores parameters for the land surface model.

    Attributes:
        sd2sc: Snow depth (mm water) corresponding to snow cover = 1.
        swcap: Soil wetness at field capacity (volume fraction).
        swwil: Soil wetness at wilting point (volume fraction).
        thrsh: Threshold for land-sea mask definition. (i.e. minimum fraction of either land or sea)
        depth_soil: Soil layer depth (m).
        depth_lice: Land-ice depth (m).
        tdland: Dissipation time (days) for land-surface temp. anomalies.
        flandmin: Minimum fraction of land for defining anomalies.
        hcapl: Heat capacity per m^2 for land.
        hcapli: Heat capacity per m^2 for land-ice.
    """
    sd2sc: jnp.ndarray 
    # Soil moisture parameters
    swcap: jnp.ndarray 
    swwil: jnp.ndarray
    thrsh: jnp.ndarray
    # Model parameters (default values)
    depth_soil: jnp.ndarray 
    depth_lice: jnp.ndarray 
    tdland: jnp.ndarray 
    flandmin: jnp.ndarray 
    hcapl: jnp.ndarray 
    hcapli: jnp.ndarray

    @classmethod
    def default(cls):
        return cls(
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
        """Checks if any parameter is NaN."""
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class Parameters:
    """A container for all physical parameter structures.

    This class aggregates all the individual parameter structs into a single
    pytree, making it easy to pass all model parameters around.

    Attributes:
        convection: Parameters for the convection scheme.
        condensation: Parameters for the condensation scheme.
        shortwave_radiation: Parameters for the radiation scheme.
        mod_radcon: Parameters for modifying radiation and convection.
        surface_flux: Parameters for the surface flux scheme.
        vertical_diffusion: Parameters for the vertical diffusion scheme.
        land_model: Parameters for the land model.
        forcing: Parameters for external forcings.
    """
    convection: ConvectionParameters
    condensation: CondensationParameters
    shortwave_radiation: ShortwaveRadiationParameters
    mod_radcon: ModRadConParameters
    surface_flux: SurfaceFluxParameters
    vertical_diffusion: VerticalDiffusionParameters
    land_model: LandModelParameters
    forcing: ForcingParameters

    @classmethod
    def default(cls):
        return cls(
            convection = ConvectionParameters.default(),
            condensation = CondensationParameters.default(),
            shortwave_radiation = ShortwaveRadiationParameters.default(),
            mod_radcon = ModRadConParameters.default(),
            surface_flux = SurfaceFluxParameters.default(),
            vertical_diffusion = VerticalDiffusionParameters.default(),
            land_model = LandModelParameters.default(),
            forcing = ForcingParameters.default()
        )

    def isnan(self):
        """Checks for NaN values across all nested parameter structures."""
        return Parameters(
            convection=self.convection.isnan(),
            condensation=self.condensation.isnan(),
            shortwave_radiation=self.shortwave_radiation.isnan(),
            mod_radcon = self.mod_radcon.isnan(),
            surface_flux = self.surface_flux.isnan(),
            vertical_diffusion = self.vertical_diffusion.isnan(),
            land_model = self.land_model.isnan(),
            forcing = self.forcing.isnan()
        )

    def any_true(self):
        """Checks if any boolean-like value in the entire parameter tree is True."""
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))
