import jax.numpy as jnp
import tree_math
from jcm.date import DateData
 
@tree_math.struct
class LWRadiationData:
    rlds: jnp.ndarray # Downward flux of long-wave radiation at the surface
    dfabs: jnp.ndarray # Flux of long-wave radiation absorbed in each atmospheric layer
    ftop: jnp.ndarray
    slr: jnp.ndarray

    @classmethod
    def zeros(self, nodal_shape, node_levels, rlds=None, dfabs=None, ftop=None, slr=None):
        return LWRadiationData(
            rlds = rlds if rlds is not None else jnp.zeros((nodal_shape)),
            dfabs = dfabs if dfabs is not None else jnp.zeros((nodal_shape + (node_levels,))),
            ftop = ftop if ftop is not None else jnp.zeros((nodal_shape)),
            slr = slr if slr is not None else jnp.zeros((nodal_shape)),
        )

    def copy(self, rlds=None, dfabs=None, ftop=None, slr=None):
        return LWRadiationData(
            rlds=rlds if rlds is not None else self.rlds,
            dfabs=dfabs if dfabs is not None else self.dfabs,
            ftop=ftop if ftop is not None else self.ftop,
            slr=slr if slr is not None else self.slr
        )

@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray # Equivalent specific humidity of clouds - set by clouds() used by get_shortwave_rad_fluxes()
    fsol: jnp.ndarray # Solar radiation at the top
    rsds: jnp.ndarray # Total downward flux of short-wave radiation at the surface
    ssr: jnp.ndarray # Net downward flux of short-wave radiation at the surface
    ozone: jnp.ndarray # Ozone concentration in lower stratosphere
    ozupp: jnp.ndarray# Ozone depth in upper stratosphere
    zenit: jnp.ndarray # The Zenit angle
    stratz: jnp.ndarray # Polar night cooling in the stratosphere
    gse: jnp.ndarray # Vertical gradient of dry static energy
    icltop: jnp.ndarray # Cloud top level
    cloudc: jnp.ndarray # Total cloud cover
    cloudstr: jnp.ndarray # Stratiform cloud cover
    ftop: jnp.ndarray # Net downward flux of short-wave radiation at the top of the atmosphere
    dfabs: jnp.ndarray #Flux of short-wave radiation absorbed in each atmospheric layer

    @classmethod
    def zeros(self, nodal_shape, node_levels, qcloud=None, fsol=None, rsds=None, ssr=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationData(
            qcloud = qcloud if qcloud is not None else jnp.zeros((nodal_shape)),
            fsol = fsol if fsol is not None else jnp.zeros((nodal_shape)),
            rsds = rsds if rsds is not None else jnp.zeros((nodal_shape)),
            ssr = ssr if ssr is not None else jnp.zeros((nodal_shape)),
            ozone = ozone if ozone is not None else jnp.zeros((nodal_shape)),
            ozupp = ozupp if ozupp is not None else jnp.zeros((nodal_shape)),
            zenit = zenit if zenit is not None else jnp.zeros((nodal_shape)),
            stratz = stratz if stratz is not None else jnp.zeros((nodal_shape)),
            gse = gse if gse is not None else jnp.zeros((nodal_shape)),
            icltop = icltop if icltop is not None else jnp.zeros((nodal_shape)),
            cloudc = cloudc if cloudc is not None else jnp.zeros((nodal_shape)),
            cloudstr = cloudstr if cloudstr is not None else jnp.zeros((nodal_shape)),
            ftop = ftop if ftop is not None else jnp.zeros((nodal_shape)),
            dfabs = dfabs if dfabs is not None else jnp.zeros((nodal_shape + (node_levels,)))
        )

    def copy(self, qcloud=None, fsol=None, rsds=None, ssr=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationData(
            qcloud=qcloud if qcloud is not None else self.qcloud,
            fsol=fsol if fsol is not None else self.fsol,
            rsds=rsds if rsds is not None else self.rsds,
            ssr=ssr if ssr is not None else self.ssr,
            ozone=ozone if ozone is not None else self.ozone,
            ozupp=ozupp if ozupp is not None else self.ozupp,
            zenit=zenit if zenit is not None else self.zenit,
            stratz=stratz if stratz is not None else self.stratz,
            gse=gse if gse is not None else self.gse,
            icltop=icltop if icltop is not None else self.icltop,
            cloudc=cloudc if cloudc is not None else self.cloudc,
            cloudstr=cloudstr if cloudstr is not None else self.cloudstr,
            ftop=ftop if ftop is not None else self.ftop,
            dfabs=dfabs if dfabs is not None else self.dfabs
        )
    
@tree_math.struct
class ModRadConData:
    # Time-invariant fields (arrays) - #FIXME: since this is time invariant, should it be intiailizd/held somewhere else?
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

    @classmethod
    def zeros(self, nodal_shape, node_levels, alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            alb_l = alb_l if alb_l is not None else jnp.zeros((nodal_shape)),
            alb_s = alb_s if alb_s is not None else jnp.zeros((nodal_shape)),
            albsfc = albsfc if albsfc is not None else jnp.zeros((nodal_shape)),
            snowc = snowc if snowc is not None else jnp.zeros((nodal_shape)),
            tau2 = tau2 if tau2 is not None else jnp.zeros((nodal_shape+(node_levels,)+(4,))),
            st4a = st4a if st4a is not None else jnp.zeros((nodal_shape+(node_levels,)+(2,))),
            stratc = stratc if stratc is not None else jnp.zeros((nodal_shape+(2,))),
            flux = flux if flux is not None else jnp.zeros((nodal_shape+(4,)))
        )

    def copy(self, alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            alb_l=alb_l if alb_l is not None else self.alb_l,
            alb_s=alb_s if alb_s is not None else self.alb_s,
            albsfc=albsfc if albsfc is not None else self.albsfc,
            snowc=snowc if snowc is not None else self.snowc,
            tau2=tau2 if tau2 is not None else self.tau2,
            st4a=st4a if st4a is not None else self.st4a,
            stratc=stratc if stratc is not None else self.stratc,
            flux=flux if flux is not None else self.flux
        )
@tree_math.struct
class SeaModelData:
    tsea: jnp.ndarray # SST, should come from sea_model.py
    
    @classmethod
    def zeros(self, nodal_shape, tsea=None):
        return SeaModelData(
            tsea = tsea if tsea is not None else jnp.zeros((nodal_shape))
        )

    @classmethod
    def copy(self, tsea=None):
        return CondensationData(
            tsea=tsea if tsea is not None else self.tsea, 
        )

@tree_math.struct
class CondensationData:
    precls: jnp.ndarray # Precipitation due to large-scale condensation
    dtlsc: jnp.ndarray
    dqlsc: jnp.ndarray

    @classmethod
    def zeros(self, nodal_shape, node_levels, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            precls = precls if precls is not None else jnp.zeros((nodal_shape)),
            dtlsc = dtlsc if dtlsc is not None else jnp.zeros((nodal_shape+(node_levels,))),
            dqlsc = dqlsc if dqlsc is not None else jnp.zeros((nodal_shape+(node_levels,))),
        )

    def copy(self, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            precls=precls if precls is not None else self.precls, 
            dtlsc=dtlsc if dtlsc is not None else self.dtlsc, 
            dqlsc=dqlsc if dqlsc is not None else self.dqlsc
        )

@tree_math.struct
class ConvectionData:
    psa: jnp.ndarray # normalized surface pressure 
    se: jnp.ndarray # dry static energy
    iptop: jnp.ndarray # Top of convection (layer index)
    cbmf: jnp.ndarray # Cloud-base mass flux
    precnv: jnp.ndarray # Convective precipitation [g/(m^2 s)]

    @classmethod
    def zeros(self, nodal_shape, node_levels, psa=None, se=None, iptop=None, cbmf=None, precnv=None):
        return ConvectionData(
            psa = psa if psa is not None else jnp.zeros((nodal_shape)),
            se = se if se is not None else jnp.zeros((nodal_shape + (node_levels,))),
            iptop = iptop if iptop is not None else jnp.zeros((nodal_shape),dtype=int),
            cbmf = cbmf if cbmf is not None else jnp.zeros((nodal_shape)),
            precnv = precnv if precnv is not None else jnp.zeros((nodal_shape)),
        )
    
    def copy(self, psa=None, se=None, iptop=None, cbmf=None, precnv=None):
        return ConvectionData(
            psa=psa if psa is not None else self.psa,
            se=se if se is not None else self.se,
            iptop=iptop if iptop is not None else self.iptop,
            cbmf=cbmf if cbmf is not None else self.cbmf,
            precnv=precnv if precnv is not None else self.precnv
        )

@tree_math.struct
class HumidityData:
    rh: jnp.ndarray # relative humidity
    qsat: jnp.ndarray # saturation specific humidity

    @classmethod
    def zeros(self, nodal_shape, node_levels, rh=None, qsat=None):
        return HumidityData(
            rh = rh if rh is not None else jnp.zeros((nodal_shape+(node_levels,))),
            qsat = qsat if qsat is not None else jnp.zeros((nodal_shape+(node_levels,)))
        )

    def copy(self, rh=None, qsat=None):
        return HumidityData(
            rh=rh if rh is not None else self.rh, 
            qsat=qsat if qsat is not None else self.qsat
        )

@tree_math.struct
class SurfaceFluxData:
    # TODO: check if any of these (fmask, phi0) need to be initialized and/or should be moved somewhere else
    stl_am: jnp.ndarray # Land surface temperature, should come from land_model.py
    soilw_am: jnp.ndarray # Soil water availability, should come from land_model.py
    lfluxland: bool # Land surface fluxes true or false, hard coded in physics.f90
    ustr: jnp.ndarray # u-stress
    vstr: jnp.ndarray # v-stress
    shf: jnp.ndarray # Sensible heat flux
    evap: jnp.ndarray # Evaporation
    slru: jnp.ndarray # Upward flux of long-wave radiation at the surface
    hfluxn: jnp.ndarray # Net downward heat flux
    tsfc: jnp.ndarray # Surface temperature
    tskin: jnp.ndarray # Skin surface temperature
    u0: jnp.ndarray # Near-surface u-wind
    v0: jnp.ndarray # Near-surface v-wind
    t0: jnp.ndarray # Near-surface temperature
    fmask: jnp.ndarray # Fractional land-sea mask, should come from boundaries.py 
    phi0: jnp.ndarray # Surface geopotential (i.e. orography), should come from boundaries.py

    @classmethod
    def zeros(self, nodal_shape, stl_am=None, soilw_am=None, lfluxland=None, ustr=None, vstr=None, shf=None, evap=None, slru=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None, fmask=None, phi0=None):
        return SurfaceFluxData(
            stl_am = stl_am if stl_am is not None else jnp.full((nodal_shape), 288.0),
            soilw_am = soilw_am if soilw_am is not None else jnp.full((nodal_shape), 0.5),
            lfluxland = lfluxland if lfluxland is not None else True,
            ustr = ustr if ustr is not None else jnp.zeros((nodal_shape)+(3,)),
            vstr = vstr if vstr is not None else jnp.zeros((nodal_shape)+(3,)),
            shf = shf if shf is not None else jnp.zeros((nodal_shape)+(3,)),
            evap = evap if evap is not None else jnp.zeros((nodal_shape)+(3,)),
            slru = slru if slru is not None else jnp.zeros((nodal_shape)+(3,)),
            hfluxn = hfluxn if hfluxn is not None else jnp.zeros((nodal_shape)+(2,)),
            tsfc = tsfc if tsfc is not None else jnp.zeros((nodal_shape)),
            tskin = tskin if tskin is not None else jnp.zeros((nodal_shape)),
            u0 = u0 if u0 is not None else jnp.zeros((nodal_shape)),
            v0 = v0 if v0 is not None else jnp.zeros((nodal_shape)),
            t0 = t0 if t0 is not None else jnp.zeros((nodal_shape)),
            fmask = fmask if fmask is not None else jnp.zeros((nodal_shape)),
            phi0 = phi0 if phi0 is not None else jnp.zeros((nodal_shape))
        )

    def copy(self, stl_am=None, soilw_am=None, lfluxland=None, ustr=None, vstr=None, shf=None, evap=None, slru=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None, fmask=None, phi0=None):
        return SurfaceFluxData(
            stl_am=stl_am if stl_am is not None else self.stl_am,
            soilw_am=soilw_am if soilw_am is not None else self.soilw_am,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland,
            ustr=ustr if ustr is not None else self.ustr,
            vstr=vstr if vstr is not None else self.vstr,
            shf=shf if shf is not None else self.shf,
            evap=evap if evap is not None else self.evap,
            slru=slru if slru is not None else self.slru,
            hfluxn=hfluxn if hfluxn is not None else self.hfluxn,
            tsfc=tsfc if tsfc is not None else self.tsfc,
            tskin=tskin if tskin is not None else self.tskin,
            u0=u0 if u0 is not None else self.u0,
            v0=v0 if v0 is not None else self.v0,
            t0=t0 if t0 is not None else self.t0,
            fmask=fmask if fmask is not None else self.fmask,
            phi0=phi0 if phi0 is not None else self.phi0
        )

#TODO: Make an abstract PhysicsData class that just describes the interface (not all the fields will be needed for all models)
@tree_math.struct
class PhysicsData:
    shortwave_rad: SWRadiationData
    longwave_rad: LWRadiationData
    convection: ConvectionData
    mod_radcon: ModRadConData
    humidity: HumidityData
    condensation: CondensationData
    surface_flux: SurfaceFluxData
    date: DateData
    sea_model: SeaModelData

    @classmethod
    def zeros(self, nodal_shape, node_levels, shortwave_rad=None, longwave_rad=None, convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, sea_model=None):
        return PhysicsData(        
            longwave_rad = longwave_rad if longwave_rad is not None else LWRadiationData.zeros(nodal_shape, node_levels),
            shortwave_rad = shortwave_rad if shortwave_rad is not None else SWRadiationData.zeros(nodal_shape, node_levels),
            convection = convection if convection is not None else ConvectionData.zeros(nodal_shape, node_levels),
            mod_radcon = mod_radcon if mod_radcon is not None else ModRadConData.zeros(nodal_shape, node_levels),
            humidity = humidity if humidity is not None else HumidityData.zeros(nodal_shape, node_levels),
            condensation = condensation if condensation is not None else CondensationData.zeros(nodal_shape, node_levels),
            surface_flux = surface_flux if surface_flux is not None else SurfaceFluxData.zeros(nodal_shape),
            date = date if date is not None else DateData.set_date(),
            sea_model = sea_model if sea_model is not None else SeaModelData.zeros(nodal_shape)
        )

    def copy(self, shortwave_rad=None,longwave_rad=None,convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, sea_model=None):
        return PhysicsData(
            shortwave_rad=shortwave_rad if shortwave_rad is not None else self.shortwave_rad,
            longwave_rad=longwave_rad if longwave_rad is not None else self.longwave_rad,
            convection=convection if convection is not None else self.convection,
            mod_radcon=mod_radcon if mod_radcon is not None else self.mod_radcon,
            humidity=humidity if humidity is not None else self.humidity,
            condensation=condensation if condensation is not None else self.condensation,
            surface_flux=surface_flux if surface_flux is not None else self.surface_flux,
            date=date if date is not None else self.date,
            sea_model=sea_model if sea_model is not None else self.sea_model
        )
