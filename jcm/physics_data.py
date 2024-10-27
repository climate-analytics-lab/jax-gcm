import jax.numpy as jnp
import tree_math

n_temperatures = 301
n_bands = 4

@tree_math.struct
class DateData:
    tyear: jnp.ndarray # Fractional time of year, should possibly be part of the model itself (i.e. not in physics_data)

    def __init__(self, tyear=None) -> None:
        self.tyear = tyear if tyear is not None else jnp.zeros((1))

    def copy(self, tyear=None):
        return DateData(
            tyear if tyear is not None else self.tyear
        )
    
@tree_math.struct
class LWRadiationData:
    rlds: jnp.ndarray # Downward flux of long-wave radiation at the surface
    dfabs: jnp.ndarray # Flux of long-wave radiation absorbed in each atmospheric layer
    
    def __init__(self, nodal_shape, node_levels, rlds=None, dfabs=None, ftop=None, slr=None) -> None:
        self.rlds = rlds if rlds is not None else jnp.zeros((nodal_shape))
        self.dfabs = dfabs if dfabs is not None else jnp.zeros((nodal_shape + (node_levels,)))
        self.ftop = ftop if ftop is not None else jnp.zeros((nodal_shape))
        self.slr = slr if slr is not None else jnp.zeros((nodal_shape))

    def copy(self, rlds=None, dfabs=None, ftop=None, slr=None):
        return LWRadiationData(
            nodal_shape=None, 
            node_levels=None, 
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

    def __init__(self, nodal_shape, node_levels, qcloud=None, fsol=None, rsds=None, ssr=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None) -> None:
        self.qcloud = qcloud if qcloud is not None else jnp.zeros((nodal_shape))
        self.fsol = fsol if fsol is not None else jnp.zeros((nodal_shape))
        self.rsds = rsds if rsds is not None else jnp.zeros((nodal_shape))
        self.ssr = ssr if ssr is not None else jnp.zeros((nodal_shape))
        self.ozone = ozone if ozone is not None else jnp.zeros((nodal_shape))
        self.ozupp = ozupp if ozupp is not None else jnp.zeros((nodal_shape))
        self.zenit = zenit if zenit is not None else jnp.zeros((nodal_shape))
        self.stratz = stratz if stratz is not None else jnp.zeros((nodal_shape))
        self.gse = gse if gse is not None else jnp.zeros((nodal_shape))
        self.icltop = icltop if icltop is not None else jnp.zeros((nodal_shape))
        self.cloudc = cloudc if cloudc is not None else jnp.zeros((nodal_shape))
        self.cloudstr = cloudstr if cloudstr is not None else jnp.zeros((nodal_shape))
        self.ftop = ftop if ftop is not None else jnp.zeros((nodal_shape))
        self.dfabs = dfabs if dfabs is not None else jnp.zeros((nodal_shape + (node_levels,)))

    def copy(self, qcloud=None, fsol=None, rsds=None, ssr=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationData(
            nodal_shape=None, 
            node_levels=None, 
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


    def __init__(self, nodal_shape, node_levels, alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None) -> None:
        self.alb_l = alb_l if alb_l is not None else jnp.zeros((nodal_shape))
        self.alb_s = alb_s if alb_s is not None else jnp.zeros((nodal_shape))
        self.albsfc = albsfc if albsfc is not None else jnp.zeros((nodal_shape))
        self.snowc = snowc if snowc is not None else jnp.zeros((nodal_shape))
        self.tau2 = tau2 if tau2 is not None else jnp.zeros((nodal_shape+(node_levels,)+(4,)))
        self.st4a = st4a if st4a is not None else jnp.zeros((nodal_shape+(node_levels,)+(2,)))
        self.stratc = stratc if stratc is not None else jnp.zeros((nodal_shape+(2,)))
        self.flux = flux if flux is not None else jnp.zeros((nodal_shape+(4,)))

    def copy(self,alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            nodal_shape=None, 
            node_levels=None, 
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
    
    def __init__(self, nodal_shape, tsea=None) -> None:
        self.tsea = tsea if tsea is not None else jnp.zeros((nodal_shape))

    def copy(self, tsea=None):
        return CondensationData(
            self.tsea.shape,
            tsea=tsea if tsea is not None else self.tsea, 
        )

@tree_math.struct
class CondensationData:
    precls: jnp.ndarray # Precipitation due to large-scale condensation

    def __init__(self, nodal_shape, node_levels, precls=None, dtlsc=None, dqlsc=None) -> None:
        self.precls = precls if precls is not None else jnp.zeros((nodal_shape))
        self.dtlsc = dtlsc if dtlsc is not None else jnp.zeros((nodal_shape+(node_levels,)))
        self.dqlsc = dqlsc if dqlsc is not None else jnp.zeros((nodal_shape+(node_levels,)))

    def copy(self, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            nodal_shape=None, 
            node_levels=None, 
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

    def __init__(self, nodal_shape, node_levels, psa=None, se=None, iptop=None, cbmf=None, precnv=None) -> None:
        self.psa = psa if psa is not None else jnp.zeros((nodal_shape))
        self.se = se if se is not None else jnp.zeros((nodal_shape + (node_levels,)))
        self.iptop = iptop if iptop is not None else jnp.zeros((nodal_shape),dtype=int)
        self.cbmf = cbmf if cbmf is not None else jnp.zeros((nodal_shape))
        self.precnv = precnv if precnv is not None else jnp.zeros((nodal_shape))

    def copy(self, psa=None, se=None, iptop=None, cbmf=None, precnv=None):
        return ConvectionData(
            nodal_shape=None, 
            node_levels=None, 
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

    def __init__(self, nodal_shape, node_levels, rh=None, qsat=None) -> None:
        self.rh = rh if rh is not None else jnp.zeros((nodal_shape+(node_levels,)))
        self.qsat = qsat if qsat is not None else jnp.zeros((nodal_shape+(node_levels,)))


    def copy(self, rh=None, qsat=None):
        return HumidityData(
            None,
            None,
            rh=rh if rh is not None else self.rh, 
            qsat=qsat if qsat is not None else self.qsat
        )

@tree_math.struct
class SurfaceFluxData:
    # TODO: check if any of these (fmask, phi0) need to be initialized and/or should be moved somewhere else
    stl_am: jnp.ndarray # Land surface temperature, should come from land_model.py
    soilw_am: jnp.ndarray # Soil water availability, should come from land_model.py
    lfluxland: jnp.bool # Land surface fluxes true or false, hard coded in physics.f90
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

    def __init__(self, nodal_shape, stl_am=None, soilw_am=None, lfluxland=None, ustr=None, vstr=None, shf=None, evap=None, slru=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None, fmask=None, phi0=None) -> None:
        self.stl_am = stl_am if stl_am is not None else jnp.full((nodal_shape), 288.0)
        self.soilw_am = soilw_am if soilw_am is not None else jnp.full((nodal_shape), 0.5)
        self.lfluxland = lfluxland if lfluxland is not None else True
        self.ustr = ustr if ustr is not None else jnp.zeros((nodal_shape)+(3,))
        self.vstr = vstr if vstr is not None else jnp.zeros((nodal_shape)+(3,))
        self.shf = shf if shf is not None else jnp.zeros((nodal_shape)+(3,))
        self.evap = evap if evap is not None else jnp.zeros((nodal_shape)+(3,))
        self.slru = slru if slru is not None else jnp.zeros((nodal_shape)+(3,))
        self.hfluxn = hfluxn if hfluxn is not None else jnp.zeros((nodal_shape)+(2,))
        self.tsfc = tsfc if tsfc is not None else jnp.zeros((nodal_shape))
        self.tskin = tskin if tskin is not None else jnp.zeros((nodal_shape))
        self.u0 = u0 if u0 is not None else jnp.zeros((nodal_shape))
        self.v0 = v0 if v0 is not None else jnp.zeros((nodal_shape))
        self.t0 = t0 if t0 is not None else jnp.zeros((nodal_shape))
        self.fmask = fmask if fmask is not None else jnp.zeros((nodal_shape))
        self.phi0 = phi0 if phi0 is not None else jnp.zeros((nodal_shape))

    
    def copy(self, stl_am=None, soilw_am=None, lfluxland=None, ustr=None, vstr=None, shf=None, evap=None, slru=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None, fmask=None, phi0=None):
        return SurfaceFluxData(
            nodal_shape=None, 
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

    def __init__(self, nodal_shape, node_levels,shortwave_rad=None, longwave_rad=None, convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, sea_model=None) -> None:
        self.longwave_rad = longwave_rad if longwave_rad is not None else LWRadiationData(nodal_shape, node_levels)
        self.shortwave_rad = shortwave_rad if shortwave_rad is not None else SWRadiationData(nodal_shape, node_levels)
        self.convection = convection if convection is not None else ConvectionData(nodal_shape, node_levels)
        self.mod_radcon = mod_radcon if mod_radcon is not None else ModRadConData(nodal_shape, node_levels)
        self.humidity = humidity if humidity is not None else HumidityData(nodal_shape, node_levels)
        self.condensation = condensation if condensation is not None else CondensationData(nodal_shape, node_levels)
        self.surface_flux = surface_flux if surface_flux is not None else SurfaceFluxData(nodal_shape)
        self.date = date if date is not None else DateData()
        self.sea_model = sea_model if sea_model is not None else SeaModelData(nodal_shape)

    def copy(self,shortwave_rad=None,longwave_rad=None,convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, sea_model=None):
        return PhysicsData(
            nodal_shape=None,
            node_levels=None,
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
