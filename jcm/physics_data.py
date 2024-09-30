import jax.numpy as jnp
import tree_math

@tree_math.struct
class DateData:
    tyear: jnp.ndarray # Fractional time of year

    def __init__(self, tyear=None) -> None:
        if tyear is not None:
            self.tyear = tyear
        else:
            self.tyear = jnp.zeros((1))

    def copy(self, tyear=None):
        return DateData(
            tyear if tyear is not None else self.tyear
        )
    
@tree_math.struct
class LWRadiationData:
    slrd: jnp.ndarray # Downward flux of long-wave radiation at the surface
    dfabs: jnp.ndarray # Flux of long-wave radiation absorbed in each atmospheric layer
    # ftop: jnp.ndarray
    # slr: jnp.ndarray
    
    def __init__(self, nodal_shape, node_levels, slrd=None, dfabs=None, ftop=None, slr=None) -> None:
        if slrd is not None:
            self.slrd = slrd
        else:
            self.slrd = jnp.zeros((nodal_shape))
        if dfabs is not None:
            self.dfabs = dfabs
        else:
            self.dfabs = jnp.zeros((nodal_shape + (node_levels,)))
        if ftop is not None:
            self.ftop = ftop
        else:
            self.ftop = jnp.zeros((nodal_shape))
        if slr is not None:
            self.slr = slr
        else:
            self.slr = jnp.zeros((nodal_shape))

    def copy(self, slrd=None, dfabs=None, ftop=None, slr=None):
        return LWRadiationData(
            (0,0), 
            0, 
            slrd=slrd if slrd is not None else self.slrd,
            dfabs=dfabs if dfabs is not None else self.dfabs,
            ftop=ftop if ftop is not None else self.ftop,
            slr=slr if slr is not None else self.slr
        )

@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray # Equivalent specific humidity of clouds - set by clouds() used by get_shortwave_rad_fluxes()
    fsol: jnp.ndarray # Solar radiation at the top
    ssrd: jnp.ndarray # Total downward flux of short-wave radiation at the surface
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

    def __init__(self, nodal_shape, node_levels, qcloud=None, fsol=None, ssrd=None, ssr=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None) -> None:
        if qcloud is not None:
            self.qcloud = qcloud
        else:
            self.qcloud = jnp.zeros((nodal_shape))
        if fsol is not None:
            self.fsol = fsol
        else:
            self.fsol = jnp.zeros((nodal_shape))
        if ssrd is not None:
            self.ssrd = ssrd
        else:
            self.ssrd = jnp.zeros((nodal_shape))
        if ssr is not None:
            self.ssr = ssr
        else:
            self.ssr = jnp.zeros((nodal_shape))
        if ozone is not None:
            self.ozone = ozone
        else:
            self.ozone = jnp.zeros((nodal_shape))
        if ozupp is not None:
            self.ozupp = ozupp
        else:
            self.ozupp = jnp.zeros((nodal_shape))
        if zenit is not None:
            self.zenit = zenit
        else:
            self.zenit = jnp.zeros((nodal_shape))
        if stratz is not None:
            self.stratz = stratz
        else:
            self.stratz = jnp.zeros((nodal_shape))
        if gse is not None:
            self.gse = gse
        else:
            self.gse = jnp.zeros((nodal_shape))
        if icltop is not None:
            self.icltop = icltop
        else:
            self.icltop = jnp.zeros((nodal_shape))
        if cloudc is not None:
            self.cloudc = cloudc
        else:
            self.cloudc = jnp.zeros((nodal_shape))
        if cloudstr is not None:
            self.cloudstr = cloudstr
        else:
            self.cloudstr = jnp.zeros((nodal_shape))
        if ftop is not None:
            self.ftop = ftop
        else:
            self.ftop = jnp.zeros((nodal_shape))
        if dfabs is not None:
            self.dfabs = dfabs
        else:
            self.dfabs = jnp.zeros((nodal_shape + (node_levels,)))


    def copy(self, qcloud=None, fsol=None, ssrd=None, ssr=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationData(
            self.cloudc.shape,
            0,
            qcloud=qcloud if qcloud is not None else self.qcloud,
            fsol=fsol if fsol is not None else self.fsol,
            ssrd=ssrd if ssrd is not None else self.ssrd,
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


    def __init__(self, nodal_shape, node_levels, fband=None,alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None) -> None:
        if fband is not None:
            self.fband = fband
        else:
            self.fband = jnp.zeros((301,4))
        if alb_l is not None:
            self.alb_l = alb_l
        else:
            self.alb_l = jnp.zeros((nodal_shape))
        if alb_s is not None:
            self.alb_s = alb_s
        else:
            self.alb_s = jnp.zeros((nodal_shape))
        if albsfc is not None:
            self.albsfc = albsfc
        else:
            self.albsfc = jnp.zeros((nodal_shape))
        if snowc is not None:
            self.snowc = snowc
        else:
            self.snowc = jnp.zeros((nodal_shape))
        if tau2 is not None:
            self.tau2 = tau2
        else:
            self.tau2 = jnp.zeros((nodal_shape+(node_levels,)+(4,)))
        if st4a is not None:
            self.st4a = st4a
        else:
            self.st4a = jnp.zeros((nodal_shape+(node_levels,)+(2,)))
        if stratc is not None:
            self.stratc = stratc
        else:
            self.stratc = jnp.zeros((nodal_shape+(2,)))
        if flux is not None:
            self.flux = flux
        else:
            self.flux = jnp.zeros((nodal_shape+(4,)))

    def copy(self,fband=None,alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            self.alb_l.shape, 
            0, # this value isn't necessary since we will pass through values for each member of the struct
            fband=fband if fband is not None else self.fband,
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
    tsea: jnp.ndarray # SST
    
    def __init__(self, nodal_shape, tsea=None) -> None:
        if tsea is not None:
            self.tsea = tsea
        else:
            self.tsea = jnp.zeros((nodal_shape))

    def copy(self, tsea=None):
        return CondensationData(
            self.tsea.shape,
            tsea=tsea if tsea is not None else self.tsea, 
        )

@tree_math.struct
class CondensationData:
    precls: jnp.ndarray # Precipitation due to large-scale condensation

    def __init__(self, nodal_shape, node_levels, precls=None, dtlsc=None, dqlsc=None) -> None:
        if precls is not None:
            self.precls = precls
        else:
            self.precls = jnp.zeros((nodal_shape))
        if dtlsc is not None:
            self.dtlsc = dtlsc
        else:
            self.dtlsc = jnp.zeros((nodal_shape+(node_levels,)))
        if dqlsc is not None:
            self.dqlsc = dqlsc
        else:
            self.dqlsc = jnp.zeros((nodal_shape+(node_levels,)))

    def copy(self, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            self.precls.shape,
            0,
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
    dfse: jnp.ndarray # Net flux of dry static energy into each atmospheric layer
    dfqa: jnp.ndarray # Net flux of specific humidity into each atmospheric layer

    def __init__(self, nodal_shape, node_levels, psa=None, se=None, iptop=None, cbmf=None, precnv=None, dfse=None, dfqa=None) -> None:
        if psa is not None:
            self.psa = psa
        else:
            self.psa = jnp.zeros((nodal_shape))
        if se is not None:
            self.se = se
        else:
            self.se = jnp.zeros((nodal_shape + (node_levels,)))
        if iptop is not None:
            self.iptop = iptop
        else:
            self.iptop = jnp.zeros((nodal_shape),dtype=int)
        if cbmf is not None:
            self.cbmf = cbmf
        else:
            self.cbmf = jnp.zeros((nodal_shape))
        if precnv is not None:
            self.precnv = precnv
        else:
            self.precnv = jnp.zeros((nodal_shape))
        if dfse is not None:
            self.dfse = dfse
        else:
            self.dfse = jnp.zeros((nodal_shape + (node_levels,)))
        if dfqa is not None:
            self.dfqa = dfqa
        else:
            self.dfqa = jnp.zeros((nodal_shape + (node_levels,)))

    def copy(self, psa=None, se=None, iptop=None, cbmf=None, precnv=None, dfse=None, dfqa=None):
        return ConvectionData(
            self.psa.shape, 
            self.se.shape[-1], 
            psa=psa if psa is not None else self.psa,
            se=se if se is not None else self.se,
            iptop=iptop if iptop is not None else self.iptop,
            cbmf=cbmf if cbmf is not None else self.cbmf,
            precnv=precnv if precnv is not None else self.precnv,
            dfse=dfse if dfse is not None else self.dfse,
            dfqa=dfqa if dfqa is not None else self.dfqa
        )

@tree_math.struct
class HumidityData:
    rh: jnp.ndarray # relative humidity
    qsat: jnp.ndarray # saturation specific humidity

    def __init__(self, nodal_shape, node_levels, rh=None, qsat=None) -> None:
        if rh is not None:
            self.rh = rh
        else:
            self.rh = jnp.zeros((nodal_shape+(node_levels,)))
        if qsat is not None:
            self.qsat = qsat
        else:
            self.qsat = jnp.zeros((nodal_shape+(node_levels,)))


    def copy(self, rh=None, qsat=None):
        return HumidityData(
            self.rh.shape[0:1],
            self.rh.shape[2],
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
        if stl_am is not None:
            self.stl_am = stl_am
        else:
            self.stl_am = jnp.full((nodal_shape), 288.0)
        if soilw_am is not None:
            self.soilw_am = soilw_am
        else:
            self.soilw_am = jnp.full((nodal_shape), 0.5)
        if lfluxland is not None:
            self.lfluxland = lfluxland
        else:
            self.lfluxland = True
        if ustr is not None:
            self.ustr = ustr
        else:
            self.ustr = jnp.zeros((nodal_shape)+(3,))
        if vstr is not None:
            self.vstr = vstr
        else:
            self.vstr = jnp.zeros((nodal_shape)+(3,))
        if shf is not None:
            self.shf = shf
        else:
            self.shf = jnp.zeros((nodal_shape)+(3,))
        if evap is not None:
            self.evap = evap
        else:
            self.evap = jnp.zeros((nodal_shape)+(3,))
        if slru is not None:
            self.slru = slru
        else:
            self.slru = jnp.zeros((nodal_shape)+(3,))
        if hfluxn is not None:
            self.hfluxn = hfluxn
        else:
            self.hfluxn = jnp.zeros((nodal_shape)+(2,))
        if tsfc is not None:
            self.tsfc = tsfc
        else:
            self.tsfc = jnp.zeros((nodal_shape))
        if tskin is not None:
            self.tskin = tskin
        else:
            self.tskin = jnp.zeros((nodal_shape))
        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = jnp.zeros((nodal_shape))
        if v0 is not None:
            self.v0 = v0
        else:
            self.v0 = jnp.zeros((nodal_shape))
        if t0 is not None:
            self.t0 = t0
        else:
            self.t0 = jnp.zeros((nodal_shape))
        if fmask is not None:
            self.fmask = fmask
        else:
            self.fmask = jnp.zeros((nodal_shape))
        if phi0 is not None:
            self.phi0 = phi0
        else:
            self.phi0 = jnp.zeros((nodal_shape))

    
    def copy(self, stl_am=None, soilw_am=None, lfluxland=None, ustr=None, vstr=None, shf=None, evap=None, slru=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None, fmask=None, phi0=None):
        return SurfaceFluxData(
            self.stl_am.shape,
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
        if longwave_rad is not None:
            self.longwave_rad = longwave_rad
        else:
            self.longwave_rad = LWRadiationData(nodal_shape, node_levels)
        if shortwave_rad is not None:
            self.shortwave_rad = shortwave_rad
        else:
            self.shortwave_rad = SWRadiationData(nodal_shape, node_levels)
        if convection is not None:
            self.convection = convection
        else:
            self.convection = ConvectionData(nodal_shape, node_levels)
        if mod_radcon is not None:
            self.mod_radcon = mod_radcon
        else:
            self.mod_radcon = ModRadConData(nodal_shape, node_levels)
        if humidity is not None:
            self.humidity = humidity
        else:
            self.humidity = HumidityData(nodal_shape, node_levels)
        if condensation is not None:
            self.condensation = condensation
        else:
            self.condensation = CondensationData(nodal_shape, node_levels)
        if surface_flux is not None:
            self.surface_flux = surface_flux
        else:
            self.surface_flux = SurfaceFluxData(nodal_shape)
        if date is not None:
            self.date = date
        else:
            self.date = DateData()
        if sea_model is not None:
            self.sea_model = sea_model
        else:
            self.sea_model = SeaModelData(nodal_shape)

    def copy(self,shortwave_rad=None,longwave_rad=None,convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, sea_model=None):
        return PhysicsData(
            (0,0),
            0,
            shortwave_rad if shortwave_rad is not None else self.shortwave_rad,
            longwave_rad if longwave_rad is not None else self.longwave_rad,
            convection if convection is not None else self.convection,
            mod_radcon if mod_radcon is not None else self.mod_radcon,
            humidity if humidity is not None else self.humidity,
            condensation if condensation is not None else self.condensation,
            surface_flux if surface_flux is not None else self.surface_flux,
            date if date is not None else self.date,
            sea_model if sea_model is not None else self.sea_model
        )
