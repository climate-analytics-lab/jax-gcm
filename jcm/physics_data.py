import jax.numpy as jnp
import tree_math
from jcm.date import DateData
from jax import tree_util

ablco2_ref = 6.0

@tree_math.struct
class LWRadiationData:
    dfabs: jnp.ndarray # Flux of long-wave radiation absorbed in each atmospheric layer
    ftop: jnp.ndarray

    @classmethod
    def zeros(self, nodal_shape, node_levels, dfabs=None, ftop=None):
        return LWRadiationData(
            dfabs = dfabs if dfabs is not None else jnp.zeros((node_levels,)+nodal_shape),
            ftop = ftop if ftop is not None else jnp.zeros(nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, dfabs=None, ftop=None):
        return LWRadiationData(
            dfabs = dfabs if dfabs is not None else jnp.ones((node_levels,)+nodal_shape),
            ftop = ftop if ftop is not None else jnp.ones(nodal_shape),
        )

    def copy(self, dfabs=None, ftop=None):
        return LWRadiationData(
            dfabs=dfabs if dfabs is not None else self.dfabs,
            ftop=ftop if ftop is not None else self.ftop,
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SWRadiationOutputData:
    qcloud: jnp.ndarray # Equivalent specific humidity of clouds - set by clouds() used by get_shortwave_rad_fluxes()
    fsol: jnp.ndarray # Solar radiation at the top
    rsds: jnp.ndarray # Total downward flux of short-wave radiation at the surface
    rsns: jnp.ndarray # Net downward flux of short-wave radiation at the surface
    ozone: jnp.ndarray # Ozone concentration in lower stratosphere
    ozupp: jnp.ndarray# Ozone depth in upper stratosphere
    zenit: jnp.ndarray # The Zenit angle
    stratz: jnp.ndarray # Polar night cooling in the stratosphere
    gse: jnp.ndarray # Vertical gradient of dry static energy
    cloudc: jnp.ndarray # Total cloud cover
    cloudstr: jnp.ndarray # Stratiform cloud cover
    ftop: jnp.ndarray # Net downward flux of short-wave radiation at the top of the atmosphere
    dfabs: jnp.ndarray #Flux of short-wave radiation absorbed in each atmospheric layer
    
    @classmethod
    def zeros(self, nodal_shape, node_levels, qcloud=None, fsol=None, rsds=None, rsns=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationOutputData(
            qcloud = qcloud if qcloud is not None else jnp.zeros(nodal_shape),
            fsol = fsol if fsol is not None else jnp.zeros(nodal_shape),
            rsds = rsds if rsds is not None else jnp.zeros(nodal_shape),
            rsns = rsns if rsns is not None else jnp.zeros(nodal_shape),
            ozone = ozone if ozone is not None else jnp.zeros(nodal_shape),
            ozupp = ozupp if ozupp is not None else jnp.zeros(nodal_shape),
            zenit = zenit if zenit is not None else jnp.zeros(nodal_shape),
            stratz = stratz if stratz is not None else jnp.zeros(nodal_shape),
            gse = gse if gse is not None else jnp.zeros(nodal_shape),
            cloudc = cloudc if cloudc is not None else jnp.zeros(nodal_shape),
            cloudstr = cloudstr if cloudstr is not None else jnp.zeros(nodal_shape),
            ftop = ftop if ftop is not None else jnp.zeros(nodal_shape),
            dfabs = dfabs if dfabs is not None else jnp.zeros((node_levels,)+nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, qcloud=None, fsol=None, rsds=None, rsns=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationOutputData(
            qcloud = qcloud if qcloud is not None else jnp.ones(nodal_shape),
            fsol = fsol if fsol is not None else jnp.ones(nodal_shape),
            rsds = rsds if rsds is not None else jnp.ones(nodal_shape),
            rsns = rsns if rsns is not None else jnp.ones(nodal_shape),
            ozone = ozone if ozone is not None else jnp.ones(nodal_shape),
            ozupp = ozupp if ozupp is not None else jnp.ones(nodal_shape),
            zenit = zenit if zenit is not None else jnp.ones(nodal_shape),
            stratz = stratz if stratz is not None else jnp.ones(nodal_shape),
            gse = gse if gse is not None else jnp.ones(nodal_shape),
            cloudc = cloudc if cloudc is not None else jnp.ones(nodal_shape),
            cloudstr = cloudstr if cloudstr is not None else jnp.ones(nodal_shape),
            ftop = ftop if ftop is not None else jnp.ones(nodal_shape),
            dfabs = dfabs if dfabs is not None else jnp.ones((node_levels,)+nodal_shape),
        )

    def copy(self, qcloud=None, fsol=None, rsds=None, rsns=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None):
        return SWRadiationOutputData(
            qcloud=qcloud if qcloud is not None else self.qcloud,
            fsol=fsol if fsol is not None else self.fsol,
            rsds=rsds if rsds is not None else self.rsds,
            rsns=rsns if rsns is not None else self.rsns,
            ozone=ozone if ozone is not None else self.ozone,
            ozupp=ozupp if ozupp is not None else self.ozupp,
            zenit=zenit if zenit is not None else self.zenit,
            stratz=stratz if stratz is not None else self.stratz,
            gse=gse if gse is not None else self.gse,
            cloudc=cloudc if cloudc is not None else self.cloudc,
            cloudstr=cloudstr if cloudstr is not None else self.cloudstr,
            ftop=ftop if ftop is not None else self.ftop,
            dfabs=dfabs if dfabs is not None else self.dfabs,
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray # Equivalent specific humidity of clouds - set by clouds() used by get_shortwave_rad_fluxes()
    fsol: jnp.ndarray # Solar radiation at the top
    rsds: jnp.ndarray # Total downward flux of short-wave radiation at the surface
    rsns: jnp.ndarray # Net downward flux of short-wave radiation at the surface
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
    compute_shortwave: jnp.ndarray

    @classmethod
    def zeros(self, nodal_shape, node_levels, qcloud=None, fsol=None, rsds=None, rsns=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None, compute_shortwave=None):
        return SWRadiationData(
            qcloud = qcloud if qcloud is not None else jnp.zeros(nodal_shape),
            fsol = fsol if fsol is not None else jnp.zeros(nodal_shape),
            rsds = rsds if rsds is not None else jnp.zeros(nodal_shape),
            rsns = rsns if rsns is not None else jnp.zeros(nodal_shape),
            ozone = ozone if ozone is not None else jnp.zeros(nodal_shape),
            ozupp = ozupp if ozupp is not None else jnp.zeros(nodal_shape),
            zenit = zenit if zenit is not None else jnp.zeros(nodal_shape),
            stratz = stratz if stratz is not None else jnp.zeros(nodal_shape),
            gse = gse if gse is not None else jnp.zeros(nodal_shape),
            icltop = icltop if icltop is not None else jnp.zeros(nodal_shape, dtype=int),
            cloudc = cloudc if cloudc is not None else jnp.zeros(nodal_shape),
            cloudstr = cloudstr if cloudstr is not None else jnp.zeros(nodal_shape),
            ftop = ftop if ftop is not None else jnp.zeros(nodal_shape),
            dfabs = dfabs if dfabs is not None else jnp.zeros((node_levels,)+nodal_shape),
            compute_shortwave = compute_shortwave if compute_shortwave is not None else jnp.array(False)
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, qcloud=None, fsol=None, rsds=None, rsns=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None, compute_shortwave=None):
        return SWRadiationData(
            qcloud = qcloud if qcloud is not None else jnp.ones(nodal_shape),
            fsol = fsol if fsol is not None else jnp.ones(nodal_shape),
            rsds = rsds if rsds is not None else jnp.ones(nodal_shape),
            rsns = rsns if rsns is not None else jnp.ones(nodal_shape),
            ozone = ozone if ozone is not None else jnp.ones(nodal_shape),
            ozupp = ozupp if ozupp is not None else jnp.ones(nodal_shape),
            zenit = zenit if zenit is not None else jnp.ones(nodal_shape),
            stratz = stratz if stratz is not None else jnp.ones(nodal_shape),
            gse = gse if gse is not None else jnp.ones(nodal_shape),
            icltop = icltop if icltop is not None else jnp.ones(nodal_shape, dtype=int),
            cloudc = cloudc if cloudc is not None else jnp.ones(nodal_shape),
            cloudstr = cloudstr if cloudstr is not None else jnp.ones(nodal_shape),
            ftop = ftop if ftop is not None else jnp.ones(nodal_shape),
            dfabs = dfabs if dfabs is not None else jnp.ones((node_levels,)+nodal_shape),
            compute_shortwave = compute_shortwave if compute_shortwave is not None else jnp.array(True)
        )

    def copy(self, qcloud=None, fsol=None, rsds=None, rsns=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None, ftop=None, dfabs=None, compute_shortwave=None):
        return SWRadiationData(
            qcloud=qcloud if qcloud is not None else self.qcloud,
            fsol=fsol if fsol is not None else self.fsol,
            rsds=rsds if rsds is not None else self.rsds,
            rsns=rsns if rsns is not None else self.rsns,
            ozone=ozone if ozone is not None else self.ozone,
            ozupp=ozupp if ozupp is not None else self.ozupp,
            zenit=zenit if zenit is not None else self.zenit,
            stratz=stratz if stratz is not None else self.stratz,
            gse=gse if gse is not None else self.gse,
            icltop=icltop if icltop is not None else self.icltop,
            cloudc=cloudc if cloudc is not None else self.cloudc,
            cloudstr=cloudstr if cloudstr is not None else self.cloudstr,
            ftop=ftop if ftop is not None else self.ftop,
            dfabs=dfabs if dfabs is not None else self.dfabs,
            compute_shortwave=compute_shortwave if compute_shortwave is not None else self.compute_shortwave
        )
    
    def isnan(self):
        self.icltop = jnp.zeros_like(self.icltop, dtype=float)
        self.compute_shortwave = jnp.zeros_like(self.compute_shortwave, dtype=float)
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class ModRadConData:
    # Time-invariant fields (arrays) - #FIXME: since this is time invariant, should it be intiailizd/held somewhere else?
    # Radiative properties of the surface (updated in fordate)
    # Albedo and snow cover arrays
    ablco2: jnp.float32 # CO2 absorptivity
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
    def zeros(self, nodal_shape, node_levels, ablco2=None, alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            ablco2 = ablco2 if ablco2 is not None else ablco2_ref,
            alb_l = alb_l if alb_l is not None else jnp.zeros(nodal_shape),
            alb_s = alb_s if alb_s is not None else jnp.zeros(nodal_shape),
            albsfc = albsfc if albsfc is not None else jnp.zeros(nodal_shape),
            snowc = snowc if snowc is not None else jnp.zeros(nodal_shape),
            tau2 = tau2 if tau2 is not None else jnp.zeros(((node_levels,)+nodal_shape+(4,))),
            st4a = st4a if st4a is not None else jnp.zeros(((node_levels,)+nodal_shape+(2,))),
            stratc = stratc if stratc is not None else jnp.zeros((nodal_shape+(2,))),
            flux = flux if flux is not None else jnp.zeros((nodal_shape+(4,)))
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, ablco2=None, alb_l=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            ablco2 = ablco2 if ablco2 is not None else ablco2_ref,
            alb_l = alb_l if alb_l is not None else jnp.ones(nodal_shape),
            alb_s = alb_s if alb_s is not None else jnp.ones(nodal_shape),
            albsfc = albsfc if albsfc is not None else jnp.ones(nodal_shape),
            snowc = snowc if snowc is not None else jnp.ones(nodal_shape),
            tau2 = tau2 if tau2 is not None else jnp.ones(((node_levels,)+nodal_shape+(4,))),
            st4a = st4a if st4a is not None else jnp.ones(((node_levels,)+nodal_shape+(2,))),
            stratc = stratc if stratc is not None else jnp.ones((nodal_shape+(2,))),
            flux = flux if flux is not None else jnp.ones((nodal_shape+(4,)))
        )

    def copy(self, alb_l=None,alb_s=None,ablco2=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            ablco2=ablco2 if ablco2 is not None else self.ablco2,
            alb_l=alb_l if alb_l is not None else self.alb_l,
            alb_s=alb_s if alb_s is not None else self.alb_s,
            albsfc=albsfc if albsfc is not None else self.albsfc,
            snowc=snowc if snowc is not None else self.snowc,
            tau2=tau2 if tau2 is not None else self.tau2,
            st4a=st4a if st4a is not None else self.st4a,
            stratc=stratc if stratc is not None else self.stratc,
            flux=flux if flux is not None else self.flux
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
@tree_math.struct
class CondensationData:
    precls: jnp.ndarray # Precipitation due to large-scale condensation
    dtlsc: jnp.ndarray
    dqlsc: jnp.ndarray

    @classmethod
    def zeros(self, nodal_shape, node_levels, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            precls = precls if precls is not None else jnp.zeros(nodal_shape),
            dtlsc = dtlsc if dtlsc is not None else jnp.zeros((node_levels,)+nodal_shape),
            dqlsc = dqlsc if dqlsc is not None else jnp.zeros((node_levels,)+nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            precls = precls if precls is not None else jnp.ones(nodal_shape),
            dtlsc = dtlsc if dtlsc is not None else jnp.ones((node_levels,)+nodal_shape),
            dqlsc = dqlsc if dqlsc is not None else jnp.ones((node_levels,)+nodal_shape),
        )

    def copy(self, precls=None, dtlsc=None, dqlsc=None):
        return CondensationData(
            precls=precls if precls is not None else self.precls,
            dtlsc=dtlsc if dtlsc is not None else self.dtlsc,
            dqlsc=dqlsc if dqlsc is not None else self.dqlsc
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)
    
@tree_math.struct
class ConvectionOutputData:
    se: jnp.ndarray # dry static energy
    cbmf: jnp.ndarray # Cloud-base mass flux
    precnv: jnp.ndarray # Convective precipitation [g/(m^2 s)]

    @classmethod
    def zeros(self, nodal_shape, node_levels, se=None, cbmf=None, precnv=None):
        return ConvectionOutputData(
            se = se if se is not None else jnp.zeros((node_levels,)+nodal_shape),
            cbmf = cbmf if cbmf is not None else jnp.zeros(nodal_shape),
            precnv = precnv if precnv is not None else jnp.zeros(nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, se=None, cbmf=None, precnv=None):
        return ConvectionOutputData(
            se = se if se is not None else jnp.ones((node_levels,)+nodal_shape),
            cbmf = cbmf if cbmf is not None else jnp.ones(nodal_shape),
            precnv = precnv if precnv is not None else jnp.ones(nodal_shape),
        )
    
    def copy(self, se=None, cbmf=None, precnv=None):
        return ConvectionOutputData(
            se=se if se is not None else self.se,
            cbmf=cbmf if cbmf is not None else self.cbmf,
            precnv=precnv if precnv is not None else self.precnv
        )
    
    # Isnan function to check if any elements of ConvectionData are NaN. This function is used after getting the gradient of something with respect to
    # a ConvectionData input object, to check if the gradient is valid. We skip the check on iptop because it is an integer and the gradient is not meaningful
    # or intended to be used.
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)


@tree_math.struct
class ConvectionData:
    se: jnp.ndarray # dry static energy
    iptop: jnp.ndarray # Top of convection (layer index)
    cbmf: jnp.ndarray # Cloud-base mass flux
    precnv: jnp.ndarray # Convective precipitation [g/(m^2 s)]

    @classmethod
    def zeros(self, nodal_shape, node_levels, se=None, iptop=None, cbmf=None, precnv=None):
        return ConvectionData(
            se = se if se is not None else jnp.zeros((node_levels,)+nodal_shape),
            iptop = iptop if iptop is not None else jnp.zeros((nodal_shape)),
            cbmf = cbmf if cbmf is not None else jnp.zeros(nodal_shape),
            precnv = precnv if precnv is not None else jnp.zeros(nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, se=None, iptop=None, cbmf=None, precnv=None):
        return ConvectionData(
            se = se if se is not None else jnp.ones((node_levels,)+nodal_shape),
            iptop = iptop if iptop is not None else jnp.ones((nodal_shape)),
            cbmf = cbmf if cbmf is not None else jnp.ones(nodal_shape),
            precnv = precnv if precnv is not None else jnp.ones(nodal_shape),
        )
    
    def copy(self, se=None, iptop=None, cbmf=None, precnv=None):
        return ConvectionData(
            se=se if se is not None else self.se,
            iptop= iptop if iptop is not None else self.iptop,
            cbmf=cbmf if cbmf is not None else self.cbmf,
            precnv=precnv if precnv is not None else self.precnv
        )
    
    # Isnan function to check if any elements of ConvectionData are NaN. This function is used after getting the gradient of something with respect to
    # a ConvectionData input object, to check if the gradient is valid. We skip the check on iptop because it is an integer and the gradient is not meaningful
    # or intended to be used.
    def isnan(self):
        self.iptop = jnp.zeros_like(self.iptop, dtype=float)
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class HumidityData:
    rh: jnp.ndarray # relative humidity
    qsat: jnp.ndarray # saturation specific humidity

    @classmethod
    def zeros(self, nodal_shape, node_levels, rh=None, qsat=None):
        return HumidityData(
            rh = rh if rh is not None else jnp.zeros((node_levels,)+nodal_shape),
            qsat = qsat if qsat is not None else jnp.zeros((node_levels,)+nodal_shape)
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, rh=None, qsat=None):
        return HumidityData(
            rh = rh if rh is not None else jnp.ones((node_levels,)+nodal_shape),
            qsat = qsat if qsat is not None else jnp.ones((node_levels,)+nodal_shape)
        )

    def copy(self, rh=None, qsat=None):
        return HumidityData(
            rh=rh if rh is not None else self.rh,
            qsat=qsat if qsat is not None else self.qsat
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class SurfaceFluxData:
    ustr: jnp.ndarray # u-stress
    vstr: jnp.ndarray # v-stress
    shf: jnp.ndarray # Sensible heat flux
    evap: jnp.ndarray # Evaporation
    rlus: jnp.ndarray # Upward flux of long-wave radiation at the surface
    rlds: jnp.ndarray # Downward flux of long-wave radiation at the surface
    rlns: jnp.ndarray # Net upward flux of long-wave radiation at the surface
    hfluxn: jnp.ndarray # Net downward heat flux
    tsfc: jnp.ndarray # Surface temperature
    tskin: jnp.ndarray # Skin surface temperature
    u0: jnp.ndarray # Near-surface u-wind
    v0: jnp.ndarray # Near-surface v-wind
    t0: jnp.ndarray # Near-surface temperature

    @classmethod
    def zeros(self, nodal_shape, ustr=None, vstr=None, shf=None, evap=None, rlus=None, rlds=None, rlns=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None):
        return SurfaceFluxData(
            ustr = ustr if ustr is not None else jnp.zeros((nodal_shape)+(3,)),
            vstr = vstr if vstr is not None else jnp.zeros((nodal_shape)+(3,)),
            shf = shf if shf is not None else jnp.zeros((nodal_shape)+(3,)),
            evap = evap if evap is not None else jnp.zeros((nodal_shape)+(3,)),
            rlus = rlus if rlus is not None else jnp.zeros((nodal_shape)+(3,)),
            rlds = rlds if rlds is not None else jnp.zeros((nodal_shape)),
            rlns = rlns if rlns is not None else jnp.zeros((nodal_shape)),
            hfluxn = hfluxn if hfluxn is not None else jnp.zeros((nodal_shape)+(2,)),
            tsfc = tsfc if tsfc is not None else jnp.zeros(nodal_shape),
            tskin = tskin if tskin is not None else jnp.zeros(nodal_shape),
            u0 = u0 if u0 is not None else jnp.zeros(nodal_shape),
            v0 = v0 if v0 is not None else jnp.zeros(nodal_shape),
            t0 = t0 if t0 is not None else jnp.zeros(nodal_shape)
        )
    
    @classmethod
    def ones(self, nodal_shape, ustr=None, vstr=None, shf=None, evap=None, rlus=None, rlds=None, rlns=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None):
        return SurfaceFluxData(
            ustr = ustr if ustr is not None else jnp.ones((nodal_shape)+(3,)),
            vstr = vstr if vstr is not None else jnp.ones((nodal_shape)+(3,)),
            shf = shf if shf is not None else jnp.ones((nodal_shape)+(3,)),
            evap = evap if evap is not None else jnp.ones((nodal_shape)+(3,)),
            rlus = rlus if rlus is not None else jnp.ones((nodal_shape)+(3,)),
            rlds = rlds if rlds is not None else jnp.ones((nodal_shape)), # FIXME these shapes
            rlns = rlns if rlns is not None else jnp.ones((nodal_shape)),
            hfluxn = hfluxn if hfluxn is not None else jnp.ones((nodal_shape)+(2,)),
            tsfc = tsfc if tsfc is not None else jnp.ones(nodal_shape),
            tskin = tskin if tskin is not None else jnp.ones(nodal_shape),
            u0 = u0 if u0 is not None else jnp.ones(nodal_shape),
            v0 = v0 if v0 is not None else jnp.ones(nodal_shape),
            t0 = t0 if t0 is not None else jnp.ones(nodal_shape)
        )

    def copy(self, ustr=None, vstr=None, shf=None, evap=None, rlus=None, rlds=None, rlns=None, hfluxn=None, tsfc=None, tskin=None, u0=None, v0=None, t0=None):
        return SurfaceFluxData(
            ustr=ustr if ustr is not None else self.ustr,
            vstr=vstr if vstr is not None else self.vstr,
            shf=shf if shf is not None else self.shf,
            evap=evap if evap is not None else self.evap,
            rlus=rlus if rlus is not None else self.rlus,
            rlds=rlds if rlds is not None else self.rlds,
            rlns=rlns if rlns is not None else self.rlns,
            hfluxn=hfluxn if hfluxn is not None else self.hfluxn,
            tsfc=tsfc if tsfc is not None else self.tsfc,
            tskin=tskin if tskin is not None else self.tskin,
            u0=u0 if u0 is not None else self.u0,
            v0=v0 if v0 is not None else self.v0,
            t0=t0 if t0 is not None else self.t0
        )
    
    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

@tree_math.struct
class LandModelData:
    stl_lm: jnp.ndarray
    stl_am: jnp.ndarray
    
    @classmethod
    def zeros(self, nodal_shape, stl_lm=None, stl_am=None):
        return LandModelData(
            stl_am = stl_am if stl_am is not None else jnp.full((nodal_shape), 288.0),
            stl_lm = stl_lm if stl_lm is not None else jnp.full((nodal_shape), 288.0)
        )
    
    @classmethod
    def ones(self, nodal_shape, stl_lm=None, stl_am=None):
        return LandModelData(
            stl_am = stl_am if stl_am is not None else jnp.ones(nodal_shape),
            stl_lm = stl_lm if stl_lm is not None else jnp.ones(nodal_shape)
        )

    def copy(self, stl_lm=None, stl_am=None):
        return LandModelData(
            stl_am = stl_am if stl_am is not None else self.stl_am,
            stl_lm = stl_lm if stl_lm is not None else self.stl_lm
        )

    def isnan(self):
        return tree_util.tree_map(jnp.isnan, self)

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
    land_model: LandModelData

    @classmethod
    def zeros(self, nodal_shape, node_levels, shortwave_rad=None,longwave_rad=None, convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, land_model=None):
        return PhysicsData(
            longwave_rad = longwave_rad if longwave_rad is not None else LWRadiationData.zeros(nodal_shape, node_levels),
            shortwave_rad = shortwave_rad if shortwave_rad is not None else SWRadiationData.zeros(nodal_shape, node_levels),
            convection = convection if convection is not None else ConvectionData.zeros(nodal_shape, node_levels),
            mod_radcon = mod_radcon if mod_radcon is not None else ModRadConData.zeros(nodal_shape, node_levels),
            humidity = humidity if humidity is not None else HumidityData.zeros(nodal_shape, node_levels),
            condensation = condensation if condensation is not None else CondensationData.zeros(nodal_shape, node_levels),
            surface_flux = surface_flux if surface_flux is not None else SurfaceFluxData.zeros(nodal_shape),
            date = date if date is not None else DateData.zeros(),
            land_model = land_model if land_model is not None else LandModelData.zeros(nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, shortwave_rad=None, longwave_rad=None, convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, land_model=None):
        return PhysicsData(
            longwave_rad = longwave_rad if longwave_rad is not None else LWRadiationData.ones(nodal_shape, node_levels),
            shortwave_rad = shortwave_rad if shortwave_rad is not None else SWRadiationData.ones(nodal_shape, node_levels),
            convection = convection if convection is not None else ConvectionData.ones(nodal_shape, node_levels),
            mod_radcon = mod_radcon if mod_radcon is not None else ModRadConData.ones(nodal_shape, node_levels),
            humidity = humidity if humidity is not None else HumidityData.ones(nodal_shape, node_levels),
            condensation = condensation if condensation is not None else CondensationData.ones(nodal_shape, node_levels),
            surface_flux = surface_flux if surface_flux is not None else SurfaceFluxData.ones(nodal_shape),
            date = date if date is not None else DateData.ones(),
            land_model = land_model if land_model is not None else LandModelData.ones(nodal_shape)        )

    def copy(self, shortwave_rad=None,longwave_rad=None,convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, date=None, land_model=None):
        return PhysicsData(
            shortwave_rad=shortwave_rad if shortwave_rad is not None else self.shortwave_rad,
            longwave_rad=longwave_rad if longwave_rad is not None else self.longwave_rad,
            convection=convection if convection is not None else self.convection,
            mod_radcon=mod_radcon if mod_radcon is not None else self.mod_radcon,
            humidity=humidity if humidity is not None else self.humidity,
            condensation=condensation if condensation is not None else self.condensation,
            surface_flux=surface_flux if surface_flux is not None else self.surface_flux,
            date=date if date is not None else self.date,
            land_model=land_model if land_model is not None else self.land_model
        )

    # Isnan function to check if any elements of PhysicsData are NaN. This function is used after getting the gradient of something with respect to 
    # a PhysicsData input object, to check if the gradient is valid. We skip the check on the date because the gradient returns NaN in 
    # valid scenarios (due to the use of arccos() in the solar() function) and we would otherwise fail this check in those cases.
    def isnan(self):
        return PhysicsData(
            shortwave_rad=self.shortwave_rad.isnan(),
            longwave_rad=self.longwave_rad.isnan(),
            convection=self.convection.isnan(),
            mod_radcon=self.mod_radcon.isnan(),
            humidity=self.humidity.isnan(),
            condensation=self.condensation.isnan(),
            surface_flux=self.surface_flux.isnan(),
            date=0,
            land_model=self.land_model.isnan()
        )
    
    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))

@tree_math.struct
class PhysicsOutputData:
    shortwave_rad: SWRadiationOutputData
    longwave_rad: LWRadiationData
    convection: ConvectionOutputData
    mod_radcon: ModRadConData
    humidity: HumidityData
    condensation: CondensationData
    surface_flux: SurfaceFluxData
    land_model: LandModelData

    @classmethod
    def zeros(self, nodal_shape, node_levels, shortwave_rad=None,longwave_rad=None, convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, land_model=None):
        return PhysicsOutputData(
            longwave_rad = longwave_rad if longwave_rad is not None else LWRadiationData.zeros(nodal_shape, node_levels),
            shortwave_rad = shortwave_rad if shortwave_rad is not None else SWRadiationOutputData.zeros(nodal_shape, node_levels),
            convection = convection if convection is not None else ConvectionOutputData.zeros(nodal_shape, node_levels),
            mod_radcon = mod_radcon if mod_radcon is not None else ModRadConData.zeros(nodal_shape, node_levels),
            humidity = humidity if humidity is not None else HumidityData.zeros(nodal_shape, node_levels),
            condensation = condensation if condensation is not None else CondensationData.zeros(nodal_shape, node_levels),
            surface_flux = surface_flux if surface_flux is not None else SurfaceFluxData.zeros(nodal_shape),
            land_model = land_model if land_model is not None else LandModelData.zeros(nodal_shape),
        )
    
    @classmethod
    def ones(self, nodal_shape, node_levels, shortwave_rad=None, longwave_rad=None, convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, land_model=None):
        return PhysicsOutputData(
            longwave_rad = longwave_rad if longwave_rad is not None else LWRadiationData.ones(nodal_shape, node_levels),
            shortwave_rad = shortwave_rad if shortwave_rad is not None else SWRadiationOutputData.ones(nodal_shape, node_levels),
            convection = convection if convection is not None else ConvectionOutputData.ones(nodal_shape, node_levels),
            mod_radcon = mod_radcon if mod_radcon is not None else ModRadConData.ones(nodal_shape, node_levels),
            humidity = humidity if humidity is not None else HumidityData.ones(nodal_shape, node_levels),
            condensation = condensation if condensation is not None else CondensationData.ones(nodal_shape, node_levels),
            surface_flux = surface_flux if surface_flux is not None else SurfaceFluxData.ones(nodal_shape),
            land_model = land_model if land_model is not None else LandModelData.ones(nodal_shape)        )

    def copy(self, shortwave_rad=None,longwave_rad=None,convection=None, mod_radcon=None, humidity=None, condensation=None, surface_flux=None, land_model=None):
        return PhysicsOutputData(
            shortwave_rad=shortwave_rad if shortwave_rad is not None else self.shortwave_rad,
            longwave_rad=longwave_rad if longwave_rad is not None else self.longwave_rad,
            convection=convection if convection is not None else self.convection,
            mod_radcon=mod_radcon if mod_radcon is not None else self.mod_radcon,
            humidity=humidity if humidity is not None else self.humidity,
            condensation=condensation if condensation is not None else self.condensation,
            surface_flux=surface_flux if surface_flux is not None else self.surface_flux,
            land_model=land_model if land_model is not None else self.land_model
        )

    # Isnan function to check if any elements of PhysicsData are NaN. This function is used after getting the gradient of something with respect to 
    # a PhysicsData input object, to check if the gradient is valid. We skip the check on the date because the gradient returns NaN in 
    # valid scenarios (due to the use of arccos() in the solar() function) and we would otherwise fail this check in those cases.
    def isnan(self):
        return PhysicsOutputData(
            shortwave_rad=self.shortwave_rad.isnan(),
            longwave_rad=self.longwave_rad.isnan(),
            convection=self.convection.isnan(),
            mod_radcon=self.mod_radcon.isnan(),
            humidity=self.humidity.isnan(),
            condensation=self.condensation.isnan(),
            surface_flux=self.surface_flux.isnan(),
            land_model=self.land_model.isnan()
        )
    
    def any_true(self):
        return tree_util.tree_reduce(lambda x, y: x or y, tree_util.tree_map(lambda x: jnp.any(x), self))
