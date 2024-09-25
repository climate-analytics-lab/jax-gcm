import jax.numpy as jnp
import tree_math

@tree_math.struct
class SWRadiationData:
    qcloud: jnp.ndarray
    fsol: jnp.ndarray
    ozone: jnp.ndarray
    ozupp: jnp.ndarray
    zenit: jnp.ndarray
    stratz: jnp.ndarray
    gse: jnp.ndarray
    icltop: jnp.ndarray
    cloudc: jnp.ndarray
    cloudstr: jnp.ndarray

    def __init__(self, nodal_shape, qcloud=None, fsol=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None) -> None:
        if qcloud is not None:
            self.qcloud = qcloud
        else:
            self.qcloud = jnp.zeros((nodal_shape))
        if fsol is not None:
            self.fsol = fsol
        else:
            self.fsol = jnp.zeros((nodal_shape))
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


    def copy(self, qcloud=None, fsol=None, ozone=None, ozupp=None, zenit=None, stratz=None, gse=None, icltop=None, cloudc=None, cloudstr=None):
        return SWRadiationData(
            self.cloudc.shape,
            qcloud=qcloud if qcloud is not None else self.qcloud,
            fsol=fsol if fsol is not None else self.fsol,
            ozone=ozone if ozone is not None else self.ozone,
            ozupp=ozupp if ozupp is not None else self.ozupp,
            zenit=zenit if zenit is not None else self.zenit,
            stratz=stratz if stratz is not None else self.stratz,
            gse=gse if gse is not None else self.gse,
            icltop=icltop if icltop is not None else self.icltop,
            cloudc=cloudc if cloudc is not None else self.cloudc,
            cloudstr=cloudstr if cloudstr is not None else self.cloudstr
        )
    
#FIXME: fband should be moved here from mod_radcon.py? since this is where it is used and set?
@tree_math.struct
class LWRadiationData:
    fsfcd: jnp.ndarray
    dfabs: jnp.ndarray
    ftop: jnp.ndarray
    fsfc: jnp.ndarray
    
    def __init__(self, nodal_shape, node_levels, fsfcd=None, dfabs=None, ftop=None, fsfc=None) -> None:
        if fsfcd is not None:
            self.fsfcd = fsfcd
        else:
            self.fsfcd = jnp.zeros((nodal_shape))
        if dfabs is not None:
            self.dfabs = dfabs
        else:
            self.dfabs = jnp.zeros((nodal_shape + (node_levels,)))
        if ftop is not None:
            self.ftop = ftop
        else:
            self.ftop = jnp.zeros((nodal_shape))
        if fsfc is not None:
            self.fsfc = fsfc
        else:
            self.fsfc = jnp.zeros((nodal_shape))

    def copy(self, fsfcd=None, dfabs=None, ftop=None, fsfc=None):
        return LWRadiationData(
            (0,0), 
            0, 
            fsfcd=fsfcd if fsfcd is not None else self.fsfcd,
            dfabs=dfabs if dfabs is not None else self.dfabs,
            ftop=ftop if ftop is not None else self.ftop,
            fsfc=fsfc if fsfc is not None else self.fsfc
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


    def __init__(self, nodal_shape, node_levels, fband=None,albl=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None) -> None:
        if fband is not None:
            self.fband = fband
        else:
            self.fband = jnp.zeros((301,4))
        if albl is not None:
            self.albl = albl
        else:
            self.albl = jnp.zeros((nodal_shape))
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

    def copy(self,fband=None,albl=None,alb_s=None,albsfc=None,snowc=None,tau2=None,st4a=None,stratc=None,flux=None):
        return ModRadConData(
            self.albl.shape, 
            0, # this value isn't necessary since we will pass through values for each member of the struct
            fband=fband if fband is not None else self.fband,
            albl=albl if albl is not None else self.albl,
            alb_s=alb_s if alb_s is not None else self.alb_s,
            albsfc=albsfc if albsfc is not None else self.albsfc,
            snowc=snowc if snowc is not None else self.snowc,
            tau2=tau2 if tau2 is not None else self.tau2,
            st4a=st4a if st4a is not None else self.st4a,
            stratc=stratc if stratc is not None else self.stratc,
            flux=flux if flux is not None else self.flux
        )

@tree_math.struct
class CondensationData:
    precls: jnp.ndarray
    dtlsc: jnp.ndarray
    dqlsc: jnp.ndarray
    
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
    dfqa: jnp.ndarray #Net flux of specific humidity into each atmospheric layer

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

