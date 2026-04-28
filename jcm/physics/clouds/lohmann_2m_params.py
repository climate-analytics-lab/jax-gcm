"""Contains the tunable parameters for the cloud microphysics.
Subroutines for intializing these values are also included.
Based on mo_echam_cloud_params from ECHAM6/ICON.
"""

import math

from jax import jit
import jax.numpy as jnp
import numpy as np
from math import pi
import tree_math

@tree_math.struct
class CloudParams2M: #(NamedTuple):
    """Cloud parameters for ECHAM6/ICON 2-m microphysical scheme"""

    # Constants
    tmelt: float       # Melting point of ice (K)
    grav: float            # Gravitational acceleration (m/s²)

    # Default values for cloud microphysics
    cthomi: float
    cn0s: float
    crhoi: float
    crhosno: float
    ccsaut: float
    clmax: float
    clmin: float
    ceffmax: float  # Max effective radius for ice cloud
    lonacc: bool

    ccsacl: float
    ccracl: float
    ccraut: float
    ceffmin: float  # Min effective radius for ice cloud
    ccwmin: float  # Cloud water limit for cover > 0
    cauloc: float
    cqtmin: float  # Total water minimum for cloud to be considered present

    # utility parameters
    epsec: float # Small number to avoid division by zero
    xsec: float
    qsec: float
    eps: float
    cri: float              # to estimate the number of produced  
                            # cloud droplets from ice melting in  
                            # case of licnc=.FALSE. [m]=> 10 um
    mi: float               # assumed mass of ice crystals with 
                            # corresponding volume mean radius cri
    ri_vol_mean_1: float # vol mean ice crystal radius, range border 1
    ri_vol_mean_2: float # vol mean ice crystal radius, range border 2
    alfased_1: float # for ice crystal fall velocity 
    alfased_2: float # for ice crystal fall velocity
    alfased_3: float # for ice crystal fall velocity
    betased_1: float  # for ice crystal fall velocity 
    betased_2: float # for ice crystal fall velocity
    betased_3: float 
    
    # Default values for cloud cover scheme
    cptop: float            # Min pressure level for condensation
    cpbot: float          # Max pressure level for tropopause calculation


    # SF #475: bounds / constants for minimum CDNC implied by max droplet size.
    cdnc_min_upper: float   # [1/m^3]
    cdnc_min_lower: float   # [1/m^3]
    rcd_vol_max: float  # [m] maximum mean-volume droplet radius used for CDNC_min

    # Ice crystal number concentration bounds
    icemin: float          # [1/m^3]
    icemax: float        # [1/m^3]

    # Lognormal droplet spectrum parameters (used for effective radius relations)
    sigmaw: float           # [-]
    # `disp = exp(0.5*sigmaw^2)` is derived, but kept as a parameter in cloud_utils.
    # Keeping it here allows exact reproduction if desired.
    disp: float  # [-]
    # Reference droplet radius/mass parameters
    dw0: float  # [m]
    cdi: float         # [-]
    mw0: float       # [kg]
    mi0: float         # [kg]
    mi0_rcp: float      # [1/kg]

    # Thermophysical / kinetic constants used in diffusional growth parameterizations
    ka: float           # [W/m/K] thermal conductivity of air (approx)
    kb: float         # [J/K] Boltzmann constant
    alpha: float            # [-] accommodation coefficient
    xmw: float       # [kg] mass of an H2O molecule
    fall: float             # [-] fall-speed tuning exponent/constant (scheme-specific)

    # Densities / conversion factors
    rhoice: float        # [kg/m^3]
    conv_effr2mvr: float     # [-] effective radius -> mean volume radius conversion
    clc_min: float          # [-] lower limit for cloud fraction in conversions

    # Exponents used by integrated sink forms (e.g., KK2000 style)
    exm1_1: float
    exp_1: float
    exm1_2: float
    exp_2: float

    # density parameters
    pirho: float  # Assuming rhoh2o = 1000 kg/m^3 (density of water)
    pirho_rcp: float
    cap: float
    cons4: float
    cons5: float

    # Snow-related collection / sedimentation tuning
    fact_coll_eff: float     # [-] temp-dependent collection efficiency factor
    fact_tke: float          # [-] turbulence factor

    # Pruppacher & Klett (1997) ice mass–size relation parameters
    fact_PK: float      # [-] (g, cm) parameter; see cloud_utils notes
    pow_PK: float          # [-]

    # Cloud scheme logical switches (TODO make them configurable. Currrently hardcoded based in ECHAM6 defaults)
    ldyn_cdnc_min: bool   # dynamic min-CDNC switch
    cdnc_min_fixed: float   # [cm^-3] fixed value when ldyn_cdnc_min is False
    nic_cirrus: int            # cirrus scheme selector

    # Resolution-dependent parameters
    # NOTE : The following parameters are normally initialized in the sucloud
    # subroutine based on model resolution. Here, default values for ICON
    # have been used as placeholders. They will be updated by calling
    # sucloud during model initialization.
    crs: float
    crt: float
    cvtfall: float
    csecfrl: float
    clwprat: float
    csatsc: float
    cinv: float

    nex: float
    nadd: float

    # Variables initialized in sucloud
    ncctop: float
    nccbot: float
    jbmin: float
    jbmax: float

    # Sub-timestep for ice sedimentation (s). Capped at the model timestep
    # in Parameters.with_timestep.
    dt_sedi: float

    # Prescribed coarse-mode aerosol number concentration (diameter > 0.5 μm)
    # used by the DeMott (2010) INP parameterization for heterogeneous ice
    # nucleation. Units: cm⁻³ at STP.
    n_aer_coarse: float

    @classmethod
    def default(
        cls,
        tmelt: float = 273.15,
        grav: float = 9.81,
        cthomi: float = 273.15 - 35.0,
        cn0s: float = 3e6,
        crhoi: float = 500.0,
        crhosno: float = 100.0,
        ccsaut: float = 95.0,
        clmax: float = 0.5,
        clmin: float = 0.0,
        ceffmax: float = 150.0,
        lonacc: bool = True,
        ccsacl: float = 0.10,
        ccracl: float = 6.0,
        ccraut: float = 15.0,
        ceffmin: float = 10.0,
        ccwmin: float = 1e-7,
        cauloc: float = 0.0,
        cqtmin: float = 1e-12,
        epsec: float = 1e-12,
        cri: float = 10e-6,
        ri_vol_mean_1: float = 2.166e-9,
        ri_vol_mean_2: float = 4.264e-8,
        alfased_1: float = 63292.4,
        alfased_2: float = 8.78,
        alfased_3: float = 329.75,
        betased_1: float = 0.5727,
        betased_2: float = 0.0954,
        betased_3: float = 0.3091,
        cptop: float = 1000.0,
        cpbot: float = 50000.0,
        cdnc_min_upper: float = 40.0e6,
        cdnc_min_lower: float = 1.0e6,
        rcd_vol_max: float = 19.0e-6,
        icemin: float = 10.0,
        icemax: float = 1.0e7,
        sigmaw: float = 0.28,
        cdi: float = 0.6,
        mw0: float = 4.19e-12,
        mi0: float = 1.0e-12,
        mi0_rcp: float = 1.0e12,
        ka: float = 0.024,
        kb: float = 1.38e-23,
        alpha: float = 0.5,
        xmw: float = 2.992e-26,
        fall: float = 3.0,
        rhoice: float = 925.0,
        conv_effr2mvr: float = 0.9,
        clc_min: float = 0.01,
        exm1_1: float = 2.47 - 1.0,
        exp_1: float = -1.0 / (2.47 - 1.0),
        exm1_2: float = 4.7 - 1.0,
        exp_2: float = -1.0 / (4.7 - 1.0),
        fact_coll_eff: float = 0.09,
        fact_tke: float = 0.7,
        fact_PK: float = 8.253e-3,
        pow_PK: float = 2.475,
        ldyn_cdnc_min: bool = False,
        cdnc_min_fixed: float = 10.0,
        nic_cirrus: int = 2,
        # resolution-dependent placeholders (ICON defaults)
        crs: float = 0.975,
        crt: float = 0.75,
        cvtfall: float = 2.5,
        csecfrl: float = 5e-6,
        clwprat: float = 4.0,
        csatsc: float = 0.7,
        cinv: float = 0.25,
        nex: float = 2,
        nadd: float = 0,
        ncctop: float = 13,
        nccbot: float = 35,
        jbmin: float = 40,
        jbmax: float = 45,
        dt_sedi: float = 10.0,
        n_aer_coarse: float = 0.5,
    ) -> 'CloudParams2M':
        """Return default cloud parameters for 2-m scheme"""
        # derived helpers — use Python/numpy math here so `default()` stays
        # safe to call under any tracing context (import-time module-level
        # initialization would otherwise trip ConcretizationTypeError when
        # called while a caller is mid-trace)
        disp = math.exp(0.5 * (sigmaw ** 2))
        dw0 = 10e-6 * disp
        xsec = 1.0 - epsec
        qsec = 1.0 - cqtmin
        eps_val = float(np.finfo(np.float32).eps)
        mi_val = 4.0 / 3.0 * cri ** 3 * pi * crhoi
        pirho_val = pi * 1000.0
        pirho_rcp_val = 1.0 / pirho_val
        cap_val = 2.0 / pi
        cons4_val = 1.0 / (pi * crhosno * cn0s) ** 0.8125
        cons5_val = 1.0 / (pi * crhosno * cn0s) ** 0.875

        return cls(
            tmelt=jnp.array(tmelt),
            grav=jnp.array(grav),
            cthomi=jnp.array(cthomi),
            cn0s=jnp.array(cn0s),
            crhoi=jnp.array(crhoi),
            crhosno=jnp.array(crhosno),
            ccsaut=jnp.array(ccsaut),
            clmax=jnp.array(clmax),
            clmin=jnp.array(clmin),
            ceffmax=jnp.array(ceffmax),
            lonacc=jnp.array(lonacc),
            ccsacl=jnp.array(ccsacl),
            ccracl=jnp.array(ccracl),
            ccraut=jnp.array(ccraut),
            ceffmin=jnp.array(ceffmin),
            ccwmin=jnp.array(ccwmin),
            cauloc=jnp.array(cauloc),
            cqtmin=jnp.array(cqtmin),
            epsec=jnp.array(epsec),
            xsec=jnp.array(xsec),
            qsec=jnp.array(qsec),
            eps=jnp.array(eps_val),
            cri=jnp.array(cri),
            mi=jnp.array(mi_val),
            ri_vol_mean_1=jnp.array(ri_vol_mean_1),
            ri_vol_mean_2=jnp.array(ri_vol_mean_2),
            alfased_1=jnp.array(alfased_1),
            alfased_2=jnp.array(alfased_2),
            alfased_3=jnp.array(alfased_3),
            betased_1=jnp.array(betased_1),
            betased_2=jnp.array(betased_2),
            betased_3=jnp.array(betased_3),
            cptop=jnp.array(cptop),
            cpbot=jnp.array(cpbot),
            cdnc_min_upper=jnp.array(cdnc_min_upper),
            cdnc_min_lower=jnp.array(cdnc_min_lower),
            rcd_vol_max=jnp.array(rcd_vol_max),
            icemin=jnp.array(icemin),
            icemax=jnp.array(icemax),
            sigmaw=jnp.array(sigmaw),
            disp=jnp.array(disp),
            dw0=jnp.array(dw0),
            cdi=jnp.array(cdi),
            mw0=jnp.array(mw0),
            mi0=jnp.array(mi0),
            mi0_rcp=jnp.array(mi0_rcp),
            ka=jnp.array(ka),
            kb=jnp.array(kb),
            alpha=jnp.array(alpha),
            xmw=jnp.array(xmw),
            fall=jnp.array(fall),
            rhoice=jnp.array(rhoice),
            conv_effr2mvr=jnp.array(conv_effr2mvr),
            clc_min=jnp.array(clc_min),
            exm1_1=jnp.array(exm1_1),
            exp_1=jnp.array(exp_1),
            exm1_2=jnp.array(exm1_2),
            exp_2=jnp.array(exp_2),
            pirho=jnp.array(pirho_val),
            pirho_rcp=jnp.array(pirho_rcp_val),
            cap=jnp.array(cap_val),
            cons4=jnp.array(cons4_val),
            cons5=jnp.array(cons5_val),
            fact_coll_eff=jnp.array(fact_coll_eff),
            fact_tke=jnp.array(fact_tke),
            fact_PK=jnp.array(fact_PK),
            pow_PK=jnp.array(pow_PK),
            ldyn_cdnc_min=jnp.array(ldyn_cdnc_min),
            cdnc_min_fixed=jnp.array(cdnc_min_fixed),
            nic_cirrus=jnp.array(nic_cirrus),
            crs=jnp.array(crs),
            crt=jnp.array(crt),
            cvtfall=jnp.array(cvtfall),
            csecfrl=jnp.array(csecfrl),
            clwprat=jnp.array(clwprat),
            csatsc=jnp.array(csatsc),
            cinv=jnp.array(cinv),
            nex=jnp.array(nex),
            nadd=jnp.array(nadd),
            ncctop=jnp.array(ncctop),
            nccbot=jnp.array(nccbot),
            jbmin=jnp.array(jbmin),
            jbmax=jnp.array(jbmax),
            dt_sedi=jnp.array(dt_sedi),
            n_aer_coarse=jnp.array(n_aer_coarse),
        )

@jit
def sucloud(nlev, vct, nn=None, is_icon=False):
    """Define highest level where condensation is allowed.
    Initializes resolution-dependent parameters.
    # TODO: allow the CloudParams2M to be updated with resolution-dependant sucloud outputs.
    # For now the default values for ICON have been used as placeholders (see note above in CloudParams2M).

    Args:
        nlev: Number of vertical levels
        vct: Vertical coordinate transformation coefficients
        nn: Truncation (optional, required if is_icon is False)
        is_icon: Whether the model is ICON (True) or ECHAM (False)

    Returns:
        Updated cloud parameters (jbmin, jbmax, ncctop, nccbot, crs, crt, cvtfall, csecfrl,
        clwprat, csatsc, cinv, nex, nadd)

    """
    global crs, crt, cvtfall, csecfrl, clwprat, csatsc, cinv, nex, nadd
    global ncctop, nccbot, jbmin, jbmax

    if is_icon:
        # ICON-specific values
        jbmin, jbmax, ncctop, nccbot = 40, 45, 13, 35
        crs, crt, cvtfall, csecfrl, clwprat, csatsc, cinv = 0.975, 0.75, 2.5, 5e-6, 4.0, 0.7, 0.25
        nex, nadd = 2, 0
    else:
        # ECHAM-specific calculations
        za = vct[:nlev + 1]
        zb = vct[nlev + 1:]
        zph = za + zb * 101320.0

        zp = (zph[:-1] + zph[1:]) * 0.5
        zh = (zph[-1] - zp) / (grav * 1.25)

        # Highest inversion level (first full level below 2000 m)
        jbmin = jnp.argmax(zh < 2000.0)

        # Lowest inversion level (first full level below 500 m)
        jbmax = jnp.argmax(zh < 500.0)

        # Pressure level cptop (Pa)
        ncctop = jnp.argmax(zp >= cptop)

        # Pressure level cpbot (Pa)
        nccbot = jnp.argmax(zp >= cpbot)

        # Resolution-dependent parameters
        if nn == 31:
            crs, crt, cvtfall, csecfrl, clwprat, csatsc, cinv = 0.95, 0.85, 3.0, 5e-7, 0.0, 0.1, 0.5
            nex, nadd = 1, 1
        elif nn == 63:
            crs, crt, cvtfall, csecfrl, clwprat, csatsc, cinv = 0.975, 0.75, 2.5, 5e-6, 4.0, 0.7, 0.25
            nex, nadd = 2, 0
        elif nn == 127:
            crs, crt, cvtfall, csecfrl, clwprat, csatsc, cinv = 0.994, 0.75, 3.0, 1e-5, 4.0, 0.7, 0.25
            nex, nadd = 2, 0
        elif nn == 255:
            crs, crt, cvtfall, csecfrl, clwprat, csatsc, cinv = 0.994, 0.75, 3.0, 1e-5, 4.0, 0.7, 0.25
            nex, nadd = 2, 0
        else:
            raise ValueError("Truncation not supported.")

    # return {
    #     "jbmin": jbmin,
    #     "jbmax": jbmax,
    #     "ncctop": ncctop,
    #     "nccbot": nccbot,
    #     "crs": crs,
    #     "crt": crt,
    #     "cvtfall": cvtfall,
    #     "csecfrl": csecfrl,
    #     "clwprat": clwprat,
    #     "csatsc": csatsc,
    #     "cinv": cinv,
    #     "nex": nex,
    #     "nadd": nadd,
    # }
    pass

# Global instance of physical constants
cloud_params = CloudParams2M.default()

# Export individual constants for convenience
tmelt = cloud_params.tmelt
grav = cloud_params.grav
cthomi = cloud_params.cthomi
cn0s = cloud_params.cn0s
crhoi = cloud_params.crhoi
crhosno = cloud_params.crhosno
ccsaut = cloud_params.ccsaut
clmax = cloud_params.clmax
clmin = cloud_params.clmin
ceffmax = cloud_params.ceffmax
lonacc = cloud_params.lonacc
ccsacl = cloud_params.ccsacl
ccracl = cloud_params.ccracl
ccraut = cloud_params.ccraut
ceffmin = cloud_params.ceffmin
ccwmin = cloud_params.ccwmin
cauloc = cloud_params.cauloc
cqtmin = cloud_params.cqtmin
epsec = cloud_params.epsec
xsec = cloud_params.xsec
qsec = cloud_params.qsec
eps = cloud_params.eps
cri = cloud_params.cri
mi = cloud_params.mi
ri_vol_mean_1 = cloud_params.ri_vol_mean_1
ri_vol_mean_2 = cloud_params.ri_vol_mean_2
alfased_1 = cloud_params.alfased_1
alfased_2 = cloud_params.alfased_2
alfased_3 = cloud_params.alfased_3
betased_1 = cloud_params.betased_1
betased_2 = cloud_params.betased_2
betased_3 = cloud_params.betased_3
cptop = cloud_params.cptop
cpbot = cloud_params.cpbot
cdnc_min_upper = cloud_params.cdnc_min_upper
cdnc_min_lower = cloud_params.cdnc_min_lower
rcd_vol_max = cloud_params.rcd_vol_max
icemin = cloud_params.icemin
icemax = cloud_params.icemax
sigmaw = cloud_params.sigmaw
disp = cloud_params.disp
dw0 = cloud_params.dw0
cdi = cloud_params.cdi
mw0 = cloud_params.mw0
mi0 = cloud_params.mi0
mi0_rcp = cloud_params.mi0_rcp
ka = cloud_params.ka
kb = cloud_params.kb
alpha = cloud_params.alpha
xmw = cloud_params.xmw
fall = cloud_params.fall
rhoice = cloud_params.rhoice
conv_effr2mvr = cloud_params.conv_effr2mvr
clc_min = cloud_params.clc_min
exm1_1 = cloud_params.exm1_1
exp_1 = cloud_params.exp_1
exm1_2 = cloud_params.exm1_2
exp_2 = cloud_params.exp_2
fact_coll_eff = cloud_params.fact_coll_eff
fact_tke = cloud_params.fact_tke
fact_PK = cloud_params.fact_PK
pow_PK = cloud_params.pow_PK
ldyn_cdnc_min = cloud_params.ldyn_cdnc_min
cdnc_min_fixed = cloud_params.cdnc_min_fixed
nic_cirrus = cloud_params.nic_cirrus
crs = cloud_params.crs
crt = cloud_params.crt
cvtfall = cloud_params.cvtfall
csecfrl = cloud_params.csecfrl
clwprat = cloud_params.clwprat
csatsc = cloud_params.csatsc
cinv = cloud_params.cinv
nex = cloud_params.nex
nadd = cloud_params.nadd
ncctop = cloud_params.ncctop
nccbot = cloud_params.nccbot
jbmin = cloud_params.jbmin
jbmax = cloud_params.jbmax
pirho = cloud_params.pirho
pirho_rcp = cloud_params.pirho_rcp
cap = cloud_params.cap
cons4 = cloud_params.cons4
cons5 = cloud_params.cons5