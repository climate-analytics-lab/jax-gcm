"""
Contains the tunable parameters for the cloud microphysics.
Subroutines for intializing these values are also included.
Based on mo_cloud_params from ECHAM6/ICON.
"""

from jax import jit
import jax.numpy as jnp
from typing import NamedTuple
from math import pi

class CloudParams2M(NamedTuple):
    """Cloud parameters for ECHAM6/ICON 2-m microphysical scheme"""

    # Constants
    tmelt: float = 273.15         # Melting point of ice (K)
    grav: float = 9.81            # Gravitational acceleration (m/s²)

    # Default values for cloud microphysics
    cthomi: float = tmelt - 35.0
    cn0s: float = 3e6
    crhoi: float = 500.0
    crhosno: float = 100.0
    ccsaut: float = 95.0
    clmax: float = 0.5
    clmin: float = 0.0
    ceffmax: float = 150.0  # Max effective radius for ice cloud
    lonacc: bool = True

    ccsacl: float = 0.10
    ccracl: float = 6.0
    ccraut: float = 15.0
    ceffmin: float = 10.0  # Min effective radius for ice cloud
    ccwmin: float = 1e-7   # Cloud water limit for cover > 0
    cauloc: float = 0.0
    cqtmin: float = 1e-12  # Total water minimum for cloud to be considered present

    # utility parameters
    epsec: float = 1e-12  # Small number to avoid division by zero
    xsec: float = 1.0 - epsec
    qsec: float  = 1.0 - cqtmin 
    eps: float = jnp.finfo(jnp.float32).eps
    cri: float   = 10e-6    # to estimate the number of produced  
                            # cloud droplets from ice melting in  
                            # case of licnc=.FALSE. [m]=> 10 um
    mi: float = 4.0/3.0*cri**3*pi*crhoi # assumed mass of ice crystals with 
                                 # corresponding volume mean radius cri
    ri_vol_mean_1: float = 2.166e-9 # vol mean ice crystal radius, range border 1
    ri_vol_mean_2: float = 4.264e-8 # vol mean ice crystal radius, range border 2
    alfased_1: float = 63292.4 # for ice crystal fall velocity 
    alfased_2: float = 8.78    # for ice crystal fall velocity
    alfased_3: float = 329.75  # for ice crystal fall velocity
    betased_1: float = 0.5727  # for ice crystal fall velocity 
    betased_2: float = 0.0954  # for ice crystal fall velocity
    betased_3: float = 0.3091 
    
    # Default values for cloud cover scheme
    cptop: float = 1000.0            # Min pressure level for condensation
    cpbot: float = 50000.0           # Max pressure level for tropopause calculation


    # SF #475: bounds / constants for minimum CDNC implied by max droplet size.
    cdnc_min_upper: float = 40.0e6   # [1/m^3]
    cdnc_min_lower: float = 1.0e6    # [1/m^3]
    rcd_vol_max: float = 19.0e-6     # [m] maximum mean-volume droplet radius used for CDNC_min

    # Ice crystal number concentration bounds
    icemin: float = 10.0            # [1/m^3]
    icemax: float = 1.0e7           # [1/m^3]

    # Lognormal droplet spectrum parameters (used for effective radius relations)
    sigmaw: float = 0.28            # [-]
    # `disp = exp(0.5*sigmaw^2)` is derived, but kept as a parameter in cloud_utils.
    # Keeping it here allows exact reproduction if desired.
    disp: float = float(jnp.exp(0.5 * (0.28 ** 2)))  # [-]
    # Reference droplet radius/mass parameters
    dw0: float = 10.0e-6 * float(jnp.exp(0.5 * (0.28 ** 2)))  # [m]
    cdi: float = 0.6              # [-]
    mw0: float = 4.19e-12         # [kg]
    mi0: float = 1.0e-12          # [kg]
    mi0_rcp: float = 1.0e12       # [1/kg]

    # Thermophysical / kinetic constants used in diffusional growth parameterizations
    ka: float = 0.024              # [W/m/K] thermal conductivity of air (approx)
    kb: float = 1.38e-23           # [J/K] Boltzmann constant
    alpha: float = 0.5             # [-] accommodation coefficient
    xmw: float = 2.992e-26         # [kg] mass of an H2O molecule
    fall: float = 3.0              # [-] fall-speed tuning exponent/constant (scheme-specific)

    # Densities / conversion factors
    rhoice: float = 925.0          # [kg/m^3]
    conv_effr2mvr: float = 0.9     # [-] effective radius -> mean volume radius conversion
    clc_min: float = 0.01          # [-] lower limit for cloud fraction in conversions

    # Exponents used by integrated sink forms (e.g., KK2000 style)
    exm1_1: float = 2.47 - 1.0
    exp_1: float = -1.0 / (2.47 - 1.0)
    exm1_2: float = 4.7 - 1.0
    exp_2: float = -1.0 / (4.7 - 1.0)

    # density parameters
    pirho: float = pi * 1000.0  # Assuming rhoh2o = 1000 kg/m^3 (density of water)
    pirho_rcp: float = 1.0 / pirho
    cap: float = 2.0 / pi
    cons4: float = 1.0 / (pi * crhosno * cn0s) ** 0.8125
    cons5: float = 1.0 / (pi * crhosno * cn0s) ** 0.875

    # Snow-related collection / sedimentation tuning
    fact_coll_eff: float = 0.09    # [-] temp-dependent collection efficiency factor
    fact_tke: float = 0.7          # [-] turbulence factor

    # Pruppacher & Klett (1997) ice mass–size relation parameters
    fact_PK: float = 8.253e-3      # [-] (g, cm) parameter; see cloud_utils notes
    pow_PK: float = 2.475          # [-]

    # Cloud scheme logical switches (TODO make them configurable. Currrently hardcoded based in ECHAM6 defaults)
    ldyn_cdnc_min: bool = False    # dynamic min-CDNC switch
    cdnc_min_fixed: float = 10.0   # [cm^-3] fixed value when ldyn_cdnc_min is False
    nic_cirrus: int = 2            # cirrus scheme selector

    # Resolution-dependent parameters (initialized later)
    crs: float = None
    crt: float = None
    cvtfall: float = None
    csecfrl: float = None
    clwprat: float = None
    csatsc: float = None
    cinv: float = None

    nex: float = None
    nadd: float = None

    # Variables initialized in sucloud
    ncctop: float = None
    nccbot: float = None
    jbmin: float = None
    jbmax: float = None

    @classmethod
    def default(cls) -> 'CloudParams2M':
        """Return default cloud parameters for 2-m scheme"""
        return cls()

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

@jit
def sucloud(nlev, vct, nn=None, is_icon=False):
    """
    Defines highest level where condensation is allowed.
    Initializes resolution-dependent parameters.
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

    return {
        "jbmin": jbmin,
        "jbmax": jbmax,
        "ncctop": ncctop,
        "nccbot": nccbot,
        "crs": crs,
        "crt": crt,
        "cvtfall": cvtfall,
        "csecfrl": csecfrl,
        "clwprat": clwprat,
        "csatsc": csatsc,
        "cinv": cinv,
        "nex": nex,
        "nadd": nadd,
    }
