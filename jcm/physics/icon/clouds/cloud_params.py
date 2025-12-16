from jax import jit
import jax.numpy as jnp

# Constants
tmelt = 273.15  # Melting temperature in Kelvin
grav = 9.81     # Gravitational acceleration (m/s^2)

# Default values for cloud microphysics
cthomi = tmelt - 35.0
cn0s = 3e6
crhoi = 500.0
crhosno = 100.0
ccsaut = 95.0
clmax = 0.5
clmin = 0.0
ceffmax = 150.0  # Max effective radius for ice cloud
lonacc = True

ccsacl = 0.10
ccracl = 6.0
ccraut = 15.0
ceffmin = 10.0  # Min effective radius for ice cloud
ccwmin = 1e-7   # Cloud water limit for cover > 0
cauloc = 0.0
cqtmin = 1e-12  # Total water minimum

# Default values for cloud cover scheme
cptop = 1000.0  # Min pressure level for condensation
cpbot = 50000.0  # Max pressure level for tropopause calculation

# Resolution-dependent parameters (initialized later)
crs = crt = cvtfall = csecfrl = clwprat = csatsc = cinv = None
nex = nadd = None

# Variables initialized in sucloud
ncctop = nccbot = jbmin = jbmax = None

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
