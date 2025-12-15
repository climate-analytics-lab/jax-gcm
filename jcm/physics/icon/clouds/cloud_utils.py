"""
Contains utility routines and constants related to the 2-m cloud microphysics scheme. Based on mo_cloud_utils from ECHAM6/ICON.

Date: 2025-12-15
"""

import jax.numpy as jnp
import jax
from jax import lax
from math import pi

from ..constants.physical_constants import (
    grav, rhoh2o
)
from cloud_params import (
    cqtmin, crhoi, cn0s, crhosno
)

# Microphysical constants
epsec = 1.0e-12
xsec = 1.0 - epsec
# cqtmin is undefined in the provided code, so it needs to be defined elsewhere
# Assuming cqtmin = 0 for now
#cqtmin = 0.0
qsec = 1.0 - cqtmin
eps = jnp.finfo(jnp.float64).eps
cri = 10.0e-6  # [m] => 10 um
pi = jnp.pi
crhoi = 1.0  # Placeholder for ice crystal density, define appropriately
mi = 4.0 / 3.0 * cri**3 * pi * crhoi
ri_vol_mean_1 = 2.166e-9
ri_vol_mean_2 = 4.264e-8
alfased_1 = 63292.4
alfased_2 = 8.78
alfased_3 = 329.75
betased_1 = 0.5727
betased_2 = 0.0954
betased_3 = 0.3091

# SF #475
cdnc_min_upper = 40.0e6
cdnc_min_lower = 1.0e6
rcd_vol_max = 19.0e-6
# SF The above has been updated on 2016.04.04 (David Neubauer, pure atm run, HAM-M7, AR&G activation)

icemin = 10.0
icemax = 1.0e7
sigmaw = 0.28
disp = jnp.exp(0.5 * sigmaw**2)
dw0 = 10.0e-6 * disp
cdi = 0.6
mw0 = 4.19e-12
mi0 = 1.0e-12
mi0_rcp = 1.0e12
ka = 0.024
kb = 1.38e-23
alpha = 0.5
xmw = 2.992e-26  # Mass of an H2O molecule [kg]
fall = 3.0
rhoice = 925.0
conv_effr2mvr = 0.9
clc_min = 0.01
exm1_1 = 2.47 - 1.0
exp_1 = -1.0 / exm1_1
exm1_2 = 4.7 - 1.0
exp_2 = -1.0 / exm1_2
rhoh2o = 1.0  # Placeholder for water density, define appropriately
pirho = pi * rhoh2o
pirho_rcp = 1.0 / pirho
cap = 2.0 / pi
crhosno = 1.0  # Placeholder for snow density, define appropriately
cn0s = 1.0  # Placeholder for some constant, define appropriately
cons4 = 1.0 / (pi * crhosno * cn0s)**0.8125
cons5 = 1.0 / (pi * crhosno * cn0s)**0.875
fact_coll_eff = 0.09  # Factor for temperature-dependent collection efficiency of snow by cold hydrometeors
fact_tke = 0.7

# SF #176 Constants for ice crystal mass to effective radius relationship, from Pruppacher & Klett 1997
# m = zfact_PK x (2r_i)**zpow_PK (m in g; r_i in cm)

# Plate P1a, from table 2.2a (size range: 10-3000 micrometers)
fact_PK = 8.253e-3  # 9.17 x 0.9, because m = rho_c x V_c, where rho_c = 0.9 g.cm^-3
pow_PK = 2.475

def get_util_var(kproma, kbdim, ktdia, klev, klevp1, paphm1, pgeo, papm1, ptm1):
    """
    Get several utility variables:
        - Geopotential at half levels (pgeoh)
        - Pressure- and height-differences (pdp, pdz)
        - Air density correction for computing ice crystal fall velocity (paaa)
        - Dynamic viscosity of air (pviscos)
    """
    grav = 9.81  # Gravitational acceleration [m/s^2]
    g_rcp = 1.0 / grav

    # Initialize output arrays
    pgeoh = jnp.zeros((kbdim, klevp1))
    pdp = jnp.zeros((kbdim, klev))
    pdpg = jnp.zeros((kbdim, klev))
    pdz = jnp.zeros((kbdim, klev))
    paaa = jnp.zeros((kbdim, klev))
    pviscos = jnp.zeros((kbdim, klev))

    # Geopotential at half levels
    pgeoh = pgeoh.at[:, ktdia+1:klev].set(
        0.5 * (pgeo[:, ktdia+1:klev] + pgeo[:, ktdia:klev-1])
    )
    pgeoh = pgeoh.at[:, ktdia].set(
        pgeo[:, ktdia] + (pgeo[:, ktdia] - pgeoh[:, ktdia+1])
    )
    pgeoh = pgeoh.at[:, klevp1-1].set(0.0)

    # Pressure differences
    pdp = pdp.at[:, ktdia:klev].set(
        paphm1[:, ktdia+1:klevp1] - paphm1[:, ktdia:klev]
    )
    pdpg = pdpg.at[:, ktdia:klev].set(
        g_rcp * pdp[:, ktdia:klev]
    )

    # Height differences
    pdz = pdz.at[:, ktdia:klev].set(
        g_rcp * (pgeoh[:, ktdia:klev] - pgeoh[:, ktdia+1:klevp1])
    )

    # Air density correction
    paaa = paaa.at[:, :].set(
        (papm1[:, :] / 30000.0)**(-0.178) * (ptm1[:, :] / 233.0)**(-0.394)
    )

    # Dynamic viscosity of air
    pviscos = pviscos.at[:, :].set(
        (1.512 + 0.0052 * (ptm1[:, :] - 233.15)) * 1.0e-5
    )

    return pgeoh, pdp, pdpg, pdz, paaa, pviscos

def get_cloud_bounds(kproma, kbdim, ktdia, klev, paclc):
    """
    Flags the top, base, and intermediate levels for each cloud.

    Parameters:
        kproma (int): Number of columns.
        kbdim (int): Number of rows.
        ktdia (int): Starting level index.
        klev (int): Number of levels.
        paclc (jax.numpy.ndarray): Cloud cover array of shape (kbdim, klev).

    Returns:
        ktop (jax.numpy.ndarray): Flag for cloud tops of shape (kbdim, klev).
        kbas (jax.numpy.ndarray): Flag for cloud bases of shape (kbdim, klev).
        kcl_minustop (jax.numpy.ndarray): Flag for all cloud levels except their top.
        kcl_minusbas (jax.numpy.ndarray): Flag for all cloud levels except their base.
    """
    epsec = 1.0e-12

    # Initialize output arrays
    ktop = jnp.zeros((kbdim, klev), dtype=jnp.int32)
    kbas = jnp.zeros((kbdim, klev), dtype=jnp.int32)
    kcl_minustop = jnp.zeros((kbdim, klev), dtype=jnp.int32)
    kcl_minusbas = jnp.zeros((kbdim, klev), dtype=jnp.int32)

    # Duplicate paclc at level-1 and level+1
    zaclcm = jnp.zeros((kbdim, klev))
    zaclcp = jnp.zeros((kbdim, klev))
    zaclcm = zaclcm.at[:, ktdia + 1 : klev].set(paclc[:, ktdia : klev - 1])
    zaclcp = zaclcp.at[:, ktdia : klev - 1].set(paclc[:, ktdia + 1 : klev])

    # Set logical switches
    ll = paclc >= epsec
    llm = zaclcm < epsec
    llp = zaclcp < epsec

    lltop = ll & llm  # True if cloud top detected
    llbas = ll & llp  # True if cloud base detected

    # Set ktop and kbas
    iindex = jnp.tile(jnp.arange(klev), (kbdim, 1))
    ktop = jnp.where(lltop, iindex, 0)
    kbas = jnp.where(llbas, iindex, 0)

    # Count the number of clouds per column
    iclnb = jnp.sum(lltop, axis=1)

    # Process each column
    def process_column(jl, kcl_minustop, kcl_minusbas):
        jnumb = iclnb[jl]
        iclbounds = jnp.zeros((2, klev // 2 + 1), dtype=jnp.int32)

        # Set the bounds in a compact array
        iclbounds = iclbounds.at[0, :jnumb].set(jnp.where(lltop[jl, :], iindex[jl, :], 0))
        iclbounds = iclbounds.at[1, :jnumb].set(jnp.where(llbas[jl, :], iindex[jl, :], 0))

        # Flag cloud levels except their base (or top)
        def update_bounds(jm, kcl_minustop, kcl_minusbas):
            jtop = iclbounds[0, jm]
            jbas = iclbounds[1, jm]
            kcl_minusbas = kcl_minusbas.at[jl, jtop:jbas].set(jbas)
            kcl_minustop = kcl_minustop.at[jl, jtop + 1 : jbas + 1].set(jtop)
            return kcl_minustop, kcl_minusbas

        kcl_minustop, kcl_minusbas = jax.lax.fori_loop(
            0, jnumb, update_bounds, (kcl_minustop, kcl_minusbas)
        )
        return kcl_minustop, kcl_minusbas

    kcl_minustop, kcl_minusbas = jax.lax.fori_loop(
        0, kproma, process_column, (kcl_minustop, kcl_minusbas)
    )

    return ktop, kbas, kcl_minustop, kcl_minusbas

def init_cloud_micro_2m(lconv):
    """
    Initializes boundary conditions for the cloud microphysics scheme.

    Parameters:
        lconv (bool): Logical flag indicating whether convection is enabled.

    Returns:
        dict: A dictionary containing boundary condition definitions.
    """
    # Define boundary condition structure
    bc_cvcbot = {"ef_type": None, "description": None, "dim": None, "active": None}
    bc_wcape = {"ef_type": None, "description": None, "dim": None, "active": None}
    bc_tconv = {"ef_type": None, "description": None, "dim": None, "active": None}
    bc_detr_cond = {"ef_type": None, "description": None, "dim": None, "active": None}

    # Initialize boundary conditions if convection is enabled
    if lconv:
        # Convective cloud base index
        bc_cvcbot["ef_type"] = "EF_MODULE"
        bc_cvcbot["description"] = "Convective cloud base index"
        bc_cvcbot["dim"] = 2
        bc_cvcbot["active"] = True

        # CAPE contribution to convective vertical velocity
        bc_wcape["ef_type"] = "EF_MODULE"
        bc_wcape["description"] = "CAPE contrib. to conv. vertical velocity"
        bc_wcape["dim"] = 2
        bc_wcape["active"] = True

        # Temperature in convective scheme
        bc_tconv["ef_type"] = "EF_MODULE"
        bc_tconv["description"] = "Temperature in convective scheme"
        bc_tconv["dim"] = 3
        bc_tconv["active"] = True

        # Detrained condensate
        bc_detr_cond["ef_type"] = "EF_MODULE"
        bc_detr_cond["description"] = "Detrained condensate"
        bc_detr_cond["dim"] = 3
        bc_detr_cond["active"] = True

    # Return the boundary condition definitions
    return {
        "bc_cvcbot": bc_cvcbot,
        "bc_wcape": bc_wcape,
        "bc_tconv": bc_tconv,
        "bc_detr_cond": bc_detr_cond,
    }