"""
Contains utility routines and constants related to the 2-m cloud microphysics scheme. Based on mo_cloud_utils from ECHAM6/ICON.

Date: 2025-12-15
"""

import jax.numpy as jnp
import jax
from jax import lax
from math import pi

from ..constants.physical_constants import (
    grav, rgrav, rhoh2o
)
from .cloud_params import (
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

def get_util_var(nproma, nbdim, ntdia, nlev, nlevp1, paphm1, pgeo, papm1, ptm1):
    """
    Get several utility variables:
        - Geopotential at half levels (pgeoh)
        - Pressure- and height-differences (pdp, pdz)
        - Air density correction for computing ice crystal fall velocity (paaa)
        - Dynamic viscosity of air (pviscos)

    Assumes that the highest level is a the surface level.
    """

    # Initialize output arrays
    pgeoh = jnp.zeros((nbdim, nlevp1))
    pdp = jnp.zeros((nbdim, nlev))
    pdpg = jnp.zeros((nbdim, nlev))
    pdz = jnp.zeros((nbdim, nlev))
    paaa = jnp.zeros((nbdim, nlev))
    pviscos = jnp.zeros((nbdim, nlev))

    # Geopotential at half levels
    pgeoh = pgeoh.at[:, ntdia+1:nlev].set(
        0.5 * (pgeo[:, ntdia+1:nlev] + pgeo[:, ntdia:nlev-1])
    )
    pgeoh = pgeoh.at[:, ntdia].set(
        pgeo[:, ntdia] + (pgeo[:, ntdia] - pgeoh[:, ntdia+1])
    )
    pgeoh = pgeoh.at[:, nlevp1-1].set(0.0) # highest half-level geopotential set to zero

    # Pressure differences
    pdp = pdp.at[:, ntdia:nlev].set(                       # absolute pressure difference
        paphm1[:, ntdia+1:nlevp1] - paphm1[:, ntdia:nlev]
    )
    pdpg = pdpg.at[:, ntdia:nlev].set(                     # pressure gradient force term
        rgrav * pdp[:, ntdia:nlev]
    )

    # Height differences
    pdz = pdz.at[:, ntdia:nlevp1].set(
        rgrav * (pgeoh[:, ntdia:nlev] - pgeoh[:, ntdia+1:nlevp1])
    )
    # Might change it to this to keep it consistent with pressure differ
    # pdz = pdz.at[:, ntdia:nlevp1].set(
    #     rgrav * (pgeoh[:, ntdia+1:nlevp1] - pgeoh[:, ntdia:nlev])
    # )

    # Air density correction
    paaa = paaa.at[:, :].set(
        (papm1[:, :] / 30000.0)**(-0.178) * (ptm1[:, :] / 233.0)**(-0.394)
    )

    # Dynamic viscosity of air
    pviscos = pviscos.at[:, :].set(
        (1.512 + 0.0052 * (ptm1[:, :] - 233.15)) * 1.0e-5
    )

    return pgeoh, pdp, pdpg, pdz, paaa, pviscos

def get_cloud_bounds(nproma, nbdim, ntdia, nlev, paclc):
    """
    Flags the top, base, and intermediate levels for each cloud.

    Assumes that the highest level is a the surface level.

    """
    # Initialize output arrays
    ktop = jnp.zeros((nbdim, nlev), dtype=jnp.int32)
    kbas = jnp.zeros((nbdim, nlev), dtype=jnp.int32)
    kcl_minustop = jnp.zeros((nbdim, nlev), dtype=jnp.int32)
    kcl_minusbas = jnp.zeros((nbdim, nlev), dtype=jnp.int32)

    # Duplicate paclc at level-1 and level+1
    zaclcm = jnp.zeros((nbdim, nlev))
    zaclcp = jnp.zeros((nbdim, nlev))
    zaclcm = zaclcm.at[:, ntdia + 1 : nlev].set(paclc[:, ntdia : nlev - 1])
    zaclcp = zaclcp.at[:, ntdia : nlev - 1].set(paclc[:, ntdia + 1 : nlev])

    # Set logical switches
    ll = paclc >= epsec
    llm = zaclcm < epsec
    llp = zaclcp < epsec

    lltop = ll & llm
    llbas = ll & llp

    # Set ktop and kbas (index-marked masks)
    iindex = jnp.tile(jnp.arange(nlev, dtype=jnp.int32), (nbdim, 1))
    ktop = jnp.where(lltop, iindex, 0)
    kbas = jnp.where(llbas, iindex, 0)

    def process_column(jl, carry_state):
        kcl_minustop, kcl_minusbas = carry_state

        # per-level event flags for this column
        is_top = lltop[jl, :]  # (nlev,) bool
        is_bas = llbas[jl, :]  # (nlev,) bool

        # Record up to nlev pairs in fixed-size arrays; unused slots stay -1.
        tops_out0 = -jnp.ones((nlev,), dtype=jnp.int32)
        bas_out0 = -jnp.ones((nlev,), dtype=jnp.int32)

        # scan state:
        # open_top: int32, -1 means "no open cloud"
        # pair_count: number of emitted pairs so far (int32)
        # tops_out, bas_out: (nlev,) arrays
        def scan_step(state, k):
            open_top, pair_count, tops_out, bas_out = state
            k = k.astype(jnp.int32)

            top_here = is_top[k]
            bas_here = is_bas[k]

            # If a top and no cloud is open, open one at k.
            open_top = jnp.where(top_here & (open_top < 0), k, open_top)

            # If a base and a cloud is open, emit a pair (open_top, k) and close.
            emit = bas_here & (open_top >= 0)

            tops_out = jax.lax.cond(
                emit,
                lambda arr: arr.at[pair_count].set(open_top),
                lambda arr: arr,
                tops_out,
            )
            bas_out = jax.lax.cond(
                emit,
                lambda arr: arr.at[pair_count].set(k),
                lambda arr: arr,
                bas_out,
            )

            pair_count = pair_count + emit.astype(jnp.int32)
            open_top = jnp.where(emit, -jnp.int32(1), open_top)

            return (open_top, pair_count, tops_out, bas_out), None

        init_state = (-jnp.int32(1), jnp.int32(0), tops_out0, bas_out0)
        (open_top, npairs, tops_list, bas_list), _ = jax.lax.scan(
            scan_step, init_state, jnp.arange(nlev)
        )

        # Apply each (top, base) pair to fill kcl_* rows using masks.
        def apply_pair(i, state):
            kcl_minustop, kcl_minusbas = state
            jtop = tops_list[i]
            jbas = bas_list[i]

            valid = (i < npairs) & (jtop >= 0) & (jbas >= 0) & (jtop < jbas)

            def do_update(st):
                kcl_minustop, kcl_minusbas = st

                idx = jnp.arange(nlev, dtype=jnp.int32)
                in_minusbas = (idx >= jtop) & (idx < jbas)   # [top, base)
                in_minustop = (idx > jtop) & (idx <= jbas)   # (top, base]

                row_minusbas = kcl_minusbas[jl, :]
                row_minustop = kcl_minustop[jl, :]

                row_minusbas = jnp.where(in_minusbas, jbas, row_minusbas)
                row_minustop = jnp.where(in_minustop, jtop, row_minustop)

                kcl_minusbas = kcl_minusbas.at[jl, :].set(row_minusbas)
                kcl_minustop = kcl_minustop.at[jl, :].set(row_minustop)
                return kcl_minustop, kcl_minusbas

            return jax.lax.cond(valid, do_update, lambda st: st, (kcl_minustop, kcl_minusbas))

        kcl_minustop, kcl_minusbas = jax.lax.fori_loop(
            0, nlev, apply_pair, (kcl_minustop, kcl_minusbas)
        )

        return kcl_minustop, kcl_minusbas

    kcl_minustop, kcl_minusbas = jax.lax.fori_loop(
        0, nproma, process_column, (kcl_minustop, kcl_minusbas)
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