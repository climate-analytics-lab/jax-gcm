"""Contains utility routines and constants related to the 2-m cloud microphysics scheme. Based on mo_cloud_utils from ECHAM6/ICON.

Date: 2025-12-15
"""

import jax.numpy as jnp
import jax
from jax import jit
from math import pi

import jcm.constants as c
from .lohmann_2m_params import (
    epsec, eps, fact_PK, pow_PK, ldyn_cdnc_min, cdnc_min_fixed,
    cdnc_min_lower, cdnc_min_upper, rcd_vol_max, cqtmin, conv_effr2mvr
)

def get_util_var(nproma, nbdim, ntdia, nlev, nlevp1, paphm1, pgeo, papm1, ptm1):
    """Get several utility variables:
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
        c.rgrav * pdp[:, ntdia:nlev]
    )

    # Height differences
    pdz = pdz.at[:, ntdia:nlevp1].set(
        c.rgrav * (pgeoh[:, ntdia:nlev] - pgeoh[:, ntdia+1:nlevp1])
    )
    # Might change it to this to keep it consistent with pressure differ
    # pdz = pdz.at[:, ntdia:nlevp1].set(
    #     c.rgrav * (pgeoh[:, ntdia+1:nlevp1] - pgeoh[:, ntdia:nlev])
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
    """Flag the top, base, and intermediate levels for each cloud.

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

def eff_ice_crystal_radius(pxice: jnp.ndarray, picnc: jnp.ndarray) -> jnp.ndarray:
    """Effective ice crystal radius following Lohmann et al. (2008, ERL), expression (1),
    using the Pruppacher & Klett (1997) mass–size relation parameters.

    Parameters
    ----------
    pxice : jnp.ndarray
        In-cloud ice mass concentration [g/m^3].
    picnc : jnp.ndarray
        Ice crystal number concentration (ICNC) [1/m^3].

    Returns
    -------
    prieff : jnp.ndarray
        Effective ice crystal radius [micron].

    """
    # fact_PK and pow_PK are constants/params from the scheme (imported in this module or via cloud_params_2m).
    return 0.5e4 * (pxice / jnp.maximum(fact_PK * jnp.maximum(picnc, eps), eps)) ** (1.0 / pow_PK)

@jit
def minimum_CDNC(pxwat, ldyn_cdnc_min=ldyn_cdnc_min, cdnc_min_fixed=cdnc_min_fixed):
    """Set the minimum cloud droplet number concentration, either statically or dynamically.

    Parameters
    ----------
        pxwat (array): In-cloud water mixing ratio [kg/m^3].
        ldyn_cdnc_min (bool): Flag to use dynamic CDNC minimum.
        cdnc_min_fixed (float): Static minimum CDNC value in cm^-3.

    Returns
    -------
        pcdnc_min (array): Minimum cloud droplet number concentration [m^-3].

    """
    if ldyn_cdnc_min:
        # Dynamic value for minimum CDNC
        pcdnc_min = rcd_vol_max**(-3.0) * (3.0 / (4.0 * pi * c.rhow)) * pxwat
        pcdnc_min = jnp.clip(pcdnc_min, cdnc_min_lower, cdnc_min_upper)
    else:
        # Static minimum CDNC
        pcdnc_min = cdnc_min_fixed * 1.0e6  # Convert from cm^-3 to m^-3
        pcdnc_min = jnp.full_like(pxwat, pcdnc_min)

    return pcdnc_min

def gridbox_frac_falling_hydrometeor(
    precip_flux_from_above: jnp.ndarray,
    precip_frac_from_above: jnp.ndarray,
    precip_flux_from_level: jnp.ndarray,
    precip_frac_from_level: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the grid box fraction covered by falling hydrometeor (e.g., rain+snow, sedimenting ice).

    Parameters
    ----------
    precip_flux_from_above : jnp.ndarray
        Flux of falling hydrometeor from above.
    precip_frac_from_above : jnp.ndarray
        Fraction of gridbox covered by falling hydrometeor from above.
    precip_flux_from_level : jnp.ndarray
        Flux of falling hydrometeor from the current level.
    precip_frac_from_level : jnp.ndarray
        Fraction of gridbox covered by falling hydrometeor from the current level.
    min_precip_flux : float
        Minimum threshold for total flux.

    Returns
    -------
    jnp.ndarray
        Total fraction of gridbox covered by falling hydrometeor.

    """
    # Determine where flux from above is greater than flux from the current level
    ll1 = precip_flux_from_above > precip_flux_from_level

    # Update fraction from above based on condition
    updated_precip_frac_from_above = jnp.where(
        ll1, precip_frac_from_above, precip_frac_from_level
    )

    # Compute total flux
    total_precip_flux = precip_flux_from_above + precip_flux_from_level

    # Determine where total flux is greater than the minimum threshold
    ll1 = total_precip_flux > cqtmin

    # Compute weighted average fraction
    weighted_precip_frac = (
        (precip_frac_from_level * precip_flux_from_level + updated_precip_frac_from_above * precip_flux_from_above)
        / jnp.maximum(total_precip_flux, cqtmin)
    )
    weighted_precip_frac = jnp.clip(weighted_precip_frac, 0.0, 1.0)

    # Compute total fraction
    total_precip_frac = jnp.where(ll1, weighted_precip_frac, 0.0)

    return total_precip_frac

def effective_2_volmean_radius_param_Schuman_2011(prieff: jnp.ndarray) -> jnp.ndarray:
    """Convert effective radius to volume-mean radius using Schumann et al. (2011) parametrisation.

    Parameters
    ----------
    prieff : jnp.ndarray
        Effective ice crystal radius (Fortran: prieff) given in units of 1.e-6 m (i.e. microns).

    Returns
    -------
    prvolmean : jnp.ndarray
        Volume-mean ice crystal radius (Fortran: prvolmean) in metres.

    Notes
    -----
    Fortran implementation:
        prvolmean = MAX(1.e-6_dp, conv_effr2mvr*1.e-6_dp*prieff)
    where conv_effr2mvr (imported) is the scheme constant converting effective -> vol-mean radius.

    """
    # Multiply prieff (1e-6 m units) by 1e-6 to get metres, apply conv_effr2mvr and enforce minimum 1e-6 m.
    return jnp.maximum(1e-6, conv_effr2mvr * 1e-6 * prieff)

def breadth_factor(pcdnc: jnp.ndarray) -> jnp.ndarray:
    """Breadth factor as a function of cloud droplet number concentration (CDNC).

    Parameters
    ----------
    pcdnc : jnp.ndarray
        Cloud droplet number concentration (Fortran: pcdnc) [1/m^3].

    Returns
    -------
    pkap : jnp.ndarray
        Breadth factor (Fortran: pkap). Parametrisation from Peng & Lohmann (2003), eq. 6:
            pkap = 0.00045e-6 * pcdnc + 1.18
        The constant 0.00045e-6 is equal to 4.5e-10.

    """
    return 4.5e-10 * pcdnc + 1.18

def threshold_vert_vel(
    sat_vap_pres_water: jnp.ndarray,  # pesw [Pa]
    sat_vap_pres_ice: jnp.ndarray,    # pesi [Pa]
    icnc: jnp.ndarray,                # picnc [1/m^3]
    ice_radius: jnp.ndarray,          # price [m] volume-mean ice crystal radius
    eta: jnp.ndarray,                 # peta [-]
) -> jnp.ndarray:
    """Threshold vertical velocity for the Wegener-Bergeron-Findeisen (WBF) criterion.

    JAX port of Fortran function `threshold_vert_vel_1d` (mo_cloud_microphysics_2m).

    The WBF process (ice growth at the expense of supercooled liquid) is active when
    the actual updraft velocity is below this threshold. The threshold is proportional
    to the supersaturation of water vapour over ice, the ice crystal number concentration,
    the crystal size, and a diffusivity-related factor `eta`.

    Parameters
    ----------
    sat_vap_pres_water : array
        Saturation vapour pressure w.r.t. liquid water `pesw` [Pa].
    sat_vap_pres_ice : array
        Saturation vapour pressure w.r.t. ice `pesi` [Pa].
    icnc : array
        Ice crystal number concentration `picnc` [1/m^3].
    ice_radius : array
        Volume-mean ice crystal radius `price` [m].
    eta : array
        Diffusivity-related variable for the WBF criterion `peta` [-].

    Returns
    -------
    pvervmax : array
        Threshold vertical velocity [m/s] (same units as `pvervx` in the calling routine,
        which is compared after scaling by 0.01 from cm/s).

    """
    return (
        (sat_vap_pres_water - sat_vap_pres_ice)
        / jnp.maximum(sat_vap_pres_ice, eps)
        * icnc
        * ice_radius
        * eta
    )

def consistency_number_to_mass(
    pthreshold: float | jnp.ndarray,
    pmass: jnp.ndarray,
    pnumber: jnp.ndarray,
    ) -> jnp.ndarray:
    """Return a "physical" number concentration/flux: whenever the corresponding mass
    is below `pthreshold`, the number is reset to 0.

    Parameters
    ----------
    pthreshold : float or jnp.ndarray
        Threshold below which `pnumber` is forced to zero.
    pmass : jnp.ndarray
        Mass-like quantity (e.g. ice flux mass) [units arbitrary].
    pnumber : jnp.ndarray
        Number-like quantity associated with `pmass`.

    Returns
    -------
    jnp.ndarray
        `pnumber` with entries zeroed where `pmass < pthreshold`.

    """
    return jnp.where(pmass < pthreshold, 0.0, pnumber)

def init_cloud_micro_2m(lconv):
    """Initialize boundary conditions for the cloud microphysics scheme.

    Parameters
    ----------
        lconv (bool): Logical flag indicating whether convection is enabled.

    Returns
    -------
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

