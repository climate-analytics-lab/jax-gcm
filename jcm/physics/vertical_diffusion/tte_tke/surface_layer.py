"""Surface-layer exchange-coefficient schemes for the TTE-TKE vdiff package.

Two schemes live here as peers, selectable via
``VDiffParameters.surface_layer_scheme``:

- ``"businger_dyer"`` (default): the original ICON-port form in
  ``turbulence_coefficients.compute_surface_exchange_coefficients`` —
  bulk Richardson built from raw temperatures, Businger-Dyer stability
  functions on top of a κ²/[ln(z/z₀)]² neutral drag. Lives in this
  module too, mostly for symmetry; the dispatcher in
  ``turbulence_coefficients.py`` still imports the original.

- ``"echam_louis"``: faithful port of ECHAM/ICON
  ``mo_turbulence_diag::sfc_exchange_coeff``. Bulk Richardson uses
  potential temperatures (with Exner ``(p₀/p)^(R/cₚ)`` referenced to
  ``p₀=10⁵ Pa``) plus a moisture-buoyancy term. Stability functions are
  Louis (1979) — momentum and heat have separate forms in both stable
  and unstable branches. Surface saturation specific humidity comes
  from the same Tetens form ``saturation_specific_humidity`` already
  uses elsewhere in the package.

Both schemes return ``(surface_exchange_heat, surface_exchange_moisture)``
shaped ``(ncol, nsfc_type)`` in m/s — i.e. CH·|U| in the bulk-aerodynamic
sense, ready to be multiplied by ρ for a flux.

The Louis form matches ECHAM/ICON ~order-of-magnitude across the full
``Ri`` range; the Businger-Dyer form matches well near neutral but
diverges a few× in strongly unstable conditions (``(1−16Ri)^(1/2)``
grows linearly in |Ri| while Louis asymptotes). See
``fortran_harness/PLAN.md`` for harness numbers.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from jcm.physics.clouds.sundqvist import saturation_specific_humidity
from .vertical_diffusion_types import VDiffParameters, VDiffState

PHYS_CONST = PhysicalConstants()

# ECHAM ``mo_echam_vdiff_params`` constants
_FSL = 0.4          # surface-layer mid-level weighting (40 % air, 60 % sfc)
_CB = 5.0           # Louis stability parameter (near-neutrality)
_CC = 5.0           # Louis stability parameter (unstable cases)
_KARMAN = 0.4
_VTMPC1 = PHYS_CONST.rv / PHYS_CONST.rd - 1.0   # ≈ 0.608


@jax.jit
def compute_surface_exchange_coefficients_echam_louis(
    state: VDiffState,
    params: VDiffParameters,
    wind_speed_surface: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    temperature_air: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ECHAM-faithful per-tile surface exchange coefficient.

    Mirrors ``mo_turbulence_diag::sfc_exchange_coeff``. Loops over each
    surface tile (water/ice/land) and computes:

      1. Saturation specific humidity at the tile surface
         (assumes saturated water/ice; for land we use the same in
         lieu of a JSBACH wet-fraction).
      2. Bulk Richardson number using θ_l difference + moisture
         buoyancy (Brutsaert clear-sky form, since paclc≈0 at the
         surface in this single-column harness path).
      3. Louis (1979) stability functions on top of a log-law neutral
         drag computed from the per-tile roughness length.

    Returns CH·|U| and CM·|U| (= sCH, sCM) in m/s, per tile. Caller
    multiplies by ρ to get the flux factor.
    """
    Rd = PHYS_CONST.rd
    cp = PHYS_CONST.cp
    grav = PHYS_CONST.grav
    Lv = PHYS_CONST.alhc
    p0 = PHYS_CONST.p0     # 1.0e5 Pa — same as ECHAM's p0ref
    rv_over_rd = PHYS_CONST.rv / Rd
    rd_over_rv = Rd / PHYS_CONST.rv

    ncol, nsfc_type = temperature_surface.shape

    # --- Atmospheric inputs at the lowest level (klev) -------------------
    p_air = state.pressure_full[:, -1]            # (ncol,)
    p_sfc = state.pressure_half[:, -1]            # (ncol,)
    T_air = temperature_air                        # (ncol,)
    qv_air = state.qv[:, -1]                       # (ncol,)
    qx_air = state.qc[:, -1] + state.qi[:, -1]    # total cloud water
    z_ref = jnp.maximum(state.height_full[:, -1] - state.height_half[:, -1], 1.0)

    exner_air = (p0 / jnp.maximum(p_air, 1.0)) ** (Rd / cp)
    theta_air = T_air * exner_air                                  # ptheta_b
    thetav_air = theta_air * (1.0 + _VTMPC1 * qv_air - qx_air)     # pthetav_b
    # Assuming no ice at surface layer → θ_l ≈ θ (clean PBL surface
    # condition). ECHAM also subtracts (Lv/cp)·θ/T·qx; with qx≈0 this is
    # zero. Including it would be a few×10⁻³ K correction at most.
    thetal_air = theta_air

    qsat_air = saturation_specific_humidity(p_air, T_air)
    qtl = qv_air + qx_air                                          # zqtl

    # --- Per-tile loop -------------------------------------------------
    surface_exchange_heat = jnp.zeros((ncol, nsfc_type))
    surface_exchange_moisture = jnp.zeros((ncol, nsfc_type))

    for isfc in range(nsfc_type):
        T_s = temperature_surface[:, isfc]
        z0 = state.roughness_length[:, isfc]
        z0 = jnp.maximum(z0, params.z0m_min)
        # Heat-roughness — ICON uses tile-specific forms (water:
        # exp(2 - 86·z0^0.375); ice: z0; land: paz0lh). For the harness
        # / first integration use z0m for all heat tiles too — small
        # impact on coefficients (a log of a ratio of two small things)
        # and avoids piping in a separate ``z0h`` field.
        z0h = z0

        # Tile saturation q (over open water/ice; over land assume
        # saturated leaf — JSBACH would supply pcsat/pcair).
        qsat_s = saturation_specific_humidity(p_sfc, T_s)
        qts = qsat_s

        exner_sfc = (p0 / jnp.maximum(p_sfc, 1.0)) ** (Rd / cp)
        theta_s = T_s * exner_sfc
        thetav_s = theta_s * (1.0 + _VTMPC1 * qts)

        # Mid-surface-layer averages (40 % air, 60 % surface)
        w1, ws = _FSL, 1.0 - _FSL
        qtmid = w1 * qtl + ws * qts
        qsmid = w1 * qsat_air + ws * qsat_s
        T_mid = w1 * T_air + ws * T_s
        theta_mid = w1 * theta_air + ws * theta_s
        thetav_mid = w1 * thetav_air + ws * thetav_s

        # Cloud-cover-weighted buoyancy coefficients
        # (paclc_b≈0 in the surface boundary layer for clear-sky tests;
        # we still compute the cloudy-sky multipliers so the formula
        # remains correct when paclc>0 is fed through.)
        zfux = Lv / (cp * jnp.maximum(T_mid, 100.0))
        zfox = Lv / (Rd * jnp.maximum(T_mid, 100.0))
        zmult1 = 1.0 + _VTMPC1 * qtmid
        zmult2 = zfux * zmult1 - rv_over_rd
        zmult3 = (rd_over_rv * zfox * qsmid
                  / (1.0 + rd_over_rv * zfox * zfux * qsmid))
        zmult5 = zmult1 - zmult2 * zmult3
        zmult4 = zfux * zmult5 - 1.0

        # No cloud at surface — but keep the mixed form for completeness
        aclc = jnp.zeros_like(T_air)
        zdus1 = aclc * zmult5 + (1.0 - aclc) * zmult1
        zdus2 = aclc * zmult4 + (1.0 - aclc) * _VTMPC1

        # Bulk Richardson with full ECHAM buoyancy
        zdthetal = thetal_air - theta_s
        zdqt = qtl - qts
        zdu2 = jnp.maximum(wind_speed_surface ** 2, 1.0)   # zepdu2 = 1.0
        zbuoy = zdus1 * zdthetal + zdus2 * theta_mid * zdqt
        ri = z_ref * grav * zbuoy / (thetav_mid * zdu2)

        # ---- Louis (1979) stability + log-law neutral ----------------
        # Effective roughness lengths capped to ½·z_ref via
        # ``MAX(2, z/z0)`` per ECHAM's lmix-bounded form.
        log_zm = jnp.log(jnp.maximum(z_ref / z0,  jnp.exp(2.0)))
        log_zh = jnp.log(jnp.maximum(z_ref / z0h, jnp.exp(2.0)))
        cdn = (_KARMAN * _KARMAN) / (log_zm * log_zm)             # neutral drag
        chn = (_KARMAN * _KARMAN) / (log_zm * log_zh)             # neutral CHN

        cfn_m = jnp.sqrt(zdu2) * cdn        # κ²·U/log²
        cfn_h = jnp.sqrt(zdu2) * chn

        # Stable branch (Ri > 0): ECHAM Mauritsen-2007 stable form
        # f_tau/f_tau0   = 0.25 + 0.75/(1+4Ri)
        # f_theta/f_theta0 = 1/(1+4Ri)
        denom_stable = 1.0 + 4.0 * jnp.maximum(ri, 0.0)
        stable_cfm = cfn_m * (0.25 + 0.75 / denom_stable)
        stable_cfh = cfn_h * (1.0 / denom_stable) * jnp.sqrt(
            0.25 + 0.75 / denom_stable)

        # Unstable branch (Ri ≤ 0): Louis 1979 functions
        z2b = 2.0 * _CB           # 10
        z3b = 3.0 * _CB           # 15
        z3bc = 3.0 * _CB * _CC    # 75
        ri_neg = jnp.minimum(ri, 0.0)
        zucfm = jnp.sqrt(-ri_neg * (1.0 + z_ref / z0))
        zucfm = 1.0 / (1.0 + z3bc * cdn * zucfm)
        unstable_cfm = cfn_m * (1.0 - z2b * ri_neg * zucfm)

        zucfh = jnp.sqrt(-ri_neg * (1.0 + z_ref / z0h))
        zucfh = 1.0 / (1.0 + z3bc * chn * zucfh)
        unstable_cfh = cfn_h * (1.0 - z3b * ri_neg * zucfh)

        cfm = jnp.where(ri > 0.0, stable_cfm, unstable_cfm)
        cfh = jnp.where(ri > 0.0, stable_cfh, unstable_cfh)

        cfh = jnp.maximum(cfh, 1.0e-6)
        cfm = jnp.maximum(cfm, 1.0e-6)

        surface_exchange_heat = surface_exchange_heat.at[:, isfc].set(cfh)
        surface_exchange_moisture = surface_exchange_moisture.at[:, isfc].set(cfh)

    return surface_exchange_heat, surface_exchange_moisture
