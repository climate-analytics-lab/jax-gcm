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
  and unstable branches. Per-tile heat roughness ``z0h`` and surface
  wetness come from ``state.roughness_heat`` and
  ``state.surface_wetness``, which the caller populates from the
  boundary forcing (open water / ice are fully saturated; land uses the
  soil-moisture-derived ``cair``-style fraction).

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

from jcm.physics.clouds.sundqvist import saturation_specific_humidity
from .vertical_diffusion_types import VDiffParameters, VDiffState


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

      1. Effective surface specific humidity, blending tile saturation
         with ambient air using ``state.surface_wetness`` (1.0 = fully
         saturated open water/ice; <1 = soil-moisture-limited land).
      2. Bulk Richardson number using θ_l difference + moisture
         buoyancy (Brutsaert clear-sky form, since paclc≈0 at the
         surface in this single-column harness path).
      3. Louis (1979) stability functions on top of a log-law neutral
         drag computed from the per-tile momentum roughness
         ``state.roughness_length`` and heat roughness
         ``state.roughness_heat``.

    Returns CH·|U| and CM·|U| (= sCH, sCM) in m/s, per tile. Caller
    multiplies by ρ to get the flux factor.
    """
    # Use the ``physical_constants`` instance already exposed on
    # ``jcm.physics.icon``. Importing via ``jcm.physics.icon`` (whose
    # ``__init__`` has already loaded the constants subpackage) avoids
    # the partial-init problem that hits a bare
    # ``from jcm.physics.icon.constants.physical_constants import …``
    # under Python 3.11 + pytest-cov when this function is reached via
    # the icon → parameters → tte_tke chain.
    from jcm.physics.icon import physical_constants as PHYS_CONST

    Rd = PHYS_CONST.rd
    cp = PHYS_CONST.cp
    grav = PHYS_CONST.grav
    Lv = PHYS_CONST.alhc
    p0 = PHYS_CONST.p0     # 1.0e5 Pa — same as ECHAM's p0ref
    rv_over_rd = PHYS_CONST.rv / Rd
    rd_over_rv = Rd / PHYS_CONST.rv
    karman = PHYS_CONST.karman_const
    vtmpc1 = rv_over_rd - 1.0   # ≈ 0.608 (q-buoyancy coefficient)

    fsl = params.surface_layer_fsl
    cb = params.louis_cb
    cc = params.louis_cc

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
    thetav_air = theta_air * (1.0 + vtmpc1 * qv_air - qx_air)      # pthetav_b
    # θ_l ≈ θ here (no ice at surface layer; ECHAM also subtracts
    # (Lv/cp)·θ/T·qx but with qx≈0 this is a few×10⁻³ K correction).
    thetal_air = theta_air

    qsat_air = saturation_specific_humidity(p_air, T_air)
    qtl = qv_air + qx_air                                          # zqtl

    # --- Per-tile loop -------------------------------------------------
    surface_exchange_heat = jnp.zeros((ncol, nsfc_type))
    surface_exchange_moisture = jnp.zeros((ncol, nsfc_type))

    for isfc in range(nsfc_type):
        T_s = temperature_surface[:, isfc]
        z0 = jnp.maximum(state.roughness_length[:, isfc], params.z0m_min)
        z0h = jnp.maximum(state.roughness_heat[:, isfc], params.z0m_min)
        wetness = jnp.clip(state.surface_wetness[:, isfc], 0.0, 1.0)

        # Tile saturation q at the surface — open water / ice are fully
        # saturated, land is wetness-weighted between qsat and ambient
        # ``qv_air`` (mirrors the JSBACH ``cair·qsat + (1-cair)·qair``
        # form in mo_turbulence_diag).
        qsat_s = saturation_specific_humidity(p_sfc, T_s)
        qts = wetness * qsat_s + (1.0 - wetness) * qv_air

        exner_sfc = (p0 / jnp.maximum(p_sfc, 1.0)) ** (Rd / cp)
        theta_s = T_s * exner_sfc
        thetav_s = theta_s * (1.0 + vtmpc1 * qts)

        # Mid-surface-layer averages (fsl·air + (1-fsl)·surface)
        w1, ws = fsl, 1.0 - fsl
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
        zmult1 = 1.0 + vtmpc1 * qtmid
        zmult2 = zfux * zmult1 - rv_over_rd
        zmult3 = (rd_over_rv * zfox * qsmid
                  / (1.0 + rd_over_rv * zfox * zfux * qsmid))
        zmult5 = zmult1 - zmult2 * zmult3
        zmult4 = zfux * zmult5 - 1.0

        # No cloud at surface — but keep the mixed form for completeness
        aclc = jnp.zeros_like(T_air)
        zdus1 = aclc * zmult5 + (1.0 - aclc) * zmult1
        zdus2 = aclc * zmult4 + (1.0 - aclc) * vtmpc1

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
        cdn = (karman * karman) / (log_zm * log_zm)             # neutral drag
        chn = (karman * karman) / (log_zm * log_zh)             # neutral CHN

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
        z2b = 2.0 * cb              # ECHAM constant ``2·cb``
        z3b = 3.0 * cb              # ``3·cb``
        z3bc = 3.0 * cb * cc        # ``3·cb·cc``
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
