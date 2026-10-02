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
  ``p₀=10⁵ Pa``) plus a moisture-buoyancy term, weighted by the lowest
  level's cloud cover (:func:`surface_bulk_richardson`). Stability functions are
  Louis (1979) — momentum and heat have separate forms in both stable
  and unstable branches. Per-tile heat roughness ``z0h`` and the
  humidity factors come from ``state.roughness_heat`` and
  ``state.surface_cair``/``surface_csat`` (open water / ice are fully
  saturated; land uses JSBACH's factors, see
  ``jcm.physics.surface.echam.jsbach_land.humidity_factors``).

Both schemes return
``(surface_exchange_heat, surface_exchange_moisture, surface_exchange_momentum)``
shaped ``(ncol, nsfc_type)`` in m/s — i.e. CH·|U|, CE·|U|, CM·|U| in the
bulk-aerodynamic sense, ready to be multiplied by ρ for a flux. These per-tile
exchange velocities are the single source the ``echam_surface`` term uses for
both the surface-flux magnitude and its implicit-damping factor, so heat,
moisture and momentum stay mutually consistent (one ECHAM ``sfc_exchange_coeff``,
not a separate momentum proxy).

The Louis form matches ECHAM/ICON ~order-of-magnitude across the full
``Ri`` range; the Businger-Dyer form matches well near neutral but
diverges a few× in strongly unstable conditions (``(1−16Ri)^(1/2)``
grows linearly in |Ri| while Louis asymptotes). See
``fortran_harness/PLAN.md`` (on the ``origin/fortran-harness-vdiff``
branch, not in this tree) for harness numbers.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

import jcm.constants as c
from jcm.physics.thermodynamics import saturation_specific_humidity
from .moist_buoyancy import cloud_weighted_buoyancy_multipliers
from .vertical_diffusion_types import VDiffParameters, VDiffState


@jax.jit
def surface_bulk_richardson(
    state: VDiffState,
    params: VDiffParameters,
    wind_speed_surface: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    temperature_air: jnp.ndarray,
) -> jnp.ndarray:
    """Bulk Richardson number of the surface layer, per tile (ncol, nsfc_type).

    ECHAM's ``zril``/``zriw``/``zrii`` (``mo_surface_land.f90::precalc_land``
    l.200-236 and the ocean and ice analogues), with the layer-mean
    quantities weighted ``fsl·air + (1 - fsl)·surface`` (``fsl`` =
    ``params.surface_layer_fsl``)::

        zbuoy = zdus1·(θ_l,air − θ_s) + zdus2·θ_mid·(q_t,air − q_t,s)
        Ri    = z_ref·g·zbuoy / (θ_v,mid·max(|U|², 1))

    where ``(zdus1, zdus2)`` are the cloud-weighted multipliers of
    :func:`~.moist_buoyancy.cloud_weighted_buoyancy_multipliers` with the
    lowest level's cloud cover ``state.cloud_fraction[:, -1]``, the ``paclc``
    ECHAM hands to the surface for every tile (``mo_surface.f90``
    l.512, 555, 595). The latent heat is that of the lowest-level air
    (``FSEL(T - tmelt, alv, als)``), the same for every tile.

    The tile's surface humidity is ``csat·q_s + (1 − cair)·q_a`` from its
    humidity factors (``state.surface_cair``/``surface_csat``; 1 = fully
    saturated open water/ice).

    Args:
        state: Atmospheric state; ``fsl`` comes from ``params`` and the tile
            humidity factors from ``state.surface_cair``/``surface_csat``
            (``state.surface_wetness`` when they are ``None``).
        params: Vertical diffusion parameters.
        wind_speed_surface: Lowest-level wind speed [m/s] (ncol,).
        temperature_surface: Tile skin temperature [K] (ncol, nsfc_type).
        temperature_air: Lowest-level air temperature [K] (ncol,).

    Returns:
        Bulk Richardson number [-], (ncol, nsfc_type).

    """
    # Read shared physical constants by attribute access on the
    # ``jcm.constants`` module so any ``set_constants`` override is honoured.
    rd = c.rd
    cp = c.cpd
    p0 = c.p0     # 1.0e5 Pa — same as ECHAM's p0ref
    vtmpc1 = c.vtmpc1
    fsl = params.surface_layer_fsl
    nsfc_type = temperature_surface.shape[1]

    # --- Atmospheric inputs at the lowest level (klev) -------------------
    p_air = state.pressure_full[:, -1]            # (ncol,)
    p_sfc = state.pressure_half[:, -1]            # (ncol,)
    T_air = temperature_air                        # (ncol,)
    qv_air = state.qv[:, -1]                       # (ncol,)
    qx_air = state.qc[:, -1] + state.qi[:, -1]    # total cloud water
    cover = state.cloud_fraction[:, -1]           # paclc(klev)
    z_ref = jnp.maximum(state.height_full[:, -1] - state.height_half[:, -1], 1.0)

    # Phase of the latent heat in the surface-layer buoyancy: vdiff.f90's
    # ``zfaxe = FSEL(T - tmelt, alv, als)`` on the lowest-level air
    # temperature — condensation above the melting point, sublimation below,
    # the same value for every tile.
    Lv = jnp.where(T_air >= c.tmelt, c.alhc, c.alhs)

    exner_air = (p0 / jnp.maximum(p_air, 1.0)) ** (rd / cp)
    theta_air = T_air * exner_air                                  # ptheta_b
    thetav_air = theta_air * (1.0 + vtmpc1 * qv_air - qx_air)      # pthetav_b
    # Liquid-water potential temperature, vdiff.f90's
    # ``zlteta1 = θ − (zfaxe/cpd)·θ/T·zx`` with the same phase-switched L.
    thetal_air = theta_air - (Lv / cp) * theta_air / T_air * qx_air

    # Saturation of the air (``zqss``) and every tile surface: ECHAM reads
    # both from the ``ua`` table (``lookup_ua_spline`` in vdiff,
    # ``lookup_ua_list_spline`` in precalc_ocean/_ice/_land): Sonntag (1990)
    # over ice at and below tmelt and over water above, with no mixed-phase
    # blend (``phase="auto"`` of jcm.physics.thermodynamics).
    qsat_air = saturation_specific_humidity(T_air, p_air)
    qtl = qv_air + qx_air                                          # zqtl
    zdu2 = jnp.maximum(wind_speed_surface ** 2, 1.0)               # zepdu2 = 1.0

    ri_tiles = []
    for isfc in range(nsfc_type):
        T_s = temperature_surface[:, isfc]
        cair_all = (state.surface_cair if state.surface_cair is not None
                    else state.surface_wetness)
        csat_all = (state.surface_csat if state.surface_csat is not None
                    else state.surface_wetness)
        cair = jnp.clip(cair_all[:, isfc], 0.0, 1.0)
        csat = jnp.clip(csat_all[:, isfc], 0.0, 1.0)

        # Tile surface humidity as ECHAM's surface layer sees it,
        # ``csat·q_s + (1 − cair)·q_a`` (precalc_land ``ztvl``/``zqmitte``/
        # ``zqddif``, mo_surface_land.f90:200-232): open water / ice are fully
        # saturated (cair = csat = 1), land carries JSBACH's factors. Ice
        # saturation below tmelt (the sea ice tile always, frozen land),
        # water above.
        qsat_s = saturation_specific_humidity(T_s, p_sfc)
        qts = csat * qsat_s + (1.0 - cair) * qv_air

        exner_sfc = (p0 / jnp.maximum(p_sfc, 1.0)) ** (rd / cp)
        theta_s = T_s * exner_sfc
        thetav_s = theta_s * (1.0 + vtmpc1 * qts)

        # Mid-surface-layer averages (fsl·air + (1-fsl)·surface)
        w1, ws = fsl, 1.0 - fsl
        qtmid = w1 * qtl + ws * qts
        qsmid = w1 * qsat_air + ws * qsat_s
        T_mid = w1 * T_air + ws * T_s
        theta_mid = w1 * theta_air + ws * theta_s
        thetav_mid = w1 * thetav_air + ws * thetav_s

        zdus1, zdus2 = cloud_weighted_buoyancy_multipliers(
            Lv, T_mid, qtmid, qsmid, cover)

        # Bulk Richardson with full ECHAM buoyancy
        zdthetal = thetal_air - theta_s
        zdqt = qtl - qts
        zbuoy = zdus1 * zdthetal + zdus2 * theta_mid * zdqt
        ri_tiles.append(z_ref * c.grav * zbuoy / (thetav_mid * zdu2))
    return jnp.stack(ri_tiles, axis=1)


@jax.jit
def compute_surface_exchange_coefficients_echam_louis(
    state: VDiffState,
    params: VDiffParameters,
    wind_speed_surface: jnp.ndarray,
    temperature_surface: jnp.ndarray,
    temperature_air: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ECHAM-faithful per-tile surface exchange coefficient.

    Mirrors ``mo_turbulence_diag::sfc_exchange_coeff``. For each surface tile
    (water/ice/land) it takes

      1. the bulk Richardson number of :func:`surface_bulk_richardson`: the
         θ_l difference plus the moisture buoyancy, weighted by the
         lowest level's cloud cover, with the tile's surface humidity
         ``csat·q_s + (1 − cair)·q_a`` (``state.surface_cair``/
         ``surface_csat``: 1 = fully saturated open water/ice);
      2. Louis (1979) stability functions on top of a log-law neutral drag
         computed from the per-tile momentum roughness
         ``state.roughness_length`` and heat roughness
         ``state.roughness_heat``.

    Returns ``(surface_exchange_heat, surface_exchange_moisture,
    surface_exchange_momentum)`` = (CH·|U|, CE·|U|, CM·|U|) in m/s, per tile
    (heat and moisture are equal in this scheme). The caller multiplies by ρ to
    get the flux factor. The momentum coefficient ``cfm`` is the Louis (1979) /
    Mauritsen (2007) drag, so the surface momentum stress is built from a real
    CM·|U| rather than the interior diffusivity.
    """
    # Read shared physical constants by attribute access on the
    # ``jcm.constants`` module so any ``set_constants`` override is
    # honoured.
    karman = c.karman_const

    cb = params.louis_cb
    cc = params.louis_cc

    ncol, nsfc_type = temperature_surface.shape

    # Neutral drag for every tile, from the one helper the 10 m reduction also
    # uses (so the reduction's stability factor is this scheme's own).
    bn_all, cfn_m_all = echam_louis_neutral_drag(state, params,
                                                 wind_speed_surface)

    z_ref = jnp.maximum(state.height_full[:, -1] - state.height_half[:, -1], 1.0)
    zdu2 = jnp.maximum(wind_speed_surface ** 2, 1.0)               # zepdu2 = 1.0
    ri_all = surface_bulk_richardson(
        state, params, wind_speed_surface, temperature_surface, temperature_air)

    # --- Per-tile loop -------------------------------------------------
    surface_exchange_heat = jnp.zeros((ncol, nsfc_type))
    surface_exchange_moisture = jnp.zeros((ncol, nsfc_type))
    surface_exchange_momentum = jnp.zeros((ncol, nsfc_type))

    for isfc in range(nsfc_type):
        z0 = jnp.maximum(state.roughness_length[:, isfc], params.z0m_min)
        z0h = jnp.maximum(state.roughness_heat[:, isfc], params.z0m_min)
        ri = ri_all[:, isfc]

        # ---- Louis (1979) stability + log-law neutral ----------------
        # Effective roughness lengths capped to ½·z_ref via
        # ``MAX(2, z/z0)`` per ECHAM's lmix-bounded form. The momentum pair
        # comes from ``echam_louis_neutral_drag`` so the 10 m reduction shares
        # this scheme's own neutral reference rather than re-deriving it.
        log_zm = bn_all[:, isfc]
        log_zh = jnp.log(jnp.maximum(z_ref / z0h, jnp.exp(2.0)))
        cdn = (karman * karman) / (log_zm * log_zm)             # neutral drag
        chn = (karman * karman) / (log_zm * log_zh)             # neutral CHN

        cfn_m = cfn_m_all[:, isfc]                                # κ²·U/log²
        cfn_h = jnp.sqrt(jnp.maximum(zdu2, 1.0e-30)) * chn

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
        zucfm = jnp.sqrt(jnp.maximum(-ri_neg * (1.0 + z_ref / z0), 1.0e-30))
        zucfm = 1.0 / (1.0 + z3bc * cdn * zucfm)
        unstable_cfm = cfn_m * (1.0 - z2b * ri_neg * zucfm)

        zucfh = jnp.sqrt(jnp.maximum(-ri_neg * (1.0 + z_ref / z0h), 1.0e-30))
        zucfh = 1.0 / (1.0 + z3bc * chn * zucfh)
        unstable_cfh = cfn_h * (1.0 - z3b * ri_neg * zucfh)

        cfm = jnp.where(ri > 0.0, stable_cfm, unstable_cfm)
        cfh = jnp.where(ri > 0.0, stable_cfh, unstable_cfh)

        cfh = jnp.maximum(cfh, 1.0e-6)
        cfm = jnp.maximum(cfm, 1.0e-6)

        surface_exchange_heat = surface_exchange_heat.at[:, isfc].set(cfh)
        surface_exchange_moisture = surface_exchange_moisture.at[:, isfc].set(cfh)
        surface_exchange_momentum = surface_exchange_momentum.at[:, isfc].set(cfm)

    return surface_exchange_heat, surface_exchange_moisture, surface_exchange_momentum


@jax.jit
def wind_10m_reduction(
    exchange_momentum: jnp.ndarray,
    neutral_exchange_momentum: jnp.ndarray,
    log_z_over_z0: jnp.ndarray,
    z_ref: jnp.ndarray,
    reference_height: float = 10.0,
) -> jnp.ndarray:
    """Per-tile ``|U(10 m)| / |U(z_ref)|`` (ECHAM ``nsurf_diag`` 10 m wind).

    ECHAM5 ``vdiff.f90`` / ICON ``mo_surface_diag::nsurf_diag`` interpolate the
    lowest-level wind down to 10 m along the surface-layer profile actually used
    for the drag, with ``zrat = 10/z_ref``::

        bn   = ln(z_ref/z0m)                     neutral profile factor
        bm   = bn·√(CM_n·|U| / CM·|U|)           stability-corrected profile
        cbn  = ln(1 + (e^bn − 1)·zrat)
        cbs  = −(bn − bm)·zrat                   stable
        cbu  = −ln(1 + (e^(bn−bm) − 1)·zrat)     unstable
        red  = (cbn + [cbs|cbu]) / bm

    The neutral profile factor and the neutral exchange velocity are supplied
    by the caller rather than rebuilt here, because they must be the surface
    layer scheme's OWN: the two schemes differ in roughness (``z0m_min``-floored
    ``state.roughness_length`` vs a hard-coded table), in the bound on
    ``z/z0``, and in whether the wind is ``zepdu2``-floored. Deriving them here
    would silently mix two drag formulations and read stability where there is
    none.

    The stable/unstable branch is selected by ``CM·|U| < CM_n·|U|``, which is
    exactly ``Ri > 0`` for both surface-layer schemes here (their stability
    factors are <1 for stable, ≥1 for unstable, and both equal 1 — with equal
    ``cbs``/``cbu`` — at ``Ri = 0``), so the Richardson number need not be
    plumbed out of the coefficient solve.

    Args:
        exchange_momentum: CM·|U| per tile [m/s] (ncol, nsfc_type).
        neutral_exchange_momentum: the same scheme's NEUTRAL CM_n·|U| [m/s]
            (ncol, nsfc_type).
        log_z_over_z0: that scheme's neutral profile factor ``ln(z_ref/z0m)``
            (ncol, nsfc_type).
        z_ref: lowest full-level height above the surface [m] (ncol,).
        reference_height: diagnostic height [m], 10 m by default.

    Returns:
        Reduction factor per tile (ncol, nsfc_type), in [0, 1].

    """
    bn = log_z_over_z0
    f_m = (jnp.maximum(exchange_momentum, 1e-12)
           / jnp.maximum(neutral_exchange_momentum, 1e-12))
    bm = bn / jnp.sqrt(f_m)

    # A lowest level below the diagnostic height leaves the wind unreduced
    # (zrat ≤ 1). This subsumes the coefficients' 1 m floor on z_ref, since
    # the diagnostic height is 10 m — flooring at 1 m first would be a no-op.
    zrat = reference_height / jnp.maximum(z_ref, reference_height)[:, None]
    cbn = jnp.log1p(jnp.expm1(jnp.clip(bn, 0.0, 30.0)) * zrat)
    cbs = -(bn - bm) * zrat
    cbu = -jnp.log1p(jnp.expm1(jnp.clip(bn - bm, -30.0, 30.0)) * zrat)
    merge = jnp.where(f_m < 1.0, cbs, cbu)
    # Math-safety clip only: the log profile cannot amplify the wind between
    # 10 m and the lowest level, nor reverse it.
    return jnp.clip((cbn + merge) / bm, 0.0, 1.0)


def echam_louis_neutral_drag(state, params, wind_speed):
    """``(ln(z/z0), CM_n·|U|)`` of the ECHAM-Louis scheme, per tile.

    THE definition, not a copy of one:
    :func:`compute_surface_exchange_coefficients_echam_louis` calls this for
    the ``log_zm``/``cfn_m`` its own drag is built on, so ``bn = κ/√CDN``
    holds for the pair by construction and the two cannot drift.
    """
    z_ref = jnp.maximum(state.height_full[:, -1] - state.height_half[:, -1], 1.0)
    z0 = jnp.maximum(state.roughness_length, params.z0m_min)
    bn = jnp.log(jnp.maximum(z_ref[:, None] / z0, jnp.exp(2.0)))
    floored = jnp.sqrt(jnp.maximum(wind_speed ** 2, 1.0))[:, None]
    return bn, floored * c.karman_const ** 2 / bn ** 2
