"""Betts-Miller convective adjustment — column physics.

A faithful JAX port of Isca's ``betts_miller.f90`` (the Frierson 2007 Simplified
Betts-Miller scheme and its ``do_shallower`` / ``do_changeqref`` siblings). The
scheme relaxes temperature and humidity toward a moist-adiabatic reference
profile at a target relative humidity ``rhbm`` over a timescale ``tau_bm``, with
an energy-consistency correction so that latent heating and column moistening
balance and precipitation stays non-negative.

Conventions
-----------
* Vertical index 0 is the model top, index ``kx-1`` is the surface (matching the
  rest of jcm; Isca uses the opposite ordering, handled internally).
* SI units throughout: temperature [K], specific humidity [kg/kg], pressure
  [Pa]. (Frierson 2007 and SpeedyWeather.jl, which the issue cites, are also
  formulated in specific humidity; Isca's mixing-ratio form differs only at
  second order, which is negligible for this idealized adjustment.)
* Physical constants are read by attribute access from :mod:`jcm.constants` so a
  ``set_constants`` override is honoured.

The per-column routine :func:`betts_miller_column` is ``vmap``-ed over the
horizontal grid by :func:`betts_miller_tendencies`; the flavor (``params.shallow``)
and modifiers are *static* and resolved by Python branching at trace time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import jcm.constants as c
from jcm.physics.convection.betts_miller.params import (
    BettsMillerParameters,
    ShallowScheme,
)

# Tetens saturation vapour pressure reference (over liquid water).
_ES0 = 610.78  # Pa at 273.16 K
_ES_A = 17.269
_ES_T0 = 273.16
_ES_TB = 35.86

_T_FLOOR = 173.16  # K — below this the parcel ascent is presumed CAPE-free (Isca).


def saturation_vapor_pressure(temperature: jnp.ndarray) -> jnp.ndarray:
    """Saturation vapour pressure over liquid water (Pa), Tetens formula."""
    return _ES0 * jnp.exp(_ES_A * (temperature - _ES_T0) / (temperature - _ES_TB))


def saturation_specific_humidity(temperature: jnp.ndarray,
                                 pressure: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity [kg/kg] at ``temperature`` [K], ``pressure`` [Pa]."""
    es = saturation_vapor_pressure(temperature)
    # Cap es below the pressure so the denominator stays positive at low p.
    es = jnp.minimum(es, 0.99 * jnp.maximum(pressure, 1.0))
    return c.eps * es / jnp.maximum(pressure - (1.0 - c.eps) * es, 1.0)


def _moist_dtdlnp(temperature, q, kappa, hlv, cp, rv):
    """Moist pseudo-adiabatic ``dT/dln(p)`` (Isca's first-order form, q for r)."""
    a = kappa * temperature + hlv / cp * q
    b = hlv * hlv * q / (cp * rv * temperature * temperature)
    return a / (1.0 + b)


def _parcel_ascent(tin, qin, pfull, phalf, buoyancy_kick):
    """Lift a surface parcel and return its profile plus CAPE / cloud mask.

    All inputs are 1-D column profiles ordered top (0) -> surface (kx-1).

    Returns:
        tp: parcel temperature [K], (kx,)
        cloud: bool mask, True on levels from the level of zero buoyancy down to
            the surface (the layers the deep adjustment relaxes), (kx,)
        cape: convective available potential energy [J/kg], scalar

    """
    kappa, hlv, cp, rv, rd = c.akap, c.alhc, c.cpd, c.rv, c.rd
    pstar = c.p0
    kx = tin.shape[0]

    # Work surface -> top: flip so lax.scan ascends.
    t_env = tin[::-1]
    p = pfull[::-1]
    # Half-level pressure bounding each full level: ph_lo = lower (towards
    # surface) interface, ph_hi = upper (towards top). phalf is (kx+1,), top..sfc.
    ph_lower = phalf[1:][::-1]   # interface below each level (larger p)
    ph_upper = phalf[:-1][::-1]  # interface above each level (smaller p)

    t0 = t_env[0] + buoyancy_kick
    q_parcel = qin[::-1][0]                      # conserved below saturation
    theta0 = t0 * (pstar / p[0]) ** kappa

    def step(carry, k):
        t_prev, q_prev, p_prev, saturated, has_cape, stopped, klzb, cape = carry
        p_k = p[k]

        # Dry-adiabatic candidate (potential temperature conserved).
        t_dry = theta0 * (p_k / pstar) ** kappa
        qsat_dry = saturation_specific_humidity(t_dry, p_k)
        newly_sat = q_parcel >= qsat_dry
        is_sat = saturated | newly_sat

        # Moist-adiabatic candidate (single ln-p step from the level below).
        dtdlnp = _moist_dtdlnp(t_prev, q_prev, kappa, hlv, cp, rv)
        t_moist = t_prev + dtdlnp * jnp.log(jnp.maximum(p_k, 1.0) / p_prev)
        t_moist = jnp.maximum(t_moist, _T_FLOOR)
        qsat_moist = saturation_specific_humidity(t_moist, p_k)

        t_k = jnp.where(is_sat, t_moist, t_dry)
        # Parcel humidity: saturated above the LCL, conserved (q_parcel) below.
        q_k = jnp.where(is_sat, qsat_moist, q_parcel)

        buoyant = t_k > t_env[k]
        # Buoyancy contribution to CAPE (energy per unit mass), guard log at top.
        dlnp = jnp.log(jnp.maximum(ph_lower[k], 1.0) / jnp.maximum(ph_upper[k], 1.0))
        contrib = rd * (t_k - t_env[k]) * dlnp

        # Track the contiguous buoyant plume above the level of free convection.
        active = is_sat & buoyant & (~stopped)
        has_cape_new = has_cape | active
        # Stop accumulating once we go non-buoyant after having had CAPE.
        stop_now = has_cape & (~buoyant)
        stopped_new = stopped | stop_now
        cape_new = cape + jnp.where(active, contrib, 0.0)
        # klzb = highest (smallest-index) level still in the plume.
        klzb_new = jnp.where(active, k_index_from_scan(kx, k), klzb)

        carry_out = (t_k, q_k, p_k, is_sat, has_cape_new, stopped_new,
                     klzb_new, cape_new)
        return carry_out, t_k

    init = (t0, q_parcel, p[0], jnp.bool_(False), jnp.bool_(False),
            jnp.bool_(False), jnp.int32(kx - 1), jnp.float32(0.0))
    (_, _, _, _, has_cape, _, klzb, cape), tp_rev = lax.scan(
        step, init, jnp.arange(kx)
    )

    tp = tp_rev[::-1]                 # back to top -> surface ordering
    # Cloud levels relaxed by the deep scheme: from klzb (top of plume) to surface.
    levels = jnp.arange(kx)
    cloud = (levels >= klzb) & has_cape
    cape = jnp.maximum(cape, 0.0)
    return tp, cloud, cape, has_cape


def k_index_from_scan(kx, scan_idx):
    """Map a surface->top scan position to a top->surface level index."""
    return (kx - 1) - scan_idx


def _reference_profiles(tin, qin, tp, pfull, cloud, rhbm, do_envsat):
    """Build reference T and q on the cloud levels (environment elsewhere)."""
    t_ref = jnp.where(cloud, tp, tin)
    if do_envsat:
        qsat_ref = saturation_specific_humidity(tin, pfull)
    else:
        qsat_ref = saturation_specific_humidity(tp, pfull)
    q_ref = jnp.where(cloud, rhbm * qsat_ref, qin)
    return t_ref, q_ref


def betts_miller_column(tin, qin, pfull, phalf, dt, params: BettsMillerParameters):
    """Betts-Miller tendencies for a single column.

    Args:
        tin: temperature [K], (kx,) top->surface.
        qin: specific humidity [kg/kg], (kx,).
        pfull: full-level pressure [Pa], (kx,).
        phalf: half-level pressure [Pa], (kx+1,).
        dt: timestep [s].
        params: static :class:`BettsMillerParameters`.

    Returns:
        (tdel, qdel, precip): temperature & humidity increments over ``dt``
        [(kx,), (kx,)] and column precipitation [kg/m^2], scalar.

    """
    hlv, cp, grav = c.alhc, c.cpd, c.grav
    dp = phalf[1:] - phalf[:-1]                       # layer thickness [Pa] (>0)

    tp, cloud, cape, has_cape = _parcel_ascent(
        tin, qin, pfull, phalf, params.buoyancy_kick)

    t_ref, q_ref = _reference_profiles(
        tin, qin, tp, pfull, cloud, params.rhbm, params.do_envsat)

    # CAPE-scaled relaxation timescale (do_taucape).
    tau = params.tau_bm
    if params.do_taucape:
        tau = jnp.sqrt(params.capetaubm) * params.tau_bm / jnp.sqrt(
            jnp.maximum(cape, 1e-10))
        tau = jnp.maximum(tau, params.tau_min)

    # Deep relaxation increments and column precip integrals.
    tdel = jnp.where(cloud, -(tin - t_ref) / tau * dt, 0.0)
    qdel = jnp.where(cloud, -(qin - q_ref) / tau * dt, 0.0)
    precip = -jnp.sum(qdel * dp) / grav                  # from moistening deficit
    precip_t = jnp.sum(cp / hlv * tdel * dp) / grav      # from latent heating

    tdel, qdel, precip = _energy_correction(
        tdel, qdel, precip, precip_t, t_ref, q_ref, tin, qin,
        cloud, dp, tau, dt, params)

    # No convection where there is no CAPE.
    tdel = jnp.where(has_cape, tdel, 0.0)
    qdel = jnp.where(has_cape, qdel, 0.0)
    precip = jnp.where(has_cape, jnp.maximum(precip, 0.0), 0.0)
    return tdel, qdel, precip


def _energy_correction(tdel, qdel, precip, precip_t, t_ref, q_ref, tin, qin,
                       cloud, dp, tau, dt, params):
    """Apply the precip/energy-consistency correction and shallow branches.

    Mirrors the post-relaxation block of Isca's ``betts_miller``: the deep
    (precip>0, precip_t>0) energy match, and the negative-precip shallow flavor.
    """
    # ---- Deep, energy-consistent case: precip > 0 and precip_t > 0 ----------
    # If the moistening implies more precip than the heating, shorten the q
    # timescale; otherwise shorten the t timescale (do_simp / Frierson 2007).
    safe_precip = jnp.where(precip != 0.0, precip, 1.0)
    safe_precip_t = jnp.where(precip_t != 0.0, precip_t, 1.0)
    q_over = precip > precip_t
    qdel_deep = jnp.where(q_over, qdel * (precip_t / safe_precip), qdel)
    tdel_deep = jnp.where(q_over, tdel, tdel * (precip / safe_precip_t))
    precip_deep = jnp.where(q_over, precip_t, precip)

    # ---- Shallow / negative-precip case: precip <= 0 < precip_t -------------
    shallow_tdel, shallow_qdel, shallow_precip = _shallow_branch(
        tdel, qdel, t_ref, q_ref, tin, qin, cloud, dp, tau, dt, params)

    deep_ok = (precip > 0.0) & (precip_t > 0.0)
    shallow_case = (precip <= 0.0) & (precip_t > 0.0)

    out_tdel = jnp.where(deep_ok, tdel_deep,
                         jnp.where(shallow_case, shallow_tdel, 0.0))
    out_qdel = jnp.where(deep_ok, qdel_deep,
                         jnp.where(shallow_case, shallow_qdel, 0.0))
    out_precip = jnp.where(deep_ok, precip_deep,
                           jnp.where(shallow_case, shallow_precip, 0.0))
    return out_tdel, out_qdel, out_precip


def _shallow_branch(tdel, qdel, t_ref, q_ref, tin, qin, cloud, dp, tau, dt,
                    params):
    """Apply the negative-precip flavor (static Python branch on params.shallow)."""
    grav = c.grav
    zeros = jnp.zeros_like(tdel)

    if params.shallow == ShallowScheme.CHANGEQREF:
        # Scale q_ref so the column-integrated precip is exactly zero, and shift
        # t_ref uniformly so the column-mean heating is zero (enthalpy-conserving,
        # non-precipitating). Faithful to Isca's do_changeqref.
        cloud_dp = jnp.where(cloud, dp, 0.0)
        deltaq = jnp.sum(jnp.where(cloud, qdel, 0.0) * tau / dt * cloud_dp)
        qrefint = jnp.sum(jnp.where(cloud, q_ref, 0.0) * cloud_dp)
        safe_qrefint = jnp.where(qrefint != 0.0, qrefint, 1.0)
        deltaqfrac2 = -deltaq / safe_qrefint * dt / tau
        new_qdel = jnp.where(cloud, qdel + deltaqfrac2 * q_ref, 0.0)
        # deltak = -mean_dp(tdel): added uniformly it zeroes the net heating.
        deltak = -jnp.sum(jnp.where(cloud, tdel, 0.0) * cloud_dp) / jnp.maximum(
            jnp.sum(cloud_dp), 1.0)
        new_tdel = jnp.where(cloud, tdel + deltak, 0.0)
        return new_tdel, new_qdel, jnp.float32(0.0)

    if params.shallow == ShallowScheme.SHALLOWER:
        return _do_shallower(tdel, qdel, cloud, dp, grav)

    if params.shallow == ShallowScheme.SIMP:
        # do_simp in the negative-precip regime simply suppresses convection
        # (the energy match already handled the positive-precip deep case).
        return zeros, zeros, jnp.float32(0.0)

    # ShallowScheme.NONE
    return zeros, zeros, jnp.float32(0.0)


def _do_shallower(tdel, qdel, cloud, dp, grav):
    """Raise the cloud top until the column-integrated precip is non-negative.

    Faithful (vectorized) port of Isca's ``do_shallower`` loop: the topmost
    cloud layers are the ones that *moisten* (drive precip < 0); remove them from
    the top down until the suffix (towards the surface) integrates to precip ≥ 0,
    then trim the boundary layer so the kept precip is exactly zero and shift the
    kept temperature increments to conserve enthalpy.
    """
    kx = tdel.shape[0]
    levels = jnp.arange(kx)
    # Per-layer precip contribution [kg/m^2]; >0 dries (rains), <0 moistens.
    lp = jnp.where(cloud, -qdel * dp / grav, 0.0)
    # Suffix sums towards the surface: cumsurf[k] = sum_{j>=k} lp[j].
    cumsurf = jnp.cumsum(lp[::-1])[::-1]

    # j = highest cloud level whose surface-ward suffix integrates to >= 0.
    # The fully-kept region is [j .. surface]; the *partially*-retained boundary
    # layer is the moistening layer just ABOVE it (j-1) — Isca scales the
    # last-removed layer back in to bring the kept precip to exactly zero. (The
    # shallow branch only runs when the full-column precip < 0, so j > klzb and
    # the boundary layer j-1 is always within the cloud.)
    qualifies = cloud & (cumsurf >= 0.0)
    j = jnp.where(qualifies, levels, kx).min()
    feasible = j < kx
    boundary = j - 1

    keep_full = (levels >= j) & cloud
    at_boundary = levels == boundary
    below = jnp.sum(jnp.where(keep_full, lp, 0.0))           # = cumsurf[j], >= 0
    lp_b = jnp.sum(jnp.where(at_boundary, lp, 0.0))          # boundary (moistening) layer
    # kept precip = below + frac*lp_b == 0  ->  frac = -below/lp_b  (in [0, 1]).
    frac = jnp.where(jnp.abs(lp_b) > 1e-12, -below / lp_b, 0.0)
    scale = jnp.where(keep_full, 1.0, jnp.where(at_boundary, frac, 0.0))

    new_tdel = tdel * scale
    new_qdel = qdel * scale
    # Zero the net heating of the kept region (Isca's deltak), conserving enthalpy.
    keep_dp = jnp.where(scale > 0.0, dp, 0.0)
    deltak = -jnp.sum(new_tdel * keep_dp) / jnp.maximum(jnp.sum(keep_dp), 1.0)
    new_tdel = jnp.where(scale > 0.0, new_tdel + deltak, 0.0)

    new_tdel = jnp.where(feasible, new_tdel, 0.0)
    new_qdel = jnp.where(feasible, new_qdel, 0.0)
    return new_tdel, new_qdel, jnp.float32(0.0)


def betts_miller_tendencies(temperature, specific_humidity, pfull, phalf, dt,
                            params: BettsMillerParameters):
    """Vectorized Betts-Miller tendencies over a column field.

    The leading axis is the vertical level; any trailing axes are the horizontal
    grid and are flattened, so this supports both the full ``(kx, ix, il)`` state
    and the column-vectorized ``(kx, ncols)`` state that ``ComposablePhysics``
    produces under ``vectorize_columns=True``.

    Args:
        temperature: [K], (kx, *horiz).
        specific_humidity: [kg/kg], (kx, *horiz).
        pfull: full-level pressure [Pa], (kx, *horiz).
        phalf: half-level pressure [Pa], (kx+1, *horiz).
        dt: timestep [s].
        params: static :class:`BettsMillerParameters`.

    Returns:
        (dtemp_dt, dq_dt, precip): tendencies [K/s], [kg/kg/s] of shape
        ``(kx, *horiz)`` and precipitation [kg/m^2/s] of shape ``horiz``.

    """
    kx = temperature.shape[0]
    horiz = temperature.shape[1:]
    n = int(np.prod(horiz)) if horiz else 1

    def to_cols(a):
        return a.reshape(a.shape[0], n).T            # (ncols, levels)

    t2 = to_cols(temperature)
    q2 = to_cols(specific_humidity)
    pf2 = to_cols(pfull)
    ph2 = to_cols(phalf)

    column = lambda t, q, pf, ph: betts_miller_column(t, q, pf, ph, dt, params)
    tdel, qdel, precip = jax.vmap(column)(t2, q2, pf2, ph2)

    dtemp_dt = tdel.T.reshape((kx,) + horiz) / dt
    dq_dt = qdel.T.reshape((kx,) + horiz) / dt
    precip_flux = precip.reshape(horiz) / dt
    return dtemp_dt, dq_dt, precip_flux
