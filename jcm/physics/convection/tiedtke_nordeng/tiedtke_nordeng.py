"""Tiedtke-Nordeng Mass-Flux Convection Scheme

This module implements the Tiedtke-Nordeng convection parameterization
in JAX, based on the ICON atmospheric model implementation.

The scheme includes:
- Deep convection with CAPE closure
- Shallow convection with moisture convergence closure  
- Mid-level convection
- Convective momentum transport
- Downdraft processes

References:
- Tiedtke, M. (1989): A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models. Mon. Weather Rev., 117, 1779-1800.
- Nordeng, T. E. (1994): Extended versions of the convective parametrization
  scheme at ECMWF and their impact on the mean and transient activity of the
  model in the tropics. ECMWF Tech. Memo. 206.

Date: 2025-01-09

"""

import jax.numpy as jnp
import jax
from jax import lax
from typing import Tuple

import jcm.constants as c
# Shared Tetens saturation thermodynamics (water+ice "auto" phase, as used
# throughout the ECHAM/Tiedtke path). Re-exported for backward compatibility.
from jcm.physics.convection.saturation import (  # noqa: F401
    cuadjtq_newton,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
)

# Import updraft, downdraft and flux modules after they're defined
# This avoids circular imports


# The scheme's data structures live in types.py; re-exported here so
# existing imports (updraft.py, downdraft.py, flux_tendencies.py, tests)
# keep working unchanged.
from jcm.physics.convection.tiedtke_nordeng.types import (  # noqa: F401
    ConvectionData,
    ConvectionParameters,
    ConvectionState,
    ConvectionTendencies,
)



def initialize_convection(temperature: jnp.ndarray,
                         humidity: jnp.ndarray,
                         pressure: jnp.ndarray,
                         u_wind: jnp.ndarray,
                         v_wind: jnp.ndarray,
                         config: ConvectionParameters) -> ConvectionState:
    """Initialize convection state variables
    
    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        config: Convection configuration
        
    Returns:
        Initial convection state

    """
    nlev = temperature.shape[0]
    
    # Initialize updraft properties with environmental values. Dtype follows
    # the inputs (not a hardcoded float32) so the scheme is correct whether the
    # model runs in float32 or float64 — both ``lax.cond`` branches must agree.
    tu = jnp.asarray(temperature)
    qu = jnp.asarray(humidity)
    lu = jnp.zeros_like(temperature)
    uu = jnp.asarray(u_wind)
    vu = jnp.asarray(v_wind)

    # Initialize downdraft properties.
    td = jnp.asarray(temperature)
    qd = jnp.asarray(humidity)
    ud = jnp.asarray(u_wind)
    vd = jnp.asarray(v_wind)

    # Initialize mass fluxes to zero.
    mfu = jnp.zeros_like(temperature)
    mfd = jnp.zeros_like(temperature)
    
    # Initialize convection diagnostics
    # int32 EXPLICITLY: under jax x64 (the JAM configuration) a bare
    # jnp.array(0) is int64, and the activation lax.cond then sees
    # int64 (inactive state) vs int32 (active branch) ktype — a
    # trace-time branch-type mismatch.
    ktype = jnp.array(0, dtype=jnp.int32)  # No convection initially
    kbase = jnp.array(nlev - 1)  # Surface level
    ktop = jnp.array(0)   # Top level
    
    # Initialize precipitation
    prate = jnp.array(0.0)
    
    return ConvectionState(
        tu=tu, qu=qu, lu=lu, uu=uu, vu=vu,
        td=td, qd=qd, ud=ud, vd=vd,
        mfu=mfu, mfd=mfd, entr=jnp.zeros_like(temperature),
        ktype=ktype, kbase=kbase, ktop=ktop,
        prate=prate
    )


def cloud_base_lift(config: ConvectionParameters,
                    thvsig: jnp.ndarray | None = None) -> jnp.ndarray:
    """ECHAM ``cubase`` sub-grid buoyancy excess ``zlift`` [K].

    ``zlift = MIN(MAX(cminbuoy, MIN(cmaxbuoy, thvsig·cbfac)), 1.0)``
    (mo_cuinitialize.f90:291, mo_cuascent.f90:444). It represents the
    thermal excess of the warmest boundary-layer plumes over the grid-mean
    parcel, and it is what allows a parcel to cross the thin
    negative-buoyancy layer between its LCL and its LFC.

    ``thvsig`` is ECHAM's ``pthvsig``: the standard deviation of virtual
    potential temperature at the second-lowest full level, produced by
    vdiff's PROGNOSTIC θ_v variance (``vdiff.f90:1338``). Passing it makes
    the convective trigger respond to what the boundary layer is actually
    doing — a well-mixed daytime layer carries a large σ(θ_v) and convects
    readily, a nocturnal stable layer carries almost none and does not. The
    scheme reads it off the ``vertical_diffusion`` diagnostic, which the
    TTE-TKE term publishes as ``thv_sigma``.

    ``config.cu_thvsig`` is the fallback for callers with no vdiff
    diagnostic — the column-mode tests and anyone driving the convection
    routine standalone. It is NOT the model path.

    Args:
        config: Convection configuration (cminbuoy / cmaxbuoy / cbfac, and
            the ``cu_thvsig`` fallback).
        thvsig: σ(θ_v) [K] from vdiff, shape ``(ncols,)`` or scalar. When
            ``None``, ``config.cu_thvsig`` is used.

    Returns:
        ``zlift`` [K], broadcasting against ``thvsig``.

    """
    sigma = config.cu_thvsig if thvsig is None else thvsig
    zlift = jnp.clip(sigma * config.cu_cbfac,
                     config.cu_cminbuoy, config.cu_cmaxbuoy)
    return jnp.minimum(zlift, 1.0)


def find_cloud_base(temperature: jnp.ndarray,
                   humidity: jnp.ndarray,
                   pressure: jnp.ndarray,
                   config: ConvectionParameters,
                   thvsig: jnp.ndarray | None = None,
                   layer_thickness: jnp.ndarray | None = None,
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find cloud base by ECHAM ``cubase``'s ``klab`` walk.

    Faithful port of ECHAM ``cubase`` (mo_cuinitialize.f90:276-320). The
    parcel starts at the lowest level with the environment's temperature and
    humidity (ECHAM seeds ``ptu = ptenh``, ``pqu = pqenh`` in ``cuini``) and
    is walked UPWARD one level at a time, conserving dry static energy::

        T_u(k) = ( cp·T_u(k+1) + geoh(k+1) - geoh(k) ) / cp

    At each level, in order:

    1. **Dry buoyancy gate.** ``zbuo = Tv_u - Tv_e + zlift``. If this is not
       positive the parcel could not have reached this level: ``klab`` falls
       to 0 and the column gets **no convection at all**. This is the
       sub-cloud test that was previously missing entirely.
    2. **Condensation** via the damped ``cuadjtq`` Newton step. If the parcel
       condenses (``pqu < zqold``) this is the LCL, ``klab`` becomes 2, and
       **the walk stops here** — ECHAM never looks higher.
    3. **Cloud-base test.** At that LCL only, with condensate loading:
       ``zbuo = T_u·(1 + vtmpc1·q_u - l_u) - Tv_e + zlift``. Cloud base
       exists iff this is positive.

    The consequence, which is ECHAM's and not an approximation of it: a
    column whose parcel is unbuoyant at its own LCL gets no convection,
    however thin the inhibition layer is. An earlier version of this function
    searched upward for the first level that was both condensing and buoyant
    (the LFC) — that is NOT what the reference does, and it let a plume start
    above a layer the parcel could never have crossed.

    ``zlift`` is the sub-grid thermal excess from vdiff's prognostic θ_v
    variance (ECHAM ``pthvsig``); see :func:`cloud_base_lift`. It is what
    covers roughly one layer of dry-adiabatic excess cooling, so in practice
    the trigger requires the LCL to be within about a layer of the surface —
    a moist, well-mixed boundary layer. That strictness is correct precisely
    because it is only half of ECHAM's trigger: elevated convection is
    ``cubasmc``, ported in :func:`find_midlevel_cloud_base` (#697). ECHAM can
    also re-seed above a ``cubase`` plume that dies partway up, which jcm
    cannot — it fixes one cloud base per column per step (#700).

    Remaining departure: ECHAM runs this walk on HALF levels (``ptenh`` /
    ``pqenh`` / ``pgeoh``, with ``cuadjtq`` at ``paphp1``). jcm's convection
    path is on full levels throughout, so the walk is evaluated there. That
    is the scheme-wide staggering gap #530, not something specific to this
    routine — the walk's logic is the reference's.

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        config: Convection configuration
        thvsig: σ(θ_v) [K] from vdiff (ECHAM ``pthvsig``). ``None`` falls
            back to ``config.cu_thvsig``.
        layer_thickness: Layer thickness (m) [nlev], used to build the
            geopotential the DSE-conserving lift needs. When ``None`` the
            lift falls back to the Exner form, which is equivalent for a
            constant ``cp`` and keeps older callers working.

    Returns:
        Tuple of (cloud_base_level, cloud_base_exists)

    """
    nlev = len(temperature)

    # Work surface-first so the walk runs in its natural direction; flip back
    # at the end. ``flip`` is a no-op when the input is already surface-first.
    is_surface_first = pressure[0] >= pressure[-1]
    flip = lambda a: jnp.where(is_surface_first, a, a[::-1])
    t_env = flip(temperature)
    q_env = flip(humidity)
    p_env = flip(pressure)

    # Geopotential of each level above the lowest one. ECHAM lifts the parcel
    # with ``(cp·T + geoh)`` conserved, so the walk needs a height coordinate;
    # ``layer_thickness`` gives it directly. Without it, fall back to the
    # equivalent Exner form.
    if layer_thickness is not None:
        dz = flip(layer_thickness)
        # Height of level k above level 0, integrating half a layer at each
        # end plus the full layers between.
        geo = c.grav * jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (dz[:-1] + dz[1:]))],
        )
        parcel_t_dry = t_env[0] + (geo[0] - geo) / c.cpd
    else:
        parcel_t_dry = t_env[0] * (p_env / p_env[0]) ** (c.rd / c.cpd)

    q_parcel = q_env[0]

    # Condense at every level (cheap, and the walk needs the result anyway).
    parcel_t, parcel_q, parcel_l = cuadjtq_newton(
        parcel_t_dry, jnp.broadcast_to(q_parcel, parcel_t_dry.shape), p_env,
    )
    condenses = parcel_l > 0.0            # ECHAM ``pqu(jk) < zqold(jk)``

    zlift = cloud_base_lift(config, thvsig)
    tv_env = t_env * (1.0 + c.vtmpc1 * q_env)

    # (1) sub-cloud dry buoyancy — the parcel still carries all its water.
    buoy_dry = parcel_t_dry * (1.0 + c.vtmpc1 * q_parcel) - tv_env + zlift
    # (3) cloud-base buoyancy with condensate loading.
    buoy_moist = (
        parcel_t * (1.0 + c.vtmpc1 * parcel_q - parcel_l) - tv_env + zlift
    )

    levels = jnp.arange(nlev)
    # ECHAM sets klab(klev)=1 unconditionally and starts testing at klevm1,
    # so level 0 is sub-cloud by definition and never a cloud base.
    # A level is REACHABLE only if every level strictly below it (above 0)
    # was both non-condensing and dry-buoyant — the walk would otherwise have
    # stopped there. ``cumprod`` of the per-level "keep walking" flag,
    # shifted by one, is exactly that.
    keeps_walking = jnp.logical_and(~condenses, buoy_dry > 0.0)
    keeps_walking = keeps_walking.at[0].set(True)      # klab(klev) = 1
    reachable = jnp.concatenate(
        [jnp.ones(1, dtype=bool), jnp.cumprod(keeps_walking.astype(jnp.int32))[:-1] > 0],
    )

    # The LCL is the lowest reachable level that condenses; ECHAM stops there.
    lcl_mask = jnp.logical_and(
        jnp.logical_and(reachable, condenses), levels > 0,
    )
    has_lcl = jnp.any(lcl_mask)
    lcl_sf = jnp.argmax(lcl_mask)          # first True, surface-first

    # Cloud base exists iff the parcel is buoyant AT that LCL.
    cloud_base_found = jnp.logical_and(
        jnp.logical_and(has_lcl, buoy_moist[lcl_sf] > 0.0),
        lcl_sf < nlev - 1,
    )

    # Back to the caller's ordering.
    cloud_base_level = jnp.where(
        is_surface_first, lcl_sf, nlev - 1 - lcl_sf,
    )
    cloud_base_level = jnp.where(
        cloud_base_found, cloud_base_level, nlev - 1,
    )
    return cloud_base_level, cloud_base_found


def find_midlevel_cloud_base(temperature: jnp.ndarray,
                             humidity: jnp.ndarray,
                             pressure: jnp.ndarray,
                             omega: jnp.ndarray,
                             layer_thickness: jnp.ndarray,
                             config: ConvectionParameters,
                             thvsig: jnp.ndarray | None = None,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ECHAM ``cubasmc``: the MID-LEVEL convection trigger.

    Faithful port of ECHAM ``cubasmc`` (mo_cuascent.f90:593-676), the
    second of the reference's two ways to start a plume. Where
    :func:`find_cloud_base` (``cubase``) lifts a *surface* parcel and so
    only fires for a moist, well-mixed boundary layer, this one starts a
    plume at a level with **no surface connection at all**::

        .NOT.ldcum .AND. klab(kk+1) == 0
                   .AND. pqen(kk)   >  0.90·pqsen(kk)
                   .AND. pverv(kk)  <  0
                   .AND. pgeoh(kk)/grav > 1500 m

    — i.e. the environment there is within 10 % of saturation, resolved-scale
    ascent is lifting it, and it is above the boundary layer. ECHAM calls
    this from inside the ascent loop at every level between ``nmctop`` (the
    300 hPa level) and ``klevm1``, so it is evaluated bottom-up and the
    LOWEST qualifying level wins. That is what the mask-plus-``argmax``
    below reproduces.

    **The plume is seeded from the environment, not from the surface.**
    ``pqu = pqen(kk)``, ``plu = 0``, and ``ptu`` is the environmental
    temperature brought adiabatically to the layer's bottom interface. Its
    cloud-base mass flux is the resolved ascent itself,
    ``clip(-omega/g, cmfcmin, cmfcmax)`` — *not* a CAPE or moisture-budget
    closure, and ECHAM never rescales it (the Nordeng CAPE rescale is gated
    on ``ktype == 1``, mo_cumastr.f90:898).

    **Retry-upward.** ECHAM does not simply take the lowest qualifying
    level: if the seeded plume is not buoyant at its first ascent step the
    ascent sets ``klab = 0`` there, and the next loop iteration lets
    ``cubasmc`` seed one level higher. The net rule is therefore *the lowest
    qualifying level whose plume survives its first step*, which is what the
    ``survives`` term encodes — one DSE-conserving lift to the next level, a
    ``cuadjtq`` adjustment, and the buoyancy test. This is also the one site
    where the ascent ``zlift`` bonus legitimately applies: ``cubasmc`` sets
    ``klab(kk+1) = 1``, and mo_cuascent.f90:449 adds ``zlift`` exactly when
    the level below is still ``klab == 1`` (see #691, which removed it from
    the ``cubase`` path where ``klab(kcbot) = 2`` makes it unreachable).

    Remaining departure, shared with ``cubase``: ECHAM re-seeds above a
    mid-level plume that took hold and then died several levels up, because
    its cloud base is a per-level quantity inside the ascent loop. jcm picks
    one cloud base per column before the scan, so a plume that survives its
    first step and dies later is simply that column's convection. Tracked as
    #700, together with the half-level staggering (#530) and the discrete
    level picks (#665).

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        omega: Pressure vertical velocity Dp/Dt (Pa/s) [nlev], negative
            upward. This is ECHAM's ``pverv``; jcm takes it from the
            dycore's ``omega`` physics field. A zero profile (no provider)
            leaves the trigger permanently off, which is the correct
            physics for a column with no resolved ascent.
        layer_thickness: Layer thickness (m) [nlev]
        config: Convection configuration
        thvsig: σ(θ_v) [K] from vdiff, for the ``zlift`` in the survival
            test. ``None`` falls back to ``config.cu_thvsig``.

    Returns:
        Tuple of (mid_level_base, mid_level_base_exists)

    """
    nlev = len(temperature)

    # Work surface-first, as in ``find_cloud_base``; flip back at the end.
    is_surface_first = pressure[0] >= pressure[-1]
    flip = lambda a: jnp.where(is_surface_first, a, a[::-1])
    t_env = flip(temperature)
    q_env = flip(humidity)
    p_env = flip(pressure)
    w_env = flip(omega)
    dz = flip(layer_thickness)

    qs_env = jax.vmap(saturation_mixing_ratio)(p_env, t_env)
    # ECHAM ``pgeoh(kk)/grav``: the height of the candidate layer's TOP
    # interface above the surface (ECHAM's ``pgeom1`` is geopotential above
    # the surface, so orography is already subtracted).
    z_top = jnp.cumsum(dz)

    levels = jnp.arange(nlev)
    eligible = (
        (q_env > config.cu_midlev_rh * qs_env)
        & (w_env < 0.0)
        & (z_top > config.cu_midlev_zmin)
        # ECHAM ``ik < klevm1``: the lowest two full levels are ``cubase``'s.
        & (levels >= 2)
        # ECHAM ``ik > nmctop``: no mid-level base at or above 300 hPa.
        # ECHAM fixes ``nmctop`` once from a reference 101320 Pa surface
        # pressure; evaluating the same cut on the live column instead makes
        # it independent of resolution and of surface pressure, which is the
        # quantity the criterion is really about.
        & (p_env > config.cu_midlev_ptop)
        & jnp.asarray(config.cu_lmfmid, dtype=bool)
    )

    # Survival of a seed at level k: one DSE-conserving lift to level k+1,
    # the damped Newton adjustment there, then the buoyancy test WITH the
    # ``zlift`` bonus (klab == 1 below, mo_cuascent.f90:449).
    dz_mid = 0.5 * (dz[:-1] + dz[1:])
    parcel_t_dry = t_env[:-1] - c.grav * dz_mid / c.cpd
    parcel_t, parcel_q, parcel_l = cuadjtq_newton(
        parcel_t_dry, q_env[:-1], p_env[1:],
    )
    zlift = cloud_base_lift(config, thvsig)
    buoy = (
        parcel_t * (1.0 + c.vtmpc1 * parcel_q - parcel_l)
        - t_env[1:] * (1.0 + c.vtmpc1 * q_env[1:])
        + zlift
    )
    survives = jnp.concatenate([
        (parcel_l > 0.0) & (buoy > 0.0),
        jnp.zeros(1, dtype=bool),      # the top level has nowhere to rise to
    ])

    ok = eligible & survives
    found = jnp.any(ok)
    base_sf = jnp.argmax(ok)           # lowest qualifying level, surface-first

    base_level = jnp.where(is_surface_first, base_sf, nlev - 1 - base_sf)
    base_level = jnp.where(found, base_level, nlev - 1)
    return base_level, found


def midlevel_mass_flux(omega_at_base: jnp.ndarray,
                       config: ConvectionParameters) -> jnp.ndarray:
    """ECHAM ``cubasmc`` cloud-base mass flux for mid-level convection.

    ``zzzmb = MIN(cmfcmax, MAX(cmfcmin, -pverv/grav))``
    (mo_cuascent.f90:643-645). The mid-level plume is driven by the
    resolved ascent that triggered it, so its amplitude is that ascent
    expressed as a mass flux — there is no CAPE or moisture-budget closure
    here, and ECHAM leaves this value alone (the Nordeng rescale is
    ``ktype == 1`` only).
    """
    return jnp.clip(-omega_at_base / c.grav, config.cmfcmin, config.cmfcmax)


def calculate_cape_cin(temperature: jnp.ndarray,
                      humidity: jnp.ndarray,
                      pressure: jnp.ndarray,
                      layer_thickness: jnp.ndarray,
                      cloud_base: int,
                      config: ConvectionParameters) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate CAPE and CIN for convective instability.

    Lifts a surface parcel: dry-adiabatic up to cloud base, moist-
    adiabatic above. CAPE is the positive-buoyancy work done between the
    LFC (level of free convection) and the EL (equilibrium level); CIN
    is the magnitude of negative-buoyancy work between cloud base and
    LFC. Stratospheric layers above the EL — where the parcel cools to
    absurdly low temperatures and gives massive bogus negative buoyancy
    — are not counted.

    Works for either input ordering:
      * TOA-first (level 0 = TOA, ICON/ECHAM convention used by the
        running model)
      * surface-first (level 0 = surface, used by some unit tests)
    Internally we always work in surface-first form so the moist-adiabat
    scan steps in the natural surface→TOA direction. The output is in
    the input's ordering for backward compatibility.

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        cloud_base: Cloud base level index in the input ordering
        config: Convection configuration

    Returns:
        Tuple of (CAPE, CIN) in J/kg

    """
    nlev = len(temperature)

    # Reorder to surface-first: level 0 = surface, level nlev-1 = TOA.
    # ``flip`` is a no-op when the input is already surface-first.
    is_surface_first = pressure[0] >= pressure[-1]
    flip = lambda a: jnp.where(is_surface_first, a, a[::-1])
    T_sf = flip(temperature)
    q_sf = flip(humidity)
    p_sf = flip(pressure)
    dz_sf = flip(layer_thickness)
    cb_sf = jnp.where(is_surface_first, cloud_base, nlev - 1 - cloud_base)

    surf_temp = T_sf[0]
    surf_humid = q_sf[0]
    surf_press = p_sf[0]
    k_levels = jnp.arange(nlev)

    # Below cloud base — dry-adiabatic ascent. q is conserved, so the
    # parcel mixing ratio stays at the surface value.
    parcel_temp_dry = surf_temp * (p_sf / surf_press) ** (c.rd / c.cpd)

    # Above cloud base — moist (pseudoadiabatic) ascent. We scan
    # surface→TOA (increasing index in surface-first) and only step the
    # parcel temperature when we are AT or above cloud base; below cb
    # the parcel just rides the dry adiabat we already computed.
    #
    # If the parcel arrives at cb already supersaturated (surf_q >
    # qsat(parcel_dry_T, p_cb) — common when find_cloud_base picks the
    # next discrete level above the true LCL), condense the excess and
    # warm the parcel by L/cp times the condensate. This raises the
    # cloud-base parcel temperature to its physically meaningful value
    # and prevents spurious cold biases that crush CAPE for warm
    # tropical columns.
    #
    # The condensation must go through the damped ``cuadjtq`` Newton step
    # (same routine the updraft uses), not an undamped ``L/cp·(q - qs(T))``:
    # condensing warms the parcel, which raises qs, so the undamped form
    # over-condenses and over-warms — +6.5 K instead of +2.0 K for a
    # 290 K / 16 g/kg parcel at 900 hPa, inflating CAPE and pushing
    # columns across the deep/shallow type threshold (issue #661).
    parcel_temp_at_cb_dry = parcel_temp_dry[cb_sf]
    p_cb = p_sf[cb_sf]
    cloud_base_temp, _, _ = cuadjtq_newton(
        parcel_temp_at_cb_dry, surf_humid, p_cb,
    )

    def _step(parcel_t, args):
        p_curr, p_next, k = args
        dp = p_next - p_curr  # negative going up
        qs = saturation_mixing_ratio(p_curr, parcel_t)
        dTdp = (1.0 / p_curr) * (c.rd * parcel_t + c.alhc * qs) / (
            c.cpd + c.alhc ** 2 * qs / (c.rv * parcel_t ** 2)
        )
        new_t = parcel_t + dTdp * dp
        # If we haven't reached cloud base yet, hold the parcel at the
        # cloud-base temperature so the moist integration starts from
        # the right pressure when k finally crosses cb.
        below_cb = k < cb_sf
        new_t = jnp.where(below_cb, cloud_base_temp, new_t)
        return new_t, new_t

    p_pairs = (p_sf[:-1], p_sf[1:], k_levels[:-1])
    _, parcel_after = lax.scan(_step, cloud_base_temp, p_pairs)
    parcel_temp_moist_sf = jnp.concatenate(
        [cloud_base_temp[jnp.newaxis], parcel_after],
    )

    is_above_cb = k_levels >= cb_sf
    parcel_temp_sf = jnp.where(
        is_above_cb, parcel_temp_moist_sf, parcel_temp_dry,
    )
    parcel_qs_sf = jax.vmap(saturation_mixing_ratio)(p_sf, parcel_temp_sf)
    # On the pseudoadiabat the parcel carries its saturation value, but it can
    # never hold *more* vapour than the total water it lifted off the surface
    # with (condensate is precipitated out, so parcel total water only ever
    # decreases). Without the cap, a cloud-base level at which the parcel is
    # still marginally subsaturated is credited with invented moisture and an
    # unearned virtual-temperature buoyancy — the CAPE-side face of #661.
    parcel_q_sf = jnp.where(
        is_above_cb, jnp.minimum(parcel_qs_sf, surf_humid), surf_humid,
    )

    env_tv_sf = T_sf * (1.0 + 0.61 * q_sf)
    parcel_tv_sf = parcel_temp_sf * (1.0 + 0.61 * parcel_q_sf)
    buoyancy_sf = c.grav * (parcel_tv_sf - env_tv_sf) / env_tv_sf

    # LFC: lowest-altitude (smallest surface-first index) at-or-above cb
    #      where buoyancy first becomes positive.
    pos_above_cb = (buoyancy_sf > 0) & is_above_cb
    has_lfc = jnp.any(pos_above_cb)
    # ``argmax`` returns the FIRST True; surface-first means lowest-
    # altitude True is the smallest index, which is what we want.
    lfc_sf = jnp.argmax(pos_above_cb)
    lfc_sf = jnp.where(has_lfc, lfc_sf, nlev)

    # EL: first level above LFC (larger index in surface-first) where
    # buoyancy turns non-positive again. Without an EL we integrate to
    # TOA (k = nlev - 1).
    above_lfc_mask = k_levels > lfc_sf
    el_candidate_mask = above_lfc_mask & ~(buoyancy_sf > 0)
    has_el = jnp.any(el_candidate_mask)
    el_sf = jnp.argmax(el_candidate_mask)
    el_sf = jnp.where(has_el, el_sf, nlev)

    in_cape_layer = (k_levels >= lfc_sf) & (k_levels < el_sf)
    in_cin_layer = is_above_cb & (k_levels < lfc_sf)

    cape_contrib = jnp.where(
        in_cape_layer & (buoyancy_sf > 0), buoyancy_sf * dz_sf, 0.0,
    )
    cin_contrib = jnp.where(
        in_cin_layer & (buoyancy_sf <= 0), -buoyancy_sf * dz_sf, 0.0,
    )

    cape = jnp.where(has_lfc, jnp.sum(cape_contrib), 0.0)
    cin = jnp.where(has_lfc, jnp.sum(cin_contrib), 0.0)

    return cape, cin


def cloud_depth_for_target_top(
    pressure: jnp.ndarray,
    cloud_base: jnp.ndarray,
    target_top_pa: float,
    min_layers: int = 2,
) -> jnp.ndarray:
    """Return the number of model levels between ``cloud_base`` and the
    level closest to ``target_top_pa`` from above — used as the updraft
    scan ceiling.

    The scan ceiling is a *maximum* depth the updraft is allowed to
    extend to, NOT the actual cloud top. The actual termination is
    decided dynamically inside ``calculate_updraft`` (negative
    buoyancy, or mfu < 1 % of mfb). ``cloud_depth`` only needs to give
    the scan enough headroom to reach physically plausible cloud tops;
    too small a value silently truncates real convection, too large
    just wastes compute on levels that would terminate dynamically
    anyway.

    A fixed level-count value would be vertical-resolution-dependent in
    surprising ways:

    * On the 47-level ICON hybrid grid we run T85×L47 on, layers are
      ~22 hPa thick in the mid-troposphere; ``cloud_depth=35`` ≈ a
      surface-to-200-hPa scan range.
    * On a coarser 8-level sigma grid (used in some bisection tests),
      ``cloud_depth=35`` would be silently clamped to ``nlev-2`` —
      the cloud is allowed to reach the model top, which both wastes
      compute and risks unphysical extension into the stratosphere.
    * On a 90-level grid, the same ``35`` would only let the cloud
      reach ~700 hPa, cutting off real deep convection.

    Deriving from a target *pressure* makes the value
    resolution-independent. Recommended targets:

    * Deep convection: 15000 Pa (150 hPa) — tropical Cb tops typically
      reach the tropopause around this pressure.
    * Shallow convection: 70000 Pa (700 hPa) — trade-cumulus cloud
      tops at ~3 km.

    Implementation: for any pressure index ordering, find the level
    closest to ``target_top_pa`` from above (i.e. the level with the
    HIGHEST pressure among levels whose pressure ≤ target). That's
    the level we want the scan to reach. ``cloud_depth`` is then the
    integer index distance ``|cloud_base - target_top_idx|``. The
    result is clipped to ``[min_layers, nlev-2]`` so the scan always
    has at least ``min_layers`` levels of headroom and stops short of
    TOA.

    Args:
        pressure: Full-level pressure profile (Pa) [nlev]
        cloud_base: Cloud-base level index (0-indexed)
        target_top_pa: Scan should reach (at least) this pressure level
        min_layers: Minimum scan depth (≥ 2 to avoid degenerate scans)

    Returns:
        Scan-ceiling depth in *levels* (int32), clipped to
        ``[min_layers, nlev-2]``.

    """
    nlev = pressure.shape[0]
    above_target = pressure <= target_top_pa
    # Among levels at or above target, pick the HIGHEST-pressure one —
    # that's the level closest to ``target_top_pa`` from above, where we
    # want the scan to reach. ``argmax`` of ``-inf`` outside the mask
    # returns 0 if no level is above target (clipped to ``min_layers``
    # so it doesn't matter for the result).
    masked_p = jnp.where(
        above_target,
        pressure,
        jnp.array(-jnp.inf, dtype=pressure.dtype),
    )
    target_top_idx = jnp.argmax(masked_p)
    depth = jnp.abs(cloud_base.astype(jnp.int32) - target_top_idx.astype(jnp.int32))
    return jnp.clip(depth, min_layers, nlev - 2)


# Minimum boundary-layer moisture supply [kg/m²/s] that counts as a real
# evaporation anchor for the deep mass-flux closure. Below this (cold start,
# numerical noise, a stack with no surface term) the moisture-budget flux is
# meaningless, so both the activation trigger AND the closure-selection gate fall
# back to the bare-CAPE closure. The two MUST use the same threshold: gating
# activation on CAPE but then forcing a ~zero moisture flux would disable
# convection while CAPE accumulates.
_MIN_MOISTURE_SUPPLY = 1.0e-7

# Minimum CAPE [J/kg] for the moisture-supply activation branch — a buoyancy
# gate standing in for ECHAM's ``ldcum`` (a genuinely buoyant updraft must
# exist). ``find_cloud_base`` returns the lifting-condensation level, which is
# present in many *statically stable* columns where a lifted parcel saturates
# but never becomes buoyant (CAPE == 0, no level of free convection). Without
# this gate the moisture-supply trigger fired deep convection — at the
# moisture-closure mass flux, up to the CFL cap — in every column with surface
# evaporation, including stable subsiding ones. On a T63L47 real-orography
# spin-up that over-activation dumped latent heat into stable tropical columns
# and ran the temperature away to NaN within ~4 days (the cloud-base flicker
# fix #529 regressed whole-model stability). Requiring a small positive CAPE
# excludes the no-buoyancy columns while staying far below the CAPE that an RCE
# column maintains while convecting continuously, so the anti-flicker behaviour
# is preserved. Strong-CAPE columns (CAPE > 100) trigger regardless.
_MIN_CAPE_FOR_MOISTURE_TRIGGER = 10.0


def tiedtke_nordeng_convection(
    temperature: jnp.ndarray,
    humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    rho: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    dt: float,
    config: ConvectionParameters = None,
    land_fraction: jnp.ndarray = jnp.array(0.0),
    moisture_supply: jnp.ndarray = jnp.array(0.0),
    moisture_tend_profile: jnp.ndarray | None = None,
    thvsig: jnp.ndarray | None = None,
    omega: jnp.ndarray | None = None,
    qte_dynamics: jnp.ndarray | None = None,
) -> Tuple[ConvectionTendencies, ConvectionState]:
    """Run Tiedtke-Nordeng convection scheme with fixed qc/qi transport

    Args:
        temperature: Environmental temperature (K) [nlev]
        humidity: Environmental specific humidity (kg/kg) [nlev]
        pressure: Environmental pressure (Pa) [nlev]
        layer_thickness: Layer thickness (m) [nlev]
        rho: Air density (kg/m³) [nlev]
        u_wind: Zonal wind (m/s) [nlev]
        v_wind: Meridional wind (m/s) [nlev]
        qc: Cloud water mixing ratio (kg/kg) [nlev]
        qi: Cloud ice mixing ratio (kg/kg) [nlev]
        dt: Time step (s)
        config: Convection configuration
        land_fraction: Fraction of column underlying land (0=ocean, 1=land).
            Selects ECHAM's per-surface ``zdnoprc`` precip-zone threshold
            via ``config.cu_dnoprc_ocean`` / ``config.cu_dnoprc_land``.
            Defaults to 0 (ocean).
        moisture_supply: Boundary-layer moisture supply rate [kg/m²/s] — the
            surface evaporation feeding the subcloud layer. Anchors the deep
            cloud-base mass flux to ECHAM's moisture-budget closure
            (``zmfub`` ≈ E/(q_u−q_e), mo_cumastr.f90). Defaults to 0, which
            falls back to the pure-CAPE closure (cold start / no surface term).
        thvsig: σ(θ_v) [K] from vdiff (ECHAM ``pthvsig``), setting the
            cloud-base ``zlift``. ``None`` falls back to ``config.cu_thvsig``.
        omega: Pressure vertical velocity [Pa/s] [nlev], negative upward
            (ECHAM ``pverv``). Drives the ``cubasmc`` mid-level trigger.
            ``None`` means no resolved ascent is known, so that trigger
            stays off and only the surface-parcel ``cubase`` path can fire.
        qte_dynamics: The DYNAMICS moisture tendency of the just-completed
            dycore step [kg/kg/s] [nlev] — advection plus hyperdiffusion,
            reconstructed one step lagged by the wrapper (see there).
            Together with ``moisture_tend_profile`` (the same-step vdiff
            part) it forms ECHAM's ``pqte``, whose column integral
            ``zdqcv`` drives the deep/shallow split. ``None`` (no host
            information) means the split sees no large-scale convergence
            and classifies by the surface budget alone.

    Returns:
        Tuple of (tendencies, final_state) with fixed qc/qi transport

    """
    if config is None:
        config = ConvectionParameters.default()
    
    nlev = len(temperature)
    
    # Initialize state
    state = initialize_convection(
        temperature, humidity, pressure, 
        u_wind, v_wind, config
    )
    
    # --- The two ECHAM plume triggers ------------------------------------
    # 1. ``cubase``: lift a surface parcel (mo_cuinitialize.f90:276).
    # 2. ``cubasmc``: seed from the environment at a nearly-saturated,
    #    rising mid-tropospheric level (mo_cuascent.f90:593).
    # ECHAM runs the second only where the first produced nothing
    # (``klab(kk+1) == 0`` everywhere in a column with no surface plume), so
    # the surface path always wins when both would fire.
    cloud_base_sfc, has_cloud_base_sfc = find_cloud_base(
        temperature, humidity, pressure, config, thvsig, layer_thickness,
    )
    if omega is None:
        omega = jnp.zeros_like(temperature)
    midlev_base, has_midlev_base = find_midlevel_cloud_base(
        temperature, humidity, pressure, omega, layer_thickness, config,
        thvsig,
    )
    use_midlev = jnp.logical_and(~has_cloud_base_sfc, has_midlev_base)
    cloud_base = jnp.where(use_midlev, midlev_base, cloud_base_sfc)
    has_cloud_base = jnp.logical_or(has_cloud_base_sfc, use_midlev)

    # ECHAM zdqpbl closure supply (mo_cumastr.f90:534-545): the moisture-
    # budget first guess integrates the PRE-CONVECTION moisture tendency
    # pqte over the levels at/below cloud base. When the host provides
    # ``moisture_tend_profile`` — the SAME-STEP vdiff moisture tendency,
    # available because the term ordering runs vdiff before convection
    # (ECHAM physc) — the supply becomes max(0, Σ_{k>=kcbot} pqte·ρ·Δz);
    # the scalar ``moisture_supply`` (surface evaporation) remains the
    # floor so a vdiff-free column keeps the #529 continuous-convection
    # anchor. Same-step supply is self-limiting: convection consumes what
    # vdiff delivered this step, so the closure cannot ramp across steps
    # (the earlier one-step-lagged total-Δq form compounded through the
    # convergence feedback and blew up — onset7 analysis). The advective
    # part of ECHAM's pqte is deliberately absent here; see the wrapper
    # for the reasoning and the follow-up note.
    # The bare surface evaporation, saved BEFORE the zdqpbl floor below:
    # ECHAM's deep/shallow test compares column moisture convergence with
    # 1.1x the SURFACE latent flux (``pqhfla``), not with the closure supply.
    surface_evap = jnp.maximum(moisture_supply, 0.0)
    if moisture_tend_profile is not None:
        below_base = jnp.arange(nlev) >= cloud_base
        zdqpbl = jnp.sum(
            jnp.where(below_base, moisture_tend_profile * rho * layer_thickness, 0.0)
        )
        moisture_supply = jnp.maximum(moisture_supply, zdqpbl)

    
    # CAPE/CIN of the SURFACE parcel. Meaningful only for the ``cubase``
    # path — a mid-level plume has no surface connection, and ECHAM
    # correspondingly never applies a CAPE closure or the Nordeng rescale to
    # ``ktype == 3`` (mo_cumastr.f90:898 gates both on ``ktype == 1``).
    cape, cin = lax.cond(
        has_cloud_base_sfc,
        lambda: calculate_cape_cin(temperature, humidity, pressure, layer_thickness,
                                 cloud_base_sfc, config),
        lambda: (jnp.array(0.0), jnp.array(0.0))
    )

    # Convection type: 0 = none, 1 = deep, 2 = shallow, 3 = mid-level.
    #
    # ``ktype`` follows ECHAM's *trigger*, not a diagnosis of the column:
    #
    #   * ``cubasmc`` sets ``ktype = 3`` outright (mo_cuascent.f90:655) —
    #     mid-level means "this plume was started with no surface
    #     connection", nothing else (#697).
    #   * ``cubase`` plumes are 1 or 2, split by the moisture-convergence
    #     test below (mo_cumastr.f90:572, #699), then demoted 1 -> 2 if
    #     the realized cloud is shallower than 200 hPa (line 752; see the
    #     demotion in ``apply_full_convection``).
    #
    # Two non-ECHAM proxies used to live here and are gone: an RH-based
    # relabel of moist-troposphere surface plumes as "mid-level" (a stand-in
    # for the missing ``cubasmc`` that handed them ``entrmid`` = 1e-4 /m,
    # 30x less entraining than the ``entrscv`` they get as shallow), and a
    # CAPE sigmoid at 1000 J/kg standing in for the moisture-convergence
    # test. Both mislabels selected the wrong entrainment for exactly the
    # regimes that matter (#699 records the measured consequences).
    # --- Smooth trigger and type selection (maintainability review B.2.2) --
    # The hard ``lax.cond(cape > 100)`` activation and the deep/shallow/mid
    # ``lax.switch`` made every convection parameter's gradient exactly zero
    # in non-convecting columns and discontinuous at the thresholds. Both
    # gates are now sigmoids: the trigger weight scales the closure mass
    # flux (a "fuzzy convective trigger"), and the type weights blend the
    # per-type entrainment rates and closures. Widths are differentiable
    # parameters; width → 0 recovers the hard scheme exactly.
    #
    # Activation semantics preserved from the #529/#535 flicker fixes:
    # trigger = [cape above the main threshold] OR [cape above a small
    # buoyancy floor AND the surface supplies moisture]. The supply gate
    # (1e-7 kg/m²/s) is a guard against a structurally absent surface
    # term, not a tunable — it stays hard.
    # Rescaled sigmoid: exactly 0 at cape = 0 (no phantom convection in
    # stable columns — a bare sigmoid leaves ~2% of the closure flux on
    # at zero CAPE), rising smoothly through ~0.5 at the threshold and
    # saturating to 1. Gradients are nonzero for any cape > 0.
    def _trigger_sigmoid(cape_v, threshold, width):
        s0 = jax.nn.sigmoid(-threshold / width)
        raw = jax.nn.sigmoid((cape_v - threshold) / width)
        return jnp.maximum((raw - s0) / (1.0 - s0), 0.0)

    w_cape_main = _trigger_sigmoid(
        cape, config.trigger_cape, config.smooth_trigger_j
    )
    w_cape_floor = _trigger_sigmoid(
        cape, _MIN_CAPE_FOR_MOISTURE_TRIGGER, config.smooth_trigger_j
    )
    supply_ok = (moisture_supply > _MIN_MOISTURE_SUPPLY).astype(cape.dtype)
    trigger_weight = jnp.maximum(w_cape_main, w_cape_floor * supply_ok)
    # ``cubasmc`` has no CAPE gate at all: the trigger conditions ARE the
    # activation (mo_cuascent.f90:631-634), so a mid-level column enters
    # with full weight and its own omega-derived mass flux.
    trigger_weight = jnp.where(use_midlev, jnp.ones_like(trigger_weight),
                               trigger_weight)

    # Type weights (deep, shallow, mid).
    #
    # Deep vs shallow is ECHAM's MOISTURE-CONVERGENCE test
    # (mo_cumastr.f90:565-574), not a CAPE threshold:
    #
    #     zdqcv  = SUM_k pqte(k)*dp(k)          [column integral]
    #     zhelp  = MAX(0, -1.1*pqhfla*grav)     [1.1x surface latent flux]
    #     ktype  = 1 (deep)  iff  zdqcv > zhelp  else 2 (shallow)
    #
    # i.e. deep convection requires large-scale moisture convergence beyond
    # ~10 % of what the surface is already evaporating; a column fed only
    # by its own surface flux is shallow. ``pqte`` here is the vdiff part
    # (same-step) plus the dynamics part (one-step-lagged reconstruction,
    # see the wrapper), matching ECHAM's leapfrog information structure.
    # With no dynamics info at all (single-column tests) the integral of
    # the vdiff tendency is exactly E, so ``zdqcv - zhelp = -0.1*E`` and
    # every surface-triggered column is shallow — which is what ECHAM
    # gives a convergence-free column too.
    #
    # This replaced a CAPE sigmoid at 1000 J/kg with no ECHAM counterpart;
    # the two disagreed systematically in exactly the interesting regimes
    # (high-CAPE non-convergent -> ECHAM shallow, low-CAPE frontal/ITCZ
    # convergence -> ECHAM deep), and the 30x entrpen/entrscv entrainment
    # gap made the mislabel expensive (#699). The width keeps the switch
    # hard at flux scales while differentiable.
    #
    # The mid-level weight is one exactly when ``cubasmc`` fired and zero
    # otherwise — a trigger identity, not a blendable diagnosis.
    pqte = (moisture_tend_profile
            if moisture_tend_profile is not None
            else jnp.zeros_like(temperature))
    if qte_dynamics is not None:
        pqte = pqte + qte_dynamics
    zdqcv = jnp.sum(pqte * rho * layer_thickness)
    zhelp = 1.1 * surface_evap
    # ECHAM's FSEL(zhelp - zdqcv, 2, 1) resolves the tie zdqcv == zhelp to
    # SHALLOW (FSEL takes the first branch at >= 0). A bare sigmoid gives
    # 0.5 there — which matters for the zero-information column (no supply,
    # no convergence: 0 vs 0) — so shift by one width to keep the tie on
    # ECHAM's side while leaving the switch differentiable.
    w_deep = jax.nn.sigmoid(
        (zdqcv - zhelp) / config.cu_dqcv_width - 1.0)
    w_mid = use_midlev.astype(cape.dtype)
    type_weights = jnp.stack([
        (1.0 - w_mid) * w_deep,                   # deep
        (1.0 - w_mid) * (1.0 - w_deep),           # shallow
        w_mid,                                    # mid
    ])

    # Discrete diagnostic ktype (consumed by the Sundqvist guard and the
    # cloud-depth ceiling): argmax of the type weights when active. The
    # ceiling is a scan bound, not the physical cloud top (dynamic
    # termination governs), so keeping it discrete costs no gradients
    # that matter.
    convection_active = jnp.logical_and(has_cloud_base, trigger_weight > 1e-3)
    conv_type = jnp.where(
        convection_active,
        jnp.argmax(type_weights) + 1,
        0,
    ).astype(jnp.int32)

    # Initialize tendencies to zero. Dtype follows the inputs so the scheme is
    # float-structure agnostic (float32 or float64) and both ``lax.cond``
    # branches agree on output types.
    dtedt = jnp.zeros_like(temperature)
    dqdt = jnp.zeros_like(humidity)
    dudt = jnp.zeros_like(u_wind)
    dvdt = jnp.zeros_like(v_wind)
    qc_conv = jnp.zeros_like(temperature)
    qi_conv = jnp.zeros_like(temperature)
    precip_conv = jnp.zeros((), dtype=temperature.dtype)
    
    # Import modules here to avoid circular imports
    from .updraft import calculate_updraft
    from .downdraft import calculate_downdraft
    from .flux_tendencies import (
        calculate_tendencies, mass_flux_closure_blend
    )
    
    # Apply full convection scheme if active (with tracer transport)
    def apply_full_convection():
        # Cloud-top scan ceiling. The ceiling is a *maximum* depth, not
        # the actual cloud top — actual termination is decided
        # dynamically inside ``calculate_updraft`` (negative buoyancy or
        # mfu < 1 % of mfb). Derive the ceiling from a target cloud-top
        # PRESSURE rather than a fixed level count so the value is
        # vertical-resolution-independent. Targets:
        #   * Deep:    150 hPa (tropical Cb tops near the tropopause)
        #   * Shallow: 700 hPa (trade-cumulus tops at ~3 km)
        # See ``cloud_depth_for_target_top`` for the derivation and a
        # detailed discussion of why a fixed level count is wrong.
        cloud_depth = lax.cond(
            conv_type == 2,
            lambda: cloud_depth_for_target_top(pressure, cloud_base, 70_000.0),
            lambda: cloud_depth_for_target_top(pressure, cloud_base, 15_000.0),
        )

        # Handle level ordering properly
        pressure_increasing = pressure[0] < pressure[-1]

        # Ensure cloud depth is at least 2 levels and doesn't extend to TOA
        # Cloud base must be at least 2 levels from the top to allow for updraft development
        min_top_level = 2  # Don't allow clouds to extend above this level

        ktop = lax.cond(
            pressure_increasing,
            lambda: jnp.maximum(cloud_base - cloud_depth, min_top_level),      # Standard: top = lower index, but not TOA
            lambda: jnp.minimum(cloud_base + cloud_depth, nlev-1-min_top_level)  # Reverse: top = higher index
        )
        
        # --- Cloud-base mass-flux closure -------------------------------------
        # ECHAM anchors the DEEP cloud-base mass flux to the boundary-layer
        # MOISTURE SUPPLY, not to instantaneous CAPE (mo_cumastr.f90 ``zmfub`` =
        # zdqpbl/(g·(q_u−q_e))). At quasi-equilibrium the updraft must export the
        # column's moisture source, so M_b = E/(q_u−q_e) with E the surface
        # evaporation [kg/m²/s] (zdqpbl/g). That source is smooth, so the mass
        # flux is steady. The bare-CAPE closure (``mass_flux_closure``) instead
        # tracks instantaneous CAPE and flips convection fully on/off each step —
        # the cloud-base flicker seen in single-column RCE. ``moisture_supply``
        # is the SAME-STEP delivered surface evaporation (the vdiff/surface
        # terms run before convection in the ECHAM physc ordering), possibly
        # raised to the zdqpbl PBL-integral above; when it is absent (E=0:
        # cold start, or a stack with no surface term) we fall back to the
        # CAPE closure so behaviour is unchanged there.
        qsat_cb = saturation_mixing_ratio(pressure[cloud_base], temperature[cloud_base])
        q_excess = qsat_cb - humidity[cloud_base]  # kg/kg, cloud-base saturation deficit
        # ECHAM ``zlo1`` validity gate (mo_cumastr.f90:268-271): the moisture-
        # budget closure ``E/(q_u−q_e)`` is only used when the cloud-base
        # saturation deficit exceeds ``zdqmin = max(0.01·q_env, 1e-10)`` — i.e.
        # the environment is not already ~saturated there. When the cloud base
        # is at/over saturation (deficit ≤ zdqmin, or negative under spectral
        # supersaturation ringing) the denominator collapses and ``E/q_excess``
        # spikes to the CFL cap, dumping a catastrophic burst of latent heat in
        # one step — the hot-cell runaway on T63L47 real-orography. ECHAM falls
        # back to a tiny flux there; we fall back to the bounded CAPE closure,
        # which keeps convection finite without losing the gentle continuous
        # moisture-anchored flux in the well-subsaturated columns where the
        # budget closure is physical.
        zdqmin = jnp.maximum(0.01 * humidity[cloud_base], 1.0e-10)
        moisture_valid = jnp.logical_and(
            q_excess > zdqmin, moisture_supply > _MIN_MOISTURE_SUPPLY
        )
        mass_flux_moisture = jnp.clip(
            moisture_supply / jnp.maximum(q_excess, zdqmin),
            config.cmfcmin, config.cmfcmax,
        )
        mass_flux_cape = mass_flux_closure_blend(
            cape, cin, jnp.array(0.0), type_weights, config
        )
        # Apply the moisture-anchored flux to any active convection (deep/
        # shallow/mid) where the budget closure is valid: with a well-
        # subsaturated cloud base the flux is a modest steady value
        # (E/(q_u−q_e)) that removes CAPE gradually, so convection stays
        # *continuously* on at the evaporation-balancing rate instead of the
        # CAPE-cap flux that empties CAPE in one step and flickers off. Columns
        # that fail ``moisture_valid`` (near-saturated cloud base, or negligible
        # evaporation 0 < E ≤ _MIN_MOISTURE_SUPPLY) keep the bounded CAPE
        # closure rather than a CFL-saturating or ~zero moisture flux.
        use_moisture = jnp.logical_and(conv_type >= 1, moisture_valid)
        mass_flux_base = jnp.where(use_moisture, mass_flux_moisture, mass_flux_cape)

        # A ``cubasmc`` plume takes neither closure: its cloud-base flux is
        # the resolved ascent that triggered it (mo_cuascent.f90:643).
        # Neither the surface moisture budget nor surface-parcel CAPE has
        # any bearing on a plume with no surface connection.
        mass_flux_base = jnp.where(
            use_midlev,
            midlevel_mass_flux(omega[cloud_base], config),
            mass_flux_base,
        )

        # Fuzzy trigger: the closure flux fades in over ~smooth_trigger_j
        # around the CAPE threshold instead of snapping on — this is what
        # gives tau/entrainment/threshold parameters nonzero gradients in
        # near-trigger columns. Fully-active columns (weight ≈ 1) are
        # unchanged.
        mass_flux_base = mass_flux_base * trigger_weight

        # ECHAM mass-flux CFL cap (``mo_cumastr.f90:582-583``):
        #
        #     zmfmax = pmref(jl, ikb-1) / dt
        #     zmfub1 = MIN(zmfub1, zmfmax)
        #
        # The convective updraft cannot evacuate more mass per unit time
        # than the source layer at cloud base contains. Without this cap
        # the closure can return arbitrarily large mass fluxes when CAPE
        # is high relative to the convective timescale, producing run-
        # away latent heating in a single step. We use the air mass of
        # the cloud-base layer itself (``rho * dz``) as the budget.
        layer_mass_at_cb = rho[cloud_base] * layer_thickness[cloud_base]
        mfu_cfl_max = layer_mass_at_cb / dt
        mass_flux_base = jnp.minimum(mass_flux_base, mfu_cfl_max)
        
        # Calculate updraft
        updraft_state = calculate_updraft(
            temperature, humidity, pressure, layer_thickness, rho,
            cloud_base, ktop, conv_type, mass_flux_base, config,
            land_fraction=land_fraction,
            type_weights=type_weights,
            # ECHAM's zlift, for the one ascent test that uses it: the
            # first step above a ``cubasmc`` (klab == 1) cloud base.
            lift=cloud_base_lift(config, thvsig),
        )
        
        # --- ECHAM depth demotion (mo_cumastr.f90:750-753) ---------------
        # A "deep" plume whose realized cloud turns out thinner than
        # 200 hPa is re-labelled shallow:
        #
        #     zpbmpt = paphp1(kcbot) - paphp1(kctop)
        #     IF (ldcum .AND. ktype==1 .AND. zpbmpt < 2.e4) ktype = 2
        #
        # The realized top is the highest level the updraft actually
        # reached (mfu above the numerical floor); full-level pressures
        # stand in for ECHAM's half levels (#530). One-pass limitation,
        # documented: ECHAM demotes BEFORE its second cuasc, so the demoted
        # column re-ascends with entrscv; jcm runs one ascent, so the
        # demotion changes the label (which gates the Nordeng rescale below
        # and the downstream ktype consumers — the Sundqvist Sc guard, the
        # tracer transport) but not the already-computed entrainment. The
        # entrainment consequence of a systematic mislabel is what the
        # moisture-convergence split above fixes at the source.
        has_plume = updraft_state.mfu > 1e-6
        p_top_realized = jnp.min(jnp.where(has_plume, pressure, jnp.inf))
        zpbmpt = pressure[cloud_base] - p_top_realized
        conv_type_final = jnp.where(
            (conv_type == 1) & (zpbmpt < 2.0e4),
            jnp.asarray(2, conv_type.dtype), conv_type,
        )

        # Calculate precipitation from updraft
        # Use the per-layer precip generated inside calculate_updraft (the
        # ECHAM ``pdmfup`` accumulator) rather than the previous
        # ``sum(lu*mfu)*cprcon`` estimator, which was ~60x too small on
        # tropical RCE columns. See ``flux_tendencies.calculate_precipitation_rate``.
        precip_rate = jnp.sum(updraft_state.pdmfup)
        
        # Calculate downdraft (now properly implemented)
        downdraft_state = calculate_downdraft(
            temperature, humidity, pressure, layer_thickness, rho,
            updraft_state, precip_rate, cloud_base, ktop, config
        )
        
        # --- Nordeng CAPE closure (deep convection; mo_cumastr.f90:812-906)
        # Rescale the trial cloud-base flux so the REALIZED flux profile
        # would consume the plume CAPE in cmftau seconds:
        #     zmfub1 = zcape·zmfub / (zheat·cmftau)
        # zheat is the CAPE-consumption rate per unit net convective mass
        # flux (environment stability × g·(mfu+mfd)/ρ summed over the cloud
        # column); zcape is the plume CAPE with virtual-T and condensate
        # loading. This replaces the dimensionally-inconsistent CAPE/(g·τ)
        # (units m/s, review finding 2.5). ECHAM applies the rescale by
        # re-running cuasc with the corrected base flux; because the parcel
        # properties are independent of the flux magnitude (fractional
        # entrainment) and every flux is linear in it, an in-place linear
        # rescale of the plume fluxes is equivalent to first order and
        # avoids the second ascent pass. The downdraft arrays are rescaled
        # by the same factor, exactly as mo_cumastr.f90:945-958.
        in_cloud = (jnp.arange(nlev) >= jnp.minimum(ktop, cloud_base)) & (
            jnp.arange(nlev) <= jnp.maximum(ktop, cloud_base)
        )
        zroi = c.rd * temperature * (1.0 + c.vtmpc1 * humidity) / pressure
        dT_up = jnp.diff(temperature, prepend=temperature[:1])   # T(k-1)-T(k), TOA-first
        dq_up = jnp.diff(humidity, prepend=humidity[:1])
        net_mf = updraft_state.mfu + downdraft_state.mfd
        zheat = jnp.sum(
            jnp.where(
                in_cloud,
                ((-dT_up + c.grav * layer_thickness / c.cpd) / temperature
                 + c.vtmpc1 * (-dq_up))
                * (c.grav * net_mf) * zroi,
                0.0,
            )
        )
        zcape_plume = jnp.sum(
            jnp.where(
                in_cloud & (updraft_state.mfu > 0),
                (c.grav * (updraft_state.tu - temperature) / temperature
                 + c.grav * c.vtmpc1 * (updraft_state.qu - humidity)
                 - c.grav * updraft_state.lu) * layer_thickness,
                0.0,
            )
        )
        zmfub = jnp.maximum(mass_flux_base, config.cmfcmin)
        zmfub1 = zcape_plume * zmfub / (jnp.maximum(zheat, 1e-10) * config.tau)
        # Bounds per ECHAM: the CFL cap above, cmfcmin (1e-10) below —
        # an invented 0.001 floor here used to bind on weak first
        # guesses, and a BOUND rescale target erases the closure/trigger
        # dependence of the amplitude (rescale = const/zmfub), deadening
        # d/d(trigger_cape) everywhere.
        zmfub1 = jnp.clip(zmfub1, config.cmfcmin, mfu_cfl_max)
        # Deliberate deviation from ECHAM: the rescale applies only when the
        # cloud-base flux came from the CAPE fallback. ECHAM rescales every
        # deep column (its first guess is always the PBL moisture budget),
        # but in the single-column RCE the rescale on top of the moisture-
        # anchored flux re-introduces the CAPE-tracking pulse the #529/#535
        # closure work eliminated (measured: max temporal heating std 7 →
        # 12 K/day). Where the moisture closure is invalid the fallback is
        # now Nordeng's zcape/(zheat·cmftau) — replacing the dimensionally
        # inconsistent CAPE/(g·τ) — so both branches are physical.
        # ECHAM rescales EVERY deep column (mo_cumastr.f90:812-906): the
        # moisture-budget flux is only the FIRST GUESS; Nordeng's
        # zmfub1 = zcape·zmfub/(zheat·cmftau) sets the final amplitude.
        # The earlier deviation (gating the rescale off when the moisture
        # closure was valid) capped deep convection at the CURRENT
        # evaporation — coupled T63L47 runs then locked into a desiccated
        # fixed point (CAPE 5000+ J/kg untouched, mass flux 7x low, TPW
        # pinned at 1.5 kg/m2). Unconditional again, as in ECHAM; the RCE
        # pulsing that motivated the gate is handled by the smoothed
        # trigger (the closure fades in over smooth_trigger_j instead of
        # snapping), which also keeps gentle convective precip alive at
        # the near-neutral equilibrium the efficient rescale produces.
        rescale = jnp.where(
            (conv_type_final == 1) & (zheat > 1e-10) & (zcape_plume > 0.0),
            zmfub1 / zmfub,
            1.0,
        )
        updraft_state = updraft_state._replace(
            mfu=updraft_state.mfu * rescale,
            pdmfup=updraft_state.pdmfup * rescale,
            plude=updraft_state.plude * rescale,
        )
        downdraft_state = downdraft_state._replace(
            mfd=downdraft_state.mfd * rescale,
            pdmfdp=downdraft_state.pdmfdp * rescale,
        )

        # Calculate final tendencies for basic variables
        tendencies = calculate_tendencies(
            temperature, humidity, u_wind, v_wind, pressure, rho, layer_thickness,
            updraft_state, downdraft_state,
            cloud_base, ktop, dt, config
        )
        
        # qc/qi tendencies come from the cudtdq ledger's detrained
        # condensate (g/Δp·plude, ECHAM pxtecl/pxteci) — computed inside
        # calculate_tendencies. The previous ``mass_flux·tracer·0.1`` /
        # ``diff(...)·0.001`` pseudo-transport and the ``lu·0.1``/``lu·0.05``
        # magic-number production had no ECHAM counterpart and were
        # dimensionally meaningless (review finding 2.4).
        dqc_dt = tendencies.dqc_dt
        dqi_dt = tendencies.dqi_dt
        qc_conv = tendencies.qc_conv
        qi_conv = tendencies.qi_conv
        
        # NOTE: ECHAM applies NO grid-mean saturation adjustment after
        # cudtdq (verified against mo_cumastr.f90 — after the tendencies
        # only cududv and the tracer mass fixer run; environment
        # supersaturation is the stratiform cloud scheme's job, fed by the
        # detrained condensate). The previous ``convective_adjustment`` over
        # the cloud column double-counted stratiform condensation and
        # contributed ~2/3 of the column heating (review finding 2.1); with
        # the faithful flux/precip ledger above, the raw tendencies flow
        # through unmodified.
        # Create enhanced tendencies with fixed qc/qi transport and the
        # adjusted saturation state.
        enhanced_tendencies = ConvectionTendencies(
            dtedt=tendencies.dtedt,
            dqdt=tendencies.dqdt,
            dudt=tendencies.dudt,
            dvdt=tendencies.dvdt,
            qc_conv=qc_conv,
            qi_conv=qi_conv,
            precip_formation=tendencies.precip_formation,
            precip_conv=tendencies.precip_conv,
            dqc_dt=dqc_dt,
            dqi_dt=dqi_dt,
        )
        
        # ECHAM-ICON convention: ktop is the smallest level index (highest
        # altitude) where the updraft mass flux is still nonzero — i.e.
        # where the dynamic termination in `calculate_updraft` last left
        # a nonzero `mfu` before zeroing it above. The previous code wrote
        # the *scan ceiling* ``ktop = kbase - cloud_depth``, which masks
        # the actual cloud top whenever the updraft terminates early.
        # Re-derive it from where ``updraft_state.mfu`` is still active.
        mfu_active = updraft_state.mfu > config.cmfcmin
        has_active = jnp.any(mfu_active)
        candidate = jnp.where(
            mfu_active, jnp.arange(nlev), jnp.array(nlev, jnp.int32),
        )
        # ``min(candidate)`` = topmost active level (smallest index in
        # ECHAM ordering). If no level is active, fall back to the scan
        # ceiling so downstream consumers don't see ``nlev``.
        actual_ktop = jnp.where(
            has_active, jnp.min(candidate).astype(jnp.int32), ktop,
        )

        # Update state
        new_state = ConvectionState(
            tu=updraft_state.tu, qu=updraft_state.qu, lu=updraft_state.lu,
            uu=u_wind, vu=v_wind,  # Simplified - would update from momentum transport
            td=downdraft_state.td, qd=downdraft_state.qd,
            ud=u_wind, vd=v_wind,  # Simplified
            mfu=updraft_state.mfu, mfd=downdraft_state.mfd,
            # Fractional entrainment (1/m) is rescale-invariant (a rate,
            # not a flux); the transport term rebuilds the absolute
            # entrainment flux against the rescaled mfu.
            entr=updraft_state.entr,
            ktype=jnp.asarray(conv_type_final, dtype=jnp.int32),
            kbase=jnp.array(cloud_base),
            ktop=actual_ktop, prate=enhanced_tendencies.precip_conv,
        )
        
        return enhanced_tendencies, new_state
    
    # No convection case (with fixed qc/qi placeholders)
    def no_convection():
        # Initialize fixed qc/qi tendencies to zero
        dqc_dt = jnp.zeros_like(qc)
        dqi_dt = jnp.zeros_like(qi)
        
        tendencies = ConvectionTendencies(
            dtedt=dtedt, dqdt=dqdt, dudt=dudt, dvdt=dvdt,
            qc_conv=qc_conv, qi_conv=qi_conv,
            precip_formation=jnp.zeros_like(qc),
            precip_conv=precip_conv,
            dqc_dt=dqc_dt, dqi_dt=dqi_dt
        )
        return tendencies, state
    
    # Apply convection if active. Both branches are pinned to the input
    # temperature dtype: under jax_enable_x64 (float64 dycore, float32
    # physics) a few float64 constants inside the full-convection branch
    # promote dudt/dvdt/dqc_dt/dqi_dt, and lax.cond requires the branch
    # output types to match exactly.
    def _pin(fn):
        def wrapped():
            tend, st = fn()
            tend = jax.tree.map(lambda x: x.astype(temperature.dtype), tend)
            st = jax.tree.map(
                lambda x: x.astype(temperature.dtype)
                if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
                else x,
                st,
            )
            return tend, st
        return wrapped

    tendencies, updated_state = lax.cond(
        conv_type > 0,
        _pin(apply_full_convection),
        _pin(no_convection)
    )

    return tendencies, updated_state


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm, TracerSpec  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402
from jcm.physics.diagnostics.moist_air_state import advance_thermo_run  # noqa: E402


class TiedtkeConvection(PhysicsTerm):
    """Tiedtke-Nordeng mass-flux convection as a composable PhysicsTerm.

    Operates on column-vectorized state ``(nlev, ncols)``. Calls the
    standalone :func:`tiedtke_nordeng_convection` scheme via ``jax.vmap``
    over columns. Holds its own :class:`ConvectionParameters` as
    ``nnx.Param`` so that gradients flow through them.

    Reads the moist-air diagnostics produced by
    :class:`~jcm.physics.diagnostics.moist_air_state.MoistAirColumnState`
    (``pressure_full``, ``layer_thickness``, ``air_density``), the
    current cloud diagnostics from ``diagnostics["clouds"]``, and the
    model timestep from ``diagnostics["_dt_seconds"]``. The environment
    (T, q) comes from the running ``thermo_run`` view (post-vdiff under
    the ECHAM physc term ordering; falls back to the raw state when
    absent), and the zdqpbl closure supply reads the same-step
    ``vertical_diffusion.qv_tendency`` profile plus the delivered surface
    evaporation from ``diagnostics["surface"]``. Writes the
    :class:`ConvectionData` sub-struct under the public ``"convection"``
    key and advances ``clouds.qc`` / ``clouds.qi`` for downstream
    microphysics in the same split step.
    """

    name: ClassVar[str] = "tiedtke_convection"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "layer_thickness", "air_density", "clouds",
    )
    provides: ClassVar[tuple[str, ...]] = ("convection", "clouds")

    requires_dycore_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: ConvectionParameters | None = None):
        """Hold the scheme-native :class:`ConvectionParameters`.

        With ECHAM's ``lmfmid`` on (the reference default, setphys.f90:71)
        the scheme runs the ``cubasmc`` mid-level trigger, which needs the
        resolved pressure vertical velocity. That is declared here as a
        genuine ``omega`` dycore-field requirement rather than degraded to
        a zero fallback: a silently-absent omega would disable an entire
        convective trigger — precisely the failure #697 exists to fix — and
        would do it invisibly. Declaring it means the dinosaur backend
        turns its omega provider on automatically (``runners._want_omega``),
        and a backend that cannot supply omega fails at Model construction
        with a pointed message instead of quietly losing elevated
        convection.

        The escape hatch for such a backend (today: pySES, which computes
        omega internally but does not expose it — #698) is ECHAM's own
        namelist switch::

            +physics.terms.tiedtke_convection.params.cu_lmfmid=false

        Column-mode callers with no dycore at all are unaffected: they
        bypass the contract check, and a zero omega correctly means "no
        resolved ascent, so no mid-level convection".
        """
        params = params or ConvectionParameters.default()
        self.params = nnx.Param(params)
        if bool(params.cu_lmfmid):
            self.requires_dycore_fields = ("omega",)

    @classmethod
    def required_tracers(cls) -> tuple[TracerSpec, ...]:
        """Declare ``qc`` and ``qi`` so the dynamics carries them across steps.

        The scheme transports cloud water and ice prognostically inside
        each updraft/downdraft, so the tendencies it returns rely on
        seeing yesterday's qc/qi at the start of every column call.
        """
        return (
            TracerSpec("qc", units="kg/kg"),
            TracerSpec("qi", units="kg/kg"),
        )

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute convective tendencies; write ``convection`` diagnostics."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        layer_thickness = diagnostics["layer_thickness"]
        air_density = diagnostics["air_density"]

        qc = state.tracers.get("qc", jnp.zeros_like(state.temperature))
        qi = state.tracers.get("qi", jnp.zeros_like(state.temperature))

        # Per-column land fraction selects between ECHAM's ocean and land
        # ``zdnoprc`` precip-zone thresholds inside the updraft.
        land_fraction = terrain.fmask.reshape(ncols)

        # Environment (T, q) for the parcel/closure calculations: the running
        # ``thermo_run`` view, which vdiff advanced with its same-step
        # tendencies (the term ordering follows ECHAM ``physc``: vdiff runs
        # before ``cucall``). This mirrors ECHAM's provisional variables
        # ``ztp1 = ptm1 + ptte·dt`` handed to ``cumastr`` — convection sees
        # the post-vdiff column, so it can consume what vdiff supplied THIS
        # step. Fall back to the raw state for stacks without the
        # ``MoistAirColumnState`` seeding term (unit tests, custom stacks).
        thermo_run = diagnostics.get("thermo_run")
        if thermo_run is None:
            temperature_env = state.temperature
            humidity_env = state.specific_humidity
        else:
            temperature_env = thermo_run["temperature"]
            humidity_env = thermo_run["specific_humidity"]

        # Boundary-layer moisture supply floor for the deep mass-flux closure:
        # the SAME-STEP grid-box surface evaporation [kg/m²/s]. The vdiff term
        # (which owns the surface exchange as the bottom row of its implicit
        # solve) and the ``EchamSurface`` term that republishes its delivered
        # fluxes both run *before* convection now, so this is this step's
        # actually-delivered flux, not a one-step-lagged carry. Absent (e.g. a
        # radiative-convective stack with no surface term) it stays zero and
        # the scheme falls back to the CAPE closure.
        #
        # Read ``effective_evaporation`` — the moisture the column actually
        # received. Since the surface exchange became the bottom boundary row
        # of the vdiff implicit solve, this equals ``evaporation`` identically
        # (reported == delivered, the ECHAM pev_vdiff identity), so the budget
        # closure can never export more water than was supplied. Fall back to
        # the raw flux for any surface state predating the field.
        surface_diag = diagnostics.get("surface")
        if surface_diag is not None and getattr(
            surface_diag, "effective_evaporation", None) is not None:
            moisture_supply = jnp.maximum(
                jnp.asarray(surface_diag.effective_evaporation).reshape(ncols), 0.0
            )
        elif surface_diag is not None and hasattr(surface_diag, "evaporation"):
            moisture_supply = jnp.maximum(
                jnp.asarray(surface_diag.evaporation).reshape(ncols), 0.0
            )
        else:
            moisture_supply = jnp.zeros(ncols)

        # Same-step pqte analog for the zdqpbl closure supply: the moisture
        # tendency the vdiff solve applied THIS step (interior mixing + the
        # surface-evaporation boundary row), read from the same-step
        # ``vertical_diffusion`` diagnostics. ECHAM's ``pqte`` at ``cucall``
        # time contains advection + vdiff; the vdiff part is the dominant PBL
        # moisture source and — crucially — is same-step, so convection
        # consumes exactly what vdiff supplied within the step and the
        # supply cannot compound across steps. The previous one-step-LAGGED
        # total-Δq snapshot form did compound (convergence→convection→
        # convergence ramped for days with heating pinned at the stability
        # cap, then NaN — the onset7 T63L47 analysis), so it was removed.
        #
        # Follow-up (documented deviation): the large-scale ADVECTIVE part of
        # ECHAM's pqte is still missing — the dycore applies advection after
        # physics, so a same-step advective moisture tendency would need new
        # host plumbing (exposing the dynamics tendency to the physics step).
        # A lagged advection increment is NOT an acceptable substitute: the
        # compounding feedback runs precisely through the lagged dynamics
        # term. vdiff-only is a strict subset of ECHAM's supply (conservative
        # closure; the max(E, zdqpbl) floor below keeps the #529 continuous-
        # convection anchor).
        vdiff_diag = diagnostics.get("vertical_diffusion")
        qv_tend_vdiff = getattr(vdiff_diag, "qv_tendency", None)
        if qv_tend_vdiff is not None:
            moisture_tend_profile = qv_tend_vdiff
        else:
            moisture_tend_profile = jnp.zeros_like(state.specific_humidity)

        # ECHAM ``pthvsig`` — σ(θ_v) at the second-lowest full level, from
        # vdiff's prognostic θ_v variance. This is what sets the cloud-base
        # ``zlift``, so convective onset follows the boundary layer's actual
        # turbulent state rather than a namelist constant: a well-mixed
        # daytime layer earns the full 1 K excess and convects readily, a
        # nocturnal stable layer earns the 0.2 K floor and does not. With no
        # vdiff term in the package (column-mode tests, dry configurations)
        # the scheme falls back to ``params.cu_thvsig``.
        thvsig = getattr(vdiff_diag, "thv_sigma", None)
        if thvsig is None:
            thvsig = jnp.full((ncols,), params.cu_thvsig)
        thvsig = jnp.broadcast_to(jnp.reshape(thvsig, (-1,)), (ncols,))

        # ECHAM ``pverv`` — the resolved pressure vertical velocity, which
        # gates and scales the ``cubasmc`` mid-level trigger. The dycore
        # supplies it as the ``omega`` physics field; ``cu_lmfmid`` makes
        # that a hard construction-time requirement (see ``__init__``), so
        # the only paths that reach the zero fallback are ones with no
        # dycore at all — the ``get_empty_data`` structural probe and
        # column-mode callers. Zero omega means no resolved ascent, which
        # correctly leaves the mid-level trigger dormant rather than
        # silently mis-firing.
        omega = (diagnostics.get("_dycore_fields") or {}).get("omega")
        if omega is None:
            omega = jnp.zeros_like(state.temperature)
        omega = jnp.reshape(omega, (nlev, ncols))

        # The DYNAMICS moisture tendency of the just-completed dycore step,
        # reconstructed from the ``_prev_step`` carry that ComposablePhysics
        # publishes: q advanced from q_prev by dt*(physics_prev + dynamics),
        # so dynamics = (q_now - q_prev)/dt - q_tend_physics_prev. It is one
        # step lagged — the SAME provenance as ECHAM's leapfrog ``pqte``
        # dynamics contribution, so this is the reference's information
        # structure, not an approximation of it. Used ONLY in the
        # deep/shallow classification integral ``zdqcv`` (a switch), never
        # in a closure AMPLITUDE: a lagged amplitude is the
        # convergence->convection->convergence compounding loop that blew
        # up the onset7 runs, and the zdqpbl closure supply deliberately
        # stays same-step vdiff-only. Absent carry (step 1, column tests):
        # zeros, i.e. no known convergence.
        prev = diagnostics.get("_prev_step")
        if prev is not None:
            q_prev = jnp.reshape(prev["specific_humidity"], (nlev, ncols))
            q_tend_prev = jnp.reshape(prev["q_tendency"], (nlev, ncols))
            qte_dynamics = (
                (state.specific_humidity - q_prev) / dt - q_tend_prev
            )
            # Step 1: the carry template is all-zeros, which would read as
            # a huge spurious "dynamics tendency" q_now/dt. A zero q_prev
            # field is not a state the model can produce; treat it as
            # "no previous step".
            qte_dynamics = jnp.where(
                jnp.any(q_prev > 0.0), qte_dynamics, 0.0,
            )
        else:
            qte_dynamics = jnp.zeros_like(state.specific_humidity)

        column_fn = jax.vmap(
            tiedtke_nordeng_convection,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, None, None, 0, 0, 1, 0, 1, 1),
            out_axes=(0, 0),
        )
        tendencies_all, _state_all = column_fn(
            temperature_env, humidity_env,
            pressure_full, layer_thickness, air_density,
            state.u_wind, state.v_wind, qc, qi,
            dt, params, land_fraction, moisture_supply,
            moisture_tend_profile, thvsig, omega, qte_dynamics,
        )

        # Hard limit on the convective T tendency: 5 K/hr, applied
        # symmetrically. Healthy deep convection over the warmest tropical
        # SSTs gives ~1 K/hr at the most active level; the cap only fires
        # when the column's parcel-vs-environment energy balance has gone
        # pathological. The companion cloud-base mass-flux CFL cap inside
        # ``tiedtke_nordeng_convection`` bounds the column-integrated mass
        # flux but does not contain per-level latent-heat spikes inside
        # the updraft loop — ECHAM bounds those via the per-level moist-
        # adjustment limits in ``mo_cuadjust.f90`` which we have not yet
        # ported. Until that lands this cap is the safety net.
        # Where the cap fires, scale the WHOLE per-level ledger by the same
        # factor rather than clipping T alone: clipping only the heating
        # decoupled the T/q pair (moistening continued at the uncapped rate
        # while its latent heating was truncated — review finding 2.8). A
        # proportional scale keeps the local energy/water pairing intact;
        # column conservation is still broken wherever the cap fires, which
        # is inherent to any such guard.
        _DTDT_MAX = 5.0 / 3600.0  # K/s
        # Per-COLUMN scale: the tightest per-level factor applies to the
        # WHOLE convective ledger — tendencies AND the precip/detrainment
        # diagnostics. The previous per-level scaling left precip_conv
        # unscaled, so every capped burst opened the composed column
        # water budget by the scaled-away amount (caught by the composed
        # closure test once the unconditional Nordeng rescale made
        # capped bursts routine at pulse peaks). A homogeneous column
        # rescale is exactly how ECHAM's own zmfub1 amplitude scaling
        # acts, so proportionality inside the ledger is preserved and
        # column conservation is exact by linearity. The cap itself
        # remains the documented stopgap for the unported mo_cuadjust
        # per-level limits.
        cap_scale = jnp.min(
            jnp.clip(
                _DTDT_MAX / jnp.maximum(jnp.abs(tendencies_all.dtedt), 1e-30),
                0.0, 1.0,
            ),
            axis=1, keepdims=True,
        )
        cap_scale_col = cap_scale[:, 0]

        tendency = PhysicsTendency(
            u_wind=tendencies_all.dudt.T,
            v_wind=tendencies_all.dvdt.T,
            temperature=(tendencies_all.dtedt * cap_scale).T,
            specific_humidity=(tendencies_all.dqdt * cap_scale).T,
            tracers={
                "qc": (tendencies_all.dqc_dt * cap_scale).T,
                "qi": (tendencies_all.dqi_dt * cap_scale).T,
            },
        )

        # Cloud-base/top / CAPE diagnostics aren't populated by the
        # wrapper today (the scheme returns the per-column state but we
        # don't reduce or surface it yet) — they stay as zeros for
        # back-compat with existing xarray field names. The heating /
        # moistening rates, by contrast, are the *applied* (post-cap)
        # tendencies returned above, surfaced so downstream analysis (e.g.
        # an RCE convective-vs-radiative heating balance) reads them straight
        # off the trajectory instead of re-running the term.
        #
        # Mass fluxes and the absolute entrainment flux (#602) carry the
        # SAME per-column cap scaling as the tendency ledger, so the
        # tracer transport they drive stays proportional to the heat and
        # moisture transport actually applied. ``entr`` is the fractional
        # rate (1/m); the absolute per-layer entrainment flux is
        # ``entr_k · mfu_{k+1} · dz_k`` (the plume entrains against the
        # flux ENTERING the layer from below — updraft.py's ``dmf_entr``).
        # Custom/test schemes may return no state — zeros then (like
        # ktype), meaning no convective tracer transport.
        if _state_all is not None:
            _mfu = (_state_all.mfu * cap_scale).T          # (nlev, ncols)
            _mfd = (_state_all.mfd * cap_scale).T
            _mfu_below = jnp.concatenate(
                [_mfu[1:], jnp.zeros_like(_mfu[:1])], axis=0
            )
            _entrain = (
                _state_all.entr.T * _mfu_below * layer_thickness
            )
            # Downdraft ledger (#622): linear in mfd, so the cap scaling
            # already applied to ``_mfd`` carries through exactly.
            # Function-level import: downdraft.py imports from this module.
            from .downdraft import downdraft_entrainment_ledger
            _entrain_dn = downdraft_entrainment_ledger(
                _mfd, layer_thickness, params.entrdd,
            )
        else:
            _mfu = jnp.zeros_like(pressure_full)
            _mfd = jnp.zeros_like(pressure_full)
            _entrain = jnp.zeros_like(pressure_full)
            _entrain_dn = jnp.zeros_like(pressure_full)
        convection = ConvectionData(
            mass_flux_up=_mfu,
            mass_flux_down=_mfd,
            entrain_up=_entrain,
            entrain_down=_entrain_dn,
            cloud_base=jnp.zeros(ncols, dtype=int),
            cloud_top=jnp.zeros(ncols, dtype=int),
            cape=jnp.zeros(ncols),
            # Per-column convection type for downstream guards (the
            # Sundqvist Sc enhancement gates on ktype == 0). Custom /
            # test schemes may return no state — treat as no convection.
            ktype=(
                _state_all.ktype.reshape(-1).astype(jnp.int32)
                if _state_all is not None
                else jnp.zeros(ncols, dtype=jnp.int32)
            ),
            precip_conv=tendencies_all.precip_conv * cap_scale_col,
            precip_formation=(
                tendencies_all.precip_formation.T * cap_scale_col[jnp.newaxis]
            ),
            qc_conv=tendencies_all.qc_conv.T,
            qi_conv=tendencies_all.qi_conv.T,
            heating_rate=tendency.temperature,
            moistening_rate=tendency.specific_humidity,
        )

        clouds = diagnostics["clouds"].copy(
            qc=jnp.maximum(
                diagnostics["clouds"].qc + tendency.tracers["qc"] * dt,
                0.0,
            ),
            qi=jnp.maximum(
                diagnostics["clouds"].qi + tendency.tracers["qi"] * dt,
                0.0,
            ),
        )

        # Advance the running thermodynamic state so the downstream cloud
        # microphysics sees the post-convection (T, q) — convective detrainment
        # already updated ``clouds.qc/qi`` above; this completes the
        # post-convection state the cloud scheme's saturation balance needs
        # (sequential convection->cloud coupling, see
        # ``jcm.physics.diagnostics.moist_air_state.advance_thermo_run``).
        # Condensate too, not just T and q: this scheme detrains convective
        # qc/qi (applied to ``clouds`` just above), so omitting it left
        # thermo_run's condensate at its PRE-convection value. The 2M scheme
        # then advanced that stale base by only its own microphysics
        # tendency, and every consumer of thermo_run qc/qi — the satellite
        # simulators and the AeroCom cloud diagnostics — lost the convective
        # detrainment entirely.
        diagnostics = advance_thermo_run(
            diagnostics,
            dt,
            d_temperature=tendency.temperature,
            d_specific_humidity=tendency.specific_humidity,
            d_qc=tendency.tracers["qc"],
            d_qi=tendency.tracers["qi"],
        )

        return tendency, {
            **diagnostics,
            "convection": convection,
            "clouds": clouds,
        }
