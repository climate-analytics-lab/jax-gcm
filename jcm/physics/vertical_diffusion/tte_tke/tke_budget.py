"""Turbulent kinetic energy (TKE) budget calculations for vertical diffusion.

This module computes the TKE budget according to the standard TKE equation:
d(TKE)/dt = Shear Production + Buoyancy Production - Dissipation + Transport

The TKE equation is:
d(e)/dt = P_s + P_b - ε + ∂/∂z(K_e ∂e/∂z)

where:
- e = TKE (turbulent kinetic energy)
- P_s = Shear production = K_m * (∂u/∂z)²
- P_b = Buoyancy production = -K_h * (g/θ) * (∂θ/∂z)
- ε = Dissipation = C_ε * e^(3/2) / l
- K_e = TKE exchange coefficient
"""

import jax
import jax.numpy as jnp
from typing import Tuple

import jcm.constants as c
from .vertical_diffusion_types import VDiffState, VDiffParameters


@jax.jit
def compute_shear_production(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dz: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray
) -> jnp.ndarray:
    """Compute shear production term in TKE budget.
    
    P_s = K_m * [(∂u/∂z)² + (∂v/∂z)²]
    
    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        dz: Increments between full level heights [m] (ncol, nlev-1)
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s] (ncol, nlev)
        
    Returns:
        Shear production [m²/s³] (ncol, nlev)

    """
    # Compute vertical wind shear
    du_dz = jnp.diff(u, axis=1) / dz
    dv_dz = jnp.diff(v, axis=1) / dz
    
    # Extend to full levels (nlev) by padding with boundary values
    du_dz_extended = jnp.concatenate([
        du_dz[:, :1],  # Extend top value
        du_dz          # Interior values (nlev-1)
    ], axis=1)
    
    dv_dz_extended = jnp.concatenate([
        dv_dz[:, :1],  # Extend top value
        dv_dz          # Interior values (nlev-1)
    ], axis=1)
    
    # Shear production: P_s = K_m * (S²)
    # where S² = (∂u/∂z)² + (∂v/∂z)²
    shear_squared = du_dz_extended**2 + dv_dz_extended**2
    
    # Multiply by exchange coefficient
    shear_production = exchange_coeff_momentum * shear_squared
    
    return shear_production


@jax.jit
def compute_buoyancy_production(
    temperature: jnp.ndarray,
    dz: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    gravity: float = c.grav
) -> jnp.ndarray:
    """Compute buoyancy production term in TKE budget.
    
    P_b = -K_h * (g/θ) * (∂θ/∂z)
    
    Args:
        temperature: Temperature [K] (ncol, nlev)
        dz: Increments between full level heights [m] (ncol, nlev-1)
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        gravity: Gravitational acceleration [m/s²]
        
    Returns:
        Buoyancy production [m²/s³] (ncol, nlev)

    """
    # Compute vertical temperature gradient
    dt_dz = jnp.diff(temperature, axis=1) / dz
    
    # Extend to full levels (nlev) by padding with boundary values
    dt_dz_extended = jnp.concatenate([
        dt_dz[:, :1],  # Extend top value
        dt_dz          # Interior values (nlev-1)
    ], axis=1)
    
    # Average temperature for buoyancy frequency
    temp_avg = temperature  # Use full level temperature directly
    
    # Buoyancy production: P_b = -K_h * (g/T) * (dT/dz + g/cp)
    # Note: The dry adiabatic lapse rate g/cp is included for stability
    lapse_rate = gravity / c.cpd
    buoyancy_freq = (gravity / temp_avg) * (dt_dz_extended + lapse_rate)
    
    # Buoyancy production (negative for stable stratification)
    buoyancy_production = -exchange_coeff_heat * buoyancy_freq
    
    return buoyancy_production


@jax.jit
def compute_dissipation(
    tke: jnp.ndarray,
    mixing_length: jnp.ndarray,
    c_dissipation: float = 0.19
) -> jnp.ndarray:
    """Compute dissipation term in TKE budget.
    
    ε = C_ε * e^(3/2) / l
    
    Args:
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        c_dissipation: Dissipation constant [-]
        
    Returns:
        Dissipation rate [m²/s³] (ncol, nlev)

    """
    # Ensure TKE is positive
    tke_positive = jnp.maximum(tke, 1e-8)
    
    # Dissipation: ε = C_ε * e^(3/2) / l
    # Use sqrt(tke) * tke / mixing_length for numerical stability
    dissipation = c_dissipation * jnp.sqrt(tke_positive) * tke_positive / mixing_length
    
    return dissipation


@jax.jit
def echam_tke_source_update(
    prev_tke: jnp.ndarray,
    shear_squared: jnp.ndarray,
    buoy_freq_squared: jnp.ndarray,
    mixing_length: jnp.ndarray,
    dt: float,
    c_m: float = 0.4,
    c_h: float = 0.5,
    c_d: float = 0.19,
    tke_min: float = 0.01,
) -> jnp.ndarray:
    """Closed-form analytic implicit update of TKE under shear/buoyancy/dissipation.

    Faithful port of ECHAM's ``vdiff.f90`` lines 837-843. The TKE prognostic
    equation, after substituting the production form ``P = sqrt(e)·zzb`` and
    dissipation ``eps = c_d · e^(3/2) / l``::

        d e/d t = sqrt(e) · l · (c_m S² − c_h N²) − c_d · e^(3/2) / l

    With ``u = sqrt(e)`` this reduces to a quadratic in ``u_new`` whose
    positive root is::

        u_new = zdisl · (sqrt(zktest) − 1)

    where::

        zzb    = l · (c_m S² − c_h N²)
        zdisl  = (l / c_d) / dt        (= "zda1·zmix/ztmst" in ECHAM)
        zktest = 1 + (zzb·dt + 2·sqrt(prev_tke)) / zdisl

    Properties:
      * Output is always ≥ 0 (we additionally floor at ``tke_min``).
      * For weak source (``zzb·dt + 2u_old`` small): ``u_new ≈
        (zzb·dt + 2u_old) / 2`` — linearised.
      * For strong source: ``e_new → l² · (c_m S² − c_h N²) / c_d``, the
        production = dissipation equilibrium.

    Replaces the previous explicit Euler step ``tke + dt·(shear + buoy
    − dissip)`` which has no stability bound; combined with the cross-
    step ``prev_physics_data`` cache in ``output_averages=true`` mode,
    the explicit form let one ill-conditioned column run TKE to ~10¹⁸
    in four timesteps and NaN'd the whole atmosphere.

    Args:
        prev_tke: TKE at previous step [m²/s²], shape (ncol, nlev).
        shear_squared: (du/dz)² + (dv/dz)² [1/s²], shape (ncol, nlev).
        buoy_freq_squared: N² (positive for stable stratification)
            [1/s²], shape (ncol, nlev).
        mixing_length: Turbulent length scale [m], shape (ncol, nlev).
        dt: Time step [s].
        c_m, c_h: Stability function values in the neutral limit
            (Mellor-Yamada constants). Match the JCM exchange-coefficient
            formula ``Km = c_m · l · sqrt(e)``.
        c_d: Dissipation constant.
        tke_min: Lower floor for output TKE [m²/s²].

    Returns:
        Post-source TKE [m²/s²], shape (ncol, nlev). Unconditionally
        non-negative, bounded by the production/dissipation equilibrium.

    """
    zzb = mixing_length * (c_m * shear_squared - c_h * buoy_freq_squared)
    zdisl = (mixing_length / c_d) / dt          # m/s
    sqrt_prev = jnp.sqrt(jnp.maximum(prev_tke, 0.0))
    arg = (zzb * dt + 2.0 * sqrt_prev) / zdisl
    zktest = 1.0 + arg
    # When net source is negative enough that zktest < 1, the implicit
    # equation has u_new = 0 → TKE = 0 (then floored to tke_min).
    zktest_safe = jnp.maximum(zktest, 1.0)
    u_new = zdisl * (jnp.sqrt(zktest_safe) - 1.0)
    return jnp.maximum(u_new * u_new, tke_min)


@jax.jit
def compute_tke_exchange_coefficient(
    tke: jnp.ndarray,
    mixing_length: jnp.ndarray,
    c_tke: float = 0.1
) -> jnp.ndarray:
    """Compute TKE exchange coefficient.
    
    K_e = C_tke * l * sqrt(e)
    
    Args:
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        c_tke: TKE transport coefficient [-]
        
    Returns:
        TKE exchange coefficient [m²/s] (ncol, nlev)

    """
    # Ensure TKE is positive
    tke_positive = jnp.maximum(tke, 1e-8)
    
    # TKE exchange coefficient: K_e = C_tke * l * sqrt(e)
    tke_exchange_coeff = c_tke * mixing_length * jnp.sqrt(tke_positive)
    
    return tke_exchange_coeff


@jax.jit
def compute_tke_diagnostics(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    mixing_length: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute TKE budget diagnostics for analysis.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s] (ncol, nlev)
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        
    Returns:
        Tuple of:
        - Shear production [m²/s³] (ncol, nlev)
        - Buoyancy production [m²/s³] (ncol, nlev)
        - Dissipation [m²/s³] (ncol, nlev)
        - TKE exchange coefficient [m²/s] (ncol, nlev)

    """
    dz = jnp.diff(state.height_full, axis=1)

    shear_production = compute_shear_production(
        state.u, state.v, dz, exchange_coeff_momentum
    )
    
    buoyancy_production = compute_buoyancy_production(
        state.temperature, dz, exchange_coeff_heat
    )
    
    dissipation = compute_dissipation(state.tke, mixing_length)
    
    tke_exchange_coeff = compute_tke_exchange_coefficient(state.tke, mixing_length)
    
    return shear_production, buoyancy_production, dissipation, tke_exchange_coeff


def echam_thv_variance_source_update(
    prev_thv_variance: jnp.ndarray,
    thv_gradient: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    tke: jnp.ndarray,
    mixing_length: jnp.ndarray,
    dt: float,
    c_d: float = 0.19,
    tke_min: float = 1.0e-10,
) -> jnp.ndarray:
    """Source/sink update for the variance of virtual potential temperature.

    Faithful port of ECHAM ``vdiff.f90`` lines 857-860::

        zthvprod = 2 * zsh * ztkesq * zthvirdif**2
        zthvdiss = pthvvar * ztkesq / (zda1 * zmix)
        zthvvar  = pthvvar + (zthvprod - zthvdiss) * ztmst
        zthvvar  = MAX(ztkemin, zthvvar)

    ECHAM's ``zsh`` already carries a factor of the mixing length, so
    ``zsh * ztkesq`` **is** the heat exchange coefficient — jcm's
    ``exchange_coeff_heat = c_h * l * sqrt(TKE)`` is the same quantity, and
    the production term is the standard ``2 * K_h * (dθ_v/dz)²``. Likewise
    ``zda1 * zmix`` is ``zmix / zsmn³``, which is jcm's ``mixing_length /
    c_d`` — the same combination already used by
    :func:`echam_tke_source_update` for ``zdisl``, so the two budgets share
    one dissipation length scale by construction.

    Unlike TKE, ECHAM does NOT solve this one analytically: the variance
    equation is linear in ``pthvvar`` (dissipation is proportional to the
    variance itself, not to a 3/2 power), so a forward step is already
    unconditionally stable in the sense that matters — the ``MAX(ztkemin,
    ...)`` floor catches the one case where a long step could overshoot
    through zero. We keep the explicit form to stay bit-faithful.

    This is the piece that makes ``pthvsig`` a physical quantity rather than
    a tunable: the variance is generated by turbulent motions acting on the
    ambient θ_v gradient and destroyed by turbulent dissipation.

    One property worth knowing before reasoning about the convective
    trigger, because it is easy to get backwards. At the production =
    dissipation fixed point the TKE **cancels**::

        var* = 2·c_h·l·√TKE·G² · l/(c_d·√TKE) = 2·c_h·l²·G²/c_d

    so the equilibrium σ(θ_v) is set by the mixing length and the ambient
    gradient alone. What distinguishes a quiescent stable layer is the
    *rate*: production carries √TKE, so a layer with √TKE ~ 0.01 m/s crawls
    toward that fixed point while a stirred one reaches it within a few
    steps. Both behaviours are pinned in ``TestThvVarianceBudget``.

    Args:
        prev_thv_variance: θ_v variance at the previous step [K²],
            shape (ncol, nlev).
        thv_gradient: ∂θ_v/∂z [K/m], shape (ncol, nlev).
        exchange_coeff_heat: K_h [m²/s], shape (ncol, nlev).
        tke: TKE used for the dissipation rate [m²/s²], shape (ncol, nlev).
            Pass the POST-source TKE, matching ECHAM's ordering where
            ``ztkesq`` is formed from ``ptkem1`` alongside the TKE update.
        mixing_length: Turbulent length scale [m], shape (ncol, nlev).
        dt: Time step [s].
        c_d: Dissipation constant, shared with the TKE budget.
        tke_min: Floor on the variance [K²] — ECHAM reuses ``ztkemin``.

    Returns:
        Post-source θ_v variance [K²], shape (ncol, nlev), ``>= tke_min``.

    """
    sqrt_tke = jnp.sqrt(jnp.maximum(tke, tke_min))
    production = 2.0 * exchange_coeff_heat * thv_gradient * thv_gradient
    # ``zda1 * zmix`` == mixing_length / c_d; guard the divide for a
    # degenerate zero-length column rather than letting it produce inf.
    dissipation_length = jnp.maximum(mixing_length / c_d, 1.0e-10)
    dissipation = prev_thv_variance * sqrt_tke / dissipation_length
    updated = prev_thv_variance + (production - dissipation) * dt
    return jnp.maximum(updated, tke_min)
