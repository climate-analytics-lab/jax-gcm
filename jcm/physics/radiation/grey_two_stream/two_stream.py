"""Two-stream radiative transfer solver

This module implements the two-stream approximation for radiative
transfer through a multi-layer atmosphere.

Each homogeneous layer is solved exactly under the Eddington closure
(Meador & Weaver 1980; Toon et al. 1989): its diffuse reflectance and
transmittance, and -- for the shortwave -- the fraction of the collimated
solar beam it scatters into the upward and downward diffuse streams. The
shortwave optical properties are delta-scaled first (the delta-Eddington
approximation of Joseph, Wiscombe & Weinman 1976), and the layers are
combined with the adding method (Shonk & Hogan 2008, eqs. 9-13), so the
column conserves energy: with no absorption, everything that enters at the
top leaves through the top or reaches the surface.

The longwave keeps its no-scattering source recurrence; only its layer
transmittance comes from the solution above.

"""

import functools
import math

import jax.numpy as jnp
import jax
from typing import Tuple, Optional
from ..radiation_types import OpticalProperties


# Below this cosine of the solar zenith angle the direct-beam path length
# ``1/mu0`` is floored. It is a numerical guard for the grazing and night-side
# columns only (``1/mu0`` would otherwise overflow): the caller masks every
# shortwave flux to zero where ``cos_zenith <= 0``, and at ``mu0 = 1e-4`` the
# incident flux on a horizontal surface is already ~0.1 W/m2.
_MU0_FLOOR = 1.0e-4

# The direct-beam layer solution is evaluated in one of two exactly
# equivalent forms, selected on the squared eigenvalue ``lambda**2``; see
# ``_direct_beam_layer``. Any threshold in (0, 1) keeps both forms away from
# their singular points; 0.25 (``lambda = 0.5``) sits well inside it.
_DIRECT_FORM_SWITCH_LAMBDA_SQ = 0.25


def delta_eddington_scaling(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Delta-Eddington scaling of the layer optical properties.

    Joseph, Wiscombe & Weinman (1976): the forward diffraction peak of the
    phase function, a fraction ``f = g**2`` of the scattered light, is
    treated as unscattered, and the two-stream solution is applied to the
    remainder:

        tau' = (1 - ssa f) tau
        ssa' = (1 - f) ssa / (1 - ssa f)
        g'   = (g - f) / (1 - f) = g / (1 + g)

    Domain: the truncation removes a *forward* peak, so it applies only to
    ``g > 0``; ``f = max(g, 0)**2``, and a backward-scattering layer
    (``-1 <= g <= 0``) passes through unscaled. Applied to ``g < 0`` the
    formula leaves the physical range (``g = -0.9`` gives ``g' = -9``,
    ``g = -1`` divides by zero). ``g'`` is written as
    ``g - g+**2/(1 + g+)`` with ``g+ = max(g, 0)``, which is ``g/(1 + g)``
    for ``g > 0`` and ``g`` otherwise, continuous with slope 1 on both
    sides of ``g = 0``.

    This is the adjustment Toon et al. (1989) prescribe with the Eddington
    coefficients for solar radiation, and it is what makes the closure hold
    for cloud droplets and aerosol: unscaled, ``g = 0.85`` makes the
    Eddington direct-beam backscatter coefficient
    ``gamma3 = (2 - 3 g mu0)/4`` negative at high sun, so a thin
    forward-scattering layer would reflect a negative flux. Scaled,
    ``g' <= 1/2`` and ``gamma3 >= 1/8`` for every ``mu0``.

    Args:
        tau: Optical depth.
        ssa: Single-scattering albedo.
        g: Asymmetry factor.

    Returns:
        The scaled ``(tau, ssa, g)``.

    """
    g_forward = jnp.maximum(g, 0.0)
    f = g_forward * g_forward
    one_minus_ssa_f = 1.0 - ssa * f
    # ``1 - ssa f`` vanishes only at ssa = g = 1: a purely forward-scattering,
    # non-absorbing layer, which the scaling makes transparent (tau' = 0).
    # Its scaled ssa is then immaterial; the safe denominator keeps the
    # division and its derivative finite there.
    safe = one_minus_ssa_f > 0.0
    denom = jnp.where(safe, one_minus_ssa_f, 1.0)
    ssa_scaled = jnp.where(safe, ssa * (1.0 - f) / denom, ssa)
    g_scaled = g - f / (1.0 + g_forward)
    return one_minus_ssa_f * tau, ssa_scaled, g_scaled


# ``(1 - exp(-x))/x = sum_n (-x)^n/(n+1)!`` is evaluated as this truncated
# series below ``_PSI_SERIES_SWITCH``. Ten terms are exact to float64 there:
# the first omitted term is ``0.1**10/11! = 2.5e-18``.
_PSI_SERIES_SWITCH = 0.1
_PSI_SERIES = tuple((-1.0) ** n / math.factorial(n + 1) for n in range(10))


def _one_minus_exp_neg_over_x(x: jnp.ndarray) -> jnp.ndarray:
    """``(1 - exp(-x)) / x`` for ``x >= 0``, smooth through ``x = 0``.

    The direct quotient ``-expm1(-x)/x`` is accurate in value for any
    ``x > 0``, but its *derivative*, ``(x exp(-x) + expm1(-x))/x**2``, is a
    difference of two ``O(x)`` terms whose ``O(x**2)`` residual is lost to
    round-off as ``x -> 0`` -- reverse and forward mode at the direct-beam
    resonance, where ``x = 0``, returned O(1) garbage. Below
    ``_PSI_SERIES_SWITCH`` the Maclaurin series is used instead, which carries
    value and derivative to working precision; above it the quotient's
    derivative loses at most ``eps/x`` relative. The safe operand keeps the
    discarded branch finite.
    """
    small = x < _PSI_SERIES_SWITCH
    x_series = jnp.where(small, x, 0.0)
    series = _PSI_SERIES[-1]
    for coeff in reversed(_PSI_SERIES[:-1]):
        series = coeff + x_series * series
    x_safe = jnp.where(small, 1.0, x)
    return jnp.where(small, series, -jnp.expm1(-x_safe) / x_safe)


@jax.jit
def two_stream_coefficients(
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    mu0: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate two-stream coefficients.
    
    Using Eddington approximation (Meador & Weaver 1980).
    
    Args:
        ssa: Single scattering albedo
        g: Asymmetry factor
        mu0: Cosine of solar zenith angle (for SW only)
        
    Returns:
        Tuple of (gamma1, gamma2, gamma3, gamma4)

    """
    # Eddington approximation coefficients
    gamma1 = (7.0 - ssa * (4.0 + 3.0 * g)) / 4.0
    gamma2 = -(1.0 - ssa * (4.0 - 3.0 * g)) / 4.0
    
    if mu0 is not None:
        # Shortwave with solar angle
        gamma3 = (2.0 - 3.0 * g * mu0) / 4.0
        gamma4 = 1.0 - gamma3
    else:
        # Longwave (no direct beam)
        gamma3 = jnp.zeros_like(ssa)
        gamma4 = jnp.ones_like(ssa)
    
    return gamma1, gamma2, gamma3, gamma4


def _direct_beam_layer(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    gamma1: jnp.ndarray,
    gamma2: jnp.ndarray,
    gamma3: jnp.ndarray,
    gamma4: jnp.ndarray,
    lambda_sq: jnp.ndarray,
    R: jnp.ndarray,
    T: jnp.ndarray,
    one_minus_T: jnp.ndarray,
    mu0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Direct-to-diffuse reflectance and transmittance of one layer.

    The two-stream equations with the collimated source (Meador & Weaver 1980
    eq. 4-5; Toon et al. 1989 eq. 23-24), for a unit horizontal flux incident
    at the layer top, ``m = 1/mu0`` and optical depth ``t`` from the top:

        dF+/dt = gamma1 F+ - gamma2 F- - ssa gamma3 m exp(-m t)
        dF-/dt = gamma2 F+ - gamma1 F- + ssa gamma4 m exp(-m t)

    with no diffuse radiation entering either face. The beam scatters
    ``ssa m exp(-m t) dt`` into the diffuse field, a fraction ``gamma3`` up and
    ``gamma4 = 1 - gamma3`` down, so for ``ssa = 1`` the net flux is constant
    and ``R_dir + T_dir + exp(-m tau) = 1``. The particular solution is
    ``F+ = A exp(-m t)``, ``F- = B exp(-m t)`` (Toon et al. 1989 eq. 23-24,
    ``C+``/``C-``), with

        A = ssa m [gamma3 (gamma1 - m) + gamma2 gamma4] / (lambda^2 - m^2)
        B = ssa m [gamma4 (gamma1 + m) + gamma2 gamma3] / (lambda^2 - m^2).

    Adding the homogeneous solution that cancels the particular solution's
    diffuse flux at each face — which is the layer's own diffuse response,
    so it enters only through ``R`` and ``T`` — gives

        R_dir = A (1 - T E) - R B
        T_dir = B (E - T) - R A E,          E = exp(-m tau).

    This form depends on the eigenvalue only through ``lambda^2`` and the
    diffuse ``R``, ``T``, which are already smooth at the conservative limit
    ``lambda = 0``, so it needs no floor there. Its only singularity is the
    resonance ``lambda = m``, a removable one (the numerators vanish with the
    denominator) that needs ``lambda >= 1``. For ``lambda^2`` above
    ``_DIRECT_FORM_SWITCH_LAMBDA_SQ`` the same solution is instead evaluated
    with the resonance factored out analytically. Writing ``e = exp(-lambda
    tau)``, ``D = (gamma1 + lambda) - (gamma1 - lambda) e^2`` and
    ``W = (E - e)/(lambda - m)``,

        R_dir = ssa m [P (1 - e E)/(lambda + m) + Q e W] / D
        T_dir = ssa m [U W + V e (1 - e E)/(lambda + m)] / D

    with ``P = gamma3 (lambda + gamma1) + gamma2 gamma4``,
    ``Q = gamma3 (lambda - gamma1) - gamma2 gamma4``,
    ``U = gamma4 (lambda + gamma1) + gamma2 gamma3`` and
    ``V = gamma4 (lambda - gamma1) - gamma2 gamma3``. ``W`` is analytic in
    ``lambda - m`` and is evaluated as ``E tau psi((lambda - m) tau)`` or
    ``e tau psi((m - lambda) tau)``, ``psi(x) = (1 - exp(-x))/x``, whichever
    has a non-negative argument, so it passes through the resonance with no
    division and no overflow. ``D >= 2 lambda >= 1`` on that branch, and
    ``|lambda^2 - m^2| >= 3/4`` on the other, so neither divides by a small
    number. Both forms are the exact solution; they agree to round-off at the
    switch.

    Both are evaluated with safe operands on the branch not taken, so reverse
    mode sees no ``0 * inf``. No clip is applied: with the delta-scaled
    coefficients ``gamma3 >= 1/8`` and the solution is non-negative, and a
    clip would break the conservation identity above.
    """
    mu0_safe = jnp.maximum(mu0, _MU0_FLOOR)
    m = 1.0 / mu0_safe
    E = jnp.exp(-m * tau)
    one_minus_E = -jnp.expm1(-m * tau)

    use_ratio_form = lambda_sq < _DIRECT_FORM_SWITCH_LAMBDA_SQ

    # --- lambda^2 below the switch: particular solution + diffuse response.
    # The resonance cannot occur here (lambda < 1/2 < 1 <= m); the safe
    # operand keeps the discarded branch's denominator away from it too.
    k2_r = jnp.where(use_ratio_form, lambda_sq, 0.0)
    resonance = k2_r - m * m                        # <= -3/4
    A = ssa * m * (gamma3 * (gamma1 - m) + gamma2 * gamma4) / resonance
    B = ssa * m * (gamma4 * (gamma1 + m) + gamma2 * gamma3) / resonance
    # 1 - T E and E - T, each assembled from non-cancelling pieces.
    R_ratio = A * (one_minus_T + T * one_minus_E) - R * B
    T_ratio = B * (one_minus_T - one_minus_E) - R * A * E

    # --- lambda^2 at or above the switch: resonance factored out.
    k2_f = jnp.where(use_ratio_form, 1.0, lambda_sq)
    k = jnp.sqrt(k2_f)
    e = jnp.exp(-k * tau)
    one_minus_eE = -jnp.expm1(-(k + m) * tau)
    D = (gamma1 + k) - (gamma1 - k) * e * e
    d = k - m
    above = d >= 0.0
    W = jnp.where(
        above,
        E * tau * _one_minus_exp_neg_over_x(jnp.where(above, d, 0.0) * tau),
        e * tau * _one_minus_exp_neg_over_x(jnp.where(above, 0.0, -d) * tau),
    )
    P = gamma3 * (k + gamma1) + gamma2 * gamma4
    Q = gamma3 * (k - gamma1) - gamma2 * gamma4
    U = gamma4 * (k + gamma1) + gamma2 * gamma3
    V = gamma4 * (k - gamma1) - gamma2 * gamma3
    scale = ssa * m / D
    R_factored = scale * (P * one_minus_eE / (k + m) + Q * e * W)
    T_factored = scale * (U * W + V * e * one_minus_eE / (k + m))

    R_dir = jnp.where(use_ratio_form, R_ratio, R_factored)
    T_dir = jnp.where(use_ratio_form, T_ratio, T_factored)
    return R_dir, T_dir


@jax.jit
def layer_reflectance_transmittance(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    mu0: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact Eddington two-stream solution of one homogeneous layer.

    The optical properties are used as given; the shortwave flux solver
    delta-scales them (``delta_eddington_scaling``) before calling this.

    Args:
        tau: Optical depth
        ssa: Single scattering albedo
        g: Asymmetry factor
        mu0: Cosine of solar zenith angle (shortwave only)

    Returns:
        Tuple of ``(R_dif, T_dif, R_dir, T_dir)``. ``R_dif``/``T_dif`` are the
        diffuse reflectance and transmittance. For the shortwave, ``R_dir`` and
        ``T_dir`` are the fractions of the collimated beam incident on the
        layer top (per unit horizontal flux) that leave it as *diffuse*
        radiation through the top and through the bottom; the unscattered
        beam ``exp(-tau/mu0)`` is not included. With ``ssa = 1``,
        ``R_dir + T_dir + exp(-tau/mu0) = 1``. For the longwave both are zero.

    """
    # Get two-stream coefficients
    gamma1, gamma2, gamma3, gamma4 = two_stream_coefficients(ssa, g, mu0)
    
    # Calculate lambda (eigenvalue).
    #
    # Factored rather than written as ``gamma1**2 - gamma2**2``. For the
    # Eddington coefficients above the two forms are algebraically the same
    # quantity — ``(gamma1 - gamma2)(gamma1 + gamma2) = 2(1 - ssa) *
    # 1.5(1 - ssa*g)`` — but the subtraction is a catastrophic cancellation
    # exactly where shortwave cloud optics lives. A liquid cloud has
    # ``ssa ~ 0.9999``, where the two squares agree to four digits and their
    # float32 difference is mostly round-off; feeding that into ``sqrt``,
    # whose derivative is ``1/(2*sqrt(x))``, amplifies the round-off instead
    # of the signal. Measured at ``ssa = 1 - 1e-6``: the subtracted form
    # reports ``d/d(ssa) ~ 6.4e3`` against a true value near 3.4e2, and at
    # ``ssa = 1`` it reports 5e5 for a derivative that does not exist. The
    # factored form has no cancellation at all and is exactly 0 at ssa = 1.
    #
    # The double-``where`` guards the square root itself: at ``ssa = 1`` the
    # masked branch is fed a 1.0 and the outer ``where`` selects the literal 0
    # the physical branch would have produced. ``lambda_val`` feeds only the
    # exponential branch below, which is never selected near ``ssa = 1``; the
    # layer solution there is evaluated as an even series in ``lambda_sq``, so
    # the endpoint derivative flows through the smooth ``lambda_sq`` rather
    # than through this cusped square root.
    lambda_sq = 3.0 * (1.0 - ssa) * (1.0 - ssa * g)
    lambda_positive = lambda_sq > 0.0
    lambda_val = jnp.where(
        lambda_positive,
        jnp.sqrt(jnp.where(lambda_positive, lambda_sq, 1.0)),
        0.0,
    )

    lambda_tau = lambda_val * tau

    # Diffuse reflectance/transmittance of a homogeneous two-stream layer
    # (Meador & Weaver 1980, eq. 14-15; Toon et al. 1989). With the eigenvalue
    # ``lambda`` above and
    #     Gamma = gamma2 / (gamma1 + lambda),   e = exp(-lambda*tau)
    # the exact solution is
    #     R = Gamma (1 - e^2) / (1 - Gamma^2 e^2)
    #     T = (1 - Gamma^2) e / (1 - Gamma^2 e^2)          ->   R_inf = Gamma.
    #
    # We evaluate an algebraically identical rearrangement that is free of the
    # singularities the ratio form carries. Multiplying the numerator and
    # denominator of both ratios by ``(gamma1 + lambda)/lambda`` and writing
    #     S = (1 - e^2) / lambda
    # collapses ``Gamma`` out entirely and gives
    #     R = gamma2 S / (gamma1 S + 1 + e^2)
    #     T = 2 e      / (gamma1 S + 1 + e^2).
    # This is preferred over the ratio form for three reasons, each of which
    # was a real defect the ratio form had here:
    #   * The denominator ``gamma1 S + 1 + e^2`` is a sum of non-negative terms
    #     (``gamma1 >= 0`` for every ssa,g; ``S >= 0``; ``e^2 > 0``) bounded
    #     below by 1, so it never cancels toward zero and needs no floor. The
    #     ratio form's ``1 - Gamma^2 e^2`` vanishes 0/0 in the conservative
    #     limit (ssa -> 1: Gamma -> 1, e -> 1), which is exactly why a
    #     non-absorbing cloud reflected nothing before this fix.
    #   * ``gamma1`` and ``gamma1 + lambda`` appear only multiplied in, never
    #     divided by, so the 0/0 at ssa = g = 1 (where ``gamma1 = 0``) that the
    #     old ``gamma2/gamma1`` needed a double-``where`` to survive simply
    #     does not arise.
    #   * Only the decaying exponentials ``e`` and ``e^2`` appear. The growing
    #     ``exp(+lambda*tau)`` that overflowed float32 above ``lambda*tau ~ 88``
    #     — the reason the old code branched to a separate asymptotic formula
    #     and the reason its two branches disagreed at the cut-off — is gone,
    #     so a single expression is valid at every optical depth. As tau -> inf
    #     both exponentials -> 0 and ``S -> 1/lambda``, giving R -> Gamma and
    #     T -> 0 continuously: the semi-infinite albedo is recovered exactly
    #     with no branch to disagree with.
    # Near the conservative limit the same solution is evaluated as an even
    # Taylor series in ``x2 = (lambda*tau)**2`` instead of via the
    # exponentials. R and T are *even* functions of ``lambda`` — substituting
    # ``lambda -> -lambda`` and multiplying numerator and denominator by
    # ``exp(-2*lambda*tau)`` reproduces them — so they are smooth functions of
    # ``lambda**2 = 3(1-ssa)(1-ssa*g)`` and therefore genuinely two-sided
    # differentiable in ``ssa`` and ``g`` at ``ssa = 1``, even though
    # ``lambda`` itself has a square-root cusp there. A guard that pins
    # ``lambda`` (and ``S``) at their limit *values* loses that channel:
    # autodiff then reported ``dT/dssa = 0.4597`` at ``(ssa=1, g=0.85,
    # tau=0.3)`` against the true 0.4789 (PR #856 review) — a silently biased
    # endpoint derivative for any optimizer pushing ssa toward 1. In the
    # hyperbolic form ``R = gamma2*(sinh x/lambda) / (cosh x +
    # gamma1*(sinh x/lambda))``, ``T = 1/(cosh x + gamma1*(sinh x/lambda))``
    # (multiply the expressions below through by ``exp(x)/2``), both
    # ``cosh x`` and ``sinh(x)/lambda = tau*sinh(x)/x`` are analytic in
    # ``x2``, so truncated series in ``x2`` route AD through the smooth
    # ``lambda_sq`` and carry the exact endpoint derivative. The series is
    # used for ``x2 < 1e-2`` (``lambda*tau < 0.1``), where its truncation
    # error is O(x2**4/4e4) ~ 2.5e-13 relative — below even float64 noise at
    # the switch, so the forward value is seamless across it; the exponential
    # branch's ``lambda_tau`` is masked to 1 inside the series region so its
    # AD stays finite on the discarded branch.
    x2 = lambda_sq * tau * tau            # (lambda*tau)**2, smooth in ssa, g
    use_series = x2 < 1e-2

    # Double-``where`` on the series input, mirroring ``lambda_tau_safe`` on
    # the opposite branch: ``where`` evaluates both branches, and at float32
    # optical depths large enough that the polynomial's powers overflow
    # (``tau * x2**3`` passes 3.4e38 around ``lambda*tau ~ 1e6``, e.g. a
    # ``tau = 1e6`` longwave layer) the discarded ``R_series`` goes
    # inf/inf = NaN, which reverse mode then hands to the *taken* branch as
    # ``0 * NaN = NaN`` — the forward value stayed (0, 0) but d/d(tau, ssa, g)
    # were all NaN. Clamping ``x2`` to 0 outside the series region keeps the
    # discarded polynomial (and its derivatives) finite; inside the region the
    # value and derivative are untouched.
    x2_safe = jnp.where(use_series, x2, 0.0)
    x2_sq = x2_safe * x2_safe
    sinhc = (1.0 + x2_safe / 6.0 + x2_sq / 120.0
             + x2_sq * x2_safe / 5040.0)             # sinh(x)/x
    cosh_x = 1.0 + x2_safe / 2.0 + x2_sq / 24.0 + x2_sq * x2_safe / 720.0
    denom_series = cosh_x + gamma1 * tau * sinhc    # >= 1: every term >= 0
    R_series = gamma2 * tau * sinhc / denom_series
    T_series = 1.0 / denom_series

    lambda_tau_safe = jnp.where(use_series, 1.0, lambda_tau)
    exp_minus = jnp.exp(-lambda_tau_safe)           # e
    exp_minus_sq = jnp.exp(-2.0 * lambda_tau_safe)  # e^2
    scaled_path = tau * (1.0 - exp_minus_sq) / lambda_tau_safe   # S
    denom = gamma1 * scaled_path + 1.0 + exp_minus_sq            # >= 1
    R_exp = gamma2 * scaled_path / denom
    T_exp = 2.0 * exp_minus / denom

    R_dif_exact = jnp.where(use_series, R_series, R_exp)
    T_dif = jnp.where(use_series, T_series, T_exp)

    # Physical bounds. R + T <= 1 already holds by construction on both
    # branches (exponential: ``R + T - 1 = (gamma2 - gamma1) S / denom``;
    # series: ``((gamma2 - gamma1) tau sinhc + (1 - cosh x)) / denom`` — both
    # non-positive since ``gamma1 - gamma2 = 2(1 - ssa) >= 0`` and
    # ``cosh x >= 1``), so the upper clip never acts. The lower clip removes
    # the small negative reflectance the Eddington closure produces for weakly
    # scattering layers, where ``gamma2 < 0`` for ``ssa < 1/(4 - 3g)`` — an
    # approximation artefact, not a physical value. Raising R to 0 there only
    # lowers the layer's absorption ``1 - R - T``, which stays >= 0.
    R_dif = jnp.clip(R_dif_exact, 0.0, 1.0)
    T_dif = jnp.clip(T_dif, 0.0, 1.0)

    if mu0 is not None:
        # ``1 - T`` in closed form, without the cancellation of forming it
        # from T for a thin layer: series ``(C - 1)/C`` with
        # ``C - 1 = (cosh x - 1) + gamma1 tau sinhc``; exponential
        # ``(gamma1 S + (1 - e)^2) / denom``. Both are sums of non-negative
        # terms over a positive denominator.
        cosh_x_m1 = x2_safe / 2.0 + x2_sq / 24.0 + x2_sq * x2_safe / 720.0
        one_minus_T = jnp.where(
            use_series,
            (cosh_x_m1 + gamma1 * tau * sinhc) / denom_series,
            (gamma1 * scaled_path + jnp.square(1.0 - exp_minus)) / denom,
        )
        # The direct-beam solution is built on the *unclipped* diffuse
        # reflectance: it is an exact identity in R (see
        # ``_direct_beam_layer``), and the clipped value would break it.
        R_dir, T_dir = _direct_beam_layer(
            tau, ssa, gamma1, gamma2, gamma3, gamma4, lambda_sq,
            R_dif_exact, T_dif, one_minus_T, mu0)
    else:
        # Longwave - no direct beam
        T_dir = jnp.zeros_like(tau)
        R_dir = jnp.zeros_like(tau)
    
    return R_dif, T_dif, R_dir, T_dir


def longwave_fluxes_single_band(
    tau: jnp.ndarray,
    ssa: jnp.ndarray, 
    g: jnp.ndarray,
    planck_layer: jnp.ndarray,
    planck_interfaces: jnp.ndarray,
    surface_emissivity: float,
    surface_planck: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate LW fluxes for a single band."""
    nlev = tau.shape[0]
    
    # Calculate layer properties (no direct beam for LW)
    R_dif, T_dif, _, _ = layer_reflectance_transmittance(tau, ssa, g, mu0=None)
    
    # Simplified approach: use source function method
    # Source = Planck * (1 - transmittance)
    source = planck_layer * (1.0 - T_dif)
    
    # Initialize arrays
    flux_up = jnp.zeros(nlev + 1)
    flux_down = jnp.zeros(nlev + 1)
    
    # Surface emission
    flux_up = flux_up.at[nlev].set(surface_emissivity * surface_planck)
    
    # Upward flux calculation using recurrence relation
    def upward_step(carry, x):
        flux_below = carry
        lev, R, T, S = x
        # CRITICAL FIX: Removed R * flux_below term
        # The R * flux_below term was incorrectly reflecting upward flux back upward
        # which doesn't make physical sense and was causing flux to be 15-25x too large
        flux_above = T * flux_below + S
        return flux_above, flux_above
        
    # Scan from bottom to top
    indices = jnp.arange(nlev - 1, -1, -1)
    _, flux_up_levels = jax.lax.scan(
        upward_step,
        flux_up[nlev],
        (indices, R_dif[::-1], T_dif[::-1], source[::-1])
    )
    flux_up = flux_up.at[:-1].set(flux_up_levels[::-1])
    
    # Downward flux from top
    def downward_step(carry, x):
        flux_above = carry
        lev, R, T, S = x
        flux_below = T * flux_above + S
        return flux_below, flux_below
        
    # Scan from top to bottom  
    indices = jnp.arange(nlev)
    _, flux_down_levels = jax.lax.scan(
        downward_step,
        0.0,  # No downward LW at TOA
        (indices, R_dif, T_dif, source)
    )
    flux_down = flux_down.at[1:].set(flux_down_levels)
    
    return flux_up, flux_down


@functools.partial(jax.jit, static_argnames=('n_bands',))
def longwave_fluxes(
    optical_properties: OpticalProperties,
    planck_layers: jnp.ndarray,
    planck_interfaces: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    surface_planck: jnp.ndarray,
    n_bands: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate longwave fluxes using two-stream method.

    Args:
        optical_properties: Layer optical properties
        planck_layers: Planck function at layer centers [nlev, n_bands]
        planck_interfaces: Planck function at interfaces [nlev+1, n_bands]
        surface_emissivity: Surface emissivity
        surface_planck: Surface Planck emission [n_bands]
        n_bands: Number of spectral bands (static for shape-polymorphic jit)

    Returns:
        Tuple of (upward_flux, downward_flux) at interfaces [nlev+1, n_bands]

    """
    # Process all bands using vmap
    def process_band(band_idx):
        flux_up_band, flux_down_band = longwave_fluxes_single_band(
            optical_properties.optical_depth[:, band_idx],
            optical_properties.single_scatter_albedo[:, band_idx],
            optical_properties.asymmetry_factor[:, band_idx],
            planck_layers[:, band_idx],
            planck_interfaces[:, band_idx],
            surface_emissivity,
            surface_planck[band_idx]
        )
        return flux_up_band, flux_down_band

    # Run only over the bands we actually use (``n_bands`` is a static int).
    # Previously this vmap ran over a fixed ``max_bands = 10`` buffer and then
    # masked out unused bands, which wasted ~70 % of the two-stream work in
    # the grey scheme (3 LW bands used out of 10).
    band_indices = jnp.arange(n_bands)
    flux_up_all, flux_down_all = jax.vmap(process_band)(band_indices)

    # Transpose to get [nlev+1, n_bands] shape
    flux_up = flux_up_all.T
    flux_down = flux_down_all.T

    # Convert from radiance to flux (multiply by π)
    flux_up *= jnp.pi
    flux_down *= jnp.pi

    return flux_up, flux_down


def shortwave_fluxes_single_band(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    cos_zenith: float,
    toa_flux: float,
    surface_albedo: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shortwave fluxes of one band: delta-Eddington layers joined by adding.

    Args:
        tau: Layer optical depth, top-first ``[nlev]``.
        ssa: Layer single-scattering albedo ``[nlev]``.
        g: Layer asymmetry factor ``[nlev]``.
        cos_zenith: Cosine of the solar zenith angle.
        toa_flux: Downward solar flux on a horizontal surface at the top.
        surface_albedo: Surface albedo, applied to the direct beam and to
            diffuse light alike.

    Returns:
        ``(flux_up, flux_down, flux_direct, flux_down_diffuse)`` at the
        ``nlev + 1`` interfaces, top-first. ``flux_down`` is the direct plus
        the diffuse downward flux. The direct beam is the delta-scaled one,
        so it includes the forward-diffraction peak.

    """
    # Delta-Eddington: the layer solution and the direct beam both use the
    # scaled optical properties (Joseph et al. 1976).
    tau, ssa, g = delta_eddington_scaling(tau, ssa, g)
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(
        tau, ssa, g, cos_zenith)

    # Unscattered beam at every interface. Accumulating the optical depth
    # rather than multiplying layer transmittances keeps a beam that has
    # decayed to exactly 0 from feeding a 0 into a product's reverse pass.
    mu0 = jnp.maximum(cos_zenith, _MU0_FLOOR)
    tau_above = jnp.concatenate(
        [jnp.zeros((1,), tau.dtype), jnp.cumsum(tau)])
    flux_direct = toa_flux * jnp.exp(-tau_above / mu0)

    # Diffuse sources: the beam arriving at each layer's top, scattered into
    # the upward and downward diffuse streams by that layer.
    src_up = R_dir * flux_direct[:-1]
    src_down = T_dir * flux_direct[:-1]

    # Adding method (Shonk & Hogan 2008, eqs. 9-13), as in the RTE solver of
    # RRTMGP. Upward pass from the surface: the albedo of everything below
    # each interface and the diffuse upward flux the sources below it emit,
    # both including the infinite series of reflections between each layer
    # and what lies beneath it.
    def adding_step(carry, layer):
        albedo_below, source_below = carry
        R, T, s_up, s_down = layer
        inv = 1.0 / (1.0 - R * albedo_below)
        albedo = R + T * T * inv * albedo_below
        source = s_up + T * inv * (source_below + s_down * albedo_below)
        return (albedo, source), (albedo, source)

    surface_source = surface_albedo * flux_direct[-1]
    _, (albedo_above, source_above) = jax.lax.scan(
        adding_step,
        (surface_albedo, surface_source),
        (R_dif[::-1], T_dif[::-1], src_up[::-1], src_down[::-1]),
    )
    # Interface-indexed, top-first: entry k is the value at the top of layer
    # k; the surface value closes the arrays.
    albedo = jnp.concatenate(
        [albedo_above[::-1], jnp.reshape(surface_albedo, (1,)).astype(tau.dtype)])
    source = jnp.concatenate(
        [source_above[::-1], jnp.reshape(surface_source, (1,))])

    # Downward pass from the top (no diffuse light enters at TOA): the flux
    # leaving the bottom of each layer is what it transmits, what it
    # reflects back of the upward emission from below, and its own downward
    # source, again summed over the multiple reflections with the albedo
    # below.
    def downward_step(flux_above, layer):
        R, T, s_down, albedo_below, source_below = layer
        flux_below = (T * flux_above + R * source_below + s_down) / (
            1.0 - R * albedo_below)
        return flux_below, flux_below

    _, flux_down_levels = jax.lax.scan(
        downward_step,
        jnp.zeros((), flux_direct.dtype),
        (R_dif, T_dif, src_down, albedo[1:], source[1:]),
    )
    flux_down_dif = jnp.concatenate(
        [jnp.zeros((1,), flux_direct.dtype), flux_down_levels])
    flux_up = albedo * flux_down_dif + source

    return flux_up, flux_direct + flux_down_dif, flux_direct, flux_down_dif


@functools.partial(jax.jit, static_argnames=('n_bands',))
def shortwave_fluxes(
    optical_properties: OpticalProperties,
    cos_zenith: float,
    toa_flux: jnp.ndarray,
    surface_albedo: jnp.ndarray,
    n_bands: int = 2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate shortwave fluxes using two-stream method.

    Args:
        optical_properties: Layer optical properties
        cos_zenith: Cosine of solar zenith angle
        toa_flux: TOA incident flux [n_bands]
        surface_albedo: Surface albedo [n_bands]
        n_bands: Number of spectral bands (static for shape-polymorphic jit)

    Returns:
        Tuple of (up_flux, down_flux, down_direct, down_diffuse)
        All at interfaces [nlev+1, n_bands]

    """
    # Process all bands using vmap
    def process_band(band_idx):
        flux_up, flux_down, flux_dir, flux_dif = shortwave_fluxes_single_band(
            optical_properties.optical_depth[:, band_idx],
            optical_properties.single_scatter_albedo[:, band_idx],
            optical_properties.asymmetry_factor[:, band_idx],
            cos_zenith,
            toa_flux[band_idx],
            surface_albedo[band_idx]
        )
        return flux_up, flux_down, flux_dir, flux_dif

    # Run only over the bands we actually use (``n_bands`` is a static int).
    band_indices = jnp.arange(n_bands)
    flux_up_all, flux_down_all, flux_direct_all, flux_diffuse_all = jax.vmap(
        process_band
    )(band_indices)

    # Transpose to get [nlev+1, n_bands] shape
    flux_up = flux_up_all.T
    flux_down = flux_down_all.T
    flux_direct = flux_direct_all.T
    flux_diffuse = flux_diffuse_all.T

    return flux_up, flux_down, flux_direct, flux_diffuse


@jax.jit
def flux_to_heating_rate(
    flux_up: jnp.ndarray,
    flux_down: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    g: float = 9.81,  # m/s^2
    cp: float = 1004.0  # J/kg/K
) -> jnp.ndarray:
    """Convert flux divergence to heating rate.
    
    dT/dt = -g/cp * dF/dp
    
    Args:
        flux_up: Upward flux at interfaces [nlev+1]
        flux_down: Downward flux at interfaces [nlev+1]
        pressure_interfaces: Pressure at interfaces [nlev+1]
        cp: Specific heat capacity
        
    Returns:
        Heating rate (K/s) [nlev]

    """
    # Net flux at interfaces
    net_flux_down = flux_down - flux_up

    # Flux divergence in layers
    flux_div = jnp.diff(net_flux_down, axis=0)

    # Pressure thickness — guard against |dp|<1 Pa to avoid div-by-zero
    dp = jnp.diff(pressure_interfaces, axis=0)
    dp_safe = jnp.where(jnp.abs(dp) < 1.0, jnp.sign(dp) * 1.0 + 1e-6, dp)

    # Heating rate
    heating = (g / cp) * (-flux_div) / dp_safe

    return heating
