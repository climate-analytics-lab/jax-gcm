"""Two-stream radiative transfer solver

This module implements the two-stream approximation for radiative
transfer through a multi-layer atmosphere.

The implementation uses the Eddington approximation with the
adding method for combining layers.

"""

import functools

import jax.numpy as jnp
import jax
from typing import Tuple, Optional
from ..radiation_types import OpticalProperties


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


@jax.jit
def layer_reflectance_transmittance(
    tau: jnp.ndarray,
    ssa: jnp.ndarray,
    g: jnp.ndarray,
    mu0: Optional[float] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate layer reflectance and transmittance.
    
    Args:
        tau: Optical depth
        ssa: Single scattering albedo  
        g: Asymmetry factor
        mu0: Cosine of solar zenith angle (for SW)
        
    Returns:
        Tuple of (R_dif, T_dif, R_dir, T_dir)
        For LW, only diffuse components are used

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

    R_dif = jnp.where(use_series, R_series, R_exp)
    T_dif = jnp.where(use_series, T_series, T_exp)

    # Physical bounds. R + T <= 1 already holds by construction on both
    # branches (exponential: ``R + T - 1 = (gamma2 - gamma1) S / denom``;
    # series: ``((gamma2 - gamma1) tau sinhc + (1 - cosh x)) / denom`` — both
    # non-positive since ``gamma1 - gamma2 = 2(1 - ssa) >= 0`` and
    # ``cosh x >= 1``), so the upper clip never acts. The lower clip removes
    # the small negative reflectance the Eddington closure produces for weakly
    # scattering layers, where ``gamma2 < 0`` for ``ssa < 1/(4 - 3g)`` — an
    # approximation artefact, not a physical value.
    R_dif = jnp.clip(R_dif, 0.0, 1.0)
    T_dif = jnp.clip(T_dif, 0.0, 1.0)
    
    if mu0 is not None:
        # Direct beam transmittance (Beer's law); guard tau/mu0 from overflow
        mu0_safe = jnp.maximum(mu0, 0.01)
        tau_over_mu = jnp.clip(tau / mu0_safe, 0.0, 100.0)
        T_dir = jnp.exp(-tau_over_mu)

        # Direct-to-diffuse reflectance (guard denom from zero when ssa*gamma4 ≈ 1).
        # NOTE (#855): this single-scattering source is not energy-conserving.
        # ``gamma3 = (2 - 3*g*mu0)/4`` goes negative for forward-scattering
        # clouds at high sun (g=0.85, mu0=1 -> -0.14), R_dir clips to 0, and the
        # scattered fraction of the attenuated direct beam is then dropped
        # entirely — a thick conservative cloud reflects ~0 at TOA. The faithful
        # fix is the Toon et al. (1989) direct-beam source functions; it is a
        # separate defect from the diffuse layer solution corrected above (#848)
        # and is tracked in #855.
        denom_dir = jnp.maximum(1.0 - ssa * gamma4, 1e-8)
        R_dir = ssa * gamma3 * (1.0 - T_dir) / denom_dir
        R_dir = jnp.clip(R_dir, 0.0, 1.0)
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
    """Calculate SW fluxes for a single band."""
    nlev = tau.shape[0]
    
    # Calculate layer properties with solar angle
    R_dif, T_dif, R_dir, T_dir = layer_reflectance_transmittance(tau, ssa, g, cos_zenith)
    
    # Direct beam transmission through atmosphere
    # Calculate cumulative direct transmission from TOA
    direct_trans = jnp.cumprod(T_dir, axis=0)
    
    # Add TOA transmission
    direct_trans_full = jnp.concatenate([jnp.array([1.0]), direct_trans])
    
    # Direct flux at each level
    flux_direct = toa_flux * direct_trans_full
    
    # Diffuse radiation calculation
    # Source from direct beam scattering (split into upward/downward components)
    source_diffuse = toa_flux * R_dir * direct_trans_full[:-1]
    source_up = 0.5 * source_diffuse
    source_down = 0.5 * source_diffuse
    
    # Initialize diffuse fluxes
    flux_down_dif = jnp.zeros(nlev + 1)
    flux_up_dif = jnp.zeros(nlev + 1)

    # Initial surface reflection of direct beam only
    flux_up_dif = flux_up_dif.at[nlev].set(surface_albedo * flux_direct[nlev])

    # Upward diffuse calculation
    def upward_diffuse_step(carry, x):
        flux_below = carry
        R, T, S = x
        flux_above = T * flux_below + S
        return flux_above, flux_above

    _, flux_up_levels = jax.lax.scan(
        upward_diffuse_step,
        flux_up_dif[nlev],
        (R_dif[::-1], T_dif[::-1], source_up[::-1])
    )
    flux_up_dif = flux_up_dif.at[:-1].set(flux_up_levels[::-1])

    # Downward diffuse calculation
    def downward_diffuse_step(carry, x):
        flux_above = carry
        R, T, S, flux_up = x
        flux_below = T * flux_above + R * flux_up + S
        return flux_below, flux_below

    _, flux_down_levels = jax.lax.scan(
        downward_diffuse_step,
        0.0,  # No diffuse at TOA
        (R_dif, T_dif, source_down, flux_up_dif[:-1])
    )
    flux_down_dif = flux_down_dif.at[1:].set(flux_down_levels)

    # CRITICAL FIX: Update surface upward flux to include diffuse reflection
    # Surface reflects both direct AND diffuse downward radiation
    flux_up_dif = flux_up_dif.at[nlev].set(
        surface_albedo * (flux_direct[nlev] + flux_down_dif[nlev])
    )

    # Recalculate upward diffuse with correct surface boundary condition
    _, flux_up_levels = jax.lax.scan(
        upward_diffuse_step,
        flux_up_dif[nlev],
        (R_dif[::-1], T_dif[::-1], source_up[::-1])
    )
    flux_up_dif = flux_up_dif.at[:-1].set(flux_up_levels[::-1])

    # Recalculate downward diffuse with updated upward flux for consistency
    _, flux_down_levels = jax.lax.scan(
        downward_diffuse_step,
        0.0,  # No diffuse at TOA
        (R_dif, T_dif, source_down, flux_up_dif[:-1])
    )
    flux_down_dif = flux_down_dif.at[1:].set(flux_down_levels)
    
    # Total fluxes
    flux_down_total = flux_direct + flux_down_dif
    flux_up_total = flux_up_dif  # No upward direct
    
    return flux_up_total, flux_down_total, flux_direct, flux_down_dif


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
