r"""Hines (1997) Doppler-spread spectral non-orographic gravity-wave drag.

Algorithm (column-mode, executed at each grid column):

1. **Brunt-Vaisala profile.** From temperature and full-level pressure,
   compute :math:`N(z)` via a sigma-coordinate finite difference of
   ``T / Pi^kappa``, then apply a single log-pressure smoother.

2. **Wave launch.** A spectrum of gravity waves is launched at a fixed
   "launch level" (counted up from the surface; default 10 levels above
   ground). The launch wind variance is split equally across 8 azimuths
   (E, NE, N, NW, W, SW, S, SE) with total RMS wind ``rms_launch_wind``
   (default 1 m/s).

3. **Doppler-spread wavenumber sweep.** Working from the launch level
   upward, at each level we evaluate the cutoff vertical wavenumber
   :math:`m_\\alpha` per azimuth. The Doppler-spread theory (Hines 1997)
   sets :math:`m_\\alpha = N / (f_1 \\sigma_\\alpha + f_2 \\sigma_t +
   v_\\alpha)`, where :math:`\\sigma_\\alpha` is the per-azimuth RMS wind
   variance from the level below, :math:`\\sigma_t` is the total RMS
   wind, and :math:`v_\\alpha` is the background wind projected onto the
   azimuth (relative to the launch level). A floor at ``min_vertical_wavenumber``
   prevents the cutoff from collapsing to zero. The variances at each
   level are then recomputed from a closed-form integral of the
   saturated spectrum.

4. **Vertical smoothing.** ``m_alpha`` and ``sigma_t`` are smoothed
   ``smoothing_passes`` times with a 1-2-1 stencil before the flux
   calculation, suppressing grid-scale noise.

5. **Flux + drag.** Per-azimuth momentum flux is
   :math:`F_\\alpha = a_\\alpha k_* (m_\\alpha - m_{\\min})`, weighted by
   the launch-level density. Zonal and meridional fluxes follow from the
   azimuth projection. Drag is the negative vertical divergence of flux,
   normalised by layer mass.

6. **Heating + diffusion** (when ``compute_heating=True``). The energy
   dissipation rate per unit mass is :math:`-f_5 \\sum_\\alpha
   d F_\\alpha / d z \\cdot (f_1 \\sigma_\\alpha + N / m_\\text{sub})`,
   with :math:`m_\\text{sub}` set to the smaller of the turbulence-limited
   and molecular-viscosity-limited cutoff wavenumbers.

**Implementation choices and gaps:**

- The four optional Hines branches in the ECHAM source are not
  implemented and intentionally omitted: the latitude-dependent launch
  variance (``lrmscon_lat``), the precipitation-modulated launch
  variance (``lozpr``), the frontal source (``lfront``), and the
  exponential damping above ``alt_cutoff`` (``icutoff``). All four are
  disabled by default in ECHAM-A configurations.

- The orographic-wave coupling (``sigsqmcw``) is hard-coded to zero —
  this matches the production ECHAM control flow in which orographic
  waves are handled by the separate Lott-Miller scheme rather than fed
  back into the Hines spectrum.

- Only ``slope = 1`` and ``naz = 8`` azimuths are supported. The other
  permitted spectral slopes (1.5, 2) and 4-azimuth case are not ported.

- ``launch_level``, ``num_azimuths``, and ``smoothing_passes`` are
  passed as Python integer kwargs to :func:`hines_gwd` rather than
  living on the parameters tree, because they control loop bounds that
  must be static at JIT trace time.

- A small handful of bug-for-bug Fortran quirks are preserved (and
  flagged at the call sites with ``# Fortran-quirk`` comments) so the
  port is bit-exact against the reference implementation:
  the leftover-loop-variable in ``mair`` indexing at the top-level
  heating term, and the 0-based shift of various 1-based loop bounds.
"""
import math
from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import lax
import tree_math

import jcm.constants as c

# 8-azimuth case uses cos(45°) projections; precompute as a Python float
# so it folds into XLA constants. ``math`` rather than ``jnp``: a jnp op at
# module level initialises the JAX backend on import (#859). The weakly-typed
# Python float is cast to the working dtype at use, so float32 physics sees the
# same correctly-rounded value the jnp.cos form produced.
_COS_PI_4 = math.cos(math.pi / 4)


@tree_math.struct
class HinesParameters:
    r"""Tunable parameters for the Hines (1997) GWD scheme.

    All fields below flow through the JAX trace as leaves (gradients,
    vmap, etc. all see them). Static loop knobs (``launch_level``,
    ``num_azimuths``, ``smoothing_passes``) live as kwargs on
    :func:`hines_gwd`.

    Attributes:
        rms_launch_wind: Total RMS wind variance of the launch spectrum
            (m/s). Main amplitude knob — typical values 0.5-2 m/s.
            Default 1.0.
        typical_horizontal_wavenumber: Characteristic horizontal
            wavenumber :math:`k_*` of the launched spectrum (1/m).
            Default 5e-5.
        min_vertical_wavenumber: Lower bound on the cutoff vertical
            wavenumber :math:`m_{\\min}` (1/m). Default 1e-4.
        compute_heating: If 1.0, compute heating + diffusion (production
            default); if 0.0, momentum flux only. Default 1.0.
        wave_amplitude_factor: :math:`f_1` in Hines (1997) — multiplies
            :math:`\\sigma_\\alpha` in the cutoff-wavenumber denominator.
            Controls the wave amplitude at which the doppler-shifted
            wavenumber saturates. Default 1.5.
        spectrum_width_factor: :math:`f_2` — multiplies :math:`\\sigma_t`
            in the same denominator. Controls how the total RMS wind
            broadens the spectrum. Default 0.3.
        mol_diffusion_factor: :math:`f_3` — divides the molecular-
            viscosity-limited cutoff wavenumber. Default 1.0.
        heating_efficiency: :math:`f_5` — scales the gravity-wave
            heating rate. Default 1.0.
        diffusion_efficiency: :math:`f_6` — scales the gravity-wave
            diffusion coefficient. Default 0.5.
        cutoff_altitude: Altitude (m) above which an exponential-decay
            damping is applied if ``cutoff_enabled`` is true. The
            cutoff is currently never enabled in production; this
            field is kept for API compatibility. Default 105e3.
        smoothing_coeff: Centre-weight of the 1-coeff-1 vertical
            smoother applied to ``m_alpha`` and ``sigma_t``. Default 2.0.

    """

    rms_launch_wind: jnp.ndarray
    typical_horizontal_wavenumber: jnp.ndarray
    min_vertical_wavenumber: jnp.ndarray
    compute_heating: jnp.ndarray
    wave_amplitude_factor: jnp.ndarray
    spectrum_width_factor: jnp.ndarray
    mol_diffusion_factor: jnp.ndarray
    heating_efficiency: jnp.ndarray
    diffusion_efficiency: jnp.ndarray
    cutoff_altitude: jnp.ndarray
    smoothing_coeff: jnp.ndarray

    @classmethod
    def default(
        cls,
        rms_launch_wind: float = 1.0,
        typical_horizontal_wavenumber: float = 5e-5,
        min_vertical_wavenumber: float = 1e-4,
        compute_heating: bool = True,
        wave_amplitude_factor: float = 1.5,
        spectrum_width_factor: float = 0.3,
        mol_diffusion_factor: float = 1.0,
        heating_efficiency: float = 1.0,
        diffusion_efficiency: float = 0.5,
        cutoff_altitude: float = 105e3,
        smoothing_coeff: float = 2.0,
    ) -> "HinesParameters":
        return cls(
            rms_launch_wind=jnp.asarray(rms_launch_wind),
            typical_horizontal_wavenumber=jnp.asarray(typical_horizontal_wavenumber),
            min_vertical_wavenumber=jnp.asarray(min_vertical_wavenumber),
            compute_heating=jnp.asarray(1.0 if compute_heating else 0.0),
            wave_amplitude_factor=jnp.asarray(wave_amplitude_factor),
            spectrum_width_factor=jnp.asarray(spectrum_width_factor),
            mol_diffusion_factor=jnp.asarray(mol_diffusion_factor),
            heating_efficiency=jnp.asarray(heating_efficiency),
            diffusion_efficiency=jnp.asarray(diffusion_efficiency),
            cutoff_altitude=jnp.asarray(cutoff_altitude),
            smoothing_coeff=jnp.asarray(smoothing_coeff),
        )


class HinesState(NamedTuple):
    """Diagnostic outputs from the Hines scheme.

    Attributes:
        flux_u: zonal momentum flux profile (Pa), full levels.
        flux_v: meridional momentum flux profile (Pa), full levels.
        diffco: vertical diffusion coefficient (m^2/s), full levels.

    """

    flux_u: jnp.ndarray
    flux_v: jnp.ndarray
    diffco: jnp.ndarray


class HinesTendencies(NamedTuple):
    """Tendencies from the Hines scheme (full-level arrays).

    Attributes:
        dudt: zonal-wind tendency (m/s^2).
        dvdt: meridional-wind tendency (m/s^2).
        dissip: energy dissipation rate per unit mass (W/kg = J/(s·kg)).
            Convert to a temperature tendency via dT/dt = dissip / cp.

    """

    dudt: jnp.ndarray
    dvdt: jnp.ndarray
    dissip: jnp.ndarray


# ---------------------------------------------------------------------------
# Helper: 8-azimuth wind projection
# ---------------------------------------------------------------------------

def _project_winds_eight_azimuths(u: jnp.ndarray, v: jnp.ndarray
                                  ) -> jnp.ndarray:
    """Project ``(u, v)`` onto 8 azimuthal directions.

    Output ordering (counterclockwise from east): 0=E, 1=NE, 2=N, 3=NW,
    4=W, 5=SW, 6=S, 7=SE. A floor of 1 mm/s is applied to each component
    and to the diagonal projections so divisions by zero don't appear in
    later stages. Returns shape ``(nlev, 8)``.
    """
    umin = 0.001
    u = jnp.where(jnp.abs(u) < umin, jnp.copysign(umin, u), u)
    v = jnp.where(jnp.abs(v) < umin, jnp.copysign(umin, v), v)
    v_plus_u = v + u
    v_plus_u = jnp.where(jnp.abs(v_plus_u) < umin,
                         jnp.copysign(umin, v_plus_u), v_plus_u)
    v_minus_u = v - u
    v_minus_u = jnp.where(jnp.abs(v_minus_u) < umin,
                          jnp.copysign(umin, v_minus_u), v_minus_u)

    east = u
    north_east = _COS_PI_4 * v_plus_u
    north = v
    north_west = _COS_PI_4 * v_minus_u
    return jnp.stack(
        [east, north_east, north, north_west,
         -east, -north_east, -north, -north_west],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Helper: total + per-azimuth RMS wind from sigsqh_alpha (8-azimuth case)
# ---------------------------------------------------------------------------

def _rms_total_and_per_azimuth_8(squared_amplitude: jnp.ndarray
                                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute total + per-azimuth RMS wind speeds at one level.

    Input ``squared_amplitude`` has shape ``(8,)`` and contains the
    per-azimuth wind variance contributions. Returns a scalar
    ``sigma_t`` (total RMS) and an array ``sigma_alpha`` of shape
    ``(8,)`` with each azimuth's RMS wind. Mirrors the 8-azimuth branch
    of the ECHAM ``hines_sigma`` routine: opposite-azimuth pairs share
    the same RMS (5..8 mirror 1..4 in 1-based ordering, i.e. 4..7 mirror
    0..3 in 0-based).
    """
    s = squared_amplitude
    sum_odd = (s[0] + s[2] + s[4] + s[6]) * 0.5
    sum_even = (s[1] + s[3] + s[5] + s[7]) * 0.5
    sa1 = jnp.sqrt(s[0] + s[4] + sum_even)
    sa2 = jnp.sqrt(s[1] + s[5] + sum_odd)
    sa3 = jnp.sqrt(s[2] + s[6] + sum_even)
    sa4 = jnp.sqrt(s[3] + s[7] + sum_odd)
    sigma_alpha = jnp.stack([sa1, sa2, sa3, sa4, sa1, sa2, sa3, sa4], axis=0)
    sigma_t = jnp.sqrt(jnp.sum(s, axis=0))
    return sigma_t, sigma_alpha


# ---------------------------------------------------------------------------
# Helper: Hines integral I_alpha (slope = 1)
# ---------------------------------------------------------------------------

def _hines_integral_slope1(
    azimuth_wind: jnp.ndarray,
    cutoff_wavenumber: jnp.ndarray,
    bv_freq_at_launch: jnp.ndarray,
    min_wavenumber: jnp.ndarray,
) -> jnp.ndarray:
    r"""Hines integral :math:`I_\alpha` for spectral slope = 1.

    This integral closes the level-by-level recursion in the doppler-
    spread sweep: the per-azimuth squared amplitude at level l is given
    by ``sigsqh_alpha[l] = sigfac[l] * ak_alpha * I_alpha``. Two
    branches: an analytic logarithmic form for typical conditions, and
    a 4-term Taylor expansion when ``|q*m| < 0.01`` or ``|q| < 1``
    where the analytic form is numerically unstable.

    The result is forced non-negative and zeroed where ``cutoff_wavenumber
    <= min_wavenumber`` (which is also where the per-azimuth do_alpha
    mask in the Fortran reference would be turned off).

    Args:
        azimuth_wind: background wind projected onto the azimuth (m/s).
        cutoff_wavenumber: per-azimuth cutoff vertical wavenumber (1/m).
        bv_freq_at_launch: scalar Brunt-Vaisala frequency at the launch
            level (rad/s).
        min_wavenumber: floor on the cutoff wavenumber (1/m).

    """
    q_min = 1.0
    qm_min = 0.01

    inv_bvfb = 1.0 / bv_freq_at_launch
    q_alpha = azimuth_wind * inv_bvfb
    qm = q_alpha * cutoff_wavenumber
    qmm = q_alpha * min_wavenumber

    # Analytic branch (slope=1): -(log(1-qm) - (1-qm) - log(1-qmm) + (1-qmm)) / q^2
    inv_q2 = 1.0 / (q_alpha * q_alpha + 1e-300)
    one_m_qm = 1.0 - qm
    one_m_qmm = 1.0 - qmm
    analytic = -(jnp.log(jnp.maximum(one_m_qm, 1e-300))
                 - one_m_qm
                 - jnp.log(jnp.maximum(one_m_qmm, 1e-300))
                 + one_m_qmm) * inv_q2

    # Taylor branch (Horner-evaluated to keep f32 accuracy):
    one_third = 1.0 / 3.0
    poly_qm = qm * qm * (0.5 + qm * (one_third + qm * (0.25 + qm * 0.2)))
    poly_qmm = qmm * qmm * (0.5 + qmm * (one_third + qmm * (0.25 + qmm * 0.2)))
    taylor = jnp.where(
        jnp.abs(q_alpha) < 1e-30,
        0.5 * (cutoff_wavenumber ** 2 - min_wavenumber ** 2),
        (poly_qm - poly_qmm) * inv_q2,
    )

    use_taylor = (jnp.abs(qm) < qm_min) | (jnp.abs(q_alpha) < q_min)
    integral = jnp.where(use_taylor, taylor, analytic)

    integral = jnp.maximum(integral, 0.0)
    integral = jnp.where(cutoff_wavenumber <= min_wavenumber, 0.0, integral)
    return integral


# ---------------------------------------------------------------------------
# Helper: 1-coeff-1 vertical smoother, applied N times, interior only
# ---------------------------------------------------------------------------

def _project_winds_eight_azimuths_axis1(
    u: jnp.ndarray, v: jnp.ndarray,
) -> jnp.ndarray:
    """Project ``(u, v)`` of shape ``(nlev, *horiz)`` onto ``(nlev, 8, *horiz)``."""
    umin = 0.001
    u = jnp.where(jnp.abs(u) < umin, jnp.copysign(umin, u), u)
    v = jnp.where(jnp.abs(v) < umin, jnp.copysign(umin, v), v)
    v_plus_u = v + u
    v_plus_u = jnp.where(
        jnp.abs(v_plus_u) < umin, jnp.copysign(umin, v_plus_u), v_plus_u
    )
    v_minus_u = v - u
    v_minus_u = jnp.where(
        jnp.abs(v_minus_u) < umin, jnp.copysign(umin, v_minus_u), v_minus_u
    )
    east = u
    north_east = _COS_PI_4 * v_plus_u
    north = v
    north_west = _COS_PI_4 * v_minus_u
    return jnp.stack(
        [east, north_east, north, north_west,
         -east, -north_east, -north, -north_west],
        axis=1,
    )


def _vertical_smoother(
    arr: jnp.ndarray,
    centre_weight: float,
    num_passes: int,
    top_index: int,
    bottom_index: int,
) -> jnp.ndarray:
    """Apply a 1-``centre_weight``-1 smoother ``num_passes`` times.

    Only indices ``top_index+1 .. bottom_index-1`` along axis 0 are averaged
    with their immediate neighbours; the boundary cells ``top_index`` and
    ``bottom_index`` keep their original values, and indices outside
    ``[top_index, bottom_index]`` are not touched. Index 0 = top. Trailing
    axes (azimuths, columns) are smoothed in lockstep without scatters.
    """
    if bottom_index <= top_index + 1 or top_index < 0:
        return arr
    sum_wts = centre_weight + 2.0
    prefix = arr[:top_index]
    window = arr[top_index:bottom_index + 1]
    suffix = arr[bottom_index + 1:]

    def one_pass(w):
        if w.shape[0] <= 2:
            return w
        smoothed = (w[:-2] + centre_weight * w[1:-1] + w[2:]) / sum_wts
        return jnp.concatenate([w[:1], smoothed, w[-1:]], axis=0)

    if isinstance(num_passes, int):
        for _ in range(num_passes):
            window = one_pass(window)
    else:
        window = lax.fori_loop(0, num_passes, lambda _, w: one_pass(w), window)
    return jnp.concatenate([prefix, window, suffix], axis=0)


# ---------------------------------------------------------------------------
# Helper: Brunt-Vaisala frequency from T profile (column or batch of columns)
# ---------------------------------------------------------------------------

def _brunt_vaisala(temperature: jnp.ndarray, pressure_full: jnp.ndarray,
                   surface_pressure: jnp.ndarray) -> jnp.ndarray:
    """Brunt-Vaisala frequency at full levels for ``(nlev, *horiz)`` inputs.

    Works in sigma = p/p_sfc coordinates and uses a finite-difference
    estimate of dT/dsigma against potential-temperature-like
    ``T / Pi^kappa``, then converts to N. Index 0 = top. Followed by a
    single forward log-pressure smoother to suppress noise.
    """
    rgocp = c.rd / c.cpd
    sigma = pressure_full / surface_pressure
    pi_kappa = sigma ** rgocp

    # Layer-centred dT/dsigma (lines 289-291 of the Fortran reference).
    dT_dsigma = ((temperature[1:] / pi_kappa[1:]
                  - temperature[:-1] / pi_kappa[:-1])
                 / (sigma[1:] - sigma[:-1]))
    dT_dsigma = jnp.minimum(dT_dsigma, -5.0 / sigma[1:])
    dT_dsigma = dT_dsigma * pi_kappa[1:]
    bvf2 = -dT_dsigma * sigma[1:] / c.rd
    bvf2 = jnp.maximum(bvf2, 0.0)
    bvfreq_inner = jnp.sqrt(bvf2) * c.grav / temperature[1:]

    # Pad index 0 with index 1 (Fortran line: ``bvfreq(:,1) = bvfreq(:,2)``).
    bvfreq = jnp.concatenate([bvfreq_inner[:1], bvfreq_inner], axis=0)

    # Single-pass log-sigma smoother (lines 304-309 of the reference).
    def smooth_step(prev, xs):
        ratio, bvf_k = xs
        cur = (prev + ratio * bvf_k) / (1.0 + ratio)
        return cur, cur

    ratios = 5.0 * jnp.log(sigma[1:] / sigma[:-1])
    _, smoothed = lax.scan(smooth_step, bvfreq[0], (ratios, bvfreq[1:]))
    return jnp.concatenate([bvfreq[:1], smoothed], axis=0)


# ---------------------------------------------------------------------------
# Main column algorithm (slope=1, naz=8)
# ---------------------------------------------------------------------------

def _hines_extro_column(
    bv_freq: jnp.ndarray,
    density: jnp.ndarray,
    layer_mass: jnp.ndarray,
    u_relative: jnp.ndarray,
    v_relative: jnp.ndarray,
    rms_launch: jnp.ndarray,
    config: HinesParameters,
    launch_idx: int,
    num_azimuths: int,
    smoothing_passes: int,
):
    """Doppler-spread sweep + flux + drag + heating for a single column.

    Args:
        bv_freq: Brunt-Vaisala frequency at full levels (rad/s),
            shape (nlev,). Index 0 = top.
        density: full-level air density (kg/m^3), shape (nlev,).
        layer_mass: full-level air mass per unit area (kg/m^2), shape
            (nlev,). Used to convert flux divergence into drag.
        u_relative, v_relative: zonal/meridional wind components
            relative to their values at the launch level (m/s), shape
            (nlev,). Above the launch level these carry the wave
            propagation; below, they are zero.
        rms_launch: scalar total RMS launch wind variance (m/s).
        config: tunable Hines parameters.
        launch_idx: 0-based index of the launch level (lower index =
            higher in the atmosphere). Static — must be a Python int.
        num_azimuths: number of azimuthal directions; must be 8.
        smoothing_passes: number of 1-2-1 smoother passes on m_alpha
            and sigma_t. Must be a Python int (for the fori_loop).

    Returns:
        ``(drag_u, drag_v, heating, diffco, flux_u, flux_v)``: all
        full-level arrays of shape (nlev,). drag in m/s^2, heating in
        W/kg, diffco in m^2/s, flux in Pa.

    """
    nlev = bv_freq.shape[0]
    horiz = bv_freq.shape[1:]
    dtype = bv_freq.dtype
    f_amp = config.wave_amplitude_factor
    f_width = config.spectrum_width_factor
    f_mol = config.mol_diffusion_factor
    f_heat = config.heating_efficiency
    f_diff = config.diffusion_efficiency
    k_horiz = config.typical_horizontal_wavenumber
    m_min = config.min_vertical_wavenumber
    visc_min = 1.0e-10
    visc_const = jnp.maximum(jnp.asarray(1.5e-5, dtype=dtype), visc_min)

    bv_freq_launch = bv_freq[launch_idx]
    density_launch = density[launch_idx]

    # Project winds onto 8 azimuths at every level: shape (nlev, 8, *horiz),
    # keeping (8, ncols) as the minor axes for 1.00x TPU tile density.
    azimuth_wind = _project_winds_eight_azimuths_axis1(u_relative, v_relative)

    # --- Initial conditions at the launch level --------------------------
    rms_b = jnp.broadcast_to(jnp.asarray(rms_launch, dtype=dtype), horiz)
    sqr_rms = rms_b * rms_b
    sqamp_launch = jnp.broadcast_to(
        sqr_rms / num_azimuths, (num_azimuths,) + horiz
    )

    sigma_t_launch, sigma_a_launch = _rms_total_and_per_azimuth_8(sqamp_launch)
    m_min_sq = m_min ** 2
    m_alpha_launch = bv_freq_launch / (
        f_amp * sigma_a_launch + f_width * sigma_t_launch)
    spectral_amplitude = (
        2.0 * sqamp_launch / (m_alpha_launch ** 2 - m_min_sq))

    # --- Sweep upward from launch_idx-1 down to index 0 ------------------
    # Carry only the previous level's state and emit per-level profiles via
    # ``ys`` instead of dynamic slice updates into full-column buffers.
    def step(carry, xs):
        sigma_t_below, sigma_a_below, m_alpha_min_running, do_alpha = carry
        bv_l, rho_l, az_wind_l = xs

        n_over_m_turb = f_width * sigma_t_below
        n_over_m_mol = jnp.cbrt(bv_l * k_horiz / visc_const) / f_mol
        n_over_m = jnp.where(
            bv_l / n_over_m_turb >= n_over_m_mol,
            bv_l / n_over_m_mol,
            n_over_m_turb,
        )

        m_trial = bv_freq_launch / (
            f_amp * sigma_a_below + n_over_m + az_wind_l)
        m_trial = jnp.where(m_trial <= 0.0, m_alpha_min_running, m_trial)
        m_trial = jnp.minimum(m_trial, m_alpha_min_running)
        m_trial = jnp.maximum(m_trial, m_min)
        m_alpha_l = jnp.where(do_alpha, m_trial, m_min)
        m_alpha_min_running_new = jnp.minimum(m_alpha_min_running, m_alpha_l)

        integral = _hines_integral_slope1(
            az_wind_l, m_alpha_l, bv_freq_launch, m_min)
        do_alpha_new = do_alpha & (m_alpha_l > m_min)

        sigfac = (density_launch / rho_l) * (bv_l / bv_freq_launch)
        sqamp_new = sigfac * spectral_amplitude * integral
        sigma_t_new, sigma_a_new = _rms_total_and_per_azimuth_8(sqamp_new)

        new_carry = (
            sigma_t_new, sigma_a_new, m_alpha_min_running_new, do_alpha_new,
        )
        return new_carry, (m_alpha_l, sigma_t_new, sigma_a_new)

    if launch_idx < 0:
        m_alpha = (
            jnp.full(
                (nlev, num_azimuths) + horiz, m_min, dtype=m_alpha_launch.dtype
            )
            .at[launch_idx]
            .set(m_alpha_launch)
        )
        sigma_t = (
            jnp.zeros((nlev,) + horiz, dtype=sigma_t_launch.dtype)
            .at[launch_idx]
            .set(sigma_t_launch)
        )
        sigma_alpha = (
            jnp.zeros((nlev, num_azimuths) + horiz, dtype=sigma_a_launch.dtype)
            .at[launch_idx]
            .set(sigma_a_launch)
        )
    else:
        init_do_alpha = jnp.full((num_azimuths,) + horiz, True)
        init = (sigma_t_launch, sigma_a_launch, m_alpha_launch, init_do_alpha)
        scan_xs = (
            bv_freq[:launch_idx],
            density[:launch_idx],
            azimuth_wind[:launch_idx],
        )
        _, (m_alpha_top, sigma_t_top, sigma_a_top) = lax.scan(
            step, init, scan_xs, reverse=True,
        )

        n_below = nlev - 1 - launch_idx
        m_alpha = jnp.concatenate(
            [
                m_alpha_top,
                m_alpha_launch[None],
                jnp.full(
                    (n_below, num_azimuths) + horiz,
                    m_min,
                    dtype=m_alpha_launch.dtype,
                ),
            ],
            axis=0,
        )
        sigma_t = jnp.concatenate(
            [
                sigma_t_top,
                sigma_t_launch[None],
                jnp.zeros((n_below,) + horiz, dtype=sigma_t_launch.dtype),
            ],
            axis=0,
        )
        sigma_alpha = jnp.concatenate(
            [
                sigma_a_top,
                sigma_a_launch[None],
                jnp.zeros(
                    (n_below, num_azimuths) + horiz, dtype=sigma_a_launch.dtype
                ),
            ],
            axis=0,
        )

    # Per-column "is the spectrum still alive at this level?" mask.
    spectrum_alive = sigma_t > 1e-30

    # --- Vertical smoothing -----------------------------------------------
    # Smooth all 8 azimuths of m_alpha and sigma_t together in one pass.
    smoothing_coeff = config.smoothing_coeff
    if smoothing_passes > 0:
        combined = jnp.concatenate(
            [m_alpha, jnp.expand_dims(sigma_t, axis=1)], axis=1
        )
        combined = _vertical_smoother(
            combined, smoothing_coeff, smoothing_passes, 0, launch_idx
        )
        m_alpha = combined[:, :num_azimuths]
        sigma_t = combined[:, num_azimuths]

    # --- Per-azimuth flux + zonal/meridional projection -------------------
    spectral_amplitude_x_k = spectral_amplitude * k_horiz
    flux_per_az = (spectral_amplitude_x_k[None] * (m_alpha - m_min)
                   * density_launch)
    flux_u = (flux_per_az[:, 0] - flux_per_az[:, 4]
              + _COS_PI_4 * (flux_per_az[:, 1] - flux_per_az[:, 3]
                             - flux_per_az[:, 5] + flux_per_az[:, 7]))
    flux_v = (flux_per_az[:, 2] - flux_per_az[:, 6]
              + _COS_PI_4 * (flux_per_az[:, 1] + flux_per_az[:, 3]
                             - flux_per_az[:, 5] - flux_per_az[:, 7]))

    # --- Drag from flux divergence ---------------------------------------
    # Top level uses a one-sided forward difference; below-launch levels
    # are masked to zero further down.
    drag_u_int = -(flux_u[:-1] - flux_u[1:]) / layer_mass[1:]
    drag_v_int = -(flux_v[:-1] - flux_v[1:]) / layer_mass[1:]
    drag_u_top = flux_u[:1] / layer_mass[:1]
    drag_v_top = flux_v[:1] / layer_mass[:1]
    drag_u = jnp.concatenate([drag_u_top, drag_u_int], axis=0)
    drag_v = jnp.concatenate([drag_v_top, drag_v_int], axis=0)

    below_launch_mask = (jnp.arange(nlev) > launch_idx).reshape(
        (nlev,) + (1,) * len(horiz)
    )
    drag_u = jnp.where(below_launch_mask, 0.0, drag_u)
    drag_v = jnp.where(below_launch_mask, 0.0, drag_v)

    # --- Heating + diffusion (when enabled) ------------------------------
    cutoff_mol = jnp.cbrt(bv_freq * k_horiz / visc_const) / f_mol
    turbulence_limited = f_width * sigma_t * cutoff_mol > bv_freq
    cutoff_turb = bv_freq / (
        f_width * jnp.where(turbulence_limited, sigma_t, 1.0))
    cutoff_eff = jnp.where(
        turbulence_limited, jnp.minimum(cutoff_turb, cutoff_mol), cutoff_mol)

    factor = f_amp * sigma_alpha + jnp.expand_dims(
        bv_freq / jnp.maximum(cutoff_eff, 1e-30), axis=1
    )
    dfdz_int = ((flux_per_az[:-1] - flux_per_az[1:])
                / jnp.expand_dims(layer_mass[1:], axis=1) * factor[1:])
    # Fortran-quirk: the top-level dfdz uses ``mair(i, l)`` where ``l`` is
    # leftover from the prior loop and equals ``launch_idx + 1`` after gfortran
    # increments the loop variable past lev2. We mirror the bug for bit-
    # exactness. ``factor[0]`` (sigma_alpha[0], cutoff_eff[0]) IS recomputed
    # at the top in the reference, so we use those values.
    dfdz_top = -flux_per_az[0] / layer_mass[launch_idx + 1] * factor[0]
    dfdz = jnp.concatenate([dfdz_top[None], dfdz_int], axis=0)

    heatng_raw = -f_heat * jnp.sum(dfdz, axis=1)
    heating = jnp.where(spectrum_alive, heatng_raw, 0.0)
    # Double-where on the cube root's argument: on a masked level (no
    # spectrum, or no heating) ``cbrt(maximum(heating, 0))`` is evaluated at
    # 0, where its derivative is infinite, and the ``where``'s zero cotangent
    # makes every ``diffco`` gradient NaN, so masked levels take the root of
    # 1 instead. Unmasked levels see their own heating: the value is the same.
    diffusing = spectrum_alive & (heatng_raw > 0.0)
    heating_diffusing = jnp.where(diffusing, heatng_raw, 1.0)
    diffco = jnp.where(
        diffusing,
        f_diff * jnp.cbrt(heating_diffusing)
        / jnp.maximum(cutoff_eff, 1e-30) ** (4.0 / 3.0),
        0.0,
    )
    heating = jnp.where(config.compute_heating > 0.5, heating, 0.0)
    diffco = jnp.where(config.compute_heating > 0.5, diffco, 0.0)

    return drag_u, drag_v, heating, diffco, flux_u, flux_v


def hines_gwd(
    pressure_half: jnp.ndarray,
    pressure_full: jnp.ndarray,
    height_half: jnp.ndarray,
    density: jnp.ndarray,
    layer_mass: jnp.ndarray,
    temperature: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    config: HinesParameters,
    *,
    launch_level: int = 10,
    num_azimuths: int = 8,
    smoothing_passes: int = 5,
) -> Tuple[HinesTendencies, HinesState]:
    """Compute Hines GWD tendencies for a single atmospheric column.

    Args:
        pressure_half: pressure on half levels (Pa), shape (nlev+1,).
            Index 0 = top, index nlev = surface.
        pressure_full: pressure on full levels (Pa), shape (nlev,).
        height_half: half-level geopotential height above sea level
            (m), shape (nlev+1,).
        density: full-level air density (kg/m^3), shape (nlev,).
        layer_mass: full-level air mass per unit area (kg/m^2), shape
            (nlev,) — i.e. ``Δp / g``.
        temperature: full-level temperature (K), shape (nlev,).
        u_wind, v_wind: full-level zonal/meridional wind (m/s), shape
            (nlev,).
        config: tunable :class:`HinesParameters`.
        launch_level: number of levels above the surface from which the
            wave spectrum is launched (1-based count). Default 10.
        num_azimuths: must be 8 (only branch implemented).
        smoothing_passes: number of vertical-smoothing passes applied
            to ``m_alpha`` and ``sigma_t`` before the flux calculation.
            Default 5.

    Returns:
        ``(tendencies, state)`` — see :class:`HinesTendencies` and
        :class:`HinesState` for field documentation.

    """
    nlev = u_wind.shape[0]
    horiz = u_wind.shape[1:]
    surface_pressure = pressure_half[-1]

    bv_freq = _brunt_vaisala(temperature, pressure_full, surface_pressure)

    # The 1-based ``levbot = nlev - launch_level`` becomes
    # ``nlev - launch_level - 1`` in 0-based indexing.
    launch_idx = nlev - launch_level - 1

    u_rel = u_wind - u_wind[launch_idx]
    v_rel = v_wind - v_wind[launch_idx]
    below_launch = (jnp.arange(nlev) > launch_idx).reshape(
        (nlev,) + (1,) * len(horiz)
    )
    u_rel = jnp.where(below_launch, 0.0, u_rel)
    v_rel = jnp.where(below_launch, 0.0, v_rel)

    drag_u, drag_v, heating, diffco, flux_u, flux_v = _hines_extro_column(
        bv_freq, density, layer_mass, u_rel, v_rel,
        config.rms_launch_wind, config,
        launch_idx, num_azimuths, smoothing_passes,
    )

    return (
        HinesTendencies(dudt=drag_u, dvdt=drag_v, dissip=heating),
        HinesState(flux_u=flux_u, flux_v=flux_v, diffco=diffco),
    )


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm import constants as _physical_constants  # noqa: E402
from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class HinesGwd(PhysicsTerm):
    """Hines (1997) doppler-spread spectral non-orographic GWD.

    Composable physics term wrapping :func:`hines_gwd`. Operates on
    column-vectorized state ``(nlev, ncols)``. Reads ``pressure_full``,
    ``pressure_half``, ``height_half``, ``air_density`` from the
    moist-air diagnostics dict, computes per-layer mass ``Δp / g`` on
    the fly, and writes only u/v/T tendencies (no Data sub-struct —
    Hines is a pure tendency producer).
    """

    name: ClassVar[str] = "hines_gwd"
    category: ClassVar[str] = "hines"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half",
        "height_half", "air_density",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: HinesParameters | None = None):
        """Hold the scheme-native :class:`HinesParameters`."""
        self.params = nnx.Param(params or HinesParameters.default())

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute u/v tendencies + heating from Hines (1997)."""
        params = self.params.get_value()
        pressure_full = diagnostics["pressure_full"]
        pressure_half = diagnostics["pressure_half"]
        height_half = diagnostics["height_half"]
        air_density = diagnostics["air_density"]

        # Evaluate directly in level-major ``(nlev, *horiz)`` layout without
        # column vmap or ``(ncols, nlev)`` transposes.
        layer_mass = (
            (pressure_half[1:] - pressure_half[:-1])
            / _physical_constants.grav
        )

        tend, _state = hines_gwd(
            pressure_half, pressure_full, height_half,
            air_density, layer_mass,
            state.temperature, state.u_wind, state.v_wind,
            params,
        )

        # Convert energy dissipation (W/kg) to a temperature tendency.
        dt_temperature = tend.dissip / _physical_constants.cpd

        return PhysicsTendency(
            u_wind=tend.dudt,
            v_wind=tend.dvdt,
            temperature=dt_temperature,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={},
        ), diagnostics
