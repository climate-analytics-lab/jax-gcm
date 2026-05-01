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
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import tree_math

from jcm.constants import grav, rd, cpd

# 8-azimuth case uses cos(45°) projections; precompute as a Python float
# so it folds into XLA constants.
_COS_PI_4 = float(jnp.cos(jnp.pi / 4))


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
    sigma_alpha = jnp.stack([sa1, sa2, sa3, sa4, sa1, sa2, sa3, sa4])
    sigma_t = jnp.sqrt(jnp.sum(s))
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

def _vertical_smoother(
    arr: jnp.ndarray,
    centre_weight: float,
    num_passes: int,
    top_index: int,
    bottom_index: int,
) -> jnp.ndarray:
    """Apply a 1-``centre_weight``-1 smoother ``num_passes`` times.

    Only indices ``top_index+1 .. bottom_index-1`` are averaged with
    their immediate neighbours; the boundary cells ``top_index`` and
    ``bottom_index`` keep their original values, and indices outside
    ``[top_index, bottom_index]`` are not touched. Index 0 = top.
    """
    sum_wts = centre_weight + 2.0
    interior = jnp.arange(top_index + 1, bottom_index)

    def one_pass(a):
        prev_vals = a[interior - 1]
        cur_vals = a[interior]
        next_vals = a[interior + 1]
        smoothed = (prev_vals + centre_weight * cur_vals + next_vals) / sum_wts
        return a.at[interior].set(smoothed)

    def body(_, a):
        return one_pass(a)

    return lax.fori_loop(0, num_passes, body, arr)


# ---------------------------------------------------------------------------
# Helper: Brunt-Vaisala frequency from T profile (column)
# ---------------------------------------------------------------------------

def _brunt_vaisala(temperature: jnp.ndarray, pressure_full: jnp.ndarray,
                   surface_pressure: jnp.ndarray) -> jnp.ndarray:
    """Brunt-Vaisala frequency at full levels from a single column.

    Works in sigma = p/p_sfc coordinates and uses a finite-difference
    estimate of dT/dsigma against potential-temperature-like
    ``T / Pi^kappa``, then converts to N. Index 0 = top. Followed by a
    single forward log-pressure smoother to suppress noise.
    """
    rgocp = rd / cpd
    sigma = pressure_full / surface_pressure
    pi_kappa = sigma ** rgocp

    # Layer-centred dT/dsigma (lines 289-291 of the Fortran reference).
    dT_dsigma = ((temperature[1:] / pi_kappa[1:]
                  - temperature[:-1] / pi_kappa[:-1])
                 / (sigma[1:] - sigma[:-1]))
    dT_dsigma = jnp.minimum(dT_dsigma, -5.0 / sigma[1:])
    dT_dsigma = dT_dsigma * pi_kappa[1:]
    bvf2 = -dT_dsigma * sigma[1:] / rd
    bvf2 = jnp.maximum(bvf2, 0.0)
    bvfreq_inner = jnp.sqrt(bvf2) * grav / temperature[1:]

    # Pad index 0 with index 1 (Fortran line: ``bvfreq(:,1) = bvfreq(:,2)``).
    bvfreq = jnp.concatenate([bvfreq_inner[:1], bvfreq_inner])

    # Single-pass log-sigma smoother (lines 304-309 of the reference).
    def smooth_step(carry, k):
        prev = carry
        ratio = 5.0 * jnp.log(sigma[k] / sigma[k - 1])
        cur = (prev + ratio * bvfreq[k]) / (1.0 + ratio)
        return cur, cur

    _, smoothed = lax.scan(smooth_step, bvfreq[0],
                           jnp.arange(1, bvfreq.shape[0]))
    return jnp.concatenate([bvfreq[:1], smoothed])


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
    f_amp = config.wave_amplitude_factor
    f_width = config.spectrum_width_factor
    f_mol = config.mol_diffusion_factor
    f_heat = config.heating_efficiency
    f_diff = config.diffusion_efficiency
    k_horiz = config.typical_horizontal_wavenumber
    m_min = config.min_vertical_wavenumber
    visc_min = 1.0e-10

    # Molecular viscosity profile — constant in this scheme (production
    # config never reaches altitudes where the real ν(z) matters).
    mol_visc = jnp.full((nlev,), 1.5e-5)

    bv_freq_launch = bv_freq[launch_idx]
    density_launch = density[launch_idx]

    # Project winds onto 8 azimuths at every level.
    azimuth_wind = jax.vmap(_project_winds_eight_azimuths)(
        u_relative, v_relative)

    # --- Initial conditions at the launch level --------------------------
    sqr_rms = rms_launch * rms_launch
    isotropic_anisotropy = jnp.full((num_azimuths,), 1.0 / num_azimuths)
    sqamp_launch = isotropic_anisotropy * sqr_rms                 # (8,)

    sigma_t_launch, sigma_a_launch = _rms_total_and_per_azimuth_8(sqamp_launch)
    m_min_sq = m_min ** 2
    m_alpha_launch = bv_freq_launch / (
        f_amp * sigma_a_launch + f_width * sigma_t_launch)
    spectral_amplitude = (
        2.0 * sqamp_launch / (m_alpha_launch ** 2 - m_min_sq))
    m_alpha_min_running_launch = m_alpha_launch                    # (8,)

    # Initialise full column buffers, indexed top..bottom.
    m_alpha = jnp.full((nlev, num_azimuths), m_min)
    m_alpha = m_alpha.at[launch_idx].set(m_alpha_launch)
    sigma_alpha = jnp.zeros((nlev, num_azimuths))
    sigma_alpha = sigma_alpha.at[launch_idx].set(sigma_a_launch)
    sqamp = jnp.zeros((nlev, num_azimuths))
    sqamp = sqamp.at[launch_idx].set(sqamp_launch)
    sigma_t = jnp.zeros((nlev,))
    sigma_t = sigma_t.at[launch_idx].set(sigma_t_launch)

    # --- Sweep upward from launch_idx-1 down to index 0 ------------------
    # At each iteration, level l is processed using sigma_t and
    # sigma_alpha at the level immediately below (already filled by the
    # previous step or by the launch initialisation).

    def step(carry, level):
        m_alpha, sigma_t, sigma_alpha, sqamp, m_alpha_min_running, do_alpha = carry
        below = level + 1

        # n_over_m: with the orographic-wave coupling switched off, the
        # ECHAM ``f2mfac`` correction vanishes and this reduces to
        # f_width * sigma_t at the level below.
        n_over_m_turb = f_width * sigma_t[below]
        # Molecular-viscosity-limited cutoff:
        visc = jnp.maximum(mol_visc[level], visc_min)
        n_over_m_mol = jnp.cbrt(bv_freq[level] * k_horiz / visc) / f_mol
        # Take the smaller cutoff (turbulence-limited or molecular-limited).
        n_over_m = jnp.where(
            bv_freq[level] / n_over_m_turb >= n_over_m_mol,
            bv_freq[level] / n_over_m_mol,
            n_over_m_turb,
        )

        # Trial cutoff wavenumber at this level:
        m_trial = bv_freq_launch / (
            f_amp * sigma_alpha[below] + n_over_m + azimuth_wind[level])
        m_trial = jnp.where(m_trial <= 0.0, m_alpha_min_running, m_trial)
        m_trial = jnp.minimum(m_trial, m_alpha_min_running)
        m_trial = jnp.maximum(m_trial, m_min)
        m_alpha_l = jnp.where(do_alpha, m_trial, m_min)
        m_alpha_min_running_new = jnp.minimum(m_alpha_min_running, m_alpha_l)

        # Hines integral, then update per-azimuth amplitude and RMS winds.
        integral = _hines_integral_slope1(
            azimuth_wind[level], m_alpha_l, bv_freq_launch, m_min)
        do_alpha_new = do_alpha & (m_alpha_l > m_min)

        sigfac = ((density_launch / density[level])
                  * (bv_freq[level] / bv_freq_launch))
        sqamp_new = sigfac * spectral_amplitude * integral
        sigma_t_new, sigma_a_new = _rms_total_and_per_azimuth_8(sqamp_new)

        m_alpha = m_alpha.at[level].set(m_alpha_l)
        sqamp = sqamp.at[level].set(sqamp_new)
        sigma_t = sigma_t.at[level].set(sigma_t_new)
        sigma_alpha = sigma_alpha.at[level].set(sigma_a_new)

        new_carry = (m_alpha, sigma_t, sigma_alpha, sqamp,
                     m_alpha_min_running_new, do_alpha_new)
        return new_carry, None

    init_do_alpha = jnp.full((num_azimuths,), True)
    indices = jnp.arange(launch_idx - 1, -1, -1)
    init = (m_alpha, sigma_t, sigma_alpha, sqamp,
            m_alpha_min_running_launch, init_do_alpha)
    (m_alpha, sigma_t, sigma_alpha, sqamp, _, _), _ = lax.scan(
        step, init, indices)

    # Per-column "is the spectrum still alive at this level?" mask.
    spectrum_alive = sigma_t > 1e-30

    # --- Vertical smoothing -----------------------------------------------
    # Only the interior between the model top and the launch level is
    # smoothed; boundaries stay fixed.
    smoothing_coeff = config.smoothing_coeff
    if smoothing_passes > 0:
        m_alpha = jnp.stack(
            [_vertical_smoother(m_alpha[:, n], smoothing_coeff,
                                smoothing_passes, 0, launch_idx)
             for n in range(num_azimuths)],
            axis=-1,
        )
        sigma_t = _vertical_smoother(sigma_t, smoothing_coeff,
                                     smoothing_passes, 0, launch_idx)

    # --- Per-azimuth flux + zonal/meridional projection -------------------
    horiz_wavenumber = jnp.full((num_azimuths,), k_horiz)
    spectral_amplitude_x_k = spectral_amplitude * horiz_wavenumber
    flux_per_az = (spectral_amplitude_x_k[None, :] * (m_alpha - m_min)
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
    drag_u_top = flux_u[0] / layer_mass[0]
    drag_v_top = flux_v[0] / layer_mass[0]
    drag_u = jnp.concatenate([jnp.array([drag_u_top]), drag_u_int])
    drag_v = jnp.concatenate([jnp.array([drag_v_top]), drag_v_int])

    below_launch_mask = jnp.arange(nlev) > launch_idx
    drag_u = jnp.where(below_launch_mask, 0.0, drag_u)
    drag_v = jnp.where(below_launch_mask, 0.0, drag_v)

    # --- Heating + diffusion (when enabled) ------------------------------
    visc_full = jnp.maximum(mol_visc, visc_min)
    cutoff_turb = bv_freq / (f_width * jnp.maximum(sigma_t, 1e-30))
    cutoff_mol = jnp.cbrt(bv_freq * k_horiz / visc_full) / f_mol
    cutoff_eff = jnp.minimum(cutoff_turb, cutoff_mol)

    factor = f_amp * sigma_alpha + (
        bv_freq / jnp.maximum(cutoff_eff, 1e-30))[:, None]
    dfdz_int = ((flux_per_az[:-1] - flux_per_az[1:])
                / layer_mass[1:, None] * factor[1:])
    # Fortran-quirk: the top-level dfdz uses ``mair(i, l)`` where ``l`` is
    # leftover from the prior loop and equals ``launch_idx + 1`` after gfortran
    # increments the loop variable past lev2. We mirror the bug for bit-
    # exactness. ``factor[0]`` (sigma_alpha[0], cutoff_eff[0]) IS recomputed
    # at the top in the reference, so we use those values.
    dfdz_top = -flux_per_az[0] / layer_mass[launch_idx + 1] * factor[0]
    dfdz = jnp.concatenate([dfdz_top[None, :], dfdz_int], axis=0)

    heatng_raw = -f_heat * jnp.sum(dfdz, axis=-1)
    heating = jnp.where(spectrum_alive, heatng_raw, 0.0)
    safe_heating = jnp.maximum(heatng_raw, 0.0)
    diffco = jnp.where(
        spectrum_alive & (heatng_raw > 0.0),
        f_diff * jnp.cbrt(safe_heating)
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
    surface_pressure = pressure_half[-1]

    bv_freq = _brunt_vaisala(temperature, pressure_full, surface_pressure)

    # The 1-based ``levbot = nlev - launch_level`` becomes
    # ``nlev - launch_level - 1`` in 0-based indexing.
    launch_idx = nlev - launch_level - 1

    u_rel = u_wind - u_wind[launch_idx]
    v_rel = v_wind - v_wind[launch_idx]
    below_launch = jnp.arange(nlev) > launch_idx
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
