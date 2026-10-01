r"""Gas-aerosol exchange: condensation of H2SO4 and SOAG onto the modes.

The formulation is MAM4's, taken from its own documentation
(``mam_doc_site/gas_aer_exch/``, pages ``eq_assumptions``,
``eq_mass_transfer_rate``, ``eq_SO4``, ``eq_SOA``, ``numerics_quadrature``,
``numerics_SO4``, ``numerics_SOA``; read 2026-10-01) rather than from the
Liu et al. (2012) supplement, which only names the references. Where the
two differ, the documentation describes the code that runs in E3SM and is
followed here; the supplement's "(Binkowski and Shankar, 1995)" names the
earlier CAM5 modal integration, which MAM4 has since replaced with the
Gauss-Hermite quadrature below.

Per particle (Fuchs and Sutugin, 1971; Seinfeld and Pandis 2016 eqs 12.12,
12.43), the molar flux to a particle of wet diameter :math:`D_p` is

.. math::

    \tilde J_L = 2\pi D_p \mathbb{D}_{g,L}\,(\tilde c_L - \tilde c_{L,sat})
                 \, f(K_n, \alpha),
    \qquad
    f = \frac{0.75\alpha(1+K_n)}{0.75\alpha + K_n(1 + K_n + 0.283\alpha)},
    \qquad K_n = \frac{2\lambda_L}{D_p},\; \lambda_L = \frac{3\mathbb{D}_{g,L}}{v_L}

with :math:`v_L = \sqrt{8RT/\pi M_L}` and the diffusivity from Fuller et
al. (1966). Integrated over a log-normal mode under the assumption that the
surface saturation concentration is uniform within the mode, the flux is
:math:`(q_L - q_{L,sat,i})\,C_{L,i}` with the mass-transfer coefficient

.. math::

    C_{L,i} = 2\pi\mathbb{D}_{g,L}\int f(K_n,\alpha)\,D_p\,n_i(\ln D_p)\,d\ln D_p

evaluated by two-point Gauss-Hermite quadrature after factoring the
integrand as :math:`[f D_p / D_p^\beta]\,D_p^\beta` with :math:`\beta` the
local log-slope of :math:`f D_p` at the mode's wet geometric-mean diameter
(:math:`1 \le \beta \le 2`, from the free-molecular to the continuum limit).

**Sulfuric acid is nonvolatile** in the tropospheric configuration. Its
linearised equation is solved analytically over the step, *including* the
step-mean chemical production :math:`\mathcal P^*` — MAM4 couples the
source and the sink because sequential splitting of the two at
climate-model time steps was shown to produce large errors (Wan et al.
2013; Wang et al. 2019). jcm's additive operator split would commit
exactly that error, so the chemistry term publishes its production and
this process reads it; the tendency returned here is the condensation
part only, so the two sum correctly.

**SOAG is semi-volatile.** Its transfer coefficient is 0.81 of sulfuric
acid's (the Fuller ratio at typical conditions, with the SOAG diffusion
volume scaled from H2SO4's by molecular weight). The equilibrium mixing
ratio over each mode follows Raoult's law over the SOA plus the 10 % of
primary organic aerosol assumed oxygenated, and the nonlinear system is
integrated with the semi-implicit Euler scheme and adaptive sub-steps of
``numerics_SOA`` — here inside a fixed-length ``lax.scan`` so the step
count is static and the whole process stays differentiable.

Everything internal is in **molar mixing ratios**, as in MAM; conversion to
the harness's kg/kg happens at the boundary of :func:`condense`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import struct

import jcm.constants as c

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

#: Molecular weight of dry air [kg/mol], MAM's gas-aerosol exchange Table 1.
M_AIR = 28.966e-3

#: Fuller et al. (1966) diffusion volumes, MAM Table 1. Dimensionless.
DIFFUSION_VOLUME_H2SO4 = 42.88
DIFFUSION_VOLUME_AIR = 20.1

#: Two-point Gauss-Hermite abscissae and weights for int exp(-y^2) g(y) dy.
_GH_ROOTS = (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
_GH_WEIGHTS = (math.sqrt(math.pi) / 2.0, math.sqrt(math.pi) / 2.0)

#: ``numerics_SO4``: thresholds on dt*C_total selecting the analytic
#: solution, its second-order Taylor expansion, or no condensation at all.
_ANALYTIC_ABOVE = 1.0e-3
_NEGLECT_BELOW = 1.0e-20

#: ``numerics_SOA`` safeguards: floors on the solvent amount and on the
#: gas/saturation mixing ratios used to size a sub-step [mol/mol].
#:
#: MAM uses 1e-20. That is NOT float32-safe: the reverse-mode derivative
#: of ``x / max(y, eps)`` carries ``1/eps^2``, and ``1e-40`` is below the
#: smallest float32 subnormal, so at an empty mode or with no gas the
#: cotangent is ``0/0``. Measured 2026-10-01: a NaN gradient in float32,
#: finite in float64 (where MAM runs and never meets it). 1e-15 mol/mol
#: squares to 1e-30, comfortably representable, and is still five orders
#: below any SOAG mixing ratio that matters. Every division by one of
#: these quantities also goes through :func:`_safe_divide`, so the branch
#: below the floor divides by 1.0 rather than by the floor.
_EPS_AEROSOL = 1.0e-15
_EPS_GAS = 1.0e-15

#: Floor on a summed uptake rate [1/s] below which it is treated as zero
#: in a division. 1e-18 squares to 1e-36, still a float32 normal; a total
#: coefficient that small puts ``dt C`` in the "neglect" regime anyway.
_RATE_FLOOR = 1.0e-18

#: Reference temperature of the SOAG equilibrium mixing ratio [K].
_T_REF_SOAG = 298.0


@struct.dataclass
class CondensationParameters:
    """Tunables of the gas-aerosol exchange, all differentiable leaves.

    Attributes:
        accommodation: sticking probability of H2SO4 on a particle surface,
            0.65 after Poschl et al. (1998); MAM applies it to SOAG too.
        soag_transfer_ratio: ``C_SOAG / C_H2SO4``, 0.81 in MAM4 (the
            Fuller diffusivity ratio at typical conditions).
        soag_equilibrium_ref: ambient equilibrium SOAG molar mixing ratio
            at 298 K and 1 atm [mol/mol], 1e-10 in MAM4.
        enthalpy_vaporization: mean SOAG enthalpy of vaporisation [J/mol],
            156 kJ/mol in MAM4.
        oxygenated_poa_fraction: fraction of primary organic aerosol taken
            to be oxygenated and so part of the SOA solution, 0.1.
        substep_fraction: the SOA sub-step may condense or evaporate at
            most this fraction of the existing/equilibrium amount under a
            forward-Euler estimate; MAM4's ``alpha = 0.05``.
        diffusion_volume_h2so4: Fuller diffusion volume of H2SO4.
        max_substeps: static length of the SOA sub-step scan. MAM4's loop
            is adaptive and unbounded; a bounded scan keeps the trace
            static and reverse-differentiable. When the adaptive estimate
            wants more steps than this, the last step is longer than the
            fraction allows — still unconditionally stable, less accurate —
            and ``condensation_soag_substep_shortfall`` reports the time
            the scan did not get to. **Measured 2026-10-01** on a
            continental-strength population (number ~1e9 kg^-1, dt 1800 s):
            the result became exactly scan-length-independent between 200
            and 400 sub-steps in every SOAG regime, and a cap of 10 changed
            SOA masses by 10-50 %. 400 is the default for that reason; a
            more polluted column scales the demand up with the total uptake
            coefficient, which is what the shortfall diagnostic is for.

    """

    accommodation: jnp.ndarray = 0.65
    soag_transfer_ratio: jnp.ndarray = 0.81
    soag_equilibrium_ref: jnp.ndarray = 1.0e-10
    enthalpy_vaporization: jnp.ndarray = 156.0e3
    oxygenated_poa_fraction: jnp.ndarray = 0.1
    substep_fraction: jnp.ndarray = 0.05
    diffusion_volume_h2so4: jnp.ndarray = DIFFUSION_VOLUME_H2SO4
    max_substeps: int = struct.field(pytree_node=False, default=400)


def gas_diffusivity(temperature, pressure, molar_mass, diffusion_volume):
    """Fuller et al. (1966) binary diffusivity in air [m^2/s].

    MAM's form: ``1e-7 T^1.75 sqrt(1/M_L + 1/M_air) / (p_atm (V_L^{1/3} +
    V_air^{1/3})^2)`` with molecular weights in g/mol and pressure in
    atmospheres. The 1e-7 (rather than the textbook 1e-3) delivers m^2/s.
    """
    m_gas_g = molar_mass * 1.0e3
    m_air_g = M_AIR * 1.0e3
    p_atm = pressure / 101325.0
    volumes = (diffusion_volume ** (1.0 / 3.0)
               + DIFFUSION_VOLUME_AIR ** (1.0 / 3.0)) ** 2
    return (1.0e-7 * temperature ** 1.75
            * jnp.sqrt(1.0 / m_gas_g + 1.0 / m_air_g)
            / (p_atm * volumes))


def mean_molecular_speed(temperature, molar_mass):
    """``sqrt(8RT / (pi M))`` [m/s]."""
    return jnp.sqrt(8.0 * c.r_universal * temperature / (math.pi * molar_mass))


def fuchs_sutugin(knudsen, accommodation):
    """Transition-regime correction ``f(Kn, alpha)`` of Fuchs and Sutugin."""
    return (0.75 * accommodation * (1.0 + knudsen)
            / (0.75 * accommodation
               + knudsen * (1.0 + knudsen + 0.283 * accommodation)))


def _log_slope_beta(knudsen, accommodation):
    """``beta = d ln(f D_p) / d ln D_p`` at the mode's wet diameter.

    ``numerics_quadrature``: ``1 - Kn [1/(1+Kn) - (1 + 2Kn + 0.283 alpha)
    / (0.75 alpha + Kn(1 + Kn + 0.283 alpha))]``. Tends to 2 in the
    free-molecular limit (``f D_p ~ D_p^2``) and 1 in the continuum
    limit (``f D_p ~ D_p``).
    """
    denominator = 0.75 * accommodation + knudsen * (
        1.0 + knudsen + 0.283 * accommodation)
    return 1.0 - knudsen * (
        1.0 / (1.0 + knudsen)
        - (1.0 + 2.0 * knudsen + 0.283 * accommodation) / denominator)


def mass_transfer_coefficient(number_density, wet_diameter, ln_sigma,
                              diffusivity, mean_free_path, accommodation):
    """``C_{L,i}`` [1/s] of one mode by two-point Gauss-Hermite quadrature.

    Args:
        number_density: particles per m^3 of air ``N_{t,i}``.
        wet_diameter: wet geometric-mean diameter ``D_{gn,w,i}`` [m].
        ln_sigma: ``ln sigma_g`` of the mode, a static Python float.
        diffusivity: gas diffusivity [m^2/s].
        mean_free_path: ``lambda_L`` [m].
        accommodation: ``alpha_L``.

    Returns:
        ``C_{L,i}`` such that the molar flux to the mode is
        ``(q_L - q_{L,sat,i}) C_{L,i}``.

    """
    x_gn = jnp.log(wet_diameter)
    beta = _log_slope_beta(2.0 * mean_free_path / wet_diameter, accommodation)
    shift = x_gn + beta * ln_sigma ** 2
    prefactor = (2.0 * math.sqrt(math.pi) * diffusivity * number_density
                 * jnp.exp(beta * x_gn + 0.5 * beta ** 2 * ln_sigma ** 2))
    total = 0.0
    for root, weight in zip(_GH_ROOTS, _GH_WEIGHTS):
        x = math.sqrt(2.0) * ln_sigma * root + shift
        diameter = jnp.exp(x)
        integrand = fuchs_sutugin(2.0 * mean_free_path / diameter,
                                  accommodation) * diameter
        total = total + weight * integrand / jnp.exp(beta * x)
    return prefactor * total


def _safe_divide(numerator, denominator, floor):
    """``numerator / denominator`` where the denominator exceeds ``floor``, else 0.

    Double ``where`` so the unselected branch never forms ``x/0`` — its
    cotangent would be NaN and would poison the gradient through the
    selected branch (see ``moments.diameter_from_volume``).
    """
    live = denominator > floor
    safe = jnp.where(live, denominator, 1.0)
    return jnp.where(live, numerator / safe, 0.0)


def solve_h2so4(q_gas, production, coefficients, dt):
    """Advance H2SO4 gas over ``dt`` against constant uptake coefficients.

    ``numerics_SO4``: the linearised ODE ``dq/dt = P* - q C_total`` with the
    coefficients frozen at the start of the step. Returned is the condensed
    amount ``q^n + P* dt - q^{n+1}`` (molar mixing ratio), chosen by the
    size of ``dt C_total``: analytic above 1e-3, second-order Taylor between
    1e-20 and 1e-3 (the exact form loses precision to cancellation there),
    and zero below.

    Args:
        q_gas: start-of-step H2SO4 molar mixing ratio ``q^n`` [mol/mol].
        production: step-mean chemical production ``P*`` [mol/mol/s].
        coefficients: ``(n_modes, *grid)`` per-mode ``C_{H2SO4,i}`` [1/s].
        dt: step [s].

    Returns:
        ``(condensed, fraction, mean, q_new)`` — the condensed molar mixing
        ratio, the end-of-step vapour (non-negative by construction),
        the per-mode partition ``C_i / C_total`` (zero where nothing
        condenses), and the time-mean vapour over the step. The mean is
        what new-particle formation evaluates its rate on
        (``numerics_coupling``): for the analytic regime it is the exact
        average of the linearised solution, ``s + (q⁰ − s)(1 − e^{−τ})/τ``;
        in the Taylor and neglect regimes the trapezoid of start and end,
        which the expansion is accurate to anyway.

    """
    total = jnp.sum(coefficients, axis=0)
    tau = dt * total

    source_over_sink = _safe_divide(production, total, _RATE_FLOOR)
    analytic = ((q_gas - source_over_sink) * jnp.exp(-tau) + source_over_sink)
    taylor = (q_gas * (1.0 - tau + 0.5 * tau ** 2)
              + dt * production * (1.0 - 0.5 * tau + tau ** 2 / 6.0))
    none = q_gas + dt * production

    q_new = jnp.where(tau > _ANALYTIC_ABOVE, analytic,
                      jnp.where(tau >= _NEGLECT_BELOW, taylor, none))
    condensed = q_gas + dt * production - q_new
    fraction = _safe_divide(coefficients, total[jnp.newaxis], _RATE_FLOOR)

    # Time-mean of the vapour over the step, for new-particle formation.
    # ``(1 − e^{−τ})/τ`` through -expm1 so it is 1 − τ/2 + … rather than a
    # cancellation as τ → 0; only used where the analytic branch is.
    analytic_mean = (source_over_sink
                     + (q_gas - source_over_sink)
                     * (-jnp.expm1(-tau)) / jnp.maximum(tau, _ANALYTIC_ABOVE))
    mean = jnp.where(tau > _ANALYTIC_ABOVE, analytic_mean,
                     0.5 * (q_gas + q_new))
    return condensed, fraction, mean, q_new


def soag_equilibrium(temperature, pressure, params):
    """Ambient equilibrium SOAG molar mixing ratio [mol/mol] (``eq_SOA``)."""
    return (params.soag_equilibrium_ref * (101325.0 / pressure)
            * jnp.exp(-params.enthalpy_vaporization / c.r_universal
                      * (1.0 / temperature - 1.0 / _T_REF_SOAG)))


def solve_soag(q_gas, q_soa, q_opoa, coefficients, q_equilibrium, dt, params):
    """Advance SOAG and per-mode SOA over ``dt`` (``numerics_SOA``).

    Semi-implicit (backward Euler) sub-steps with the solvent factor
    ``S*_i = q_equ / max(q_opoa,i + q*_soa,i, eps)`` frozen per sub-step,
    where ``q*_soa,i`` is the forward-Euler estimate for a condensing mode
    and the old value for an evaporating one. The sub-step length is the
    largest for which a forward-Euler step would move no mode by more than
    ``substep_fraction`` of its existing or equilibrium amount, capped by
    the time remaining. Total SOA (gas + every mode) is conserved exactly
    by construction of the closed form for ``q_gas^{new}``.

    The adaptive loop is unrolled into a fixed-length scan: a sub-step that
    finds no time remaining is a no-op, so the result is independent of
    ``max_substeps`` whenever the adaptive count fits inside it.

    All quantities molar [mol/mol]; ``coefficients`` and ``q_soa`` are
    ``(n_modes, *grid)``, the gas and ``dt`` are ``(*grid)``.

    Returns:
        ``(q_gas_new, q_soa_new, remaining)`` — ``remaining`` is the time
        [s] the scan did not reach because ``max_substeps`` bound; zero
        whenever the adaptive count fitted.

    """
    alpha = params.substep_fraction
    solvent_floor = _EPS_AEROSOL
    gas_floor = _EPS_GAS

    def substep(carry, _):
        gas, soa, remaining = carry

        # Equilibrium over each mode with the current SOA, Raoult's law.
        sat = (_safe_divide(soa, q_opoa + soa, solvent_floor)
               * q_equilibrium[jnp.newaxis])

        # Fractional driving force per mode and the sub-step it permits.
        scale = jnp.maximum(gas[jnp.newaxis], sat)
        phi = _safe_divide(gas[jnp.newaxis] - sat, scale, gas_floor)
        rate = jnp.sum(coefficients * jnp.abs(phi), axis=0)
        wanted = _safe_divide(alpha, rate, _RATE_FLOOR)
        wanted = jnp.where(rate > _RATE_FLOOR, wanted, remaining)
        step = jnp.minimum(wanted, remaining)

        # Solvent factor frozen over the sub-step: forward-Euler estimate of
        # the SOA for condensing modes, the old value for evaporating ones.
        forward = soa + step[jnp.newaxis] * coefficients * (gas[jnp.newaxis] - sat)
        soa_star = jnp.where(phi > 0.0, forward, soa)
        s_star = _safe_divide(q_equilibrium[jnp.newaxis], q_opoa + soa_star,
                              solvent_floor)

        # Backward Euler, closed over total-SOA conservation.
        beta = step[jnp.newaxis] * coefficients
        denominator = 1.0 + beta * s_star
        gas_new = ((gas + jnp.sum(soa, axis=0)
                    - jnp.sum(soa / denominator, axis=0))
                   / (1.0 + jnp.sum(beta / denominator, axis=0)))
        soa_new = (soa + beta * gas_new[jnp.newaxis]) / denominator
        return (gas_new, soa_new, remaining - step), None

    (gas, soa, remaining), _ = jax.lax.scan(
        substep, (q_gas, q_soa, jnp.full_like(q_gas, dt)),
        None, length=params.max_substeps)
    return gas, soa, remaining


def condense(population, env, spec, params):
    """Run the gas-aerosol exchange: population in, population out.

    Reads ``g_h2so4`` and ``g_soag`` from ``population.gases``, the
    step-mean H2SO4 chemical production from ``env.h2so4_production``, and
    the wet radii of the start-of-step κ-Köhler state from
    ``env.state_start``. Returns the population with gas and aerosol mass
    advanced, with the soluble mass condensed onto the primary-carbon mode
    routed to the accumulation mode and recorded in ``population.coating``
    for the ageing process.

    **Why pcm's condensate goes to accumulation.** The primary-carbon mode
    carries no sulfate or SOA slot (``mam4_data._PCARBON_SPECIES``), yet it
    has surface area and takes up vapour like any mode. MAM moves the
    condensed soluble species to accumulation "along with" the aged
    fraction of the carbon; the carbon's own transfer is AER-06's job and
    needs the amount deposited, which is what ``coating`` carries. Routing
    the soluble mass now rather than holding it is also what the
    ``mam4-jax`` adapter had to add to close a sulfur leak.
    """
    masses = dict(population.masses)
    gases = dict(population.gases)
    coating = dict(population.coating)
    diagnostics = dict(population.diagnostics)

    temperature, pressure, dt = env.temperature, env.pressure, env.dt
    zeros = jnp.zeros_like(temperature)
    m_h2so4 = GAS_SPECIES["h2so4"].molar_mass
    m_soag = GAS_SPECIES["soag"].molar_mass
    m_so4 = spec.species_props("so4").molar_mass
    to_molar_h2so4 = M_AIR / m_h2so4
    to_molar_soag = M_AIR / m_soag

    # --- transfer coefficients for H2SO4, one per mode, on WET diameters ---
    diffusivity = gas_diffusivity(temperature, pressure, m_h2so4,
                                  params.diffusion_volume_h2so4)
    mean_free_path = 3.0 * diffusivity / mean_molecular_speed(temperature, m_h2so4)
    coefficients = []
    for index, mode in enumerate(spec.modes):
        number_density = population.numbers[number_name(mode.short)] * env.air_density
        wet_diameter = 2.0 * env.state_start.r_wet[index]
        coefficients.append(mass_transfer_coefficient(
            number_density, wet_diameter, math.log(mode.geom_std_dev),
            diffusivity, mean_free_path, params.accommodation))
    coefficients = jnp.stack(coefficients, axis=0)

    # --- H2SO4: nonvolatile, with chemical production coupled in ---
    q_h2so4 = gases.get(gas_name("h2so4"), zeros) * to_molar_h2so4
    production = env.h2so4_production * to_molar_h2so4
    condensed, fraction, q_mean, q_new = solve_h2so4(
        q_h2so4, production, coefficients, dt)
    # The vapour becomes the SOLVED end-of-step value, which is
    # non-negative by construction. Subtracting ``condensed * sum(fraction)``
    # instead left a residual of -1.6e-17 kg/kg from a 1e-10 input at full
    # depletion (float32: the partition sums to 1 ± eps), and a negative
    # tracer poisoned the new-particle clipping downstream. The cost is a
    # conservation error of |sum(fraction) - 1| * condensed, i.e. float32
    # rounding, which the budget test tolerates; positivity is exact.
    # Note the gas tendency this produces is the condensation part only
    # (q_new is solved from q^n, the production enters as a separate term
    # from the chemistry's own tendency): q_new - q^n - dt P*.
    gases[gas_name("h2so4")] = (gases.get(gas_name("h2so4"), zeros)
                                + (q_new - q_h2so4 - dt * production)
                                / to_molar_h2so4)
    so4_per_mode = condensed[jnp.newaxis] * fraction * (m_so4 / M_AIR)
    # A fraction by construction; clipped because float32 rounding can put
    # it a few ulp outside [0, 1].
    diagnostics["condensation_h2so4_depletion"] = jnp.clip(_safe_divide(
        condensed, q_h2so4 + dt * production, _EPS_GAS), 0.0, 1.0)
    # The coupling quantities new-particle formation reads (molar mixing
    # ratio and 1/s); see ``numerics_coupling`` on MAM's page.
    diagnostics["condensation_h2so4_step_mean"] = q_mean
    diagnostics["condensation_h2so4_uptake_sum"] = jnp.sum(coefficients, axis=0)

    # --- SOAG: semi-volatile, Raoult's law over SOA + oxygenated POA ---
    q_soag = gases.get(gas_name("soag"), zeros) * to_molar_soag
    soa_modes, opoa_modes = [], []
    for mode in spec.modes:
        soa = masses.get(mass_name("soa", mode.short), zeros)
        poa = masses.get(mass_name("poa", mode.short), zeros)
        soa_modes.append(soa * to_molar_soag)
        opoa_modes.append(params.oxygenated_poa_fraction * poa * to_molar_soag)
    q_soa = jnp.stack(soa_modes, axis=0)
    q_opoa = jnp.stack(opoa_modes, axis=0)
    q_soag_new, q_soa_new, remaining = solve_soag(
        q_soag, q_soa, q_opoa, params.soag_transfer_ratio * coefficients,
        soag_equilibrium(temperature, pressure, params), dt, params)
    gases[gas_name("soag")] = q_soag_new / to_molar_soag
    soa_delta = (q_soa_new - q_soa) / to_molar_soag
    # Fraction of the step the bounded scan did not integrate; nonzero
    # means ``max_substeps`` bound and the last sub-step was oversized.
    diagnostics["condensation_soag_substep_shortfall"] = remaining / dt

    # --- apply per mode; route pcm's condensate to accumulation ---
    accumulation = spec.mode("accum").short
    for index, mode in enumerate(spec.modes):
        target = accumulation if mode.short == "pcm" else mode.short
        for species, delta in (("so4", so4_per_mode[index]),
                               ("soa", soa_delta[index])):
            key = mass_name(species, target)
            if key in masses:
                masses[key] = masses[key] + delta
            if mode.short == "pcm":
                coating[species] = coating.get(species, zeros) + delta

    return population.__class__(
        masses=masses, numbers=dict(population.numbers), gases=gases,
        coating=coating, diagnostics=diagnostics)
