r"""New-particle formation (NPF) from sulfuric acid vapour.

MAM4's parameterization, taken from its own documentation
(``mam_doc_site/aerosol_nucleation/``, pages ``notation_and_assumptions``,
``equations_binary``, ``equations_growth``, ``equations_PBL``,
``numerics_clipping``, ``numerics_coupling``; read 2026-10-01). Three
pieces, applied in this order:

1. **A formation rate of critical clusters** :math:`J^*` [cm⁻³ s⁻¹] from
   two pathways, the larger of which wins (``equations_PBL``):

   * binary H₂SO₄–H₂O homogeneous formation after Vehkamäki et al. (2002):
     polynomial fits for the H₂SO₄ mole fraction :math:`x^*` of the
     critical cluster, the rate, the cluster's total molecule count
     :math:`n_{tot}` and its radius :math:`r^*`, in :math:`T`,
     :math:`\ln RH` and :math:`\ln c_{H_2SO_4}` (molecules cm⁻³). Valid
     for :math:`T \in [230.15, 305.15]` K, :math:`RH \in [10^{-4}, 1]`
     and :math:`c \in [10^4, 10^{11}]`; :math:`T` and :math:`RH` are
     clipped to the range and the rate is zero below the lower
     concentration bound;
   * an empirical first-order rate :math:`J^* = 10^{-6}\,c_{H_2SO_4}`
     within the boundary layer — below the PBL top or 100 m, whichever
     is higher — after Sihto et al. (2006) with the coefficient of Wang
     and Penner (2009). Its clusters are 1 nm pure H₂SO₄. This is what
     addresses the continental Aitken-number deficit that binary
     formation alone leaves (D-A2).

2. **Survival of those clusters to the Aitken mode** (``equations_growth``),
   MAM's revision of Kerminen and Kulmala (2002): the apparent rate is
   :math:`J = J^* \exp(\eta/D_{fin} - \eta/D_{ini})` with
   :math:`\eta = \gamma\, CS'/GR`, where the condensation sink :math:`CS'`
   is derived from the gas-aerosol exchange's summed uptake coefficient
   (``numerics_coupling``) and the growth rate :math:`GR` from the
   step-mean H₂SO₄ of that same step. New particles enter the Aitken
   mode at its lower bound.

3. **Clipping** (``numerics_clipping``): if the sulfate the new particles
   would carry exceeds the available vapour, the rate is scaled down by
   that ratio, and dropped to zero if the scaled rate falls below
   :math:`10^{-12}` m⁻³ s⁻¹.

**On units in the growth expressions.** MAM's page labels :math:`GR` as
m s⁻¹, but with its own constant ``3.0e-9`` and inputs — Kerminen and
Kulmala's eq. 20 — :math:`GR` comes out in nm h⁻¹, :math:`\gamma` in
nm² m² h⁻¹ and so :math:`\eta` in nm, which is the only reading under which
:math:`\eta/D` with :math:`D` in nm is dimensionless. That reading is
implemented and checked by a sanity test (:math:`\eta` of order nm at
typical conditions). It is recorded in the design doc as an inference from
dimensional consistency, not a quotation.

All internal mixing ratios are molar, as in MAM; conversion happens at the
boundary of :func:`form_new_particles`.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
from flax import struct

import jcm.constants as c

from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics import numerics
from jcm.physics.aerosol.jam.microphysics.condensation import M_AIR
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

#: Avogadro's number [mol⁻¹], from the two constants jcm does carry.
AVOGADRO = c.r_universal / c.ak

#: Density assumed for both sulfuric acid and sulfate aerosol [kg/m³]
#: (``notation_and_assumptions``).
RHO_H2SO4 = 1770.0

#: Vehkamäki validity ranges (``equations_binary``).
_T_RANGE = (230.15, 305.15)
_RH_RANGE = (1.0e-4, 1.0)
_C_RANGE = (1.0e4, 1.0e11)
#: Below this molar mixing ratio no formation is computed at all.
_Q_FLOOR = 4.0e-16

#: ``numerics_clipping``: an apparent rate below this [m⁻³ s⁻¹] is zero.
_RATE_FLOOR_M3 = 1.0e-12
#: MAM's early return (``modal_aero_newnuc.F90``: ``rateloge <= -13.82``): a
#: critical-cluster rate below this [cm⁻³ s⁻¹] — about 0.1 cm⁻³ per day — is
#: no formation, BEFORE survival and clipping are evaluated. Found by the
#: AER-12 A/B: below Vehkamäki's validity the fit returns a cluster larger
#: than the Aitken bound, the survival exponent turns positive and
#: ``0 · exp(+large)`` is NaN (measured in float64 at 291 K, 9.9e3 cm⁻³).
_RATE_FLOOR_CM3 = 1.0e-6
#: Molar mass [kg/mol] at which MAM counts the new particles' sulfate
#: (``mwso4``, pure sulfate) — the host mass is booked at the aerosol
#: species' 0.115, so a particle of the booked diameter and density holds
#: 115/96 of its "pure" mass. Counting from the pure mass is what MAM does
#: (``qnuma_del = qso4a_del·mw_so4a/mass_part``); counting from the host
#: mass gave 1.19× MAM's number (AER-12).
_M_SO4_PURE = 0.09606
#: MAM's early return (``modal_aero_newnuc.F90``: ``rateloge <= -13.82``): a
#: critical-cluster rate below this [cm⁻³ s⁻¹] — about 0.1 cm⁻³ per day — is
#: no formation, BEFORE survival and clipping are evaluated. Found by the
#: AER-12 A/B: below Vehkamäki's validity the fit returns a cluster larger
#: than the Aitken bound, the survival exponent turns positive and
#: ``0 · exp(+large)`` is NaN (measured in float64 at 291 K, 9.9e3 cm⁻³).
_RATE_FLOOR_CM3 = 1.0e-6
#: Molar mass [kg/mol] at which MAM counts the new particles' sulfate
#: (``mwso4``, pure sulfate) — the host mass is booked at the aerosol
#: species' 0.115, so a particle of the booked diameter and density holds
#: 115/96 of its "pure" mass. Counting from the pure mass is what MAM does
#: (``qnuma_del = qso4a_del·mw_so4a/mass_part``); counting from the host
#: mass gave 1.19× MAM's number (AER-12).
_M_SO4_PURE = 0.09606
#: Sulfate [kg/kg per step] below which formation books nothing: 1e-30 is
#: a hundred-thousandth of one H₂SO₄ molecule per kg of air.
_MASS_FLOOR = 1.0e-30

#: Aitken mode bounds used for the survival target (``equations_growth``);
#: the documented ``D_p,lo`` is a log-blend of MAM4's 8.7 and 26 nm, ≈12.5 nm.
_AITKEN_LO = math.exp(0.67 * math.log(8.7e-9) + 0.33 * math.log(26.0e-9))

#: Hygroscopicity of ammonium bisulfate used for the nuclei's water uptake.
_KAPPA_NH4HSO4 = 0.56

#: Growth rate [nm/h] below which nuclei are taken not to survive at all.
_GROWTH_FLOOR = 1.0e-9


@struct.dataclass
class NewParticleFormationParameters:
    """Tunables of new-particle formation, all differentiable leaves.

    Attributes:
        pbl_rate_coefficient: first-order coefficient of the empirical
            boundary-layer pathway [s⁻¹], 1e-6 after Wang and Penner (2009).
        pbl_minimum_depth: the empirical pathway applies below the PBL top
            or this height [m], whichever is higher; MAM uses 100 m.
        pbl_enabled: 1.0 keeps the empirical pathway, 0.0 leaves binary
            formation alone. A leaf rather than a flag so the two can be
            compared by differentiating through the switch.

    """

    pbl_rate_coefficient: jnp.ndarray = 1.0e-6
    pbl_minimum_depth: jnp.ndarray = 100.0
    pbl_enabled: jnp.ndarray = 1.0


def _poly(t, p0, p1, p2, p3, px, x):
    """``p0 + p1 T + p2 T² + p3 T³ + px / x``, the Vehkamäki coefficient form."""
    return p0 + p1 * t + p2 * t ** 2 + p3 * t ** 3 + px / x


def binary_formation(temperature, relative_humidity, concentration):
    """Vehkamäki et al. (2002) binary H₂SO₄–H₂O formation.

    Args:
        temperature: [K]; clipped to the parameterization's range.
        relative_humidity: as a fraction; clipped likewise.
        concentration: H₂SO₄ [molecules cm⁻³].

    Returns:
        ``(rate, n_h2so4, radius)`` — the critical-cluster formation rate
        [cm⁻³ s⁻¹] (zero below the concentration bound), the number of
        H₂SO₄ molecules in a critical cluster, and its radius [m].

    """
    t = jnp.clip(temperature, *_T_RANGE)
    ln_rh = jnp.log(jnp.clip(relative_humidity, *_RH_RANGE))
    # Concentration is clipped INTO the range for the polynomial's sake,
    # and the rate is then zeroed below the lower bound, as MAM does.
    conc = jnp.clip(concentration, *_C_RANGE)
    ln_c = jnp.log(conc)

    x = (0.740997 - 0.00266379 * t - 0.00349998 * ln_c
         + 0.0000504022 * t * ln_c + 0.00201048 * ln_rh
         - 0.000183289 * t * ln_rh + 0.00157407 * ln_rh ** 2
         - 0.0000179059 * t * ln_rh ** 2 + 0.000184403 * ln_rh ** 3
         - 1.50345e-6 * t * ln_rh ** 3)

    ln_rate = (
        _poly(t, 0.14309, 2.21956, -0.0273911, 0.0000722811, 5.91822, x)
        + _poly(t, 0.117489, 0.462532, -0.0118059, 0.0000404196, 15.7963, x) * ln_rh
        + _poly(t, -0.215554, -0.0810269, 0.00143581, -4.7758e-6, -2.91297, x) * ln_rh ** 2
        + _poly(t, -3.58856, 0.049508, -0.00021382, 3.10801e-7, -0.0293333, x) * ln_rh ** 3
        + _poly(t, 1.14598, -0.600796, 0.00864245, -0.0000228947, -8.44985, x) * ln_c
        + _poly(t, 2.15855, 0.0808121, -0.000407382, -4.01957e-7, 0.721326, x) * ln_rh * ln_c
        + _poly(t, 1.6241, -0.0160106, 0.0000377124, 3.21794e-8, -0.0113255, x) * ln_rh ** 2 * ln_c
        + _poly(t, 9.71682, -0.115048, 0.000157098, 4.00914e-7, 0.71186, x) * ln_c ** 2
        + _poly(t, -1.05611, 0.00903378, -0.0000198417, 2.46048e-8, -0.0579087, x) * ln_rh * ln_c ** 2
        + _poly(t, -0.148712, 0.00283508, -9.24619e-6, 5.00427e-9, -0.0127081, x) * ln_c ** 3
    )

    ln_ntot = (
        _poly(t, -0.00295413, -0.0976834, 0.00102485, -2.18646e-6, -0.101717, x)
        + _poly(t, -0.00205064, -0.00758504, 0.000192654, -6.7043e-7, -0.255774, x) * ln_rh
        + _poly(t, 0.00322308, 0.000852637, -0.0000154757, 5.66661e-8, 0.0338444, x) * ln_rh ** 2
        + _poly(t, 0.0474323, -0.000625104, 2.65066e-6, -3.67471e-9, -0.000267251, x) * ln_rh ** 3
        + _poly(t, -0.0125211, 0.00580655, -0.000101674, 2.88195e-7, 0.0942243, x) * ln_c
        + _poly(t, -0.038546, -0.000672316, 2.60288e-6, 1.19416e-8, -0.00851515, x) * ln_rh * ln_c
        + _poly(t, -0.0183749, 0.000172072, -3.71766e-7, -5.14875e-10, 0.00026866, x) * ln_rh ** 2 * ln_c
        + _poly(t, -0.0619974, 0.000906958, -9.11728e-7, -5.36796e-9, -0.00774234, x) * ln_c ** 2
        + _poly(t, 0.0121827, -0.00010665, 2.5346e-7, -3.63519e-10, 0.000610065, x) * ln_rh * ln_c ** 2
        + _poly(t, 0.000320184, -0.0000174762, 6.06504e-8, -1.42177e-11, 0.000135751, x) * ln_c ** 3
    )

    n_total = jnp.exp(ln_ntot)
    n_h2so4 = n_total * x
    radius_nm = jnp.exp(-1.6524245 + 0.42316402 * x + 0.3346648 * ln_ntot)
    rate = jnp.where(concentration > _C_RANGE[0], jnp.exp(ln_rate), 0.0)
    return rate, n_h2so4, radius_nm * 1.0e-9


def boundary_layer_formation(concentration, params):
    """Empirical first-order formation ``J* = k c`` (``equations_PBL``).

    Returns ``(rate, n_h2so4, radius)`` with 1 nm pure-H₂SO₄ clusters:
    ``r* = 0.5 nm`` and ``n = N_A (π/6)(1 nm)³ ρ / M_w``.
    """
    rate = params.pbl_rate_coefficient * concentration
    m_h2so4 = GAS_SPECIES["h2so4"].molar_mass
    n_h2so4 = AVOGADRO * (math.pi / 6.0) * (1.0e-9) ** 3 * RHO_H2SO4 / m_h2so4
    return rate, jnp.full_like(rate, n_h2so4), jnp.full_like(rate, 0.5e-9)


def survival_to_aitken(rate, n_h2so4, radius, temperature, relative_humidity,
                       concentration, uptake_sum, air_moles):
    """Carry critical clusters to the Aitken lower bound (``equations_growth``).

    Returns ``(apparent_rate, particle_dry_diameter)``: the rate reaching
    the Aitken mode [cm⁻³ s⁻¹] and the dry diameter [m] the new particles
    are booked at — the cluster's own if it already exceeds the bound,
    else the bound.

    The condensation sink is MAM's, ``CS' = C_sum / (4π D_g α)`` from the
    summed uptake coefficient of the gas-aerosol exchange step rather than
    Kerminen and Kulmala's own form. Units: see the module docstring.
    """
    m_so4 = 0.115       # MAM books new-particle mass at the sulfate aerosol MW
    m_h2so4 = GAS_SPECIES["h2so4"].molar_mass

    # Dry size of the critical cluster, against the Aitken lower bound.
    dry_volume = n_h2so4 * m_so4 / (AVOGADRO * RHO_H2SO4)
    dry_diameter = jnp.cbrt(6.0 * dry_volume / math.pi)
    already_large = dry_diameter > _AITKEN_LO

    # Wet-to-dry volume ratio of ammonium-bisulfate nuclei, no curvature.
    rh = jnp.clip(relative_humidity, 0.1, 0.95)
    f_v = 1.0 - _KAPPA_NH4HSO4 / jnp.log(rh)
    rho_nuc = RHO_H2SO4 / f_v

    d_initial_nm = jnp.maximum(2.0 * radius * 1.0e9, 1.0)
    d_final_nm = 1.0e9 * _AITKEN_LO * jnp.cbrt(f_v)

    gamma = (0.23 * d_initial_nm ** 0.2 * (d_final_nm / 3.0) ** 0.075
             * (rho_nuc / 1000.0) ** (-0.33) * (temperature / 293.0) ** (-0.75))
    speed = 14.7 * jnp.sqrt(temperature)
    growth_rate = 3.0e-9 * speed * (m_h2so4 * 1.0e3) * concentration / rho_nuc
    diffusivity = 6.7037e-6 * temperature ** 0.75 / air_moles
    sink = uptake_sum / (4.0 * math.pi * diffusivity * 0.65)

    # No vapour means no growth and no survival. Double-where, not a
    # floored divisor: ``gamma sink / max(GR, tiny)`` has reverse-mode
    # cotangent ``-gamma sink / tiny^2``, and ``1e-60`` underflows in
    # float32 to ``x/0`` (measured 2026-10-01 as a NaN gradient through
    # the full core at an empty mode with no gas). 1e-9 nm/h is a growth
    # rate that would take a century to cross the Aitken bound.
    growing = growth_rate > _GROWTH_FLOOR
    safe_growth = jnp.where(growing, growth_rate, 1.0)
    eta = jnp.where(growing, gamma * sink / safe_growth, 0.0)
    # Survival is a fraction: the exponent is clamped at zero so a cluster
    # the fit puts above the Aitken bound (only outside Vehkamäki's
    # validity, where the rate is already zeroed) cannot make it overflow.
    survival = jnp.where(
        growing, jnp.exp(jnp.minimum(eta / d_final_nm - eta / d_initial_nm, 0.0)), 0.0)
    apparent = jnp.where(already_large, rate, rate * survival)
    diameter = jnp.where(already_large, dry_diameter, _AITKEN_LO)
    return apparent, diameter


def form_new_particles(population, env, spec, params):
    """Run new-particle formation: population in, population out.

    Reads the step-mean H₂SO₄ and the summed uptake coefficient that the
    gas-aerosol exchange published in ``population.diagnostics``
    (``numerics_coupling``: MAM evaluates formation on the exchange step's
    time-mean vapour and uses its uptake rate for the growth sink), and
    the lagged boundary-layer height from ``env``. Adds number and sulfate
    to the Aitken mode and removes the sulfate from the vapour.
    """
    masses = dict(population.masses)
    numbers = dict(population.numbers)
    gases = dict(population.gases)
    diagnostics = dict(population.diagnostics)

    temperature, pressure, dt = env.temperature, env.pressure, env.dt
    zeros = jnp.zeros_like(temperature)
    m_h2so4 = GAS_SPECIES["h2so4"].molar_mass
    m_so4 = spec.species_props("so4").molar_mass
    to_molar = M_AIR / m_h2so4

    # Vapour: the exchange step's time-mean where it ran, else the current
    # value; molecules per cm³ from the molar mixing ratio.
    q_now = gases.get(gas_name("h2so4"), zeros) * to_molar
    q_mean = diagnostics.get("condensation_h2so4_step_mean", q_now)
    uptake_sum = diagnostics.get("condensation_h2so4_uptake_sum", zeros)
    air_moles = pressure / (c.r_universal * temperature)          # mol m⁻³
    concentration = q_mean * air_moles * AVOGADRO * 1.0e-6         # cm⁻³

    # Two pathways; the larger rate wins, and its cluster properties go
    # with it. The empirical pathway applies only within the boundary layer.
    rate_b, n_b, r_b = binary_formation(temperature, env.saturation, concentration)
    rate_p, n_p, r_p = boundary_layer_formation(concentration, params)
    in_pbl = env.height_above_surface <= jnp.maximum(
        env.pbl_height[jnp.newaxis], params.pbl_minimum_depth)
    rate_p = rate_p * params.pbl_enabled * in_pbl
    use_pbl = rate_p > rate_b
    rate = jnp.where(use_pbl, rate_p, rate_b)
    n_h2so4 = jnp.where(use_pbl, n_p, n_b)
    radius = jnp.where(use_pbl, r_p, r_b)
    rate = jnp.where((q_mean > _Q_FLOOR) & (rate >= _RATE_FLOOR_CM3), rate, 0.0)

    apparent, diameter = survival_to_aitken(
        rate, n_h2so4, radius, temperature, env.saturation, concentration,
        uptake_sum, air_moles)

    # Tendencies (``notation_and_assumptions``): number per kg of air, and
    # the sulfate each new particle carries at its booked dry diameter.
    rate_per_m3 = apparent * 1.0e6
    number_tendency = rate_per_m3 / env.air_density                 # kg⁻¹ s⁻¹
    particle_mass = (math.pi / 6.0) * diameter ** 3 * RHO_H2SO4    # kg
    mass_tendency = particle_mass * number_tendency                 # kg/kg/s

    # Clipping: never book more sulfate than the vapour can supply.
    # ``available`` is floored at zero so a vapour that is negative by
    # rounding reads as empty.
    available = jnp.maximum(gases.get(gas_name("h2so4"), zeros), 0.0) * (m_so4 / m_h2so4)
    wanted = mass_tendency * dt
    # Formation wanting less than ``_MASS_FLOOR`` books nothing — number and
    # mass alike, so nothing is overdrawn — and the clipping ratio is only
    # evaluated where it is live. Both are for the reverse pass: with the
    # vapour fully consumed and ``wanted`` ~1e-32 kg/kg the ratio is live
    # and ``available / wanted`` has the rule ``cot · wanted⁻²``, which
    # overflows float32 to ``0 · inf = NaN`` even though the true
    # sensitivity is zero (measured 2026-10-01 on the AER-11 sweep, a mode
    # with number and no mass). The floor keeps the live divisor ≥ 1e-30
    # and :func:`numerics.divide` never forms its square.
    forming = wanted > _MASS_FLOOR
    clipping = forming & (wanted > available)
    safe_wanted = jnp.where(clipping, wanted, 1.0)
    factor = jnp.where(forming,
                       jnp.where(clipping, numerics.divide(available, safe_wanted), 1.0),
                       0.0)
    factor = jnp.where(rate_per_m3 * factor < _RATE_FLOOR_M3, 0.0, factor)
    so4_delta = wanted * factor
    # Number from the pure-sulfate mass of what was booked (MAM's
    # bookkeeping, see ``_M_SO4_PURE``), not from the host mass: the same
    # as ``so4_delta·(96/115)/particle_mass``, written without dividing by
    # a ~1e-21 kg particle mass, whose reverse rule would overflow float32
    # (the AER-11 class).
    number_delta = number_tendency * dt * factor * (_M_SO4_PURE / m_so4)

    aitken = spec.mode(spec.aitken_mode).short
    numbers[number_name(aitken)] = numbers[number_name(aitken)] + number_delta
    masses[mass_name("so4", aitken)] = masses[mass_name("so4", aitken)] + so4_delta
    gases[gas_name("h2so4")] = (gases.get(gas_name("h2so4"), zeros)
                                - so4_delta * (m_h2so4 / m_so4))

    diagnostics["npf_rate_apparent"] = apparent
    diagnostics["npf_boundary_layer_pathway"] = use_pbl.astype(temperature.dtype)
    return dataclasses.replace(
        population, masses=masses, numbers=numbers, gases=gases, diagnostics=diagnostics)
