r"""Brownian coagulation among the Aitken, accumulation and primary-carbon modes.

MAM4's coagulation, taken from E3SM's ``modal_aero_coag.F90`` (R. Easter,
2007, adapted from MIRAGE2 and CMAQ v4.6; read 2026-10-01 — the MAM
documentation site's coagulation pages are empty stubs, so the Fortran is
the reference here, and it is BSD-3). Three pieces:

1. **Rate coefficients** from CMAQ's "fast" ``getcoags`` (F. S. Binkowski,
   2003): the 0th- and 3rd-moment coagulation integrals over a pair of
   log-normal modes are evaluated analytically in the free-molecular and
   near-continuum limits (Whitby et al. 1991, eqs. H.7, H.10–H.12), each
   corrected by a tabulated factor fitted to the Gauss–Hermite quadratures
   of Binkowski and Roselle (2003), and the two limits are combined by a
   harmonic mean. The tables live in :mod:`.coagulation_tables`. The
   near-continuum form uses a slip correction linear in Knudsen number
   with coefficient ``A = 1.246``; the mean free path and viscosity are the
   1962 U.S. Standard Atmosphere expressions. Everything is evaluated on
   the **wet** geometric-mean diameters and wet densities of the
   start-of-step κ-Köhler state, as MAM does.

2. **Number changes**, semi-analytic over the step (``pair_option_acoag =
   3``): accumulation loses number only to itself,
   :math:`N' = N/(1 + \Delta t\,\beta_{jj}N)`; primary carbon loses to
   itself and to accumulation; Aitken loses to itself, to accumulation and
   to primary carbon. The donor's equation :math:`dN/dt = -(a N + b N^2)`
   is solved exactly, with the receiver's number taken as its step mean,
   and MAM's two small-coefficient approximations where the exact form is
   ill-conditioned. Inter-modal coagulation conserves the receiver's
   number — no mode ever gains number.

3. **Mass transfer** as a first-order volume loss from the donor at the
   receiver's step-mean number: the Aitken mode's species all go to
   accumulation, whether the collision partner was accumulation or primary
   carbon (MAM's "effective" receiver — ``pcm`` carries no sulfate or SOA
   slot), and the primary-carbon species go to accumulation by direct
   coagulation. The share of the Aitken sulfate and SOA whose partner was
   primary carbon is a soluble shell on that carbon and is recorded in
   ``population.coating`` for the ageing process (D-A1); MAM computes the
   aged fraction here and again inside condensation, and this scaffold
   keeps that in one place.

Coagulation with the coarse mode is neglected, as in MAM, as much slower
than these three. There are no tunables: MAM's coagulation has none.

**A transcription note on the 0th-moment free-molecular form.** CMAQ's
``getcoags`` has the last term of Whitby's H.7a as ``2 r⁻¹ es_i(4) +
es_j(1)`` where the moment expansion gives ``2 r⁻¹ es_i(4) es_j(1)`` (two
terms where one belongs; see the derivation in the design doc). The
expression here is MAM's as carried, because the correction tables were
fitted with it and the A/B of AER-12 compares against it; the design doc
records the size of the difference at MAM4 geometry.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp

import jcm.constants as c

from jcm.physics.aerosol.jam.microphysics import numerics
from jcm.physics.aerosol.jam.microphysics.coagulation_tables import BM0, BM0IJ, BM3I
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

#: Slip-correction coefficient of the near-continuum kernel (``getcoags``
#: ``a``; CMAQ once used 2.492, MAM carries 1.246).
SLIP_CORRECTION = 1.246
#: Sea-level mean free path [m] at ``_P_STD``/``_T_STD`` (U.S. Standard
#: Atmosphere 1962, table I.2.8), scaled as ``T/p``.
_MEAN_FREE_PATH_STD = 6.6328e-8
_P_STD = 101325.0
_T_STD_OFFSET = 15.0            # t0 = tmelt + 15 K
#: Sutherland viscosity of air, U.S. Standard Atmosphere 1962 p. 14.
_VISCOSITY_BETA = 1.458e-6
_VISCOSITY_S = 110.4
#: MAM's ``xferfrac_max`` is ``1 - 10 ε`` in float64; in float32 that literal
#: rounds to exactly 1.0, so the cap is taken in the working dtype.
_TRANSFER_ULPS = 10.0


def _transfer_max(like):
    """Largest fraction a step may move: ten ulps short of everything."""
    return 1.0 - _TRANSFER_ULPS * jnp.finfo(jnp.asarray(like).dtype).eps
#: Switch points of the donor number solution (``modal_aero_coag_sub``).
_SMALL_TOTAL = 0.01
_SMALL_INTER = 0.001
#: ``max(dumloss, 1e-37)`` in MAM; here a double-``where`` guard instead.
_TINY_LOSS = 1.0e-37

#: Species whose coagulation onto primary carbon counts as a soluble
#: shell (MAM: so4, nh4, soa; JAM carries no nh4).
_SHELL_SPECIES = ("so4", "soa")


def _sigma_class(geom_std_dev: float) -> int:
    """Whitby's σ_g class, 0-based: ``nint(4 (σ − 0.75))`` clipped to 1..10."""
    return max(1, min(10, int(math.floor(4.0 * (geom_std_dev - 0.75) + 0.5)))) - 1


def _harmonic_mean(a, b):
    """Combine two limits as ``a b / (a + b)``, zero where both vanish.

    Through :func:`numerics.harmonic_mean`: the plain quotient's reverse
    pass forms ``cot · (a+b)⁻²``, which overflows float32 under the
    cotangent the number equation sends back (AER-11).
    """
    return numerics.harmonic_mean(a, b)


def _moment_powers(ln_sigma: float) -> dict[int, float]:
    """``es(k) = exp(k ln²σ / 8)`` for the exponents ``getcoags`` uses."""
    base = 0.125 * ln_sigma * ln_sigma
    return {k: math.exp(k * base)
            for k in (1, 4, 5, 8, 9, 16, 20, 25, 36, 49, 64, 100)}


def air_properties(temperature, pressure):
    """Mean free path [m] and dynamic viscosity [kg m⁻¹ s⁻¹] of air.

    The U.S. Standard Atmosphere 1962 expressions ``getcoags_wrapper_f``
    uses: the sea-level mean free path scaled by ``T/p``, and Sutherland's
    law for the viscosity.
    """
    t_std = c.tmelt + _T_STD_OFFSET
    mean_free_path = _MEAN_FREE_PATH_STD * _P_STD * temperature / (t_std * pressure)
    viscosity = (_VISCOSITY_BETA * temperature * jnp.sqrt(temperature)
                 / (temperature + _VISCOSITY_S))
    return mean_free_path, viscosity


def intramodal_coefficient(diameter, density, geom_std_dev, temperature,
                           mean_free_path, viscosity):
    """0th-moment self-coagulation coefficient ``β_ii`` [m³ s⁻¹].

    ``dN/dt = −β_ii N²`` for a single log-normal mode (so ``β_ii`` is half
    the monodisperse kernel). Whitby et al. (1991) eqs. H.11a/H.12a with
    the ``bm0`` correction, combined by the harmonic mean.
    """
    ln_sigma = math.log(geom_std_dev)
    es = _moment_powers(ln_sigma)
    knc = (2.0 / 3.0) * c.ak * temperature / viscosity
    kfm = jnp.sqrt(3.0 * c.ak * temperature / density)
    knudsen = 2.0 * mean_free_path / diameter

    near_continuum = knc * (1.0 + es[8] + SLIP_CORRECTION * knudsen * (es[20] + es[4]))
    free_molecular = (kfm * jnp.sqrt(diameter) * BM0[_sigma_class(geom_std_dev)]
                      * (es[1] + es[25] + 2.0 * es[5]))
    return jnp.maximum(_harmonic_mean(near_continuum, free_molecular), 0.0)


def intermodal_coefficients(diameter_i, density_i, sigma_i,
                            diameter_j, density_j, sigma_j,
                            temperature, mean_free_path, viscosity):
    """Donor-to-receiver coefficients ``(β_ij0, β_ij3)``.

    ``β_ij0`` [m³ s⁻¹]: the donor's number loss, ``dN_i/dt = −β_ij0 N_i N_j``.
    ``β_ij3`` [m³ s⁻¹]: the donor's *volume* loss as a first-order rate at
    the receiver's number, ``dV_i/dt = −β_ij3 N_j V_i`` — CMAQ's 3rd-moment
    transfer coefficient divided by the donor's ``M₃/N`` (MAM's
    ``betaij3 = c3ij / (dg³ exp(4.5 ln²σ))``).

    Whitby et al. (1991) eqs. H.7a/H.10a (0th) and H.7b/H.10b (3rd) with
    the ``bm0ij``/``bm3i`` corrections, each limit combined by the
    harmonic mean. The diameter-ratio class of the tables is a function of
    the state and so a (piecewise-constant) traced index; the σ_g classes
    are static.
    """
    ln_sigma_i, ln_sigma_j = math.log(sigma_i), math.log(sigma_j)
    ei, ej = _moment_powers(ln_sigma_i), _moment_powers(ln_sigma_j)
    class_i, class_j = _sigma_class(sigma_i), _sigma_class(sigma_j)
    a = SLIP_CORRECTION

    knc = (2.0 / 3.0) * c.ak * temperature / viscosity
    kfm = jnp.sqrt(6.0 * c.ak * temperature / (density_i + density_j))
    kn_i = 2.0 * mean_free_path / diameter_i
    kn_j = 2.0 * mean_free_path / diameter_j
    sqrt_d = jnp.sqrt(diameter_i)
    r = jnp.sqrt(diameter_j / diameter_i)
    r2 = r * r
    r4 = r2 * r2
    ri1 = 1.0 / r
    ri2 = 1.0 / r2
    ri3 = ri2 * ri1

    # Diameter-ratio class: 1 + nint(log(d_j/d_i) / ln √2), clipped to 1..10
    # (Whitby et al. 1991, p. H.5). Fortran's nint rounds half away from
    # zero; floor(x + 0.5) agrees for every x that is not exactly −k−½.
    ratio_class = jnp.clip(
        1.0 + jnp.floor(jnp.log(diameter_j / diameter_i) / math.log(math.sqrt(2.0)) + 0.5),
        1.0, 10.0).astype(jnp.int32) - 1
    bm0ij = jnp.take(jnp.asarray(BM0IJ[:, class_i, class_j], dtype=r.dtype), ratio_class)
    bm3i = jnp.take(jnp.asarray(BM3I[:, class_i, class_j], dtype=r.dtype), ratio_class)

    # --- 0th moment: donor number loss ---
    near_continuum_0 = knc * (
        2.0 + a * (kn_i * (ei[4] + r2 * ei[16] * ej[4])
                   + kn_j * (ej[4] + ri2 * ej[16] * ei[4]))
        + (r2 + ri2) * ei[4] * ej[4])
    # Last two terms carried as MAM has them; see the module docstring.
    free_molecular_0 = kfm * sqrt_d * bm0ij * (
        ei[1] + r * ej[1] + 2.0 * r2 * ei[1] * ej[4]
        + r4 * ei[9] * ej[16] + ri3 * ei[16] * ej[9]
        + 2.0 * ri1 * ei[4] + ej[1])
    beta_0 = jnp.maximum(_harmonic_mean(near_continuum_0, free_molecular_0), 0.0)

    # --- 3rd moment: donor volume transfer ---
    # Both limits carry the donor's d³ (Whitby's H.7b/H.10b are 3rd-moment
    # rates), and MAM divides it back out to get a first-order rate. Factor
    # it out BEFORE the harmonic mean: with d³ ~ 1e-23 the two limits are
    # ~1e-37 and their product underflows float32 to zero (measured
    # 2026-10-01 as β_ij3 = 0 at MAM4 geometry).
    near_continuum_3 = knc * (
        2.0 * ei[36]
        + a * kn_i * (ei[16] + r2 * ei[4] * ej[4])
        + a * kn_j * (ei[36] * ej[4] + ri2 * ei[64] * ej[16])
        + r2 * ei[16] * ej[4] + ri2 * ei[64] * ej[4])
    free_molecular_3 = kfm * sqrt_d * bm3i * (
        ei[49] + r * ei[36] * ej[1] + 2.0 * r2 * ei[25] * ej[4]
        + r4 * ei[9] * ej[16] + ri3 * ei[100] * ej[9]
        + 2.0 * ri1 * ei[64] * ej[1])
    transfer_3_per_d3 = _harmonic_mean(near_continuum_3, free_molecular_3)
    beta_3 = jnp.maximum(transfer_3_per_d3 / math.exp(4.5 * ln_sigma_i * ln_sigma_i), 0.0)
    return beta_0, beta_3


def receiver_number(number, self_coefficient, dt):
    """``N' = N / (1 + Δt β N)``: self-coagulation only, exact."""
    return number / (1.0 + dt * self_coefficient * number)


def donor_number(number, inter_rate, self_coefficient, dt):
    """Advance a donor mode's number through ``dN/dt = −(inter_rate N + β N²)``.

    MAM's three forms: with ``a = Δt·inter_rate`` and ``b = Δt·β``,
    ``N exp(−(a + bN))`` when the total loss is small, ``exp(−a) N/(1 + bN)``
    when the inter-modal part is small, else the exact solution
    ``N g (1 − f)/(1 − g f)`` with ``f = bN/(a + bN)``, ``g = exp(−a)``,
    clipped to ``[0, N]``. Every divisor is guarded in both branches so the
    dead branches stay finite under reverse-mode differentiation.
    """
    a = dt * inter_rate
    b = dt * self_coefficient
    total = a + b * number
    small_total = jnp.abs(total) < _SMALL_TOTAL
    small_inter = jnp.abs(a) < _SMALL_INTER

    safe_total = jnp.where(small_total, 1.0, total)
    fraction = jnp.where(small_total, 0.0, b * number / safe_total)
    g = jnp.exp(-a)
    denominator = 1.0 - g * fraction
    general_live = ~(small_total | small_inter)
    safe_denominator = jnp.where(general_live, denominator, 1.0)
    general = jnp.where(general_live, g * (1.0 - fraction) / safe_denominator, 1.0)
    general = jnp.clip(general, 0.0, 1.0)

    return jnp.where(
        small_total, number * jnp.exp(-total),
        jnp.where(small_inter, g * number / (1.0 + b * number), number * general))


def coagulate(population, env, spec, params=None):
    """Run coagulation: population in, population out.

    ``params`` is unused — MAM's coagulation has no tunables — and is
    accepted only so the process signature is uniform.
    """
    masses = dict(population.masses)
    numbers = dict(population.numbers)
    coating = dict(population.coating)
    diagnostics = dict(population.diagnostics)

    aitken = spec.mode(spec.aitken_mode)
    accum = spec.mode(spec.accumulation_mode)
    pcarbon = spec.mode(spec.primary_carbon_mode)
    temperature, dt = env.temperature, env.dt
    rho_air = env.air_density
    mean_free_path, viscosity = air_properties(temperature, env.pressure)

    index = {mode.short: i for i, mode in enumerate(spec.modes)}

    def geometry(mode):
        i = index[mode.short]
        return (2.0 * env.state_start.r_wet[i], env.state_start.rho[i],
                mode.geom_std_dev)

    def inter(donor, receiver):
        return intermodal_coefficients(*geometry(donor), *geometry(receiver),
                                       temperature, mean_free_path, viscosity)

    def intra(mode):
        return intramodal_coefficient(*geometry(mode), temperature,
                                      mean_free_path, viscosity)

    # Number concentrations [m⁻³], negatives read as empty (MAM).
    def concentration(mode):
        return jnp.maximum(numbers[number_name(mode.short)], 0.0) * rho_air

    beta_ait_acc_0, beta_ait_acc_3 = inter(aitken, accum)
    beta_pcm_acc_0, beta_pcm_acc_3 = inter(pcarbon, accum)
    beta_ait_pcm_0, beta_ait_pcm_3 = inter(aitken, pcarbon)
    beta_ait_ait = intra(aitken)
    beta_acc_acc = intra(accum)
    beta_pcm_pcm = intra(pcarbon)

    # --- number, in MAM's order: receiver first, each donor at the
    # receivers' step-mean numbers ---
    n_acc = concentration(accum)
    n_acc_new = receiver_number(n_acc, beta_acc_acc, dt)
    n_acc_avg = 0.5 * (n_acc + n_acc_new)

    n_pcm = concentration(pcarbon)
    n_pcm_new = donor_number(n_pcm, beta_pcm_acc_0 * n_acc_avg, beta_pcm_pcm, dt)
    n_pcm_avg = 0.5 * (n_pcm + n_pcm_new)

    n_ait = concentration(aitken)
    n_ait_new = donor_number(
        n_ait, beta_ait_acc_0 * n_acc_avg + beta_ait_pcm_0 * n_pcm_avg,
        beta_ait_ait, dt)

    for mode, n_old, n_new in ((accum, n_acc, n_acc_new),
                               (pcarbon, n_pcm, n_pcm_new),
                               (aitken, n_ait, n_ait_new)):
        key = number_name(mode.short)
        numbers[key] = numbers[key] + (n_new - n_old) / rho_air

    # --- Aitken mass: one first-order loss at both partners' numbers, all
    # of it to accumulation; the primary-carbon share is a soluble shell ---
    loss_to_acc = beta_ait_acc_3 * n_acc_avg
    loss_to_pcm = beta_ait_pcm_3 * n_pcm_avg
    loss = loss_to_acc + loss_to_pcm
    live = loss > _TINY_LOSS
    # Through the safe quotient: ``loss`` can sit at its 1e-37 floor, whose
    # square is not a float32 number (AER-11).
    pcm_share = jnp.where(live, numerics.divide(loss_to_pcm, jnp.where(live, loss, 1.0)), 0.0)
    transfer_max = _transfer_max(temperature)
    aitken_fraction = jnp.clip(1.0 - jnp.exp(-loss * dt), 0.0, transfer_max)
    zeros = jnp.zeros_like(temperature)
    for species in aitken.species:
        source = mass_name(species, aitken.short)
        moved = masses[source] * aitken_fraction
        masses[source] = masses[source] - moved
        target = mass_name(species, accum.short)
        if target in masses:        # MAM: a species with no receiver slot is lost
            masses[target] = masses[target] + moved
        if species in _SHELL_SPECIES:
            coating[species] = coating.get(species, zeros) + moved * pcm_share

    # --- primary-carbon mass by direct coagulation with accumulation; the
    # ageing transfer that MAM adds here is AER-06's ---
    pcm_fraction = jnp.clip(1.0 - jnp.exp(-beta_pcm_acc_3 * n_acc_avg * dt),
                            0.0, transfer_max)
    for species in pcarbon.species:
        source = mass_name(species, pcarbon.short)
        moved = masses[source] * pcm_fraction
        masses[source] = masses[source] - moved
        target = mass_name(species, accum.short)
        if target in masses:
            masses[target] = masses[target] + moved

    diagnostics["coagulation_aitken_mass_fraction"] = aitken_fraction
    diagnostics["coagulation_aitken_share_to_pcm"] = pcm_share
    diagnostics["coagulation_pcm_mass_fraction"] = pcm_fraction
    return dataclasses.replace(
        population, masses=masses, numbers=numbers, coating=coating, diagnostics=diagnostics)
