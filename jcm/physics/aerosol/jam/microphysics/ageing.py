r"""Ageing of primary carbon into the accumulation mode.

MAM4's criterion, from E3SM's ``modal_aero_gasaerexch.F90`` (the
primary-carbon ageing block of ``modal_aero_gasaerexch_sub``, and the
duplicate of it in ``modal_aero_coag_sub``; read 2026-10-01 — the MAM
documentation site's ``aging/`` page is a stub). Freshly emitted carbon is
hydrophobic: the primary-carbon mode is ``can_activate=False`` and so is
excluded from droplet activation, in-cloud scavenging and the cloud-borne
phase. Once a soluble shell has grown on it the particles behave like
accumulation-mode ones, and this process moves them there.

**The criterion.** With :math:`V_{shell}` the volume of hygroscopic
material deposited on the mode this step and :math:`V_{core}` the mode's
dry volume, the aged fraction is the shell thickness over the thickness of
:math:`n` monolayers of sulfate:

.. math::

   f_{age} = \min\!\left(1,\;
       \frac{V_{shell}}{V_{core}} \cdot
       \frac{D_{g}\, e^{2.5 \ln^2\sigma_g}}{6\, n\, \delta}\right),
   \qquad \delta = 4.76\times10^{-10}\ \mathrm{m},

where :math:`6 e^{-2.5\ln^2\sigma_g}/D_g` is the mode's surface-to-volume
ratio, so :math:`V_{shell}/(\text{surface})` is the mean shell thickness.
:math:`\delta` is the size of one (bi)sulfate molecule, from 1 mol = 65 cm³
at 1.77 g cm⁻³; MAM does not distinguish sulfuric acid, bisulfate and
ammonium sulfate in the criterion. SOA counts toward the shell scaled by
:math:`\kappa_{SOA}/\kappa_{SO_4}` (MAM's ``soa_equivso4_factor``), the
volume of sulfate with the same hygroscopicity contribution.

**What moves.** That fraction of every primary-carbon species and of the
mode's number goes to accumulation. The soluble shell itself was already
booked to accumulation by the processes that deposited it (condensation,
coagulation), since the primary-carbon mode has no sulfate or SOA slot;
aerosol water is not transferred (MAM bypasses it). The fraction is capped
at :math:`1 - 10\epsilon` (``xferfrac_max``).

**Where the shell comes from, and the operator split (D-A1).** MAM
evaluates this criterion twice per step, inside gas-aerosol exchange with
the condensed shell and inside coagulation with the coagulated one, and
transfers twice. Here both processes record their deposit in
``population.coating`` and the criterion is evaluated once on the sum,
after both; first-order equivalent, and the single place to revisit if
the BC lifetime disagrees. The core volume and diameter are the
start-of-step ones, as elsewhere in the core.

**The monolayer count.** ``n_so4_monolayers_pcage = 8`` in both E3SM's and
CESM's MAM4 sources; Liu et al. (2012) describe MAM3/7 with 3. The default
here is MAM4's 8, as a differentiable leaf; the design doc records the
discrepancy with the plan's earlier reading of the supplement.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
from flax import struct

from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

#: Thickness of one sulfate monolayer [m] (``4.76e-10`` in MAM: the cube
#: root of one molecule's volume at 65 cm³ mol⁻¹).
MONOLAYER_THICKNESS = 4.76e-10
#: MAM's ``xferfrac_max`` is ``1 - 10 ε`` in float64; in float32 that literal
#: rounds to exactly 1.0, so the cap is taken in the working dtype.
_TRANSFER_ULPS = 10.0


def _transfer_max(like):
    """Largest fraction a step may move: ten ulps short of everything."""
    return 1.0 - _TRANSFER_ULPS * jnp.finfo(jnp.asarray(like).dtype).eps
#: Core volume [m³ per kg air] below which the mode is empty for the
#: criterion: 1e-19 kg/kg of carbon, and the largest value at which a
#: float32 reverse derivative through ``V_shell/V_core`` stays finite.
_CORE_FLOOR = 1.0e-22
#: Species whose deposit counts toward the shell, and the hygroscopicity
#: the shell is measured against.
_SHELL_SPECIES = ("so4", "soa")
_REFERENCE_SHELL = "so4"


@struct.dataclass
class AgeingParameters:
    """Tunables of primary-carbon ageing, all differentiable leaves.

    Attributes:
        n_so4_monolayers: sulfate monolayers that age a carbon particle
            completely. MAM4's sources use 8; Liu et al. (2012) describe 3
            for MAM3/7; ECHAM-HAM's ``m7_coat`` uses 1.

    """

    n_so4_monolayers: jnp.ndarray = 8.0


def shell_equivalence(spec) -> dict[str, float]:
    """Per-species factor converting deposit volume to sulfate-equivalent volume.

    ``κ_species / κ_so4`` for each shell species (1 for sulfate itself),
    MAM's ``soa_equivso4_factor`` generalised. Static, from the spec.
    """
    reference = spec.species_props(_REFERENCE_SHELL).hygroscopicity
    return {species: spec.species_props(species).hygroscopicity / reference
            for species in _SHELL_SPECIES}


def aged_fraction(shell_volume, core_volume, dry_diameter, geom_std_dev,
                  n_monolayers):
    """Fraction of the mode that ages: shell thickness over ``n`` monolayers.

    Both volumes are per kg of air; only their ratio matters. MAM forms
    ``tmp1 = V_shell D_g e^{2.5 ln²σ}`` over ``tmp2 = 6 n δ V_core`` to
    avoid overflow in float64; in float32 that quotient's reverse
    derivative, ``−tmp1/tmp2²`` with ``tmp2 ~ 1e-21``, underflows to a
    division by zero (measured 2026-10-01 as ``−inf`` d/dn). So the same
    number is built as ``(V_shell/V_core) · (D_g e^{2.5 ln²σ} / 6 n δ)``,
    two ratios of comparably sized quantities, and a core below
    ``_CORE_FLOOR`` reads as empty — the derivative of the first ratio is
    ``−V_shell/V_core²`` and needs a floor that keeps it representable.
    """
    core = jnp.maximum(core_volume, 0.0)
    live = core > _CORE_FLOOR
    ratio = jnp.where(live, shell_volume / jnp.where(live, core, 1.0), 0.0)
    scale = dry_diameter * math.exp(2.5 * math.log(geom_std_dev) ** 2) / (
        6.0 * n_monolayers * MONOLAYER_THICKNESS)
    # MAM: tmp1 >= tmp2 → xferfrac_max, else min(tmp1/tmp2, xferfrac_max).
    fraction = ratio * scale
    return jnp.clip(fraction, 0.0, _transfer_max(fraction))


def age(population, env, spec, params):
    """Run ageing: population in, population out.

    Consumes ``population.coating`` — zeroing it, so a second evaluation
    in the same step cannot age the same deposit twice — and moves the
    aged fraction of the primary-carbon species and number to accumulation.
    """
    masses = dict(population.masses)
    numbers = dict(population.numbers)
    diagnostics = dict(population.diagnostics)

    pcarbon = spec.mode("primary_carbon")
    accum = spec.mode("accum")
    zeros = jnp.zeros_like(env.temperature)

    equivalence = shell_equivalence(spec)
    shell_volume = zeros
    for species, factor in equivalence.items():
        deposit = population.coating.get(species, zeros)
        shell_volume = shell_volume + factor * deposit / spec.species_props(species).density
    core_volume = moments.mode_volume(masses, pcarbon, spec)
    index = {mode.short: i for i, mode in enumerate(spec.modes)}[pcarbon.short]
    dry_diameter = 2.0 * env.state_start.r_dry[index]

    fraction = aged_fraction(shell_volume, core_volume, dry_diameter,
                             pcarbon.geom_std_dev, params.n_so4_monolayers)

    for species in pcarbon.species:
        source = mass_name(species, pcarbon.short)
        moved = masses[source] * fraction
        masses[source] = masses[source] - moved
        target = mass_name(species, accum.short)
        if target in masses:        # MAM: a species with no receiver slot is lost
            masses[target] = masses[target] + moved
    source = number_name(pcarbon.short)
    moved = numbers[source] * fraction
    numbers[source] = numbers[source] - moved
    numbers[number_name(accum.short)] = numbers[number_name(accum.short)] + moved

    diagnostics["ageing_fraction"] = fraction
    diagnostics["ageing_shell_volume"] = shell_volume
    return dataclasses.replace(
        population, masses=masses, numbers=numbers, coating={}, diagnostics=diagnostics)
