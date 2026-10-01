r"""Mode renaming: the Aitken upper tail becomes accumulation after growth.

MAM4's renaming, from E3SM's ``modal_aero_rename.F90``
(``modal_aero_rename_sub``, R. Easter 2007 after MIRAGE2; read 2026-10-01
— the MAM documentation site's ``renaming/`` page is a stub), the scheme
of Easter et al. (2004). After a continuous-growth process has moved
Aitken particles up in size, the part of the distribution above a cut
diameter between the two modes is accumulation-mode by definition and is
transferred, number and mass, so condensation cannot inflate the Aitken
mode without bound. MAM has one pair, Aitken → accumulation, and this
module implements that pair.

**The cut.** :math:`D_{cut} = \sqrt{D_{g,ait}\,e^{1.5\ln^2\sigma_{ait}}
\cdot D_{g,acc}\,e^{1.5\ln^2\sigma_{acc}}}`, the geometric mean of the two
modes' volume-mean diameters at their reference sizes (≈ 55 nm for MAM4).
The fraction of a log-normal above it is :math:`\tfrac12\operatorname{erfc}
\big((\ln D_{cut} - \ln D_g)/(\sqrt2\ln\sigma)\big)` for number, and the
same with :math:`\ln D_g + 3\ln^2\sigma` for volume.

**The transfer.** With the mode's dry volume before the growth
:math:`V_{old}` and after it :math:`V_{new}`, at the (unchanged) number
:math:`N`, the diameters :math:`D_{old}` and :math:`D_{new}` follow from
the volume-number relation, and the transferred volume is the growth of
the tail, :math:`F_v(D_{new})V_{new} - F_v(D_{old})V_{old}`, as a fraction
of :math:`V_{new}`; the number fraction is :math:`F_n(D_{new}) -
F_n(D_{old})`, capped at the volume fraction. MAM's guards, all kept:
nothing happens if the mode is empty, if it did not grow by more than
1e-6 of itself, or if :math:`D_{new}` is still below the mode's reference
diameter; the number used for the diameters is relaxed into
:math:`[V/(f D_{hi}^3)/27,\ 27\,V/(f D_{lo}^3)]` so a number badly out of
step with the volume cannot produce an absurd diameter; and when
:math:`D_{new}` is already past the cut, :math:`D_{old}` is taken no
larger than :math:`0.99 D_{cut}` to guarantee some transfer.

**Both phases.** MAM computes the fractions once from the combined
interstitial + cloud-borne volume and number of the Aitken mode and
applies them to both phases; so does this (D-A5). The cloud-borne part
does not grow during gas-aerosol exchange, so it enters the old and the
new volume alike. MAM also renames after stratiform aqueous chemistry;
jcm's ``AqueousSulfur`` puts its product into the cloud-borne
accumulation and coarse modes only, never Aitken, so that call has
nothing to act on here and is not reproduced.

The growth this process sees is whatever ran before it in the step —
gas-aerosol exchange, in D-A1's order — measured as the change in the
mode's dry volume from ``env.dry_volume_start``. No tunables: ``frelax``
and the 1e-6 growth threshold are numerical guards, not physics.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
from jax.scipy.special import erfc

from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.microphysics.condensation import M_AIR
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

#: MAM's ``frelax``: how far the number may stray from the volume-implied
#: bounds before it is clipped for the diameter calculation.
RELAXATION = 27.0
#: MAM's ``dryvol_smallest``, 1e-25 m³ per kmol of air (a 1 nm particle at
#: 1e-5 cm⁻³), here per kg of air.
_SMALLEST_VOLUME = 1.0e-25 / (1000.0 * M_AIR)
#: Growth below this fraction of the old volume is not growth.
_MIN_RELATIVE_GROWTH = 1.0e-6
#: ``dp_belowcut = 0.99 dp_cut``.
_BELOW_CUT = 0.99
_TRANSFER_ULPS = 10.0


def _transfer_max(like):
    """Largest fraction a step may move: ten ulps short of everything."""
    return 1.0 - _TRANSFER_ULPS * jnp.finfo(jnp.asarray(like).dtype).eps


def cut_diameter(donor, receiver) -> float:
    """``D_cut``: geometric mean of the two modes' reference volume-mean diameters."""
    def volume_mean(mode):
        return mode.dgnum * math.exp(1.5 * math.log(mode.geom_std_dev) ** 2)
    return math.sqrt(volume_mean(donor) * volume_mean(receiver))


def tail_fraction(ln_cut, ln_diameter, ln_sigma):
    """Fraction of a log-normal's number above ``exp(ln_cut)``.

    Pass ``ln_diameter + 3 ln²σ`` for the volume fraction.
    """
    return 0.5 * erfc((ln_cut - ln_diameter) * math.sqrt(0.5) / ln_sigma)


def transfer_fractions(volume_old, volume_new, number, donor, receiver):
    """``(volume_fraction, number_fraction)`` of the donor to move.

    Pure function of the donor's dry volume before and after growth
    [m³/kg], its number [kg⁻¹] and the two modes' static geometry; MAM's
    ``modal_aero_rename_sub`` per cell. Every division and log is guarded
    in both branches, so an empty or unchanged mode has a finite
    reverse-mode gradient.
    """
    ln_sigma = math.log(donor.geom_std_dev)
    three_ln2 = 3.0 * ln_sigma * ln_sigma
    f_volume = moments.volume_factor(donor)
    ln_cut = math.log(cut_diameter(donor, receiver))
    ln_below_cut = math.log(_BELOW_CUT * cut_diameter(donor, receiver))
    number_per_volume_hi = RELAXATION / (f_volume * donor.dgnum_lo ** 3)
    number_per_volume_lo = 1.0 / (f_volume * donor.dgnum_hi ** 3 * RELAXATION)

    old_bounded = jnp.maximum(volume_old, _SMALLEST_VOLUME)
    growth = volume_new - volume_old
    grew = (volume_new > _SMALLEST_VOLUME) & (growth > _MIN_RELATIVE_GROWTH * old_bounded)

    number_bounded = jnp.clip(jnp.maximum(number, 0.0),
                              old_bounded * number_per_volume_lo,
                              old_bounded * number_per_volume_hi)
    safe_new = jnp.where(grew, volume_new, 1.0)
    diameter_new = jnp.cbrt(safe_new / (number_bounded * f_volume))
    active = grew & (diameter_new > donor.dgnum)
    ln_new = jnp.log(jnp.where(active, diameter_new, 1.0))
    tail_number_new = tail_fraction(ln_cut, ln_new, ln_sigma)
    tail_volume_new = tail_fraction(ln_cut, ln_new + three_ln2, ln_sigma)

    diameter_old = jnp.cbrt(old_bounded / (number_bounded * f_volume))
    ln_old = jnp.log(diameter_old)
    # Past the cut already: take the old diameter below it, so some
    # transfer happens (MAM's dp_belowcut rule).
    ln_old = jnp.where(diameter_new >= math.exp(ln_cut),
                       jnp.minimum(ln_old, ln_below_cut), ln_old)
    tail_number_old = tail_fraction(ln_cut, ln_old, ln_sigma)
    tail_volume_old = tail_fraction(ln_cut, ln_old + three_ln2, ln_sigma)

    moved_volume = tail_volume_new * volume_new - tail_volume_old * volume_old
    active = active & (moved_volume > 0.0)
    volume_fraction = jnp.where(
        active, jnp.minimum(moved_volume, safe_new) / safe_new, 0.0)
    volume_fraction = jnp.minimum(volume_fraction, _transfer_max(volume_fraction))
    number_fraction = jnp.clip(tail_number_new - tail_number_old, 0.0, volume_fraction)
    number_fraction = jnp.where(active, number_fraction, 0.0)
    return volume_fraction, number_fraction


def rename_aitken(population, env, spec, params=None):
    """Run renaming: population in, population out.

    ``params`` is unused — renaming has no tunables — and is accepted only
    so the process signature is uniform.
    """
    if env.dry_volume_start is None:
        raise ValueError(
            "renaming needs env.dry_volume_start, the per-mode dry volume "
            "before the continuous-growth processes ran")
    masses = dict(population.masses)
    numbers = dict(population.numbers)
    cloud_borne = dict(population.cloud_borne)
    diagnostics = dict(population.diagnostics)

    donor = spec.mode("aitken")
    receiver = spec.mode("accum")
    zeros = jnp.zeros_like(env.temperature)

    def cb_mass(species, mode):
        return cloud_borne.get(mass_name(species, mode.short, cloud_borne=True), zeros)

    def cb_number(mode):
        return cloud_borne.get(number_name(mode.short, cloud_borne=True), zeros)

    # Cloud-borne volume is unchanged by exchange: in both old and new.
    volume_cloud_borne = zeros
    for species in donor.species:
        volume_cloud_borne = volume_cloud_borne + jnp.maximum(
            cb_mass(species, donor), 0.0) / spec.species_props(species).density
    volume_new = moments.mode_volume(masses, donor, spec) + volume_cloud_borne
    volume_old = env.dry_volume_start[donor.short] + volume_cloud_borne
    number = numbers[number_name(donor.short)] + jnp.maximum(cb_number(donor), 0.0)
    volume_fraction, number_fraction = transfer_fractions(
        volume_old, volume_new, number, donor, receiver)

    def move(store, source, target, fraction):
        if source not in store:
            return
        moved = jnp.maximum(store[source], 0.0) * fraction
        store[source] = store[source] - moved
        if target in store:        # MAM aborts at init if a slot is missing
            store[target] = store[target] + moved

    for species in donor.species:
        move(masses, mass_name(species, donor.short),
             mass_name(species, receiver.short), volume_fraction)
        move(cloud_borne, mass_name(species, donor.short, cloud_borne=True),
             mass_name(species, receiver.short, cloud_borne=True), volume_fraction)
    move(numbers, number_name(donor.short), number_name(receiver.short), number_fraction)
    move(cloud_borne, number_name(donor.short, cloud_borne=True),
         number_name(receiver.short, cloud_borne=True), number_fraction)

    diagnostics["rename_volume_fraction"] = volume_fraction
    diagnostics["rename_number_fraction"] = number_fraction
    return dataclasses.replace(
        population, masses=masses, numbers=numbers, cloud_borne=cloud_borne,
        diagnostics=diagnostics)
