"""Dust/BC ice-nucleating-particle populations from the modal aerosol (#494).

The freezing schemes need, for dust and black carbon, an IN **number** [m⁻³]
(to cap the nucleated crystals) and a **surface-area** concentration [m²/m³]
(the active-site schemes scale with area), split into a **soluble** pool
(immersion freezing, inside cloud droplets) and an **insoluble** pool
(deposition/contact freezing, bare particles) — the analog of HAMMOZ's
``mo_ham_freezing.f90::get_aerofreez_nc``.

``_jam_state`` only carries per-mode totals, so the per-species dust/BC mass is
read from the ``m_du_*``/``m_bc_*`` tracers and converted with each mode's
log-normal geometry:

* number-equiv:  ``n = (m/ρ) / v_p``,         ``v_p = (π/6)·dg³·exp(4.5 ln²σ)``
* surface area:  ``A = (m/ρ) · 6/(dg·exp(2.5 ln²σ))``    [m²/m³ after ×ρ_air]

Solubility (MAM4-MOM): BC in accum/coarse is treated as aged/coated → soluble,
BC in primary_carbon → insoluble; dust (only in accum/coarse) is split by the
differentiable ``frac_du_soluble`` (most aged dust is immersion-active, a small
bare fraction is available for deposition).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name

_SOLUBLE_MODES = ("acc", "cor")     # aged/coated → immersion
_INSOLUBLE_MODES = ("pcm",)         # primary carbon → deposition/contact


def _geometry_factors(spec: ModalAerosolSpec, mode_short: str):
    """``(number_factor [1/m³], area_factor [1/m])`` for a mode's log-normal."""
    mode = spec.mode(mode_short)
    ln_sigma = math.log(mode.geom_std_dev)
    v_p = (math.pi / 6.0) * mode.dgnum ** 3 * math.exp(4.5 * ln_sigma ** 2)
    area_factor = 6.0 / (mode.dgnum * math.exp(2.5 * ln_sigma ** 2))
    return 1.0 / v_p, area_factor


def in_populations(
    spec: ModalAerosolSpec,
    tracers: dict,
    air_density: jnp.ndarray,        # (nlev, ncols)
    frac_du_soluble: jnp.ndarray,    # scalar in [0, 1]
) -> dict[str, jnp.ndarray]:
    """Dust/BC IN number [m⁻³] and area [m²/m³], soluble + insoluble.

    Returns keys ``{du,bc}_{number,area}_{sol,insol}`` each ``(nlev, ncols)``.
    """
    zeros = jnp.zeros_like(air_density)

    def species_number_area(species, mode_short):
        mass = tracers.get(mass_name(species, mode_short), zeros)
        density = spec.species_props(species).density
        vol = jnp.maximum(mass, 0.0) / density        # m³/kg-air
        n_fac, a_fac = _geometry_factors(spec, mode_short)
        number = vol * n_fac * air_density            # m⁻³
        area = vol * a_fac * air_density              # m²/m³
        return number, area

    # Dust lives only in the soluble (accum/coarse) modes; split by solubility.
    du_n = zeros
    du_a = zeros
    for m in _SOLUBLE_MODES:
        if "du" in spec.mode(m).species:
            n, a = species_number_area("du", m)
            du_n = du_n + n
            du_a = du_a + a
    fsol = jnp.clip(frac_du_soluble, 0.0, 1.0)

    # BC solubility follows the mode (soluble modes vs primary carbon).
    bc_n_sol = zeros
    bc_a_sol = zeros
    for m in _SOLUBLE_MODES:
        if "bc" in spec.mode(m).species:
            n, a = species_number_area("bc", m)
            bc_n_sol = bc_n_sol + n
            bc_a_sol = bc_a_sol + a
    bc_n_insol = zeros
    bc_a_insol = zeros
    for m in _INSOLUBLE_MODES:
        if "bc" in spec.mode(m).species:
            n, a = species_number_area("bc", m)
            bc_n_insol = bc_n_insol + n
            bc_a_insol = bc_a_insol + a

    return {
        "du_number_sol": du_n * fsol,
        "du_area_sol": du_a * fsol,
        "du_number_insol": du_n * (1.0 - fsol),
        "du_area_insol": du_a * (1.0 - fsol),
        "bc_number_sol": bc_n_sol,
        "bc_area_sol": bc_a_sol,
        "bc_number_insol": bc_n_insol,
        "bc_area_insol": bc_a_insol,
    }
