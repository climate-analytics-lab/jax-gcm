"""Dust/BC ice-nucleating-particle populations from the aerosol spec (#494).

The freezing schemes need, for dust and black carbon, an IN **number** [m⁻³]
(to cap the nucleated crystals) and a **surface-area** concentration [m²/m³]
(the active-site schemes scale with area), split into a **soluble** pool
(immersion freezing, inside cloud droplets) and an **insoluble** pool
(deposition/contact freezing, bare particles) — the analog of HAMMOZ's
``mo_ham_freezing.f90::get_aerofreez_nc``.

This is written against the *generic* aerosol-population interface rather than
any particular microphysics: for each population class it reads only the
``species`` it carries, its ``soluble`` flag, and the family-agnostic
``number_factor`` / ``area_factor`` geometry (see
:class:`jcm.physics.aerosol.jam.population.AerosolMode`). A sectional spec
(#491) exposes the same per-class interface, so this code is invariant to the
modal-vs-sectional choice. Per-species dust/BC mass comes from the
``m_du_*``/``m_bc_*`` (interstitial) and ``mc_*`` (cloud-borne) tracers:

* number-equiv:  ``n = (m/ρ) · number_factor``
* surface area:  ``A = (m/ρ) · area_factor``        [both ×ρ_air for /m³]

Solubility comes from the spec, not hard-coded class names. **Cloud-borne**
dust/BC (``mc_*``) is already inside cloud droplets, so it is immersion-active
(soluble) regardless of class. **Interstitial** (``m_*``) aerosol is classed by
the population's own ``soluble`` flag (aged/coated classes → immersion;
hydrophobic classes such as primary carbon → deposition/contact). Interstitial
dust in a soluble class is further split by the differentiable
``frac_du_soluble`` (most aged dust is immersion-active, a small bare fraction
is available for deposition).
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name


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

    def number_area(species, mode, *, cloud_borne):
        mass = tracers.get(
            mass_name(species, mode.short, cloud_borne=cloud_borne), zeros
        )
        density = spec.species_props(species).density
        vol = jnp.maximum(mass, 0.0) / density        # m³/kg-air
        return (vol * mode.number_factor * air_density,
                vol * mode.area_factor * air_density)

    def pools(species):
        """``(soluble, insoluble, cloud_borne)`` (number, area) pairs.

        Interstitial mass is sorted into the soluble / insoluble pool by each
        class's own ``soluble`` flag; cloud-borne mass is immersion-active
        (soluble) in any class.
        """
        sol = ins = cb = (zeros, zeros)
        for mode in spec.modes:
            if species not in mode.species:
                continue
            dn, da = number_area(species, mode, cloud_borne=True)
            cb = (cb[0] + dn, cb[1] + da)
            dn, da = number_area(species, mode, cloud_borne=False)
            if mode.soluble:
                sol = (sol[0] + dn, sol[1] + da)
            else:
                ins = (ins[0] + dn, ins[1] + da)
        return sol, ins, cb

    fsol = jnp.clip(frac_du_soluble, 0.0, 1.0)
    (du_sol, du_ins, du_cb) = pools("du")
    (bc_sol, bc_ins, bc_cb) = pools("bc")

    return {
        # Dust: soluble-class interstitial dust is split by frac_du_soluble
        # into an immersion-active and a bare (deposition) fraction; cloud-borne
        # dust is always immersion-active; any insoluble-class dust stays bare.
        "du_number_sol": du_sol[0] * fsol + du_cb[0],
        "du_area_sol": du_sol[1] * fsol + du_cb[1],
        "du_number_insol": du_sol[0] * (1.0 - fsol) + du_ins[0],
        "du_area_insol": du_sol[1] * (1.0 - fsol) + du_ins[1],
        # BC: soluble-class interstitial + cloud-borne is immersion-active;
        # hydrophobic-class (e.g. primary carbon) interstitial is bare.
        "bc_number_sol": bc_sol[0] + bc_cb[0],
        "bc_area_sol": bc_sol[1] + bc_cb[1],
        "bc_number_insol": bc_ins[0],
        "bc_area_insol": bc_ins[1],
    }
