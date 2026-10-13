"""Per-species aerosol mass-budget gauge (#713).

The July-2026 aerosol runaway hid for weeks because a burden drift is a
small residual of large opposing process fluxes — and the health gates
(NaN / temperature / humidity) sailed through a ×390 dust explosion with
the meteorology entirely normal. This gauge closes the budget *in the
step*, where the pairing between a physics call and the host's transport
is exact, instead of reconstructing it from time-averaged output:

``budget_mass_<sp>``
    Column mass of the ADVECTED interstitial tracers (``m_*``) of species
    ``sp`` at physics entry [kg/m²], per column. The cloud-borne carry is
    deliberately excluded — the host never transports it, so it cannot
    contribute to the dynamics residual, and its exchange transfers show
    up in ``budget_ptend`` (as they should: they are real sources/sinks
    of advected mass).

``budget_ptend_<sp>``
    Net physics column tendency [kg/m²/s]: the SUM over every term of the
    accumulated ``m_*`` tracer tendencies. Unlike the ``emi_*`` /
    ``wet_*`` / ``dry_*`` ledgers this includes *every* pathway —
    chemistry production, cloud-borne exchange, convective transport,
    re-injection — so ``ptend − (emi − wet − dry)`` measures the
    unledgered physics per species.

``budget_dyn_<sp>``
    The DYNAMICS residual [kg/m²/s]: ``(mass_now − expected_prev)/dt``
    with ``expected_prev = mass_prev + dt·ptend_prev`` carried across the
    step. The host applies exactly ``q += dt·tend`` and then transports
    (semi-Lagrangian advection, filters, the dynamics→physics positivity
    projection), so this residual is everything the transport machinery
    created or destroyed. A conservative core reads ~0; dinosaur's
    semi-Lagrangian tracer transport is *documented* non-conservative
    (its own validation test: "semi-Lagrangian transport does not
    conserve mass exactly"), and this field is where that error becomes
    visible per species, per column, per step. Zero on the first step
    (no previous expectation).

All three are per-column 2D fields riding the ordinary diagnostics
output (time-averaged like everything else under ``output_averages``).
Global closure: ``d(budget_mass)/dt == budget_ptend + budget_dyn`` by
construction, so the *global-mean* ``budget_dyn`` is the transport leak
and ``budget_ptend − ledgers`` the unledgered-physics leak.

The expectation carried between steps lives under the internal
``_budget_expected`` key (excluded from output).
"""

from __future__ import annotations

import jax.numpy as jnp

#: Internal carry key for the lagged expectation.
CARRY_KEY = "_budget_expected"


def gauge_aerosol_budget(
    diagnostics: dict,
    state,
    tracer_tends: dict,
    dt: float,
) -> dict:
    """Emit the per-species budget fields; thread the lagged expectation.

    No-op (returns ``diagnostics`` unchanged) when the composition carries
    no aerosol mass tracers or no air-mass diagnostics — both are
    trace-time-static properties of the physics package, so the carry
    structure is stable across steps.
    """
    rho = diagnostics.get("air_density")
    dz = diagnostics.get("layer_thickness")
    if rho is None or dz is None:
        return diagnostics
    from jcm.physics.aerosol.jam.emissions.flux_diagnostic import (
        DEPOSITED_SPECIES,
        _species_of,
    )
    by_species: dict[str, list[str]] = {}
    for nm in state.tracers:
        if not nm.startswith("m_"):
            continue
        sp = _species_of(nm)
        if sp in DEPOSITED_SPECIES:
            by_species.setdefault(sp, []).append(nm)
    if not by_species:
        return diagnostics

    dm = rho * dz                                   # (nlev, ncols) [kg/m²]
    prev_expected = diagnostics.get(CARRY_KEY)
    # Validity flag rides the carry: ``get_empty_data`` traces this
    # function and ZERO-FILLS the resulting carry template, and the
    # initial physics carry hands that template back on the first step —
    # so ``prev_expected`` is present-but-structural there, and treating
    # it as a real expectation would report the entire initial burden as
    # a fictitious dynamics source on step one (harmless from a zero
    # cold start, badly wrong from a warm start). The flag is written as
    # 1 by every real step and is 0 exactly when the expectation is the
    # zero-filled seed.
    valid = jnp.asarray(0.0, dm.dtype)
    if prev_expected is not None:
        valid = prev_expected.get("_valid", valid) * jnp.ones((), dm.dtype)
    out = dict(diagnostics)
    expected: dict[str, jnp.ndarray] = {}
    for sp, names in sorted(by_species.items()):
        mass = sum(
            jnp.sum(jnp.asarray(state.tracers[nm]) * dm, axis=0)
            for nm in names
        )
        ptend = sum(
            jnp.sum(tracer_tends[nm] * dm, axis=0)
            for nm in names if nm in tracer_tends
        )
        ptend = ptend + jnp.zeros_like(mass)
        out[f"budget_mass_{sp}"] = mass
        out[f"budget_ptend_{sp}"] = ptend
        if prev_expected is not None and sp in prev_expected:
            out[f"budget_dyn_{sp}"] = (
                valid * (mass - prev_expected[sp]) / dt
            )
        else:
            out[f"budget_dyn_{sp}"] = jnp.zeros_like(mass)
        expected[sp] = mass + dt * ptend
    expected["_valid"] = jnp.ones((), dm.dtype)
    out[CARRY_KEY] = expected
    return out
