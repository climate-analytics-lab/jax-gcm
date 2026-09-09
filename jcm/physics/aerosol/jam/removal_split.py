"""Operator-split tracer view shared by the aerosol removal terms.

Sedimentation, dry deposition and wet scavenging each bound their own
removal at 100 % of what they see, but ``ComposablePhysics`` hands every
term the STEP-START state and sums the returned tendencies, so three
independently-bounded sinks can sum past the available mass (a raining
marine surface cell removed 164 % of its coarse sea salt).

ECHAM and CAM avoid this by operator splitting: each removal process acts
on the state the previous one left. ``split_view`` reproduces that from the
running tendency ``ComposablePhysics`` already publishes
(``_tendency_run``), so the removal chain is sequential without the driver
having to apply tendencies mid-step. Removing a fraction of what remains
can never exceed the whole, and each term still reports exactly the mass it
took, keeping the ``dry_*``/``wet_*`` ledgers consistent with the state
change.

Cloud-borne tracers live in the physics carry, which the removal terms
already integrate sequentially via ``cloud_borne_store.apply_updates``, so
they pass through unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.cloud_borne_store import tracer_view


def split_view(spec, state, diagnostics) -> dict:
    """Tracer values as the terms already run this step left them.

    Falls back to the step-start values when no running tendency is
    published (a term exercised standalone, or the structural probe).
    """
    view = tracer_view(spec, state, diagnostics)
    run = diagnostics.get("_tendency_run")
    if run is None:
        return view
    dt = diagnostics.get("_dt_seconds", 1800.0)
    updated = dict(view)
    for name, tendency in run.get("tracers", {}).items():
        prev = updated.get(name)
        if prev is None or jnp.ndim(tendency) != jnp.ndim(prev):
            continue
        updated[name] = prev + dt * tendency
    return updated
