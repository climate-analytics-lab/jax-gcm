"""Operator-split tracer view for the aerosol removal terms.

``ComposablePhysics`` hands every term the STEP-START state and sums the
tendencies. Sedimentation, dry deposition and wet scavenging each bound
their own removal at 100 % of what they see, so their sum is unbounded and
a raining marine cell can lose more mass than it holds.

ECHAM and CAM avoid this by operator splitting: each process acts on the
working copy the previous ones left. ``split_view`` reconstructs that copy
from the running tendency the driver publishes as ``_tendency_run``, so
removing a fraction of what remains can never exceed the whole.

The reconstruction folds in every term already run this step that returns
tracer TENDENCIES, not only the removal chain: emissions, tracer vertical
diffusion, convective transport and the sulfur chemistry all precede
sedimentation in ``jam_aerosol_physics``. Aerosol emitted or formed this
step is therefore present to be removed, and aerosol convection has
already exported is not. Terms that transform tracers in place rather
than through the tendency dict — the MAM4 core and ice nucleation read
``tracer_view`` — are outside it.

Cloud-borne tracers live in the physics carry, which the removal terms
already integrate sequentially through ``cloud_borne_store.apply_updates``,
so they pass through unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.cloud_borne_store import tracer_view


def split_view(spec, state, diagnostics) -> dict:
    """Tracer values as the terms already run this step left them.

    Falls back to the step-start values when no running tendency is
    published (a term exercised standalone, or the structural probe); both
    driver hosts publish it.
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
