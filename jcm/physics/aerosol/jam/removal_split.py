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
already exported is not. The microphysics core, optics and ice nucleation read the same working
copy; each process returns only its own change, so sources and removal are
applied once. Cloud-borne updates already integrated in the carry are
not reconstructed from interstitial tendencies.

Cloud-borne tracers live in the physics carry, which the removal terms
already integrate sequentially through ``cloud_borne_store.apply_updates``,
so they pass through unchanged.
"""

from __future__ import annotations

from jcm.physics.aerosol.jam.cloud_borne_store import tracer_view
from jcm.physics_interface import working_tracers


def split_view(spec, state, diagnostics) -> dict:
    """Tracer values as the terms already run this step left them.

    Falls back to the step-start values when no running tendency is
    published (a term exercised standalone, or the structural probe); both
    driver hosts publish it.
    """
    view = (state.tracers if spec is None
            else tracer_view(spec, state, diagnostics))
    return working_tracers(state, diagnostics, base=view)
