"""Runnable examples for ``docs/source/gradients.rst`` (#241).

Each function below is literalincluded into that page, so the code shown
there is exactly the code ``docs/gradients_examples_test.py`` runs in the
fast test suite (under 60 s on CPU combined) — the examples cannot go stale
the way a hand-copied snippet in a doc can. See ``docs/source/gradients.rst``
for the narrative and the two gotchas that matter (the surrogate-gradient
pattern, and ``predictions.times`` under ``jvp``); this module is deliberately
just the code, with comments only where the CHOICE of example needs one.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


def held_suarez_temperature_response():
    """Build ``f(delta_k)``: mean temperature after a 1-day Held-Suarez run.

    Held-Suarez (``jcm.physics.held_suarez``) is the cheapest physics
    package in jcm — pure Newtonian relaxation + Rayleigh friction, no
    boundary-condition files, no moisture physics — so a short run through
    it is fast enough to differentiate interactively. ``delta_k`` is a
    uniform offset (Kelvin) added to the model's default initial
    temperature; only ``temperature_variation`` is perturbed (every other
    field stays exactly at its unperturbed value), which keeps the
    perturbation well inside Held-Suarez's smooth operating range at any
    ``delta_k`` used below. The returned ``f`` closes over one Model and one
    base state, built once, so repeated ``jvp``/``vjp``/``check_vjp`` calls
    on it reuse the same compiled integration rather than re-tracing it.
    """
    from jcm.model import Model
    from jcm.terrain import TerrainData
    from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
    from jcm.physics.held_suarez.utils import get_held_suarez_coords

    coords = get_held_suarez_coords(spectral_truncation=21)
    terrain = TerrainData.from_coords(coords)
    model = Model(coords=coords, terrain=terrain, time_step=180,
                  physics=held_suarez_physics())
    state0 = model._prepare_initial_dycore_state()

    def f(delta_k):
        perturbed = dataclasses.replace(
            state0, temperature_variation=state0.temperature_variation + delta_k)
        predictions = model.run(initial_state=perturbed,
                                save_interval=1, total_time=1)
        return predictions.dynamics.temperature.mean()

    return f


def speedy_mean_temperature_wrt_trvdi():
    """Build ``diagnostic(trvdi)``: jacfwd target for a differentiable parameter.

    ``trvdi`` (moisture-diffusion relaxation time, hours) is a
    ``VerticalDiffusionParameters`` leaf held in an ``nnx.Param`` on the
    vertical-diffusion ``PhysicsTerm`` (``jcm/physics/speedy/speedy_terms.py``;
    the CLAUDE.md "Differentiable physics parameters" pattern), so it is a
    pytree leaf ``jax.grad``/``jacfwd`` can see — the same leaf
    ``JAX_gotchas.md``'s "Static arguments hold whole objects" section edits
    by hand to show what NOT to do (mutate it in place after the physics has
    run). Differentiating through ``speedy_physics(parameters=...)`` is the
    supported way to change it: build a fresh ``Parameters`` (here, via
    ``dataclasses.replace`` so every other default stays untouched) and a
    fresh physics/Model from it, inside the traced function.
    """
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics
    from jcm.physics.speedy.params import Parameters

    coords = get_speedy_coords(spectral_truncation=21)
    base_params = Parameters.default()

    def diagnostic(trvdi):
        params = dataclasses.replace(
            base_params,
            vertical_diffusion=dataclasses.replace(
                base_params.vertical_diffusion, trvdi=trvdi))
        model = Model(coords=coords, physics=speedy_physics(parameters=params),
                     time_step=30)
        predictions = model.run(save_interval=1, total_time=1)
        return predictions.dynamics.temperature.mean()

    return diagnostic


def toy_function_for_check_gradients():
    """Return a cheap, non-physics function for demonstrating the checker APIs.

    ``jcm.testing.check_gradients`` and ``jax.test_util.check_jvp``/
    ``check_vjp`` all compare automatic differentiation against a
    finite-difference secant; their *mechanics* are independent of whether
    ``f`` happens to be a physics scheme. This toy keeps the demo of those
    mechanics itself fast; the two functions above are where the examples
    actually run jcm. (``jcm/model_test.py``'s ``test_speedy_model_*gradient*``
    tests and ``jcm/testing_test.py`` are the full-model worked examples for
    these same checkers; this module does not re-run those.)
    """
    return lambda x: jnp.sin(x) * x**2
