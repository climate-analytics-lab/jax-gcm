"""Reverse-mode gradients must stay finite through the ECHAM physics stack.

Regression guard for issue #558: multi-step reverse-mode differentiation
through the ECHAM column physics used to return NaN while the forward pass
and a central finite difference were both finite. The cause was a class of
degenerate-state cotangent poisons — ``sqrt(0)``, ``x**p`` at ``x == 0`` for
fractional ``p``, and ``a/0`` — sitting in ``where``-masked branches. The
forward masks the bad value, but the masked branch's derivative is ``inf`` and
``0 * inf = nan`` poisons the gradient once a second step chains through it.

These are triggered by degenerate states that are entirely normal in practice:
an aquaplanet (zero sub-grid orography → every SSO orography denominator is 0)
and a balanced isothermal start (zero wind → every ``sqrt(u**2 + v**2)`` has an
infinite derivative), plus clear/ice-free cells (cloud fraction / condensate 0
under a fractional power).

The tests below differentiate ``mean(temperature)`` after two model steps with
respect to the solar constant, for the 1M and 2M cloud schemes with and without
the JAM prognostic-aerosol chain — the four configurations calibration work
relies on. All must be finite; the headline config is additionally checked
against a central finite difference.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from jcm.forcing import default_forcing
from jcm.model import Model
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.runners import inject_balanced_isothermal_profile
from jcm.utils import get_coords


@pytest.fixture(autouse=True)
def _enable_x64():
    """Enable float64 for the duration of each test, then restore.

    x64 is required to reproduce the issue-#558 configuration and for a
    meaningful FD comparison, but ``jax_enable_x64`` is a *process-global*
    flag. Flipping it at import time leaks into sibling tests (pytest imports
    this module during collection even under ``-m "not slow"``), corrupting
    their float32 dtype assertions. Scope it here and restore the prior value.
    """
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)

_STEPS = 2
_S0 = 1361.0


def _mean_temperature_after_two_steps(solar_constant, *, cloud_scheme, aerosol_module):
    """d/dS0 target: mean air temperature after ``_STEPS`` op-split steps.

    Uses the model's per-step function directly (a plain Python loop rather
    than the outer ``lax.scan``) so the graph is a fixed unroll — this is
    exactly the chained backward that exposed #558.
    """
    coords = get_coords(get_echam_levels(47), spectral_truncation=21)
    forcing = default_forcing(coords.horizontal)
    rad = dataclasses.replace(RadiationParameters.default(), solar_constant=solar_constant)
    kwargs = dict(
        radiation=rad, radiation_scheme="grey", checkpoint_terms=False,
        cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
    )
    if aerosol_module == "jam":
        # Exercise the JAM chain without the optional GPL MAM4 extra.
        kwargs["jam_microphysics"] = "placeholder"
    physics = echam_physics(**kwargs)
    model = Model(coords=coords, physics=physics, time_step=15.0)
    inject_balanced_isothermal_profile(model)

    step = model._get_op_split_step_fn(forcing)
    state = model._final_dycore_state
    physics_state = model._final_physics_state
    for _ in range(_STEPS):
        state, physics_state = step(state, physics_state)
    return jnp.mean(model.dycore.to_physics_state(state).temperature)


_CONFIGS = [
    ("1m", "macv2sp"),
    ("2m", "macv2sp"),
    ("1m", "jam"),
    ("2m", "jam"),
]


@pytest.mark.slow
@pytest.mark.parametrize("cloud_scheme,aerosol_module", _CONFIGS)
def test_two_step_gradient_is_finite(cloud_scheme, aerosol_module):
    """Reverse-mode d(meanT)/d(solar_constant) is finite for every config."""
    grad = jax.grad(
        lambda s: _mean_temperature_after_two_steps(
            s, cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
        )
    )(jnp.asarray(_S0))
    assert jnp.isfinite(grad), (
        f"{cloud_scheme}/{aerosol_module}: reverse-mode gradient is "
        f"{grad} — a degenerate-state cotangent poison has been "
        "re-introduced (issue #558)."
    )


@pytest.mark.slow
def test_two_step_gradient_matches_finite_difference():
    """AD gradient matches a central finite difference (headline config).

    Uses 2M + JAM — the most comprehensive stack (2-moment microphysics plus
    the full prognostic-aerosol chain) — so the check exercises the largest
    set of guarded terms.
    """
    cfg = dict(cloud_scheme="2m", aerosol_module="jam")
    ad = jax.grad(
        lambda s: _mean_temperature_after_two_steps(s, **cfg)
    )(jnp.asarray(_S0))

    eps = 5.0
    fd = (
        _mean_temperature_after_two_steps(jnp.asarray(_S0 + eps), **cfg)
        - _mean_temperature_after_two_steps(jnp.asarray(_S0 - eps), **cfg)
    ) / (2.0 * eps)

    assert jnp.isfinite(ad)
    # Loose tolerance: the gradient is ~5e-6 and the FD carries O(eps**2)
    # truncation error; we only need to confirm AD is right, not exact.
    assert abs(float(ad) - float(fd)) <= 1e-8, (
        f"AD {float(ad):.6e} disagrees with central FD {float(fd):.6e}"
    )
