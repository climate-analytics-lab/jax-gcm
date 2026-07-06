"""Reverse-mode gradients must stay finite through the ECHAM physics stack.

Regression guard for issue #558: multi-step reverse-mode differentiation
through the ECHAM column physics used to return NaN while the forward pass
and a central finite difference were both finite. The cause was a class of
degenerate-state cotangent poisons — ``sqrt(0)``, ``x**p`` at ``x == 0`` for
fractional ``p``, and ``a/0`` — sitting in ``where``-masked branches. The
forward masks the bad value, but the masked branch's derivative is ``inf`` and
``0 * inf = nan`` poisons the gradient once a second step chains through it.

These are triggered by states that are entirely normal in practice: an
aquaplanet (zero sub-grid orography → every SSO orography denominator is 0)
and a balanced isothermal start (zero wind → every ``sqrt(u**2 + v**2)`` has an
infinite derivative), plus clear/ice-free cells (cloud fraction / condensate 0
under a fractional power).

The test differentiates ``mean(temperature)`` after two model steps with
respect to the solar constant, for the 1M and 2M cloud microphysics schemes.
Both exercise the shared SSO / vertical-diffusion / surface / convection terms;
1M adds ``echam_1m`` (ice sedimentation guard) and 2M adds ``lohmann_2m`` +
``cloud_utils`` (effective-radius guard).

Precision: the ``0 * inf = nan`` poison is dtype-agnostic, so this runs at the
session's default (float32) — cheap, in-process, and it never touches the
process-global ``jax_enable_x64`` flag (toggling it mid-session corrupts the
xdist-shared compilation cache). The gradient matches its float64 value to
well within the tolerance here (validated against a central FD, 5.2105e-6, in
PR #559). The JAM prognostic-aerosol chain has a *separate* float32-only
degeneracy (finite in float64) and is validated manually in x64 rather than
here — see PR #559 / issue #558.
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

_STEPS = 2
_S0 = 1361.0
# d(meanT)/d(solar_constant) after two steps from the balanced isothermal
# aquaplanet start with the production-seeded physics carry. Radiation-dominated,
# so identical across cloud configs (1M and 2M both give 5.2104e-6). Matches a
# float64 central FD (5.2105e-6); validated in PR #559.
_EXPECTED_GRAD = 5.21e-6

_CONFIGS = [
    ("1m", "macv2sp"),
    ("2m", "macv2sp"),
]


def _mean_temperature_after_two_steps(solar_constant, *, cloud_scheme, aerosol_module):
    """d/dS0 target: mean air temperature after ``_STEPS`` op-split steps.

    Uses the model's per-step function directly (a plain Python loop rather
    than the outer ``lax.scan``) so the graph is a fixed unroll — this is
    exactly the chained backward that exposed #558.
    """
    coords = get_coords(get_echam_levels(47), spectral_truncation=21)
    forcing = default_forcing(coords.horizontal)
    rad = dataclasses.replace(RadiationParameters.default(), solar_constant=solar_constant)
    physics = echam_physics(
        radiation=rad, radiation_scheme="grey", checkpoint_terms=False,
        cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
    )
    model = Model(coords=coords, physics=physics, time_step=15.0)
    inject_balanced_isothermal_profile(model)
    # Seed the cross-step physics carry exactly as the production rollout /
    # resume path does (Model.run and Model.resume both build it when None —
    # model.py). ``inject_*`` only populates ``_final_dycore_state``, leaving
    # ``_final_physics_state`` at its ``None`` construction default; stepping
    # with ``None`` synthesises a *zero* carry, so e.g. the TTE-TKE term would
    # start from TKE=0 instead of its seeded ECHAM 0.01 floor. That is a state
    # production never produces, so we must seed here to differentiate the same
    # trajectory the model actually runs. (The #558 poison triggers — SSO
    # zero-orography denominators, the zero-wind ``sqrt(u**2+v**2)``, and
    # clear/ice-free fractional powers — are all independent of this carry and
    # remain exercised.)
    model._final_physics_state = model._build_initial_physics_carry()

    step = model._get_op_split_step_fn(forcing)
    state = model._final_dycore_state
    physics_state = model._final_physics_state
    for _ in range(_STEPS):
        state, physics_state = step(state, physics_state)
    return jnp.mean(model.dycore.to_physics_state(state).temperature)


@pytest.mark.slow
@pytest.mark.parametrize("cloud_scheme,aerosol_module", _CONFIGS)
def test_two_step_gradient_is_finite_and_correct(cloud_scheme, aerosol_module):
    """Reverse-mode d(meanT)/d(solar_constant) is finite and correct.

    Finiteness is the #558 guard (a re-introduced degenerate-state poison NaNs
    the cotangent); the value check additionally catches a guard that silently
    changes the physics.
    """
    grad = jax.grad(
        lambda s: _mean_temperature_after_two_steps(
            s, cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
        )
    )(jnp.asarray(_S0))

    assert jnp.isfinite(grad), (
        f"{cloud_scheme}/{aerosol_module}: reverse-mode gradient is {grad} — "
        "a degenerate-state cotangent poison has been re-introduced (#558)."
    )
    # 2% tolerance absorbs the float32-vs-float64 difference; the poison-vs-clean
    # signal is NaN-vs-finite, and a wrong-but-finite guard would miss by far
    # more than 2%.
    assert float(grad) == pytest.approx(_EXPECTED_GRAD, rel=2e-2), (
        f"{cloud_scheme}/{aerosol_module}: gradient {float(grad):.4e} is far "
        f"from the validated {_EXPECTED_GRAD:.4e}."
    )
