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

The tests below differentiate ``mean(temperature)`` after two model steps with
respect to the solar constant, for the 1M and 2M cloud schemes with and without
the JAM prognostic-aerosol chain — the four configurations calibration work
relies on. The ``0 * inf = nan`` poison is dtype-agnostic, so these run under
the session's default precision (no process-global ``jax_enable_x64`` toggle,
which would corrupt sibling tests' compilation caches under xdist); the
gradient is checked both for finiteness and against its known-correct value
(cross-checked against a central finite difference in float64, 5.2105e-6, when
this fix was validated — see PR #559 / issue #558).
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
# aquaplanet start. Radiation-dominated, so identical across cloud/aerosol
# configs. Validated against a float64 central FD (5.2105e-6) in PR #559.
_EXPECTED_GRAD = 5.21e-6


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
def test_two_step_gradient_is_finite_and_correct(cloud_scheme, aerosol_module):
    """Reverse-mode d(meanT)/d(solar_constant) is finite and correct.

    Finiteness is the #558 guard (a re-introduced degenerate-state poison
    NaNs the cotangent); the value check additionally catches a guard that
    silently changes the physics.
    """
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
    # 2% tolerance absorbs float32-vs-float64 differences; the poison-vs-clean
    # signal is NaN-vs-finite, and a wrong-but-finite guard would miss by far
    # more than 2%.
    assert float(grad) == pytest.approx(_EXPECTED_GRAD, rel=2e-2), (
        f"{cloud_scheme}/{aerosol_module}: gradient {float(grad):.4e} is far "
        f"from the validated {_EXPECTED_GRAD:.4e}."
    )
