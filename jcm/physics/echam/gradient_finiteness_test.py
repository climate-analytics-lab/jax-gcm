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


# --- Cloud-parameter gradient through a long rollout (#558 residual) ----------
#
# A distinct #558 poison surfaced only when differentiating a *cloud* parameter
# (not the solar constant) through a longer rollout: finite at <=8 steps, NaN at
# >=12. Root cause: ``combine_optical_properties`` (grey radiation) and
# ``cloud_sw_optics`` (cloud_optics) combined single-scatter-albedo / asymmetry
# with a bare ``jnp.where(tau>0, scattering/tau, 0)`` — the true branch divides
# by the *same* quantity the mask tests, so a clear/aerosol-free layer computes
# ``x/0`` whose derivative is ``inf`` and ``where``'s VJP forms ``0*inf = nan``.
# Backward-only (forward finite), cloud-parameter-driven (crt -> cloud fraction
# -> whether a layer's tau hits 0), rollout-dependent (a layer crosses into
# clear ~10 steps in). Fixed with the safe-denominator double-``where`` in both
# functions. See jcm/physics/radiation/{grey_two_stream/radiation_scheme.py,
# cloud_optics.py}.
#
# This gradient is genuinely tiny (~1e-81: a cloud parameter barely moves mean-T
# over a near-clear aquaplanet), so the meaningful signal is finite-vs-NaN, not
# a value. It also has no float32 representation — intermediate cotangents
# overflow float32's ~3.4e38 ceiling and NaN regardless of the (fixed) poison —
# so unlike the solar-constant test above it must run in float64. Toggling the
# process-global ``jax_enable_x64`` in the shared pytest process would corrupt
# the xdist-shared compilation cache and leak float64 into sibling tests, so the
# rollout runs in an isolated subprocess (below). A ``lax.scan`` (not a Python
# unroll) keeps the compiled graph a single step body: a 12-step unroll of the
# full ECHAM column exhausts the LLVM section-memory / mmap map-count on CI.
_CLOUD_ROLLOUT_STEPS = 12


def _run_cloud_gradient_subprocess() -> None:
    """Entry point executed in the isolated float64 subprocess.

    Differentiates ``mean(temperature)`` after ``_CLOUD_ROLLOUT_STEPS`` op-split
    steps w.r.t. the Sundqvist ``crt`` critical-relative-humidity parameter and
    prints ``FINITE``/``NONFINITE`` for the parent test to assert on.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jcm.physics.clouds.sundqvist import CloudParameters

    cld0 = CloudParameters.default()

    def mean_temperature(crt):
        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        forcing = default_forcing(coords.horizontal)
        physics = echam_physics(
            clouds=dataclasses.replace(cld0, crt=crt),
            radiation_scheme="grey", checkpoint_terms=False,
        )
        model = Model(coords=coords, physics=physics, time_step=15.0)
        inject_balanced_isothermal_profile(model)
        model._final_physics_state = model._build_initial_physics_carry()
        step = model._get_op_split_step_fn(forcing)
        carry0 = (model._final_dycore_state, model._final_physics_state)

        def body(carry, _):
            return step(*carry), None

        (state, _), _ = jax.lax.scan(
            body, carry0, xs=None, length=_CLOUD_ROLLOUT_STEPS,
        )
        return jnp.mean(model.dycore.to_physics_state(state).temperature)

    grad = jax.grad(mean_temperature)(jnp.asarray(float(cld0.crt)))
    print(f"grad={float(grad):.6e}")
    print("FINITE" if bool(jnp.isfinite(grad)) else "NONFINITE")


@pytest.mark.slow
def test_cloud_parameter_gradient_finite_through_long_rollout():
    """d(meanT)/d(cloud crt) through a 12-step rollout stays finite (#558).

    Guards the optical-property combination fix: a re-introduced
    divide-by-the-``where``-condition in radiation_scheme / cloud_optics NaNs
    this backward while the forward and the shorter solar-constant gradient
    above both stay finite. Runs float64 in an isolated subprocess (see the
    module note) so it never perturbs the in-process float32 tests.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, __file__, "__cloud_gradient__"],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"cloud-gradient subprocess failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "FINITE" in result.stdout.splitlines(), (
        "cloud-parameter gradient through a 12-step rollout is not finite — a "
        "divide-by-condition cotangent poison has been re-introduced (#558).\n"
        f"{result.stdout}"
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "__cloud_gradient__":
        _run_cloud_gradient_subprocess()
