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

The tests differentiate ``mean(temperature)`` after two model steps with
respect to the solar constant, for the 1M and 2M cloud schemes with and without
the JAM prognostic-aerosol chain — the four configurations calibration work
relies on.

They run in **float64** (the calibration / JEM-Cal use case and the precision
#558 was reported at). ``jax_enable_x64`` is a *process-global* flag, and
toggling it mid-session — even via a scoped context manager — corrupts the JAX
compilation cache shared by other tests under pytest-xdist (an int64-compiled
program served int32 inputs: ``RuntimeProgramInputMismatch``). So each config
is differentiated in a **fresh subprocess** that enables x64 from startup; the
parent process (and its float32 sibling tests) is never touched. The subprocess
entry point is this module's ``__main__`` block.
"""

import dataclasses
import subprocess
import sys

import pytest

_STEPS = 2
_S0 = 1361.0
# d(meanT)/d(solar_constant) after two steps from the balanced isothermal
# aquaplanet start. Radiation-dominated, so identical across cloud/aerosol
# configs. Validated against a float64 central FD (5.2105e-6) in PR #559.
_EXPECTED_GRAD = 5.21e-6

_CONFIGS = [
    ("1m", "macv2sp"),
    ("2m", "macv2sp"),
    ("1m", "jam"),
    ("2m", "jam"),
]


def _grad_in_subprocess(cloud_scheme, aerosol_module):
    """Run the two-step gradient for one config in a fresh x64 subprocess.

    Returns the gradient as a Python float. Raises ``AssertionError`` with the
    captured output if the subprocess fails or the marker line is missing.
    """
    proc = subprocess.run(
        [sys.executable, __file__, cloud_scheme, aerosol_module],
        capture_output=True, text=True, timeout=1800,
    )
    marker = "GRADRESULT "
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith(marker)), None
    )
    assert line is not None, (
        f"{cloud_scheme}/{aerosol_module}: subprocess produced no result "
        f"(returncode {proc.returncode}).\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    return float(line[len(marker):])


@pytest.mark.slow
@pytest.mark.parametrize("cloud_scheme,aerosol_module", _CONFIGS)
def test_two_step_gradient_is_finite_and_correct(cloud_scheme, aerosol_module):
    """Reverse-mode d(meanT)/d(solar_constant) is finite and correct.

    Finiteness is the #558 guard (a re-introduced degenerate-state poison
    NaNs the cotangent, which ``float(...)`` surfaces as ``nan``); the value
    check additionally catches a guard that silently changes the physics.
    """
    grad = _grad_in_subprocess(cloud_scheme, aerosol_module)

    import math
    assert math.isfinite(grad), (
        f"{cloud_scheme}/{aerosol_module}: reverse-mode gradient is {grad} — "
        "a degenerate-state cotangent poison has been re-introduced (#558)."
    )
    assert grad == pytest.approx(_EXPECTED_GRAD, rel=2e-2), (
        f"{cloud_scheme}/{aerosol_module}: gradient {grad:.4e} is far from "
        f"the validated {_EXPECTED_GRAD:.4e}."
    )


def _main(cloud_scheme, aerosol_module):
    """Subprocess entry point: print ``GRADRESULT <float>`` (nan if poisoned)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jcm.forcing import default_forcing
    from jcm.model import Model
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.physics.echam.echam_terms import echam_physics
    from jcm.physics.radiation.radiation_types import RadiationParameters
    from jcm.runners import inject_balanced_isothermal_profile
    from jcm.utils import get_coords

    def objective(solar_constant):
        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        forcing = default_forcing(coords.horizontal)
        rad = dataclasses.replace(
            RadiationParameters.default(), solar_constant=solar_constant,
        )
        kwargs = dict(
            radiation=rad, radiation_scheme="grey", checkpoint_terms=False,
            cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
        )
        if aerosol_module == "jam":
            # Exercise the JAM chain without the optional GPL MAM4 extra.
            kwargs["jam_microphysics"] = "placeholder"
        model = Model(
            coords=coords, physics=echam_physics(**kwargs), time_step=15.0,
        )
        inject_balanced_isothermal_profile(model)
        # Use the per-step function directly (a plain Python loop, not the
        # outer lax.scan) so the graph is a fixed unroll — exactly the chained
        # backward that exposed #558.
        step = model._get_op_split_step_fn(forcing)
        state = model._final_dycore_state
        physics_state = model._final_physics_state
        for _ in range(_STEPS):
            state, physics_state = step(state, physics_state)
        return jnp.mean(model.dycore.to_physics_state(state).temperature)

    grad = jax.grad(objective)(jnp.asarray(_S0))
    print(f"GRADRESULT {float(grad)}")


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
