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
respect to the solar constant, for the 1M and 2M cloud microphysics schemes,
through the ECHAM composition ``echam_physics()`` — RRTMGP radiation. RRTMGP's
reverse pass holds the per-g-point profiles of every column live at once
(``RRTMGPRadiation``), which at T21L47 takes the process past 100 GB; the test
sets ``JCM_RRTMGP_COL_CHUNKS`` so the backward rematerializes the columns in
blocks, at the cost of one extra forward radiation evaluation.

**What this does not cover, and where that lives.** One scalar input is a
narrow direction: a poison on a path the solar constant never reaches stays
invisible, and when one does fire this test can only say the total is NaN, not
which input carried it. ``term_gradients_test`` is the per-leaf counterpart —
it takes the same composition on a single column and differentiates each term,
and the whole package, one input leaf at a time, which is how the grey-radiation
Planck defect that this test walks straight past was found. It is kept there
rather than here because the leaf-by-leaf sweep wants a one-column
``compute_tendencies``, not a T21 rollout: the rollout is what makes this test
slow and it buys nothing that repeating it per leaf would not.
Both exercise the shared SSO / vertical-diffusion / surface / convection terms;
1M adds ``echam_1m`` (ice sedimentation guard) and 2M adds ``lohmann_2m`` +
``cloud_utils`` (effective-radius guard).

**What each case differentiates.** The 1M case is ``jax.jit(jax.grad(f))`` of
the whole two-step function: one compiled program, the form in which a
production run differentiates. A derivative that overflows only once XLA has
fused the program is NaN there and finite eagerly, which is what the tracer
mass fixer's was (#987), so an eager gradient cannot stand in for it. The 2M
case is the eager op-by-op gradient, the graph that exposed #558 and the one
``term_gradients_test`` replays term by term. Each way of differentiating is
therefore covered by one scheme. 2M under ``jit`` and 1M eagerly were each
measured finite when the split was made (#987), and are not run because each
is a further case of 10-13 GiB in the shard.

**Memory budget.** The job that runs this test is one process on a GitHub
runner with 4 vCPU and 15989 MiB (14809 MiB free at start, 3071 MiB of swap),
shared with every other test of its shard. The cases are arranged around three
facts, with peaks measured as resident set size of a fresh pytest process on
four pinned cores of the dev workstation (jax 0.10.2):

* *Rematerialisation pays only where the gradient is eager.* ``echam_physics``
  wraps each term in ``jax.checkpoint`` (``checkpoint_terms=True``, its
  default, which a model run uses), so an eager backward recomputes a term's
  forward instead of keeping its tape. The land skin balance reads the
  radiation's surface fluxes (#979), so every term after vertical diffusion is
  on the S0 path in step 1 as well as in step 2, and without rematerialisation
  Tiedtke alone would keep about 1.7 GB of tape per step. With it the live
  arrays at the end of the eager forward pass are 1468 MiB (5864 MiB without)
  and the 2M case peaks at 12042 MiB alone (14941 MiB without it, as measured
  for #979). A compiled program has no tape to drop, and there the same
  wrapper costs memory: the 1M ``jit(grad)`` with it peaked at 13217 MiB, took
  404 s to compile and needs 5.2 GiB of static temporaries; without it
  9896 MiB, 204 s and 3.8 GiB. The jitted case therefore sets
  ``checkpoint_terms=False``. With it the gradient is as finite and correct
  (1.19572e-05 either way), so the choice is memory, not coverage.
* *Each case starts from a cleared process.* The root ``conftest.py`` drops
  JAX's compiled-executable caches and returns the freed heap to the OS only at
  a class or module boundary, and the cases are parametrizations of one
  function of one module, so ``_fresh_process_memory`` does it before and after
  each. Without it the second case starts from the 10-14 GB the first (or the
  module before it) left mapped. The eager case runs first because it is the
  larger and the one that suffers most from a heap another case left behind
  (the other order peaks 0.5 GiB higher, at 12956 MiB). All four tests of the
  module in one fresh pytest process peak at 12448 MiB, 2361 MiB under the
  runner's free memory, and take 16 min on the loaded workstation.
* *About 5 GB of the eager peak is not array data*, but eager-mode executables
  and allocator slack. ``jax.jit`` of the differentiated function removes most
  of it: the jitted 1M case runs in 428 s at 9896 MiB, against 718 s at
  12042 MiB for the eager 2M case.

Precision: the ``0 * inf = nan`` poison is dtype-agnostic, so this runs at the
session's default (float32) — cheap, in-process, and it never touches the
process-global ``jax_enable_x64`` flag (toggling it mid-session corrupts the
xdist-shared compilation cache). The gradient matches its float64 value to
well within the tolerance here (validated against a central FD, 5.2105e-6, in
PR #559). The JAM prognostic-aerosol chain has a *separate* float32-only
degeneracy (finite in float64) and is validated manually in x64 rather than
here — see PR #559 / issue #558.
"""

import ctypes
import ctypes.util
import dataclasses
import gc

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import pytest

from jcm.date import DateData
from jcm.forcing import default_forcing
from jcm.model import Model
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.initial_states import balanced_isothermal_state
from jcm.utils import get_coords

_STEPS = 2
_S0 = 1361.0
# d(meanT)/d(solar_constant) after two steps from the balanced isothermal
# aquaplanet start with the production-seeded physics carry, RRTMGP radiation.
# Radiation-dominated, so identical across cloud configs (1M and 2M both give
# 1.1957e-5). A float32 central difference agrees within its own noise: 1.190e-5
# at a +-50 W m-2 step, where the float32 resolution of the mean temperature
# alone is ~1.5 %.
_EXPECTED_GRAD = 1.196e-5
# Column blocks for RRTMGP's reverse pass (see the module docstring): 2048 T21
# columns in blocks of 128 bring the peak RSS to ~13 GB on CPU. Finer blocks do
# not lower it further; the rest is the model's own backward.
_RRTMGP_COL_CHUNKS = "16"

# (cloud scheme, aerosol module, differentiate under ``jax.jit``). 1M is the
# whole-function ``jit(grad)`` a production run's compiled step is, 2M the eager
# op-by-op gradient; see "What each case differentiates" in the module docstring.
_CONFIGS = [
    pytest.param("2m", "macv2sp", False, id="2m-macv2sp-eager"),
    pytest.param("1m", "macv2sp", True, id="1m-macv2sp-jit"),
]


def _mean_temperature_after_two_steps(solar_constant, *, cloud_scheme,
                                      aerosol_module, checkpoint_terms=True):
    """d/dS0 target: mean air temperature after ``_STEPS`` op-split steps.

    Uses the model's per-step function directly (a plain Python loop rather
    than the outer ``lax.scan``) so the graph is a fixed unroll — this is
    exactly the chained backward that exposed #558.
    """
    # The model is built from concrete values. Under ``jax.jit`` those
    # constructions must run at trace time rather than be staged, because the
    # build reads their values (``int(model.dt_si.m)``, grid constants); only
    # the solar constant is traced. Eagerly this context changes nothing.
    with jax.ensure_compile_time_eval():
        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        forcing = default_forcing(coords.horizontal)
        rad = dataclasses.replace(
            RadiationParameters.default(), solar_constant=solar_constant)
        # ``checkpoint_terms`` is ``echam_physics``' default, the setting a
        # model run uses, wherever the gradient is eager; see "Memory budget"
        # in the module docstring for why the jitted case turns it off.
        physics = echam_physics(
            radiation=rad, checkpoint_terms=checkpoint_terms,
            cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
        )
        model = Model(coords=coords, physics=physics, time_step=15.0)
        # ``bootstrap_state`` populates ``_final_dycore_state`` from the
        # balanced-isothermal start AND seeds the cross-step physics carry
        # exactly as the production rollout / resume path does (Model.run and
        # Model.resume both build it when None — model.py). Stepping with a
        # ``None`` carry would synthesise a *zero* carry, so e.g. the TTE-TKE
        # term would start from TKE=0 instead of its seeded ECHAM 0.01 floor —
        # a state production never produces — so we bootstrap here to
        # differentiate the same trajectory the model actually runs. (The #558
        # poison triggers — SSO zero-orography denominators, the zero-wind
        # ``sqrt(u**2+v**2)``, and clear/ice-free fractional powers — are all
        # independent of this carry and remain exercised.)
        model.bootstrap_state(balanced_isothermal_state(model))

    step = model._get_op_split_step_fn(forcing)
    state = model._final_dycore_state
    physics_state = model._final_physics_state
    clock = model.start_time
    for model_step in range(_STEPS):
        date = DateData(
            clock, jnp.int32(model_step), int(model.dt_si.m))
        state, physics_state = step(state, physics_state, date)
        clock = clock + jdt.Timedelta(seconds=jnp.int32(model.dt_si.m))
    return jnp.mean(model.dycore.to_physics_state(state).temperature)


def _release_process_memory():
    """Drop compiled executables and return the freed heap to the OS.

    The same three steps ``conftest.py`` takes at a class or module boundary
    (``jax.clear_caches``, ``gc.collect``, glibc ``malloc_trim``), unconditional
    here: clearing only frees memory to the allocator, and glibc keeps freed
    heap mapped, so without the trim the process size does not go down.
    """
    jax.clear_caches()
    gc.collect()
    name = ctypes.util.find_library("c")
    trim = getattr(ctypes.CDLL(name), "malloc_trim", None) if name else None
    if trim is not None:
        trim(0)


@pytest.fixture(autouse=True)
def _fresh_process_memory():
    """Run each case in a cleared, trimmed process (see "Memory budget")."""
    _release_process_memory()
    yield
    _release_process_memory()


@pytest.mark.slow
@pytest.mark.parametrize("cloud_scheme,aerosol_module,jit", _CONFIGS)
def test_two_step_gradient_is_finite_and_correct(cloud_scheme, aerosol_module,
                                                  jit, monkeypatch):
    """Reverse-mode d(meanT)/d(solar_constant) is finite and correct.

    Finiteness is the #558 guard (a re-introduced degenerate-state poison NaNs
    the cotangent); the value check additionally catches a guard that silently
    changes the physics. With ``jit`` the whole differentiated function is one
    compiled program, the form in which a production run differentiates; a
    cotangent that is NaN only once XLA has fused the program (#987) is
    invisible to the eager case.
    """
    # Read when RRTMGP's column map is traced, i.e. inside jax.grad below.
    monkeypatch.setenv("JCM_RRTMGP_COL_CHUNKS", _RRTMGP_COL_CHUNKS)
    gradient = jax.grad(
        lambda s: _mean_temperature_after_two_steps(
            s, cloud_scheme=cloud_scheme, aerosol_module=aerosol_module,
            checkpoint_terms=not jit,
        )
    )
    grad = (jax.jit(gradient) if jit else gradient)(jnp.asarray(_S0))
    mode = "jit" if jit else "eager"

    assert jnp.isfinite(grad), (
        f"{cloud_scheme}/{aerosol_module} ({mode}): reverse-mode gradient is "
        f"{grad} — a degenerate-state cotangent poison has been re-introduced "
        "(#558; under jit also a derivative that overflows only in the compiled "
        "program, #987)."
    )
    # 2% tolerance absorbs the float32-vs-float64 difference; the poison-vs-clean
    # signal is NaN-vs-finite, and a wrong-but-finite guard would miss by far
    # more than 2%.
    assert float(grad) == pytest.approx(_EXPECTED_GRAD, rel=2e-2), (
        f"{cloud_scheme}/{aerosol_module} ({mode}): gradient "
        f"{float(grad):.4e} is far from the validated {_EXPECTED_GRAD:.4e}."
    )


# --- Radiation optical-property combination: divide-by-condition poison (#558)
#
# A distinct #558 poison surfaced only when differentiating a *cloud* parameter
# (not the solar constant) through a longer rollout: finite at <=8 steps, NaN at
# >=12, forward finite at every step. Root cause: two grey-radiation optical
# combinations weighted single-scatter albedo / asymmetry with a bare
# ``jnp.where(tau > 0, scattering / tau, 0)`` — the true branch divides by the
# *same* quantity the mask tests, so a clear (and aerosol-free) layer, where that
# denominator is 0, computes ``x/0``: finite-masked in the forward, but its
# derivative is ``inf`` and ``where``'s VJP forms ``0 (mask) * inf = nan``. That
# poisons the gradient of any upstream cloud parameter and only accumulates
# enough sensitivity to surface past ~10 rollout steps. Fixed (safe-denominator
# double-``where``) in ``combine_optical_properties`` and ``cloud_optics``.
#
# These guards test the two functions directly with a deliberately clear layer
# (zero denominator) in the differentiated cell — the exact poison, without the
# expense of a multi-step model rollout. The end-to-end x64 rollout gradient is
# ~1e-81 and float32-unrepresentable, so it lives in the JEM-Cal calibration
# suite; here we pin the mechanism cheaply and deterministically.


def test_combine_optical_properties_gradient_finite_in_clear_layer():
    """``combine_optical_properties`` backward is finite where total tau/scat = 0.

    A clear, aerosol-free layer drives ``total_tau_with_aerosol`` and
    ``total_scattering`` to 0; a re-introduced ``where(x>0, .../x, 0)`` there
    NaNs the reverse pass while the forward stays finite (#558).
    """
    from jcm.physics.radiation.grey_two_stream.radiation_scheme import (
        combine_optical_properties,
    )
    from jcm.physics.radiation.radiation_types import OpticalProperties

    nlev, nbands = 6, 8
    # Layer 0 has zero gas optical depth too, so total tau (not just scattering)
    # vanishes there — exercises both guarded divisions.
    gas_tau = jnp.zeros((nlev, nbands)).at[1:].set(0.01)
    clear_cloud = OpticalProperties(
        optical_depth=jnp.zeros((nlev, nbands)),
        single_scatter_albedo=jnp.zeros((nlev, nbands)),
        asymmetry_factor=jnp.zeros((nlev, nbands)),
    )
    zeros = jnp.zeros((nlev, nbands))

    def summed(aerosol_optical_depth):
        combined = combine_optical_properties(
            gas_tau, clear_cloud, aerosol_optical_depth, zeros, zeros,
        )
        return jnp.sum(
            combined.single_scatter_albedo + combined.asymmetry_factor
        )

    grad = jax.grad(summed)(zeros)
    assert jnp.all(jnp.isfinite(grad)), (
        "combine_optical_properties gradient is non-finite in a clear layer — a "
        "divide-by-the-where-condition cotangent poison is back (#558)."
    )


def test_cloud_optics_gradient_finite_in_clear_layer():
    """``cloud_optics`` SW backward is finite where a layer is cloud-free.

    A layer with zero liquid *and* ice path has ``tau_total == 0``, so the
    tau-weighted ssa / asymmetry combination divides by 0; the guard keeps the
    differentiated branch finite while the outer ``where`` returns clear-sky
    values (#558).
    """
    from jcm.physics.radiation.cloud_optics import cloud_optics

    # Interleave cloudy and cloud-free (cw == ci == 0) layers.
    cloud_water_path = jnp.array([0.0, 1e-3, 0.0, 2e-3, 0.0, 0.0])
    cloud_ice_path = jnp.array([0.0, 0.0, 1e-3, 0.0, 0.0, 0.0])
    layer_thickness = jnp.full((6,), 500.0)
    cdnc_factor = jnp.array(1.0)

    def summed(cwp):
        sw_optics, _ = cloud_optics(
            cwp, cloud_ice_path, layer_thickness, cdnc_factor,
        )
        return jnp.sum(
            sw_optics.single_scatter_albedo + sw_optics.asymmetry_factor
        )

    grad = jax.grad(summed)(cloud_water_path)
    assert jnp.all(jnp.isfinite(grad)), (
        "cloud_optics SW gradient is non-finite in a cloud-free layer — a "
        "divide-by-the-where-condition cotangent poison is back (#558)."
    )
