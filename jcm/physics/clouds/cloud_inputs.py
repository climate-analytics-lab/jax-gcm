"""The inputs of the tendency-driven stratiform cloud schemes.

ECHAM's ``cloud`` (``mo_cloud.f90``) and ``cloud_micro_interface``
(``mo_cloud_micro_2m.f90``) are driven by tendencies. They receive the state
at the previous time level (``ptm1``, ``pqm1``, ``pxlm1``, ``pxim1`` and, in
the 2M, the number tracers; ``physc.f90:1073-1074``) and the tendencies
accumulated since (``ptte``, ``pqte``, ``pxlte``, ``pxite``): the explicit
dynamics (``dyn.f90:244-467``, tracer transport ``mo_tpcore.f90:581-583``),
vertical diffusion (``physc.f90:678-717``), radiative heating (776-794),
gravity-wave drag (835-884) and convection (987-999). Convective detrainment
arrives separately as ``pxtecl``/``pxteci`` (``physc.f90:1081``;
``mo_cloud.f90:666-680``). Condensation is the humidity increment the
saturation humidity does not absorb, ``zqcdif = (ztmst·pqte − zdqsat)·paclc``
(``mo_cloud.f90:706-730``): the cloudy part is taken as saturated at the
anchor, and what the increments move it away by condenses.

jcm splits the physics sequentially from the dynamics: physics runs on the
post-dynamics state ``x_n``, the dycore adds the projected physics tendency
(giving the post-physics state ``x_ap``) and then runs the dynamics to
``x_{n+1}``. The cloud scheme left its cloudy part saturated at the previous
step's ``x_ap``, so the faithful mapping of ECHAM's inputs is

* anchor := the previous step's ``x_ap``, carried by the model
  (``_post_physics_state``, written from
  :meth:`jcm.dycore.base.DynamicalCore.after_physics_state`);
* increment := ``(x_n − x_ap) + dt·P_upstream − detrainment``, where
  ``x_n − x_ap`` is the dynamics of the last step and ``P_upstream`` is the
  running sum of the tendencies of the terms upstream of the cloud scheme
  (``_tendency_run``);
* detrainment := ``dt ×`` the convection scheme's detrained-condensate rate
  (``_convective_detrainment``), per step;
* provisional state := anchor + increment + detrainment, which is
  ``x_n + dt·P_upstream``.

Where no carried anchor is valid — the first step of a run, a checkpoint
written before the slot existed, a host with no dynamical core (single
column, RCE), or a composition none of whose terms asks for the slot — the
anchor is ``x_n`` and the dynamics increment is exactly zero. Which case
applies is decided by the carry's explicit validity flag, never by
inspecting field values.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from jcm.physics_interface import POST_PHYSICS_STATE_KEY, PhysicsState

#: Step-local diagnostics key under which a convection scheme publishes its
#: detrained condensate as ``{"qc": rate, "qi": rate}`` [kg/kg/s] — the
#: detrainment part of the qc/qi tendency it returns (ECHAM ``pxtecl``,
#: ``pxteci``). Dropped before the cross-step carry (see
#: ``ComposablePhysics._STEP_LOCAL_KEYS``); absent where the scheme detrains
#: nothing, which reads as zero.
CONVECTIVE_DETRAINMENT_KEY = "_convective_detrainment"

#: The condensate tracers that receive convective detrainment.
_DETRAINED = ("qc", "qi")


class CloudFields(NamedTuple):
    """Temperature, humidity and the cloud tracers at one time level.

    Every array is ``(nlev, *horiz)``, the layout of the state the term
    receives. ``tracers`` holds exactly the names the caller asked for.
    """

    temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    tracers: dict[str, jnp.ndarray]


class CloudSchemeInputs(NamedTuple):
    """ECHAM's cloud-scheme inputs, formed by :func:`cloud_scheme_inputs`.

    Attributes:
        anchor: ``ptm1``, ``pqm1``, ``pxlm1``, ``pxim1`` (and the number
            tracers): the carried post-physics state of the previous step,
            or the state the term receives where no valid carry exists.
        increment: ``ztmst`` times the accumulated tendencies ``ptte``,
            ``pqte``, ``pxlte``, ``pxite``: the dynamics since the anchor plus
            every upstream physics term, EXCLUDING convective detrainment.
        detrained_qc: ``ztmst·pxtecl``, detrained cloud liquid [kg/kg per
            step].
        detrained_qi: ``ztmst·pxteci``, detrained cloud ice [kg/kg per step].
        provisional: the state the cloud scheme's tendencies are relative to.
            Exactly ``anchor + (increment + detrained)`` for qc and qi and
            ``anchor + increment`` for every other field, bit for bit.
        dynamics_valid: scalar, 1 where the anchor is the carried
            post-physics state and 0 where it is the received state.

    """

    anchor: CloudFields
    increment: CloudFields
    detrained_qc: jnp.ndarray
    detrained_qi: jnp.ndarray
    provisional: CloudFields
    dynamics_valid: jnp.ndarray


def _running_tendency(diagnostics: dict, name: str, like: jnp.ndarray):
    """Return one field's upstream running tendency; zeros where none is published."""
    run = diagnostics.get("_tendency_run")
    if run is None:
        return jnp.zeros_like(like)
    if name in ("temperature", "specific_humidity"):
        return run[name]
    return run["tracers"].get(name, jnp.zeros_like(like))


def cloud_scheme_inputs(
    state: PhysicsState,
    diagnostics: dict,
    tracers: tuple[str, ...] = ("qc", "qi"),
) -> CloudSchemeInputs:
    """Form the anchor, increments, detrainment and provisional state.

    Args:
        state: The state the cloud term receives (``x_n``).
        diagnostics: The term's diagnostics. Reads ``_dt_seconds``, and
            where present ``_tendency_run`` (the upstream running
            tendency), :data:`CONVECTIVE_DETRAINMENT_KEY` and the carried
            ``_post_physics_state``.
        tracers: The cloud tracers wanted besides temperature and humidity.
            A name absent from ``state.tracers`` is zeros.

    Returns:
        A :class:`CloudSchemeInputs`.

    """
    dt = diagnostics["_dt_seconds"]
    zeros = jnp.zeros_like(state.temperature)
    names = ("temperature", "specific_humidity") + tuple(tracers)

    def received(name):
        if name == "temperature":
            return state.temperature
        if name == "specific_humidity":
            return state.specific_humidity
        return state.tracers.get(name, zeros)

    x_n = {name: received(name) for name in names}

    # The carried post-physics state. The slot's presence is a static
    # property of the composition; its validity is the carried flag, 0 in
    # the construction-time template (first step, a restored checkpoint that
    # predates the slot) and on hosts that never write it (single column,
    # RCE). With the flag at 0 the anchor IS x_n, so the dynamics increment
    # below is x_n - x_n = 0 exactly and carries no gradient into the
    # (zero-filled) slot.
    carry = diagnostics.get(POST_PHYSICS_STATE_KEY)
    if carry is None:
        valid = jnp.zeros((), dtype=state.temperature.dtype)
        anchor = dict(x_n)
    else:
        valid = jnp.asarray(carry["valid"], dtype=state.temperature.dtype)
        use_carry = valid > 0.5

        def carried(name):
            if name in ("temperature", "specific_humidity"):
                return carry[name]
            return carry["tracers"].get(name)

        anchor = {}
        for name in names:
            value = carried(name)
            anchor[name] = (x_n[name] if value is None
                            else jnp.where(use_carry, value, x_n[name]))

    detrainment_rates = diagnostics.get(CONVECTIVE_DETRAINMENT_KEY) or {}
    detrained = {
        name: dt * detrainment_rates.get(name, zeros) for name in _DETRAINED
    }

    increment = {}
    provisional = {}
    for name in names:
        upstream = dt * _running_tendency(diagnostics, name, zeros)
        dynamics = x_n[name] - anchor[name]
        if name in _DETRAINED:
            # The detrained condensate is inside the running tendency (the
            # convection scheme returns it as its qc/qi tendency); take it out
            # so it can be passed on by itself, as ECHAM passes pxtecl/pxteci.
            increment[name] = dynamics + (upstream - detrained[name])
            provisional[name] = anchor[name] + (increment[name] + detrained[name])
        else:
            increment[name] = dynamics + upstream
            provisional[name] = anchor[name] + increment[name]

    def fields(values):
        return CloudFields(
            temperature=values["temperature"],
            specific_humidity=values["specific_humidity"],
            tracers={name: values[name] for name in tracers},
        )

    return CloudSchemeInputs(
        anchor=fields(anchor),
        increment=fields(increment),
        detrained_qc=detrained["qc"],
        detrained_qi=detrained["qi"],
        provisional=fields(provisional),
        dynamics_valid=valid,
    )
