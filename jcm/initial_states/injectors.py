"""Initial-condition builders: analytic profiles and warm starts.

Each builder takes a built :class:`~jcm.model.Model` and RETURNS a starting
state (a dycore-native state, or — for ERA5, re-exported from
:mod:`jcm.data.era5` — a :class:`~jcm.physics_interface.PhysicsState`). The
natural call pattern from Python is to hand the returned state straight to
``Model.run``::

    model = Model(coords=coords, terrain=terrain, physics=echam_physics())
    predictions = model.run(
        initial_state=jw_state(model, rh=0.6),
        total_time=10.0, save_interval=1.0,
    )

``Model.run(initial_state=...)`` accepts ``None``, a ``PhysicsState``, or a
dycore-native state (``model.py`` ``bootstrap_state``): ``run`` bootstraps
from the supplied state and integrates it.

These are the library homes of the initial conditions the Hydra CLI exposes
as ``init.kind={jw,balanced_isothermal,era5,from_state}``; ``jcm.runners``
calls straight through to them (#640).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from jcm.model import Model

logger = logging.getLogger(__name__)


# Standard-atmosphere lapse rate and surface temperature for the JW init.
_JW_T_SFC = 288.0       # K, mid-latitude mean surface T
_JW_LAPSE = 6.5e-3      # K/m, ICAO standard tropospheric lapse rate
_JW_T_FLOOR = 250.0     # K, cold-tail cap so semi-implicit reference T stays
                        # close (dycore goes unstable for ΔT ~ 50 K).
# Reference temperature used for the column-mean hydrostatic balance applied
# to surface pressure over orography. ~ midpoint between troposphere and
# stratosphere — exact value matters very little for the surface-pressure
# field, but the nondimensionalisation is sensitive to changes here.
_HYDROSTATIC_T_REF = 260.0

# Tetens / Bolton coefficients for saturation vapour pressure over water.
_ES0 = 611.2     # Pa
_ES_A = 17.67
_ES_B = 29.65    # K offset
_T0_C = 273.15   # K, melting point reference

# Tropopause cap above which we set RH = 0 in the JW humidity profile.
_RH_CAP_PRESSURE_PA = 20000.0   # 200 hPa


def balanced_isothermal_state(model: Model):
    """Build an isothermal-rest atmosphere with orography-balanced ``ps``.

    Same ps-rebalance logic as :func:`jw_state` (so air doesn't end up below
    ground over tall topography), but keeps the temperature field at a uniform
    288 K and humidity at zero. Useful as a robust starting state for
    moist-physics runs over real terrain when the full JW lapse-rate profile
    is unstable at the chosen resolution.

    Returns a dycore-native state; pass it to
    ``model.run(initial_state=...)``.
    """
    from dinosaur.scales import units
    from jcm.constants import grav, p0s1_bg, rd

    # Bootstrap the default rest state so the physics' seeded tracers (cloud
    # water, GHG VMRs, aerosol modes, ...) are already in place; we only
    # override the temperature/ps fields below.
    state = model.initial_state(
        physics_state=None, random_seed=0,
    )
    p0_pa = p0s1_bg

    orog = jnp.asarray(model.terrain.orog)
    if jnp.any(orog > 1.0):
        # Hydrostatic balance with the actual isothermal T (288 K), not
        # ``_HYDROSTATIC_T_REF`` (260 K which is appropriate for the
        # JW lapse-rate profile). Using the matching T avoids an
        # initial-step pressure-temperature inconsistency.
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (rd * _JW_T_SFC))
        scale = float(model.dycore.physics_specs.nondimensionalize(1.0 * units.pascal))
        log_ps_nodal = jnp.log(ps_pa_nodal * scale)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]
        )
    return state


def jw_state(model: Model, rh: float = 0.6):
    """Build a Jablonowski-Williamson-style lapse-rate initial condition.

    Starts from the default isothermal rest atmosphere (so the physics'
    seeded tracers are preserved) and overrides T/ps/q with a vertical
    profile suitable for moist physics:

    * Temperature: 288 K at the surface, ICAO standard lapse 6.5 K/km, capped
      at 250 K so the semi-implicit reference temperature stays close.
    * Surface pressure: hydrostatically balanced against the model's
      orography when present (otherwise the isothermal init places air below
      ground on tall mountains and the run blows up).
    * Humidity: ``rh`` × q_sat(T) below ~200 hPa, zero above; clipped to a
      sensible range for q.

    Returns a dycore-native state; pass it to
    ``model.run(initial_state=...)``.
    """
    from dinosaur.hybrid_coordinates import HybridCoordinates
    from dinosaur.scales import units

    from jcm.constants import grav, p0s1_bg, rd

    # Bootstrap the default rest state; ``initial_state(None)``
    # is what seeds the physics' prognostic tracers, which we preserve below.
    state = model.initial_state(
        physics_state=None, random_seed=0,
    )

    nlon, nlat = model.coords.horizontal.nodal_shape
    p0_pa = p0s1_bg
    if isinstance(model.coords.vertical, HybridCoordinates):
        sigma = jnp.asarray(model.coords.vertical.get_sigma_centers(p0_pa))
    else:
        sigma = jnp.asarray(model.coords.vertical.centers)
    nlev = sigma.size

    # Hypsometric height for an isothermal column at T = 288 K. The scale
    # height H = R_d * T / g comes out to ~ 8400 m; we use it to convert
    # sigma to z so the lapse-rate profile can be evaluated.
    p = sigma * p0_pa
    scale_height = rd * _JW_T_SFC / grav
    z = scale_height * jnp.log(p0_pa / p)
    T_profile = jnp.maximum(_JW_T_SFC - _JW_LAPSE * z, _JW_T_FLOOR)

    # Hydrostatically rebalance surface pressure when there's nontrivial
    # orography, otherwise the isothermal-rest init produces air below ground.
    orog = jnp.asarray(model.terrain.orog)
    if jnp.any(orog > 1.0):
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (rd * _HYDROSTATIC_T_REF))
        scale = float(model.dycore.physics_specs.nondimensionalize(1.0 * units.pascal))
        log_ps_nodal = jnp.log(ps_pa_nodal * scale)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]
        )

    # Humidity: rh * q_sat(T) below the tropopause cap, dry above.
    es = _ES0 * jnp.exp(_ES_A * (T_profile - _T0_C) / (T_profile - _ES_B))
    q_sat = 0.622 * es / jnp.maximum(p - es, 1.0)
    rh_profile = jnp.where(p > _RH_CAP_PRESSURE_PA, rh, 0.0)
    q_profile = jnp.clip(rh_profile * q_sat, 1e-8, 0.03)

    T_ref = jnp.asarray(model.dycore.primitive.reference_temperature)
    # Temperature is the physical air temperature. Dinosaur now receives the
    # physical kg/kg humidity and applies its own virtual-temperature coupling;
    # pre-dividing T by (1 + 0.61 q) would apply that correction twice and hand
    # physics a colder state than this initializer documents.
    T_var_profile = T_profile - T_ref
    T_var_nodal = jnp.broadcast_to(
        T_var_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(state.temperature_variation.dtype)
    state.temperature_variation = model.coords.horizontal.to_modal(T_var_nodal)

    # kg/kg is a dimensionless mass fraction. This is the same conversion as
    # ``physics_state_to_dynamics_state`` and is intentionally distinct from
    # the g/kg conversion retained for non-humidity mass tracers.
    q_nondim = model.dycore.physics_specs.nondimensionalize(
        q_profile * units.dimensionless
    )
    q_dtype = state.tracers["specific_humidity"].dtype
    q_nodal = jnp.broadcast_to(
        q_nondim[:, None, None], (nlev, nlon, nlat)
    ).astype(q_dtype)
    # Preserve the other prognostic tracers (qc, qi, qnc, qni, qr, qs, GHG VMRs,
    # aerosol modes, ...) that ``bootstrap_state`` seeded — only the JW analytic
    # humidity profile is injected here. Overwriting the whole dict would drop
    # the cloud tracers, leaving radiation with zero cloud water (CRE ≡ 0). The
    # in-cloud RRTMGP inflation this persisting cloud water could otherwise
    # cause is bounded by the mo_psrad in-cloud zeroing (mcica.in_cloud_path).
    state.tracers = {
        **state.tracers,
        "specific_humidity": model.coords.horizontal.to_modal(q_nodal),
    }
    return state


def checkpoint_state(model: Model, path: str, *, unstamped_scale=None):
    """Warm-start: load a saved model state as the initial condition.

    ``path`` points at a state written by
    :func:`jcm.checkpoint.save_checkpoint` (e.g. the hosted equilibrated
    states under ``bundles/<grid>_<levels>/init_states/``). Unlike a
    checkpoint *resume*, the recorded elapsed-day count is DISCARDED —
    the clock starts at zero / the model's ``start_time`` — so a hosted
    state skips the ~9-month from-cold spin-up (#638) without inheriting
    the donor run's calendar. The state's pytree must match the composed
    model (grid, levels, physics tracer set); ``load_checkpoint`` fails
    loudly on any mismatch it cannot migrate by field name.

    A donor written before the checkpoint schema stamp (jcm 3.0) is
    refused, because it does not record which unit convention its dycore
    state uses. ``unstamped_scale`` is passed through to
    :func:`jcm.checkpoint.load_checkpoint` as the caller's explicit
    assertion of that convention; see
    ``docs/source/design/checkpoint_compatibility.md``.

    Returns ``(state, physics_carry, donor_days)``:

    * ``state`` — the loaded dycore state with its clock reset;
    * ``physics_carry`` — the donor's restored cross-step physics carry
      (radiation sub-cycle cache, prior-step TKE, …), read back off the
      template after ``load_checkpoint``;
    * ``donor_days`` — the donor state's elapsed sim-days (before the reset,
      for logging).

    Pass both forward as
    ``model.run(initial_state=state, initial_physics_state=physics_carry)`` so
    the warm start inherits the donor's carry fidelity rather than resetting
    it to a fresh carry at the run seam. The carry is structurally the pytree
    :meth:`Model.initial_physics_carry` builds for this model — it was
    just deserialized against exactly that template — so ``run`` threads it
    straight through.

    Uses ``model`` as the deserialization template: ``bootstrap_state`` +
    ``load_checkpoint`` replace ``model.dycore_state`` / ``physics_carry``
    with the donor's contents. The caller immediately
    re-bootstraps from the returned ``state`` (via ``model.run``), which
    rebuilds the template's dycore state; the physics carry we return here is
    what keeps the donor's carry alive across that re-bootstrap.
    """
    from jcm.checkpoint import load_checkpoint

    # bootstrap_state builds both state pytrees, which load_checkpoint
    # needs as deserialization templates (their values are overwritten).
    model.bootstrap_state()
    days = load_checkpoint(model, path, unstamped_scale=unstamped_scale,
                           as_initial_condition=True)
    # Initial-condition mode resets both the exact clock and the dycore's
    # elapsed counter while preserving the donor's physical fields.
    state = model.dycore_state
    # The restored donor carry, to be re-seeded through
    # ``model.run(initial_physics_state=...)``. Without this the warm start's
    # run would rebuild a fresh carry and silently lose the donor's radiation
    # sub-cycle cache / prior-step TKE.
    physics_carry = model.physics_carry
    return state, physics_carry, days
