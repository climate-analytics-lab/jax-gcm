"""The cloud-borne aerosol phase: physics-carry fields (CAM's pbuf pattern).

#602: with the cloud-borne cycle closed (item 1) and physics-side
vertical transport in place (item 2), the ``mc_*``/``nc_*`` phase lives
in the cross-step physics carry — never in dycore tracers. The 30-day
controlled A/B (2026-08-13, figures on #602) settled it: Eulerian-spectral
advection rang the episodic mirrors ~90%-of-cells negative (global-mean
mass net-negative) at 2.2x the carry cost, and even against the fair
semi-Lagrangian baseline (quasi-monotone nodal transport, unreleased
dinosaur#135) the carry agreed to 3-8% with r(zonal)=0.997 at 2.1x less
cost. A tracers-storage escape hatch was initially retained for an
FV-dycore re-check and then REMOVED by decision: pySES is the CAM-SE
dycore, CAM itself keeps ``qqcw`` in pbuf rather than advecting it, and
pySES is the configuration most sensitive to tracer count (#595) — there
is no configuration left where advected mirrors could win.

What the carry gives up is resolved-scale advection of in-droplet
aerosol (CAM accepts the same trade); what it keeps is the full explicit
cycle at ~19% over the implicit treatment instead of ~165%.

One semantic nuance of the sequential carry updates: aerosol the
exchange transfers into the carry can be scavenged by wetdep within the
SAME step (a tracer-mode accumulator only exposed it next step) — an
O(dt·rate) strengthening of wet removal.

The access helpers keep per-term code uniform:

* :func:`tracer_view` — read view merging ``state.tracers`` with the
  carry (identity for implicit populations). Reads are SEQUENTIAL within
  the step: each term sees the previous term's update, which bounds every
  removal by current content.
* :func:`apply_updates` — integrates cloud-borne rate updates into the
  carry (``rate·dt``).

:class:`CloudBorneCarryStore` owns the carry slot: it declares the
``initial_carry_state`` seed, guarantees the key exists with identical
structure on every step (including the ``get_empty_data`` probe, which
keeps the scan-carry pytree stable), and applies the turbulent vertical
mixing of the carry with the same TTE-TKE exchange coefficients and
implicit solve ``TracerVerticalDiffusion`` gives the interstitial
tracers — without it, carry storage would be exactly the column-local
frozen reservoir #602 warned about. Carry-resident state is not threaded
by ``PrescribedStateModel`` (#623).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
from flax import nnx

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.vertical_diffusion.tracer_diffusion import (
    TracerDiffusionParameters,
    diffuse_tracers_implicit,
)
from jcm.physics_interface import PhysicsTendency

CARRY_KEY = "_jam_cloud_borne"


def mirror_names(spec: ModalAerosolSpec) -> tuple[str, ...]:
    """Cloud-borne tracer names in canonical (mode, number-then-mass) order."""
    names: list[str] = []
    for mode in spec.modes:
        names.append(number_name(mode.short, cloud_borne=True))
        names.extend(
            mass_name(sp, mode.short, cloud_borne=True)
            for sp in mode.species
        )
    return tuple(names)


def carry_mode(spec: ModalAerosolSpec) -> bool:
    """Whether this population carries an explicit cloud-borne phase."""
    return spec.cloud_borne


def tracer_view(spec, state, diagnostics) -> dict:
    """Read view over interstitial tracers plus the cloud-borne store.

    For an implicit population this is just ``state.tracers``; with an
    explicit phase the carry entries (when present — the probe may run
    before the store term) overlay it. Cloud-borne names are never dycore
    tracers, so the merge cannot shadow a live tracer.
    """
    if not carry_mode(spec):
        return state.tracers
    carry = diagnostics.get(CARRY_KEY)
    if carry is None:
        return state.tracers
    return {**state.tracers, **carry}


def apply_updates(
    spec, diagnostics, updates: dict, dt,
) -> tuple[dict, dict]:
    """Integrate cloud-borne rate updates [.../s] into the carry.

    Returns ``(diagnostics, passthrough)`` where ``passthrough`` is
    always empty for an explicit population (kept in the signature so
    call sites read uniformly) and echoes ``updates`` for an implicit
    one, where no store exists and the caller decides the fate of any
    stray cloud-borne update (there should be none).
    """
    if not carry_mode(spec):
        return diagnostics, updates
    carry = dict(diagnostics.get(CARRY_KEY) or {})
    zeros = None
    for name, rate in updates.items():
        prev = carry.get(name)
        if prev is None:
            zeros = jnp.zeros_like(rate) if zeros is None else zeros
            prev = zeros
        carry[name] = prev + dt * rate
    return {**diagnostics, CARRY_KEY: carry}, {}


class CloudBorneCarryStore(PhysicsTerm):
    """Own the cloud-borne carry: seed it, keep it stable, mix it."""

    name: ClassVar[str] = "jam_cloud_borne_store"
    category: ClassVar[str] = "aerosol_cloud_borne_store"
    requires: ClassVar[tuple[str, ...]] = (
        "air_density", "layer_thickness",
    )
    provides: ClassVar[tuple[str, ...]] = (CARRY_KEY,)

    def __init__(
        self,
        params: TracerDiffusionParameters | None = None,
        *,
        spec: ModalAerosolSpec | None = None,
        vertical_mixing: bool = True,
    ):
        """Hold the population, mixing params, and the mixing gate."""
        self._spec = spec or MAM4_SPEC
        if not carry_mode(self._spec):
            raise ValueError(
                "CloudBorneCarryStore needs a population with an explicit "
                "cloud-borne phase (spec.cloud_borne=True)."
            )
        self._names = mirror_names(self._spec)
        self._vertical_mixing = bool(vertical_mixing)
        self.params = nnx.Param(
            params or TracerDiffusionParameters.default()
        )

    def initial_carry_state(self, coords) -> dict:
        """Zero-seed one carry field per mirror (cold start, spin-up).

        Shapes assume the flattened-columns physics layout
        (``vectorize_columns=True``, the only JAM configuration); a 3D
        composition would need the ``(nlev, nlon, nlat)`` layout instead
        and fails loudly at the first scan step if composed anyway.
        """
        col_shape = (
            coords.horizontal.nodal_shape[0]
            * coords.horizontal.nodal_shape[1],
        )
        nlev = coords.nodal_shape[0]
        return {
            CARRY_KEY: {
                nm: jnp.zeros((nlev,) + col_shape) for nm in self._names
            }
        }

    def __call__(self, state, diagnostics, forcing, terrain):
        params = self.params.get_value()
        zeros = jnp.zeros_like(state.temperature)
        carry = diagnostics.get(CARRY_KEY)
        if carry is None:
            # Structural probe (or a composition that skipped the seed):
            # materialise the zero store so every downstream term — and
            # the scan-carry template — sees the same structure.
            carry = {nm: zeros for nm in self._names}
        else:
            # Fixed name set regardless of what rode in on the carry.
            carry = {nm: carry.get(nm, zeros) for nm in self._names}

        vd = diagnostics.get("vertical_diffusion")
        if self._vertical_mixing and vd is not None:
            dt = diagnostics.get("_dt_seconds", 1800.0)
            q = jnp.stack([carry[nm] for nm in self._names])
            q_new = diffuse_tracers_implicit(
                q,
                params.diffusion_scale * jnp.maximum(vd.kh, 0.0),
                diagnostics["air_density"],
                diagnostics["layer_thickness"],
                dt,
            )
            carry = {nm: q_new[k] for k, nm in enumerate(self._names)}

        tendency = PhysicsTendency.zeros(state.temperature.shape)
        return tendency, {**diagnostics, CARRY_KEY: carry}
