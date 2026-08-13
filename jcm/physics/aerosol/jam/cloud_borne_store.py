"""Cloud-borne aerosol storage: dycore tracers or the physics carry.

#602 item 3 (CAM's ``pbuf`` pattern): with the cloud-borne cycle closed
(#602 item 1) and physics-side vertical transport in place (item 2), the
``mc_*``/``nc_*`` mirrors no longer *need* to be dycore-advected tracers —
they can live in the cross-step physics carry, skipping the spectral
transforms and advection entirely (the dominant tracer cost at ne30, #595)
while the exchange/wet/dry/aqueous cycle runs on them unchanged in physics
space. What carry storage gives up is resolved-scale advection of
in-droplet aerosol; CAM accepts exactly that trade for ``qqcw``.

``ModalAerosolSpec.cloud_borne_storage`` selects the mode ("tracers" |
"carry"; only meaningful with an explicit cloud-borne phase). Carry is
the factory default: the 30-day controlled A/B (2026-08-13) measured the
dycore-advected mirrors being rung ~90%-of-cells negative by spectral
advection of these episodic fields, at 2.2x the carry cost, while carry
storage stayed positive-definite and recovered ~85% of the implicit
mode's saving. "tracers" remains reachable (spec field, factory kwarg,
``echam_physics(jam_cloud_borne_storage=...)``) for FV-dycore
re-evaluation, where nothing rings and resolved advection of in-droplet
aerosol could be measured cleanly.

One semantic nuance of the sequential carry updates: aerosol the
exchange transfers into the carry can be scavenged by wetdep within the
SAME step (the tracers mode only exposes it next step) — an O(dt·rate)
difference that slightly strengthens wet removal in carry mode.

The two access helpers keep the per-term diffs small:

* :func:`tracer_view` — a read view merging ``state.tracers`` with the
  carry, so every consumer reads mirrors the same way in both modes. In
  carry mode reads are SEQUENTIAL within the step (each term sees the
  previous term's update), which bounds each removal by the current
  content — stronger positivity than the parallel tracer accumulator.
* :func:`apply_updates` — the write path: in tracers mode the updates are
  handed back for the ordinary tendency accumulator; in carry mode they
  are integrated into the carry immediately (backward-compatible
  ``rate·dt`` semantics, so terms emit rates in both modes).

:class:`CloudBorneCarryStore` owns the carry slot: it declares the
``initial_carry_state`` seed, guarantees the key exists with identical
structure on every step (including the ``get_empty_data`` probe, which
keeps the scan-carry pytree stable), and applies the turbulent vertical
mixing of the carry with the same TTE-TKE exchange coefficients and
implicit solve the tracer-mode mirrors get from
``TracerVerticalDiffusion`` — without it, carry storage would be exactly
the column-local frozen reservoir #602 warned about.
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
    """Whether this population keeps its cloud-borne phase in the carry."""
    return spec.cloud_borne and spec.cloud_borne_storage == "carry"


def tracer_view(spec, state, diagnostics) -> dict:
    """Read view over interstitial tracers plus the cloud-borne store.

    In tracers mode this is just ``state.tracers``; in carry mode the
    carry entries (when present — the probe may run before the store
    term) overlay it. Mirrors are absent from ``state.tracers`` in carry
    mode, so the merge cannot shadow a live tracer.
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
    """Route cloud-borne rate updates [.../s] to the active store.

    Returns ``(diagnostics, passthrough)``: in tracers mode the updates
    come back as ``passthrough`` for the term's ordinary tendency dict;
    in carry mode they are integrated into the carry now (sequential
    semantics) and ``passthrough`` is empty. Callers must use both
    returns.
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
                "CloudBorneCarryStore needs spec.cloud_borne_storage='carry' "
                "(and an explicit cloud-borne phase)."
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
