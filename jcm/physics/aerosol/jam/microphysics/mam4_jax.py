"""Real MAM4-JAX modal microphysics core (issue #490).

Wraps the MAM4-JAX box model (``reflective-org/MAM4-JAX``) as a JAM
:class:`ModalMicrophysicsTerm`, replacing the κ-Köhler
:class:`PlaceholderMicrophysics` with the actual modal microphysics —
``calcsize`` (size redistribution) → ``wateruptake`` (Köhler water) →
``amicphys`` (gas–aerosol exchange, rename, binary H₂SO₄ nucleation,
coagulation). The core owns rename/aging, so the harness does **not**
duplicate them.

Tracer adapter
--------------
jcm carries aerosol as flat ``state.tracers`` keys (``m_/mc_`` interstitial/
cloud-borne mass, ``n_/nc_`` number — see :mod:`..tracer_layout`). MAM4-JAX
works on flat ``q``/``qqcw`` arrays of length ``pcnst=35``. The mapping is the
MAM4 index bookkeeping from ``mam4_jax.data`` (``NUMPTR_AMODE`` /
``LMASSPTR_AMODE`` and their cloud-borne mirrors), precomputed once at
construction. Each step packs jcm tracers into ``q``/``qqcw``, advances one
timestep, and unpacks the change back into per-tracer tendencies
(``Δtracer/Δt``). ``q[..., 0]`` is water-vapour mixing ratio, seeded from
``state.specific_humidity``; ``_jam_state`` is filled from the core's
``dgncur_a``/``dgncur_awet``/``wetdens``.

Licensing & precision
----------------------
MAM4-JAX is **GPL-3.0**; jcm is Apache-2.0. It is therefore an *optional*
dependency (``pip install jcm[mam4]``) imported lazily here, so a plain jcm
import never pulls in GPL code. The core's diffrax SOA-exchange ODE solver
only converges in **float64** (it diverges in float32), so this term enables
``jax_enable_x64`` at construction and runs the core in float64, casting
jcm's float32 tracers to float64 at the boundary and the resulting
tendencies / ``_jam_state`` back to the model dtype. The dynamical core
keeps creating float32 arrays and is unaffected.

Deliberately not coupled yet (follow-ups)
-----------------------------------------
* **Gas-phase tracers** (H₂SO₄, SOAG, SO₂, DMS) are seeded to zero — jcm's
  harness does not prognose them yet. The core's internal hardcoded H₂SO₄
  production (``1e-16`` mol/mol/s) still drives weak nucleation, so this adds
  a tiny non-conservative sulfate source; negligible per step.
* **Cloud-borne activation** stays the harness's job (the core runs clear-sky,
  ``cldn=0``); ``amicphys``'s cloudy sub-area path is not ported upstream.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.gas_species import MAM4_GAS
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
)
from jcm.physics.convection.saturation import saturation_specific_humidity
from jcm.physics_interface import PhysicsTendency

# amicphys ``name_gas`` order (igas): 0 = SOA gas, 1 = H₂SO₄. ``data.LMAP_GAS``
# maps each to its pcnst slot, so jcm's gas tokens resolve to q indices.
_GAS_IGAS: dict[str, int] = {"soag": 0, "h2so4": 1}

# jcm aerosol species token -> MAM4 ``SPECNAME_AMODE`` type index. This is the
# physical correspondence between jcm's canonical tokens and MAM4's species
# types (``mam4_jax.data.SPECNAME_AMODE``); jcm's per-mode species sets were
# built to mirror MAM4's slots exactly, so every (mode, token) resolves.
_TOKEN_TO_TYPE: dict[str, int] = {
    "so4": 0,   # sulfate
    "poa": 3,   # p-organic
    "soa": 4,   # s-organic
    "bc": 5,    # black-c
    "ss": 6,    # seasalt
    "du": 7,    # dust
    "moa": 8,   # m-organic
}

_TINY_VOL = 1.0e-40   # m³/kg — floor for the volume-weighted κ guard.

#: Cached ``(calcsize, wateruptake, amicphys, data)`` from MAM4-JAX. Imported
#: once, lazily, so plain jcm import never touches the GPL dependency.
_CORE = None


def _core():
    """Lazily import MAM4-JAX and (re-)enable float64.

    Importing ``mam4_jax`` enables ``jax_enable_x64`` globally. The core only
    converges in float64 (its implicit diffrax solver diverges otherwise), so
    we keep it on; because jcm builds arrays with ``dtype=float``, the whole
    model then runs in float64 while the core is active. The flag is re-asserted
    on every call (not just the cached first import) because a caller may have
    toggled it off in between — e.g. test teardown.
    """
    global _CORE
    jax.config.update("jax_enable_x64", True)
    if _CORE is None:
        import mam4_jax  # noqa: F401  — also enables jax_enable_x64 on import
        from mam4_jax import data
        from mam4_jax.processes.amicphys import amicphys
        from mam4_jax.processes.calcsize import calcsize
        from mam4_jax.processes.wateruptake import wateruptake
        _CORE = (calcsize, wateruptake, amicphys, data)
    return _CORE


def relative_humidity(temperature, specific_humidity, pressure):
    """Ambient relative humidity (0–1) for ``amicphys``'s nucleation path.

    ``RH = q / q_sat`` using the shared Tetens saturation thermodynamics; the
    core re-clamps to [0.01, 0.99].
    """
    qs = saturation_specific_humidity(temperature, pressure)
    return jnp.clip(specific_humidity / jnp.maximum(qs, 1.0e-30), 0.0, 1.0)


class Mam4JaxMicrophysics(ModalMicrophysicsTerm):
    """MAM4-JAX modal aerosol microphysics core (interstitial + cloud-borne)."""

    name: ClassVar[str] = "jam_mam4_jax_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full", "height_full")
    spec: ClassVar[ModalAerosolSpec] = MAM4_SPEC

    def __init__(self, spec: ModalAerosolSpec | None = None):
        """Import the core, enable float64, and precompute the index maps."""
        if spec is not None:
            self.spec = spec
        _, _, _, data = _core()

        # Precompute static (jcm tracer name -> pcnst index) packings and the
        # per-mode index/property tables used to fill ``_jam_state``. All
        # plain Python / numpy so nnx treats them as static metadata.
        q_pack: list[tuple[str, int]] = []
        qqcw_pack: list[tuple[str, int]] = []
        num_pcnst: list[int] = []
        mode_species: list[list[tuple[int, float, float]]] = []
        for i, mode in enumerate(self.spec.modes):
            q_pack.append((number_name(mode.short), int(data.NUMPTR_AMODE[i])))
            qqcw_pack.append(
                (number_name(mode.short, cloud_borne=True),
                 int(data.NUMPTRCW_AMODE[i]))
            )
            num_pcnst.append(int(data.NUMPTR_AMODE[i]))
            types = tuple(data.LSPECTYPE_AMODE[i])
            sp_list: list[tuple[int, float, float]] = []
            for sp in mode.species:
                slot = types.index(_TOKEN_TO_TYPE[sp])
                midx = int(data.LMASSPTR_AMODE[i][slot])
                mcidx = int(data.LMASSPTRCW_AMODE[i][slot])
                q_pack.append((mass_name(sp, mode.short), midx))
                qqcw_pack.append(
                    (mass_name(sp, mode.short, cloud_borne=True), mcidx)
                )
                props = self.spec.species_props(sp)
                sp_list.append((midx, props.density, props.hygroscopicity))
            mode_species.append(sp_list)

        # Gas tracers fed to the core: g_h2so4/g_soag -> their pcnst slots.
        self._gas_pack = tuple(
            (gas_name(g), int(data.LMAP_GAS[_GAS_IGAS[g]])) for g in MAM4_GAS
        )

        self._q_pack = tuple(q_pack)
        self._qqcw_pack = tuple(qqcw_pack)
        self._num_pcnst = tuple(num_pcnst)
        self._mode_species = tuple(mode_species)
        self._pcnst = int(data.PCNST)
        self._ntot = int(data.NTOT_AMODE)
        self._dgnum = np.asarray(data.DGNUM_AMODE, np.float64)
        self._initialized = True

    def _jam_state(self, q_new, dgncur_a, dgncur_awet, wetdens, out_dtype):
        """Build ``_jam_state`` from the post-step core fields (mode axis 0)."""
        r_dry, r_wet, rho, kappa, mass, number = [], [], [], [], [], []
        for i, sp_list in enumerate(self._mode_species):
            tot_mass = jnp.zeros_like(q_new[..., 0])
            tot_vol = jnp.zeros_like(q_new[..., 0])
            vol_kappa = jnp.zeros_like(q_new[..., 0])
            for midx, dens, hyg in sp_list:
                m = q_new[..., midx]
                v = m / dens
                tot_mass = tot_mass + m
                tot_vol = tot_vol + v
                vol_kappa = vol_kappa + v * hyg
            safe = jnp.maximum(tot_vol, _TINY_VOL)
            r_dry.append(0.5 * dgncur_a[..., i])
            r_wet.append(0.5 * dgncur_awet[..., i])
            rho.append(wetdens[..., i])
            kappa.append(jnp.where(tot_vol > _TINY_VOL, vol_kappa / safe, 0.0))
            mass.append(tot_mass)
            number.append(q_new[..., self._num_pcnst[i]])
        stack = lambda xs: jnp.stack(xs, axis=0).astype(out_dtype)
        return JamAerosolState(
            r_dry=stack(r_dry), r_wet=stack(r_wet), rho=stack(rho),
            kappa=stack(kappa), mass=stack(mass), number=stack(number),
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        calcsize, wateruptake, amicphys, _ = _core()
        out_dtype = state.temperature.dtype
        shape = state.temperature.shape
        zeros64 = jnp.zeros(shape, jnp.float64)
        dt = jnp.asarray(diagnostics["_dt_seconds"], jnp.float64)

        def fetch(name):
            return jnp.asarray(
                state.tracers.get(name, jnp.zeros(shape)), jnp.float64
            )

        # Pack jcm tracers into the flat MAM4 arrays (water vapour at slot 0).
        q = jnp.zeros(shape + (self._pcnst,), jnp.float64)
        q = q.at[..., 0].set(jnp.asarray(state.specific_humidity, jnp.float64))
        for name, idx in self._q_pack:
            q = q.at[..., idx].set(fetch(name))
        # Gas-phase H₂SO₄/SOAG from the sulfur chemistry → MAM4 gas slots; the
        # core condenses/nucleates them (other gas slots stay zero).
        for name, idx in self._gas_pack:
            q = q.at[..., idx].set(fetch(name))
        qqcw = jnp.zeros(shape + (self._pcnst,), jnp.float64)
        for name, idx in self._qqcw_pack:
            qqcw = qqcw.at[..., idx].set(fetch(name))

        rh = relative_humidity(
            state.temperature, state.specific_humidity,
            diagnostics["pressure_full"],
        )

        # PBL height [m] for boundary-layer-enhanced nucleation. TTE-TKE
        # publishes it in the ``vertical_diffusion`` diagnostic, which (like
        # ARG's updraft) is read from the previous step's carry — it runs after
        # the aerosol block. Absent on the first step → 0, so only the
        # free-tropospheric binary nucleation path fires there.
        vdiff = diagnostics.get("vertical_diffusion")
        pblh = (
            jnp.asarray(jnp.broadcast_to(vdiff.pbl_height, shape), jnp.float64)
            if vdiff is not None else zeros64
        )

        core_state = {
            "q": q,
            "qqcw": qqcw,
            "dgncur_a": jnp.broadcast_to(
                jnp.asarray(self._dgnum, jnp.float64), shape + (self._ntot,)
            ),
            "t": jnp.asarray(state.temperature, jnp.float64),
            "pmid": jnp.asarray(diagnostics["pressure_full"], jnp.float64),
            "cldn": zeros64,
            "zmid": jnp.asarray(diagnostics["height_full"], jnp.float64),
            "pblh": pblh,
            "relhum": jnp.asarray(rh, jnp.float64),
            "deltat": dt,
        }

        # One operator-splitting step (clear-sky, cldn=0). amicphys runs the
        # full physics (mdo_gasaerexch=rename=newnuc=coag=1, its defaults).
        #
        # The step is ``vmap``-ed over a flattened cell axis so each (level,
        # column) point solves its OWN box-model ODE. This is essential, not
        # cosmetic: amicphys's gas-exchange uses an *implicit* diffrax solver,
        # and passing the whole grid as one batched state makes that solver
        # form a Jacobian coupled across every cell — its compile cost grows
        # with (n_cells)² and explodes (>80 GB at T21). Per-cell vmap keeps the
        # Jacobian at the single-cell size (the upstream box model only ever
        # ran one cell), collapsing T21 compile to ~1 GB while giving each cell
        # its own adaptive timestep — the physically correct box-model
        # semantics.
        n_cells = int(np.prod(shape)) if shape else 1

        def to_cells(a):
            return a.reshape((n_cells,) + a.shape[len(shape):])

        flat_state = {
            k: (v if k == "deltat" else to_cells(v))
            for k, v in core_state.items()
        }
        in_axes = ({k: (None if k == "deltat" else 0) for k in flat_state},)
        one_step = lambda s: amicphys(wateruptake(calcsize(s)))
        flat_out = jax.vmap(one_step, in_axes=in_axes)(flat_state)

        def from_cells(a):
            return a.reshape(shape + a.shape[1:])

        out = {
            k: from_cells(flat_out[k])
            for k in ("q", "qqcw", "dgncur_a", "dgncur_awet", "wetdens")
        }
        q_new, qqcw_new = out["q"], out["qqcw"]

        tracer_tends: dict[str, jnp.ndarray] = {}
        for name, idx in self._q_pack:
            tracer_tends[name] = (
                (q_new[..., idx] - q[..., idx]) / dt
            ).astype(out_dtype)
        for name, idx in self._qqcw_pack:
            tracer_tends[name] = (
                (qqcw_new[..., idx] - qqcw[..., idx]) / dt
            ).astype(out_dtype)
        # Gas consumed by condensation/nucleation → sink on the gas tracers
        # (the matching aerosol gain flows through the aerosol-slot readback).
        for name, idx in self._gas_pack:
            tracer_tends[name] = (
                (q_new[..., idx] - q[..., idx]) / dt
            ).astype(out_dtype)

        jam_state = self._jam_state(
            q_new, out["dgncur_a"], out["dgncur_awet"], out["wetdens"],
            out_dtype,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )
        return tendency, {**diagnostics, "_jam_state": jam_state}
