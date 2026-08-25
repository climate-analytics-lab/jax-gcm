"""Real MAM4-JAX modal microphysics core (issue #490).

Wraps the MAM4-JAX box model (``reflective-org/MAM4-JAX``) as a JAM
:class:`ModalMicrophysicsTerm`, replacing the κ-Köhler
:class:`PlaceholderMicrophysics` with the actual modal microphysics —
``calcsize`` (size redistribution) → ``wateruptake`` (Köhler water) →
``amicphys`` (gas–aerosol exchange, rename, binary H₂SO₄ nucleation,
coagulation, carbonaceous ageing). The core owns rename and ageing, so the
harness does **not** duplicate them.

Carbonaceous ageing (jax-gcm#721)
---------------------------------
The core's ``mam_pcarbon_aging_1subarea`` port (mam4-jax ≥ the #721 pin,
``mdo_pcarbonaging`` on by default) moves the sulfate-coated fraction of
the primary-carbon mode — number plus pom/bc/mom mass by the
monolayer-criterion fraction, condensed so4/soa wholesale — into the
accumulation mode each step. This is what turns fresh hydrophobic BC/POA
into wet-scavengable CCN (the pcm mode is ``can_activate=False`` by
design), and it also closes the core's pcm repack leak (condensed so4/soa
on pcm has no state slot and was silently dropped). The monolayer
threshold is the ``n_so4_monolayers`` constructor knob (default 3.0 —
the amicphys-path reference value, fed via phys_control in
CAM5/ACME/E3SM; the oft-quoted 8.0 belongs to the legacy
modal_aero_coag aging path; ECHAM-HAM's ``m7_coat`` uses 1.0).

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
dependency (``pip install jcm[mam4]``). It is imported at this module's top, but
this adapter module is itself loaded only when JAM selects the mam4_jax core
(lazily, via ``jam_terms``), so a plain jcm import never pulls in GPL code. The condensation is integrated with the
operator-split ``substep`` / ``astem`` backends (the original adaptive diffrax
solver is not supported — too expensive), both float32-safe FORWARD. The core
precision is selectable per-instance (``core_dtype``): ``"float32"`` runs the
~1M-cell amicphys vmap — the dominant JAM cost — under a *scoped*
``jax.enable_x64(False)`` context while the host model keeps its own precision
(pySES's float64 dynamics untouched; the RRTMGP wrapper's scoped-context
pattern), with boundary casts jcm dtype → core dtype on entry and back on the
tendencies / ``_jam_state``. The default is ``"float64"`` because the float32
core's reverse pass is unusable (non-finite gradients inside ``amicphys`` —
upstream issue); forward-only production drivers opt into float32 for the
speed. ``enable_x64=False`` still runs the whole model float32.

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

import contextlib
import os
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.cloud_borne_store import (
    CARRY_KEY,
    apply_updates,
    carry_mode,
    tracer_view,
)
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

# MAM4-JAX (GPL-3.0) core. Imported at module level — this whole adapter module
# is itself only imported when JAM selects the mam4_jax core (lazily, via
# ``jam_terms``), so a plain jcm import never reaches this GPL dependency. The
# import enables ``jax_enable_x64`` by default; ``__init__`` sets the final
# precision per-instance (see ``enable_x64``). If the ``jcm[mam4]`` extra isn't
# installed this raises ``ImportError`` here, which is the right signal.
import mam4_jax  # noqa: F401
from mam4_jax.core import data
from mam4_jax.coupling import amicphys as _amicphys
from mam4_jax.coupling.amicphys import amicphys
from mam4_jax.physics.calcsize import calcsize
from mam4_jax.physics.wateruptake import wateruptake

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

    def __init__(
        self,
        spec: ModalAerosolSpec | None = None,
        *,
        condensation_backend: str = "substep",
        n_substeps: int = 4,
        enable_x64: bool | None = None,
        core_dtype: str | None = None,
        n_so4_monolayers: float = 3.0,
    ):
        """Import the core, set precision, select the condensation backend.

        The original adaptive ``diffrax`` (Kvaerno5) condensation solve is **not
        supported** here — it is ~40x the cost of the operator-split backends and
        not worth it for this model. Only the two operator-split backends are
        offered:

        * ``"substep"`` (default) — analytic H2SO4 + ``n_substeps``
          frozen-``g_star`` SOA substeps, each the exact closed form of the
          linear sub-ODE. No adaptive while-loop, so no ``max_steps`` fragility
          or worst-cell gating; ``n_substeps`` is speed-insensitive (set it for
          accuracy).
        * ``"astem"`` — the Fortran-faithful adaptive scheme (upstream
          semi-implicit step1/step2 SOA with adaptive ``dtcur = alpha/tmpa``,
          plus the same analytic H2SO4). Use this to match the CAM/E3SM
          reference exactly.

        Both need a mam4_jax with ``configure_condensation``
        (reflective-org/MAM4-JAX#59).

        ``n_so4_monolayers`` sets the carbonaceous-ageing coating
        threshold (see the module docstring; default 3.0, the amicphys
        reference value). Smaller ages faster ⇒ shorter BC/POA
        lifetime. Trace-time static — not a differentiable parameter.

        ``enable_x64`` controls the GLOBAL model precision. Both backends are
        float32-safe (the coag ``qv12`` underflow was fixed upstream), so
        float32 runs the *whole* coupled model in float32 — useful memory
        headroom (the dynamics + 60-tracer spectral transport ~halve their
        traffic). ``None`` (default) reads the ``MAM4_JAX_ENABLE_X64`` env var
        (default ``"1"`` → float64, the safe default); ``True`` / ``False``
        override it. Applied here, at construction, so the dycore state built
        afterwards inherits it.

        ``core_dtype`` controls THIS CORE's precision independently of the
        global flag: ``"float32"`` runs the ~1M-cell amicphys vmap — the
        dominant JAM cost — in float32 under a *scoped*
        ``jax.enable_x64(False)`` context, even when the host model is float64
        (pySES CAM-SE dynamics require global x64; the old global-flag route
        to a float32 core would break them). This is the same scoped-context
        pattern the RRTMGP wrapper uses, and the float32 FORWARD pass is the
        casper-validated configuration (MAM4-JAX #60). ``"float64"``
        (default) keeps the full-precision core. ``None`` reads
        ``MAM4_JAX_CORE_DTYPE`` (default ``"float64"``).

        The default stays float64 because the float32 core's REVERSE pass is
        not usable: gradients through ``amicphys`` come out non-finite in
        float32 (``calcsize``/``wateruptake`` are grad-clean; the failure is
        inside the amicphys sub-processes — upstream issue). Forward-only
        production drivers should pass ``core_dtype="float32"`` for the
        speed; gradient/calibration work must keep float64.
        """
        if spec is not None:
            self.spec = spec

        # Let mam4_jax validate the backend and own its own capability errors.
        _amicphys.configure_condensation(
            backend=condensation_backend, n_substeps=n_substeps,
        )
        self._condensation_backend = str(condensation_backend)
        self._n_substeps = int(n_substeps)

        # Disable the core's hard-coded "other-process" gas production stub
        # (driver.F90:1248, 1e-16 mol/mol/s on H2SO4). jcm supplies its own
        # sulfur via jam_sulfur_gas_chemistry, so the stub is a spurious
        # sulfur source: left on, it creates ~1e-7 kg-S/m²/day per column —
        # ~10× the emitted sulfur globally — and drove the unbounded
        # secondary-aerosol growth of jax-gcm#642 (a previous soft hasattr
        # guard skipped silently on cores that predate the hook, which is
        # how a full corrupted model year shipped). A core without the hook
        # cannot conserve sulfur, so REFUSE it rather than run.
        if not hasattr(_amicphys, "configure_gas_netprod"):
            raise ImportError(
                "The installed mam4-jax has no configure_gas_netprod, so its "
                "hard-coded H2SO4 production stub (1e-16 mol/mol/s, "
                "driver.F90:1248) cannot be disabled and every JAM run "
                "creates sulfur mass without bound (jax-gcm#642). Install "
                "the pinned version: pip install 'jcm[mam4]'."
            )
        _amicphys.configure_gas_netprod(h2so4=0.0, soa=0.0)

        # Carbonaceous ageing (jax-gcm#721). The core's pcarbon-aging
        # transfer (mam_pcarbon_aging_1subarea, on by default upstream) is
        # the ONLY pathway that moves fresh BC/POA out of the
        # non-activatable pcm mode into accum where wet removal reaches it
        # — and, mechanically, the routine that rescues so4/soa condensed
        # onto pcm before the LMAP_AER repack drops it (a per-step sulfur
        # leak otherwise). A core without the hook has neither, so refuse
        # it like the gas-netprod guard above rather than silently run
        # 21-day BC lifetimes with a sulfur sink.
        if not hasattr(_amicphys, "configure_pcarbon_aging"):
            raise ImportError(
                "The installed mam4-jax has no pcarbon aging "
                "(configure_pcarbon_aging): BC/POA would never leave the "
                "primary-carbon mode (jax-gcm#721) and so4/soa condensed "
                "onto it is silently dropped at the state repack. Install "
                "the pinned version: pip install 'jcm[mam4]'."
            )
        # Monolayer threshold: 3.0 is what the MAM4 amicphys path
        # actually receives (via phys_control; the 8.0 in
        # modal_aero_gasaerexch.F90 belongs to the legacy
        # modal_aero_coag path), ECHAM-HAM's counterpart (m7_coat) uses
        # 1.0 — the spread directly sets the BC/POA lifetime, so it is
        # exposed here as a calibration knob. Held on
        # the instance and passed PER CALL to the core (a static jit
        # argument) — NOT installed into the core's process-global config,
        # which is read at trace time and would make several
        # differently-configured instances in one process order-dependent
        # (Codex P1 on #726). Trace-time static, so not differentiable.
        self._n_so4_monolayers = float(n_so4_monolayers)

        # Precision — applied during construction so the dycore state built
        # afterwards (in bootstrap/run) inherits it; toggling it later would
        # leave an f64 state meeting f32 tendencies (mixed-dtype errors).
        if enable_x64 is None:
            want_x64 = os.environ.get("MAM4_JAX_ENABLE_X64", "1") != "0"
        else:
            want_x64 = bool(enable_x64)
        jax.config.update("jax_enable_x64", want_x64)
        self._enable_x64 = want_x64

        if core_dtype is None:
            core_dtype = os.environ.get("MAM4_JAX_CORE_DTYPE", "float64")
        if core_dtype not in ("float32", "float64"):
            raise ValueError(
                f"core_dtype must be 'float32' or 'float64', got {core_dtype!r}"
            )
        # A float64 core is only expressible when x64 is on; a float32 core
        # works under either global setting (scoped ctx is a no-op when x64
        # is already off).
        self._core_f32 = core_dtype == "float32" or not want_x64

        # Precompute static (jcm tracer name -> pcnst index) packings and the
        # per-mode index/property tables used to fill ``_jam_state``. All
        # plain Python / numpy so nnx treats them as static metadata.
        q_pack: list[tuple[str, int]] = []
        qqcw_pack: list[tuple[str, int]] = []
        num_pcnst: list[int] = []
        mode_species: list[list[tuple[int, float, float]]] = []
        # The qqcw side is packed/unpacked only when the population prognoses
        # a cloud-borne phase (#602); without one the core still receives a
        # zero qqcw array (its API needs it) but no tendencies are read back.
        explicit_cb = self.spec.cloud_borne
        for i, mode in enumerate(self.spec.modes):
            q_pack.append((number_name(mode.short), int(data.NUMPTR_AMODE[i])))
            if explicit_cb:
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
                if explicit_cb:
                    qqcw_pack.append(
                        (mass_name(sp, mode.short, cloud_borne=True), mcidx)
                    )
                props = self.spec.species_props(sp)
                sp_list.append((midx, props.density, props.hygroscopicity))
            mode_species.append(sp_list)

        # Gas tracers (H2SO4/SOAG) resolve their pcnst slot from a *different*
        # MAM4 index table (LMAP_GAS) than the aerosol tracers, but once mapped
        # they are packed into ``q`` and read back as tendencies exactly like
        # any other tracer — so they just join ``q_pack``.
        for g in MAM4_GAS:
            q_pack.append((gas_name(g), int(data.LMAP_GAS[_GAS_IGAS[g]])))

        self._q_pack = tuple(q_pack)
        self._qqcw_pack = tuple(qqcw_pack)
        self._num_pcnst = tuple(num_pcnst)
        self._mode_species = tuple(mode_species)
        self._pcnst = int(data.PCNST)
        self._ntot = int(data.NTOT_AMODE)
        self._dgnum = np.asarray(data.DGNUM_AMODE, np.float64)
        self._initialized = True
        if carry_mode(self.spec):
            # In carry mode the store term must run upstream each step
            # (name-set fixing + vertical mixing); requiring its key makes
            # _validate_ordering enforce that, instead of apply_updates
            # silently seeding an unmixed, unmanaged dict.
            self.requires = (*type(self).requires, CARRY_KEY)

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
        # Scoped core precision: with a float32 core under a float64 host
        # (pySES), everything from tracer packing to the amicphys vmap runs
        # inside jax.enable_x64(False) so the core's own dtype-less literals
        # come out float32 too — the RRTMGP-wrapper pattern (commit 27bb36f).
        # No-op when the host already runs float32, or for a float64 core.
        ctx = (jax.enable_x64(False) if self._core_f32
               else contextlib.nullcontext())
        with ctx:
            return self._step(state, diagnostics)

    def _step(self, state, diagnostics):
        cdt = jnp.float32 if self._core_f32 else jnp.float64
        out_dtype = state.temperature.dtype
        shape = state.temperature.shape
        zeros_c = jnp.zeros(shape, cdt)
        dt = jnp.asarray(diagnostics["_dt_seconds"], cdt)

        view = tracer_view(self.spec, state, diagnostics)

        def fetch(name):
            # Floor gas/aerosol tracers at 0. Spectral advection of the JAM
            # tracers leaves small NEGATIVE mass/number on the near-zero field
            # (Gibbs ringing, same root as the optics #543 / ARG #544 floors).
            # A negative aerosol mass makes wateruptake compute a NEGATIVE wet
            # density, and coag's coefficient/scatter math then NaNs on it
            # (single-cell repro: wetdens=-2172 kg/m^3 -> qnum=nan at
            # _mam_coag_1subarea). The core floors qaer/qnum internally but not
            # wetdens, so guard at the boundary where the tracers enter.
            return jnp.maximum(
                jnp.asarray(view.get(name, jnp.zeros(shape)), cdt),
                0.0,
            )

        # Pack jcm tracers into the flat MAM4 arrays (water vapour at slot 0).
        q = jnp.zeros(shape + (self._pcnst,), cdt)
        q = q.at[..., 0].set(jnp.asarray(state.specific_humidity, cdt))
        for name, idx in self._q_pack:
            q = q.at[..., idx].set(fetch(name))
        qqcw = jnp.zeros(shape + (self._pcnst,), cdt)
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
            jnp.asarray(jnp.broadcast_to(vdiff.pbl_height, shape), cdt)
            if vdiff is not None else zeros_c
        )

        core_state = {
            "q": q,
            "qqcw": qqcw,
            "dgncur_a": jnp.broadcast_to(
                jnp.asarray(self._dgnum, cdt), shape + (self._ntot,)
            ),
            "t": jnp.asarray(state.temperature, cdt),
            "pmid": jnp.asarray(diagnostics["pressure_full"], cdt),
            "cldn": zeros_c,
            "zmid": jnp.asarray(diagnostics["height_full"], cdt),
            "pblh": pblh,
            "relhum": jnp.asarray(rh, cdt),
            "deltat": dt,
        }

        # One operator-splitting step (clear-sky, cldn=0). amicphys runs the
        # full physics (mdo_gasaerexch=rename=newnuc=coag=1, its defaults).
        #
        # The step is ``vmap``-ed over a flattened cell axis so each (level,
        # column) point integrates its OWN box-model — the upstream MAM4 box
        # model only ever ran a single cell, and per-cell vmap is the faithful
        # structure for the operator-split substep / astem integrators.
        n_cells = int(np.prod(shape)) if shape else 1

        def to_cells(a):
            return a.reshape((n_cells,) + a.shape[len(shape):])

        flat_state = {
            k: (v if k == "deltat" else to_cells(v))
            for k, v in core_state.items()
        }
        in_axes = ({k: (None if k == "deltat" else 0) for k in flat_state},)
        one_step = lambda s: amicphys(
            wateruptake(calcsize(s)),
            n_so4_monolayers=self._n_so4_monolayers)
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
        cb_updates: dict[str, jnp.ndarray] = {}
        for name, idx in self._qqcw_pack:
            cb_updates[name] = (
                (qqcw_new[..., idx] - qqcw[..., idx]) / dt
            ).astype(out_dtype)
        if carry_mode(self.spec):
            diagnostics, passthrough = apply_updates(
                self.spec, diagnostics,
                cb_updates, jnp.asarray(dt, out_dtype),
            )
            tracer_tends.update(passthrough)
        else:
            tracer_tends.update(cb_updates)

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
