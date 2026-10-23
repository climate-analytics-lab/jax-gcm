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
The core's ``mam_pcarbon_aging_1subarea`` port (mam4-jax ≥ 0.4.0,
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
modal_aero_coag aging path; ECHAM-HAM's ``m7_coat`` uses 1.0). It is a
**differentiable parameter**: an ``nnx.Param`` leaf on the term, passed
to the core per call as a traced ``AmicphysParams`` leaf, so gradients
flow and a calibration sweep reuses one compile.

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

Per-species mass conservation (jax-gcm#778)
--------------------------------------------
Only sulfate and SOA have sources or sinks in the core (gas exchange,
nucleation). Every other pathway — Aitken/accumulation transfer,
renaming, coagulation (including the pcarbon→accumulation ageing pair) —
moves mass as an equal-and-opposite donor/receiver update, so every
other species' column total is invariant to round-off; ``fetch()``'s
floor cannot leak because the floored value is what the core transports.
The per-species conservation test in ``mam4_jax_test.py`` pins this on
both backends with negative-cell and ageing stress.

Organic-mass closure, "substep" backend (jax-gcm#1064)
--------------------------------------------------------
The "substep" condensation backend (the default, differentiable core)
integrates the SOA gas/aerosol exchange ODE in closed form per substep,
which conserves gas + aerosol organic mass only for the *unconstrained*
problem: flooring a mode's aerosol (or the gas) at zero when the raw
update would go negative can only ADD organic mass, never remove it, and
a warm cell with a small SOA reservoir can evaporate past what it holds.
The real fix is a bounded corrector inside the kernel itself
(``mam4_jax.coupling.amicphys``, reflective-org/MAM4-JAX#85, branch
``fix/substep-condensation-organic-closure``): until a mam4-jax release
carrying it is pinned, :meth:`Mam4JaxMicrophysics._close_organic_budget`
applies the equivalent correction at step level (whole
``calcsize -> wateruptake -> amicphys`` call), gated off automatically by
``mam4_jax.__version__`` once the pin moves past the fixed release — see
that method's docstring for the full derivation, and
``docs/source/design/jam_soa_dust.md`` for the measured size. The
"astem" backend never had this defect (its step-2 solve is already
self-consistent) and the closure never runs for it.

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
from flax import nnx
from packaging.version import Version

from jcm.filters import stable_quotient
from jcm.physics.aerosol.jam.cloud_borne_store import (
    CARRY_KEY,
    apply_updates,
    carry_mode,
)
from jcm.physics.aerosol.jam.removal_split import split_view
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

@contextlib.contextmanager
def _preserved_x64():
    """Undo any process-wide ``jax_enable_x64`` flip made inside the block."""
    prior = jax.config.read("jax_enable_x64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


# MAM4-JAX (GPL-3.0) core. Imported at module level — this whole adapter module
# is itself only imported when JAM selects the mam4_jax core (lazily, via
# ``jam_terms``), so a plain jcm import never reaches this GPL dependency. If
# the ``jcm[mam4]`` extra isn't installed this raises ``ImportError`` here,
# which is the right signal.
#
# The import turns ``jax_enable_x64`` on process-wide; it is restored here so
# merely importing the adapter cannot change the dtype of unrelated code
# (issue #729). ``__init__`` sets the precision each instance needs, so the
# import-time flip is redundant.
#
# Public-namespace split (jax-gcm#737): ``mam4_jax/__init__.py`` states that
# only its OWN top-level attributes are covered by semver — a submodule path
# "is internal and DOES move" (v0.3.0 renamed ``mam4_jax.processes.*`` out of
# existence with no deprecation period). As of mam4-jax 0.5.1 the top level
# re-exports ``AmicphysParams``, ``configure_gas_netprod`` and
# ``configure_pcarbon_aging``, so this adapter reads those three off
# ``mam4_jax.<name>`` at their point of use below (kept as attribute lookups,
# not a ``from mam4_jax import ...``, so the ``hasattr`` capability guards
# below still name the actual missing symbol rather than failing at import
# time with a generic ``ImportError``). The five imports here are every
# symbol that is NOT yet re-exported: jcm drives ``calcsize -> wateruptake ->
# amicphys`` as three separate stages, vmapped together over jcm's OWN
# flattened cell axis (not MAM4's), which is why ``run_step``/``run_timesteps``
# don't fit and the per-stage functions plus ``configure_condensation`` (the
# backend/substep selector) and the ``core.data`` index tables
# (``NUMPTR_AMODE`` / ``LMASSPTR_AMODE`` etc., the tracer-packing basis below)
# have to come from their submodules. Requested upstream at
# reflective-org/MAM4-JAX#84; until it lands the exact ``mam4-jax==0.5.1`` pin
# in ``pyproject.toml`` is the mitigation and every bump is deliberate.
with _preserved_x64():
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

# First mam4-jax release expected to carry the upstream organic-mass-closure
# fix for the "substep" SOA-exchange kernel (jax-gcm#1064,
# reflective-org/MAM4-JAX#85, branch fix/substep-condensation-organic-
# closure, commit fece9fb3 on top of the pinned v0.5.1). jcm's step-level
# closure below (``_close_organic_budget``) is a stopgap for every mam4-jax
# release OLDER than this; once a release carrying the fix is pinned in
# ``pyproject.toml``, this constant's comparison retires the closure on its
# own (see ``_organic_closure_active`` below) — no code change needed here,
# only the pin bump and deleting the now-dead closure once it is confirmed
# inactive.
_ORGANIC_CLOSURE_FIXED_IN = "0.5.2"


def relative_humidity(temperature, specific_humidity, pressure):
    """Ambient relative humidity (0–1) for ``amicphys``'s nucleation path.

    ``RH = q / q_sat`` using the shared Tetens saturation thermodynamics; the
    core re-clamps to [0.01, 0.99].

    ``q_sat`` decays continuously toward 0 as ``T`` falls and is already
    below float32's ~1.08e-19 derivative-safety bound at T = 100 K,
    p = 900 hPa -- an ordinary polar-night/model-top state, not an exotic
    one -- so ``d(RH)/dT`` was NaN there (#971). Two prior fixes both
    changed the forward value instead of only the gradient: raising the
    floor on ``q_sat`` (1e-30 -> 1e-12) silently rescaled any ultra-cold
    cell with ``q_sat`` between the two floors (q = 1e-13 at T = 100 K,
    q_sat ~= 6.5e-21, reads back as RH = 0.1 instead of the correct
    clip-to-1); a subsequent double-``where`` short-circuit (RH = 1
    whenever ``q > 0`` and ``q_sat`` below a threshold) in turn forced a
    genuinely *unsaturated* cold cell to RH = 1 (T = 120 K, p = 900 hPa,
    q_sat ~= 4.7e-16, q = 4.7e-17 is 10 % saturated, not 100 %) -- both
    caught in review, PR #1113.

    :func:`jcm.filters.stable_quotient` resolves this properly: its value
    is ``q / max(q_sat, 1e-30)``, bit-identical to the original formula (so
    the forward value never changes, for *any* input), but its reverse-mode
    rule is ``(dq - ratio*d(q_sat)) / q_sat`` rather than the bare
    division's ``-q * q_sat**-2`` -- the latter needs ``q_sat**2``, which
    underflows float32 far above where ``1/q_sat`` itself would; the former
    only needs ``1/q_sat``, finite down to any ``q_sat`` this Tetens fit
    can produce before it underflows to exactly 0 (where the ``maximum``
    floor, as in the original, takes over).
    """
    qs = saturation_specific_humidity(temperature, pressure)
    return jnp.clip(stable_quotient(specific_humidity,
                                    jnp.maximum(qs, 1.0e-30)), 0.0, 1.0)


class Mam4JaxMicrophysics(ModalMicrophysicsTerm):
    """MAM4-JAX modal aerosol microphysics core (interstitial + cloud-borne)."""

    name: ClassVar[str] = "jam_mam4_jax_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full", "height_full")
    # Read in ``_step`` (see its "PBL height" comment): a cross-step read
    # of last step's TTE-TKE boundary-layer height for the nucleation
    # path -- this term (the "core") runs in the pre-cloud aerosol block,
    # upstream of ``TteTkeVerticalDiffusion`` in the default
    # ``echam_physics`` order; absent on the first step -> 0, which only
    # disables the boundary-layer-enhanced nucleation branch.
    optional_requires: ClassVar[tuple[str, ...]] = ("vertical_diffusion",)
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
        lifetime. A differentiable ``nnx.Param`` leaf — the gradient is
        well-defined (piecewise; zero once the mode saturates) and the
        upstream core takes it as a traced pytree field. Must be >= 0
        (validated here for direct construction); optimizer updates
        that wander negative are clamped to 0 in the core — where the
        loss surface is flat (saturated branch, zero gradient) — so
        keep calibration search bounds positive.

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
        if not hasattr(mam4_jax, "configure_gas_netprod"):
            raise ImportError(
                "The installed mam4-jax has no configure_gas_netprod, so its "
                "hard-coded H2SO4 production stub (1e-16 mol/mol/s, "
                "driver.F90:1248) cannot be disabled and every JAM run "
                "creates sulfur mass without bound (jax-gcm#642). Install "
                "the pinned version: pip install 'jcm[mam4]'."
            )
        # configure_gas_netprod is re-exported at the top level as of
        # mam4-jax 0.5.1 (jax-gcm#737) — read it off ``mam4_jax`` rather than
        # the internal ``mam4_jax.coupling.amicphys`` submodule.
        mam4_jax.configure_gas_netprod(h2so4=0.0, soa=0.0)

        # Carbonaceous ageing (jax-gcm#721). The core's pcarbon-aging
        # transfer (mam_pcarbon_aging_1subarea, on by default upstream) is
        # the ONLY pathway that moves fresh BC/POA out of the
        # non-activatable pcm mode into accum where wet removal reaches it
        # — and, mechanically, the routine that rescues so4/soa condensed
        # onto pcm before the LMAP_AER repack drops it (a per-step sulfur
        # leak otherwise). A core without the hook has neither, so refuse
        # it like the gas-netprod guard above rather than silently run
        # 21-day BC lifetimes with a sulfur sink.
        if not hasattr(mam4_jax, "AmicphysParams"):
            raise ImportError(
                "The installed mam4-jax has no pcarbon aging / params API "
                "(AmicphysParams): BC/POA would never leave the "
                "primary-carbon mode (jax-gcm#721) and so4/soa condensed "
                "onto it is silently dropped at the state repack. Install "
                "the pinned version (mam4-jax 0.5.1): pip install 'jcm[mam4]'."
            )
        # AmicphysParams is re-exported at the top level as of mam4-jax 0.5.1
        # (jax-gcm#737) — read it off ``mam4_jax`` rather than the internal
        # ``mam4_jax.coupling.amicphys`` submodule.
        if not {"soa_uptake_mask", "soa_equilibrium_molecular_weight"}.issubset(
                mam4_jax.AmicphysParams._fields):
            raise ImportError(
                "The installed mam4-jax lacks per-call CAM6 SOA configuration. "
                "Its MOM reference allows coarse SOA uptake, whereas CAM6 "
                "SOA occupies the Aitken/accumulation modes. Install the "
                "released mam4-jax 0.5.1: pip install -U '.[mam4]'.")
        # Monolayer threshold: 3.0 is what the MAM4 amicphys path
        # actually receives (via phys_control; the 8.0 in
        # modal_aero_gasaerexch.F90 belongs to the legacy
        # modal_aero_coag path), ECHAM-HAM's counterpart (m7_coat) uses
        # 1.0 — the spread directly sets the BC/POA lifetime, making it
        # a key uncertain parameter. Held as an ``nnx.Param`` LEAF (per
        # the repo's differentiable-parameters convention: numeric
        # tunables must be visible to jax.grad / optimizers) and passed
        # to the core per call as a TRACED AmicphysParams leaf — never
        # via the core's process-global config, which is read at trace
        # time and would make several differently-configured instances
        # order-dependent (Codex P1 on #726). A calibration sweep over
        # it reuses one compile.
        if float(n_so4_monolayers) < 0.0:
            raise ValueError(
                f"n_so4_monolayers must be >= 0, got {n_so4_monolayers} "
                "(0 means a zero-thickness coating requirement = instant "
                "full ageing, NOT off — use the core's mdo_pcarbonaging "
                "toggle to disable ageing)."
            )
        self.n_so4_monolayers = nnx.Param(
            jnp.asarray(float(n_so4_monolayers)))

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
        # pcnst slots carrying the "soa" species token in any mode (jax-gcm#1064
        # closure below): accum/aitken/coarse per mam4_data.py, NOT the
        # primary-carbon mode (its LMAP_AER row has no SOA column — ageing
        # moves pcm's condensed soa/so4 as part of the lumped transfer, not a
        # standalone soa tracer), discovered here rather than hardcoded so it
        # tracks whatever ``self.spec`` actually prognoses.
        soa_mode_idxs: list[int] = []
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
                if sp == "soa":
                    soa_mode_idxs.append(midx)
            mode_species.append(sp_list)

        # Gas tracers (H2SO4/SOAG) resolve their pcnst slot from a *different*
        # MAM4 index table (LMAP_GAS) than the aerosol tracers, but once mapped
        # they are packed into ``q`` and read back as tendencies exactly like
        # any other tracer — so they just join ``q_pack``.
        soag_idx = None
        for g in MAM4_GAS:
            idx = int(data.LMAP_GAS[_GAS_IGAS[g]])
            q_pack.append((gas_name(g), idx))
            if g == "soag":
                soag_idx = idx

        self._q_pack = tuple(q_pack)
        self._qqcw_pack = tuple(qqcw_pack)
        self._num_pcnst = tuple(num_pcnst)
        self._mode_species = tuple(mode_species)
        self._pcnst = int(data.PCNST)
        self._ntot = int(data.NTOT_AMODE)
        self._dgnum = np.asarray(data.DGNUM_AMODE, np.float64)

        # jax-gcm#1064 step-level organic-mass closure (see
        # ``_close_organic_budget`` for the full rationale). Gated OFF
        # whenever the installed mam4_jax already carries the upstream
        # per-substep fix (reflective-org/MAM4-JAX#85) — including for the
        # "astem" backend, which never had this defect (its step-2 solve is
        # self-consistent) — so the closure retires itself the moment the
        # ``pyproject.toml`` pin moves past ``_ORGANIC_CLOSURE_FIXED_IN``,
        # with no code change required here.
        self._soag_pcnst_idx = soag_idx
        self._soa_mode_pcnst_idxs = tuple(soa_mode_idxs)
        # MAM4-JAX's own vmr unit-conversion tables use the gas tracer's
        # literal carbon atomic weight (12.011) but the nominal integer
        # carbon weight (12.0) for the aerosol species — a small, intentional
        # upstream convention (confirmed empirically: this ratio is exactly
        # what equalizes the two slots' units to float64 machine precision
        # for the ASTEM backend, which has no closure defect of its own), so
        # the gas slot must be rescaled onto the aerosol slots' basis before
        # the two are comparable as a single organic-mass total.
        self._organic_gas_to_aer_factor = float(
            data.FCVT_GAS[_GAS_IGAS["soag"]]
            / data.FCVT_AER[int(data.AMICPHYS_IAER_SOA)]
        )
        self._organic_closure_active = (
            self._condensation_backend == "substep"
            and Version(mam4_jax.__version__) < Version(_ORGANIC_CLOSURE_FIXED_IN)
        )
        self._initialized = True
        if carry_mode(self.spec):
            # In carry mode the store term must run upstream each step
            # (name-set fixing + vertical mixing); requiring its key makes
            # _validate_ordering enforce that, instead of apply_updates
            # silently seeding an unmixed, unmanaged dict.
            self.requires = (*type(self).requires, CARRY_KEY)

    def _close_organic_budget(self, q_old, q_new):
        """Step-level organic-mass closure for the ``substep`` backend
        (jax-gcm#1064), active only while ``self._organic_closure_active``.

        The fixed-substep SOA-exchange kernel
        (``mam4_jax.coupling.amicphys._mam_soaexch_1subarea_substep``)
        integrates the gas/aerosol uptake ODE pair in closed form each
        substep, which conserves gas + aerosol organic mass only for the
        *unconstrained* problem, then floors the result (aerosol at zero
        per mode, gas at zero) to keep it physical. For any real ``x``,
        ``max(0, x) - x == max(0, -x) >= 0``, so EVERY floor hit in this
        step — on any mode's aerosol or on the gas — can only ADD organic
        mass, never remove it (confirmed empirically: a cold/condensing
        cell, where no floor engages, closes to float64 machine precision
        with no correction at all; a warm cell whose small SOA reservoir
        evaporates past what it holds manufactures organic mass growing
        with how far past the floor it goes). A warm, gas-poor cell is
        exactly this case: the gas update already collects credit for the
        full (unrealizable) evaporative flux before the aerosol is floored,
        so the floored aerosol never actually supplies it (jax-gcm#1052's
        1.27% excess, condensing direction; jax-gcm#1064's ~14 units/step
        on a March global state, evaporating direction).

        Coagulation and nucleation never touch this budget: coagulation
        (including the pcarbon -> accumulation ageing transfer) moves
        aerosol mass donor/receiver between modes — never creating or
        destroying it in total (the module docstring's #778 per-species
        invariant) — and nucleation forms new sulfate particles from
        H2SO4, not organics. So any drift between the pre-step and
        post-step (soag + sum of every mode's soa) total is attributable
        entirely to this floor.

        Repaying the excess (PR #1170 review, Codex P2): the natural
        donor is the gas, since in the *evaporating* case above it is the
        gas that over-collected. But the gas cannot give back more than
        it has — and in a cell dominated by *condensation* (gas pulled
        down toward a near-zero aggregate equilibrium by one or more
        strongly-absorbing modes) the gas can end this step already at,
        or within round-off of, its own floor, with NO headroom left,
        while a *different* mode's own aerosol floor (the exact
        evaporating-cell mechanism above) still created excess elsewhere
        in the same cell. Taking the whole excess from the gas in that
        situation is a no-op (``max(0, 0 - excess) == 0``) and the budget
        stays open. So the repayment is two-sided: take what the gas can
        actually give first (bounded by its own post-step value), then
        take any remainder from the SOA aerosol mode(s) that received it
        — identified as the modes with a POSITIVE mass increment this
        step, since a mode that was itself floored can only have
        *decreased*, never increased (the same ``max(0,x)-x>=0``
        identity), so a positive increment is exactly the signature of
        "received more than the gas could really supply." The remainder
        is split across those modes in proportion to their own
        increment, each capped at that mode's own post-step mass so the
        correction cannot drive it negative. Any shortfall that remains
        after both the gas and every over-received mode are exhausted is
        left uncorrected rather than forced to zero — a genuine physical
        non-closure (more mass manufactured than this cell has anywhere
        left to take it back from) would show up as a nonzero residual
        in the closure tests rather than being silently hidden.

        Applied at STEP level, covering the whole
        ``calcsize -> wateruptake -> amicphys`` call, not per substep: the
        kernel's internal ``lax.scan`` does not expose the per-substep
        shortfall to this adapter, only the post-step ``q``. The within-
        step difference between a per-substep correction (inside the
        kernel) and this one-shot step-level correction (outside it) is
        that a later substep would see a very slightly lower gas mixing
        ratio — the deficit already returned — which only changes how
        much FURTHER condensation/evaporation that later substep computes:
        a second-order effect on what is already a small correction.

        ``q_old``/``q_new`` are the packed MAM4 state before/after that
        call, in the core's own pcnst-indexed mass-mixing-ratio units.
        SOAG's gas slot is not directly comparable to the SOA aerosol
        slots in those raw units (see ``_organic_gas_to_aer_factor``), so
        the gas is rescaled onto the aerosol slots' basis before comparing
        or correcting them; a correction computed on that basis is
        converted back to the gas's own raw units before being applied.

        Static-shape and side-effect-free (``jnp.maximum``/``jnp.where``,
        no branching on traced values), so this stays reverse-mode
        differentiable like the rest of the step.
        """
        gas_to_aer = self._organic_gas_to_aer_factor
        soa_idxs = self._soa_mode_pcnst_idxs
        g_old = q_old[..., self._soag_pcnst_idx]
        g_new = q_new[..., self._soag_pcnst_idx]
        a_olds = [q_old[..., idx] for idx in soa_idxs]
        a_news = [q_new[..., idx] for idx in soa_idxs]
        total_before = gas_to_aer * g_old + sum(a_olds)
        total_after = gas_to_aer * g_new + sum(a_news)
        # Clip at 0: the floor can only ADD organic mass (see docstring
        # above), so a cell that never hits a floor (the common case) gets
        # an exact no-op here, not a round-off-sized perturbation.
        excess = jnp.maximum(0.0, total_after - total_before)

        # First take what the gas can actually give (on the aerosol-
        # equivalent basis ``excess`` is already in), bounded by what is
        # there -- it can give at most its own post-step value, never
        # driving itself negative.
        gas_available = gas_to_aer * g_new
        taken_from_gas = jnp.minimum(excess, gas_available)
        g_corrected = jnp.maximum(0.0, g_new - taken_from_gas / gas_to_aer)

        # Gas-limited condensation (the case the gas-only correction
        # missed): whatever the gas could not give must come back from
        # the mode(s) that received it -- the ones with a positive mass
        # increment this step, weighted by the size of that increment.
        residual = excess - taken_from_gas
        increments = [jnp.maximum(0.0, a_n - a_o)
                      for a_n, a_o in zip(a_news, a_olds)]
        increment_total = sum(increments)
        # Safe division: when increment_total is 0 every increment_i is
        # also exactly 0 (each is individually >= 0 and they sum to 0),
        # so the numerator is 0 and this is 0/tiny = 0, never 0/0 = nan,
        # for every mode simultaneously -- no further guard is needed to
        # keep this differentiable.
        safe_total = jnp.where(increment_total > 0.0, increment_total, 1.0)
        a_corrected = []
        for a_new_i, increment_i in zip(a_news, increments):
            share = jnp.where(
                increment_total > 0.0, increment_i / safe_total, 0.0,
            )
            # Capped at the mode's own post-step mass: the correction
            # must not drive it negative. Any shortfall this leaves (this
            # mode's share of the residual exceeded what it actually
            # holds) is deliberately NOT redistributed to other modes --
            # it is a genuine non-closure, surfaced as a nonzero residual
            # in the budget tests rather than forced away.
            a_corrected.append(a_new_i - jnp.minimum(share * residual, a_new_i))

        q_new = q_new.at[..., self._soag_pcnst_idx].set(g_corrected)
        for idx, a_corrected_i in zip(soa_idxs, a_corrected):
            q_new = q_new.at[..., idx].set(a_corrected_i)
        return q_new

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
            core_result = self._step(state, diagnostics)
        # Host casts and carry integration must run with host precision
        # restored: requesting float64 inside the scope yields float32.
        return self._assemble_outputs(state, diagnostics, core_result)

    def _step(self, state, diagnostics):
        cdt = jnp.float32 if self._core_f32 else jnp.float64
        shape = state.temperature.shape
        zeros_c = jnp.zeros(shape, cdt)
        dt = jnp.asarray(diagnostics["_dt_seconds"], cdt)

        view = split_view(self.spec, state, diagnostics)

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
        # AmicphysParams is re-exported at the top level as of mam4-jax
        # 0.5.1 (jax-gcm#737).
        core_params = mam4_jax.AmicphysParams(
            n_so4_monolayers=jnp.asarray(
                self.n_so4_monolayers.get_value(), cdt),
            # Jo et al. (2023), section 2.2: CAM6 SOA has fine-mode
            # reservoirs. Keep transient primary-carbon coating, which
            # aging transfers to accumulation, but exclude coarse uptake.
            soa_uptake_mask=jnp.asarray(
                [mode.short != "cor" for mode in self.spec.modes], dtype=bool),
            # CAM6 p0=1e-10 atm corresponds to C*=1.02 ug/m3 with
            # molecular weight 250, rather than the MOM reference's 150.
            soa_equilibrium_molecular_weight=jnp.asarray(250.0, cdt))
        one_step = lambda s: amicphys(
            wateruptake(calcsize(s)), core_params)
        flat_out = jax.vmap(one_step, in_axes=in_axes)(flat_state)

        def from_cells(a):
            return a.reshape(shape + a.shape[1:])

        out = {
            k: from_cells(flat_out[k])
            for k in ("q", "qqcw", "dgncur_a", "dgncur_awet", "wetdens")
        }
        q_new, qqcw_new = out["q"], out["qqcw"]
        if self._organic_closure_active:
            q_new = self._close_organic_budget(q, q_new)
            out = {**out, "q": q_new}

        tracer_tends = {
            name: (q_new[..., idx] - q[..., idx]) / dt
            for name, idx in self._q_pack
        }

        cb_updates = {
            name: (qqcw_new[..., idx] - qqcw[..., idx]) / dt
            for name, idx in self._qqcw_pack
        }

        # Downstream optics/removal need the population AFTER condensation,
        # ageing and coagulation. Repeating calcsize here would advance its
        # number adjustment and mode transfer twice; diagnose only its size
        # relation, then recompute equilibrium water for the updated mass.
        diameters = []
        for i, sp_list in enumerate(self._mode_species):
            volume = sum(q_new[..., midx] / density
                         for midx, density, _ in sp_list)
            number = q_new[..., self._num_pcnst[i]]
            live = (volume > _TINY_VOL) & (number > 0.0)
            v = jnp.where(live, volume, 1.0)
            n = jnp.where(live, number, 1.0)
            mode = self.spec.modes[i]
            diameter = (6.0 * v / (jnp.pi * n)) ** (1.0 / 3.0) * np.exp(
                -1.5 * np.log(mode.geom_std_dev) ** 2)
            diameters.append(jnp.where(
                live, jnp.clip(diameter, mode.dgnum_lo, mode.dgnum_hi),
                mode.dgnum))
        out = wateruptake({**core_state, **out,
                           "dgncur_a": jnp.stack(diameters, axis=-1)})

        jam_state = self._jam_state(
            q_new, out["dgncur_a"], out["dgncur_awet"], out["wetdens"],
            cdt,
        )
        return tracer_tends, cb_updates, dt, jam_state


    def _assemble_outputs(self, state, diagnostics, core_result):
        """Restore host precision before assembling tendencies and carry."""
        tracer_tends, cb_updates, dt, jam_state = core_result
        host_dtype = state.temperature.dtype
        tracer_tends = {
            name: value.astype(host_dtype)
            for name, value in tracer_tends.items()
        }

        cb_updates = {
            name: value.astype(host_dtype)
            for name, value in cb_updates.items()
        }
        dt = jnp.asarray(dt, dtype=host_dtype)

        if carry_mode(self.spec):
            diagnostics, passthrough = apply_updates(
                self.spec, diagnostics, cb_updates, dt,
            )
            tracer_tends.update(passthrough)
        else:
            tracer_tends.update(cb_updates)

        jam_state = jax.tree_util.tree_map(
            lambda value: value.astype(host_dtype),
            jam_state,
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracer_tends,
        )

        return tendency, {**diagnostics, "_jam_state": jam_state}
