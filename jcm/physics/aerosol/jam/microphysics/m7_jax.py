"""ECHAM-HAM M7 modal microphysics core over ``m7-jax`` (jax-gcm#1017).

Wraps ``m7_jax.model.step_all`` — the complete native ``m7`` call on HAM's
18-component, seven-mode state, validated against the compiled M7 Fortran —
as a JAM :class:`ModalMicrophysicsTerm` on :data:`M7_SPEC`. This module is the
jcm side of ECHAM-HAM's ``mo_ham_subm.f90::ham_subm_interface``: it builds
M7's inputs from jcm's state, calls the core and turns the result back into
tracer tendencies and the ``_jam_state`` the harness reads. The design and the
reasons for each mapping are in
``docs/source/design/ham_m7_configuration.md``.

Operator split
--------------
ECHAM calls M7 on ``pxtm1 + pxtte·Δt`` — the state with every process that ran
earlier in the step applied — and then overwrites ``pxtte`` so the step ends at
M7's result. jcm sums tendencies computed against the step-start state and
publishes the running sum as ``_tendency_run``. The adapter therefore feeds M7
the aerosol ``x₀ + run·Δt`` (:func:`split_view`), gives it the H2SO4 gas at its
step-start value with the accumulated H2SO4 tendency as the production rate
(ECHAM's ``zgso4m1``/``zdgso4``, ``mo_ham_subm.f90`` the ``immr2molec``
branch for ``isubm_so4g``), and returns ``(x_M7 − (x₀ + run·Δt))/Δt``, so the
summed tendency over the step is exactly ``(x_M7 − x₀)/Δt``.

Units
-----
M7 works in its own units (sulfate molecules cm⁻³, other species µg m⁻³,
number cm⁻³); the conversions are ``m7_jax.interface``'s ports of
``ham_subm_interface``. Sulfur crosses the boundary as molecules: aerosol SO₄
converts with SO₄'s molar mass (96.0631 g/mol, the population's) and the
H2SO4 gas tracer with H2SO4's (jcm's ``g_h2so4`` is kg/kg of H2SO4), so the
gas→particle transfer conserves sulfur atom for atom.

Host diagnostics
----------------
* Relative humidity is the CLEAR-SKY value ``(q − q_s·c)/(1 − c)`` with
  ``c = min(cloud cover, 1 − 1e-10)`` and ``q_s`` over water from ECHAM's
  Sonntag table, clipped to [0, 1] (``ham_subm_interface``). The cloud cover is
  the previous step's (``clouds`` carry), as ECHAM's ``paclc`` is when M7 runs.
* The organic-nucleation PBL test ``jk ≥ int(ppbl)`` uses ECHAM's own PBL-top
  level ``ihpbl`` (``vdiff.f90:737-759``), recomputed here from the dry static
  energy and the dynamic height ``min(z_top, 0.3·u*/|f|)`` with the previous
  step's surface friction velocity, as ECHAM uses the previous step's.
* Forest fraction: ``ForcingData.forest_fraction`` (ECHAM's ``forest``).
* Ion-pair production is not supplied: H2SO4/H2O nucleation runs
  ``nucleation_scheme=1`` (Vehkamäki), and the reference ``nsnucl=2``
  (Kazil–Lovejoy, ion-mediated) is refused until its table and the GCR rates
  are staged (#1017 F5).

Precision
---------
``enable_x64`` controls the GLOBAL model precision, exactly as
``Mam4JaxMicrophysics``: ``None`` (default) reads the ``M7_JAX_ENABLE_X64``
env var (default ``"1"`` -> float64); ``True``/``False`` override it.
Applied here, at construction, so the dycore state built afterwards
inherits it.

``core_dtype`` controls THIS CORE's precision independently of that global
flag (#1017 W1 task 3, forward only -- m7-jax's reverse pass is untested in
float32, so gradient/calibration work must keep ``"float64"``):
``"float32"`` runs ``step_all`` under a *scoped* ``jax.enable_x64(False)``
context -- the same scoped-context pattern ``Mam4JaxMicrophysics`` and the
RRTMGP wrapper use -- with boundary casts jcm dtype -> core dtype on entry
and back on the tendencies / ``_jam_state``; the κ lookup table is cast to
the core dtype inside that same scope (loaded once as float64 numpy at
construction -- a jnp array built once at import/construction time would
freeze its dtype to whichever ``jax_enable_x64`` was live THEN, not track
later per-call scoping, the same trap m7-jax's own ``_SECTION4_MASK`` had).
``None`` (default) reads ``M7_JAX_CORE_DTYPE`` (default ``"float64"``) --
bit-identical to before this precision option existed.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import jcm.constants as c
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.jam_state import JamAerosolState
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.removal_split import split_view
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
)
from jcm.physics.coords_util import column_lat_lon
from jcm.physics.thermodynamics import saturation_specific_humidity
from jcm.physics_interface import PhysicsTendency

# m7-jax (BSD-3) is the optional ``jcm[m7]`` extra; this adapter module is
# imported only when JAM selects the ``m7_jax`` core (lazily, via
# ``jam_terms``), so a plain jcm import never needs it. Unlike mam4_jax,
# importing m7_jax has no jax_enable_x64 side effect, so no _preserved_x64
# wrapper is needed around it.
from m7_jax import interface as m7_interface  # noqa: E402
from m7_jax.model import FullState, step_all  # noqa: E402
from m7_jax.properties import (  # noqa: E402
    KappaTable,
    default_kappa_table_path,
    load_kappa_table,
)

#: HAM ``aerocomp`` order (``mo_ham_m7_trac.f90``): the 18 (species, class)
#: components of the M7 state, the layout ``m7_jax`` expects.
M7_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("so4", "ns"), ("so4", "ks"), ("so4", "as"), ("so4", "cs"),
    ("bc", "ks"), ("bc", "as"), ("bc", "cs"), ("bc", "ki"),
    ("oc", "ks"), ("oc", "as"), ("oc", "cs"), ("oc", "ki"),
    ("ss", "as"), ("ss", "cs"),
    ("du", "as"), ("du", "cs"), ("du", "ai"), ("du", "ci"),
)
#: M7's class order (``inucs`` .. ``icoai``).
M7_CLASSES: tuple[str, ...] = ("ns", "ks", "as", "cs", "ki", "ai", "ci")

# vdiff.f90 PBL-top constants (574, 596).
_ZEPCOR = 5.0e-5
_ZCHNEU = 0.3
# ham_subm_interface's cap on the cloud cover in the clear-sky humidity.
_ZEPS_CLOUD = 1.0e-10


def clear_sky_relative_humidity(q, qs, cloud_cover):
    """``ham_subm_interface``'s clear-sky RH: ``max(0,(q − qs·c)/(1 − c))/qs`` in [0, 1]."""
    cc = jnp.minimum(cloud_cover, 1.0 - _ZEPS_CLOUD)
    q_amb = jnp.maximum(0.0, (q - qs * cc) / (1.0 - cc))
    return jnp.clip(q_amb / qs, 0.0, 1.0)


def pbl_top_level(dry_static_energy, height, ustar, coriolis):
    """ECHAM's PBL-top level ``ihpbl`` (``vdiff.f90:737-759``), 1-based, top-first.

    ``dry_static_energy`` and ``height`` (above the surface) are ``(nlev, *horiz)``
    top-first; ``ustar`` and ``coriolis`` are ``(*horiz)``. Scanning up from the
    level above the lowest, ``ihpblc`` is the first level whose dry static energy
    exceeds the lowest level's and ``ihpbld`` the first whose height reaches the
    dynamic height ``min(z_top, 0.3·u*/max(|f|, 5e-5))``; both default to
    ``klev`` and ``ihpbl = min(ihpblc, ihpbld)``. The first level met scanning
    upward is the LARGEST qualifying index.
    """
    nlev = dry_static_energy.shape[0]
    zcor = jnp.maximum(jnp.abs(coriolis), _ZEPCOR)
    zhdyn = jnp.minimum(height[0], _ZCHNEU * ustar / zcor)
    jk = jnp.arange(1, nlev + 1).reshape((nlev,) + (1,) * (dry_static_energy.ndim - 1))
    above = jk < nlev
    cond_c = above & (dry_static_energy - dry_static_energy[-1] > 0.0)
    cond_d = above & (height - zhdyn >= 0.0)
    ihpblc = jnp.max(jnp.where(cond_c, jk, 0), axis=0)
    ihpbld = jnp.max(jnp.where(cond_d, jk, 0), axis=0)
    ihpblc = jnp.where(ihpblc == 0, nlev, ihpblc)
    ihpbld = jnp.where(ihpbld == 0, nlev, ihpbld)
    return jnp.minimum(ihpblc, ihpbld)


def _default_kappa_table_path() -> Path:
    """``M7_JAX_KAPPA_TABLE`` if set, else the table ``m7-jax`` ships as package data."""
    env = os.environ.get("M7_JAX_KAPPA_TABLE")
    return Path(env) if env else Path(default_kappa_table_path())


class M7JaxMicrophysics(ModalMicrophysicsTerm):
    """ECHAM-HAM M7 aerosol microphysics (``m7-jax``) on :data:`M7_SPEC`."""

    name: ClassVar[str] = "jam_m7_jax_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full", "air_density", "height_full")
    spec: ClassVar[ModalAerosolSpec] = M7_SPEC

    def __init__(
        self,
        spec: ModalAerosolSpec | None = None,
        *,
        nucleation_scheme: int = 1,
        organic_scheme: int = 1,
        coagulation: bool = True,
        condensation: bool = True,
        enable_x64: bool | None = None,
        core_dtype: str | None = None,
        kappa_table: str | os.PathLike | None = None,
    ):
        """Validate the population, the switches and the precision; load the κ table.

        ``nucleation_scheme``/``organic_scheme``/``coagulation``/``condensation``
        are M7's ``nsnucl``/``nonucl``/``lscoag``/``lscond`` (static). The
        reference ``nsnucl=2`` is refused until its data are staged.

        ``enable_x64``/``core_dtype``: see the module docstring's Precision
        section.
        """
        if spec is not None:
            self.spec = spec
        missing = [(sp, cl) for sp, cl in M7_COMPONENTS
                   if cl not in self.spec.mode_shorts
                   or sp not in self.spec.mode(cl).species]
        if missing or self.spec.mode_shorts != M7_CLASSES:
            raise ValueError(
                "M7JaxMicrophysics needs the M7 population (classes "
                f"{M7_CLASSES} carrying HAM's 18 components); missing {missing}.")
        if self.spec.cloud_borne:
            raise ValueError("M7 has no explicit cloud-borne phase; use cloud_borne=False.")
        if nucleation_scheme == 2:
            raise NotImplementedError(
                "nsnucl=2 (Kazil-Lovejoy) needs parnuc.15H2SO4.nc and the GCR ion-pair "
                "tables, which are not staged yet (jax-gcm#1017 F5).")
        if nucleation_scheme not in (0, 1) or organic_scheme not in (0, 1, 2):
            raise ValueError("nucleation_scheme must be 0/1 and organic_scheme 0/1/2.")

        # Precision -- applied here, at construction, so the dycore state
        # built afterwards inherits it; toggling it later would leave an f64
        # state meeting f32 tendencies (mixed-dtype errors). Mirrors
        # Mam4JaxMicrophysics exactly (see the module docstring).
        if enable_x64 is None:
            want_x64 = os.environ.get("M7_JAX_ENABLE_X64", "1") != "0"
        else:
            want_x64 = bool(enable_x64)
        jax.config.update("jax_enable_x64", want_x64)
        self._enable_x64 = want_x64

        if core_dtype is None:
            core_dtype = os.environ.get("M7_JAX_CORE_DTYPE", "float64")
        if core_dtype not in ("float32", "float64"):
            raise ValueError(f"core_dtype must be 'float32' or 'float64', got {core_dtype!r}")
        # A float64 core is only expressible when x64 is on; a float32 core
        # works under either global setting (the scoped context in __call__
        # is a no-op when x64 is already off).
        self._core_f32 = core_dtype == "float32" or not want_x64

        self._options = dict(nucleation_scheme=int(nucleation_scheme),
                             organic_scheme=int(organic_scheme),
                             coagulation=bool(coagulation),
                             condensation=bool(condensation))
        # Loaded once as float64 NUMPY (never a jnp array stored on the
        # instance): a jnp array built here would freeze its dtype to
        # whichever jax_enable_x64 was live at CONSTRUCTION time, not track
        # the scoped float32 context __call__ enters per step (the same
        # trap m7-jax's own module-level _SECTION4_MASK had -- #1017 W1 task
        # 3). Rebuilt at the step's working dtype in _step instead.
        _table64 = load_kappa_table(kappa_table or _default_kappa_table_path())
        # nnx.data: a tuple of (numpy, so jax_enable_x64-immune) arrays is
        # still a pytree of data leaves as far as nnx's static/data check is
        # concerned, same as self._coriolis below.
        self._table_np = nnx.data(tuple(np.asarray(x, dtype=np.float64) for x in _table64))
        self._mw_so4_g = self.spec.species_props("so4").molar_mass * 1000.0
        self._mw_h2so4_g = GAS_SPECIES["h2so4"].molar_mass * 1000.0
        self._mass_names = tuple(mass_name(sp, cl) for sp, cl in M7_COMPONENTS)
        self._number_names = tuple(number_name(cl) for cl in M7_CLASSES)
        # Per-class (component index, density, κ) for _jam_state's κ and mass.
        self._class_components = tuple(
            tuple((i, self.spec.species_props(sp).density,
                   self.spec.species_props(sp).hygroscopicity)
                  for i, (sp, cl) in enumerate(M7_COMPONENTS) if cl == cls)
            for cls in M7_CLASSES)
        self._coriolis = nnx.data(None)

    def cache_coords(self, coords) -> None:
        """Cache the per-column Coriolis parameter for the PBL-top diagnostic."""
        super().cache_coords(coords)
        lat, _ = column_lat_lon(coords.horizontal)
        self._coriolis = nnx.data(jnp.asarray(2.0 * c.omega * np.sin(np.asarray(lat))))

    def __call__(self, state, diagnostics, forcing, terrain):
        # Scoped core precision (#1017 W1 task 3): with a float32 core under
        # a float64 host, everything from tracer packing to step_all runs
        # inside jax.enable_x64(False) so the core's own dtype-less literals
        # come out float32 too -- the same pattern Mam4JaxMicrophysics and
        # the RRTMGP wrapper use. No-op when the host already runs float32,
        # or for a float64 core.
        out_dtype = state.temperature.dtype
        ctx = (jax.enable_x64(False) if self._core_f32
               else contextlib.nullcontext())
        with ctx:
            tracer_tends, jam_state = self._step(state, diagnostics, forcing, terrain)
        # Cast the core-dtype outputs back to the host dtype OUTSIDE the
        # scoped context: `jnp.asarray(x, jnp.float64)` /
        # `x.astype(jnp.float64)` INSIDE a `jax.enable_x64(False)` block
        # silently truncates to float32 (a real JAX property, not specific
        # to this cast) rather than restoring float64 once the array is
        # already float32 core output -- doing the cast here, after `ctx`
        # has exited and the global flag is back to whatever it was, is
        # what actually gets the host its own dtype back. `zeros_like` on an
        # already-float64 state field is unaffected by the scope (it mirrors
        # an existing array's dtype rather than requesting a new one), so
        # the other PhysicsTendency fields need no such care.
        tracers = {name: value.astype(out_dtype) for name, value in tracer_tends.items()}
        jam_state = jax.tree.map(lambda x: x.astype(out_dtype), jam_state)
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracers,
        )
        return tendency, {**diagnostics, "_jam_state": jam_state}

    def _step(self, state, diagnostics, forcing, terrain):
        """Everything from tracer packing through step_all, in core dtype.

        Returns ``(tracer_tends, jam_state)`` still in ``cdt`` -- __call__
        (outside the scoped x64 context) casts both back to the host dtype.
        """
        cdt = jnp.float32 if self._core_f32 else jnp.float64
        shape = state.temperature.shape            # (nlev, *horiz)
        horiz = shape[1:]
        dt = jnp.asarray(diagnostics["_dt_seconds"], cdt)
        rho = jnp.asarray(diagnostics["air_density"], cdt)
        pressure = jnp.asarray(diagnostics["pressure_full"], cdt)
        temperature = jnp.asarray(state.temperature, cdt)
        q = jnp.asarray(state.specific_humidity, cdt)
        zeros = jnp.zeros(shape, cdt)
        # The kappa table must be passed in the core dtype: rebuilt fresh
        # here (not cached as a jnp array on the instance -- see __init__)
        # so it tracks cdt even though self._core_f32 is fixed per instance.
        table = KappaTable(*(jnp.asarray(x, dtype=cdt) for x in self._table_np))

        # Aerosol after every earlier process of the step (ECHAM's pxtm1+pxtte·dt).
        view = split_view(self.spec, state, diagnostics)
        run = (diagnostics.get("_tendency_run") or {}).get("tracers", {})

        def fetch(name):
            return jnp.maximum(jnp.asarray(view.get(name, zeros), cdt), 0.0)

        mass_mmr = jnp.stack([fetch(n) for n in self._mass_names], axis=-1)
        number_mmr = jnp.stack([fetch(n) for n in self._number_names], axis=-1)
        rho_e = rho[..., None]
        mass = jnp.concatenate([
            m7_interface.so4_mixing_ratio_to_native(mass_mmr[..., :4], rho_e, self._mw_so4_g),
            m7_interface.mass_mixing_ratio_to_native(mass_mmr[..., 4:], rho_e),
        ], axis=-1)
        number = m7_interface.number_mixing_ratio_to_native(number_mmr, rho_e)

        h2so4_name = gas_name("h2so4")
        h2so4_0 = jnp.maximum(jnp.asarray(state.tracers.get(h2so4_name, zeros), cdt), 0.0)
        h2so4_run = jnp.asarray(run.get(h2so4_name, zeros), cdt)
        gas = m7_interface.so4_mixing_ratio_to_native(h2so4_0, rho, self._mw_h2so4_g)
        production = m7_interface.so4_mixing_ratio_to_native(h2so4_run, rho, self._mw_h2so4_g)

        clouds = diagnostics.get("clouds")
        cloud_cover = (jnp.clip(jnp.asarray(clouds.cloud_fraction, cdt), 0.0, 1.0)
                       if clouds is not None else zeros)
        # ECHAM's water-only Sonntag table and its 0.5 cap (qsat_from_es).
        qs = saturation_specific_humidity(temperature, pressure, phase="water")
        rh = clear_sky_relative_humidity(q, qs, cloud_cover)

        forest = getattr(forcing, "forest_fraction", None) if forcing is not None else None
        forest = (jnp.broadcast_to(jnp.asarray(forest, cdt).reshape(horiz), shape)
                  if forest is not None else zeros)
        in_pbl = self._in_pbl(state, diagnostics, terrain, temperature, q, shape, cdt)

        result = step_all(
            FullState(mass, number, gas), temperature, pressure, rh, dt,
            production, cloud_cover, table, forest, in_pbl, **self._options)
        new_mass, new_number, new_gas = result.state

        new_mass_mmr = jnp.concatenate([
            m7_interface.native_to_so4_mixing_ratio(new_mass[..., :4], rho_e, self._mw_so4_g),
            m7_interface.native_to_mass_mixing_ratio(new_mass[..., 4:], rho_e),
        ], axis=-1)
        new_number_mmr = m7_interface.native_to_number_mixing_ratio(new_number, rho_e)
        new_h2so4 = m7_interface.native_to_so4_mixing_ratio(new_gas, rho, self._mw_h2so4_g)

        tracers = {}
        for i, n in enumerate(self._mass_names):
            tracers[n] = (new_mass_mmr[..., i] - mass_mmr[..., i]) / dt
        for i, n in enumerate(self._number_names):
            tracers[n] = (new_number_mmr[..., i] - number_mmr[..., i]) / dt
        # Gas: the core integrated the running production itself, so the
        # step's H2SO4 change is (new − x₀); remove what the earlier terms
        # already contribute through the running sum.
        tracers[h2so4_name] = (new_h2so4 - h2so4_0) / dt - h2so4_run

        jam_state = self._jam_state(result, new_mass_mmr, new_number_mmr)
        return tracers, jam_state

    def _in_pbl(self, state, diagnostics, terrain, temperature, q, shape, cdt):
        """Per-cell ``jk >= ihpbl`` with ECHAM's PBL-top level (see :func:`pbl_top_level`)."""
        nlev = shape[0]
        horiz = shape[1:]
        if self._coriolis is None:
            raise RuntimeError("M7JaxMicrophysics needs cache_coords (the Coriolis "
                               "parameter of ECHAM's PBL-top diagnostic).")
        vdiff = diagnostics.get("vertical_diffusion")
        # The previous step's u*, as ECHAM's ustarm. On the first step there is
        # none: u* = 0 makes the dynamic height 0 m, as ECHAM's would be.
        ustar = (jnp.asarray(vdiff.surface_friction_velocity, cdt).reshape(horiz)
                 if vdiff is not None else jnp.zeros(horiz, cdt))
        coriolis = jnp.asarray(self._coriolis, cdt).reshape(horiz)
        geopotential = jnp.asarray(state.geopotential, cdt)
        if terrain is not None and getattr(terrain, "orog", None) is not None:
            geopotential = geopotential - c.grav * jnp.asarray(terrain.orog, cdt).reshape(horiz)
        dse = geopotential + temperature * c.cpd * (1.0 + c.vtmpc2 * q)
        ihpbl = pbl_top_level(dse, geopotential / c.grav, ustar, coriolis)
        jk = jnp.arange(1, nlev + 1).reshape((nlev,) + (1,) * len(horiz))
        return jk >= ihpbl

    def _jam_state(self, result, mass_mmr, number_mmr):
        """``_jam_state`` per class from the post-call state (``ham_subm_interface``'s rdry/rwet/densaer).

        Returned in core dtype; ``__call__`` casts to the host dtype outside
        the scoped x64 context (see its comment).
        """
        props = result.properties
        wet = props.wet_radius * 1.0e-2                       # cm -> m
        dry = jnp.concatenate([props.dry_radius * 1.0e-2, wet[..., 4:]], axis=-1)
        rho_p = props.density * 1.0e3                         # g cm-3 -> kg m-3
        kappa, mass = [], []
        for comps in self._class_components:
            vol = sum(mass_mmr[..., i] / d for i, d, _ in comps)
            vk = sum(mass_mmr[..., i] / d * k for i, d, k in comps)
            kappa.append(jnp.where(vol > 1e-40, vk / jnp.maximum(vol, 1e-40), 0.0))
            mass.append(sum(mass_mmr[..., i] for i, _, _ in comps))
        move = lambda a: jnp.moveaxis(a, -1, 0)  # noqa: E731
        return JamAerosolState(
            r_dry=move(dry), r_wet=move(wet), rho=move(rho_p),
            kappa=jnp.stack(kappa, axis=0),
            mass=jnp.stack(mass, axis=0),
            number=move(number_mmr),
        )
