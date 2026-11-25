"""ECHAM-HAM M7 modal microphysics core over ``m7-jax`` (jax-gcm#1017).

Wraps ``m7_jax.model.step_all_soa`` — the complete native ``m7`` call on HAM's
18-component, seven-mode state, validated against the compiled M7 Fortran, in
its cell-minor (structure-of-arrays) form — as a JAM
:class:`ModalMicrophysicsTerm` on :data:`M7_SPEC`. This module is the jcm side
of ECHAM-HAM's ``mo_ham_subm.f90::ham_subm_interface``: it builds M7's inputs
from jcm's state, calls the core and turns the result back into tracer
tendencies and the ``_jam_state`` the harness reads. The design and the
reasons for each mapping are in
``docs/source/design/jam_m7_configuration.md``.

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
  ``c = min(cloud cover, 1 − 1e-10)`` (one float32 epsilon below 1 in float32,
  where HAM's cap rounds to 1) and ``q_s`` over water from ECHAM's
  Sonntag table, clipped to [0, 1] (``ham_subm_interface``). The cloud cover is
  the previous step's (``clouds`` carry), as ECHAM's ``paclc`` is when M7 runs.
* The organic-nucleation PBL test ``jk ≥ int(ppbl)`` uses ECHAM's own PBL-top
  level ``ihpbl`` (``vdiff.f90:737-759``), recomputed here from the dry static
  energy and the dynamic height ``min(z_top, 0.3·u*/|f|)`` with the previous
  step's surface friction velocity, as ECHAM uses the previous step's.
* Forest fraction: ``ForcingData.forest_fraction`` (ECHAM's ``forest``).
* Ion-pair production for ``nucleation_scheme=2`` (Kazil–Lovejoy,
  ion-mediated) comes from :mod:`jam.chemistry.gcr_ionisation`'s
  ``gcr_ion_pair_rate``, fed the per-step solar activity/date carried on
  ``forcing.solar`` (jax-gcm#1017 Kazil/GCR task, Part B) -- the same
  date-derived-quantity pattern every other date-dependent term in this
  codebase uses, not ``DateData`` directly. The PARNUC and GCR tables are
  read only when ``nucleation_scheme=2`` is constructed, so the other schemes
  never need ``HAM_INPUT_DIR``.

Precision & Chunking
--------------------
``enable_x64`` controls the GLOBAL model precision, ``core_dtype`` controls
THIS CORE's working precision (``"float32"`` or ``"float64"``), and
``chunk_size`` optionally tiles grid cells into ``(n_chunks, chunk_size)``
blocks via ``jax.lax.scan`` + ``jax.checkpoint`` through
:meth:`ModalMicrophysicsTerm.configure_core_execution`. Lookup tables are
stored as ``jax_enable_x64``-immune NumPy arrays pre-transposed and cast once
at construction to the core's working NumPy dtype, avoiding per-step 45–200 MB
host/device casts or 4-D transposes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import jcm.constants as c
from jcm.data import ham_inputs
from jcm.physics.aerosol.jam.chemistry import gcr_ionisation
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.microphysics.base import ModalMicrophysicsTerm
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.population import ModalAerosolSpec
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
)
from jcm.physics.coords_util import column_lat_lon
from jcm.physics.thermodynamics import saturation_specific_humidity
from jcm.physics_interface import PhysicsTendency

# m7-jax is the optional ``jcm[m7]`` extra, pinned exactly (pyproject.toml);
# this adapter module is imported only when JAM selects the ``m7_jax`` core
# (lazily, via ``jam_terms``), so a plain jcm import never needs it.
from m7_jax import interface as m7_interface  # noqa: E402
from m7_jax.model import SoAState, step_all_soa  # noqa: E402
from m7_jax.nucleation import (  # noqa: E402
    PreparedKazilLovejoyTable,
    load_kazil_lovejoy_table,
)
from m7_jax.properties import (  # noqa: E402
    PreparedKappaTable,
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
_TINY_M7_VOL = 1.0e-40


def clear_sky_relative_humidity(q, qs, cloud_cover):
    """``ham_subm_interface``'s clear-sky RH: ``max(0,(q − qs·c)/(1 − c))/qs`` in [0, 1].

    The cover is capped at ``1 − 1e-10`` as HAM caps it. In a precision where
    that cap rounds to exactly 1 (float32) the cap is one machine epsilon below
    1 instead: otherwise an overcast cell divides by zero, which is ``0/0`` (a
    NaN that reaches the M7 state) wherever the cell is exactly saturated,
    and a NaN reverse-mode derivative in every overcast cell. Float64 keeps
    HAM's cap bit for bit.
    """
    dtype = jnp.result_type(cloud_cover, q, float)
    cap = 1.0 - max(_ZEPS_CLOUD, float(jnp.finfo(dtype).epsneg))
    cc = jnp.minimum(cloud_cover, jnp.asarray(cap, dtype))
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


_KAZIL_TABLE_NAMES = ham_inputs.KAZIL_TABLE_NAMES


def _ham_input_dir() -> Path:
    """Return the directory holding both ``nucleation_scheme=2`` datasets."""
    return ham_inputs.ham_input_dir()


def _kazil_table_path(directory: Path) -> Path:
    for name in _KAZIL_TABLE_NAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of {_KAZIL_TABLE_NAMES} found in {directory}; "
        "nucleation_scheme=2 needs the Kazil/Lovejoy PARNUC table."
    )


class M7JaxMicrophysics(ModalMicrophysicsTerm):
    """ECHAM-HAM M7 aerosol microphysics (``m7-jax``) on :data:`M7_SPEC`."""

    name: ClassVar[str] = "jam_m7_jax_microphysics"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full", "air_density", "height_full")
    optional_requires: ClassVar[tuple[str, ...]] = ("clouds", "vertical_diffusion")
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
        chunk_size: int | None = None,
        kappa_table: str | os.PathLike | None = None,
    ):
        """Validate the population, switches, precision, and chunking; load lookup tables."""
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
        if nucleation_scheme not in (0, 1, 2) or organic_scheme not in (0, 1, 2):
            raise ValueError("nucleation_scheme must be 0/1/2 and organic_scheme 0/1/2.")

        self.configure_core_execution(
            enable_x64=enable_x64,
            core_dtype=core_dtype,
            chunk_size=chunk_size,
            env_prefix="M7_JAX",
        )
        self._options = dict(
            nucleation_scheme=int(nucleation_scheme),
            organic_scheme=int(organic_scheme),
            coagulation=bool(coagulation),
            condensation=bool(condensation),
        )

        # Store lookup tables as NumPy arrays already cast to the core's working
        # dtype (jax_enable_x64-immune at construction AND free of per-step
        # float64->float32 casts or 4-D transposes).
        np_cdt = np.float32 if self._core_f32 else np.float64
        _table64 = load_kappa_table(kappa_table or _default_kappa_table_path())
        _growth_np, _radius_np, _temp_np, _hum_np, _kap_np = (
            np.asarray(x, dtype=np_cdt) for x in _table64)
        self._kappa_dims = tuple(int(d) for d in _growth_np.shape)
        self._kappa_growth_flat_np = nnx.data(
            np.ascontiguousarray(np.transpose(_growth_np, (1, 2, 0, 3)).reshape(-1))
        )
        _log_r_np = np.log(_radius_np)
        self._kappa_bounds = tuple(
            float(v)
            for v in (
                _log_r_np[0], _log_r_np[-1],
                _temp_np[0], _temp_np[-1],
                _kap_np[0], _kap_np[-1],
                _hum_np[0], _hum_np[1], _hum_np[2],
            )
        )

        self._kazil_table_np = nnx.data(None)
        self._kazil_flat_np = nnx.data(None)
        self._kazil_strides: tuple[int, int, int, int] | None = None
        self._gcr_table_np = nnx.data(None)
        if nucleation_scheme == 2:
            ham_input_dir = _ham_input_dir()
            _kazil64 = load_kazil_lovejoy_table(_kazil_table_path(ham_input_dir))
            self._kazil_table_np = nnx.data(
                tuple(np.asarray(x, dtype=np_cdt) for x in _kazil64)
            )
            # PARNUC's 5-D log-rate grid flattened once, with its row strides,
            # for PreparedKazilLovejoyTable's 1-D corner gathers.
            _pfr_np = self._kazil_table_np[5]
            _, n_rh, n_h, n_i, n_c = _pfr_np.shape
            s_i = int(n_c)
            s_h = int(n_i) * s_i
            s_rh = int(n_h) * s_h
            s_t = int(n_rh) * s_rh
            self._kazil_strides = (s_t, s_rh, s_h, s_i)
            self._kazil_flat_np = nnx.data(np.ascontiguousarray(_pfr_np.reshape(-1)))

            _gcr64 = gcr_ionisation.read_obrien_gcr_ipr(ham_input_dir)
            self._gcr_table_np = nnx.data(
                tuple(np.asarray(x, dtype=np_cdt) for x in _gcr64)
            )

        self._mw_so4_g = self.spec.species_props("so4").molar_mass * 1000.0
        self._mw_h2so4_g = GAS_SPECIES["h2so4"].molar_mass * 1000.0
        self._mass_names = tuple(mass_name(sp, cl) for sp, cl in M7_COMPONENTS)
        self._number_names = tuple(number_name(cl) for cl in M7_CLASSES)
        self._aerosol_names = self._mass_names + self._number_names
        # Per-class (component index, density, κ) for _jam_state's κ and mass.
        self._class_components = tuple(
            tuple((i, self.spec.species_props(sp).density,
                   self.spec.species_props(sp).hygroscopicity)
                  for i, (sp, cl) in enumerate(M7_COMPONENTS) if cl == cls)
            for cls in M7_CLASSES)
        self._coriolis = nnx.data(None)
        self._lat = nnx.data(None)
        self._lon = nnx.data(None)

    def cache_coords(self, coords) -> None:
        """Cache the per-column Coriolis parameter and lat/lon."""
        super().cache_coords(coords)
        lat, lon = column_lat_lon(coords.horizontal)
        self._coriolis = nnx.data(jnp.asarray(2.0 * c.omega * np.sin(np.asarray(lat))))
        self._lat = nnx.data(jnp.asarray(lat))
        self._lon = nnx.data(jnp.asarray(lon))

    def __call__(self, state, diagnostics, forcing, terrain):
        out_dtype = state.temperature.dtype
        with self.core_precision_scope():
            tracer_tends, jam_state = self._step(state, diagnostics, forcing, terrain)
        # Cast core-dtype outputs back to host dtype OUTSIDE the scoped context.
        tracers = self.cast_tree_to_dtype(tracer_tends, out_dtype)
        jam_state = self.cast_tree_to_dtype(jam_state, out_dtype)
        tendency = PhysicsTendency(
            u_wind=jnp.zeros_like(state.u_wind),
            v_wind=jnp.zeros_like(state.v_wind),
            temperature=jnp.zeros_like(state.temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers=tracers,
        )
        return tendency, {**diagnostics, "_jam_state": jam_state}

    def _prepared_kappa_table(self, cdt):
        """Wrap pre-transposed NumPy kappa table arrays in ``PreparedKappaTable``."""
        sg = lambda v: jax.lax.stop_gradient(jnp.asarray(v, dtype=cdt))  # noqa: E731
        n_r, n_t, n_rh, n_k = self._kappa_dims
        b = self._kappa_bounds
        return PreparedKappaTable(
            growth_thrk_flat=sg(self._kappa_growth_flat_np),
            n_r=n_r, n_t=n_t, n_rh=n_rh, n_k=n_k,
            r_log_lo=sg(b[0]), r_log_hi=sg(b[1]),
            t_lo=sg(b[2]), t_hi=sg(b[3]),
            k_lo=sg(b[4]), k_hi=sg(b[5]),
            rh_0=sg(b[6]), rh_1=sg(b[7]), rh_2=sg(b[8]),
        )

    def _prepared_kazil_table(self, cdt):
        """Build the Kazil-Lovejoy PARNUC lookup in the core dtype, pre-flattened."""
        sg = lambda v: jax.lax.stop_gradient(jnp.asarray(v, dtype=cdt))  # noqa: E731
        s_t, s_rh, s_h, s_i = self._kazil_strides
        return PreparedKazilLovejoyTable(
            temperature=sg(self._kazil_table_np[0]),
            relative_humidity=sg(self._kazil_table_np[1]),
            h2so4=sg(self._kazil_table_np[2]),
            ion_pair_rate=sg(self._kazil_table_np[3]),
            condensation_sink=sg(self._kazil_table_np[4]),
            log_pfr_flat=sg(self._kazil_flat_np),
            s_t=s_t, s_rh=s_rh, s_h=s_h, s_i=s_i,
        )

    def _step(self, state, diagnostics, forcing, terrain):
        """Everything from tracer extraction through M7 integration, in core dtype."""
        cdt = self.working_dtype
        shape = state.temperature.shape            # (nlev, *horiz)
        horiz = shape[1:]
        dt = jnp.asarray(diagnostics["_dt_seconds"], cdt)
        rho = jnp.asarray(diagnostics["air_density"], cdt)
        pressure = jnp.asarray(diagnostics["pressure_full"], cdt)
        temperature = jnp.asarray(state.temperature, cdt)
        q = jnp.asarray(state.specific_humidity, cdt)
        zeros = jnp.zeros(shape, cdt)

        fetched = self.fetch_working_tracers(
            state, diagnostics, self._aerosol_names, cdt, cell_shape=shape
        )
        mass_mmr18 = fetched[:18]
        number_mmr7 = fetched[18:]

        run = (diagnostics.get("_tendency_run") or {}).get("tracers", {})
        h2so4_name = gas_name("h2so4")
        raw_h2so4 = state.tracers.get(h2so4_name)
        h2so4_0 = (
            jnp.maximum(jnp.asarray(raw_h2so4, cdt), 0.0)
            if raw_h2so4 is not None else zeros
        )
        raw_h2so4_run = run.get(h2so4_name)
        h2so4_run = jnp.asarray(raw_h2so4_run, cdt) if raw_h2so4_run is not None else zeros
        gas = m7_interface.so4_mixing_ratio_to_native(h2so4_0, rho, self._mw_h2so4_g)
        production = m7_interface.so4_mixing_ratio_to_native(h2so4_run, rho, self._mw_h2so4_g)

        clouds = diagnostics.get("clouds")
        cloud_cover = (jnp.clip(jnp.asarray(clouds.cloud_fraction, cdt), 0.0, 1.0)
                       if clouds is not None else zeros)
        qs = saturation_specific_humidity(temperature, pressure, phase="water")
        rh = clear_sky_relative_humidity(q, qs, cloud_cover)

        forest = getattr(forcing, "forest_fraction", None) if forcing is not None else None
        forest = (jnp.broadcast_to(jnp.asarray(forest, cdt).reshape(horiz), shape)
                  if forest is not None else zeros)
        in_pbl = self._in_pbl(state, diagnostics, terrain, temperature, q, shape, cdt)

        step_kwargs = dict(self._options)
        if self._options["nucleation_scheme"] == 2:
            step_kwargs["kazil_lovejoy_table"] = self._prepared_kazil_table(cdt)
            step_kwargs["ion_pair_rate"] = self._gcr_ion_pair_rate(
                forcing, pressure, temperature, horiz, shape, cdt)

        table = self._prepared_kappa_table(cdt)
        mass18 = m7_interface.aerosol_mass_to_native_soa(mass_mmr18, rho, self._mw_so4_g)
        number7 = m7_interface.number_mixing_ratio_to_native_soa(number_mmr7, rho)
        result = step_all_soa(
            SoAState(mass18, number7, gas),
            temperature, pressure, rh, dt,
            production, cloud_cover, table, forest, in_pbl,
            compute_valid=False,
            compute_water=False,
            chunk_size=self._chunk_size,
            **step_kwargs,
        )
        new_mass18, new_number7, new_gas = result.state
        new_mass_mmr18 = m7_interface.native_to_aerosol_mass_soa(
            new_mass18, rho, self._mw_so4_g
        )
        new_number_mmr7 = m7_interface.native_to_number_mixing_ratio_soa(new_number7, rho)
        new_h2so4 = m7_interface.native_to_so4_mixing_ratio(new_gas, rho, self._mw_h2so4_g)

        tracers = {
            n: (new_mass_mmr18[i] - mass_mmr18[i]) / dt
            for i, n in enumerate(self._mass_names)
        }
        for i, n in enumerate(self._number_names):
            tracers[n] = (new_number_mmr7[i] - number_mmr7[i]) / dt
        tracers[h2so4_name] = (new_h2so4 - h2so4_0) / dt - h2so4_run
        jam_state = self._jam_state_soa(result.properties, new_mass_mmr18, new_number_mmr7)
        return tracers, jam_state

    def _gcr_ion_pair_rate(self, forcing, pressure, temperature, horiz, shape, cdt):
        """GCR ion-pair production rate ``pipr`` [cm-3 s-1] for nucleation_scheme=2."""
        if self._lat is None or self._lon is None:
            raise RuntimeError("M7JaxMicrophysics needs cache_coords (lat/lon for "
                               "the GCR-ionisation geomagnetic coordinate transform).")
        if forcing is None or getattr(forcing, "solar", None) is None:
            raise RuntimeError("nucleation_scheme=2 needs forcing.solar (calendar_year/"
                               "day_of_year/tyear) for the GCR ionisation date dependence.")
        solar = forcing.solar
        lat = jnp.asarray(self._lat, cdt).reshape(horiz)
        lon = jnp.asarray(self._lon, cdt).reshape(horiz)
        year = jnp.asarray(solar.calendar_year, cdt)
        day_of_year = jnp.asarray(solar.day_of_year, cdt)
        activity = gcr_ionisation.solar_activity(year, jnp.asarray(solar.tyear, cdt))
        gcr_table = gcr_ionisation.ObrienGcrTable(
            *(jnp.asarray(x, dtype=cdt) for x in self._gcr_table_np))
        rate = gcr_ionisation.gcr_ion_pair_rate(
            lat, lon, pressure, temperature, activity, gcr_table,
            year=year, day_of_year=day_of_year,
        )
        return jnp.broadcast_to(rate, shape)

    def _in_pbl(self, state, diagnostics, terrain, temperature, q, shape, cdt):
        """Per-cell ``jk >= ihpbl`` with ECHAM's PBL-top level (see :func:`pbl_top_level`)."""
        nlev = shape[0]
        horiz = shape[1:]
        if self._coriolis is None:
            raise RuntimeError("M7JaxMicrophysics needs cache_coords (the Coriolis "
                               "parameter of ECHAM's PBL-top diagnostic).")
        vdiff = diagnostics.get("vertical_diffusion")
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

    def _jam_state_soa(self, props, mass_mmr18, number_mmr7):
        """Assemble ``_jam_state`` directly from cell-minor SoA properties without transposes."""
        wet_m = tuple(r * 1.0e-2 for r in props.wet_radius)
        dry_m = (*(r * 1.0e-2 for r in props.dry_radius), *wet_m[4:])
        rho_m = tuple(d * 1.0e3 for d in props.density)
        kappa_m, mass_m = self.mode_kappa_and_mass(
            [[(mass_mmr18[i], d, k) for i, d, k in comps] for comps in self._class_components],
            tiny_vol=_TINY_M7_VOL,
        )
        return self.build_jam_state(dry_m, wet_m, rho_m, kappa_m, mass_m, number_mmr7)
