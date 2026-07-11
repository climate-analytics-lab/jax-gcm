"""pySES CAM-SE implementation of the :class:`DynamicalCore` protocol.

Couples the pyses spectral-element CAM-SE hydrostatic core (float64,
explicit RK3-5STAGE, lumped physics-dynamics coupling) to jcm's gridpoint
physics packages through a pg2 finite-volume physics grid.

Architecture (one physics ``dt``)
---------------------------------
::

    dycore state {"model_state": pyses dict, "sim_time": s}      float64
        │ to_physics_state:  gll_to_fv gather → (nlev, 1, ncol)  → float32
        ▼
    jcm physics (ECHAM / SPEEDY / Held-Suarez, column-local)      float32
        │ PhysicsTendency (nlev, 1, ncol)                         float32
        ▼ step: scatter_3d (fv_to_gll, cast → float64) + DSS
    pyses ``advance_coupling_step`` with
        physics_forcing = {"dynamics": wrap_dynamics(du, dT, 0·d_mass),
                           "tracers":  wrap_tracers(dr, passive, dry=0)}
    (coupling_types.lump_all: forward-Euler add of physics_dt × forcing,
     then subcycled RK3-5STAGE dynamics + tensor hyperviscosity + nu_top
     sponge + vertical remap + tracer advection)                  float64

Precision contract (deliberate, from the developer prototype):
**float64 dynamics / float32 physics.** pyses's jax backend enables
``jax_enable_x64`` process-wide (the explicit SE core needs it); the two
casts live at exactly two seams — ``to_physics_state`` casts the gathered
state *down* to float32, and ``FVPhysicsGrid.scatter_3d`` casts the physics
tendency *up* to float64 before it re-enters the dynamics.

Key configuration decisions inherited from the developer prototype (see the
docstrings of the helpers for the full rationale):

* full 47-level ECHAM/ICON hybrid grid with a finite ~1 Pa top
  (:func:`jcm.dycore.pyses.coords.full_echam_hybrid`);
* resting, dry U.S. Standard Atmosphere 1976 initial state over the real
  orography (:mod:`jcm.dycore.pyses.initial_states`) — the analytic
  baroclinic state goes negative-temperature at the ~1 Pa top;
* native CAM-SE ``nu_top`` Laplacian upper sponge, deepened to ``n_sponge=8``
  (the full L47 column adds ~15 thin levels over the reduced grid pyses's
  ``n_sponge=5`` default was tuned for); at higher resolution lower
  ``nu_top`` (the ne30 runs used 2.5e4) to cut sponge sub-cycling;
* element-local bilinear metric (``init_quasi_uniform_grid_elem_local``) —
  the equiangular analytic metric's inverse is ill-conditioned and
  deprecated in pyses;
* real-geography boundary data bilinearly interpolated onto the physics
  columns *and* the GLL dynamics nodes at build time (host-side numpy; see
  :mod:`jcm.dycore.pyses.interp` for when to interpolate offline instead).

Tracers
-------
CAM-SE is a *dry-mixing-ratio* core: moisture is carried as
``r = q / (1 - q)`` (kg water per kg dry air) in
``tracers["moisture_species"]["water_vapor"]`` and participates in the
pressure/thermodynamics; every tracer declared through the protocol's
``tracer_specs`` (``qc``/``qi``, GHG VMRs, aerosol fields, ...) is carried as
a pyses *passive* tracer — advected and vertically remapped by the dynamics,
forced only by the physics tendency scattered back each step.
``TracerSpec.nondimensionalize`` is a dinosaur-bridge concept (modal
scaling); this backend carries every tracer in its physical units unchanged
on both sides, so the flag is ignored (identity either way).

Model integration
-----------------
``Model`` drives this backend through the protocol only. Construct with a
``dt_seconds`` and pass ``Model(dycore=..., time_step=dycore.dt_seconds/60)``
so the Model's date/forcing bookkeeping agrees with the step the dycore
actually takes.

Historical caveat (fixed): earlier revisions required ``physics_dtype=
jnp.float64`` when
driving the shipped ECHAM physics (through ``Model`` *or* a caller-owned
loop). Two independent reasons, both external to this backend:

* ``Model``'s cross-step physics carry is templated by
  ``Physics.get_empty_data`` (a probe at JAX's default dtype — float64 once
  pyses enables x64) and its forcing is cast to that same default, so the
  ``lax.scan`` carry only type-checks at the process default dtype;
* with x64 enabled, the shipped ECHAM terms are not float32-dtype-stable:
  e.g. the grey/RRTMGP radiation terms hold strong float64 table constants
  that promote *some* (not all) output leaves, so their compute-vs-cached
  ``lax.cond`` cannot type-check against any uniform-dtype carry when fed
  float32 inputs.

The float32 side of the precision split therefore currently covers the
*seam* (float32 ``PhysicsState`` out, float32 ``PhysicsTendency`` accepted
and upcast at ``scatter_3d``) and any dtype-stable physics; running the
full ECHAM stack at float32 needs a physics-side dtype-stability fix
(tracked as an open issue for the ne30 GPU production run, where the f32
physics memory/throughput win matters).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

import jcm.constants as jcm_constants
from jcm.dycore.base import DynamicalCore, Predictions
from jcm.dycore.pyses._pyses import require_pyses
from jcm.dycore.pyses.coords import (
    DEFAULT_MODEL_TOP_FRACTION,
    EchamSECoords,
    full_echam_hybrid,
    make_echam_se_coords,
)
from jcm.dycore.pyses.initial_states import ussa_pressure, ussa_temperature
from jcm.dycore.pyses.interp import interp_grid_to_points
from jcm.dycore.pyses.physics_grid import FVPhysicsGrid
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData, _SSO_NAMES


# Default deepened upper sponge for the finite-top full-L47 column (see the
# module docstring). ``nu_top`` is pyses's default Laplacian viscosity at the
# ne3-ish resolutions the tests run; production ne30 runs used 2.5e4.
DEFAULT_SPONGE_N_LEVELS = 8
DEFAULT_SPONGE_NU_TOP = 2.5e5   # m^2/s

# Internal pyses tracer names that protocol tracer_specs must not collide
# with (water vapour is the prognostic moisture species, not a passive slot).
_RESERVED_TRACER_NAMES = frozenset({"water_vapor", "specific_humidity"})


def cast_to_physics_dtype(tree, dtype=jnp.float32):
    """Cast every floating-point leaf of a pytree to the physics working dtype.

    The direct-drive float32-physics loop (see the module docstring) must
    hand the physics *dtype-consistent* inputs: with pyses's x64 enabled,
    freshly built :class:`~jcm.forcing.ForcingData` and the per-term carry
    slots from ``Physics.initial_carry_state`` come out float64, and terms
    that ``lax.cond`` a computed branch against a cached carry slot (e.g.
    sub-cycled radiation) require the two to have equal dtypes. Apply this
    to the forcing (after ``select``) and to the initial physics carry —
    the developer prototype's ``_to_f32``. Integer / bool leaves pass
    through unchanged.
    """
    import jax

    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        tree,
    )


class PysesCamSEDycore(DynamicalCore):
    """CAM-SE (pySES) dynamical core on the cubed sphere with a pg2 physics grid.

    Args:
        nx: Elements per cube-face edge (``ne``; 3 for tests, 30 for
            production).
        npt: GLL points per element edge (4 = cubic, the CAM-SE standard).
        dt_seconds: Physics coupling step (s). ``None`` uses pyses's
            resolution-based default (``round(900 * 30 / nx)`` to two
            significant figures).
        nlev: Vertical layers from the ECHAM/ICON hybrid tables (47).
        top_fraction: Finite-top replacement factor for the singular
            ``a[0] = 0`` interface (see :func:`full_echam_hybrid`).
        n_sponge: Levels covered by the native CAM-SE ``nu_top`` Laplacian
            upper sponge (0 disables). Deepened default 8 for the thin
            full-L47 top.
        nu_top: Sponge Laplacian viscosity (m^2/s) at the model top; ramped
            by pyses over the ``n_sponge`` levels and sub-cycled at its own
            CFL inside the dynamics.
        terrain_file: Optional path to a jcm-canonical terrain netCDF
            (``orog``/``lsm`` + the six SSO fields on a regular lon/lat
            grid, e.g. ``jcm/data/bc/t63/terrain.nc``). ``None`` builds a
            flat all-ocean aquaplanet.
        tracer_specs: Optional ``name -> TracerSpec`` mapping; ``Model``
            overwrites this at construction with the active physics
            package's declarations.
        physics_dtype: Working dtype of the physics-facing state
            (default float32 — the precision-split contract).

    """

    def __init__(
        self,
        nx: int = 3,
        npt: int = 4,
        dt_seconds: float | None = None,
        *,
        nlev: int = 47,
        top_fraction: float = DEFAULT_MODEL_TOP_FRACTION,
        n_sponge: int = DEFAULT_SPONGE_N_LEVELS,
        nu_top: float = DEFAULT_SPONGE_NU_TOP,
        terrain_file: str | None = None,
        tracer_specs: Mapping[str, Any] | None = None,
        physics_dtype=jnp.float32,
    ):
        """Build the SE grid, vertical grid, configs and boundary data."""
        self._be = require_pyses()
        from pyses.dynamical_cores.hyperviscosity import init_hypervis_config_tensor
        from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
        from pyses.dynamical_cores.model_config import (
            hypervis_opts,
            init_default_config,
        )
        from pyses.dynamical_cores.model_info import models
        from pyses.dynamical_cores.physics_config import init_physics_config
        from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
        from pyses.dynamical_cores.time_step import time_step_options
        from pyses.dynamical_cores.time_stepping import init_timestep_config
        from pyses.mesh_generation.element_local_metric import (
            init_quasi_uniform_grid_elem_local,
        )

        bnp = self._be.np
        self.model = models.cam_se
        self.nx, self.npt = int(nx), int(npt)
        self.n_sponge, self.nu_top = int(n_sponge), float(nu_top)
        # Public: Model reads this to cast its physics carry template so
        # the lax.scan carry matches the per-step compute dtype (float32
        # physics under the float64/x64 dynamics this backend enables).
        self.physics_dtype = physics_dtype
        self._physics_dtype = physics_dtype

        # Element-local bilinear metric: the supported quasi-uniform
        # cubed-sphere grid (the analytic equiangular metric is deprecated —
        # its inverse is ill-conditioned near panel corners).
        self.h_grid, self.dims = init_quasi_uniform_grid_elem_local(
            self.nx, self.npt, calc_smooth_tensor=True,
        )

        # Full nlev-layer finite-top hybrid grid, shared (same numbers) by
        # the float64 dynamics v_grid and the float32 physics coords.
        constants = jcm_constants.physical_constants
        self.p0 = float(constants.p0)
        a_pa, b = full_echam_hybrid(nlev, top_fraction)
        self._a_boundaries_pa, self._b_boundaries = a_pa, b
        self.v_grid = init_vertical_grid(
            bnp.asarray(a_pa / self.p0), bnp.asarray(b), self.p0, self.model,
        )
        self.nlev = int(self.v_grid["hybrid_b_m"].shape[0])

        # Dynamics constants come from jcm.constants (the single source of
        # truth shared with the physics), not pyses's own defaults — the
        # same bridge DinosaurDycore makes via physics_specs_from_constants.
        # epsilon = Rd/Rv is the molecular-weight ratio pyses expects.
        self.physics_config = init_physics_config(
            self.model,
            Rgas=float(constants.rd),
            radius_earth=float(constants.rearth),
            angular_freq_earth=float(constants.omega),
            gravity=float(constants.grav),
            p0=self.p0,
            cp=float(constants.cpd),
            cp_water_vapor=float(constants.cpv),
            R_water_vapor=float(constants.rv),
            epsilon=float(constants.rd) / float(constants.rv),
        )
        _, _default_diffusion, base_tc = init_default_config(
            self.nx, self.h_grid, self.v_grid, self.dims, self.model,
            physics_dt=(-1.0 if dt_seconds is None else float(dt_seconds)),
            hypervis_type=hypervis_opts.variable_resolution,
            physics_config=self.physics_config,
        )
        # Rebuild the diffusion config so the nu_top sponge covers the
        # deeper set of thin top levels (init_default_config hard-codes
        # n_sponge=5, tuned for a reduced-top grid).
        self.diffusion_config = init_hypervis_config_tensor(
            self.h_grid, self.v_grid, self.dims, self.physics_config,
            nu_top=self.nu_top, n_sponge=self.n_sponge,
        )
        self.timestep_config = init_timestep_config(
            base_tc["physics_dt"], self.h_grid, self.physics_config,
            self.diffusion_config, self.dims, self.model,
            dynamics_tstep_type=time_step_options.RK3_5STAGE,
            physics_dynamics_coupling=coupling_types.lump_all,
        )
        self.dt_seconds = float(self.timestep_config["physics_dt"])
        self._gravity = float(self.physics_config["gravity"])

        # pg2 physics grid + the coords adapter physics caches against.
        self.colmap = FVPhysicsGrid(self.h_grid, self.dims, nf=2)
        self.coords: EchamSECoords = make_echam_se_coords(
            self.colmap.latitudes, self.colmap.longitudes, a_pa, b,
            dtype=self._physics_dtype,
        )

        # Boundary conditions (also fills the GLL-side orography caches the
        # initial state integrates over). Model overwrites tracer_specs.
        self.terrain = self.build_terrain(source_file=terrain_file)
        self.tracer_specs = dict(tracer_specs) if tracer_specs else {}

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------

    def build_terrain(self, *, source_file: str | None = None, **kwargs) -> TerrainData:
        """Build :class:`TerrainData` on the physics columns (and cache GLL orography).

        ``source_file=None`` gives a flat, all-ocean aquaplanet. With a
        jcm-canonical terrain netCDF, orography / land-sea mask / the six
        SSO descriptors are bilinearly sampled onto the pg2 physics columns
        (the ``(1, ncol)`` fields physics reads) **and** the orography onto
        the GLL dynamics nodes, whose surface geopotential ``g·orog`` anchors
        the initial state and the hydrostatic geopotential reconstruction.

        Smoothing note: there is no separate spectral truncation here (that
        is a dinosaur-basis concept). The bilinear sample of ~T63 data onto
        an ne3–ne30 SE grid is itself smooth at the model's resolvable
        scales; the SE core's own DSS + tensor hyperviscosity control any
        residual grid-scale forcing. ``phis0`` is therefore exactly
        ``g·orog`` on the columns.

        Side effect (documented): refreshes the internal GLL orography
        caches (``_orog_gll`` / ``_phi_surf_gll`` / ``_orog_col``) *and*
        ``self.terrain``, so a later :meth:`initial_state` is consistent
        with the returned physics terrain.
        """
        if kwargs:
            raise TypeError(
                f"PysesCamSEDycore.build_terrain got unsupported kwargs {sorted(kwargs)}; "
                "orography enveloping/truncation options are dinosaur-backend concepts."
            )
        ncol = self.colmap.num_cols
        gll = np.asarray(self.h_grid["physical_coords"], dtype=np.float64)
        gll_shape = gll.shape[:3]

        if source_file is None:
            zero_col = jnp.zeros((1, ncol))
            terrain = TerrainData(
                orog=zero_col, phis0=zero_col, fmask=zero_col,
                lfluxland=jnp.bool_(False),
                **{name: zero_col for name in _SSO_NAMES},
            )
            self._orog_col = np.zeros(ncol)
            self._orog_gll = self._be.np.zeros(gll_shape)
        else:
            import xarray as xr

            ds = xr.open_dataset(source_file)
            lon = np.asarray(ds["lon"].values)
            lat = np.asarray(ds["lat"].values)

            col_lon = np.degrees(self.colmap.longitudes)
            col_lat = np.degrees(self.colmap.latitudes)

            def sample(name, points_lon, points_lat):
                return interp_grid_to_points(
                    lon, lat, ds[name].transpose("lon", "lat").values,
                    points_lon, points_lat,
                )

            orog_col = np.maximum(sample("orog", col_lon, col_lat), 0.0)
            fmask_col = np.clip(sample("lsm", col_lon, col_lat), 0.0, 1.0)
            sso_col = {}
            for name in _SSO_NAMES:
                vals = sample(name, col_lon, col_lat)
                if name != "orothe":     # orothe is a signed angle
                    vals = np.maximum(vals, 0.0)
                sso_col[name] = vals

            def col2d(arr):
                return jnp.asarray(arr.reshape(1, ncol), dtype=self._physics_dtype)

            terrain = TerrainData(
                orog=col2d(orog_col),
                phis0=col2d(self._gravity * orog_col),
                fmask=col2d(fmask_col),
                lfluxland=jnp.bool_(True),
                **{name: col2d(sso_col[name]) for name in _SSO_NAMES},
            )
            self._orog_col = orog_col

            # Orography on the GLL dynamics nodes for the initial state.
            orog_gll = np.maximum(
                sample("orog",
                       np.degrees(gll[..., 1]).reshape(-1),
                       np.degrees(gll[..., 0]).reshape(-1)),
                0.0,
            ).reshape(gll_shape)
            self._orog_gll = self._be.np.asarray(orog_gll)

        self._phi_surf_gll = self._gravity * self._orog_gll
        self.terrain = terrain
        return terrain

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def initial_state(
        self,
        physics_state: PhysicsState | None,
        *,
        sim_time: float = 0.0,
        random_seed: int = 0,
        tracer_specs: Mapping[str, Any] | None = None,
    ):
        """Build the CAM-SE native state.

        ``physics_state=None`` (the default path): a resting, dry U.S.
        Standard Atmosphere 1976 column over the real orography — the
        surface pressure at each GLL node is the USSA pressure at its
        orographic height, so the state carries a genuine land/sea
        surface-pressure contrast and a positive temperature all the way to
        the ~1 Pa top. ``random_seed`` is unused: the resting state is
        deterministic, and the model spins its circulation up from the
        (longitudinally asymmetric) boundary forcing rather than from a
        seeded perturbation.

        With a gridpoint ``physics_state`` (``(nlev, 1, ncol)`` fields), the
        columns are scattered onto the GLL nodes (FV→GLL + DSS), the given
        surface pressure is distributed hydrostatically through the hybrid
        tables, and each layer's dry mass is recovered as
        ``Δp_moist / (1 + r)`` — the standard CAM approximation for
        ingesting a moist-pressure analysis into the dry-mass core.

        Declared tracers (``tracer_specs``, falling back to
        ``self.tracer_specs``) are seeded at ``spec.initial_value`` on the
        default path or scattered from ``physics_state.tracers`` when
        present there.
        """
        from pyses.dynamical_cores.cam_se.se_state import init_model_struct
        from pyses.dynamical_cores.initialization import init_analytic_state
        from pyses.dynamical_cores.mass_coordinate import surface_mass_to_d_mass

        bnp = self._be.np
        specs = dict(tracer_specs) if tracer_specs is not None else dict(self.tracer_specs)

        if physics_state is None:
            orog_gll = self._orog_gll

            def z_pi_surf_func(lat, lon):
                return orog_gll, ussa_pressure(orog_gll)

            def zeros_of(lat, lon, z):
                return jnp.zeros_like(z)

            model_state = init_analytic_state(
                z_pi_surf_func,
                ussa_pressure,
                lambda lat, lon, z: ussa_temperature(z),   # dry → Tv == T
                zeros_of, zeros_of, zeros_of,              # u = v = q = 0
                self.h_grid, self.v_grid, self.physics_config,
                self.dims, self.model, eps=1e-6,
            )
            # The state is dry *by construction* (Q ≡ 0), but pyses recovers
            # the moisture mixing ratio as Δp_moist/Δp_dry − 1 from two
            # independently bisected pressure columns, leaving an
            # eps-of-the-bisection residual (up to ~1e-4 in the thin top
            # layers). Zero it exactly rather than carry spurious
            # stratospheric "moisture" into the physics.
            moisture = model_state["tracers"]["moisture_species"]
            moisture["water_vapor"] = bnp.zeros_like(moisture["water_vapor"])
            template = moisture["water_vapor"]
            passive = model_state["tracers"]["tracers"]
            for spec in specs.values():
                passive[spec.name] = float(spec.initial_value) * bnp.ones_like(template)
        else:
            cm = self.colmap
            ps_pa = bnp.asarray(
                physics_state.normalized_surface_pressure, dtype=bnp.float64,
            ) * self.p0
            ps_gll = cm.dss(cm.scatter_2d(ps_pa))
            T_gll = cm.dss(cm.scatter_3d(physics_state.temperature))
            u_gll = cm.dss(cm.scatter_3d(physics_state.u_wind))
            v_gll = cm.dss(cm.scatter_3d(physics_state.v_wind))
            q_gll = bnp.clip(cm.dss(cm.scatter_3d(physics_state.specific_humidity)),
                             0.0, 0.5)
            r_gll = q_gll / (1.0 - q_gll)

            dp_moist = surface_mass_to_d_mass(ps_gll, self.v_grid)
            d_mass = dp_moist / (1.0 + r_gll)
            model_state = init_model_struct(
                bnp.stack((u_gll, v_gll), axis=-1), T_gll, d_mass,
                self._phi_surf_gll, {"water_vapor": r_gll}, {},
                self.h_grid, self.dims, self.physics_config, self.model,
            )
            passive = model_state["tracers"]["tracers"]
            template = r_gll
            for spec in specs.values():
                if spec.name in physics_state.tracers:
                    passive[spec.name] = cm.dss(
                        cm.scatter_3d(physics_state.tracers[spec.name]))
                else:
                    passive[spec.name] = float(spec.initial_value) * bnp.ones_like(template)

        return {
            "model_state": model_state,
            "sim_time": jnp.asarray(float(sim_time), dtype=jnp.float64),
        }

    # ------------------------------------------------------------------
    # Gridpoint bridge
    # ------------------------------------------------------------------

    def to_physics_state(self, state) -> PhysicsState:
        """Gather the float64 GLL state onto the pg2 columns as float32.

        Moisture converts dry mixing ratio → specific humidity
        ``q = r / (1 + r)`` (with ``r`` floored at 0 — FV averaging of a
        near-zero field can leave tiny negatives); every declared tracer is
        gathered as-is (non-negativity clamping is ``verify_state``'s job at
        the physics boundary). The surface pressure handed to physics is the
        full **moist** pressure (dry column mass + water), normalised by
        ``p0``, so the physics' hybrid pressure reconstruction matches the
        dynamics' actual column mass.
        """
        from pyses.dynamical_cores.cam_se.thermodynamics import eval_sum_species
        from pyses.dynamical_cores.mass_coordinate import d_mass_to_surface_mass

        bnp = self._be.np
        cm = self.colmap
        dtype = self._physics_dtype

        def f32_3d(x):
            return cm.gather_3d(x).astype(dtype)

        ms = state["model_state"]
        dyn = ms["dynamics"]
        u = f32_3d(dyn["horizontal_wind"][..., 0])
        v = f32_3d(dyn["horizontal_wind"][..., 1])
        T = f32_3d(dyn["T"])

        moisture = ms["tracers"]["moisture_species"]
        r_pos = bnp.maximum(f32_3d(moisture["water_vapor"]), dtype(0.0))
        q = r_pos / (1.0 + r_pos)

        passive = ms["tracers"]["tracers"]
        tracers = {name: f32_3d(passive[name]) for name in self.tracer_specs}

        d_pressure = eval_sum_species(moisture) * dyn["d_mass"]
        ps = d_mass_to_surface_mass(d_pressure, self.v_grid)
        nsp = (cm.gather_2d(ps) / self.p0).astype(dtype)

        phi = cm.gather_3d(self._geopotential(ms)).astype(dtype)
        return PhysicsState(
            u_wind=u, v_wind=v, temperature=T,
            specific_humidity=q, geopotential=phi,
            normalized_surface_pressure=nsp, tracers=tracers,
        )

    def _geopotential(self, model_state):
        """Hydrostatic mid-level geopotential on the GLL nodes over real orography.

        Rebuilt diagnostically each step from the state (CAM-SE hydrostatic
        carries no prognostic ``phi``): virtual temperature from the moist /
        dry-air species, moist pressure from ``d_mass``, integrated upward
        from ``phi_surf = g·orog``.
        """
        from pyses.dynamical_cores.cam_se.thermodynamics import (
            eval_Rgas_dry,
            eval_balanced_geopotential,
            eval_interface_pressure,
            eval_midlevel_pressure,
            eval_sum_species,
            eval_virtual_temperature,
        )

        dyn = model_state["dynamics"]
        tracers = model_state["tracers"]
        moisture = tracers["moisture_species"]

        R_dry = eval_Rgas_dry(tracers["dry_air_species"], self.physics_config)
        total_mixing_ratio = eval_sum_species(moisture)
        T_v = eval_virtual_temperature(
            dyn["T"], moisture, total_mixing_ratio, R_dry, self.physics_config,
        )
        ptop = self.v_grid["hybrid_a_i"][0] * self.v_grid["reference_surface_mass"]
        d_pressure = total_mixing_ratio * dyn["d_mass"]
        p_mid = eval_midlevel_pressure(eval_interface_pressure(d_pressure, ptop))
        return eval_balanced_geopotential(
            T_v, d_pressure, p_mid, R_dry, self._phi_surf_gll,
        )

    def _forcing_from_tendency(self, tend: PhysicsTendency, model_state):
        """Scatter a gridpoint :class:`PhysicsTendency` into a pyses forcing struct.

        Each column tendency is scattered FV→GLL (float64 from here on) and
        DSS-projected to C0 continuity — an element-discontinuous forcing
        would inject spurious boundary modes into the explicit RK stages.
        The moisture tendency converts specific humidity to dry mixing
        ratio via the chain rule ``dr/dt = (dq/dt) / (1 - q)²`` evaluated at
        the *current* GLL moisture. Dry-air layer mass and the dry-air
        species receive zero forcing (physics never creates or destroys dry
        air). Tracers with a ~zero physics tendency (e.g. GHG VMRs no term
        writes) are thereby genuinely passive.
        """
        from pyses.dynamical_cores.model_state import wrap_dynamics, wrap_tracers

        bnp = self._be.np
        cm = self.colmap
        dyn = model_state["dynamics"]
        ts = model_state["tracers"]

        du = cm.dss(cm.scatter_3d(tend.u_wind))
        dv = cm.dss(cm.scatter_3d(tend.v_wind))
        dT = cm.dss(cm.scatter_3d(tend.temperature))
        dyn_forcing = wrap_dynamics(
            bnp.stack((du, dv), axis=-1), dT,
            bnp.zeros(dyn["d_mass"].shape), self.model,
        )

        r_old = ts["moisture_species"]["water_vapor"]
        r_pos = bnp.maximum(r_old, 0.0)
        q_now = r_pos / (1.0 + r_pos)
        dr_gll = cm.scatter_3d(tend.specific_humidity) \
            / bnp.maximum(1.0 - q_now, 1e-6) ** 2
        moist_forcing = {"water_vapor": cm.dss(dr_gll)}

        passive_forcing = {
            name: cm.dss(cm.scatter_3d(tend.tracers[name]))
            for name in self.tracer_specs
        }
        zero_dry = {k: bnp.zeros(v.shape) for k, v in ts["dry_air_species"].items()}
        tracer_forcing = wrap_tracers(
            moist_forcing, passive_forcing, self.model, dry_air_species=zero_dry,
        )
        return {"dynamics": dyn_forcing, "tracers": tracer_forcing}

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, state, physics_tendency: PhysicsTendency | None):
        """One physics ``dt``: lumped forcing add + subcycled CAM-SE dynamics.

        Delegates to pyses's jitted ``advance_coupling_step`` (the same
        entry point pyses's own ``init_simulator`` production loop uses):
        with ``coupling_types.lump_all`` it forward-Euler-adds
        ``physics_dt × forcing`` to dynamics and tracers, then runs the
        CFL-subcycled RK3-5STAGE dynamics with tensor hyperviscosity, the
        ``nu_top`` sponge, conservative vertical remap (Zerroukat) and
        consistent tracer advection. Pure-JAX and jit-able end to end;
        ``physics_tendency=None`` takes an unforced (adiabatic) step.
        """
        from pyses.dynamical_cores.run_dycore import advance_coupling_step

        ms = state["model_state"]
        forcing = (
            self._forcing_from_tendency(physics_tendency, ms)
            if physics_tendency is not None else None
        )
        ms_next = advance_coupling_step(
            ms, self.h_grid, self.v_grid, self.physics_config,
            self.diffusion_config, self.timestep_config, self.dims, self.model,
            physics_forcing=forcing,
        )
        return {
            "model_state": ms_next,
            "sim_time": state["sim_time"] + self.dt_seconds,
        }

    # ------------------------------------------------------------------
    # Sim-time accounting
    # ------------------------------------------------------------------

    def sim_time(self, state) -> jnp.ndarray:
        return state["sim_time"]

    def with_sim_time(self, state, sim_time):
        return {
            "model_state": state["model_state"],
            "sim_time": jnp.asarray(sim_time, dtype=jnp.float64),
        }

    # ------------------------------------------------------------------
    # Tracer compatibility
    # ------------------------------------------------------------------

    def required_tracers_ok(self, specs: Sequence[Any]) -> None:
        """CAM-SE carries any scalar tracer passively; reject only name clashes.

        Raises:
            ValueError: if a spec collides with the core's internal moisture
                species name, or the same name is declared twice.

        """
        seen = set()
        for spec in specs:
            if spec.name in _RESERVED_TRACER_NAMES:
                raise ValueError(
                    f"Tracer name {spec.name!r} is reserved by the pyses CAM-SE "
                    "backend (moisture is carried internally as the "
                    "'water_vapor' dry mixing ratio, exposed to physics as "
                    "specific_humidity). Rename the physics tracer."
                )
            if spec.name in seen:
                raise ValueError(f"Duplicate tracer spec {spec.name!r}.")
            seen.add(spec.name)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _regrid_targets(self):
        """Lazily build the column → regular lat/lon bin-average regrid.

        Choice (documented per the deliverable): **bin average** of the pg2
        cell values over a regular lat/lon target grid sized to roughly one
        column per box (``nlat = round(sqrt(ncol / 2))``, ``nlon = 2·nlat``),
        with empty boxes filled from their nearest column (great-circle /
        chord distance). Bin-averaging is monotone, cheap, sparse and
        faithful to what the model resolves; it deliberately trades
        smoothness for not inventing extrema, and empty-box fill keeps the
        output gap-free at coarse ne. Host-side numpy, computed once.
        """
        if getattr(self, "_regrid_cache", None) is not None:
            return self._regrid_cache

        ncol = self.colmap.num_cols
        nlat = max(4, int(round(np.sqrt(ncol / 2.0))))
        nlon = 2 * nlat
        lat_edges = np.linspace(-90.0, 90.0, nlat + 1)
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_centers = (np.arange(nlon) + 0.5) * (360.0 / nlon)

        col_lat = np.degrees(self.colmap.latitudes)
        col_lon = np.degrees(self.colmap.longitudes) % 360.0
        ilat = np.clip(np.searchsorted(lat_edges, col_lat, side="right") - 1,
                       0, nlat - 1)
        ilon = np.clip((col_lon / (360.0 / nlon)).astype(int), 0, nlon - 1)
        bin_idx = ilat * nlon + ilon                       # (ncol,)
        counts = np.bincount(bin_idx, minlength=nlat * nlon)

        # Nearest column (chord distance) for every empty target box.
        empty = np.nonzero(counts == 0)[0]
        fill_src = np.zeros(empty.shape, dtype=int)
        if empty.size:
            tgt_lat = np.radians(lat_centers[empty // nlon])
            tgt_lon = np.radians(lon_centers[empty % nlon])
            tgt_xyz = np.stack([np.cos(tgt_lat) * np.cos(tgt_lon),
                                np.cos(tgt_lat) * np.sin(tgt_lon),
                                np.sin(tgt_lat)], axis=-1)
            c_lat = self.colmap.latitudes
            c_lon = self.colmap.longitudes
            col_xyz = np.stack([np.cos(c_lat) * np.cos(c_lon),
                                np.cos(c_lat) * np.sin(c_lon),
                                np.sin(c_lat)], axis=-1)
            fill_src = np.argmax(tgt_xyz @ col_xyz.T, axis=1)

        self._regrid_cache = dict(
            nlat=nlat, nlon=nlon,
            lat_centers=lat_centers, lon_centers=lon_centers,
            bin_idx=bin_idx, counts=np.maximum(counts, 1),
            empty=empty, fill_src=fill_src,
        )
        return self._regrid_cache

    def _regrid_columns(self, values: np.ndarray) -> np.ndarray:
        """``(..., ncol)`` column data → ``(..., nlon, nlat)`` lat/lon boxes."""
        rg = self._regrid_targets()
        lead = values.shape[:-1]
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
        out = np.zeros((flat.shape[0], rg["nlat"] * rg["nlon"]))
        np.add.at(out, (slice(None), rg["bin_idx"]), flat)
        out /= rg["counts"][None, :]
        if rg["empty"].size:
            out[:, rg["empty"]] = flat[:, rg["fill_src"]]
        # (lead, nlat, nlon) -> jcm-canonical (lead, nlon, nlat)
        out = out.reshape(lead + (rg["nlat"], rg["nlon"]))
        return np.swapaxes(out, -1, -2)

    def to_xarray(
        self,
        predictions: Predictions,
        times,
        *,
        additional_coords: Mapping[str, Any] | None = None,
    ):
        """Regrid a saved trajectory onto a regular lat/lon xarray Dataset.

        Dynamics fields (and tracers) come out with dims
        ``(time, level, lon, lat)`` / ``(time, lon, lat)``. Following the
        repo-wide output convention the level axis is flipped to
        **surface-first** (level index 0 ≈ σ 0.996) — the physics-internal /
        pyses-native frame is top-first — and both a nominal-σ ``level``
        coordinate and the hybrid ``(a, b)`` mid-level tables are attached so
        analysis selects by coordinate value, never by blind index. Physics
        diagnostics from ``predictions.physics`` are included wherever their
        trailing shape matches the column layout (``(1, ncol)``, or a
        level/interface axis followed by it); other leaves are skipped.
        """
        import xarray as xr
        from jax.tree_util import tree_flatten_with_path

        rg = self._regrid_targets()
        a_full = 0.5 * (self._a_boundaries_pa[:-1] + self._a_boundaries_pa[1:])
        b_full = 0.5 * (self._b_boundaries[:-1] + self._b_boundaries[1:])
        # Nominal σ at mid-levels, flipped surface-first for the output file.
        sigma_full = (a_full / self.p0 + b_full)[::-1]

        times = np.asarray(times)
        coords = {
            "time": times,
            "level": ("level", sigma_full),
            "lon": ("lon", rg["lon_centers"]),
            "lat": ("lat", rg["lat_centers"]),
            "hybrid_a_full": ("level", a_full[::-1]),
            "hybrid_b_full": ("level", b_full[::-1]),
        }
        ds = xr.Dataset(coords=coords)
        ds["level"].attrs["long_name"] = (
            "nominal sigma (a/p0 + b) at layer mid-level, surface-first"
        )

        n_time = times.shape[0]
        ncol = self.colmap.num_cols

        def add_field(name, arr):
            # Accept both the (1, ncol) physics-state layout and the
            # flattened (ncols,) layout the column-vectorized physics
            # diagnostics carry, with an optional leading level /
            # level-interface axis. Anything else is skipped (per-band
            # optics, scalars, ...).
            arr = np.asarray(arr)
            if arr.shape[:1] != (n_time,):
                return
            trailing = arr.shape[1:]
            if trailing in ((1, ncol), (ncol,)):
                ds[name] = (("time", "lon", "lat"),
                            self._regrid_columns(arr.reshape(n_time, ncol)))
                return
            surf_ok = trailing[1:] in ((1, ncol), (ncol,)) if len(trailing) > 1 else False
            if surf_ok and trailing[0] in (self.nlev, self.nlev + 1):
                data = self._regrid_columns(arr.reshape(n_time, trailing[0], ncol))
                if trailing[0] == self.nlev:
                    ds[name] = (("time", "level", "lon", "lat"), data[:, ::-1])
                else:
                    ds[name] = (("time", "level_interface", "lon", "lat"),
                                data[:, ::-1])

        dyn = predictions.dynamics
        for field in ("u_wind", "v_wind", "temperature", "specific_humidity",
                      "geopotential", "normalized_surface_pressure"):
            add_field(field, getattr(dyn, field))
        for tracer_name, arr in dyn.tracers.items():
            add_field(tracer_name, arr)

        if predictions.physics is not None:
            leaves, _ = tree_flatten_with_path(predictions.physics)
            for path, leaf in leaves:
                if not hasattr(leaf, "shape"):
                    continue
                name = ".".join(
                    str(getattr(p, "key", getattr(p, "name", p))) for p in path
                )
                add_field(name, leaf)

        for key, value in (additional_coords or {}).items():
            ds.coords[key] = value
        return ds
