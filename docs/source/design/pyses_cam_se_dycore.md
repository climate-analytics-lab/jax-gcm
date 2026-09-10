# The pySES CAM-SE dynamical core

`jcm.dycore.pyses.PysesCamSEDycore` (registry name `"pyses_cam_se"`, optional
dependency `pip install jcm[pyses]`) is a spectral-element backend for the
`DynamicalCore` protocol: the pySES CAM-SE hydrostatic core on a quasi-uniform
cubed sphere, coupled to jcm's column physics through a pg2 finite-volume
physics grid. This page is the design record; the module docstrings under
`jcm/dycore/pyses/` carry the per-decision detail.

## One physics step

```text
{"model_state": pyses dict, "sim_time"}                     float64, GLL nodes
    │  to_physics_state: gll_to_fv area-average → (nlev, 1, ncol), cast f32
    ▼
jcm physics (ECHAM / SPEEDY / …), column-local on pg2 cells
    │  PhysicsTendency (nlev, 1, ncol)
    ▼  fv_to_gll scatter (cast → f64) + DSS (C0 projection)
pyses advance_coupling_step(..., physics_forcing={"dynamics", "tracers"})
    lump_all coupling: +physics_dt × forcing, then subcycled RK3-5STAGE
    dynamics, tensor hyperviscosity, nu_top sponge, Zerroukat vertical
    remap, consistent tracer advection
```

The step path is pure JAX (pyses's `advance_coupling_step` is itself jitted);
host-side numpy appears only at construction time (grid build, terrain /
forcing interpolation, output regrid weights).

## Key decisions

- **pg2 physics grid, from pyses itself.** Physics on the GLL nodes imprints
  the element structure onto the forcing; physics therefore runs on 2×2
  quasi-equal-area finite-volume cells per element (Hannah et al. 2021).
  pySES ≥ 0.1.3.1 ships the remap machinery
  (`pyses.dynamical_cores.finite_volume_grid`); jcm only wraps it
  (`FVPhysicsGrid`) into the `(nlev, 1, ncol)` layout, adds seam-safe
  cell-centre coordinates (Cartesian averaging — direct longitude averaging
  wraps wrongly across 0/2π), and pairs the scatter with pyses's DSS. The
  FV→GLL→FV round trip is exact (the paper's R1 identity); GLL→FV→GLL is not
  pointwise (J-weighted reconstruction + DSS ⇒ ~1% at ne3, shrinking with
  resolution).
- **`(1, ncol)` horizontal layout.** `ComposablePhysics(vectorize_columns=True)`
  needs exactly two horizontal dims, so the scattered columns pose as a
  degenerate 2-D grid. Limitation: the shipped physics caches lat/lon via a
  separable `meshgrid(latitudes, longitudes)`, which cannot represent
  per-column *longitudes* on this layout — every column sees a single
  reference longitude in zenith-angle / MACv2-SP placement (latitudes are
  exact). Open issue for production; the true coordinates are on
  `coords.horizontal.column_{latitudes,longitudes}`.
- **Full L47 with a finite top.** The whole ECHAM/ICON 47-level hybrid table
  is kept, with the singular `a[0]=0` top interface nudged to
  `0.5 × a[1]` (~1 Pa) so the pressure→height inversion is well posed and
  the explicit core sees a finite top layer. The thin lid is stabilised by
  CAM-SE's native `nu_top` Laplacian sponge (the analogue of ECHAM's
  `lmidatm` upper sponge), deepened to `n_sponge=8`; at ne30 use
  `nu_top≈2.5e4` to cut sponge sub-cycling.
- **Resting USSA-1976 initial state.** The analytic baroclinic base state
  extrapolates to negative temperatures at the ~1 Pa top; the piecewise
  U.S. Standard Atmosphere is positive to 84.852 km. The initial state is a
  dry column at rest over the real orography (surface pressure = USSA at
  each GLL node's orographic height); circulation and moisture spin up from
  the boundary forcing.
- **Constants from `jcm.constants`.** The pyses `physics_config` is built
  from the live jcm constants (`grav`, `rd`, `cpd`, `p0`, …) so dynamics and
  physics share one source of truth, mirroring the dinosaur backend's
  `physics_specs_from_constants`.
- **Tracers.** CAM-SE is a dry-mixing-ratio core: moisture is prognostic as
  `r = q/(1−q)`; every `TracerSpec` the physics declares becomes a pyses
  *passive* tracer (advected + vertically remapped, forced only by the
  scattered physics tendency). Dry air mass receives zero physics forcing.
  `TracerSpec.nondimensionalize` is a dinosaur-bridge concept and is ignored
  (identity in both directions).
- **Precision: float64 dynamics / float32 physics seam.** pySES's jax
  backend enables x64 process-wide (the explicit SE core needs it). The
  backend casts down to float32 in `to_physics_state` and back up in
  `FVPhysicsGrid.scatter_3d` — the only two seams. Caveat: the shipped ECHAM
  terms are not float32-dtype-stable under x64 (strong f64 table constants
  promote some carry leaves), so ECHAM runs — via `Model` or a caller-owned
  loop — currently need `physics_dtype=jnp.float64`; full-f32 physics awaits
  a physics-side dtype-stability fix.
- **Boundary data by bilinear sampling.** Terrain (orog/lsm/SSO) and monthly
  forcing climatology are bilinearly sampled from their regular lon/lat
  grids onto the columns (and orography onto the GLL nodes) at build time;
  `jcm.dycore.pyses.build_forcing` wraps the result as ordinary
  `ForcingData` `TimeSeries` leaves (WRAP_YEAR), so `Model`'s
  `forcing.select(date)` machinery is reused untouched. Interpolate offline
  for higher-resolution forcing.
- **Output.** `to_xarray` bin-averages the columns onto a regular lat/lon
  grid sized to ~1 column per box (empty boxes filled from the nearest
  column), flips both vertical axes to the repo's surface-first output
  convention, and hands off to `jcm.cf_metadata.finalize_output` for the
  nominal-σ / hybrid (a, b) coordinates and CF attributes, so analysis selects
  by value, never by blind index. The interface axis is named `level_i`, the
  same as the dinosaur backend's — see
  [output_vertical_conventions](output_vertical_conventions.md).

## Usage

```python
import jax.numpy as jnp
from jcm.dycore.pyses import PysesCamSEDycore, build_forcing
from jcm.model import Model
from jcm.physics.echam.echam_terms import echam_physics

dycore = PysesCamSEDycore(
    nx=30, npt=4, dt_seconds=900.0,
    terrain_file="jcm/data/bc/t63/terrain.nc",
    nu_top=2.5e4,                       # ne30 sponge setting
    physics_dtype=jnp.float64,          # required for the ECHAM stack
)
model = Model(
    dycore=dycore,
    time_step=dycore.dt_seconds / 60.0,  # keep Model minutes == dycore dt
    physics=echam_physics(radiation_scheme="grey"),
)
forcing = build_forcing("jcm/data/bc/t63/forcing.nc", dycore)
predictions = model.run(forcing=forcing, save_interval=1.0, total_time=5.0)
ds = dycore.to_xarray(...)              # lat/lon regridded output
```

## Open issues for the ne30 GPU production run

- Per-column longitudes in the shipped physics `cache_coords` (see above).
- Full-float32 ECHAM physics (dtype-stable radiation carry).
- Device sharding: pyses shards on the element axis via
  `PYSES_SHARD_CPU_COUNT` / its jax device setup; the jcm physics column
  axis sharding (`_flattened_column_sharding`) is a no-op without an
  `spmd_mesh` on the coords adapter — multi-GPU runs need these two schemes
  reconciled.
- Forcing/terrain interpolation is bilinear from T63; higher-resolution
  boundary data should be regridded offline (conservative) onto the column
  set.
