Getting Started
===============

.. _installation:

Installation
------------

To use JAX-GCM, first install it:

.. code-block:: console

   $ pip install jcm

or for the development version:

.. code-block:: console

   $ git clone https://github.com/climate-analytics-lab/jax-gcm.git
   $ cd jax-gcm
   $ git switch dev
   $ pip install -e .

Requirements
^^^^^^^^^^^^

- Python ≥ 3.11
- JAX
- Dinosaur (the dynamical-core backend shipped with v2.0)
- XArray (for I/O and data handling)

See ``requirements.txt`` for the complete list of dependencies.

Command-line interface
----------------------

Most simulations can be launched without writing any Python via the bundled
Hydra CLI. ``jcm/main.py`` is executable so it can be invoked either as a
module or directly::

   ./jcm/main.py                                               # direct invocation
   python -m jcm.main                                          # equivalent module form
   python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid
   python -m jcm.main physics=echam grid=echam_t63_l47_hybrid
   python -m jcm.main physics=held_suarez grid=held_suarez_t31_l8 \
       run.total_time=30 run.save_interval=1
   python -m jcm.main physics=echam +physics.terms.tiedtke_convection.params.entrpen=4e-4
   python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid run=longrun
   python -m jcm.main physics=echam-emulated-2m grid=echam_t63_l47_hybrid
   python -m jcm.main run.mode=scm run.state_file=path/to/state.nc \
       run.column.lat_deg=0 run.column.lon_deg=180

The state-file modes (``run.mode=scm`` and ``run.mode=prescribed``) read a
netCDF written by an earlier run. Both the vertical orientation and the tracer
list are handled for you: output files are surface-first and are flipped into
the top-first physics frame on load, and with ``run.tracer_vars`` unset (the
default) every tracer the configured physics declares — ``qc``/``qi`` for the
one-moment cloud scheme, plus ``qnc``/``qni`` for the two-moment one — is
loaded from the file when it carries it. Pass an explicit mapping to rename
variables, or ``run.tracer_vars={}`` to load none.

Inspect the available config groups and the fully-composed config::

   python -m jcm.main --help                                   # config-group choices
   python -m jcm.main --cfg job                                # composed config
   python -m jcm.main --cfg job grid=echam_t63_l47_hybrid       # with overrides

Config groups live under ``jcm/config/``: ``physics``, ``grid``, ``run``,
``init``, ``terrain``, ``forcing``, ``diffusion``.

Quick Start Examples
--------------------

Aquaplanet Simulation
^^^^^^^^^^^^^^^^^^^^^

An aquaplanet simulation is the simplest configuration - a water-covered planet with no orography and constant (zonally symmetric) forcing. This is ideal for learning the model and testing new physics:

.. code-block:: python

   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Create a model with default aquaplanet configuration
   model = Model(
      coords=get_speedy_coords(),  # T31 spectral resolution with 8 vertical levels
      time_step=30.0  # minutes
   )

   # Run a 120-day simulation
   predictions = model.run(
      save_interval=10.0,  # save every 10 days
      total_time=120.0     # total simulation time in days
   )

   # Convert output to xarray Dataset for analysis
   ds = predictions.to_xarray()
   print(ds)

This creates a T31 spectral resolution model (96x48 grid points) with 8 vertical levels using the SPEEDY physics package. The default forcing includes zonally symmetric sea surface temperatures and no land.

Realistic Simulation
^^^^^^^^^^^^^^^^^^^^

For a more realistic simulation with orography and time-varying boundary conditions, you can load data from files:

.. code-block:: python

   from jcm.model import Model
   from jcm.terrain import TerrainData
   from jcm.forcing import ForcingData
   from importlib import resources

   coords = get_speedy_coords()  # T31 spectral resolution with 8 vertical levels

   # Load realistic orography and land-sea mask, interpolated to T31 grid
   data_dir = resources.files("jcm.data.bc.t30.clim")
   terrain_file = data_dir / "terrain.nc"
   terrain = TerrainData.from_file(terrain_file, coords=coords)

   # Load realistic forcing data (SST, sea ice, soil moisture, etc.) interpolated to T31 grid.
   # Time-varying variables are wrapped as `TimeSeries` leaves; the Model
   # picks the right slice each step via `forcing.select(date)`. By default
   # `from_file` auto-detects climatology vs date-aligned mode from the
   # netCDF time axis (one-year files wrap, multi-year files align by date).
   forcing_file = data_dir / "forcing.nc"
   forcing = ForcingData.from_file(forcing_file, coords=coords)

   # Create model with realistic configuration. SPEEDY assumes a 365-day
   # no-leap calendar by construction; pass `calendar='gregorian'` if you
   # want the model clock to advance against real Gregorian timestamps.
   model = Model(
      coords,
      time_step=30.0,
      terrain=terrain
   )

   # Run simulation
   predictions = model.run(
      forcing=forcing,
      save_interval=5.0,   # save every 5 days
      total_time=30.0      # 30-day simulation
   )

   # Convert to xarray and save
   ds = predictions.to_xarray()
   ds.to_netcdf("output.nc")

Customizing the Model
^^^^^^^^^^^^^^^^^^^^^

You can customize various aspects of the model:

**Resolution**: Change the horizontal and vertical resolution

.. code-block:: python

   from jcm.terrain import TerrainData
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Higher resolution: T85 (256x128 grid). time_step is omitted, so the
   # Model picks a numerically stable step for this resolution (see
   # "Choosing the time step" below).
   coords = get_speedy_coords(spectral_truncation=85)
   terrain = TerrainData.aquaplanet(coords=coords)

   model = Model(
      coords=coords,
      terrain=terrain
   )

**Choosing the time step**: ``time_step`` (minutes) is optional and is
resolved from a single source of truth:

* An explicit ``time_step=...`` always wins. If you also pass an
  explicitly-constructed dycore, the two must agree — a mismatch raises,
  because the dycore bakes its step into its integrator at construction
  and physics/dates/saves would otherwise silently advance by a different
  ``dt`` than the dynamics.
* With an explicit ``dycore=...`` and no ``time_step``, the Model adopts
  the dycore's ``dt_seconds`` — whoever constructs the dycore owns the
  step.
* With ``coords=...`` and no ``time_step`` (the Model builds the dycore
  itself), the active physics is consulted via
  :py:meth:`jcm.physics_interface.Physics.stable_time_step_minutes`.
  Physics without a grid-dependent stability limit (ECHAM, Held–Suarez,
  ...) keep the historical 30-minute default; SPEEDY shortens the step
  only for high-vertical-level / high-truncation grids where its explicit
  surface drag would otherwise be unstable (standard 7/8-level SPEEDY
  runs stay at exactly 30 minutes). See
  :doc:`design/speedy_variable_levels` for the stability analysis.

**Physics**: Use different physics packages or configurations

.. code-block:: python

   from jcm.physics.speedy.speedy_terms import speedy_physics
   from jcm.physics.speedy.params import Parameters
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Customize physics parameters
   params = Parameters.default()
   params = params.replace(...)  # modify parameters as needed

   physics = speedy_physics(parameters=params)

   model = Model(
      coords=get_speedy_coords(),
      time_step=30.0,
      physics=physics
   )

Parameters must be set **before** the Model is built, as above. jcm binds
them into the compiled executable when the physics is first traced
(``Model._run_from_state`` takes ``self`` as a static jit argument), and
editing the parameters a live model already holds is unreliable: whether a
later run sees the edit depends on which of JAX's compilation caches it
hits, so the results cannot be trusted either way. jcm logs a warning and
flags ``preds.params`` when it detects that. To sweep a parameter, build a
Model per value inside a single ``jax.jit``, which makes the rebuild a
trace-time cost paid once instead of a recompile per iteration::

   @jax.jit
   def forecast(albsea):
       p = Parameters.default()
       p = p.replace(mod_radcon=p.mod_radcon.replace(albsea=albsea))
       model = Model(coords=coords, physics=speedy_physics(parameters=p))
       return summarize(model.run(save_interval=0.25, total_time=0.25))

A Model with ``observers=`` needs one extra argument under an outer
``jit``, because the observers' sampling tables are built on the host from
the window's start time and that is a tracer there: pass a concrete
``observer_t0_days``, or, to reuse one compilation across *different*
windows, pass tables from ``model.prepare_observers(t0_days,
save_interval, total_time)`` as ``observer_xs``.

**Logging**: ``Model(log_level=...)`` defaults to ``logging.WARNING`` and is
applied to the ``jcm`` logger rather than the root logger, so jcm's warnings
about a run stay audible without jcm reconfiguring logging for your
application. Pass ``logging.CRITICAL`` to quieten it. The Hydra CLI exposes
the same knob as ``run.log_level`` (also ``WARNING`` by default).

**Dynamical core**: Pass a backend explicitly when you need backend-specific
configuration. ``Model(coords=...)`` remains the shorthand for constructing
the shipped Dinosaur backend with default settings. The v2.0 Hydra CLI also
uses Dinosaur; explicit backend selection is currently a Python-API workflow.
An explicitly-constructed backend owns the time step: the Model adopts its
``dt_seconds``, and passing a conflicting ``Model(time_step=...)`` raises
(see "Choosing the time step" above).

.. code-block:: python

   from jcm.diffusion import DiffusionFilter
   from jcm.dycore.dinosaur import DinosaurDycore
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords
   from jcm.terrain import TerrainData

   coords = get_speedy_coords()
   dycore = DinosaurDycore(
       coords=coords,
       terrain=TerrainData.aquaplanet(coords),
       dt_seconds=1800.0,
       diffusion=DiffusionFilter.default(),
   )
   model = Model(dycore=dycore)  # adopts the dycore's 30-minute step

**Initial Conditions**: Start from a specific state

The simplest path is to hand :meth:`~jcm.model.Model.run` a
:class:`~jcm.physics_interface.PhysicsState` you built yourself:

.. code-block:: python

   from jcm.physics_interface import PhysicsState

   # Create or load initial state
   # initial_state = PhysicsState(...)

   predictions = model.run(
       initial_state=initial_state,
       save_interval=1.0,
       total_time=10.0
   )

For the common starting states there are ready-made *state builders* in
:mod:`jcm.initial_states` — the same ones the CLI's ``init`` config group
exposes. Each one **returns** a starting state; hand it to
:meth:`~jcm.model.Model.run` via ``initial_state=``.

.. code-block:: python

   from jcm.model import Model
   from jcm.terrain import TerrainData
   from jcm.physics.echam.echam_terms import echam_physics
   from jcm.initial_states import jw_state

   coords = ...                       # your CoordinateSystem
   terrain = TerrainData.from_coords(coords)
   model = Model(coords=coords, terrain=terrain, physics=echam_physics())

   # Jablonowski–Williamson-style lapse-rate atmosphere at 60 % RH,
   # with surface pressure rebalanced over the orography.
   predictions = model.run(
       initial_state=jw_state(model, rh=0.6),
       total_time=10.0, save_interval=1.0,
   )

The other builders follow the identical pattern:

* :func:`~jcm.initial_states.balanced_isothermal_state` — a uniform
  288 K rest state with the same orography-balanced surface pressure; a robust
  spin-up state for moist physics over real terrain.
* :func:`~jcm.initial_states.era5_state` ``(coords, date)`` — a ``PhysicsState``
  seeded from an ERA5 (WeatherBench2) slice at an ISO date, regridded via
  :mod:`jcm.data.era5` (re-exported here for discoverability).
* :func:`~jcm.initial_states.checkpoint_state` ``(model, path)`` — a
  **warm start** from a saved state (e.g. a hosted equilibrated state under
  ``bundles/<grid>_<levels>/init_states/``). It returns
  ``(state, physics_carry, donor_days)``; unlike a checkpoint *resume* the
  donor's elapsed-day count is discarded, so the clock starts at the model's
  ``start_date`` — this skips the ~9-month from-cold spin-up without
  inheriting the donor run's calendar. Pass ``physics_carry`` to
  ``initial_physics_state`` so the warm start keeps the donor's cross-step
  physics carry (radiation sub-cycle cache, prior-step TKE) rather than
  resetting it at the run seam:

  .. code-block:: python

     from jcm.initial_states import checkpoint_state

     state, physics_carry, _ = checkpoint_state(
         model, 'bundles/echam_t63_l47_hybrid/init_states/spun_up.msgpack')
     predictions = model.run(
         initial_state=state, initial_physics_state=physics_carry,
         forcing=forcing, total_time='1 year', save_interval='1 day')

  (Restoring a *checkpoint* to continue a preempted run of your own — keeping
  the elapsed clock — is the separate :func:`jcm.checkpoint.load_checkpoint`
  path documented under "Checkpointing for preemptible runs" below.)


Calendar-aware durations and resampling
---------------------------------------

``Model.run`` and ``Model.resume`` accept either a numeric day count or a
calendar-string for ``save_interval`` and ``total_time``. Strings like
``'1 month'`` and ``'1 year'`` are resolved against the model's calendar
(``'365_day'`` by default; pass ``Model(calendar='gregorian')`` for the
365.2425-day approximation). The integrator itself stays fixed-cadence —
each "month" is a fixed 365/12-day chunk, not aligned to calendar month
boundaries — so this is mostly an ergonomic shortcut.

For *calendar-aligned* monthly / annual statistics, run the model at a
daily ``save_interval`` and post-resample the trajectory using xarray's
standard ``resample`` API. The trajectory's ``time`` coord is real
``datetime64``, so xarray's resampler does the calendar bookkeeping:

.. code-block:: python

   predictions = model.run(save_interval='1 day', total_time='1 year')
   ds = predictions.to_xarray()

   # Calendar-aligned monthly means.
   monthly = ds.resample(time='1MS').mean()

   # Daily total precipitation summed into calendar months, etc.
   monthly_precip = ds['precipitation'].resample(time='1MS').sum()

The cost of this pattern is keeping daily output in memory for the
duration of the run.

Long forcing time-series and chunked runs
-----------------------------------------

For multi-year forcing files, it's often convenient to run the model one
year at a time. This keeps memory bounded and lets you save output as you
go. Use ``xarray.Dataset.groupby('time.year')`` to slice the forcing,
then ``Model.run`` for the first year and ``Model.resume`` for subsequent
years to continue from the previous state:

.. code-block:: python

   import xarray as xr
   from jcm.forcing import ForcingData

   ds = xr.open_dataset('era5_1980_2010.nc')
   yearly_outputs = []

   year_iter = iter(ds.groupby('time.year'))

   year, year_ds = next(year_iter)
   forcing = ForcingData.from_dataset(year_ds, coords=coords)
   preds = model.run(forcing=forcing, save_interval='1 day',
                     total_time='1 year')
   yearly_outputs.append(preds.to_xarray())

   for year, year_ds in year_iter:
       forcing = ForcingData.from_dataset(year_ds, coords=coords)
       preds = model.resume(forcing=forcing, save_interval='1 day',
                            total_time='1 year')
       yearly_outputs.append(preds.to_xarray())

   trajectory = xr.concat(yearly_outputs, dim='time')

xarray's lazy loading means each year's slice only pulls the data it
actually needs from disk, so this stays memory-efficient even for very
long forcing records.

Yearly forcing bundles
^^^^^^^^^^^^^^^^^^^^^^^

The transient AMIP boundary conditions ship as one file per year (download
only the years you run, append new years without rewriting history). A config
points at a ``{year}`` pattern plus an inclusive range;
:func:`jcm.forcing.expand_yearly_files` turns that into the concrete file list
that :meth:`~jcm.forcing.ForcingData.from_file` concatenates along ``time``:

.. code-block:: python

   from jcm.forcing import ForcingData, expand_yearly_files

   files = expand_yearly_files(
       'hf://bundles/t63/forcing_amip/{year}.nc',
       years=[1979, 1983],            # inclusive
       available=[1979, 2022],        # optional: product's source coverage
   )
   forcing = ForcingData.from_file(files, coords=coords)

Passing ``available`` widens the expansion by one year on each side (clipped to
coverage) so the mid-month samples bracket the run's start/end instead of
clamping for ~half a month. Non-pattern specs (plain paths, lists, ``None``)
pass through untouched, so a run can mix a yearly SST pattern with a static
dust climatology under one ``forcing.years`` range. When you hand-assemble a
:class:`~jcm.forcing.ForcingData` rather than loading a validated bundle,
:func:`jcm.forcing.validate_emissions_grid` and
:func:`jcm.forcing.validate_oxidant_levels` guard the grid/level layout the
physics expects.


Checkpointing for preemptible runs
----------------------------------

Multi-day integrations on preemptible compute (spot instances, Slurm
``--requeue`` queues, NRP Nautilus) can be killed at short notice. Set
``run.checkpoint_path`` to make a chunked run resumable: after each
chunk the runner persists the modal + physics state and the elapsed
sim-day count to that file (atomic write via tmpfile + rename, so a
kill mid-write leaves the previous checkpoint intact). When the same
command is launched again with the file already in place, the run
restores from the checkpoint and only steps the remaining chunks.

.. code-block:: bash

   python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid \
       run=longrun run.checkpoint_path=/scratch/$JOB_ID.ckpt

The same primitives are available directly to bring-your-own-driver
workflows via :py:mod:`jcm.checkpoint`:

.. code-block:: python

   from jcm.checkpoint import save_checkpoint, load_checkpoint

   model.run(forcing=forcing, total_time=10)
   save_checkpoint(model, '/scratch/run.ckpt', elapsed_days=10.0)

   # ... later, in a fresh process ...
   model = build_model(cfg)            # same coords + physics
   model.bootstrap_state()             # populate template pytrees
   elapsed = load_checkpoint(model, '/scratch/run.ckpt')
   model.resume(forcing=forcing, total_time=20 - elapsed)

The on-disk format is flax's msgpack codec applied to flattened lists
of arrays — small (state pytrees are a few MB even at T63L47) and
portable across hosts as long as the destination ``Model`` was built
with the same coords and physics term composition.


Nudging the model toward an external state
-------------------------------------------

The model can be relaxed toward an external reference state ("nudging")
to suppress internal variability that's unrelated to the question you're
asking — useful for comparing model fields to specific dates of
observations, or for reducing noise in calibration runs.

Nudging is implemented as a gridpoint-space ``PhysicsTerm``:

.. math::

   \frac{\mathrm{d}X}{\mathrm{d}t}\bigg|_\mathrm{nudge}
   = \frac{X_\mathrm{ref} - X}{\tau}

where ``X`` is a gridpoint wind or temperature field and ``τ`` is the
relaxation timescale.
The most common pattern is to nudge winds above the boundary layer and
let everything else evolve freely, so the model gets the right
synoptic-scale circulation while its physics still has the freedom to
respond.

**From config** the whole setup is one flag: ``nudging=era5`` pulls the
run window from WeatherBench2's public cloud ERA5 (regridded to the
model grid and cached locally by :mod:`jcm.data.era5`), and
``init=era5`` starts the run from the ERA5 state at the same date:

.. code-block:: console

   $ python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid \
         init=era5 nudging=era5 run.start_date=2010-01-01 run.total_time=30

Prefetch on a login node first when compute nodes lack internet
(``python -m jcm.data.era5 --grid echam_t63_l47_hybrid --start
2010-01-01 --end 2010-01-31 --init``). The WB2 stores carry 13 pressure
levels up to 50 hPa, so nudging is automatically masked off above
``nudging.min_pressure_hpa`` (default 60), and requires internet or a
warm cache; see ``jcm/config/nudging/era5.yaml`` for the knobs
(``tau_hours``, ``pbl_levels``, ``nudge_temperature``, ``freq``).

**In code**, wire it manually against any reference dataset:

.. code-block:: python

   import xarray as xr
   from jcm.forcing import ForcingData
   from jcm.model import Model
   from jcm.nudging import NudgingTarget, NudgingConfig, with_nudging

   ref_ds = xr.open_dataset('era5_2010.nc')   # u, v, T on (time, lev, lat, lon)

   # The target is loaded straight off the netCDF in gridpoint space and
   # attached to forcing — it's just another per-step input. The Model
   # slices it inside ``forcing.select(date, calendar)`` like every other
   # time-varying leaf, so the nudging term never sees the date.
   target = NudgingTarget.from_dataset(ref_ds)
   forcing = ForcingData.from_file('boundary_conditions.nc', coords=coords)
   forcing = forcing.replace(nudging_target=target)

   config = NudgingConfig.winds_only(
       nlev=coords.vertical.layers,
       tau_seconds=21600.0,        # 6 h relaxation
       pbl_levels=2,               # leave the bottom 2 levels free
   )

   nudged_physics = with_nudging(physics, config)
   nudged = Model(coords=coords, terrain=terrain, physics=nudged_physics)
   predictions = nudged.run(forcing=forcing, save_interval='1 day', total_time='1 month')

The reference data can be a single climatology (passed with
``time_var=None``) or a multi-year time series; the latter aligns
against the model's calendar through the same machinery the regular
forcing uses.

Nudging is dycore-agnostic — it's just another :class:`PhysicsTerm`,
producing a gridpoint :class:`PhysicsTendency` that the dycore consumes
through the standard physics-coupling path. The same setup works under
SPEEDY, ECHAM, or any other physics package, on any
:class:`DynamicalCore` backend.

Composing extra terms: the upper sponge
----------------------------------------

Because physics is *composable*, adding a scheme is just ``+``-ing a
:class:`~jcm.physics.physics_term.PhysicsTerm` onto the package. An
:class:`~jcm.physics.dissipation.UpperSponge` — Rayleigh drag on the winds
plus zonal-mean relaxation of temperature at the top few levels — damps
spectral ringing near a rigid model lid:

.. code-block:: python

   from jcm.physics.dissipation import UpperSponge
   from jcm.physics.echam.echam_terms import echam_physics

   physics = echam_physics() + UpperSponge(n_sponge_levels=5,
                                           sponge_timescale_s=3 * 3600.0)
   model = Model(coords=coords, terrain=terrain, physics=physics)

The relaxation timescales that both the sponge and the nudging term use follow
a masked per-level ``1/tau`` profile; :func:`jcm.nudging.inv_tau_profile`
builds one from a dycore vertical coordinate (zeroing the boundary-layer
levels and everything above ``min_pressure_hpa``). See the
:mod:`jcm.nudging` and :mod:`jcm.physics.dissipation.upper_sponge` module
docstrings for the full set of knobs, and :doc:`design/composable_physics`
for the composition API (``+``, ``replace``, ``remove``).

Multi-Device Parallelization
-----------------------------

JCM supports multi-device parallelization using JAX's SPMD (Single Program Multiple Data) sharding. This allows you to split computation across multiple GPUs or TPUs for faster execution, especially useful for higher resolution simulations.

If you don't specify ``spmd_mesh`` when building your coords, JCM runs on a single device by default. This is the recommended approach for smaller resolutions (T31, T42) or when you only have a single GPU/TPU available.

Basic Concepts
^^^^^^^^^^^^^^

**SPMD Mesh**: Defines how to partition data across devices. The mesh has three dimensions corresponding to ``(x, y, z)`` or ``(longitude, latitude, vertical)``.

**Sharding Strategy**: Typically, for SPEEDY Physics simulations,  you want to shard the longitude dimension first since it usually has the most grid points. 
For Physics implementations with more layers (e.g. 32 or 64 layers) however, you may find that sharding the dycore in the vertical dimension to be most effective. 
Future implementations may allow for more flexible sharding strategies.

Enabling Parallelization
^^^^^^^^^^^^^^^^^^^^^^^^

To enable multi-device parallelization, pass ``spmd_mesh`` to the coords helper
(e.g. ``get_speedy_coords`` or ``get_coords``) and build the ``Model`` with those coords:

.. code-block:: python

   import jax
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Check available devices
   print(f"Available devices: {jax.devices()}")
   print(f"Number of devices: {len(jax.devices())}")

   # Define a mesh to split longitude across 4 devices
   # Mesh shape (4, 1, 1) means:
   #   - Split longitude dimension across 4 devices
   #   - Don't split latitude (1)
   #   - Don't split vertical (1)
   coords = get_speedy_coords(spmd_mesh=(4, 1, 1))
   model = Model(coords=coords)
   predictions = model.run(save_interval=5.0, total_time=30.0)

Mesh Configuration Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The product of mesh dimensions must equal the number of available devices:

- ``(4, 1, 1)``: Split longitude across 4 devices
- ``(2, 2, 1)``: Split longitude (2) and latitude (2) across 4 devices total
- ``(8, 1, 1)``: Split longitude across 8 devices (for higher resolutions)

**Rules of thumb:**

1. Product of mesh dimensions = number of devices
2. Longitude (x) usually has most grid points → split first
3. Higher resolutions (T85+) benefit more from sharding

Analyzing Output
----------------

The model output is a :py:class:`Predictions` object containing the model state trajectory. Convert it to xarray for analysis:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Convert to xarray Dataset
   ds = predictions.to_xarray()

   # Print variables
   print(ds.data_vars)

   # Plot surface temperature evolution. Output is surface-first, so index 0
   # is the level nearest the ground on both vertical axes.
   ds['temperature'].isel(level=0).mean(dim='lon').plot()
   plt.title('Zonal Mean Surface Temperature')
   plt.show()

   # Calculate global mean quantities (see jcm.analysis below for the
   # conservation-grade weights)
   global_mean_temp = ds['temperature'].weighted(
       ds['lat'].pipe(lambda x: np.cos(np.deg2rad(x)))
   ).mean(dim=['lon', 'lat'])

Post-processing with ``jcm.analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:mod:`jcm.analysis` is the one home for the xarray post-processing recipes that
otherwise get re-implemented per script — area weights, global means, layer
pressure thicknesses and column burdens, all computed on *saved* netCDF output:

.. code-block:: python

   import xarray as xr
   from jcm import analysis

   ds = xr.open_dataset('output.nc')

   # Area-weighted global mean over the horizontal dims (everything except
   # time / level / level_i / mode). On a dinosaur (Gauss-Legendre) output
   # grid this uses the *exact* quadrature weights, not the cos(lat)
   # approximation — so conservation residuals actually integrate to zero.
   T_global = analysis.global_mean(ds['temperature'])

   # Column burden [kg/m^2] of a tracer, mass-weighted with the file's own
   # layer thickness. layer_pressure_thickness() prefers the model's
   # pressure_thickness diagnostic and falls back to differencing
   # pressure_half; both output vertical axes are surface-first (#710).
   dp = analysis.layer_pressure_thickness(ds)
   qc_burden = analysis.column_integral(ds['qc'], dp)   # or, in one step:
   qc_burden = analysis.column_burden(ds, 'qc')

   # column_burden already time-broadcasts, so a global-mean burden time series is:
   burden_ts = analysis.global_mean(analysis.column_burden(ds, 'qc'))

:func:`~jcm.analysis.area_weights` deliberately returns a dims-only
``DataArray`` (no ``lat`` coordinate) so ``.weighted()`` broadcasts it by
dimension name without float32/float64 coordinate-alignment surprises. Files
written before the #710 vertical-convention unification are **not** supported by
``layer_pressure_thickness`` — see :doc:`design/output_vertical_conventions`.

Vertical coordinates in the output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output files carry two vertical axes: ``level`` (``nlev`` layer mid-levels —
temperature, tracers, ``pressure_full``) and ``level_i`` (``nlev+1``
interfaces — ``pressure_half``, radiative fluxes). **Both run surface-first**,
so index 0 is the level nearest the ground and ``level[k]`` sits between
``level_i[k]`` and ``level_i[k+1]``.

To mass-weight a ``level`` field — a column burden, say — use the layer
pressure thickness ``pressure_thickness`` [Pa], which is written directly on
the ``level`` axis (positive, already aligned with the tracers) by the ECHAM
physics stacks:

.. code-block:: python

   burden = (ds['qc'] * ds['pressure_thickness'] / 9.81).sum('level')  # kg/m^2

For a file written before ``pressure_thickness`` existed (or a SPEEDY run,
which does not compute it), reconstruct Δp from ``pressure_half`` instead —
mixing the two axes is safe because both run surface-first:

.. code-block:: python

   # diff() keeps the *interface* sigma labels, so after renaming the dim the
   # mid-level coordinate must be assigned explicitly — otherwise xarray's
   # alignment finds no matching labels and the product is silently empty.
   dp = (-ds['pressure_half'].diff('level_i')
         .rename(level_i='level').assign_coords(level=ds['level']))
   burden = (ds['qc'] * dp / 9.81).sum('level')     # kg/m^2

Both axes are CF-labelled nominal sigma (``a/p0 + b``) and carry
``positive = "down"``; the hybrid ``(a, b)`` tables travel with the file as the
``hybrid_a_full`` / ``hybrid_b_full`` / ``hybrid_a_half`` / ``hybrid_b_half``
coordinates, so ``p = a + b * p_s`` is reproducible from the file alone. See
:doc:`design/output_vertical_conventions` — including for how to read files
written before this convention was unified, where the interface axis was
stored top-first.

Overriding physical constants
-----------------------------

All shared physical constants live in a single source of truth,
:class:`jcm.constants.PhysicalConstants`, exposed as a process-global singleton.
Each quantity has exactly one canonical name (e.g. dry-air specific heat is
``cpd``, the dry-air gas constant ``rd``, the melting point ``tmelt``).
*Derived* quantities (``rd = akap·cpd``, ``cvd``, ``rgrav``, the ``vtmpc*``
coefficients) are computed on access, so they always stay consistent with the
base values.

.. note::

   jcm's default gravitational acceleration is ``grav = 9.81`` m/s² (the value
   the physics ports were tuned against), **not** the WMO standard 9.80665.
   When you compare a burden or mass budget against an external tool, weight
   with the *same* ``g`` the model used — read it from :mod:`jcm.constants`
   (``import jcm.constants as c; c.grav``) rather than hardcoding a literal.
   :mod:`jcm.analysis`'s column integrals already use the live singleton for
   exactly this reason.

To run with non-default constants — say for a different planet or a sensitivity
study — call :func:`jcm.constants.set_constants` **before constructing the
model**. Only *base* fields are set; derived constants follow automatically, and
both the dynamical core and the physics pick up the override:

.. code-block:: python

   import jcm.constants as c
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Override base constants (derived values recompute automatically)
   c.set_constants(grav=9.80665, rearth=6.371229e6, cpd=1005.0)
   assert c.rd == c.akap * c.cpd     # derived value follows

   coords = get_speedy_coords(layers=8, spectral_truncation=31)
   model = Model(coords=coords)       # honours the override

From the CLI, use the ``constants`` config group (applied before the model is
built):

.. code-block:: bash

   python -m jcm.main +constants.grav=9.80665 +constants.rearth=6.4e6

.. note::

   The override is **process-global** and must be set *before* the model is
   constructed. Read constants by attribute access (``import jcm.constants as
   c; c.grav``) — a ``from jcm.constants import grav`` captures the value at
   import time and will not track later overrides.

.. warning::

   Set constants **once, at the start of the process, before building any
   model** — think of them as fixed for the run (hence *constants*), in contrast
   to calibratable scheme parameters which are threaded through the model as
   explicit, differentiable arguments.

   Constants are baked into a model at construction/trace time: the dynamical
   core reads the singleton when it is built, and physics functions read the
   current values when JAX first traces them. Because JAX caches compiled
   functions, calling :func:`~jcm.constants.set_constants` *after* a model has
   been built — or building a **second** model with different constants in the
   same process — is **not** guaranteed to take effect; already-traced/compiled
   code keeps the values it was first traced with. To compare several constant
   sets, run each in a **separate process** (e.g. a fresh interpreter or a
   separate CLI invocation).

Next Steps
----------

- See :doc:`design` to understand the model architecture
- See :doc:`api` for detailed API documentation
- Check example notebooks in the ``notebooks/`` directory of the GitHub repo
- Read :doc:`developer` for contribution guidelines
