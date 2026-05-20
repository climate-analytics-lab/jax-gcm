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
- Dinosaur (dynamical core)
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
   python -m jcm.main run.mode=scm run.state_file=path/to/state.nc \
       run.column.lat_deg=0 run.column.lon_deg=180

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

   # Higher resolution: T85 (256x128 grid)
   coords = get_speedy_coords(spectral_truncation=85)
   terrain = TerrainData.aquaplanet(coords=coords)

   model = Model(
      coords=coords,
      time_step=20.0,  # smaller timestep for stability
      terrain=terrain
   )

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

**Initial Conditions**: Start from a specific state

.. code-block:: python

   from jcm.physics_interface import PhysicsState

   # Create or load initial state
   # initial_state = PhysicsState(...)

   predictions = model.run(
       initial_state=initial_state,
       save_interval=1.0,
       total_time=10.0
   )


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

Nudging is implemented as a Newtonian relaxation in spectral space:

.. math::

   \frac{\mathrm{d}X}{\mathrm{d}t}\bigg|_\mathrm{nudge}
   = \frac{X_\mathrm{ref} - X}{\tau}

where ``X`` is one of the dycore state variables (vorticity, divergence,
temperature, log surface pressure) and ``τ`` is the relaxation timescale.
The most common pattern is to nudge winds above the boundary layer and
let everything else evolve freely, so the model gets the right
synoptic-scale circulation while its physics still has the freedom to
respond:

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

   # Plot surface temperature evolution
   ds['temperature'].isel(level=7).mean(dim='lon').plot()
   plt.title('Zonal Mean Surface Temperature')
   plt.show()

   # Calculate global mean quantities
   global_mean_temp = ds['temperature'].weighted(
       ds['lat'].pipe(lambda x: np.cos(np.deg2rad(x)))
   ).mean(dim=['lon', 'lat'])

Next Steps
----------

- See :doc:`design` to understand the model architecture
- See :doc:`api` for detailed API documentation
- Check example notebooks in the ``notebooks/`` directory of the GitHub repo
- Read :doc:`developer` for contribution guidelines
