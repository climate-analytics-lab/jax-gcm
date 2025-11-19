Getting Started
===============

.. _installation:

Installation
------------

To use JAX-GCM, first install it using pip:

.. code-block:: console

   $ git clone https://github.com/climate-analytics-lab/jax-gcm.git
   $ cd jax-gcm
   $ pip install -e .

Requirements
^^^^^^^^^^^^

- Python ≥ 3.11
- JAX
- Dinosaur (dynamical core)
- XArray (for I/O and data handling)

See ``requirements.txt`` for the complete list of dependencies.

Quick Start Examples
--------------------

Aquaplanet Simulation
^^^^^^^^^^^^^^^^^^^^^

An aquaplanet simulation is the simplest configuration - a water-covered planet with no orography and constant (zonally symmetric) forcing. This is ideal for learning the model and testing new physics:

.. code-block:: python

   from jcm.model import Model

   # Create a model with default aquaplanet configuration
   model = Model(
       time_step=30.0,  # minutes
   )

   # Run a 120-day simulation
   predictions = model.run(
       save_interval=10.0,  # save every 10 days
       total_time=120.0     # total simulation time in days
   )

   # Convert output to xarray Dataset for analysis
   ds = model.predictions_to_xarray(predictions)
   print(ds)

This creates a T31 spectral resolution model (96x48 grid points) with 8 vertical levels using the SPEEDY physics package. The default forcing includes zonally symmetric sea surface temperatures and no land.

Realistic Simulation
^^^^^^^^^^^^^^^^^^^^

For a more realistic simulation with orography and time-varying boundary conditions, you can load data from files:

.. code-block:: python

   from jcm.model import Model
   from jcm.geometry import Geometry
   from jcm.forcing import ForcingData
   from pathlib import Path

   # Load realistic orography and land-sea mask
   data_dir = Path("jcm/data/bc")
   terrain_file = data_dir / "terrain_t31.nc"
   geometry = Geometry.from_file(terrain_file)

   # Load realistic forcing data (SST, sea ice, soil moisture, etc.)
   forcing_file = data_dir / "forcing_t31.nc"
   forcing = ForcingData.from_file(forcing_file)

   # Create model with realistic configuration
   model = Model(
       time_step=30.0,
       geometry=geometry,
   )

   # Run simulation
   predictions = model.run(
       forcing=forcing,
       save_interval=5.0,   # save every 5 days
       total_time=30.0      # 30-day simulation
   )

   # Convert to xarray and save
   ds = model.predictions_to_xarray(predictions)
   ds.to_netcdf("output.nc")

Customizing the Model
^^^^^^^^^^^^^^^^^^^^^

You can customize various aspects of the model:

**Resolution**: Change the horizontal and vertical resolution

.. code-block:: python

   from jcm.geometry import Geometry

   # Higher resolution: T63 (192x96 grid) with 12 levels
   geometry = Geometry.from_grid_shape(
       nodal_shape=(192, 96),
       num_levels=12
   )

   model = Model(
       time_step=15.0,  # smaller timestep for stability
       geometry=geometry
   )

**Physics**: Use different physics packages or configurations

.. code-block:: python

   from jcm.physics.speedy.speedy_physics import SpeedyPhysics
   from jcm.physics.speedy.parameters import Parameters

   # Customize physics parameters
   params = Parameters.default()
   params = params.copy(...)  # modify parameters as needed

   physics = SpeedyPhysics(parameters=params)

   model = Model(
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


Multi-Device Parallelization
-----------------------------

JCM supports multi-device parallelization using JAX's SPMD (Single Program Multiple Data) sharding. This allows you to split computation across multiple GPUs or TPUs for faster execution, especially useful for higher resolution simulations.

Basic Concepts
^^^^^^^^^^^^^^

**SPMD Mesh**: Defines how to partition data across devices. The mesh has three dimensions corresponding to ``(x, y, z)`` or ``(longitude, latitude, vertical)``.

**Sharding Strategy**: Typically, you want to shard the longitude dimension first since it usually has the most grid points.

Enabling Parallelization
^^^^^^^^^^^^^^^^^^^^^^^^

To enable multi-device parallelization, pass ``spmd_mesh`` when creating your geometry:

.. code-block:: python

   import jax
   from jcm.model import Model
   from jcm.geometry import Geometry

   # Check available devices
   print(f"Available devices: {jax.devices()}")
   print(f"Number of devices: {len(jax.devices())}")

   # Create a mesh to split longitude across 4 devices
   # Mesh shape (4, 1, 1) means:
   #   - Split longitude dimension across 4 devices
   #   - Don't split latitude (1)
   #   - Don't split vertical (1)
   mesh = jax.make_mesh((4, 1, 1), ('x', 'y', 'z'))

   # Create geometry with sharding enabled
   geometry = Geometry.from_spectral_truncation(
       spectral_truncation=85,  # Higher resolution benefits from parallelization
       num_levels=8,
       spmd_mesh=mesh,
   )

   # Create and run model as usual
   model = Model(geometry=geometry, time_step=30.0)
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
4. Match mesh to your device count: 4 GPUs → mesh product should equal 4

Using with Terrain Files
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also enable sharding when loading realistic terrain:

.. code-block:: python

   from jcm.forcing import ForcingData

   mesh = jax.make_mesh((4, 1, 1), ('x', 'y', 'z'))

   geometry = Geometry.from_terrain_file(
       terrain_file='jcm/data/bc/terrain_t85.nc',
       spmd_mesh=mesh,
   )

   forcing = ForcingData.from_file('jcm/data/bc/forcing_daily_t85.nc')

   model = Model(geometry=geometry, time_step=30.0)
   predictions = model.run(
       save_interval=5.0,
       total_time=30.0,
       forcing=forcing
   )

Single Device (Default Behavior)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't specify ``spmd_mesh``, JCM runs on a single device:

.. code-block:: python

   # No sharding - runs on single device
   geometry = Geometry.from_spectral_truncation(
       spectral_truncation=31,
       num_levels=8
   )

This is the recommended approach for smaller resolutions (T31, T42) or when you only have a single GPU/TPU available.

Analyzing Output
----------------

The model output is a :py:class:`Predictions` object containing the model state trajectory. Convert it to xarray for analysis:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Convert to xarray Dataset
   ds = model.predictions_to_xarray(predictions)

   # Print variables
   print(ds.data_vars)

   # Plot surface temperature evolution
   ds['temperature'].isel(z=7).mean(dim='lon').plot()
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
- Check example notebooks in the ``notebooks/`` directory
- Read :doc:`developer` for contribution guidelines
