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

   # Higher resolution: T85 (256x128 grid)
   geometry = Geometry.from_spectral_truncation(spectral_truncation=85)

   model = Model(
       time_step=20.0,  # smaller timestep for stability
       geometry=geometry
   )

**Physics**: Use different physics packages or configurations

.. code-block:: python

   from jcm.physics.speedy.speedy_physics import SpeedyPhysics
   from jcm.physics.speedy.params import Parameters

   # Customize physics parameters
   params = Parameters.default()
   params = params.replace(...)  # modify parameters as needed

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
- Check example notebooks in the ``notebooks/`` directory
- Read :doc:`developer` for contribution guidelines
