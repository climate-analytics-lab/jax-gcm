v1 to v2 Migration Guide
========================

This short guide covers the API changes most likely to affect existing v1
scripts and notebooks.

Install the Beta
----------------

Pin the beta explicitly:

.. code-block:: console

   $ pip install "jcm==2.0.0b1"

or opt into pre-releases generally:

.. code-block:: console

   $ pip install --pre --upgrade jcm

Model Construction
------------------

Most simple SPEEDY scripts still construct a model from coordinates:

.. code-block:: python

   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   coords = get_speedy_coords()
   model = Model(coords=coords, time_step=30.0)

The old geometry-style construction has been replaced by explicit coordinates
and terrain:

.. code-block:: python

   from jcm.model import Model
   from jcm.terrain import TerrainData
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   coords = get_speedy_coords()
   terrain = TerrainData.aquaplanet(coords)
   model = Model(coords=coords, terrain=terrain)

If you need backend-specific configuration, construct the Dinosaur dycore
explicitly. ``dt_seconds`` is in seconds; ``Model(time_step=...)`` is in
minutes, so the two values must represent the same duration.

.. code-block:: python

   from jcm.dycore.dinosaur import DinosaurDycore
   from jcm.model import Model

   dycore = DinosaurDycore(coords=coords, terrain=terrain, dt_seconds=1800.0)
   model = Model(dycore=dycore, time_step=30.0)

Output Conversion
-----------------

``Model.run()``, ``Model.resume()``, and ``Model.run_from_state()`` return a
``ModelPredictions`` wrapper. Convert it directly:

.. code-block:: python

   predictions = model.run(save_interval=1.0, total_time=10.0)
   ds = predictions.to_xarray()

Do not pass the physics package to ``to_xarray``.

Composable Physics
------------------

Physics packages are now assembled from ``PhysicsTerm`` instances. SPEEDY keeps
its package-level ``Parameters`` object:

.. code-block:: python

   from jcm.physics.speedy.params import Parameters
   from jcm.physics.speedy.speedy_terms import speedy_physics

   params = Parameters.default()
   physics = speedy_physics(parameters=params)

ECHAM uses per-scheme parameter structs rather than a monolithic ECHAM
``Parameters`` object:

.. code-block:: python

   from jcm.physics.echam.echam_terms import echam_physics
   from jcm.physics.convection.tiedtke_nordeng import ConvectionParameters

   convection = ConvectionParameters.default(tau=10800.0)
   physics = echam_physics(convection=convection, radiation_scheme="rrtmgp")

Swap or remove process terms through the composable API:

.. code-block:: python

   physics = echam_physics().remove("hines")
   physics = echam_physics(radiation_scheme="emulated")

For RRTMGP production runs, the CLI path remains the recommended default:

.. code-block:: console

   $ python -m jcm.main physics=echam-rrtmgp grid=echam_t63_l47_hybrid

Forcing and Calendars
---------------------

The model selects time-varying forcing before physics runs. Physics terms no
longer receive ``DateData`` directly. Use ``ForcingData.from_file(...,
coords=coords)`` and pass the result to ``Model.run`` or ``Model.resume``.

``save_interval`` and ``total_time`` accept day counts or calendar strings:

.. code-block:: python

   predictions = model.run(save_interval="1 day", total_time="1 year")

Physical Constants
------------------

Shared physical constants now live in :mod:`jcm.constants`. To override them,
call :func:`jcm.constants.set_constants` before constructing the model:

.. code-block:: python

   import jcm.constants as constants

   constants.set_constants(grav=9.80665, rearth=6.371229e6)
   model = Model(coords=coords)

Read constants through module attribute access, for example
``constants.grav``. Avoid ``from jcm.constants import grav`` in code that needs
to honour runtime overrides.

