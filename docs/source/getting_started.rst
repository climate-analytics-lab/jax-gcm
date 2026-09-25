Getting Started
===============

This guide walks you from an install to a running simulation and a plot, using
the Python API. By the end you will have run an idealised aquaplanet, run a
validated configuration, and pulled global means out of the output.

.. note::

   To drive the model from the **command line** instead — the
   ``python -m jcm.main`` Hydra CLI, chunked and resumable production runs,
   Docker and batch queues — see :doc:`running_at_scale`. For capabilities
   beyond a first run (nudging, custom steppers, overriding constants) see
   :doc:`advanced_features`.

.. _installation:

Installation
------------

.. code-block:: console

   $ pip install jcm

or, for the development version:

.. code-block:: console

   $ git clone https://github.com/climate-analytics-lab/jax-gcm.git
   $ cd jax-gcm
   $ git switch dev
   $ pip install -e .

Requirements
^^^^^^^^^^^^

- Python ≥ 3.11
- JAX ≥ 0.10, < 0.11 (validated on 0.10.2; the 0.11 line drops the
  ``jax.experimental.hijax`` API that Flax imports, so it is capped in
  ``requirements.txt`` — see #853)
- Dinosaur ≥ 1.5.0 (the dynamical-core backend)
- Flax ≥ 0.12.1
- XArray (for I/O and data handling)
- pySES ≥ 0.1.3.1 for the optional CAM-SE backend (``pip install jcm[pyses]``)

See ``requirements.txt`` for the complete list, and :doc:`v2_to_v3` for why
each of those floors is a floor.

Upgrading from v2? Read :doc:`v2_to_v3` first: v3.0 changes unit contracts,
public APIs, configuration group names and checkpoint compatibility.

Example notebooks
-----------------

If you would rather read working code than prose, the
`notebooks directory <https://github.com/climate-analytics-lab/jax-gcm/tree/main/notebooks>`_
has runnable end-to-end examples:

- `01_jcm_demo <https://github.com/climate-analytics-lab/jax-gcm/blob/main/notebooks/01_jcm_demo.ipynb>`_
  — build a SPEEDY aquaplanet, run it, and plot the output. Start here.
- `02_optimization_example <https://github.com/climate-analytics-lab/jax-gcm/blob/main/notebooks/02_optimization_example.ipynb>`_
  — differentiate through a simulation and tune a physics parameter by
  gradient descent, which is the thing a differentiable GCM is *for*.
- `04_jcm_era5_example <https://github.com/climate-analytics-lab/jax-gcm/blob/main/notebooks/04_jcm_era5_example.ipynb>`_
  — initialise from ERA5 reanalysis and compare the forecast against it.
- `05_jcm_echam_demo <https://github.com/climate-analytics-lab/jax-gcm/blob/main/notebooks/05_jcm_echam_demo.ipynb>`_
  — the full ECHAM physics package on a hybrid vertical grid, the
  configuration the production runs use.

Quick Start Examples
--------------------

Aquaplanet Simulation
^^^^^^^^^^^^^^^^^^^^^

The quickest way to see the model work is an aquaplanet: a water-covered
planet, no orography, zonally symmetric forcing. Nothing to download, and it
runs on a laptop CPU — which makes it the right thing to reach for when you
are learning the model or testing a new physics scheme in isolation.

.. code-block:: python

   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   model = Model(
       coords=get_speedy_coords(),   # T31 (96x48), 8 vertical levels
       time_step=30.0,               # minutes
   )

   predictions = model.run(
       save_interval=10.0,           # save a snapshot every 10 days
       total_time=120.0,             # days
   )

   ds = predictions.to_xarray()
   print(ds)

That is the whole loop: build a model, run it, convert to xarray. The defaults
give you the SPEEDY physics package on a T31 grid with zonally symmetric sea
surface temperatures and no land.

Realistic Simulation
^^^^^^^^^^^^^^^^^^^^

Next, give the model real orography and real boundary conditions. The shape of
the code is the same — you are adding two objects, ``TerrainData`` (what the
surface looks like) and ``ForcingData`` (what it does over time):

.. code-block:: python

   from importlib import resources

   from jcm.forcing import ForcingData
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords
   from jcm.terrain import TerrainData

   coords = get_speedy_coords()

   # Orography and land-sea mask, interpolated onto the T31 grid.
   data_dir = resources.files("jcm.data.bc.t30.clim")
   terrain = TerrainData.from_file(data_dir / "terrain.nc", coords=coords)

   # SST, sea ice, soil moisture and friends. Time-varying fields become
   # TimeSeries leaves and the Model picks the right slice each step.
   forcing = ForcingData.from_file(
       data_dir / "forcing.nc", coords=coords, align_mode="wrap_year")

   model = Model(coords=coords, time_step=30.0, terrain=terrain)

   predictions = model.run(
       forcing=forcing,
       save_interval=5.0,
       total_time=30.0,
   )

   predictions.to_xarray().to_netcdf("output.nc")

``from_file`` treats input timestamps as real dates by default. Select
``align_mode="wrap_year"`` explicitly for a repeating monthly or daily
climatology. All physics packages use the same Gregorian model clock.

.. _configurations-from-python:

Ready-made configurations
-------------------------

A GCM run is not one choice but dozens of co-dependent ones: resolution,
vertical levels, radiation scheme, cloud microphysics, aerosol core, droplet
activation variant, terrain source, sponge, time step. Most combinations are
merely *composable*; only a few have been run for a full year and checked
against observations. Picking them yourself is how you end up debugging a
configuration rather than doing science.

So beyond the simple defaults above, jcm ships **named configurations** that
pin a validated set of those choices, and :func:`jcm.configurations.load`
builds one for you:

.. code-block:: python

   import jcm.configurations as configurations

   configurations.available()        # {name: one-line summary}

   exp = configurations.load("t63-echam-jam")
   predictions = exp.model.run(**exp.run_kwargs)

   # Override any key with Hydra dotted syntax:
   exp = configurations.load("t63-echam-jam", **{"run.total_time": 30})

``exp.model`` is a built :class:`~jcm.model.Model`, ``exp.forcing`` the built
:class:`~jcm.forcing.ForcingData`, and ``exp.config`` a plain resolved dict.
The initial state the configuration calls for is already applied, so
``model.run(**exp.run_kwargs)`` reproduces exactly what the CLI's
``+configuration=<name>`` does — the same recipes, built in-process with Hydra
invisible.

Configurations that select an optional component need its extra installed —
``jcm[mam4]`` for the JAM aerosol recipes, ``jcm[pyses]`` for the CAM-SE
backend, ``jcm[cosp]`` for the COSP variant — and fail with a clear
``ModuleNotFoundError`` at load otherwise.

:doc:`science/configurations` lists every shipped configuration and how far
each has been validated; :doc:`design/packaged_config_tree` describes the tree
itself, and :doc:`running_at_scale` the CLI equivalent.

Fetching input data
-------------------

Real runs need real boundary data: orography, sea surface temperature, ozone,
emissions, dust soil properties. That data is far too large to ship inside the
package, so it lives on a Hugging Face **data mirror** as per-grid bundles and
is fetched on demand into a local cache. One resolution engine serves both the
Python API and the CLI, so the two doors get identical inputs.

:meth:`~jcm.forcing.ForcingData.from_bundles` is that door from Python. It
composes the whole canonical forcing set for a composition in one call:

.. code-block:: python

   from jcm.forcing import ForcingData
   from jcm.initial_states import jw_state
   from jcm.model import Model
   from jcm.physics.echam.echam_levels import get_echam_levels
   from jcm.physics.echam.echam_terms import echam_physics
   from jcm.terrain import TerrainData
   from jcm.utils import get_coords

   coords = get_coords(vertical_coords=get_echam_levels(47),
                       spectral_truncation=63)          # ECHAM T63L47 hybrid

   # JAM reads the 2-moment scheme's process ledger, and RRTMGP consumes the
   # aerosol optics the grey scheme would ignore — so these three go together.
   physics = echam_physics(aerosol_module="jam", cloud_scheme="2m",
                           radiation_scheme="rrtmgp")
   terrain = TerrainData.from_coords(coords)   # flat ocean; terrain_file= for orography
   model = Model(coords=coords, terrain=terrain, physics=physics)

   # Surface, ozone, and the emission / DMS / dust / oxidant set, all on the
   # model grid. aerosol=None gives surface + ozone only.
   forcing = ForcingData.from_bundles(coords, aerosol="jam", surface="pd")

   predictions = model.run(
       initial_state=jw_state(model, rh=0.0),
       forcing=forcing, total_time="365 days", save_interval="1 day",
   )

Use ``from_bundles`` when you are composing your **own** model, as above. It
supplies only the forcing — a validated end-to-end setup such as
``t63-echam-jam`` also pins the radiation scheme, aerosol core, activation
variant, native-grid terrain and sponge — so to reproduce one of those, use
``configurations.load`` from the previous section instead of rebuilding it by
hand.

Two behaviours worth knowing before your first bundle run:

- **Ozone is not silently approximated.** On a hybrid grid, if neither the
  packaged climatology nor a mirror bundle resolves, ``auto`` raises rather
  than falling back to the analytic profile, which is a genuinely low-fidelity
  path. Ask for it explicitly with ``forcing.ozone_file=analytic`` if you want
  it.
- **Dust needs five products, not one**, and the first four are mandatory
  together — setting only ``dust_file`` raises. See
  :doc:`science/boundary_conditions` for what each one carries.
- **Dust's saturation gate reads the surface bundle's relative soil
  wetness** (``soilw_rel``, an ECHAM-like ws/wsmx). A bundle built before that
  channel existed leaves the gate inert, logged rather than silent — rebuild
  the bundle to get the intended gating.

:doc:`design/data_mirror` describes the mirror and the caching;
:doc:`running_at_scale` covers the ``auto`` defaults and ``hf://`` paths from
the CLI.

Customizing the Model
---------------------

Everything above used defaults. Each piece can be swapped independently.

**Resolution.** Pass a different truncation; omit ``time_step`` and the Model
picks a stable one for the grid.

.. code-block:: python

   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords
   from jcm.terrain import TerrainData

   coords = get_speedy_coords(spectral_truncation=85)   # T85, 256x128
   model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords=coords))

``time_step`` (minutes) is optional: an explicit value always wins, otherwise
the Model asks the active physics for a stable one — 12 minutes for ECHAM and
Held-Suarez, 30 for standard SPEEDY. :doc:`running_at_scale` has the full
resolution order, including what happens when you also pass a dycore.

**Physics.** Build a package with your own parameters:

.. code-block:: python

   from jcm.model import Model
   from jcm.physics.speedy.params import Parameters
   from jcm.physics.speedy.speedy_coords import get_speedy_coords
   from jcm.physics.speedy.speedy_terms import speedy_physics

   params = Parameters.default()
   # params = params.replace(...)          # your changes here
   physics = speedy_physics(parameters=params)

   model = Model(coords=get_speedy_coords(), time_step=30.0, physics=physics)

Set parameters **before** building the Model. jcm binds them into the compiled
executable when the physics is first traced, so editing them on a live model is
unreliable — whether a later run sees the edit depends on which compilation
cache it hits. jcm logs a warning and flags ``preds.params`` when it detects
that. To sweep a parameter, build a Model per value *inside* one ``jax.jit``,
so the rebuild is a trace-time cost rather than a recompile per iteration.

**Dynamical core.** ``Model(coords=...)`` builds the Dinosaur backend with
default settings. Pass one explicitly when you need backend-specific
configuration; it then owns the time step.

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
   model = Model(dycore=dycore)          # adopts the dycore's 30-minute step

*Transport scheme.* ``DinosaurDycore(advection=None)`` (the default; Hydra
``dycore.advection=null``) lets the physics choose: SPEEDY runs the Eulerian
spectral core it was formulated on, which on CPU is ~4x faster than
semi-Lagrangian for SPEEDY; ECHAM, JAM, Held–Suarez and any composition that
carries extra tracers run semi-Lagrangian. Pass ``advection="semi_lagrangian"``
or ``"eulerian"`` to force one — Eulerian with tracer-carrying physics runs
but warns, since it rings negative on sharp tracer fields. See
:doc:`design/dinosaur_transport_selection`.

**Initial conditions.** For the common starting states there are ready-made
builders in :mod:`jcm.initial_states` — the same ones the CLI's ``init`` group
exposes. Each returns a state; hand it to ``run`` as ``initial_state=``:

.. code-block:: python

   from jcm.initial_states import jw_state

   predictions = model.run(
       initial_state=jw_state(model, rh=0.6),
       total_time=10.0, save_interval=1.0,
   )

The others follow the same pattern:
:func:`~jcm.initial_states.balanced_isothermal_state` (a 288 K rest state,
robust for moist physics over real terrain),
:func:`~jcm.initial_states.era5_state` ``(coords, date)`` (from a
WeatherBench2 ERA5 slice), and
:func:`~jcm.initial_states.checkpoint_state` ``(model, path)`` (a warm start
from a saved state, skipping the ~9-month spin-up). You can also pass a
:class:`~jcm.physics_interface.PhysicsState` you built yourself.

.. note::

   A state file written before jcm 3.0 is **refused**: it carries no schema
   stamp, so nothing records which unit convention its numbers follow. See
   :doc:`design/checkpoint_compatibility` for the escape hatch, and
   :doc:`v2_to_v3` if you are carrying state across the v2/v3 boundary.

**Logging.** jcm is a library, so it configures no logging — no handlers, no
levels, no ``basicConfig``. To see its output from your own code::

    import logging
    logging.basicConfig()                            # a handler, once, for your app
    logging.getLogger("jcm").setLevel(logging.INFO)  # then jcm's verbosity

Both lines are needed: a level alone creates no handler, and with none
configured records fall through to Python's last-resort handler, whose level is
``WARNING``. That is why warnings reach you with no setup and INFO does not.
Under the CLI, jcm *is* the application and does configure logging — see
:doc:`running_at_scale` for ``run.log_level``.

Analyzing Output
----------------

``model.run`` returns a predictions object holding the state trajectory.
Convert it to xarray and everything from there is ordinary xarray:

.. code-block:: python

   import matplotlib.pyplot as plt

   ds = predictions.to_xarray()
   print(ds.data_vars)

   # Output is surface-first, so level=0 is the level nearest the ground.
   ds['temperature'].isel(level=0).mean(dim='lon').plot()
   plt.title('Zonal Mean Surface Temperature')
   plt.show()

Post-processing with ``jcm.analysis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:mod:`jcm.analysis` is the one home for the recipes that otherwise get
re-implemented in every analysis script — area weights, global means, layer
thicknesses and column burdens, all computed on *saved* netCDF output:

.. code-block:: python

   import xarray as xr
   from jcm import analysis

   ds = xr.open_dataset('output.nc')

   # Area-weighted global mean. On a dinosaur output grid this uses the exact
   # Gauss-Legendre quadrature weights, not the cos(lat) approximation, so
   # conservation residuals really do integrate to zero.
   T_global = analysis.global_mean(ds['temperature'])

   # Column burden [kg/m^2], mass-weighted with the file's own layer thickness.
   qc_burden = analysis.column_burden(ds, 'qc')

   # column_burden time-broadcasts, so a global-mean burden time series is:
   burden_ts = analysis.global_mean(analysis.column_burden(ds, 'qc'))

Prefer these over hand-rolled weights: :func:`~jcm.analysis.area_weights`
returns a dims-only ``DataArray`` so ``.weighted()`` broadcasts by dimension
name without float32/float64 coordinate-alignment surprises, and the column
integrals use the model's own ``g`` rather than a hardcoded 9.81.

Vertical coordinates in the output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Output files carry two vertical axes: ``level`` (``nlev`` layer mid-levels —
temperature, tracers, ``pressure_full``) and ``level_i`` (``nlev+1``
interfaces — ``pressure_half``, radiative fluxes). **Both run surface-first**,
so index 0 is nearest the ground and ``level[k]`` sits between ``level_i[k]``
and ``level_i[k+1]``.

To mass-weight a ``level`` field by hand, use ``pressure_thickness`` [Pa],
which the ECHAM stacks write directly on the ``level`` axis:

.. code-block:: python

   burden = (ds['qc'] * ds['pressure_thickness'] / 9.81).sum('level')  # kg/m^2

For a SPEEDY run, or a file written before ``pressure_thickness`` existed,
reconstruct Δp from ``pressure_half``. Mixing the two axes is safe because both
run surface-first, but the rename needs care:

.. code-block:: python

   # diff() keeps the *interface* labels, so the mid-level coordinate must be
   # assigned explicitly — otherwise xarray's alignment finds no matching
   # labels and the product is silently empty.
   dp = (-ds['pressure_half'].diff('level_i')
         .rename(level_i='level').assign_coords(level=ds['level']))
   burden = (ds['qc'] * dp / 9.81).sum('level')     # kg/m^2

Both axes are CF-labelled nominal sigma with ``positive = "down"``, and the
hybrid ``(a, b)`` tables travel with the file, so ``p = a + b * p_s`` is
reproducible from the file alone. See
:doc:`design/output_vertical_conventions`, including for how to read files
written before this convention was unified.

Next Steps
----------

- :doc:`advanced_features` — nudging, custom steppers, calendar-aware
  durations, long forcing records, overriding constants.
- :doc:`running_at_scale` — the Hydra CLI, chunked production runs, Docker and
  batch queues.
- :doc:`design` — how the model is put together.
- :doc:`api` — the full API reference.
