Architecture & Design
=====================

JAX-GCM is designed to be a fully differentiable climate model that balances ease of use for novices with extensibility for experts. This document is the high-level architectural overview; the design references below cover the same machinery in depth.

.. toctree::
   :maxdepth: 1
   :caption: In-depth design references

   design/composable_physics
   design/operator_split_physics
   design/water_positivity_conservation
   design/writing_a_physics_scheme
   design/surface_exchange
   design/forcing_time_semantics
   design/parallelization
   design/output_vertical_conventions
   design/checkpoint_compatibility
   design/pyses_cam_se_dycore
   design/dinosaur_sl_jam_configuration
   design/jam
   design/jam_carbon_aging
   design/jam_aerosol_removal
   design/speedy_variable_levels
   design/frontal_gravity_wave_drag
   design/lohmann_2m_column_processes
   design/convective_trigger_soundings
   design/aerosol_optics_diagnostics
   design/jam_optics_mode_seam
   design/radiation_nn_emulator
   design/aerocom_erfari_sampling
   design/jam_regression
   design/cloud_cover_gate
   design/data_mirror
   design/packaged_config_tree
   design/science_register_enforcement
   design/test_suite_memory
   design/observers
   design/pyses_performance_review

Core Architecture
-----------------

Model Structure
^^^^^^^^^^^^^^^

The :py:class:`jcm.model.Model` class is the central orchestrator. It links a
pluggable dynamical-core backend to a physics package without reaching into the
backend's native state representation. See :doc:`design/operator_split_physics`
for the per-step coupling between the dycore and physics.

.. code-block:: text

   ┌───────────────────────────────────────────────┐
   │ Model                                         │
   │                                               │
   │   DynamicalCore                               │
   │   native state ↔ gridpoint PhysicsState       │
   │          │                                    │
   │          ▼                                    │
   │   physics_interface                           │
   │   verify state → tendencies → verify tendency │
   │          │                                    │
   │          ▼                                    │
   │   ComposablePhysics                           │
   │   ordered PhysicsTerm modules                 │
   └───────────────────────────────────────────────┘

Pluggable Dynamical Cores
^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`jcm.dycore.base.DynamicalCore` protocol owns every operation
that depends on a dycore's native representation: initial-state construction,
one-step integration, terrain preparation, conversion to gridpoint
:py:class:`~jcm.physics_interface.PhysicsState`, simulation-time accounting,
and xarray output. The shipped
:py:class:`jcm.dycore.dinosaur.dycore.DinosaurDycore` backend wraps Dinosaur's
spectral primitive-equation state, its two-time-level semi-Lagrangian
semi-implicit Crank–Nicolson RK2 step, and its spectral filters. Tracer
transport on that backend is semi-Lagrangian only — see
:doc:`science/dynamical_core` for the integrator and transport contract. The
optional :py:class:`jcm.dycore.pyses.PysesCamSEDycore` backend
(``pip install jcm[pyses]``, registry name ``"pyses_cam_se"``) wraps the
pySES CAM-SE spectral-element core on the cubed sphere, coupling to the
column physics through a pg2 finite-volume physics grid — see
:doc:`design/pyses_cam_se_dycore`.

The physics-dynamics boundary is gridpoint space on the dycore's native
horizontal layout, so physics never sees spectral coefficients. The protocol
permits non-lat/lon layouts, although the current column-vectorized
``ComposablePhysics`` path still expects exactly two horizontal dimensions;
a backend with an element-plus-GLL layout must flatten or adapt that layout
before using the shipped column physics packages. Backends perform any output
regridding in ``to_xarray()``, outside the per-step physics path.

For convenience, ``Model(coords=...)`` constructs ``DinosaurDycore``
automatically. Expert callers can construct and pass a backend explicitly:

.. code-block:: python

   from jcm.dycore.dinosaur import DinosaurDycore
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords
   from jcm.terrain import TerrainData

   coords = get_speedy_coords()
   dycore = DinosaurDycore(
       coords=coords,
       terrain=TerrainData.aquaplanet(coords),
       dt_seconds=1800.0,
   )
   model = Model(dycore=dycore, time_step=30.0)

The Hydra CLI selects the backend through the ``dycore`` config group, which
``jcm.runners.build_model()`` dispatches on: ``dycore=dinosaur`` (the
default) or one of the canonical pySES configurations. A whole pySES run is
therefore a single command:

.. code-block:: console

   $ python -m jcm.main +configuration=ma-ne30-l47

That packaged preset (see ``jcm/config/configuration/``) pins the backend, the
matching forcing bundles and the one physics switch pySES needs. Composing the
groups by hand works too, but the switch is not optional — the backend exposes
no ``omega``, so ECHAM's mid-level convection trigger has to be turned off or
``Model`` construction refuses to build (#698):

.. code-block:: console

   $ python -m jcm.main dycore=pyses_ne30l47 physics=echam-rrtmgp-2m run=pyses_year \
       +physics.terms.tiedtke_convection.params.cu_lmfmid=false

The spelling of that override follows the physics group's own shape: the
term-list presets take ``+physics.terms.tiedtke_convection.params.cu_lmfmid=false``
as above, while the factory-built ones (``physics=echam-jam*``, which set
``builder: echam_physics``) take the scalar ``+physics.cu_lmfmid=false`` and
reject a ``terms`` node outright.

The ``dycore`` group owns what is backend-specific: on ``pyses_*`` the
resolution comes from ``nx``/``npt``/``nlev`` in that group and the ``grid``
group is ignored, and the Model adopts the group's ``dt_seconds`` rather than
``run.time_step``. Each shipped ``dycore/*.yaml`` documents its own settings
and limits.

An explicitly-constructed backend owns the time step: ``Model`` adopts its
``dt_seconds`` when ``time_step`` is omitted, and raises on a conflicting
explicit ``time_step``. On the ``coords`` path the Model builds the backend
itself and, when ``time_step`` is omitted, consults the active physics'
``stable_time_step_minutes`` for grid-dependent stability limits (see
:doc:`design/speedy_variable_levels`).

The Physics Interface
^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`jcm.physics_interface.Physics` base class defines the contract
between the gridpoint state supplied by the dycore and a physics package:

.. code-block:: python

   class Physics:
       def compute_tendencies(
           self,
           state: PhysicsState,
           forcing: ForcingData,
           terrain: TerrainData,
           prev_physics_data=None,
       ) -> tuple[PhysicsTendency, PhysicsCarryState]:
           """Compute physics tendencies for the current state.

           Args:
               state: Current atmospheric state (temperature, winds, etc.)
               forcing: Boundary conditions for the *current step*. The
                   Model collapses every `TimeSeries` leaf and populates
                   `forcing.solar` via ``forcing.select(date)``
                   before this call, so physics terms see only flat 2-D
                   spatial fields and a precomputed `SolarGeometry` —
                   no time axis, no `DateData`.
               terrain: Orography / land-sea mask information
               prev_physics_data: Cross-step physics carry from the
                   preceding timestep.

           Returns:
               tendencies: Changes to apply to the state
               updated_data: Updated diagnostics and cross-step carry
           """
           raise NotImplementedError

This interface enables:

- **Modularity**: Swap physics packages without changing the dynamical core
- **Composability**: Combine different physics implementations
- **Testability**: Test physics in isolation from dynamics

Design Principles
-----------------

Functional Programming Paradigm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The physics code follows functional programming principles:

**Pure Functions**: Each physics term (convection, radiation, etc.) is a pure function that takes inputs and returns outputs without side effects:

.. code-block:: python

   def compute_convection(
       state: PhysicsState,
       diagnostics: dict,
       parameters: Parameters,
   ) -> tuple[PhysicsTendency, dict]:
       """Pure function computing convective tendencies."""
       # No global state, no mutations
       tendencies = ...
       diagnostics = ...
       return tendencies, diagnostics

**Clear Separation**: Each physics term is clearly separated, making the code easy to understand and modify. The ``speedy_physics()`` factory builds an ordered list of ``PhysicsTerm`` instances:

.. code-block:: python

   def speedy_physics(parameters: Parameters | None = None) -> ComposablePhysics:
       params = parameters or Parameters.default()
       return ComposablePhysics(terms=[
           SpeedyFlags(),
           SpeedyForcing(...),
           SpeedyConvection(params.convection),
           SpeedyLargeScaleCondensation(params.condensation),
           SpeedyShortwaveRadiation(params.shortwave_radiation),
           SpeedyDownwardLongwaveRadiation(...),
           SpeedySurfaceFlux(params.surface_flux),
           SpeedyUpwardLongwaveRadiation(...),
           SpeedyVerticalDiffusion(params.vertical_diffusion),
       ])

This design makes it easy to:

- Add new physics terms
- Remove or reorder existing terms
- Debug individual components
- Test each term independently

Composability
^^^^^^^^^^^^^

The model is composable at multiple levels through the ``ComposablePhysics`` framework. See :doc:`design/composable_physics` for the contract, validation rules, diagnostics-dict convention, and differentiability patterns; :doc:`design/writing_a_physics_scheme` for a single-file plugin walkthrough.

**Composable Physics**: Individual parameterizations (``PhysicsTerm`` instances) can be mixed across packages:

.. code-block:: python

   from jcm.physics.speedy.speedy_terms import speedy_physics
   from jcm.physics.echam.echam_terms import echam_physics
   from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
       ConvectionParameters,
       TiedtkeConvection,
   )

   # Use pre-built SPEEDY defaults
   physics = speedy_physics()

   # Use ECHAM with the NN radiation emulator
   physics = echam_physics(radiation_scheme="emulated")

   # Replace one term with a separately configured instance
   convection = TiedtkeConvection(
       params=ConvectionParameters.default(tau=10800.0),
   )
   physics = echam_physics().replace("convection", convection)

   # Remove a term
   physics = echam_physics().remove("hines")

Each ``PhysicsTerm`` is a ``flax.nnx.Module`` that stores its own tunable parameters as ``nnx.Param`` attributes and coordinate caches as ``nnx.Variable``. Terms communicate through a ``diagnostics`` dict threaded through the term list. The dict serves a dual role: keys without a leading underscore are exposed as user-facing diagnostic output (written to xarray); keys prefixed with ``_`` (e.g. ``_radiation``, ``_convection``) are internal inter-term state and are filtered out of the user-facing output.

**Configurations**: Model components can be configured independently:

.. code-block:: python

   coords = get_speedy_coords(nodal_shape=(256, 128), layers=8, spectral_truncation=85)
   terrain = TerrainData.from_coords(coords)
   physics = speedy_physics(parameters=custom_params)

   model = Model(
       coords,
       terrain=terrain,
       physics=physics,
   )

Differentiability
^^^^^^^^^^^^^^^^^

A core design goal is full differentiability through the model. This enables:

**Gradient-Based Optimization**: Tune parameters using gradients:

.. code-block:: python

   def loss(params):
       physics = speedy_physics(parameters=params)
       model = Model(coords=get_speedy_coords(), physics=physics)
       predictions = model.run(...)
       return compute_loss(predictions, observations)

   grad_fn = jax.grad(loss)
   gradients = grad_fn(initial_params)

**Per-Scheme Optimization** (using ``nnx.grad`` to differentiate w.r.t.
individual term parameters):

.. code-block:: python

   from flax import nnx

   physics = speedy_physics()
   physics.cache_coords(coords)

   def loss_fn(physics):
       model = Model(coords=coords, terrain=terrain, physics=physics)
       return compute_loss(model.run(total_time=...))

   # Gradient w.r.t. all physics parameters
   grads = nnx.grad(loss_fn)(physics)

   # Gradient w.r.t. convection parameters only
   convection_filter = nnx.PathContains("convection")
   grads = nnx.grad(loss_fn, wrt=convection_filter)(physics)

**Sensitivity Analysis**: Understand how initial conditions affect outcomes:

.. code-block:: python

   def run_model(initial_state):
       model = Model(coords=get_speedy_coords())
       return model.run(initial_state=initial_state, total_time=1.0)

   # Gradients with respect to initial conditions
   sensitivity = jax.grad(run_model)

**Data Assimilation**: Incorporate observations using gradient-based methods.

**Coupling**: Enable differentiable coupling between atmosphere and other Earth system components (ocean, land, chemistry).

All code is written to be compatible with JAX transformations:

- **JIT Compilation**: Entire model can be JIT compiled for performance
- **Automatic Differentiation**: Forward and reverse mode AD through all operations
- **Vectorization**: Batch multiple runs efficiently with ``vmap``

JAX Compatibility
^^^^^^^^^^^^^^^^^

The codebase uses JAX-compatible data structures and operations:

**Immutable Structures**: Data classes using ``tree_math.struct`` or ``dataclasses``:

.. code-block:: python

   @tree_math.struct
   class PhysicsState:
       temperature: jnp.ndarray
       u_wind: jnp.ndarray
       v_wind: jnp.ndarray
       specific_humidity: jnp.ndarray
       # ... other fields

**Pure Transformations**: State updates return new objects rather than mutating:

.. code-block:: python

   # Good: Returns new state
   new_state = state.replace(temperature=state.temperature + dt * tendency)

   # Bad: Would mutate (not JAX compatible)
   # state.temperature += dt * tendency

**Static Shapes**: Array shapes are known at compile time for efficient JIT compilation.

Ease of Use
-----------

For Novices
^^^^^^^^^^^

The default configuration provides a working model out of the box:

.. code-block:: python

   # Just works - sensible defaults for everything
   model = Model(coords=get_speedy_coords())
   predictions = model.run(total_time="10 days")

For Experts
^^^^^^^^^^^

Every component can be customized or extended:

- **Custom Physics**: Add a new ``PhysicsTerm`` — see :doc:`design/writing_a_physics_scheme` for the one-file plugin contract.
- **Custom Dynamical Core**: Implement the :py:class:`jcm.dycore.base.DynamicalCore` protocol and pass the backend to ``Model(dycore=...)``.
- **Custom Forcing**: Create specialized boundary condition handlers
- **Custom Diagnostics**: Add new output variables and computations
- **Integration**: Couple with other models or ML components

Code Quality
------------

The codebase maintains high standards to support future complexity:

**Testing**: High unit test coverage ensures correctness:

.. code-block:: bash

   # Tests are co-located with source in process directories
   pytest jcm/physics/convection/speedy_convection_test.py
   pytest jcm/physics/radiation/speedy_shortwave_test.py
   pytest jcm/physics/radiation/grey_two_stream/radiation_scheme_test.py
   # ... etc

**Documentation**: All public APIs are documented with clear docstrings.

**Type Hints**: Function signatures use type hints for clarity and IDE support.

**Continuous Integration**: Automated testing ensures changes don't break existing functionality.

Physics Directory Organization
-------------------------------

Physics code is organized by **physical process**, with files named after the
**scheme** rather than the model they were ported from. New schemes drop in
beside existing ones without nesting:

.. code-block:: text

   jcm/physics/
   ├── radiation/
   │   ├── grey_two_stream/      # fast grey two-stream package
   │   ├── rrtmgp.py             # RRTMGP wrapper
   │   ├── nn_emulator.py        # NN radiation emulator
   │   ├── speedy_shortwave.py
   │   └── speedy_longwave.py
   ├── convection/
   │   ├── tiedtke_nordeng/      # Tiedtke-Nordeng mass flux
   │   └── speedy_convection.py
   ├── clouds/
   │   ├── sundqvist.py          # Sundqvist diagnostic cloud fraction
   │   ├── echam_1m.py           # ECHAM 1-moment microphysics
   │   ├── speedy_humidity.py
   │   └── speedy_condensation.py
   ├── vertical_diffusion/
   │   ├── tte_tke/              # TTE-TKE closure
   │   └── speedy_vdiff.py
   ├── gravity_waves/             # hines/ (Hines 1997), sso/ (Lott-Miller 1997), simple/
   ├── aerosol/macv2_sp.py       # Stevens MACv2-SP simple plumes
   ├── chemistry/simple_chemistry.py
   ├── surface/                  # SPEEDY and ECHAM surface schemes
   ├── speedy/                   # SPEEDY infrastructure (params, coords)
   └── echam/                    # ECHAM infrastructure (params, coords)

Model-specific *infrastructure* (parameter containers, coordinate caches,
data structs) lives under ``speedy/`` and ``echam/``. Everything else is
named after the scheme so an ECHAM port and a CAM port of the same
parameterization sit side-by-side without per-model subfolders.

Future Directions
-----------------

The composable architecture is designed to support:

- **Hybrid Models**: Combine traditional physics with machine learning — a neural network ``PhysicsTerm`` slots into the composable term list and automatically participates in gradient computation
- **Multi-Component Coupling**: Ocean, land surface, chemistry models
- **Ensemble Workflows**: Efficient parallel ensemble generation with ``vmap``
- **Adjoint Sensitivity**: Large-scale sensitivity studies through end-to-end differentiability
- **Parameter Estimation**: Per-scheme gradient-based calibration using ``nnx.grad``
- **New Parameterizations**: Add new schemes (e.g., Betts-Miller convection) as ``PhysicsTerm`` subclasses that drop into existing workflows

The modular, functional design with clean interfaces makes these extensions straightforward while maintaining the core simplicity of the base model.
