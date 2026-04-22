Architecture & Design
=====================

JAX-GCM is designed to be a fully differentiable climate model that balances ease of use for novices with extensibility for experts. This document describes the key architectural decisions and design principles.

Core Architecture
-----------------

Model Structure
^^^^^^^^^^^^^^^

The :py:class:`jcm.model.Model` class serves as the central orchestrator, linking the Dinosaur dynamical core with physics implementations through a clean interface:

.. code-block:: text

   ┌──────────────────────────────────────────────┐
   │             Model                            │
   │  ┌────────────────────────────────────────┐  │
   │  │   Dinosaur Dynamical Core              │  │
   │  │   (Spectral, Primitive Equations)      │  │
   │  └────────────────────────────────────────┘  │
   │                  ↕                           │
   │  ┌────────────────────────────────────────┐  │
   │  │   Physics Interface                    │  │
   │  │   (PhysicsState ↔ PhysicsTendency)     │  │
   │  └────────────────────────────────────────┘  │
   │                  ↕                           │
   │  ┌────────────────────────────────────────┐  │
   │  │   ComposablePhysics                    │  │
   │  │   (ordered list of PhysicsTerm)        │  │
   │  │   built by speedy_physics(),           │  │
   │  │   icon_physics(), held_suarez_physics()│  │
   │  └────────────────────────────────────────┘  │
   └──────────────────────────────────────────────┘

The Physics Interface
^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`jcm.physics_interface.Physics` abstract base class defines a clean contract between the dynamical core and physics packages:

.. code-block:: python

   class Physics:
       def __call__(
           self,
           state: PhysicsState,
           physics_data: PhysicsData,
           forcing: ForcingData,
           terrain: TerrainData,
       ) -> tuple[PhysicsTendency, PhysicsData]:
           """Compute physics tendencies for the current state.

           Args:
               state: Current atmospheric state (temperature, winds, etc.)
               physics_data: Diagnostic data from previous timesteps
               forcing: Boundary conditions (SST, orography, etc.)
               terrain: Orography/terrain information

           Returns:
               tendencies: Changes to apply to the state
               updated_data: Updated diagnostic information
           """
           pass

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
       physics_data: PhysicsData,
       parameters: Parameters,
   ) -> tuple[PhysicsTendency, ConvectionData]:
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
           SpeedyCondensation(params.condensation),
           SpeedyShortwaveRadiation(params.shortwave_radiation),
           SpeedyLongwaveRadiation(...),
           SpeedySurfaceFlux(params.surface_flux),
           SpeedyVerticalDiffusion(params.vertical_diffusion),
       ])

This design makes it easy to:

- Add new physics terms
- Remove or reorder existing terms
- Debug individual components
- Test each term independently

Composability
^^^^^^^^^^^^^

The model is composable at multiple levels through the ``ComposablePhysics`` framework.

**Composable Physics**: Individual parameterizations (``PhysicsTerm`` instances) can be mixed across packages:

.. code-block:: python

   from jcm.physics.speedy.speedy_terms import speedy_physics
   from jcm.physics.icon.icon_terms import icon_physics, IconRadiationRRTMGP

   # Use pre-built SPEEDY defaults
   physics = speedy_physics()

   # Use ICON with NN radiation emulator
   physics = icon_physics(radiation_scheme="emulated")

   # Replace SPEEDY's shortwave radiation with an ICON scheme
   physics = speedy_physics().replace("radiation_sw", IconRadiationRRTMGP())

   # Remove a term
   physics = icon_physics().remove("gravity_waves")

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
       return model.run(initial_state=initial_state, ...)

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
   predictions = model.run()

For Experts
^^^^^^^^^^^

Every component can be customized or extended:

- **Custom Physics**: Implement the ``Physics`` interface for new parameterizations
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
   pytest jcm/physics/radiation/icon/radiation_test.py
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
   │   ├── grey_two_stream/      # ICON-style grey two-stream package
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
   ├── gravity_waves/hines/
   ├── aerosol/macv2_sp.py       # Stevens MACv2-SP simple plumes
   ├── chemistry/simple_chemistry.py
   ├── surface/                  # speedy + icon (multi-tile bundle in icon/)
   ├── speedy/                   # SPEEDY infrastructure (params, coords)
   └── icon/                     # ICON infrastructure (params, coords)

Model-specific *infrastructure* (parameter containers, coordinate caches,
data structs) lives under ``speedy/`` and ``icon/``. Everything else is
named after the scheme so an "ICON" port and a "CAM" port of the same
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
