Architecture & Design
=====================

JAX-GCM is designed to be a fully differentiable climate model that balances ease of use for novices with extensibility for experts. This document describes the key architectural decisions and design principles.

Core Architecture
-----------------

Model Structure
^^^^^^^^^^^^^^^

The :py:class:`jcm.model.Model` class serves as the central orchestrator, linking the Dinosaur dynamical core with physics implementations through a clean interface:

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │             Model                       │
   │  ┌───────────────────────────────────┐  │
   │  │   Dinosaur Dynamical Core         │  │
   │  │   (Spectral, Primitive Equations) │  │
   │  └───────────────────────────────────┘  │
   │                  ↕                      │
   │  ┌───────────────────────────────────┐  │
   │  │   Physics Interface               │  │
   │  └───────────────────────────────────┘  │
   │                  ↕                      │
   │  ┌───────────────────────────────────┐  │
   │  │   Physics Implementations         │  │
   │  │   • SpeedyPhysics                 │  │
   │  │   • (Future: ICON, custom, ...)   │  │
   │  └───────────────────────────────────┘  │
   └─────────────────────────────────────────┘

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
           geometry: Geometry,
       ) -> tuple[PhysicsTendency, PhysicsData]:
           """Compute physics tendencies for the current state.

           Args:
               state: Current atmospheric state (temperature, winds, etc.)
               physics_data: Diagnostic data from previous timesteps
               forcing: Boundary conditions (SST, orography, etc.)
               geometry: Grid and coordinate information

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

**Clear Separation**: Each physics term is clearly separated, making the code easy to understand and modify:

.. code-block:: python

   class SpeedyPhysics(Physics):
       def __init__(self, parameters: Parameters = None):
           self.parameters = parameters or Parameters.default()

           # Physics terms are explicit and ordered
           self.terms = [
               compute_convection,
               compute_large_scale_condensation,
               compute_shortwave_radiation,
               compute_longwave_radiation,
               compute_surface_fluxes,
               compute_vertical_diffusion,
           ]

This design makes it easy to:

- Add new physics terms
- Remove or reorder existing terms
- Debug individual components
- Test each term independently

Composability
^^^^^^^^^^^^^

The model is designed to be composable at multiple levels:

**Physics Packages**: Different physics implementations can be easily swapped:

.. code-block:: python

   # Use SPEEDY physics
   model = Model(physics=SpeedyPhysics())

   # Use custom physics (future)
   model = Model(physics=CustomPhysics())

   # Combine multiple physics packages (future)
   model = Model(physics=HybridPhysics([speedy_radiation, ml_convection]))

**Configurations**: Model components can be configured independently:

.. code-block:: python

   geometry = Geometry.from_grid_shape(nodal_shape=(256, 128), num_levels=8)
   physics = SpeedyPhysics(parameters=custom_params)
   
   model = Model(
       geometry=geometry,
       physics=physics,
   )

Differentiability
^^^^^^^^^^^^^^^^^

A core design goal is full differentiability through the model. This enables:

**Gradient-Based Optimization**: Tune parameters using gradients:

.. code-block:: python

   def loss(params):
       physics = SpeedyPhysics(parameters=params)
       model = Model(physics=physics)
       predictions = model.run(...)
       return compute_loss(predictions, observations)

   # Compute gradients with respect to physics parameters
   grad_fn = jax.grad(loss)
   gradients = grad_fn(initial_params)

**Sensitivity Analysis**: Understand how initial conditions affect outcomes:

.. code-block:: python

   def run_model(initial_state):
       model = Model()
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
   model = Model()
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

   # Tests for each physics module
   pytest jcm/physics/speedy/convection_test.py
   pytest jcm/physics/speedy/radiation_test.py
   # ... etc

**Documentation**: All public APIs are documented with clear docstrings.

**Type Hints**: Function signatures use type hints for clarity and IDE support.

**Continuous Integration**: Automated testing ensures changes don't break existing functionality.

Future Directions
-----------------

The architecture is designed to support:

- **Multiple Physics Packages**: ICON physics, custom ML-based physics
- **Hybrid Models**: Combine traditional physics with machine learning
- **Multi-Component Coupling**: Ocean, land surface, chemistry models
- **Ensemble Workflows**: Efficient parallel ensemble generation
- **Adjoint Sensitivity**: Large-scale sensitivity studies
- **Optimization**: Parameter estimation, model calibration

The modular, functional design with clean interfaces makes these extensions straightforward while maintaining the core simplicity of the base model.
