Developer Guide
===============

Contributing to JAX-GCM
-----------------------

We welcome contributions to JAX-GCM! Whether you're fixing bugs, adding features, improving documentation, or expanding the physics packages, your help is appreciated.

Getting Started
^^^^^^^^^^^^^^^

1. **Find or Create an Issue**

   - Check the `GitHub Issues <https://github.com/climate-analytics-lab/jax-gcm/issues>`_ for existing work
   - Pick up an existing issue or create a new one describing what you'd like to work on
   - Assign yourself to the issue to let others know you're working on it

2. **Fork and Clone**

   .. code-block:: console

      $ git clone https://github.com/your-username/jax-gcm.git
      $ cd jax-gcm
      $ pip install -e .

3. **Create a Branch**

   .. code-block:: console

      $ git checkout -b fix-issue-123

Issue Management
^^^^^^^^^^^^^^^^

Good issue management helps everyone stay coordinated:

- **Keep Issues Updated**: If you make progress on an issue, add a comment. If you get stuck or need help, mention it.
- **Assign Yourself**: When you start working on an issue, assign yourself. When you stop, unassign yourself.
- **Be Specific**: When creating issues, clearly describe the problem or feature request with examples if possible.

Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^

Submitting Quality PRs
"""""""""""""""""""""""

- **One Issue Per PR**: Keep pull requests focused on a single issue or feature
- **Small is Beautiful**: Smaller, incremental changes are easier to review and merge
- **Link to Issues**: Every PR should reference an issue that explains *why* the change is needed
- **Write Tests**: Except for documentation changes, PRs should include tests that:

  - Demonstrate the issue (if it's a bug fix)
  - Show that the issue is now fixed
  - Cover the new functionality (if it's a feature)

PR Checklist
""""""""""""

Before submitting your PR, ensure:

.. code-block:: text

   ☐ Code follows the existing style and conventions
   ☐ New tests are added and all tests pass
   ☐ Documentation is updated if needed
   ☐ The PR description clearly explains what and why
   ☐ The PR is linked to a relevant issue
   ☐ Code is rebased on the latest dev branch

Testing Your Changes
^^^^^^^^^^^^^^^^^^^^^

Run the test suite to ensure your changes don't break existing functionality:

.. code-block:: console

   # Run all tests
   $ pytest

   # Run specific test file
   $ pytest jcm/model_test.py

   # Run only fast tests (skip slow integration tests)
   $ pytest -m "not slow"

   # Match the CI fast-test coverage gate
   $ pytest -m "not slow" --cov=jcm --cov-fail-under=90

   # Run the linter
   $ ruff check .

Write tests for your changes in the appropriate test file (e.g., ``jcm/module_name_test.py``). We aim for high unit test coverage to support the increasing complexity of physics going forward.

Code Quality
^^^^^^^^^^^^

We strive for high-quality, maintainable code:

- **Functional Design**: Follow the functional programming paradigm used in the physics code. This makes individual physics terms clear and composable.
- **Type Hints**: Add type hints to function signatures where appropriate.
- **Documentation**: Add docstrings to public functions and classes using NumPy style.
- **JAX Compatibility**: Ensure code is compatible with JAX transformations (jit, grad, vmap).

Example of well-documented function:

.. code-block:: python

   def compute_temperature_tendency(
       state: PhysicsState,
       parameters: Parameters
   ) -> jnp.ndarray:
       """Compute temperature tendency from heating rates.

       Args:
           state: Current physics state containing temperature and pressure.
           parameters: Model parameters for physics calculations.

       Returns:
           Temperature tendency array of shape (levels, lon, lat).
       """
       # Implementation here
       pass

Development Tips
----------------

JAX Considerations
^^^^^^^^^^^^^^^^^^

When writing code for JAX-GCM, keep in mind:

- **Pure Functions**: Functions should be pure (no side effects) to work with JAX transformations
- **Immutable Data**: Use ``tree_math.struct`` for data structures
- **No Python Control Flow**: Use ``jax.lax.cond`` instead of ``if`` in JIT-compiled code
- **Static Shapes**: Array shapes should be statically known where possible

See :doc:`jax_gotchas` for more details.

Profiling
^^^^^^^^^

Where a timestep's time goes
""""""""""""""""""""""""""""

For the usual question — *which physics term is this configuration spending its
time in?* — use the ready-made tool rather than a hand-rolled trace:

.. code-block:: console

   $ python tools/profile_terms.py --preset ma-t63-l47 --gpu 3

It runs the preset twice to warm up (both discarded), traces a third run, and
prints milliseconds per step for the dynamical core, each direction of the
dynamics/physics bridge, and every physics term individually, alongside the
fraction of device time it could not attribute cleanly.

The default window is two radiation sub-cycles. Resist lengthening it: the
profiler's event buffer holds about a million events and a T63L47 JAM step
emits ~19,000 kernels, so a longer window overflows it and undercounts. The
short window loses nothing, because kernel shapes are static and the model
takes no data-dependent branches, so a step's cost does not vary with the
state.

The attribution works because :mod:`jcm.profiling` wraps the dycore call, the
bridge and each :class:`~jcm.physics.physics_term.PhysicsTerm` in a
:func:`jax.named_scope`. XLA records that scope in the ``op_name`` metadata of
every instruction traced inside it, and the profiler reports the instruction
behind each GPU kernel, so kernel time joins back to the component that emitted
it. The scopes are always on and cost nothing at runtime. A new driver-level
stage that deserves its own line in the report needs a scope adding there;
individual physics terms need no changes, since the term loop already labels
them by ``PhysicsTerm.name``.

Note that ``profile_terms.py`` disables CUDA graph capture — otherwise every
kernel reports the same synthetic instruction and nothing is attributable — so
its *total* step time reads high. For throughput use ``tools/benchmark.py`` and
the ``jcm-benchmark`` skill's methodology.

Raw traces
""""""""""

To inspect the timeline directly:

.. code-block:: python

   import jax.profiler
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Start a trace and create a Perfetto trace file
   jax.profiler.start_trace("./tensorboard_logs", create_perfetto_trace=True)

   model = Model(coords=get_speedy_coords(),time_step=30.0)

   # Run the model
   predictions = model.run(
       save_interval=0.5/24,
       total_time=1/24,
   )

   # Ensure all computations are complete
   jax.tree_util.tree_map(
       lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
       predictions
   )

   # Stop the trace
   jax.profiler.stop_trace()

You can visualize the generated trace file using **Perfetto**, a performance analysis tool for a variety of platforms.
To use Perfetto, navigate to https://ui.perfetto.dev/ in your web browser. Then, click "Open trace file" and select the
`.perfetto-trace` file generated by :py:func:`jax.profiler.start_trace`. This will display a detailed timeline of your
model's execution, showing CPU and GPU activity, memory usage, and other performance metrics, which is useful for debugging performance bottlenecks.

Documentation
^^^^^^^^^^^^^

Documentation is built with Sphinx. To build locally:

.. code-block:: console

   $ cd docs
   $ make html

Then open ``docs/build/html/index.html`` in your browser.

Communication
-------------

- **GitHub Issues**: For bugs, feature requests, and discussions
- **Pull Requests**: For code reviews and merging changes
- **Code Comments**: For explaining complex logic in the code

We appreciate your contributions and look forward to working with you!
