.. role:: py:class(xref)
.. role:: py:meth(xref)
.. role:: py:func(xref)
.. role:: py:attr(xref)
.. role:: py:mod(xref)

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
   ☐ Code is rebased on the latest main branch

Testing Your Changes
^^^^^^^^^^^^^^^^^^^^^

Run the test suite to ensure your changes don't break existing functionality:

.. code-block:: console

   # Run all tests
   $ pytest

   # Run specific test file
   $ pytest jcm/model_test.py

   # Run with verbose output
   $ pytest -v

   # Run only fast tests (skip slow integration tests)
   $ pytest -m "not slow"

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
- **Immutable Data**: Use ``dataclasses`` or ``tree_math.struct`` for data structures
- **No Python Control Flow**: Use ``jax.lax.cond`` instead of ``if`` in JIT-compiled code
- **Static Shapes**: Array shapes should be statically known where possible

See ``JAX_gotchas.md`` in the repository for more details.

Profiling
^^^^^^^^^

To profile the model and identify performance bottlenecks:

.. code-block:: python

   import jax.profiler

   # Start a trace and create a Perfetto trace file
   jax.profiler.start_trace("./tensorboard_logs", create_perfetto_trace=True)

   model = Model(time_step=30.0)

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
