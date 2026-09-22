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

   # Match the CI fast-test coverage gate (both steps, as CI runs them)
   $ pytest -m "not slow" --cov=jcm --cov-fail-under=90
   $ coverage report --fail-under=90

   # Run the linter
   $ ruff check .

Write tests for your changes in the appropriate test file (e.g., ``jcm/module_name_test.py``). We aim for high unit test coverage to support the increasing complexity of physics going forward.

The suite is memory-bound, and a process killed by a memory ceiling shows up
as an arbitrary set of "failures" rather than an error. Before running it in
parallel — in particular on a Derecho login node, where a 10 GiB per-user
cgroup makes ``-n 12`` an OOM rather than a test result — see
:doc:`design/test_suite_memory`, which also covers the ``jax_enable_x64``
isolation the root ``conftest.py`` provides.

What CI runs
^^^^^^^^^^^^

``ruff check .`` is a gate, not a parallel job: it runs first and both test
jobs hang off it, so a lint error costs about twenty seconds instead of two
runner-hours. Behind it the fast suite (90% coverage) and — on pull requests
only — the slow suite (80%, against ``.coveragerc-pr``) run in parallel. If
the fast suite fails it cancels the whole run, taking the in-flight slow job
with it, so **a cancelled slow result never means the slow tests passed**. It
does not tell you *why* on its own: ``cancel-in-progress: true`` cancels that
job identically when a newer push supersedes the run, and so does cancelling
by hand, so open the ``fast-tests`` job to tell a real failure from a
superseded run.

Two limits by design: the cancel only fires on pull requests, since a push to
``main`` or ``dev`` has no slow job to stop and a cancelled run there would
mute the failure notification; and it is best-effort, because a pull request
from a fork gets a read-only token, so there the slow suite runs to
completion.

The workflow is triggered by pull requests, and by pushes to ``main`` and
``dev`` only. A branch with no open pull request gets no CI at all, so run the
commands above locally before opening one.

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

The default window is one radiation sub-cycle — 10 steps at T63L47's 12 min
step and 2 h radiation. Resist lengthening it: the profiler's event buffer
holds about a million events and a T63L47 JAM step emits ~19,000 kernels, so a
longer window overflows it and undercounts. The window must *also* span a whole
number of sub-cycles, so on that configuration one cycle is the only window
both admissible and within the buffer; the previous two-cycle default sat on
the ceiling and failed. The short window loses nothing, because kernel shapes
are static and the model takes no data-dependent branches, so a step's cost
does not vary with the state.

The attribution works because :mod:`jcm.profiling` wraps the dycore call, the
bridge and each :class:`~jcm.physics.physics_term.PhysicsTerm` in a
:func:`jax.named_scope`. XLA records that scope in the ``op_name`` metadata of
every instruction traced inside it, and the profiler reports the instruction
behind each GPU kernel, so kernel time joins back to the component that emitted
it. The scopes are always on and cost nothing at runtime. A new driver-level
stage that deserves its own line in the report needs a scope adding there;
individual physics terms need no changes, since the term loop already labels
them by ``PhysicsTerm.name``.

Before writing a report the tool checks that the trace is complete: the
``dynamics``, ``bridge_to_physics`` and ``bridge_to_dynamics`` scopes sit
outside every loop and branch in the step, so each must show up in the
attribution exactly once per step. It fails — naming the labels — if any of the
three is missing (the HLO-metadata join broke, so every number would be
misattributed) or if any is short of the step count (the event buffer
overflowed, so every number would be an undercount). The overflow error names
the largest window that is both within the buffer and a whole number of
sub-cycles, or says plainly that no admissible window fits when even one cycle
is too long.

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

The strict build is the gate
""""""""""""""""""""""""""""

``make html`` is the convenient loop, but CI
(``.github/workflows/run_docs.yaml``) builds the **whole** tree with warnings
as errors, and that is what a documentation change has to clear. Run the
identical command before pushing, **from the repository root** (the path is
``docs/source``, so this fails if you are still inside ``docs/`` from the
``make html`` above):

.. code-block:: console

   $ cd <repo root>
   $ sphinx-build -W --keep-going -b html docs/source /tmp/docs-html

``-W`` turns every warning into an error; ``--keep-going`` reports all of them
in one pass instead of stopping at the first, so a page with several problems
takes one run to fix rather than several. The build must end in
``build succeeded`` with no warnings at all.

Two things to know when a warning does appear:

- **Fix it at the source, not with a suppression.** ``conf.py`` deliberately
  sets no ``suppress_warnings``; a blanket entry there would silently disarm
  the gate for everyone.
- **Docstrings are part of the tree.** ``api.rst`` autosummarises ``jcm``
  recursively, so a malformed ``Args:`` block in a documented module fails the
  docs build even though the code imports fine. Co-located ``*_test.py``
  modules and ``conftest.py`` are filtered out of that walk by
  ``docs/source/_templates/autosummary/module.rst`` — tests are not public API,
  and the pyses backend's tests need an optional extra the docs environment
  does not install.

Because the docs workflow is not triggered by ``jcm/**/*.py`` (see the comment
in ``run_docs.yaml``), a docstring change that breaks the strict build surfaces
on the next documentation PR rather than on the code PR that introduced it.
Running the command above after editing a public docstring avoids handing that
to someone else.

Communication
-------------

- **GitHub Issues**: For bugs, feature requests, and discussions
- **Pull Requests**: For code reviews and merging changes
- **Code Comments**: For explaining complex logic in the code

We appreciate your contributions and look forward to working with you!
