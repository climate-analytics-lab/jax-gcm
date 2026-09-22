Advanced features
=================

:doc:`getting_started` covers building a model, running it and looking at the
output. This page collects the capabilities you reach for after that — each one
useful, none of them needed on day one.

The sections are independent; read the one you need. Every one of them opens
with what it is for, so you can tell from the first paragraph whether it is the
thing you were looking for.

.. contents::
   :local:
   :depth: 1


.. _overriding-constants:

Overriding physical constants
-----------------------------

Use this for a different planet, or for a sensitivity study on a constant
the physics treats as fixed. The important part is *when*: constants are
baked in at construction and trace time, so the override has to happen
before you build anything, and comparing several sets means separate
processes.

All shared physical constants live in a single source of truth,
:class:`jcm.constants.PhysicalConstants`, exposed as a process-global singleton.
Each quantity has exactly one canonical name (e.g. dry-air specific heat is
``cpd``, the dry-air gas constant ``rd``, the melting point ``tmelt``).
*Derived* quantities (``rd = akap·cpd``, ``cvd``, ``rgrav``, the ``vtmpc*``
coefficients) are computed on access, so they always stay consistent with the
base values.

.. note::

   jcm's default gravitational acceleration is ``grav = 9.81`` m/s² (the value
   the physics ports were tuned against), **not** the WMO standard 9.80665.
   When you compare a burden or mass budget against an external tool, weight
   with the *same* ``g`` the model used — read it from :mod:`jcm.constants`
   (``import jcm.constants as c; c.grav``) rather than hardcoding a literal.
   :mod:`jcm.analysis`'s column integrals already use the live singleton for
   exactly this reason.

To run with non-default constants — say for a different planet or a sensitivity
study — call :func:`jcm.constants.set_constants` **before constructing the
model**. Only *base* fields are set; derived constants follow automatically, and
both the dynamical core and the physics pick up the override:

.. code-block:: python

   import jcm.constants as c
   from jcm.model import Model
   from jcm.physics.speedy.speedy_coords import get_speedy_coords

   # Override base constants (derived values recompute automatically)
   c.set_constants(grav=9.80665, rearth=6.371229e6, cpd=1005.0)
   assert c.rd == c.akap * c.cpd     # derived value follows

   coords = get_speedy_coords(layers=8, spectral_truncation=31)
   model = Model(coords=coords)       # honours the override

The CLI exposes the same override through the ``constants`` config group
(``+constants.grav=9.80665``) — see :doc:`running_at_scale`.

.. note::

   The override is **process-global** and must be set *before* the model is
   constructed. Read constants by attribute access (``import jcm.constants as
   c; c.grav``) — a ``from jcm.constants import grav`` captures the value at
   import time and will not track later overrides.

.. warning::

   Set constants **once, at the start of the process, before building any
   model** — think of them as fixed for the run (hence *constants*), in contrast
   to calibratable scheme parameters which are threaded through the model as
   explicit, differentiable arguments.

   Constants are baked into a model at construction/trace time: the dynamical
   core reads the singleton when it is built, and physics functions read the
   current values when JAX first traces them. Because JAX caches compiled
   functions, calling :func:`~jcm.constants.set_constants` *after* a model has
   been built — or building a **second** model with different constants in the
   same process — is **not** guaranteed to take effect; already-traced/compiled
   code keeps the values it was first traced with. To compare several constant
   sets, run each in a **separate process** (e.g. a fresh interpreter or a
   separate CLI invocation).

Real dates and monthly interval means
-------------------------------------

``Model`` uses one real Gregorian clock. Set ``start_time`` when constructing
it, then supply either a fixed ``total_time`` (numeric days or a string such
as ``"24h"``) or an absolute ``end_time``. Months and years are not duration
aliases: use an endpoint when their varying lengths matter.

For monthly output, save daily interval means and aggregate their bounds:

.. code-block:: python

   model = Model(..., start_time="2000-01-01")
   daily = model.run(end_time="2001-01-01", save_interval="1D",
                     output_averages=True)
   monthly = daily.monthly_means()

The helper returns an xarray Dataset with duration-weighted means, bounds
and coverage. It rejects snapshots and intervals crossing month boundaries,
because their monthly means cannot be reconstructed. A daily grid should
therefore start at midnight. Observer datasets remain separate and retain
their sampling cadence.

Categorical integer and boolean diagnostics have no defined arithmetic mean.
They are omitted from interval-mean primary output and named in the Dataset's
``omitted_interval_mean_variables`` attribute. Instantaneous output retains
them.

Long runs can feed each chunk's daily Dataset to
``jcm.temporal_aggregation.MonthlyMeanAccumulator.update``. Write any returned
completed months, then call ``finish()`` for the last, possibly partial month.
Its resumable state contains sums and valid durations per variable; it does
not retain a month of daily fields.

Long forcing time-series and chunked runs
-----------------------------------------

Reach for this when the forcing record is too large to hold in memory — a
multi-decade reanalysis, say. Running a year at a time keeps memory bounded
and lets you write output as you go. (This is about the *forcing*; chunking
the integration for preemptible jobs is a separate mechanism, in
:doc:`running_at_scale`.)

For multi-year forcing files, it's often convenient to run the model one
year at a time. This keeps memory bounded and lets you save output as you
go. Use ``xarray.Dataset.groupby('time.year')`` to slice the forcing,
then ``Model.run`` for the first year and ``Model.resume`` for subsequent
years to continue from the previous state:

.. code-block:: python

   import xarray as xr
   from jcm.forcing import ForcingData

   ds = xr.open_dataset('era5_1980_2010.nc')
   yearly_outputs = []

   year_iter = iter(ds.groupby('time.year'))

   year, year_ds = next(year_iter)
   model = Model(coords=coords, start_time=f'{year}-01-01')
   forcing = ForcingData.from_dataset(year_ds, coords=coords)
   preds = model.run(forcing=forcing, save_interval='1 day',
                     end_time=f'{year + 1}-01-01')
   yearly_outputs.append(preds.to_xarray())

   for year, year_ds in year_iter:
       forcing = ForcingData.from_dataset(year_ds, coords=coords)
       preds = model.resume(forcing=forcing, save_interval='1 day',
                            end_time=f'{year + 1}-01-01')
       yearly_outputs.append(preds.to_xarray())

   trajectory = xr.concat(yearly_outputs, dim='time')

xarray's lazy loading means each year's slice only pulls the data it
actually needs from disk, so this stays memory-efficient even for very
long forcing records. (The Hydra runner's chunked/checkpointed loop — see
:doc:`running_at_scale` — chunks the *integration* for preemptible runs,
but it builds the full ``ForcingData`` up front; for a forcing record too
large for memory, this manual per-year loop is the memory-efficient
pattern.)

Yearly forcing bundles
^^^^^^^^^^^^^^^^^^^^^^^

The transient AMIP boundary conditions ship as one file per year (download
only the years you run, append new years without rewriting history). A config
points at a ``{year}`` pattern plus an inclusive range;
:func:`jcm.forcing.expand_yearly_files` turns that into the concrete file list
that :meth:`~jcm.forcing.ForcingData.from_file` concatenates along ``time``:

.. code-block:: python

   from jcm.forcing import ForcingData, expand_yearly_files

   files = expand_yearly_files(
       'hf://bundles/t63/forcing_amip/{year}.nc',
       years=[1979, 1983],            # inclusive
       available=[1979, 2022],        # optional: product's source coverage
   )
   forcing = ForcingData.from_file(
       files, coords=coords, align_mode="by_date_interp")

Passing ``available`` widens the expansion by one year on each side (clipped to
coverage) so the mid-month samples bracket the run's start/end instead of
clamping for ~half a month. Non-pattern specs (plain paths, lists, ``None``)
pass through untouched, so a run can mix a yearly SST pattern with a static
dust climatology under one ``forcing.years`` range. When you hand-assemble a
:class:`~jcm.forcing.ForcingData` rather than loading a validated bundle,
:func:`jcm.forcing.validate_emissions_grid` and
:func:`jcm.forcing.validate_oxidant_levels` guard the grid/level layout the
physics expects.

Nudging the model toward an external state
-------------------------------------------

Use nudging when you want the model's large-scale circulation to track a
specific reality — comparing fields against observations on given dates, or
cutting internal variability out of a calibration run. It relaxes chosen
fields toward a reference dataset while leaving the physics free to respond.

The model can be relaxed toward an external reference state ("nudging")
to suppress internal variability that's unrelated to the question you're
asking — useful for comparing model fields to specific dates of
observations, or for reducing noise in calibration runs.

Nudging is implemented as a gridpoint-space ``PhysicsTerm``:

.. math::

   \frac{\mathrm{d}X}{\mathrm{d}t}\bigg|_\mathrm{nudge}
   = \frac{X_\mathrm{ref} - X}{\tau}

where ``X`` is a gridpoint wind or temperature field and ``τ`` is the
relaxation timescale.
The most common pattern is to nudge winds above the boundary layer and
let everything else evolve freely, so the model gets the right
synoptic-scale circulation while its physics still has the freedom to
respond.

To wire it manually against any reference dataset:

.. code-block:: python

   import xarray as xr
   from jcm.forcing import ForcingData
   from jcm.model import Model
   from jcm.nudging import NudgingTarget, NudgingConfig, with_nudging

   ref_ds = xr.open_dataset('era5_2010.nc')   # u, v, T on (time, lev, lat, lon)

   # The target is loaded straight off the netCDF in gridpoint space and
   # attached to forcing — it's just another per-step input. The Model
   # slices it inside ``forcing.select(date)`` like every other
   # time-varying leaf, so the nudging term never sees the date.
   target = NudgingTarget.from_dataset(ref_ds)
   forcing = ForcingData.from_file('boundary_conditions.nc', coords=coords)
   forcing = forcing.replace(nudging_target=target)

   config = NudgingConfig.winds_only(
       nlev=coords.vertical.layers,
       tau_seconds=21600.0,        # 6 h relaxation
       pbl_levels=2,               # leave the bottom 2 levels free
   )

   nudged_physics = with_nudging(physics, config)
   nudged = Model(coords=coords, terrain=terrain, physics=nudged_physics)
   predictions = nudged.run(forcing=forcing, save_interval='1 day', total_time='30 days')

The reference data can be a single climatology (passed with
``time_var=None``) or a multi-year time series; the latter aligns
against the model's calendar through the same machinery the regular
forcing uses.

Nudging is dycore-agnostic — it's just another :class:`PhysicsTerm`,
producing a gridpoint :class:`PhysicsTendency` that the dycore consumes
through the standard physics-coupling path. The same setup works under
SPEEDY, ECHAM, or any other physics package, on any
:class:`DynamicalCore` backend. The whole setup is also available as a single
CLI flag (``nudging=era5``, pulling the run window from cloud ERA5) — see
:doc:`running_at_scale`.

.. note::
   ``NudgingTarget`` fields use the model-state units: winds in m/s,
   temperature in K, and specific humidity in **kg/kg**. ERA5 stores
   humidity in the same units, so pass it through without rescaling.

Composing extra terms: the upper sponge
----------------------------------------

This is the worked example of the composition API: adding a scheme to a
package is a ``+``. Reach for the sponge itself when a run rings or blows up
near the model lid, which is common on the high-top L47/L95 grids.

Because physics is *composable*, adding a scheme is just ``+``-ing a
:class:`~jcm.physics.physics_term.PhysicsTerm` onto the package. An
:class:`~jcm.physics.dissipation.UpperSponge` — Rayleigh drag on the winds
plus zonal-mean relaxation of temperature at the top few levels — damps
spectral ringing near a rigid model lid:

.. code-block:: python

   from jcm.physics.dissipation import UpperSponge
   from jcm.physics.echam.echam_terms import echam_physics

   physics = echam_physics() + UpperSponge(n_sponge_levels=5,
                                           sponge_timescale_s=3 * 3600.0)
   model = Model(coords=coords, terrain=terrain, physics=physics)

The relaxation timescales that both the sponge and the nudging term use follow
a masked per-level ``1/tau`` profile; :func:`jcm.nudging.inv_tau_profile`
builds one from a dycore vertical coordinate (zeroing the boundary-layer
levels and everything above ``min_pressure_hpa``). See the
:mod:`jcm.nudging` and :mod:`jcm.physics.dissipation.upper_sponge` module
docstrings for the full set of knobs, and :doc:`design/composable_physics`
for the composition API (``+``, ``replace``, ``remove``).

Plugging in an out-of-tree aerosol-optics backend
-------------------------------------------------

Reach for this if you have your own Mie pathway — a neural emulator, or a
different quadrature — and want it inside the JAM aerosol chain without
forking the term. You implement one method; everything around it is reused.

The JAM aerosol optics delegate the one genuinely optical step — the
mode-integrated extinction, scattering and forward-scattering of a lognormal
mode at one wavelength — to an overridable ``_mode_optics`` hook on
``jcm.physics.aerosol.jam.optics.optics_term.JamOpticsTerm``. An
alternative Mie pathway (a neural emulator, say) is therefore a
``JamOpticsTerm`` subclass living in its own package, attached by category:

.. code-block:: python

   from jcm.physics.echam.echam_terms import echam_physics
   from some_optics_package import SomeOpticsTerm   # subclasses JamOpticsTerm

   physics = echam_physics(aerosol_module="jam").replace(
       "aerosol_optics", SomeOpticsTerm())
   model = Model(coords=coords, terrain=terrain, physics=physics)

``replace`` keeps the term's position in the validated JAM ordering and hands
the displaced term to the replacement's ``adopt_runtime_configuration``, so
settings the factory applied after composition — the radiation cadence the
optics gate rides, in particular — are carried over rather than silently reset
to constructor defaults.

Everything around the hook — the modal volumes, hygroscopic water, the
empty-mode mass gate, the SSA/asymmetry weighting, the AeroCom per-species
apportionment and the 550 nm column diagnostics — stays with the base class
and applies unchanged to any backend. The contract an override must satisfy
(finite values *and* derivatives on empty and degenerate modes, non-negative
extinction, no Python branching on traced values) is in
:doc:`design/jam_optics_mode_seam`.

External steppers and transformed predictions
---------------------------------------------

Reach for this when something other than ``Model.run`` drives the
integration — a coupler, a custom stepper, or an optimisation loop that
steps the model itself. It is also what you need after any ``jax.tree``
operation on a trajectory, because a transformed ``ModelPredictions`` loses
the coordinate and physics objects ``to_xarray()`` depends on.

Couplers and custom steppers can obtain both initial pytrees without reading
private model attributes. The two builder methods are pure; they do not change
the state retained by ``model``:

.. code-block:: python

   state = model.initial_state()
   physics_carry = model.initial_physics_carry()
   run_state, predictions = model.run_from_state_with_carry(
       initial_state=state,
       initial_physics_state=physics_carry,
       initial_time=model.start_time,
       initial_step=0,
       forcing=forcing,
       total_time=1.0,
       save_interval=1.0,
   )

For the next window, pass ``run_state.dynamics``, ``run_state.physics``,
``run_state.time`` and ``run_state.step`` together. The exact clock is part
of the resumable state; do not reconstruct it from a floating elapsed counter.

``model.bootstrap_state()`` is the stateful alternative: it installs and
returns a matched ``(state, physics_carry)`` pair for a later ``resume()``.
The installed pair is available through the read-only ``model.dycore_state``
and ``model.physics_carry`` properties. Checkpoint readers replace both values
atomically through ``model.restore_state(state, physics_carry, time=..., step=...)``.
The complete installed state is available as ``model.run_state``.

``ModelPredictions`` deliberately drops coordinate and physics objects when it
crosses a JAX pytree boundary. Reattach that static context before converting a
transformed trajectory to xarray:

.. code-block:: python

   import jax

   transformed = jax.tree.map(lambda value: value, predictions)
   ds = transformed.with_context(model).to_xarray()

The rebuilt parameter record is labelled as a live-model read rather than a
trace-time claim. If observer or snapshot arrays were not pytree children, pass
their run-specific metadata explicitly to ``with_context``.

Two related plumbing details, both of which bite only once you drive the
model yourself:

**Warm starts carry a physics carry.**
:func:`~jcm.initial_states.checkpoint_state` returns
``(state, physics_carry, donor_days)``. Unlike a checkpoint *resume* the
donor's elapsed-day count is discarded, so the clock starts at the model's
``start_time`` — that is what lets a hosted equilibrated state skip the
~9-month from-cold spin-up without inheriting the donor run's calendar. Pass
the carry through, or the run resets the radiation sub-cycle cache and
prior-step TKE at the seam:

.. code-block:: python

   from jcm.initial_states import checkpoint_state

   state, physics_carry, _ = checkpoint_state(
       model, 'bundles/echam_t63_l47_hybrid/init_states/spun_up.msgpack')
   predictions = model.run(
       initial_state=state, initial_physics_state=physics_carry,
       forcing=forcing, total_time='365 days', save_interval='1 day')

(Restoring a checkpoint to continue a preempted run of your own — keeping the
elapsed clock — is :func:`jcm.checkpoint.load_checkpoint`, in
:doc:`running_at_scale`.)

**Observers prepare sampling tables on the host.** When the window's exact
clock is traced under an outer jit, build tables first with
``model.prepare_observers(start_time, save_interval, total_time)`` and pass
them as ``observer_xs``. The tables and exact clock can then vary between
windows without making their values static compilation parameters.


Where to next
-------------

- :doc:`running_at_scale` — the same capabilities from the Hydra CLI, plus
  chunked production runs, multi-device parallelism and batch queues.
- :doc:`design` — how the pieces fit together, and the design notes behind
  each mechanism above.
- :doc:`api` — the full API reference.
