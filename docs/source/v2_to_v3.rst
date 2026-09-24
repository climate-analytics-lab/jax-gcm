v2 to v3 Migration Guide
========================

v3.0 is a deliberate major release: it changes public APIs, unit contracts,
configuration names and — in several places — the climate the model produces.
This guide covers what an existing v2 script, notebook, config or checkpoint
has to change, and what to expect from a run that changes nothing.

The baseline is **v2.0.1**, the last tagged release on ``main``. If you are
coming from v1, read :doc:`v1_to_v2` first.

.. contents::
   :local:
   :depth: 1

Read this first: the three changes that silently alter results
--------------------------------------------------------------

Most items below fail loudly. These three do not, so check them before
comparing any v3 number against a v2 one.

1. **Specific humidity is kg/kg everywhere** (:ref:`v3-q-units`). A v2-written
   state file read as-is gives an atmosphere 1000x too wet. (A v2 *checkpoint*
   is the safe case — it is refused outright, see
   :ref:`v3-checkpoints` — but only because it is unstamped; override that
   refusal carelessly and you get the same class of error.)
2. **Chemistry volume mixing ratios are ppmv, not ppbv**
   (:ref:`v3-chem-units`). Code that followed the old docstrings now supplies
   1000x too much ozone and methane.
3. **The dynamics is moist on hybrid levels** (:ref:`v3-moist-dynamics`) and
   **semi-Lagrangian tracer transport now has a mass fixer on by default**
   (:ref:`v3-mass-fixer`). Both change the climate of an unchanged
   configuration.

Installation and dependencies
-----------------------------

.. code-block:: console

   $ pip install --pre --upgrade jcm          # 3.0.0rc1 is a pre-release
   $ pip install "jcm==3.0.0rc1"              # or pin it

Until 3.0.0 is tagged, a plain ``pip install --upgrade jcm`` keeps serving the
2.x line: ``3.0.0rc1`` is a PEP 440 pre-release and pip ignores those unless
asked.

Minimum versions that changed, and why each floor is a floor rather than a
preference:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Requirement
     - Minimum
     - Why
   * - ``dinosaur``
     - ``>= 1.5.0``
     - 1.4.0 added the semi-Lagrangian transport jcm now requires
       unconditionally; 1.5.0 fixes the hybrid-coordinate temperature equation
       (neuralgcm/dinosaur#144), which ECHAM's hybrid levels depend on.
   * - ``flax``
     - ``>= 0.12.1``
     - 0.12.1 added ``nnx.Variable.get_value()``, which jcm reads every
       parameter through. On exactly 0.12.0 the lookup falls through to the
       wrapped object and a ``Model`` fails to build.
   * - ``pyses`` (extra)
     - ``>= 0.1.3.1``
     - Earlier builds lower the spectral-element contractions to per-gridpoint
       GEMMs on GPU (1.4x slower, 1.8x the memory at ne30L47) and carry an
       upstream tracer-hyperviscosity bug on the ``quasi_uniform`` path every
       canonical ne30 configuration selects. **ne30 results produced with an
       older pyses should be treated as provisional.**

Public API changes
------------------

.. _v3-q-units:

Specific humidity is kg/kg on every interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PhysicsState.specific_humidity`` is **kg/kg**,
``PhysicsTendency.specific_humidity`` is **kg/kg/s**, and serialized output
advertises the CF unit ``kg kg-1``. Before v3 the dynamics/physics bridge
tagged the field g/kg while ECHAM's schemes were kg/kg-native, so the stored
tracer was ``q/1000`` and the moisture term in the virtual temperature was
weakened by a factor of 1000. SPEEDY still works in g/kg internally, behind a
single adapter.

.. code-block:: python

   # v2: a nudging target or a hand-built state needed the g/kg conversion
   target = q_kgkg * 1000.0

   # v3: pass the physical value
   target = q_kgkg

Two **opposite** 1000x hazards follow, and they are easy to conflate:

* a **checkpoint** (``run.checkpoint_path``, msgpack) written by v2 holds the
  dycore-native value. On ECHAM that is physical/1000, so forcing such a file
  in without the ``x1000`` gives an atmosphere 1000x too **dry**. v3 refuses
  unstamped files precisely so this cannot happen by accident — see
  :ref:`v3-checkpoints`, and note the factor is ECHAM's, not SPEEDY's;
* a **netCDF** written by v2 and fed back as a gridpoint state holds g/kg.
  Read as kg/kg it is 1000x too **wet**.

The reliable tell is magnitude, not provenance: a near-surface
``specific_humidity`` above 0.1 is g/kg, because that value is impossible in
kg/kg.

.. _v3-chem-units:

Chemistry mixing ratios are ppmv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ChemistryData``, ``ChemistryState``, ``ChemistryTendencies`` and
``ChemistryParameters`` are **ppmv** (rates ppmv s⁻¹). The v2 docstrings said
ppbv while the RRTMGP boundary converted as if ppmv, so a caller who followed
the documentation arrived at gas optics 1000x high.

.. code-block:: python

   # v2 docstrings said ppbv, so a caller following them passed a ppbv value
   ozone_vmr = ozone_ppbv

   # v3: divide any value that followed the old ppbv documentation by 1000
   ozone_vmr = ozone_ppbv / 1000.0

   # on a struct you already hold, that is one replace:
   chem = chem.replace(ozone_vmr=chem.ozone_vmr / 1000.0)

Use the explicit ``ozone_mole_fraction()`` / ``methane_mole_fraction()``
helpers where a mol/mol value is wanted, rather than writing the ``1e-6``
yourself.

Model state, physics carry and the clock are public
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The private members external drivers were reaching into are now public
methods and properties.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - v2
     - v3
   * - ``model._prepare_initial_dycore_state(...)``
     - ``model.initial_state(...)``
   * - ``model._build_initial_physics_carry()``
     - ``model.initial_physics_carry()``
   * - ``model._final_dycore_state``
     - ``model.dycore_state`` (property)
   * - ``model._final_physics_state``
     - ``model.physics_carry`` (property)
   * - ``model.bootstrap_state(...)`` returning ``None``
     - returns ``(dycore_state, physics_carry)``
   * - ``model._date_from_sim_time(t)``
     - ``model.date_from_sim_time(t)``

``_date_from_sim_time`` remains as a thin compatibility alias. It is slated for
removal, but the release it goes in is a maintainer decision rather than
something this guide can promise.

A zero-length round trip through ``run_from_state_with_carry`` purely to obtain
an initial state is no longer needed:

.. code-block:: python

   # v2
   state = model._prepare_initial_dycore_state()
   carry = model._build_initial_physics_carry()

   # v3
   state = model.initial_state()
   carry = model.initial_physics_carry()

Reattaching context after a JAX transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ModelPredictions`` is a pytree whose non-array context (coords, physics,
dycore, observer metadata) cannot survive ``tree_map`` / ``scan`` / ``jit``.
In v2 the rebuilt object raised ``AttributeError`` inside ``to_xarray()`` and
silently lost ``observation_datasets()``, ``snapshot_dataset()`` and
``params``. v3 gives it a public repair:

.. code-block:: python

   # v2 (hand-rebuilt, and only coords + physics came back)
   ModelPredictions(predictions._predictions, model.coords, model.physics).to_xarray()

   # v3
   predictions.with_context(model).to_xarray()

The explicit ``with_context(coords, physics, ...)`` form is for custom drivers
that have no ``Model``. Parameters re-derived at this point are labelled as
such, so they cannot be mistaken for trace-time provenance.

One effective timestep
^^^^^^^^^^^^^^^^^^^^^^

``run.time_step`` (**minutes**) and ``dycore.dt_seconds`` (**seconds**) are
resolved by one helper: an explicit ``run.time_step`` wins, otherwise the built
model's step is used. Delegated configurations such as
``jcm/config/run/pyses_year.yaml`` set ``run.time_step: null`` deliberately and
the Model adopts the dycore's value.

What happens on a disagreement depends on which door you came in by, and the
difference matters:

* **Python API.** ``Model(dycore=..., time_step=...)`` with a step that
  disagrees with the dycore's ``dt_seconds`` **raises** ``ValueError``. The
  dycore bakes its step into its integrator at construction, so the Model
  cannot honour the other value; drop ``time_step=`` or rebuild the dycore.
* **Hydra, pySES.** ``run.time_step`` is **ignored with a warning** and the
  dycore's ``dt_seconds`` is used. It is not forwarded to ``Model`` at all.
  This is deliberate: ``run/default.yaml`` sets 12 minutes, so forwarding it
  would make ``dycore=pyses_ne30l47`` fail to build unless the user had also
  selected ``run=pyses_year`` — a value they never chose vetoing the group
  that owns the step. **So a pySES run that sets** ``run.time_step`` **does not
  change its timestep**; change ``dycore.dt_seconds`` instead, and watch for
  the ``pySES owns the timestep`` warning.

Either way, code that read ``float(cfg.run.time_step)`` directly must go
through the resolver, since that value is legitimately ``null``.

``Model(log_level=...)`` is removed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passing it now raises ``TypeError``. Setting a level was never a property of a
model object: ``Model.__init__`` applied it to the ``jcm`` logger as a
construction side effect, resetting process-global verbosity.

.. code-block:: python

   # v2
   model = Model(coords=coords, log_level="CRITICAL")

   # v3 — configure the logger, as with any library
   import logging
   logging.getLogger("jcm").setLevel(logging.CRITICAL)

Related, and visible without any code change:

* ``import jcm`` no longer calls ``logging.basicConfig``. It installs no root
  handler and sets no format, so an application that configures logging keeps
  its own configuration — and one that silences logging now gets silence.
* ``run.log_level`` (default ``WARNING``) now actually takes effect. It
  previously did nothing in ``run.mode=prescribed`` and ``run.mode=scm``, and
  nothing before the model was built in ``full`` mode. **A default CLI run
  therefore prints less** than it did: the resolved ``ozone_file=auto``
  product, the JAX compilation-cache directory, the ERA5 store and the SCM
  column resolution are all INFO. Pass ``run.log_level=INFO`` to get them
  back. Numeric levels are accepted; an unrecognised level raises.
* **Single-column runs may select a different column.** ``run.column.lon_deg``
  is now folded onto the circle, so a negative value such as ``-120`` selects
  120°W rather than a column up to 180° away. A ``lat_deg`` outside
  ``[-90, 90]``, or a non-finite value, raises instead of being clamped to the
  polar-most row. A ``lon_deg`` already in ``[0, 360)`` is unchanged *except*
  within half a cell of 360°, which used to fall back to the last axis centre
  and now wraps to the first.

``set_constants`` now reaches the modules it promised
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nine modules captured constants at import time and so ignored
``set_constants``: JAM activation, dry-deposition resistances, sedimentation,
ice nucleation, the WMO tropopause diagnostic, JAM aqueous chemistry, two
TTE-TKE modules and the emissions preparation tool. They now read the live
singleton.

**This changes results** for any run that overrides ``grav``, ``cpd``,
``m_air``, ``r_universal`` or ``ak`` and composes one of those terms: such a
run was previously computing with a mixed constant set. No magnitude is
quoted because none was measured.

Three capture forms are all now rejected by a lint-style test that AST-parses
every module:

.. code-block:: python

   # all three ignore a later set_constants()
   from jcm.constants import grav                 # value bound at import
   from jcm.constants import physical_constants   # stale singleton reference
   _MW_AIR = c.m_air * 1000.0                     # derived at import time
   def f(x, gravity=c.grav): ...                  # default evaluated at import

   # correct: read through the module on every use
   import jcm.constants as c

   def f(x, gravity=None):
       gravity = c.grav if gravity is None else gravity

Override before the model is built. A constant read inside a jitted term is
fixed at trace time, and constants internal to ``mam4-jax`` are outside jcm's
control.

Widened, not broken: ``vertical_interp_log_p``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``jcm.data.bc.interpolate_ozone.vertical_interp_log_p`` was documented as
field-generic but unpacked exactly four dimensions. It now honours that
contract for 2-D through 5-D input, validates the level count against axis 1
and requires a 1-D, strictly monotonic ``plev_source``. The parameter keeps its
published ``o3_source`` name and the result is bit-identical on 4-D input, so
**no migration is required** and shipped boundary-condition files do not move.
Only the ``Returns`` docstring changed: it no longer claims a fixed
``(time, nplev_target, lat, lon)`` shape.

Configuration changes
---------------------

``+experiment=`` is now ``+configuration=``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The packaged preset group was renamed, with **no compatibility alias group**:

.. code-block:: console

   $ python -m jcm.main +experiment=t63-echam-jam      # v2
   $ python -m jcm.main +configuration=t63-echam-jam   # v3

The Python door moved with it — ``jcm.experiments`` → ``jcm.configurations``
(``load()``, ``available()``). A downstream project that composes the tree
through ``hydra.searchpath: [pkg://jcm.config]`` must also change
``+experiment@<node>=<name>`` to ``+configuration@<node>=<name>``; JAX-ESM
composes ``+experiment@atmosphere=<name>`` and has to move in the same cycle.

``+advection=`` is gone
^^^^^^^^^^^^^^^^^^^^^^^

Semi-Lagrangian tracer transport is the only transport on the Dinosaur
backend; the Eulerian spectral path was removed because it rang negative on
sharp emission sources and NaN'd the aerosol microphysics.
``+advection=semi_lagrangian`` and ``+advection=eulerian`` are both rejected,
and the backend refuses to build on a dinosaur without the SL classes, naming
what to install.

``diffusion.tracer_positivity`` is **not** gone. It survives as a
mass-conserving hole-filler at the dynamics-to-physics boundary, rather than
the positivity mechanism it had to be on the Eulerian path, and it still
resolves the same way:

* ``auto`` (the default) enables it **only** when the physics advects
  prognostic aerosol — that is, ``physics=echam-jam*``, where
  ``aerosol_module == "jam"``. Every other composition (``speedy``, ``echam``,
  ``echam-rrtmgp-2m``, ``held_suarez``, a custom term list) resolves ``auto``
  to **off**, so those runs stay bit-identical to a run without the filter.
* ``true`` / ``false`` force it on or off regardless.

**So do not blanket-delete** ``diffusion.tracer_positivity=true`` **from a v2
command line.** On a JAM command it is redundant — ``auto`` already enables it,
which is why the canonical configuration in
:doc:`design/dinosaur_sl_jam_configuration` no longer passes it. On any
**non-JAM** command dropping it *turns the filter off*, which is a real change
wherever a tracer reaches the boundary negative. Keep the explicit value there,
or delete it deliberately.

A shape consequence worth knowing if you index a dycore-native state:
``specific_humidity`` stays **modal** for the implicit q↔Tᵥ coupling while
every extra tracer is **nodal**, so the two no longer share a shape.

``physics=echam`` composes RRTMGP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``physics=echam`` used to compose the grey two-stream scheme, which is an
unsupported pairing (the supported ones are grey-for-SPEEDY and
RRTMGP-for-ECHAM). It now composes RRTMGP — and because that made the separate
``echam-rrtmgp`` group redundant, **that group was deleted**:

.. code-block:: console

   $ python -m jcm.main physics=echam-rrtmgp   # v2 — now a Hydra missing-config error
   $ python -m jcm.main physics=echam          # v3 — the same term list

(The ``+configuration=t63-echam-rrtmgp`` *configuration* preset is a different
group and still exists.) **There is deliberately no CLI route to a grey-ECHAM
composition.** The Python factory still defaults to grey for the cheap A/B, so
the two doors disagree on purpose:

.. code-block:: python

   echam_physics()                          # grey (unchanged, cheap for tests)
   echam_physics(radiation_scheme="rrtmgp")  # what physics=echam now gives you

JAM no longer composes MACv2-SP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``echam_physics(aerosol_module="jam")`` no longer also builds
``Macv2SpAerosol``. MACv2-SP was a stopgap that filled the shared ``aerosol``
optics/Twomey slot before JAM had its own coupling; JAM now owns that slot
through an ``AerosolCarrySeeder`` and supplies the direct effect through
``JamOpticsTerm``, including the grey scheme's broadband 550 nm profile fields
so grey+JAM keeps a direct effect. With ``jam_optics=False`` the aerosol is
radiatively passive (all-zero optics) — a clean A/B control.
``aerosol_module="macv2sp"`` is unchanged and still the default.

**Breaking: JAM + the one-moment cloud scheme is rejected at compose time.**
``echam_physics(aerosol_module="jam", cloud_scheme="1m")`` raises. JAM's wet
scavenging keys to the process-time ledger the two-moment scheme publishes,
and the cover-keyed reconstruction that let 1M work was removed rather than
kept as an unvalidated path.

**Breaking: aerosol output variables are renamed into namespaces.** The
top-level ``aerosol_optical_depth`` key is removed — it collided with the
unrelated per-band ``RadiationInput`` field — and its value lives on as
``jam_optics.aod_550``. The internal ``aerosol`` struct that radiation and the
microphysics read by attribute is unchanged; only the output keys move.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - v2 output key
     - v3 output key
   * - ``aerosol.aod_total``
     - ``macsp.od550aer``
   * - ``aerosol.aod_anthropogenic``
     - ``macsp.aod_anthropogenic``
   * - ``aerosol.aod_background``
     - ``macsp.aod_background``
   * - ``aerosol.aod_profile``
     - ``macsp.aod_profile``
   * - ``aerosol.ssa_profile``
     - ``macsp.ssa_profile``
   * - ``aerosol.asy_profile``
     - ``macsp.asy_profile``
   * - ``aerosol.cdnc_factor``
     - ``macsp.cdnc_factor``
   * - ``aerosol.Nccn``
     - ``macsp.nccn``
   * - ``aerosol.angstrom``
     - ``macsp.angstrom``
   * - ``aerosol_optical_depth``
     - ``jam_optics.aod_550``

JAM's own column optics publish under ``jam_optics.*``
(``aod_550``, ``aod_profile``, ``ssa_profile``, ``asy_profile``, ``angstrom``,
plus ``aod_sw_per_band`` / ``aod_lw_per_band``). Note that
``jam_optics.aod_550`` is a band-centre approximation, distinct from the
Mie-based ``od550aer`` of the ``aerocom_optics`` pass.

.. _v3-align:

Forcing files must declare climatology or dated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

v2's ``align: auto`` looked at a file's time axis and treated anything
spanning at most ~one year as a climatology, replaying it every model year. A
year of monthly samples from one real year looks exactly like a monthly
climatology, so a one-year transient archive was silently recycled. v3 does
not guess (#884): ``auto`` resolves only data-mirror and packaged products,
from the kind the mirror manifest records, and **raises for any other file**.

What now errors, and the one-line fix:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - v2 usage
     - v3 fix
   * - ``forcing.file=/my/sst_clim.nc`` (align left ``auto``)
     - add ``forcing.align=wrap_year`` (or ``by_date`` / ``by_date_interp``
       for dated samples)
   * - a user ``forcing.ozone_file`` / ``emissions_file`` / ``oxidants_file``
     - add ``forcing.ozone_align=...`` / ``forcing.emissions_align=...``
       (scalar, or one mode per list product) / ``forcing.oxidants_align=...``
   * - ``forcing.prescribed_surface_flux.file`` with a time axis
     - add ``forcing.prescribed_surface_flux.align=...``
   * - ``forcing.prescribed_surface_flux`` with an interactive physics preset
     - compose a forced-mode consumer (``physics=speedy-forced-flux`` /
       ``echam-forced-flux``); without one the block is rejected, never
       silently ignored
   * - ``ForcingData.from_dataset(ds)`` with a time axis
     - pass ``align_mode="wrap_year"`` (an in-memory dataset has no manifest
       identity, so ``auto`` always raises)
   * - ``ForcingData.from_file(path)`` / ``OzoneClimatology.from_file(path)``
       / ``read_anthropogenic_emissions(ds)`` / ``read_prescribed_aerosol_emissions(ds)``
       on a user file
     - pass ``align_mode=...``

``hf://`` mirror paths, their fetched Hugging Face cache files, and the files
packaged under ``jcm/data/bc`` (the SPEEDY T30 and T63 climatologies) keep
resolving under ``auto``; every shipped configuration and the ``amip`` /
``era5`` presets are unchanged. The declared mode is also checked: a
prescribed-flux file declared ``wrap_year`` must hold exactly twelve monthly
samples January to December, a declared ozone climatology must have twelve
months, and a date-aligned flux archive must cover the run window.

Other config-surface changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **One run schema.** ``run/default.yaml`` is the complete base and
  ``longrun`` / ``smoke`` / ``pyses_year`` inherit from it, so
  ``run.checkpoint_path=...`` works on every run group without guessing
  between ``run.``, ``+run.`` and ``++run.``.
* **Emission inputs default to** ``auto``. ``physics=echam-jam* grid=...``
  alone now composes the per-grid emission bundles, resolving
  ``hf://bundles/<grid>/{emissions_pd,dms,dust}.nc`` plus level-matched
  oxidants at build time, the same convention ``ozone_file: auto`` already
  used. **The fetch is eager**, so a cold cache with no network fails at
  startup rather than mid-run. An explicit ``null`` still opts out.
* **MACv2-SP weights are a mandatory key** when you want real plumes:
  ``forcing=macv2_sp`` with ``forcing.macv2_sp_file=...``.
  ``MACv2.0-SP_v1.nc`` is not on the data mirror, and the default all-ones
  weights are a perpetual year 2005.
* **The emulated radiation scheme ships trained weights.**
  ``echam_physics(radiation_scheme="emulated")`` used to build random weights
  that NaN'd within a step; ``weights_file`` now defaults to ``"auto"``
  (packaged weights). An ``emulator_weights_file`` given for a non-emulated
  scheme is rejected.
* **The L40 vertical table is deleted.** Its truncated ``vct`` put the model
  top at 274 hPa. ``40`` is gone from both supported-count lists and
  ``get_echam_levels`` asserts a composed top below 1000 Pa.
* **New config-trap warnings** (warnings, never errors) fire on: JAM with
  aquaplanet terrain; ``forcing=from_file`` with aquaplanet terrain; a
  prognostic aerosol module with every emission input explicitly nulled;
  MACv2-SP on default all-ones weights; and transient by-date forcing composed
  with present-day JAM emission bundles.

.. _v3-datetime:

One real datetime clock
-----------------------

Model time now follows Gregorian dates, including February 29, at whole-second
precision. Timezone-aware inputs are normalized to UTC; naive inputs use the
same timeline. Days contain 86,400 seconds; leap-second timestamps are not
supported. Replace
``start_date`` with ``start_time`` and remove the ``calendar`` argument.
Use exactly one of a fixed ``total_time`` or an absolute ``end_time``:

.. code-block:: python

   model = Model(..., start_time="2000-01-01")
   daily = model.run(forcing=forcing, end_time="2001-01-01",
                     save_interval="1D", output_averages=True)
   monthly = daily.monthly_means()

Numeric run durations remain days. Strings such as ``"6h"`` and ``"1D"``
are fixed durations; ``"1 month"`` and ``"1 year"`` are rejected because
months and years have different lengths. In Hydra, select an endpoint with
``run.total_time=null run.end_time=2001-01-01``.

Run and save durations must divide exactly into model steps, and the run must
contain complete save intervals. An interval mean is labelled at the midpoint
of its exact bounds, which may fall on a half second for an odd-length
interval (e.g. a 1 s step with ``save_interval="3 seconds"``).

Seasonal physics now evaluates January 1 at phase zero in every year. The v2
default instead inherited an epoch-dependent offset (seven days on
2000-01-01), so one-day ECHAM and longer climate fingerprints change even
though the physics equations do not. The ECHAM regression shift was isolated
by running the v3 integrator with the legacy phase before updating that
reference; see :doc:`design/datetime_v3_scope`.

Interval means include ``time_bounds`` and midpoint labels. Monthly means
weight each contributing interval by its duration; snapshots cannot be
converted into interval means after the run. Observers keep their own
sampling: this helper aggregates only the primary output stream. See
:doc:`design/datetime_v3_scope` for the forcing and partial-month contracts.

Whether an input repeats every year or is dated is always declared
(:ref:`v3-align`); the clock decides what each declared mode selects. A
``wrap_year`` climatology is replayed on the real calendar: twelve records are
January to December, each held from the 1st of its month (#805); a 365/366
record table is a nominal-date daily climatology, where a 365-record table
holds February 28 on February 29 and March 1 still selects March 1; any other
length (e.g. MACv2-SP's weekly annual cycle) keeps equal fractions of the
actual year. A climatology's own stamps are read only for their month and day,
so idealised-calendar climatologies still load. Dated ``by_date`` /
``by_date_interp`` input is placed on the exact clock: no-leap dated inputs
retain their nominal date components, and ``by_date_interp`` interpolates
across the actual bracketing dates, including a missing leap day. Unsupported
transient 360-day and Julian axes are rejected at ingestion; preprocess those
explicitly with xarray. Dated lookup holds endpoint values outside its axis
for the surface, ozone, emission and oxidant inputs; only prescribed surface
fluxes are checked to cover the run window, so check forcing coverage when
constructing an experiment.

Coupled output must use the shared public conversion instead of multiplying
floating epoch days into nanoseconds:

.. code-block:: python

   from jcm.predictions import output_time_labels

   # exact_times is a jax_datetime.Datetime axis shared by the components.
   labels = output_time_labels(exact_times)
   ocean = ocean.assign_coords(time=labels)

The result is exact ``datetime64[ms]``; integer-second model labels are
preserved, and interval midpoints are exact to the millisecond: an
odd-length interval's midpoint falls on a half second and is represented
exactly (``ModelPredictions.time_labels`` and ``to_xarray`` compute it from
the exact bounds). Floating days-since-epoch input is rejected. The trajectory serializer uses
the same conversion, preventing tiny timestamp differences from expanding
an xarray merge into two interleaved axes (#862).

A saved run now includes its exact datetime and integer step count. External
couplers must preserve the complete ``RunState`` or pass ``time`` and ``step``
explicitly to ``restore_state``. Dycore ``sim_time`` alone is insufficient.
``run_from_state_with_carry`` now requires ``initial_time`` and
``initial_step`` and returns ``(RunState, ModelPredictions)``; continue with
all four fields of ``RunState`` (``dynamics``, ``physics``, ``time``, ``step``).
See :doc:`advanced_features` for a complete external-stepper example.
``run.mode=prescribed`` places each state at its own time. A dated state
file (every v3 output) supplies the first state's time itself, so
``run.start_time`` may be omitted; if it is set and differs, the config wins
with a warning naming both times. An older output whose ``time`` axis is
elapsed time carries no date and requires ``run.start_time``.
Checkpoints predating the exact clock can only be imported as initial
conditions (``as_initial_condition=True``), starting at the new model's
``start_time``. Unstamped files additionally require the unit assertion
explained below.

.. _v3-checkpoints:

Checkpoint compatibility
------------------------

.. warning::

   **A v2 checkpoint is refused by v3, by design.** It carries no
   ``schema_version`` stamp, and without one the loader cannot tell which unit
   convention its numbers follow — which matters because v3 changed that
   convention, and changed it *differently per physics package*. Rather than
   restore plausible-looking numbers that are wrong by a factor of 1000,
   ``load_checkpoint`` refuses and names the file.

The mechanism, and how to get a v2 state in anyway
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``save_checkpoint`` now stamps every file with ``schema_version``, the writing
``jcm`` version, and every state array **under its pytree name** rather than
its position. Matching on names is what makes ordinary upgrades survive: a
physics-carry field a newer jcm added is seeded from the freshly bootstrapped
carry, one it removed is dropped, both logged at INFO. **The struct-growth
breakage that invalidated every checkpoint in the 2.x line is therefore gone**
— that was reason (1) in earlier drafts of this guide and it no longer applies.

Still refused, because no automatic answer is safe:

* a different grid, level count, precision or physics composition — the file
  names the leaf;
* a changed field set under a carry slot a term declares **prognostic**
  (``PhysicsTerm.prognostic_carry_slots``). JAM's cloud-borne aerosol phase is
  the one case today: it lives only in the carry, so seeding it would invent
  mass and dropping it would destroy mass;
* **any unstamped file**, i.e. anything written before v3.

For that last case there is a deliberate escape hatch. You assert the file's
convention yourself, per leaf:

.. code-block:: python

   from jcm.checkpoint import load_checkpoint

   # a pre-#824 ECHAM donor: its stored mass mixing ratios are 1000x small
   load_checkpoint(model, path, as_initial_condition=True, unstamped_scale={
       "tracers.specific_humidity": 1000.0,
       "tracers.qc": 1000.0,
       "tracers.qi": 1000.0,
   })

   # or, having checked the file is already in the current convention:
   load_checkpoint(model, path, as_initial_condition=True, unstamped_scale={})

and, for a *fresh start from* a saved state, from the CLI through the ``init``
group:

.. code-block:: console

   $ python -m jcm.main init=from_state init.file=<path> \
       'init.unstamped_scale=["tracers.specific_humidity=1000","tracers.qc=1000"]'

An empty mapping asserts "already in the current convention"; a name that is
not a leaf of this model, or is not a floating-point leaf, is rejected rather
than silently ignored. Use it only when you know how the file was written.

.. warning::

   **A pre-3.0 campaign cannot be resumed from the command line.** This is a
   deliberate break, not an oversight: ``run.checkpoint_path`` has **no**
   rescale knob, and is not getting one. The hatch above is on the ``init``
   group and on the Python API only, so
   ``python -m jcm.main run.checkpoint_path=<pre-3.0 file>`` refuses and there
   is no override to add.

   You have two options:

   1. **Regenerate the state on 3.0** — the clean choice if the spin-up is
      affordable.
   2. **Launder the file once through Python**: load it with the assertion, then
      save it back out. The result is an ordinary stamped 3.0 checkpoint that
      the CLI resumes normally.

   .. code-block:: python

      from jcm.checkpoint import load_checkpoint, save_checkpoint

      model = ...                      # the same composition the donor used
      model.bootstrap_state()
      days = load_checkpoint(model, "old.msgpack", unstamped_scale={...})
      save_checkpoint(model, "migrated.msgpack", elapsed_days=days)

   ``save_checkpoint`` stamps whatever it writes, so ``migrated.msgpack``
   carries ``schema_version`` and loads with no assertion at all — verified
   end to end, including that the elapsed-day count survives the round trip, so
   the resumed run continues on the donor's clock rather than restarting it.

.. important::

   **The 1000x factor is ECHAM's, not SPEEDY's — do not copy the entries
   across.** Before v3 the bridge stored ``gridpoint_q x 1e-3``, while the
   *gridpoint* humidity itself was package-specific: ECHAM's schemes were
   kg/kg-native, so an ECHAM donor's store is physical/1000 and needs the
   ``x1000``; SPEEDY's were g/kg-native, so the two factors cancelled and a
   SPEEDY donor's store is **already physical**. Scaling a SPEEDY donor by 1000
   would be the same error in the opposite direction.

The full policy, the evidence behind it and the rule for bumping the schema
are in :doc:`design/checkpoint_compatibility`; the shipped
``jcm/config/init/from_state.yaml`` carries the same warning next to the knob.

What changed underneath
^^^^^^^^^^^^^^^^^^^^^^^

For reference, the two convention changes the stamp exists to disambiguate:

* **Gridpoint humidity** is kg/kg on every package (#666) — see
  :ref:`v3-q-units`. What the *dycore* stored for it changed with it, and
  differently per package, as above.
* **Mass mixing-ratio tracers** — cloud condensate, aerosol mass, gas mass —
  are stored unscaled in kg/kg rather than nondimensionalised (#824), so a
  pre-v3 file's values are 1000x smaller than the new convention. Tracers
  declaring ``nondimensionalize=False`` (number concentrations, volume mixing
  ratios) are unaffected.

If none of this applies to your situation, the simplest path is unchanged:
start from a fresh initial state, or from a gridpoint ``PhysicsState``, which
carries no dycore convention at all.

Science-changing fixes
----------------------

These change the climate of a configuration you did not otherwise touch.
Directions and magnitudes are quoted only where a measurement exists; where a
change was not quantified, this guide says so rather than estimating.

.. _v3-moist-dynamics:

Moist dynamics on hybrid levels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The geopotential handed to physics is now built from **virtual** temperature,
and the hybrid primitive equations carry the condensate loading too:
``Tv = T (1 + (Rv/Rd - 1) q - Σ q_condensate)``, following ECHAM6's
``dyn.f90::ztv`` and the identical expression for the physics geopotential in
``physc.f90::ztvm1``. The condensate set is whatever the composition declares
of ``qc``/``qi``/``qr``/``qs``; including prognostic rain and snow is a
deliberate departure from ECHAM6, whose one-moment scheme carries no
prognostic precipitation.

Measured at T21L47: with the coupling off, ``T`` and ``q`` are bit-identical
to the previous store and global condensate mass agrees to 1 part in 1e7. With
it on, **temperature responds by up to 0.37 K locally against a 0.26 K Tᵥ
deficit at peak condensate**. Pure-sigma (SPEEDY) configurations keep a dry
dynamics; only their physics geopotential changes.

.. _v3-mass-fixer:

A mass fixer on semi-Lagrangian transport, on by default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Semi-Lagrangian transport was **creating** tracer mass: the quasi-monotone
limiter rectifies interpolation error, so clipping an undershoot can only add
mass. Measured over 90 days at T63L47 with ``echam-jam-aerocom``, dust gained
**+61 to +67 ng/m²/s, about 20% of its own emission flux**, steadily; sea
salt's entire net budget balance was the transport term.

v3 applies an ECMWF-style proportional mass fixer (Diamantakis & Flemming
2014) — one global rescaling factor per nodal tracer per step, clipped to
``[2/3, 1.5]``. Dust's transport term falls to **+0.02 to +0.15 ng/m²/s** and
every species closes to ±0.2 ng/m²/s. Because ``qc``/``qi`` also ride nodal
transport, **cloud-condensate advection becomes conserving too** — a
deliberate behaviour change, not a side effect. Opt out with:

.. code-block:: python

   from jcm.dycore.dinosaur import DinosaurDycore

   dycore = DinosaurDycore(
       coords=coords, terrain=terrain, dt_seconds=1800.0,
       sl_options={"mass_fixer": False},
   )

Per-species ``budget_mass_<sp>`` / ``budget_ptend_<sp>`` / ``budget_dyn_<sp>``
diagnostics and one greppable log line per species per chunk make the residual
visible.

SPEEDY shortwave heating is applied every step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Every release from v1.0.0 through v2.0.1 is affected, and so is the
configuration described in the GMD paper.** The shortwave heating tendency was
returned only on a radiation step and zeroed on the other two of every three,
while fluxes and diagnostics were still recomputed every step — so the output
looked self-consistent. v3 caches the tendency and replays it.

At T31/L8 the TOA-minus-surface residual falls from **39.2 to 2.9 W/m²**
(against 2.5 W/m² for ``nstrad=1``) and the mean atmospheric temperature rises
from **237.1 to 242.5 K**. Over a three-year realistic-Earth run the imbalance
goes **44.8 → 4.5 W/m²**, the atmosphere warms **5.6 K**, and global
precipitation falls **3.48 → 2.75 mm/day**. The run is also ~11% faster.

Cloud cover is reported under maximum-random overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Total cloud cover was scored as a column maximum, which is only a lower bound.
v3 uses ECHAM's own maximum-random overlap (``aclcov``) through the new public
``jcm.analysis.total_cloud_cover(cloud_fraction, dim="level")``. This is a
change of **definition**, not of physics — the audit found no defect in the
cover itself. The offset measured between the two definitions is **+0.147 /
+0.148** on a two-moment year, **+0.124 / +0.128** on a one-moment year and
**+0.110 / +0.111** on a two-moment JAM 90-day segment; the spread across the
three available definitions is about **0.27 to 0.30**. The validation band
moved with it, to 0.5-0.9 for ECHAM. SPEEDY keeps 0.4-0.8 under its own key,
because its RH-based column cover has no profile to overlap.

**If you are comparing pre-release cover numbers, establish which definition
produced them first.**

Convection: ECHAM's own trigger chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two non-ECHAM proxies were replaced by the reference logic: ``cubasmc``
(mid-level trigger) and ``zdqcv`` (deep iff column moisture convergence
exceeds 1.1x the surface latent flux, with a 200 hPa depth demotion). The
cloud-base saturation adjustment became ECHAM's damped ``cuadjtq`` Newton step,
replacing one that **created up to 42% water**; the downdraft wet-bulb now
closes moist static energy to ~1e-2 J/kg where it had been broken by up to
5 kJ/kg. θᵥ variance is a real prognostic budget published as ``thv_sigma``.

Over a 3-day T63L47 A/B the convection-type partition (none/deep/shallow/mid)
moves **38/9/33/21% → 38/10/52/0.2%** and convective/large-scale precipitation
**0.90/0.23 → 0.35/0.52 mm/day**. Treat these as mechanism and stability
evidence, not climate validation.

``cubasmc`` needs ``omega`` as a declared dycore field. ``Model`` switches the
provider on automatically for a capable backend; on pySES, which has none, use
ECHAM's own ``cu_lmfmid=false`` switch (see
:ref:`v3-limitation-omega`).

Two-moment microphysics: the process chain is inside the column scan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The whole two-moment chain now runs inside the vertical scan, so each level
consumes the precipitation state the levels above produced in the same step.
Two processes were previously **dead code**: accretion by rain from above and
snow riming both read never-declared ``qr``/``qs`` tracers that were
identically zero, and are now Marshall-Palmer inversions of the carry fluxes.
Three latent defects were fixed: the mean ice radius was receiving an effective
radius in microns where the volume-mean radius in metres belongs (``r³`` off by
1e18), ``zqrho`` was ``1/ρ`` instead of ``1.3/ρ``, and ``peta`` was receiving a
dimensionless saturation ratio where ECHAM's ζ belongs. The
condensation-derivative ``zdqsdt`` was stepping 1.0 K while keeping the
factor for ECHAM's 0.001 K lookup step: ``zqcon`` at 285 K goes from
**1/644 to 0.357**.

**Breaking within the term:** the Sundqvist bolt-on is removed and the
two-moment scheme owns saturation adjustment itself. **Output semantics
change:** the one-moment term now also performs ECHAM's ``paclc``
write-back, so ``clouds.cloud_fraction`` means the post-microphysics cover
under *both* schemes. ``alhf`` became a derived property (``alhs - alhc``),
shifting Tiedtke's melting heat by 0.3%.

JAM wet scavenging keys to the process-time ledger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The in-cloud scavenging rate scaled with the **end-of-step** cloud cover, so a
cell whose condensate fully converted within one step (cover 0, condensate 0)
received **zero removal in exactly the step with the largest removal**; and the
cloud-borne reservoir drained on the resuspension timescale whenever cover read
zero, letting up to **86% of it escape scavenging at a 1800 s step**. v3 reads
the two-moment scheme's ``ScavengingLedger`` — formation rates, process-time
cover and the evaporation ledger — and computes HAMMOZ's
``prep_wetdep_hydro`` fractions from it, with resuspension keyed to evaporation
rather than to cover.

One deliberate deviation from the reference: ECHAM-HAM itself has the dead zone
(a fully-converting cell gets a zero fraction); jcm reads zero pool with
positive formation as a scavenged fraction of 1.

JAM hygroscopic water volume
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Aerosol water was computed from the number-median radius as if it were the
volume-mean, understating the water volume by **2.70x** at σ_g = 1.6 (Aitken,
primary carbon) and **4.73x** at σ_g = 1.8 (accumulation, coarse). The fix is a
number-free growth-ratio form, exact for the whole lognormal at any σ_g. In a
6-day single-column A/B, ``od550aer`` falls **8.3%** while ``od550_wat`` nearly
doubles (**x1.96**) and water's share of ``od550aer`` goes **0.239 → 0.509**;
``od550_so4`` falls 59% purely as re-apportionment, not as a sulfate change.

Also from the dependency floor, with no jcm-side change: **dinosaur 1.5.0
fixes the hybrid-coordinate temperature equation**, so results on ECHAM hybrid
levels differ from runs made with earlier dinosaur builds. Sigma-level runs are
unchanged.

.. _v3-support-matrix:

Support matrix
--------------

:doc:`science/configurations` is the authoritative, exhaustive roster: all 19
packaged configurations, tiered, with grid, timestep and forcing per row. This
table is the other cut through the same information — **backend against physics
package** — because that is the axis a v2 user is most likely to assume is
free.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - physics package
     - Dinosaur (semi-Lagrangian)
     - pySES CAM-SE (``jcm[pyses]``)
   * - ``speedy``
     - **Release-validated**, T31 L8 sigma (``speedy-t31``)
     - Composes and builds; no packaged configuration, never validated
   * - ``echam`` (1M)
     - **Release-validated**, T63 and T106 L47
     - No packaged configuration; composes only with ``cu_lmfmid=false``
       (:ref:`v3-limitation-omega`)
   * - ``echam-rrtmgp-2m``
     - **Release-validated**, T63 and T106 L47
     - No packaged configuration; composes only with ``cu_lmfmid=false``
       (:ref:`v3-limitation-omega`)
   * - ``echam-jam``
     - **Release-validated**, T63 L47 and T63 L95
     - Benchmark-validated at ne30 L47/L95, with ``cu_lmfmid=false`` and
       prescribed-emission-free (below)
   * - ``echam-emulated-2m``
     - Benchmark-validated, T63 L47
     - No packaged configuration
   * - ``held_suarez``
     - Composes and builds; no packaged configuration
     - Composes and builds; no packaged configuration
   * - Betts-Miller / RCE
     - Python-only (``jcm.rce.rce_physics``); unit tests, no Hydra group
     - No packaged configuration

Betts-Miller is the default convection of the single-column RCE layer
(``jcm.rce``), which is a Python entry point rather than a Hydra group: no
``physics=`` or ``+configuration=`` option composes it, and its coverage is the
``rce_test.py`` / ``betts_miller_test.py`` unit suites. The separate
``tools/release_validation/scm_check.py`` is **not** an RCE check despite
borrowing ``jcm.rce``'s column setup — it drives ECHAM+JAM with Tiedtke
convection on one prescribed column.

"Release-validated" means a member of ``tools/release_validation/matrix.yaml``:
a full A100 year with 5-day means, scored by ``health.py`` against TOA net,
precipitation, cloud cover, near-surface temperature and AOD bands (plus the
JAM burden-drift and budget-residual gates). **All seven release-validated
members run on the Dinosaur backend.** "Benchmark-validated" means a
configuration-group preset exercised by ``tools/benchmark.py`` and long
campaign runs, but not scored by the release matrix.

Accepted limitations (proposed)
-------------------------------

Each item below is the shipped behaviour of v3.0, documented rather than
fixed. They are marked **documented limitation (proposed)** because the
decision to accept rather than fix each one is the maintainer's, recorded
against the release tracker (#831) — any of them can be flipped to a code fix
before the tag.

.. _v3-limitation-omega:

pySES publishes no ``omega``, so mid-level convection is off there
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Documented limitation (proposed) — #698.*

pySES computes the vertical pressure velocity internally but does not publish
it, and ``PysesCamSEDycore`` has no ``compute_omega`` provider to switch on.
ECHAM's Tiedtke scheme declares ``omega`` whenever ``cu_lmfmid`` is true, which
is the reference default, so **model construction raises** — it does not warn
and does not silently substitute zero:

.. code-block:: text

   ValueError: The composed physics requires dycore-supplied fields ['omega'],
   but this backend provides none, and has no compute_<field> provider to
   switch on. ...

(As shipped, the ne30 configurations set ``compute_frontogenesis: false``, so
the message reads ``provides none``.) The escape hatch is ECHAM's own switch,
and **its spelling depends on the shape of the physics group**, because
``cu_lmfmid`` reaches the term by a different route in each:

.. code-block:: console

   # term-list groups (physics=echam, echam-rrtmgp-2m, ...)
   $ python -m jcm.main dycore=pyses_ne30l47 physics=echam-rrtmgp-2m run=pyses_year \
       +physics.terms.tiedtke_convection.params.cu_lmfmid=false

   # factory-built groups (physics=echam-jam*, i.e. builder: echam_physics)
   $ python -m jcm.main dycore=pyses_ne30l47 physics=echam-jam run=pyses_year \
       +physics.cu_lmfmid=false

The wrong spelling fails loudly rather than falling back to the default. The
packaged ``+configuration=ma-ne30-l{47,95}`` presets already set the right one.
From Python the door is ``ConvectionParameters.default(cu_lmfmid=False)``.

**The cost is physical, not procedural:** a run with the trigger off has no
elevated convection above a stable layer at all.

Only four PhysicsTerms are verified layout-agnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Documented limitation (proposed) — #626.*

``ComposablePhysics`` is presented as host-agnostic — the same term running
unchanged on a whole ``(kx, ix, il)`` grid or a column-vectorized
``(kx, ncols)`` block. The dynamic audit (``jcm/physics/layout_audit_test.py``)
classifies all **62** shipped concrete terms, and behaviourally checks
grid-versus-column agreement for exactly **four**:

* ``AerocomDiagnostics``
* ``MoistAirColumnState``
* ``NudgingTerm``
* ``UpperSponge``

Eleven more are inert in the harness (they emit nothing on the audit's
environment, so a comparison would be vacuous) and the remaining 47 are not
audited. Two honest caveats: the audit **forces classification** but never
drives the 58 unchecked terms in both hosts, and the claim that column-only
terms "raise loudly on a 3-D state" is an assertion in a docstring, not
something the audit tests. The ``ClassVar``-declared-host proposal in #626 is
not implemented.

In practice: a non-lat/lon backend still needs the column-vectorized path, and
a term outside the list of four has not been shown to give the same answer in
both hosts.

The water positivity cap is a small artificial source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Documented limitation (proposed) — #806.*

``verify_tendencies`` clamps a negative tendency at ``-max(q, 0)/dt`` for the
retained water fields, while TTE-TKE emits a conservative vertical
redistribution whose donor layer can be overdrawn by the summed operator-split
sinks. The donor is clamped, the receivers keep their gain, and column water
mass is created. The cap is kept because the moist physics downstream requires
``q >= 0``.

v3 makes it **measurable rather than invisible**, and nothing more: exact
stop-gradient ``applied - raw`` diagnostics per water field, their total, and —
for ECHAM-family physics, which publishes pressure thickness — a
pressure-weighted ``column_water_source`` in kg m⁻² s⁻¹. The documented
monitoring target is cumulative correction below **0.1% of cumulative
precipitation**; it is informational, not a runtime failure threshold.

Two gaps remain inside the instrumentation: ``verify_state``'s entry clip is
the same class of source and is **not** in the ledger, so the reported number
understates the total; and SPEEDY could publish ``column_water_source`` from
its own layer thicknesses but does not. Note also which tracers are in scope —
the cap covers ten names (the water fields plus ``qnc``/``qni`` and the three
VMRs) while the ledger covers only ``specific_humidity`` and
``qc``/``qi``/``qr``/``qs``. JAM aerosol and gas tracers are deliberately
**not** capped; their removal is bounded where it is produced.

Native HAMMOZ dust inputs exist only at T63
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Documented limitation (proposed) — #810, with the calibration half in #808.*

The five HAMMOZ soil and source fields exist on disk only at T63. T106 is
published as a **nearest-neighbour refinement** of T63 — the honest choice,
since conservative regridding cannot refine a grid and the region mask is
categorical, but not the native field HAMMOZ would use. So T106 gains
resolution in the dynamics and none in the dust source, and the ``ndust = 3``
tuning polynomial it feeds is itself fitted only up to T63.

There is **no ne30 dust product at all**, so ``auto`` resolves to nothing on
the pySES backend: a shipped ``+configuration=ma-ne30-l{47,95}`` run has the
dust term composed but inert, leaving online Gong sea salt as its only aerosol
source. A file supplied by hand is sampled onto the physics columns —
continuous fields bilinearly, the categorical region mask nearest-neighbour —
so an ne30 dust field is two removes from a native one. See
:doc:`science/boundary_conditions` and :doc:`science/aerosol`.

Validation gaps in the release matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Documented limitation (proposed) — #638.*

Three things the matrix does not currently establish, each worth knowing before
quoting a validated configuration:

* the **T106 members' multi-GPU mesh configurations have never been run for a
  full year**;
* ``echam-jam`` at **L95** needs L95 oxidant and ozone inputs staged, which is
  a data dependency rather than a code one;
* the single-column JAM check (``scm_check.py``) composes **grey** radiation,
  while the stated pairing policy for the matrix is RRTMGP for ECHAM. Either
  the check or the policy should move.

A ``FAIL`` from ``health.py`` is also a recorded verdict rather than
automatically a blocker: several members fail a gate by design until the
underlying calibration item is fixed, and the harness is designed to post the
table as-is.

Where to look next
------------------

* :doc:`release_notes` — the full v3.0.0 change list, including everything that
  is additive rather than breaking.
* :doc:`science/configurations` — the authoritative configuration roster and
  its tiers.
* :doc:`getting_started` and :doc:`running_at_scale` — the CLI, the config-group
  tree and the override semantics.
* :doc:`v1_to_v2` — if you are coming from v1.
