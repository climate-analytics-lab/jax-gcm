Release Notes
=============

v3.0.0 (unreleased)
-------------------

v3.0 is a deliberate major release. It makes **online interactive aerosol**
(JAM/MAM4) a working configuration end to end, adds the pySES CAM-SE
dynamical-core backend alongside a Dinosaur backend whose tracer transport is
semi-Lagrangian by default, and settles a set of unit, API and output
contracts that were inconsistent in the
2.x line. Several of those corrections change the climate a configuration
produces.

**Read the** :doc:`v2-to-v3 migration guide <v2_to_v3>` **before upgrading.** It carries the before/after
snippets, the checkpoint-compatibility rules, the support matrix and the list
of accepted limitations; this page is the change list.

Breaking changes
^^^^^^^^^^^^^^^^

Every item here requires a change to code, a config, a saved file, or a reader
of the output. :doc:`v2_to_v3` has the migration for each.

One exact Gregorian clock and real monthly output
"""""""""""""""""""""""""""""""""""""""""""""""""

- ``Model(start_time=...)`` replaces ``start_date`` and removes the
  ``calendar`` switch. Runs use exactly one ``total_time`` or absolute
  ``end_time``; a month/year ``run.total_time`` (``12 months``) is resolved
  against ``run.start_time`` into the exact end, while the fixed-duration APIs
  (``save_interval``, ``Model.run``) reject month/year aliases, and silently
  truncated intervals are rejected. The exact datetime and step counter travel with the resumable
  ``RunState`` and schema-2 checkpoints.
- ``wrap_year`` climatologies select by real calendar position: twelve
  monthly records switch at civil month boundaries, including leap years
  (#805), and 365/366-record tables select by nominal month/day. Noleap input
  dates preserve their nominal date components on the Gregorian clock (#449).
  Whether a file repeats annually is declared, never inferred (see the #884
  entry below).
- The chunked CLI streams calendar-month means (#901):
  ``run.monthly_means=true`` writes ``{output_prefix}_monthly_YYYY-MM.nc``
  independent of chunk length, with the pending month persisted and rotated
  with the checkpoint; ``run.save_chunks=false`` drops the per-chunk files.
  ``run=longrun`` and ``run=pyses_year`` now default to a calendar year
  (``12 months``) of daily means in 5-day chunks written only as monthly
  files — previously 365 days of 5-day means in 30-/10-day chunk files.
  A chunk's host work (output conversion, health check, netCDF, monthly
  and checkpoint writes) runs on a background thread while the next chunk
  integrates, and the monthly accumulator works on NumPy buffers in place;
  at T63L47 JAM this recipe runs at the integration's own speed (it was 22%
  slower with the host work in series). A bailing run now stops one chunk
  later; see :doc:`design/chunked_run_io_overlap`.
- ``ModelPredictions.monthly_means()`` reduces bounded interval means by
  real month; save daily means with ``output_averages=True`` first. Observer
  sampling stays independent. Exact shared ``output_time_labels`` replaces
  floating epoch-day conversion for coupled output (#862).
- SPEEDY seasonal phase follows the actual Gregorian year. This changes
  seasonal timing and requires climate validation; empirical local constants
  using 365 days do not define a separate clock. See :ref:`v3-datetime` and
  `issue #876 <https://github.com/climate-analytics-lab/jax-gcm/issues/876>`_
  for migration and release gates.

Checkpoints carry a schema stamp and migrate by field name
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ``save_checkpoint`` now writes a ``schema_version`` stamp, the ``jcm``
  version and every state array under its pytree name, and
  ``load_checkpoint`` matches those names against the destination model: a
  physics-carry field a newer jcm added is seeded from the freshly
  bootstrapped carry, one it removed is dropped, both logged at INFO, so an
  upgrade that touches a diagnostic struct no longer invalidates a restart
  (#731). A grid, level-count, precision or physics-composition difference is
  still refused, naming the file and the leaf, as is a changed field set under
  a carry slot a term declares prognostic (``PhysicsTerm.prognostic_carry_slots``
  — JAM's cloud-borne aerosol phase, which nothing recomputes). **Breaking:** a checkpoint
  written before this release carries no stamp and is refused, because it does
  not record which unit convention its dycore state uses (#824 changed what a
  stored mass mixing ratio means, and #666 changed what a gridpoint humidity
  means, differently per physics package) — start from a fresh initial state,
  or assert the file's convention explicitly with
  ``load_checkpoint(..., unstamped_scale=..., as_initial_condition=True)`` /
  ``init.unstamped_scale``. Files predating schema 2 can supply initial
  conditions but cannot resume an exact v3 clock.
  The policy, its evidence and the rule for bumping the schema are in
  :doc:`design/checkpoint_compatibility`.

jcm configures no logging; ``Model(log_level=...)`` removed
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- **Breaking:** ``Model(log_level=...)`` is gone. It applied a level to the
  ``jcm`` logger, so building a model reconfigured logging for the whole
  process — resetting a verbosity the host application had chosen. A library
  emits records and leaves handlers and levels to whoever assembles the
  process; jcm now does only that. Code passing the argument raises
  ``TypeError`` and should set the level itself instead::

      logging.getLogger("jcm").setLevel(logging.CRITICAL)

- ``import jcm`` no longer calls ``logging.basicConfig``, so it installs no
  handler or format on your root logger. Warnings still reach an unconfigured
  program through Python's last-resort handler (WARNING and above, on stderr);
  to see INFO, configure logging as you would for any library. This never
  affected ``python -m jcm.main``, whose handlers come from Hydra's
  ``job_logging`` regardless.
- An application that silences logging now gets silence — jcm will not
  override it, which it previously did by design (#735). Findings that must
  survive a quiet application are recorded in the run's provenance instead,
  or raised as Python warnings rather than log records: the
  parameters-changed-after-compilation case #735 was about is a
  ``UserWarning`` and the ``live_parameters_differ_from_compiled`` provenance
  key.
- **A default CLI run prints less.** ``run.log_level`` (default ``WARNING``)
  now actually takes effect: previously it was applied only by
  ``Model.__init__``, so it did nothing at all in ``run.mode=prescribed`` and
  ``run.mode=scm``, nothing before the model was built in ``full``, and
  nothing for the four modules that logged to the root logger directly.
  Messages that used to appear regardless — the resolved ``ozone_file=auto``
  product, the JAX compilation-cache directory, the ERA5 store, the SCM
  column resolution — are INFO and now obey it. Use ``run.log_level=INFO``
  to restore them. It also now accepts a numeric level, and refuses an
  unrecognised one rather than silently running at ``WARNING``.
- ``SingleColumnModel`` column selection is fixed on both axes. A westward
  ``run.column.lon_deg`` (``-120`` for 120W) selected a column up to 180
  degrees away, because longitude was matched on the number line rather than
  the circle; requests near the 0/360 seam and above 360 were wrong too. A
  ``lat_deg`` outside [-90, 90] was silently clamped to the polar-most row
  and now raises, as does a non-finite coordinate. A ``lon_deg`` already in
  [0, 360) is unaffected **unless its nearest cell lies across the 0/360
  seam** — a request within half a cell of 360 used to fall back to the
  axis's last centre and now correctly wraps to its first, so such a run
  selects a different column than it did before.
- **Breaking:** a single-column request the state file does not cover is
  refused (#818). A regional or single-column file used to run its nearest
  column however far away (``lat_deg=90`` ran the 80°N row of a
  ``lat = 20..80`` file; a single-column file ran its one column for any
  request); ``select_column`` now raises, naming the request, the file's
  coverage and the distance to the nearest column. Coverage is the axis span
  extended by half a grid cell at each end, with longitude periodic only when
  the file's longitudes close the circle, a latitude end within one row
  spacing of its pole reaching the pole, and a length-1 axis covering only its
  own coordinate. Global files — what jcm writes — are unaffected. See
  :ref:`v3-scm-coverage`.

One trajectory conversion for every backend
"""""""""""""""""""""""""""""""""""""""""""

- ``DinosaurDycore.to_xarray`` converts a real run (#951): it failed on the
  nested physics diagnostics every run returns (``_prev_step``,
  ``water_positivity_correction``) and rewrote the exact ``datetime64``
  labels into an elapsed axis. It is now where the dinosaur trajectory
  conversion lives, ``ModelPredictions.to_xarray()`` delegates to the
  attached dycore for every backend, and the labels are kept as given; a
  numeric axis raises ``TypeError``.
- The physics names its own diagnostics, once, for every model trajectory
  output (``ModelPredictions.to_xarray()``, the chunked CLI's files and each
  backend's ``DynamicalCore.to_xarray``):
  ``jcm.predictions.physics_output_fields`` (the physics'
  ``data_struct_to_dict``) is used by the dinosaur and pySES conversions
  alike, and a trajectory fetched to the host with ``jax.device_get`` now
  writes the same variables (host arrays were dropped). The
  ``run.mode=prescribed`` output keeps its own minimal ``diag.*`` layout. **Breaking for protocol implementers:**
  ``DynamicalCore.to_xarray`` takes a keyword-only ``physics``, and a dycore
  carries the ``output_physics`` a ``Model`` binds to it at construction, so
  a direct ``model.dycore.to_xarray(...)`` names the variables the model's
  output does; a call with diagnostics and no physics to name them raises.
- **pySES output names physics diagnostics as the dinosaur output does.**
  It used to take each leaf's pytree path, which named SPEEDY's typed
  structs by position (``_condensation.0``, ``_shortwave_rad.10``) and wrote
  the ``_prev_step`` carry plumbing; it now writes ``condensation.dqlsc`` and
  the rest of the dinosaur names, drops what the dinosaur output drops
  (``_prev_step``, per-term withheld fields), applies per-term output renames,
  and splits multi-channel fields per channel. See :ref:`v3-dycore-to-xarray`.

Specific humidity has one kg/kg contract
""""""""""""""""""""""""""""""""""""""""

- **Breaking for direct SPEEDY-state and output consumers:**
  ``PhysicsState.specific_humidity`` is now kg/kg for every dycore and physics
  package, and ``PhysicsTendency.specific_humidity`` is kg/kg/s. Dinosaur stores
  that dimensionless mass fraction directly so its hybrid moist dynamics sees
  the physical humidity; pySES/ECHAM and raw ERA5 inputs are unchanged. The
  translated SPEEDY routines still calculate internally in g/kg behind a
  centralized adapter. Serialized ``specific_humidity`` now contains kg/kg
  and advertises the equivalent CF unit ``kg kg-1``. Remove any ``* 1000``
  conversion previously applied when
  constructing a nudging target, and divide old saved g/kg humidity values by
  1000 before supplying them as a new ``PhysicsState`` (#666).
- **Migration, both production resume paths.** A ``run.checkpoint_path``
  msgpack checkpoint stores the dycore-native humidity, which was
  physical/1000 before this release; resuming one without conversion gives a
  1000x too dry atmosphere with no error. An ``init=from_state`` netCDF stores
  g/kg, so reading it as kg/kg gives a 1000x too wet one. The ``units``
  attribute does not distinguish the two — output written since the CF
  metadata pass already advertises ``kg kg-1`` on g/kg values — but the
  magnitude does: a near-surface ``specific_humidity`` above 0.1 is g/kg, and
  is impossible in kg/kg.

``relative_humidity`` has one definition
""""""""""""""""""""""""""""""""""""""""

- **Breaking for readers of** ``relative_humidity`` **from ECHAM runs:** the
  saved field is now always ``MoistAirColumnState``'s water-saturation RH
  (``e/e_s,w``, WMO; what ``tools/aerocom_cmor.py`` writes as ``hur``) and
  carries CF ``standard_name = relative_humidity``. Previously the Sundqvist
  cloud-cover term overwrote it with its own ``q/q_s``, whose ``q_s`` switches
  to ice saturation where a cold cell holds cloud ice — up to ~25 % higher in
  icy cold cells, and discontinuous across the ice threshold. That closure
  humidity is now published separately as ``cover_relative_humidity``
  (#615).

``ChemistryData`` uses ppmv consistently
""""""""""""""""""""""""""""""""""""""""

- **Breaking for direct simple-chemistry callers:** ``ChemistryData``,
  ``ChemistryState``, ``ChemistryTendencies`` and ``ChemistryParameters`` now
  consistently use ppmv (and ppmv s⁻¹ for rates). The old documentation said
  ppbv even though ECHAM boundary conditions supplied ppmv and radiation
  treated the values as ppmv. Divide caller-provided values that followed the
  old ppbv documentation by 1000. Existing callers that supplied the actual
  ECHAM/RRTMGP ppmv convention are unchanged. Field-specific
  ``ozone_mole_fraction()`` / ``methane_mole_fraction()`` helpers make the
  conversion to gas-optics mol/mol explicit (#749).
- **Breaking for direct callers:** ``ChemistryParameters.ozone_stratosphere_coeff``
  is removed. The analytic ozone profile never read it, so it had an exactly
  zero gradient; drop it from any constructor call (#799).

Delegated timesteps have one effective value
""""""""""""""""""""""""""""""""""""""""""""

- Runner and profiling paths now resolve an explicit ``run.time_step`` in
  minutes or, when it is ``null``, adopt the built model/dycore timestep.
  A pySES configuration owns its timestep in ``dycore.dt_seconds``: on that
  Hydra path an explicit ``run.time_step`` is ignored with a warning rather
  than forwarded, so it cannot veto the group that owns the step (the Python
  API is the strict door — ``Model(dycore=..., time_step=...)`` raises on a
  disagreement). Prescribed-state runs, single-column runs, chunk budget
  tolerances
  and term profiles therefore use the same number of seconds as the model
  instead of raising on ``None``, silently falling back to 900 seconds, or
  reporting against a conflicting config value (#801).

Packaged config-tree contract; the ``experiment`` group is renamed
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ``jcm/config`` **is now a documented public, packaged Hydra config tree**
  (#757). A downstream Hydra app reaches every jcm group through
  ``hydra.searchpath: [pkg://jcm.config]`` and can re-root a whole validated
  configuration under one of its own nodes with ``+configuration@<node>=<name>``.
  ``jcm/config/__init__.py`` was added so Hydra's ``pkg://`` provider reports the
  tree as available (a namespace-package ``jcm.config`` was read but flagged
  "not available"). The public group names, the load-bearing
  ``# @package _global_`` header plus absolute-override recipe style, and the
  rename policy are documented in :doc:`design/packaged_config_tree`. There is
  deliberately
  **no** back-compatibility alias group — the contract plus a release note plus
  a lockstep downstream update is the policy.
- **Breaking for searchpath users:** the ``experiment`` config group was
  renamed to ``configuration`` (the word "experiment" already means a *realized
  simulation* elsewhere in the project). The CLI is now ``+configuration=<name>``
  and the Python door is ``jcm.configurations`` (was ``jcm.experiments``). A
  downstream app composing jcm through ``pkg://jcm.config`` must change
  ``+experiment@<node>=<name>`` to ``+configuration@<node>=<name>``; JAX-ESM in
  particular composes ``+experiment@atmosphere=<name>`` and must update in the
  same release cycle.

The ECHAM factory composes RRTMGP; ``radiation_scheme="grey"`` is rejected
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ``echam_physics()`` defaults to ``radiation_scheme="rrtmgp"`` and accepts
  exactly ``"rrtmgp"`` and ``"emulated"`` (the neural-network emulator of
  RRTMGP, the fast option) or a radiation ``PhysicsTerm`` instance (#918).
  **Breaking:** ``radiation_scheme="grey"`` raises ``ValueError``, and a bare
  ``echam_physics()`` — which used to compose the grey two-stream — now runs
  RRTMGP, silently changing the climate and cost of an unchanged script. The
  grey two-stream is an idealized scheme (like Betts-Miller convection) with no
  ECHAM reference and no validation in an ECHAM composition; it remains
  available as a scheme, composed explicitly with
  ``echam_physics(radiation_scheme=GreyTwoStreamRadiation())``. The
  factory-built Hydra presets reject ``physics.radiation_scheme=grey`` with the
  same message. A radiation term instance now drives the rest of the
  composition with its own parameters (the JAM optics cadence follows its
  ``radiation_interval``), and ``radiation=`` alongside an instance is
  rejected. Cheap tests compose the idealized stack through
  ``jcm.physics.echam.testing.idealized_echam_physics``. The package gradient
  harnesses, the ECHAM regression reference and the single-column JAM release
  check run RRTMGP. See :ref:`v3-echam-radiation`.

Forcing time alignment is declared, never inferred
""""""""""""""""""""""""""""""""""""""""""""""""""

- **``align: auto`` no longer guesses from a file's time axis** (#884). v2
  treated any file spanning at most ~one year as a climatology and replayed it
  every model year, so a one-year *transient* archive (a year of monthly SST,
  ozone, emissions or fluxes) was silently recycled. Now ``auto`` resolves
  only a data-mirror or packaged product, from the ``alignment`` the mirror
  manifest records (``climatology`` → ``wrap_year``, ``transient`` →
  ``by_date``; ``by_date_interp`` for transient ozone). **For any other file
  ``auto`` raises**, naming the knob to set. One rule covers every
  time-resolved input on both backends: ``forcing.align`` (SST/sea-ice file),
  the new ``forcing.ozone_align`` / ``forcing.emissions_align`` (a scalar, or
  one mode per ``emissions_file`` product) / ``forcing.oxidants_align``,
  ``forcing.prescribed_surface_flux.align``, and the Python readers'
  ``align_mode`` (``ForcingData.from_dataset`` has no file identity, so its
  ``auto`` always raises). **Fix:** declare the file's kind, e.g.
  ``forcing.align=wrap_year`` for a climatology or ``forcing.align=by_date``
  for dated samples. Every shipped configuration and the ``amip`` / ``era5``
  presets resolve unchanged. See :doc:`v2_to_v3`.
- **Every dated input must cover the run, or declare a hold** (#900). v2
  clamped a ``by_date`` / ``by_date_interp`` input to its first or last sample
  outside its time axis, and a ``{year}`` range past a product's
  ``available_years`` to the edge-year file, with no warning, so a transient
  run past its SST, ozone, emission or oxidant archive silently reused the
  last record. Each dated input now declares its out-of-range policy next to
  its alignment: ``forcing.persist`` / ``ozone_persist`` /
  ``emissions_persist`` / ``oxidants_persist`` / ``macv2_persist`` /
  ``prescribed_surface_flux.persist`` (and ``persist=`` on the Python
  readers and ``ForcingData.from_bundles``). The default ``strict`` checks
  every dated leaf against the run window at every entry point (the CLI with
  the configured window right after assembly, before compilation; ``Model.run``
  / ``resume`` / ``PrescribedStateModel.run`` with their exact windows), and
  rejects out-of-coverage ``forcing.years`` before fetching. ``hold`` holds the
  edge samples deliberately, warns once, and records the policy in the output
  provenance (``jcm_prov_dated_input_persistence``). ``forcing=era5`` declares
  ``ozone_persist: hold`` for its 2023–24 years. The MACv2-SP ``year_weight``
  axis now ends at the file's last real year instead of forward-filling the
  fill years. **Fix:** a transient run past its archive adds
  ``forcing.<input>_persist=hold`` (or covers the run with data). See
  :ref:`v3-persist`.

MACv2-SP removed from JAM; namespaced aerosol output
""""""""""""""""""""""""""""""""""""""""""""""""""""

- **MACv2-SP and JAM are now mutually exclusive aerosol sources** (#640).
  ``echam_physics(aerosol_module="jam")`` no longer also composes MACv2-SP
  (which was a stopgap for the shared ``aerosol`` optics/Twomey diagnostic
  before JAM had its own coupling). JAM now owns the ``aerosol`` slot through a
  minimal ``AerosolCarrySeeder`` and supplies the direct effect via
  ``JamOpticsTerm`` — including the grey two-stream scheme's broadband 550 nm
  profile fields, so grey+JAM keeps a direct effect. With ``jam_optics=False``
  the aerosol is radiatively passive (all-zero optics), a clean A/B control.
  MACv2-SP is unchanged for ``aerosol_module="macv2sp"``.
- **Activation fallback.** In the JAM path the 2M scheme falls back, where
  ARG's ``activated_cdnc`` is empty, to its own ECHAM-HAM minimum-CDNC floor
  (``cdnc_min_fixed`` = 40 cm⁻³, or the dynamic max-radius floor; #674) rather
  than the MACv2-SP SPA floor. The SPA floor remains the ``macv2sp`` + 2M path's
  Twomey link.
- **Breaking: aerosol output variables are renamed into explicit namespaces.**
  MACv2-SP's ``aerosol.*`` output moves to ``macsp.*`` with CF/AeroCom names
  where they exist (``aerosol.aod_total`` → ``macsp.od550aer``); JAM's column
  optics publish under ``jam_optics.*`` (``jam_optics.aod_550`` is the
  band-centre-approx column AOD, distinct from the Mie-based ``od550aer`` of the
  ``aerocom_optics`` pass). The top-level ``aerosol_optical_depth`` key — which
  collided with the unrelated per-band ``RadiationInput`` field — is **removed**;
  its value lives on as ``jam_optics.aod_550``. The internal ``aerosol`` struct
  that radiation and the microphysics read by attribute is unchanged; only the
  output keys move. ``tools/aerocom_cmor.py`` and
  ``tools/release_validation/health.py`` are updated for the new names.

One vertical direction in the output, and CF metadata
"""""""""""""""""""""""""""""""""""""""""""""""""""""

- **Breaking: interface variables are now written surface-first**, the same
  direction as the full-level fields (#710). Previously an output file ran its
  two vertical dimensions in *opposite* directions — ``level`` (mid-levels)
  surface-first, ``level_i`` (interfaces) TOA-first — with nothing in the file
  to say so. Every natural pairing of the two was therefore silently upside
  down: a heating rate computed from the saved radiative fluxes and compared
  against the saved temperature, or a tracer burden mass-weighted with
  ``diff(pressure_half)``, came out vertically reversed with a plausible
  magnitude and no error.
- **Anything reading** ``pressure_half``, ``height_half`` or the radiative
  fluxes (``radiation.lw_flux_up`` and friends) **gets the opposite order to
  before.** Code that compensated for the old mismatch must drop that
  compensation. There is no read-time shim: ``tools/jam_burden_report.py``
  used to detect the disagreement and flip Δp, and no longer does.
- **Old files are not converted and are not supported.** A pre-release
  trajectory is identifiable by ``level_i`` being a bare integer index with no
  attributes; a current one carries descending nominal sigma plus
  ``positive = "down"``. Re-run rather than re-read.
- ``level_interface`` **is gone**: the pyses backend now names its interface
  axis ``level_i``, matching the dinosaur backend, so a reader needs one name
  rather than two.
- **The file is now self-describing.** Both vertical axes are real coordinate
  variables holding nominal sigma (``a/p0 + b``) with CF ``standard_name``,
  ``units``, ``axis`` and ``positive``; the hybrid ``(a, b)`` tables travel
  with the file as the ``hybrid_a_full`` / ``hybrid_b_full`` /
  ``hybrid_a_half`` / ``hybrid_b_half`` coordinates so ``p = a + b·p_s`` is
  reproducible from the file alone, and CF ``formula_terms`` names them.
  ``lat``/``lon``/``time`` and the pressure, height and core prognostic
  variables gain standard names and units. Files are stamped
  ``Conventions = CF-1.11``.
- Observer profile curtains follow the same convention (their ``level`` axis
  was top-first with no coordinate values), and ``jcm.cf_metadata`` is now the
  single place any backend converts the physics-internal frame to the file
  frame. See ``docs/source/design/output_vertical_conventions.md``.
- **New diagnostic** ``pressure_thickness`` **[Pa]** — the per-layer Δp on the
  ``level`` axis, written by the ECHAM physics stacks. Mass-weight a ``level``
  field with ``(field * pressure_thickness / g).sum('level')`` instead of
  reconstructing Δp from ``pressure_half``, which invites the interface/
  mid-level alignment trap (a documented burden example silently evaluated to
  ``0.0``). Present wherever the moist-air prepare term runs; SPEEDY output
  does not carry it.
- **Reading states back is orientation-aware** (#741):
  ``jcm.utils.load_states_from_xarray`` detects a surface-first file from its
  ``level`` coordinate values and always returns the top-first physics frame —
  previously a trajectory file loaded through it came back vertically
  inverted. ``PrescribedStateModel`` output joins the file convention too
  (#739): its ``level`` axis was top-first under the same dim name every other
  product now guarantees is surface-first.
- **Physics diagnostics can carry their own CF metadata** (#740): a
  ``PhysicsTerm`` declares ``output_attrs`` (units, ``standard_name``,
  ``long_name``) for the output keys it provides, next to the code that
  computes them. The radiation flux and heating-rate set, the cloud
  diagnostics and the convection diagnostics now reach the file with units
  and CF standard names instead of empty attributes.

SPEEDY output flattening, hyperdiffusion coverage, and backlog fixes
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- **SPEEDY surface fluxes are published as flat 2D maps** (#645, #328,
  #390). ``ustr``, ``vstr``, ``shf``, ``evap`` and ``rlus`` carried land,
  sea and area-weighted values in a trailing channel axis, so output files
  held ``surface_flux.shf.0/.1/.2``. Only the weighted grid mean ever
  reached the atmosphere, and ``hfluxn`` had no channel for it at all —
  ``hfluxn[:, :, 2]`` clamped to the sea value instead of raising, which a
  coupled run consumed as its grid-mean heat flux. Each of these is now a
  single 2D variable holding the grid mean: ``surface_flux.shf.2``
  **becomes** ``surface_flux.shf``, and the per-surface ``.0``/``.1``
  variables are gone. ``hfluxn`` gains the grid mean it never had. The
  merged values themselves are unchanged.

  A coupled surface model that was reading per-tile heat fluxes out of
  ``hfluxn``'s channels now needs them from its own land/ocean components
  rather than from the atmosphere's diagnostics.

  **Existing SPEEDY checkpoints will not load**, including the ``t31_l8``
  init state on the data mirror — the diagnostic struct changed shape.
  ``load_checkpoint`` now names the file and the reason instead of
  surfacing a bare leaf-count error. Regenerate the state, or pin the
  previous release.
- **Every SPEEDY output variable carries units and a description** (#390).
  The SPEEDY units table never reached output at all: the composable
  physics container defined no table, so all 67 variables shipped bare.
  Terms now declare their own table and the container gathers them, with
  a test that fails if a new diagnostic arrives without a row.
- **ECHAM hyperdiffusion now covers every hybrid grid, including L95**
  (#579). ``diffusion.kind=auto`` matched on ``layers == 47``, so the L95
  middle-atmosphere grids — which exist precisely to resolve the
  stratosphere — silently fell back to SPEEDY's uniform del² profile, and
  pinning an L47 profile on them failed with an opaque broadcast error.
  The ECHAM6.3 ``mo_hdiff.f90::sudif`` tables are now ported in full
  (T31L47, T63L47, **T63L95**, T127L95, T255L95) and the ``setdyn.f90``
  ``dampth`` timescales with them, selected per ``(truncation, layers)``.
  Truncations ECHAM does not tabulate borrow the nearest tabulated
  profile in log space and interpolate ``dampth`` along ECHAM's own
  slope. A hybrid grid that still finds no profile now warns instead of
  falling back silently, and a length-mismatched pin is rejected at build
  time with a message naming the config key and the grid.

  **This changes results on T85L47.** Its base timescale was a hard-coded
  3 h that did not sit on ECHAM's own T63→T127 slope; it is now derived
  as 3.63 h, so damping is slightly weaker. T63L47 — the tuned and
  validated target — is bit-identical, as is every SPEEDY and
  Held-Suarez configuration. T106/T119 at both level counts, and all L95
  grids, move from the SPEEDY uniform profile to their ECHAM profile.
- **Run provenance recorded in every output file** (#591): netCDF global
  attributes (``jcm_prov_*``) carry the git SHA/branch/dirty state of
  every imported editable library, precision flags and devices, the
  resolved boundary files actually opened (size+mtime; content sha256
  with ``JCM_HASH_INPUTS=1``), whether ozone was prescribed or analytic,
  and a single 12-hex run hash for at-a-glance "same setup" checks. A
  ``<output>.nc.provenance.json`` sidecar holds the fully composed
  Hydra config, and one log line at startup summarises SHAs / precision
  / ozone source.
- **Betts-Miller default shallow flavor is now** ``SHALLOWER`` (was
  Isca's nominal ``SIMP``, which zeroes the shallow branch and is always
  overridden in practice — #524). Runs using the default Betts-Miller
  configuration will now do non-precipitating shallow adjustment. The
  flip exposed and fixes a latent NaN reverse-mode gradient in the
  ``SHALLOWER`` boundary-layer division (masked-branch 0/0, the #558
  pattern), previously hidden by the ``SIMP`` default.
- Surface albedo/emissivity per surface type are differentiable
  parameters (``SurfaceOpticsParameters``, #347); defaults unchanged.
- ``TerrainData.from_file`` ignores SSO fields at a different resolution
  than the model grid (deriving them instead) rather than crashing later
  in physics (#578).
- The conservative regridder accepts rectilinear sources (1-D lon/lat
  axes + 2-D area, #533) and remaps them with the exact-overlap first-order
  conservative operator (CDO ``remapcon``'s scheme), exact at any resolution
  ratio; unstructured sources keep nearest-cell binning, but a target cell no
  source centre reaches takes the nearest source value instead of zero.
- **JAX persistent compilation cache on by default** (#592):
  ``$SCRATCH/jcm-jax-cache`` (else ``~/.cache/jcm/jax``), relocatable
  via ``JCM_CACHE_DIR``, disable with ``JCM_CACHE_DIR=off``. Entries
  are keyed on the compiled HLO, so code edits miss rather than
  wrongly hit; reruns of an already-compiled configuration skip the
  multi-minute compile entirely.
- SPEEDY outputs no longer carry the unexplained ``wvi_id`` /
  ``hsg_level`` dimensions (#391); channelized ``units_table.csv``
  entries now say what each channel is (#238); ``Model`` has a
  ``__repr__`` (#322); the JAX-gotchas guide is part of the Sphinx docs
  (#157).
- **A failed checkpoint save leaves the run resumable** (#1006). The chunked
  loop writes the new checkpoint beside the old one and renames it over
  ``run.checkpoint_path``, keeping the old as ``.prev`` only once the new is
  whole; a resume falls back to ``.prev`` (with a warning) when the live file
  is missing or undecodable. **Behaviour change:** a run with
  ``run.checkpoint_path`` that finds nothing to resume but finds its own chunk
  files, archives or checkpoint remnants in the directory now raises instead of
  re-initialising over them; restore a checkpoint or use an empty directory
  (:doc:`design/checkpoint_compatibility`).


New capabilities
^^^^^^^^^^^^^^^^

Headline capabilities of the 3.0 line, then the individual mechanisms.

Interactive aerosol (JAM)
"""""""""""""""""""""""""

- **End-to-end online aerosol.** Prescribed CEDS/biomass emissions plus
  interactive sea salt (Gong 2003), Tegen/HAMMOZ dust and DMS; the MAM4-JAX
  modal microphysics core (``pip install jcm[mam4]``); gas-phase and aqueous
  sulfur chemistry; ARG droplet activation; heterogeneous ice nucleation on
  dust and BC; dry deposition, sedimentation, and in-cloud and below-cloud wet
  scavenging keyed to the two-moment scheme's process-time ledger.
- **Dust emission** follows Tegen et al. (2002) as HAM2 configures it
  (``ndust = 4``), with the saltation threshold gated on relative soil
  wetness: ``forcing.soilw_rel`` carries soil water as a fraction of each
  cell's own field capacity (ERA5 ``swvl1`` over the HTESSEL capacity of its
  soil type), the quantity ECHAM's ``ws/wsmx > 0.99`` cut-off is defined
  against. The channel is optional — a forcing file without it leaves the
  cut-off inert and says so in the log — and the mirror's surface bundles
  carry it (#787). Because the flux lives in the far tail of the 10 m wind
  distribution, HAM's threshold vector is scaled for jcm's own winds by a
  single global multiplier, ``NDUSCALE_JCM_T63_SCALE`` (per run,
  ``physics.jam_dust_nduscale_scale``), set to **0.344** at T63 (fitted
  to the observed dust optical depth by the aerosol retune, below, with the T63
  cloud-cover set in place); the
  regional ratios stay HAM's, and T106 and ne30 take the Fortran's uniform
  default since their inputs are interpolated from T63. A full
  ``echam-jam-t63-l47`` year emits 2351 Tg/yr of D < 10 µm dust, 3.7 times
  the 642 Tg/yr that the parent model's published budget becomes once
  converted to this window and above the AeroCom medians (1640; 1123
  in the 15-model dust intercomparison); the annual
  budget is a release-validation gate on any T63 run of 300 days or more
  (``DUST_EMISSION_TG_PER_YR``, 400-2600 Tg/yr; the calibrated year is 10 %
  below the upper edge) (#808).
- **The release configuration is the T63 cloud-cover set and two aerosol scales
  calibrated with it** (#682). The scales are the Gong sea-salt source, **4**
  (``physics.seasalt.scale``, from 1), and the dust threshold multiplier,
  **0.344** (``physics.jam_dust_nduscale_scale``, from 0.5). They are the best arm
  of a 20-arm Gaussian-process sweep over exactly these two (30-day January and
  July windows) with the cover set of *The cloud-cover defaults at T63 are
  calibrated*, below, held fixed, on a loss over cloud radiative effects, cloud
  cover, precipitation, liquid water path and the ESA-CCI total and dust AOD,
  each against its inter-annual spread. An earlier 40-arm Sobol sweep and a
  25-arm Gaussian-process sweep of four aerosol levers on ECHAM's cover
  parameters had put the same two scales at 0.379095663 and 2; the cover set's
  larger cloud fraction raises wet removal and strips the aerosol those values
  delivered, so the scales are fitted again with it in place. A 365-day T63 L47
  year of the release configuration confirms them. Against the control year
  (ECHAM's cover parameters, dust 0.5, sea salt 1), global AOD at 550 nm rises
  from 0.059 to 0.074 (observed 0.145), dust AOD from 0.0032 to 0.0109 (0.0213),
  the dust burden from 5.3 to 20.9 mg/m² (AeroCom mean 37.6) and the sea-salt
  burden from 6.6 to 16.4 mg/m² (AeroCom mean 14.7, median 12.5); sulphate
  stays inside the AeroCom band (5.3 to 4.1 mg SO4/m², ion basis, against a band
  of 1.95-5.85). **What it gains** on the JAM host: the annual net TOA flux falls
  from +7.4 to +2.7 W/m² (CERES +1.0), the ``cloud_cover`` release gate passes
  (0.52 on the offline overlap, 0.65 as the radiation sees it, against 0.46 and
  0.58), the reflected SW rises from 92.6 to 99.3 W/m² (observed 99.0) and the
  LW CRE from 25.5 to 27.4 (27.9), and the loss on the Stage-2 windows falls from
  469.6 to 325.8. On the 2M host the cover set takes the TOA bias from +9.8 to
  +5.6 W/m² and passes the same gate (0.51). **What it costs:** on the JAM host
  the liquid water path is 57 g/m² against 36 observed (49 for the control) and
  the SW CRE is 3.1 W/m² too strong (-48.8 against -45.7; the control's is 2.9
  too weak); the sea-salt source is four times HAM's (8475 Tg/yr, AeroCom median
  6280) and its burden is 12 % above the AeroCom mean; the total AOD is below
  that of an aerosol-only configuration on ECHAM's cover parameters (0.074
  against 0.082); and the year's POA drift is -0.0039 /day on the six-month line
  of the 40-save recipe (limit 0.002; the whole-year fit with the annual harmonic
  gives -0.0025 against its limit of 0.003, and every aerosol gate passes under
  it). The year's results are tabulated in
  :doc:`design/jam_aerosol_retune`. The DMS, wet-removal,
  convection and microphysics defaults are unchanged: the 1M and 2M levers
  were swept as well: the 2M's optimum
  sat on the lower edge of four of its five levers, the 1M's best arm took
  `cprcon` to the edge of its range, and both bought cloud radiative effect
  with liquid water path. What the calibration cannot fix is
  structural and is listed under *Known limitations*; the design page records
  the targets, the stages and the evidence.
- **Aerosol direct radiative effect from the modal population**: per-band Mie
  optics integrated over each mode's lognormal size distribution and fed to
  RRTMGP, with a broadband 550 nm path so the grey two-stream scheme keeps a
  direct effect. ``jam_optics=False`` makes the aerosol radiatively passive,
  which is a clean A/B control rather than a disabled feature.
- **JAM is composed through** ``physics=echam-jam*``, which is factory-built
  (``builder: echam_physics``) because the aerosol chain splits around the
  cloud term. It requires the two-moment cloud scheme; JAM with the one-moment
  scheme is rejected at compose time.

Dynamical cores and grids
"""""""""""""""""""""""""

- **New pySES CAM-SE backend** (:class:`jcm.dycore.pyses.PysesCamSEDycore`,
  ``pip install jcm[pyses]``): spectral elements on the cubed sphere, float64
  dynamics driving float32 column physics, coupled through a pg2
  finite-volume physics grid, with multi-GPU element sharding and a
  frontogenesis physics-fields provider. Selected from Hydra with
  ``dycore=pyses_ne30l{47,95}`` or the ``+configuration=ma-ne30-l{47,95}``
  presets. See :doc:`design/pyses_cam_se_dycore`.
- ``physics=speedy`` runs on the pySES backend (#797). Under pySES's
  float64-dynamics / float32-physics split the forcing and SPEEDY's cached
  vertical tables stay float64, so the ``lax.cond`` branches that recompute
  from them (the shortwave cloud and flux caches, the near-surface humidity
  blend) came out float64 against a float32 pass-through branch and the
  first physics step raised a ``TypeError``. The recomputed branches are
  pinned to their operand's dtypes (``jcm.utils.cast_like``), and the scatters
  that write those float64 values into float32 fields (lowest-level surface
  tendencies, longwave boundary temperatures, shortwave fluxes, large-scale
  condensation) cast explicitly, which JAX deprecates doing implicitly. Both
  are no-ops in an all-float32 or all-float64 run.
- **Semi-Lagrangian is the Dinosaur backend's default tracer transport.**
  Every extra tracer rides nodally with a Bermejo-Staniforth quasi-monotone
  limiter, so aerosol non-negativity is structural in transport rather than
  imposed afterwards. The top-level ``+advection`` switch is gone;
  ``dycore.advection`` (default ``null`` = the physics decides) selects the
  scheme; tracer-carrying physics defaults to semi-Lagrangian (an explicit
  Eulerian with tracers warns). Tracer-free
  SPEEDY declares the Eulerian core it was formulated on, so SPEEDY runs keep
  2.x transport and CPU speed (semi-Lagrangian cost ~4x the SPEEDY step on
  CPU) — see :doc:`design/dinosaur_transport_selection`. ``diffusion.tracer_positivity`` survives, defaulting to ``auto``
  (on for JAM), but only as a mass-conserving hole-filler at the
  dynamics-to-physics boundary — see
  :doc:`design/dinosaur_sl_jam_configuration`.
- **ECHAM6 middle-atmosphere L95 vertical table** (lid ~0.01 hPa) with
  T63/T106/T119 grid presets and matching ECHAM hyperdiffusion profiles, plus
  an ``ne30`` L95 dycore preset.
- **The Hydra** ``dycore`` **group selects the backend**, dispatched by
  ``jcm.runners.build_model``; a whole pySES run is one command.
- **ECHAM T127 and T255 grids** — *supported, not validated*.
  ``get_coords(spectral_truncation=127|255)`` builds ECHAM's own 384×192 /
  768×384 Gaussian grids (dinosaur has no ``Grid.T127``/``T255`` factory; they
  are constructed like T63), with ``grid=echam_t{127,255}_l{47,95}_hybrid``
  presets. Every climatological and static input is on the data mirror for
  them; they are outside the release matrix and nothing is tuned for them —
  see :doc:`science/configurations`.

Diagnostics and output
""""""""""""""""""""""

- **AeroCom phase-4 diagnostic suite** with CMOR post-processing
  (``tools/aerocom_cmor.py``), and the CALIPSO and MODIS satellite simulators
  alongside CloudSat, including COSP joint histograms (``clmodis`` tau/Reff,
  LWP+IWP/Reff, the lidar scattering-ratio CFAD and ISCCP). The 10 m wind
  ``uas``/``vas`` applies the vertical-diffusion term's stability-corrected
  10 m reduction, the one the surface-exchange contract's wind uses, to the
  post-physics lowest-level wind, so it shares the time level of the other
  AeroCom winds (#911).
- **AeroCom number and PM diagnostics take each mode's width from the aerosol
  spec**: ``aerocom_N70``/``aerocom_N100`` and ``aerocom_PM1``/``aerocom_PM10``
  integrate every mode with its own ``geom_std_dev``, and an explicit
  ``AerocomDiagnostics(mode_sigma_g=...)`` lists one width per mode in the order
  of the spec's modes. Values from earlier development builds used swapped
  Aitken/accumulation widths — about 7–8 % high in near-surface N70/N100 and
  about 16 % high in PM1 on a spun-up T63 JAM state; PM10 and the model state
  are unaffected (#917).
- **Virtual observation operators** — stations, tracks and solar-time swaths —
  sampled every model timestep, each producing its own output dataset. See
  :doc:`design/observers`.
- **Per-species, per-mode and per-wavelength aerosol optics**, microphysical
  process-rate and emission-flux diagnostics, and per-species aerosol
  mass-budget terms (``budget_mass_*`` / ``budget_ptend_*`` / ``budget_dyn_*``)
  that make the transport residual visible.
- ``tools/jam_burden_report.py`` reports column burdens against climatological
  anchors for any dycore and grid, with inferred per-species lifetimes from an
  emissions file.

Radiation, clouds and gravity waves
"""""""""""""""""""""""""""""""""""

- **Climatological ozone is the default** (``forcing.ozone_file: auto``). The
  analytic profile it replaces carried roughly 7.6x the climatological
  *tropospheric* ozone column and biased clear-sky OLR about 12 W/m² low. It
  is still selectable as ``forcing.ozone_file=analytic``; ``auto`` raises on a
  hybrid grid it cannot resolve rather than substituting it silently, and
  falls back with a warning only on sigma grids. T63 is the only grid whose
  climatology is packaged with the wheel; the others come from the data-mirror
  bundle.
- **New** ``radiation.total_cloud_cover`` **diagnostic**: cover as the McICA
  sub-columns see it, under the same overlap rule the flux solve integrates.
  Under the grey two-stream scheme it is the weight of the cloudy beam (the
  column's largest cover under maximum-random and exponential overlap), and
  the NN emulator publishes the analytic expectation of the McICA draw
  rather than sampling it.
  This is *not* the same number as ``clouds.total_cloud_cover`` below, which
  the release-validation gate scores — see :doc:`design/cloud_cover_gate`.
- **New** ``clouds.total_cloud_cover`` **diagnostic**: ECHAM's total cloud cover
  ``aclcov`` (``mo_cloud.f90`` section 10.2), the maximum-random overlap of the
  step's final cloud fraction, computed in the model every step. Under
  ``run.output_averages`` the saved frame is its time mean over the output
  interval, which is ECHAM's accumulation; the overlap of a saved *mean* profile
  (``jcm.analysis.total_cloud_cover``) is the offline approximation, usually lower, for
  output that lacks the field. The release-validation ``cloud_cover`` gate
  scores the online field when the file has it (and prints the observed
  reference beside it), and falls back to the offline overlap with a note when
  it does not. Checkpoints written before the field existed resume with it
  seeded to zero until the first step.
- **CAM spectral frontal gravity-wave drag**, selectable with
  ``gw_scheme="frontal"`` or ``gw_scheme="both"`` to run it alongside Hines.
- **Per-level precipitation flux profiles** and a CloudSat COSP warm-rain
  hook.

Coupling to an external surface component
"""""""""""""""""""""""""""""""""""""""""

- **A package-independent surface-exchange contract.** Every physics package
  that resolves a surface publishes a
  :class:`~jcm.physics.surface.surface_exchange.SurfaceExchange` struct under
  ``diagnostics["surface_exchange"]`` — net downward heat flux, sensible and
  latent heat, evaporation, total precipitation, wind stress, near-surface
  wind, and lowest-level air density / potential temperature, with one
  documented sign convention (turbulent fluxes positive up, net heat flux
  positive down). Grid-mean fields are guaranteed; per-tile and rain/snow-split
  fields are optional and absent (not zero) where a package cannot fill them
  faithfully. SPEEDY and ECHAM publish it; Held-Suarez opts out;
  ``ComposablePhysics.require_surface_exchange()`` fails a coupler fast at
  composition time (#754).
- **The near-surface wind vector on the contract.** ``SurfaceExchange``
  carries the eastward/northward wind ``wind_u``/``wind_v`` at the same
  reference as ``wind_speed``, which each package keeps as its own and names
  in the static ``wind_reference`` field and the netCDF attributes: ECHAM's
  stability-corrected 10 m wind (``"10m"``), SPEEDY's ``fwind0``-scaled
  lowest-level wind (``"lowest_level"``). ECHAM also fills the optional
  per-tile ``wind_u_tile``/``wind_v_tile``/``wind_speed_tile`` with
  ``tile_fraction`` (water, sea ice, land); ``SurfaceExchange.validate()``
  checks the vector and tile invariants. ``ForcingData.ocean_u``/``ocean_v``
  are reserved for a coupled ocean surface current but are **not yet used**:
  the vertical diffusion still takes the stress against a surface at rest
  (#911; implementation tracked in #915).
- **Forced surface mode.** ``physics=speedy-forced-flux`` /
  ``physics=echam-forced-flux`` deliver externally prescribed sensible-heat,
  evaporation and momentum fluxes in place of the package's own surface
  exchange, entering the same tendency pathways (SPEEDY's bottom-level source;
  ECHAM's ``TteTkeVerticalDiffusion(couple_surface=False)`` plus an explicit
  ``PrescribedSurfaceFlux`` term). Fluxes ride
  ``forcing.prescribed_surface_flux`` (a ``constants`` block or a grid file
  whose climatology-vs-dated alignment is declared by its ``align`` key) or the
  ``ForcingData.prescribed_*`` fields a coupler sets directly, in the
  published contract's units and signs (#301). Prescribed fluxes supplied to a
  composition with no forced-mode consumer are rejected rather than silently
  ignored, and a date-aligned flux archive must cover the run (its CF
  ``time_bnds`` when present). See :doc:`design/surface_exchange`.

Mechanisms
""""""""""

The individual public mechanisms behind the capabilities above.

JAM optics: a per-mode backend seam
"""""""""""""""""""""""""""""""""""

- ``JamOpticsTerm`` exposes the one genuinely optical step as a hook, so an
  out-of-tree Mie pathway is a subclass implementing ``_mode_optics`` rather
  than a fork of the whole term (#791). ``_map_bands`` and ``_build_mie_lut``
  are overridable alongside it, for a backend whose per-band intermediates
  are large or that never reads the built lookup table. ``ModeOpticsInputs``
  carries both normalisations — a column number per area and a total volume,
  with the ``col_factor`` that converts between them — because backends
  disagree about which one they predict. The hook returns optical depths
  rather than ``(k_ext, ssa, g)``, which keeps the base class from dividing
  by a possibly-zero number or volume. Compose one with
  ``echam_physics(aerosol_module="jam").replace("aerosol_optics", ...)``.
  The default pathway is untouched and its answers are unchanged; see
  :doc:`design/jam_optics_mode_seam`.

Public state and transformed-output contracts
"""""""""""""""""""""""""""""""""""""""""""""

- ``Model.initial_state()`` and ``Model.initial_physics_carry()`` return fresh
  dycore/carry pytrees for external steppers. ``bootstrap_state()`` now returns
  the pair it installs, ``dycore_state`` and ``physics_carry`` expose the
  resumable pair read-only, and checkpoint restore replaces both atomically
  (#755).
- ``ModelPredictions.with_context(model)`` reattaches the static coordinates,
  physics, dycore and observer metadata intentionally omitted at JAX pytree
  boundaries. The explicit ``with_context(coords, physics, ...)`` form supports
  custom drivers; re-derived live parameters are labelled so they cannot be
  mistaken for trace-time provenance (#756).
- ``ModelPredictions.is_interval_mean()`` is the public reading of whether a
  trajectory's frames are interval means (#907). A coupler scanning
  ``run_from_state_with_carry`` over chunks gets the per-trajectory flag
  stacked with every other leaf; the accessor returns it when the chunks
  agree and raises ``ValueError`` when a stack mixes means with
  instantaneous samples. ``time_labels()`` and ``to_xarray()`` read the flag
  through it, so a stacked trajectory labels directly (keeping its chunk
  axis) and serializes once the chunk axis is merged into time — no private
  field has to be rewritten. See :doc:`advanced_features`.

Public model clock conversion
"""""""""""""""""""""""""""""

- :meth:`jcm.model.Model.date_from_sim_time` is now the public, JIT-safe way
  to convert elapsed simulation seconds into the same :class:`jcm.date.DateData`
  used by forcing and physics. It documents the stop-gradient boundary,
  nearest-second date rounding and day rollover, and the independently
  timestep-derived ``model_step``. ``Model._date_from_sim_time`` remains a
  compatibility alias in 3.0 and is planned for removal in a later release
  (#758).

Scheme parameters on the factory-built presets
""""""""""""""""""""""""""""""""""""""""""""""

- The factory-built physics presets (``physics=echam-jam``,
  ``echam-forced-flux``) take per-scheme parameters from the command line,
  as the term-list presets always have (#933). The block is the
  ``echam_physics`` argument: ``+physics.convection.entrpen=4e-4``,
  ``+physics.radiation.cloud_inhomogeneity_liquid=0.7``. Each field is
  applied on top of the object the factory would otherwise build, so the
  factory's own choices for the other fields (the JAM radiation defaults,
  ``cu_lmfmid``) are kept. Both preset styles share one conversion
  (:func:`jcm.physics.physics_term.with_field_overrides`): an unknown field
  is an error listing the valid fields (on the term-list presets too, where
  it used to surface as a bare ``TypeError``), and numeric fields stay
  differentiable pytree leaves. A mapping for a scheme the composition does
  not include (``microphysics`` under ``cloud_scheme: 2m``, the MACv2-SP ``aerosol``
  under ``aerosol_module: jam``) and ``cu_lmfmid``
  set both as the scalar flag and in ``convection`` are rejected. In Python,
  ``echam_physics`` accepts the same mappings in place of ``Parameters``
  objects.
- The JAM sea-salt and DMS emission scales, the other calibration levers of
  the retune (#682), are reachable the same way: ``+physics.seasalt.scale=``
  (Gong sea salt) and ``+physics.dms.flux_scale=`` (Nightingale DMS) on the
  factory-built JAM presets, or ``seasalt=`` / ``dms=`` of ``echam_physics``,
  a mapping or a ``Parameters`` object. They need ``aerosol_module="jam"``
  and are rejected without it rather than ignored.
- The other JAM process schemes take their parameters the same way (#995),
  chiefly wet deposition, the retune's reserve lever on the aerosol burden:
  ``+physics.wetdep.incloud_scale=`` (the multiplier on stratiform in-cloud
  removal) and ``+physics.wetdep.impact_scale=`` (on the inertial-impaction
  efficiency of below-cloud removal). The convective in-plume scavenging,
  which the convective tracer transport owns, is scaled as a whole by
  ``+physics.conv_transport.conv_scav_scale=`` (default 1: HAMMOZ's per-mode
  ``csr_conv`` times the scale, at most one). The arguments of
  ``echam_physics`` are ``wetdep``, ``drydep``, ``sedimentation``,
  ``activation``, ``oxidants``, ``sulfur_gas``, ``aqueous``,
  ``cloud_borne_exchange``, ``tracer_diffusion``, ``conv_transport`` and
  ``anthropogenic_params``, each a mapping or a ``Parameters`` object, with
  the same rules as ``seasalt``: an unknown field is an error listing the
  valid ones, and an argument whose scheme is not composed is rejected
  (without ``aerosol_module="jam"``; ``anthropogenic_params`` without
  ``jam_anthropogenic``; ``cloud_borne_exchange`` without ``jam_cloud_borne``;
  ``conv_transport`` without ``jam_convective_transport``).
  A valid override of a field no run reads warns instead
  (``wetdep.conv_scav_ratio`` with convective tracer transport; the fallbacks
  ``oxidants.o3_fallback_vmr``, ``oxidants.cos_zenith_fallback``,
  ``activation.updraft_default`` and ``drydep.u_star_default``), and the
  ``oxidants`` proxies are read only when the run supplies no oxidant
  climatology (``forcing.oxidants_file``, ``auto`` by default).
  ``incloud_scale`` scales the stratiform in-cloud removal only. The dust
  parameters beyond the ``jam_dust_*`` flags have no such argument yet (#995).
- A numeric override takes the dtype and shape of the field it replaces, on
  both preset styles (:func:`jcm.physics.physics_term.with_field_overrides`).
  The 1M cloud scheme casts its parameters to the state's precision, so
  ``++physics.terms.echam_1m_microphysics.params.ccraut=20.0`` (likewise
  ``ccsaut`` and ``ccsacl``) stopped with ``AttributeError: 'float' object has
  no attribute 'astype'`` before the first step; the override is now the
  array its default is. A scalar field takes a number, a profile field a list
  of its length, and an integer or boolean field only a value it can
  represent. A value of another shape is an error naming the config key, never
  a broadcast (#682 retune pilot).

Provenance records the parameters
"""""""""""""""""""""""""""""""""

- **Every output now records the physics parameter values the run
  actually used** (#732). The composed Hydra config that #591 stamped is
  not the same thing: each scheme's ``params`` block is deliberately
  absent from the shipped yamls so unspecified fields fall back to
  ``Parameters.default()`` in code, meaning the config recorded the
  *overrides* and said nothing about the effective values, and a model
  built in Python or one whose parameters a calibration loop replaced
  had no config behind it at all. ``jcm_prov_params`` (with
  ``jcm_prov_params_sha``) now carries them, read off the *built*
  physics, keyed as ``<term>.<variable>.<field>``
  (``tiedtke_convection.params.entrpen``). Read it with
  ``jcm.provenance.read_params(ds.attrs)``, or off the predictions object
  as ``predictions.params``. Everything else about a run stays where it
  was: the term composition, dycore and resolution are already in the
  config record this sits beside.
- Both kinds of parameter variable are covered. An ``nnx.Param`` is
  recorded in full, including tuned arrays such as the MACv2-SP plume
  shapes. A plain ``nnx.Variable`` is recorded where it is knob-shaped
  (scalars, 0-d arrays, structs of those), because a parameter block
  holding a bool cannot be a ``Param`` — ``SpeedySurfaceFlux.surface_params``,
  ``EchamSurface.params`` and every Held-Suarez tuning constant are plain
  Variables — while the coordinate caches terms also hold as Variables
  stay out. Arrays over 64 elements (embedded NN weights) are summarized
  by shape, dtype and hash; values captured under ``jit``/``grad`` read
  ``"<traced>"``.
- **The record is captured at trace time, not from the live module.**
  ``Model._run_from_state`` is jitted with ``self`` static, so parameters
  are constants inside the compiled executable and changing one in place
  afterwards does not reach the computation. Reading the module at the
  handoff would therefore stamp a trajectory with values that never ran.
  Where the live values disagree with the compiled ones, the record
  reports the compiled ones and a ``live_parameters_differ_from_compiled``
  key says so (as does the warning below): that disagreement means an
  in-place parameter change may not have reached the run. Build the physics
  anew to change parameters.
- **An in-place parameter change after the physics has run now fails
  loudly** (#735). The change still does not reliably reach a later run —
  of that model, or of a new ``Model`` built on the same physics object at
  the same grid, since each checkpointed term's trace is cached — so the
  next such run raises a ``UserWarning`` before it starts, naming each
  changed field once per model; a sensitivity loop that edits one physics
  object therefore hears about it on its second iteration instead of
  returning a flat response. Where a new Model can reuse those traces, the
  first-compiled parameter record is the physics object's, so the record
  holds the values the traces were built with, flagged; a physics without
  per-term checkpointing (Held-Suarez), or a new grid, retraces with the
  live values and neither warns nor flags. An edit made before the physics
  first runs is simply the value it compiles with. Making edits take effect would need the
  parameters passed through the jit as traced arguments, a change to the
  compiled hot path; the supported loop builds the physics and the ``Model``
  inside one ``jax.jit`` (:doc:`advanced_features`).
- The record travels on the predictions object, so it reaches every
  output stream that object produces (trajectory, snapshots and the
  per-observer datasets), including a bare
  ``model.run(...).to_xarray().to_netcdf(...)`` that never touches the
  Hydra runners, and a later run cannot retroactively change an earlier
  one's record.
- ``jcm_prov_run_hash`` **values change**, because the parameters are now
  folded into the hash. They have to be: every member of a parameter
  sweep shares one code state, config and input set, so without them a
  sweep produced a single run hash for every member.

Transient AMIP forcing, ERA5 nudging and ERA5 initial states
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- **Historical (AMIP-style) runs from config** (#610): yearly transient
  bundles on the data mirror (``bundles/<grid>/forcing_amip/<year>.nc``
  with PCMDI-AMIP mid-month SST/sea-ice, ERA5 land climatology and
  CR-CMIP global-mean GHGs; matching ``emissions_amip`` and
  ``ozone_amip`` files), a ``forcing=amip`` preset
  (``forcing.years=[first,last]`` expands ``{year}`` patterns and
  concatenates along time), a ``run.start_date`` key so the model
  calendar lands on the forcing dates, and a ``by_date_interp``
  time-alignment mode that linearly interpolates between samples —
  required for the AMIP boundary (``tosbcs``) convention to reconstruct
  observed monthly means. Plain ``by_date`` series stay
  piecewise-constant.
- **ERA5 nudging and initial conditions from config** (#610):
  ``nudging=era5`` relaxes winds (optionally temperature) toward
  WeatherBench2's public cloud ERA5, windowed to the run dates,
  regridded to the model grid and cached locally (``jcm.data.era5``,
  ``pip install jcm[era5]``); ``init=era5`` starts from the ERA5 state
  at ``run.start_date``. Nudging is masked off above the WB2 stores'
  50 hPa top and below ``nudging.pbl_levels``. Prefetch CLI:
  ``python -m jcm.data.era5 --grid <grid> --start <d0> --end <d1>``.

Boundary-condition and emissions data mirror
""""""""""""""""""""""""""""""""""""""""""""

All boundary conditions and emissions now come from the Hugging Face
dataset ``climate-analytics-lab/jax-gcm-data`` (issue #515), buildable
end-to-end with ``python -m jcm.data.mirror.build_mirror`` and
reachable from any config path via the ``hf://`` prefix (see
``docs/source/design/data_mirror.md``). **Runs forced from the mirror
differ scientifically from the packaged/prepared files** — intended
corrections, listed here because they change climate:

- ``soilw_am`` and ``snowc`` are computed with the
  ``jcm.data.bc.compile`` fraction formulas from ERA5 sources. Land
  means move from 0.161 → 0.54 (soil availability) and 0.019 → 0.115
  (snow cover): the packaged climatology was systematically dry/
  snow-poor, consistent with a source-unit mismatch in its original
  derivation. Expect wetter land, more evaporation and stronger snow
  albedo.
- Anthropogenic emissions are sector-resolved: ``elevated_industrial``
  (CEDS ENE+IND, ~50 m injection — 81 % of anthropogenic SO2) and
  ``shipping`` are separate channels instead of being emitted at the
  surface.
- SSO fields derive from the GMTED2010 30″ DEM with the gradient tensor
  on 10′ block means (calibrated against the ECHAM T127 reference);
  the packaged T63 ``orosig`` was ≈0 everywhere, so SSO drag
  strengthens.
- Ozone is the CMIP7 FZJ product (real mesospheric decline), oxidants
  the WACCM CCMI full-lid decade climatologies (real H2O2 and
  mesospheric values; required for L95), dust erodibility the
  0.23×0.31° source, and the land mask is fractional (coastal cells and
  small islands retain orography).
- PI (1850s; SST/ice = 1870–1879 mean) and PD (2005–2014) eras ship for
  every product.
- The mirror publishes ``t127`` and ``t255`` bundles (terrain, PI/PD
  forcing, emissions, DMS, dust, and ozone/oxidants at L47/L95); the yearly
  transient series stay at ``t63``/``t106`` (#888). The five Tegen dust inputs
  now come from the ECHAM-HAMMOZ pool at every resolution it ships — native at
  T63, T127 and T255 — and the **t106 dust bundles change**: they are
  conservatively coarsened from the native T255 files instead of
  nearest-neighbour refined from T63 (area means now match T63 to round-off;
  RMS departure from a T127-derived reference drops ~3×), and the region mask
  is regenerated from HAMMOZ's own lon/lat-box recipe (566 of 51,200 T106
  cells change region along box edges). The builder runs on NCAR Glade or DKRZ
  Levante (``jcm/data/mirror/sites.py``) and adds a grid with ``--grids``
  without rebuilding the others.
- **The t63/t106 emission bundles change** (``emissions_{pi,pd}`` and every
  ``emissions_amip`` year): they are remapped with the exact-overlap
  conservative operator instead of nearest-centre binning, which carried
  1-3 % global-mean flux errors and misplaced point sources by a cell (up to
  ~55 % of a field's maximum locally). Global-mean fluxes now agree across
  every published grid to round-off; the release-validated T63 JAM
  configurations see correspondingly changed emissions.
- The native ne30pg3 terrain published as ``bundles/ne30pg3/sso.nc``
  carried a DEM-validity placeholder ``lsm`` (99.8 % land) instead of a
  land-sea mask; it is replaced by the assembled
  ``bundles/ne30pg3/terrain.nc`` (CESM ``LANDFRAC`` land fraction, exact
  GLL orography), and the pySES ``build_terrain`` now rejects any
  terrain file averaging >0.9 land as a placeholder (#596).
- Every mirror read is pinned to one dataset commit (``MIRROR_REVISION`` in
  ``jcm/data/remote.py``; ``JCM_MIRROR_REVISION=<commit sha>`` overrides it,
  and a branch name is refused). Two machines therefore no longer read
  different copies of a republished bundle depending on their caches. Runs,
  checkpoints, release-validation launches, benchmarks and fixture bands
  record the commit. The pin is the 2026-09-28 commit whose forcing bundles
  carry the land-surface convention below (``lsm``, ``forest``, ``glac``) on
  top of the conservatively remapped t63/t106 ``emissions_{pd,pi}``, so a
  cache holding earlier bundles re-fetches once; prefetch before running
  offline.

Importing jcm does not touch the GPU
""""""""""""""""""""""""""""""""""""

``import jcm`` — and importing any jcm module, including the physics packages
directly — no longer initialises a JAX backend (#859). Previously the SPEEDY
sigma tables on ``jcm``'s import chain were built as jax arrays at import, the
first device query of the process, so on a GPU host a bare ``import jcm``
brought up CUDA and, under JAX's default ``XLA_PYTHON_CLIENT_PREALLOCATE``,
claimed 75 % of the card (61,222 MiB of an 80 GB A100) in a process that might
never do device work — an orchestrator whose integrations run in subprocesses,
or a REPL inspecting output. The device is now first touched when a model or
physics term actually builds arrays. Module-level tables (the SPEEDY and
Held-Suarez sigma boundaries, the grey cloud-optics band tables, the TTE
Businger-Dyer roughness tables) are stored as Python floats and materialised
where used, and ``def`` defaults that were jax arrays are ``None`` or numpy
scalars. Two consequences: ``physical_constants.SIGMA_LAYER_BOUNDARIES`` and
``held_suarez.utils.DEFAULT_SIGMA_BOUNDARIES`` are now tuples of floats (use
``compute_sigma_boundaries(nlev)`` for an array), and the dtype of those tables
follows the ``jax_enable_x64`` setting in force when they are used rather than
whichever was in force at import. ``jcm/import_side_effects_test.py`` enforces
the property for every module.

ECHAM physics traces with 64-bit mode on
""""""""""""""""""""""""""""""""""""""""

With ``jax_enable_x64`` on, which importing ``mam4_jax`` does unless
``MAM4_JAX_ENABLE_X64=0`` is set, every ECHAM configuration failed at trace
time (#945). The Tiedtke no-convection state built its cloud-base and
cloud-top indices as int64 while the convecting branch returned int32.
Separately, the RRTMGP aerosol-free companion with
``aerosol_free_interval > 1`` returned float32 fluxes from the solve branch and
float64 fluxes from the hold branch. The convection indices are now int32 in
every branch. The companion's fluxes and fractions now keep the dtype of the
slots they fill. A fast test steps the composed package under x64, so CI covers
this without the ``mam4`` extra. The float32 forward result is bit-identical.

Reference-exact values with surrogate derivatives
"""""""""""""""""""""""""""""""""""""""""""""""""

- ``jcm.physics.surrogate_gradient.with_surrogate_gradient(exact, surrogate)``
  returns a function whose value is ``exact``'s, bit for bit, and whose
  derivatives in forward and reverse mode are those of ``surrogate``,
  obtained by differentiating ``surrogate`` itself rather than from a
  hand-written rule. A non-smooth point inside the range a scheme visits (a
  clip, a phase switch, a power law with an unbounded slope) keeps the
  reference value and still gives an optimiser a bounded, informative
  derivative. A surrogate's width is a static (``pytree_node=False``) field of
  the scheme's parameters, and a width of 0 selects the reference derivative.
  ``jcm.testing.check_surrogate_gradient`` checks such a function: its value
  equals ``exact``'s, its jvp and vjp equal ``surrogate``'s, and its two AD
  modes are adjoint. The ECHAM cover, the 1M scheme and the Tiedtke-Nordeng
  convection use it (see the corrected physics entries). See :doc:`design/surrogate_gradients`.

JAM runs float32 physics under 64-bit mode without mixed-dtype scatters
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With ``jax_enable_x64`` on and float32 physics (pySES, or any run that imports
``mam4_jax``), the float parameters are float64 while the state is float32, and
three JAM/ECHAM sites scattered float64 values into float32 tendencies — a JAX
``FutureWarning`` now and an error in later JAX releases (#770): the DMS
emission, the dry deposition of every deposited moment, and the ECHAM land
surface's soil heat tendency. Each value is now pinned to its operand's dtype.
A fast test traces the ECHAM+JAM package under x64 with ``FutureWarning`` as an
error. Results with x64 off are unchanged.

Finite parameter gradients at degenerate inputs
"""""""""""""""""""""""""""""""""""""""""""""""

Several schemes returned a NaN derivative — with respect to the state or to a
tunable parameter — at an ordinary degenerate input, although their value was
fine (#663): JAM activation and ice nucleation at zero TKE (activation's
``tke_factor`` among them), JAM in-cloud sulfur chemistry in every clear-sky
cell, the TTE-TKE Businger-Dyer surface-layer option on a stable surface,
Lott-Miller SSO on an exactly calm column and just above its orography-std
floor, Hines' dissipation and diffusion coefficient (``spectrum_width_factor``
among the affected parameters), and RRTMGP with the sun exactly overhead.
Each is a guard on the branch its value discards, so forward results are
bit-identical. RRTMGP's ``cloud_decorrelation_km`` no longer blocks tracing the
radiation parameters. A slow test now differentiates every term's outputs with
respect to every float parameter of the ECHAM, ECHAM+JAM (2M), grey and SPEEDY
packages.

Finite gradients under ``jax.jit`` through the tracer mass fixer
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

``jax.jit`` of the ECHAM two-step ``d(mean T)/d(solar constant)`` with the 1M
cloud scheme returned NaN while the eager gradient was finite (#987). The
dycore's semi-Lagrangian tracer mass fixer rescales each tracer by
``target / current``, the ratio of its global mass before and after transport,
behind a mask that the totals are positive. The reverse rule of that quotient
needs ``current**-2``, which is ``inf`` in float32 below about 1.08e-19, far
above the mask's 1e-30. In a compiled program the 1M composition's ice
tendency in clear cells is a rounding residue (~1e-29 kg/kg/s; op-by-op
evaluation gives exactly zero) whose global total, ~3e-21, passes the mask, and
a zero cotangent times the ``inf`` was NaN over the whole dynamical state. The fixer, and the column hole-filler
``jcm.filters.mass_conserving_positivity`` (whose gradient was also NaN in any
column with no positive mass and ``inf`` in one holding only a residue), now
divide through ``jcm.filters.stable_quotient``: the plain quotient, with its
exact derivative evaluated without the square. Forward results are
bit-identical. The two-step gradient under ``jax.jit`` now agrees with the
eager one to float32 rounding (1.19573e-05 and 1.19572e-05) for both cloud
schemes.


Corrected physics
^^^^^^^^^^^^^^^^^

Fixes that change the climate of a configuration you did not otherwise touch.
:doc:`v2_to_v3` quotes the measured direction and magnitude for each, where one
was measured.

Sub-grid orographic drag acts on the low-level flow, not the whole column
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The ECHAM configurations had jets about half their observed strength and
  net easterly low-level flow (measured in the 1M and JAM T63L47 years alike):
  a year had zonal wind 9 m/s at 200 hPa
  and −3.3 m/s at 850 hPa in the global mean (ERA5 15.6 and +1.0), the
  equator-to-60° 500 hPa height drop a quarter (NH) to a half (SH) of ERA5's,
  and no Southern Ocean trough. The Lott-Miller SSO drag ran with
  ``nktopg = 1``. In ECHAM ``nktopg`` is a grid level that ``sugwd`` sets
  (level 45 of L47) and ``orosetup`` applies as ``kknu = MIN(kknu, nktopg)``,
  a floor on the depth of the low-level layer; at 1 every such layer reached
  the model top, so the "incident" wind, stability and density were column
  means. The scheme then removed 0.052 N/m² of westerly momentum (area mean of
  the column force × cos φ; the surface friction torque was 0.026), with
  column forces up to 10 N/m² and drag of −12 m/s/day at 300 hPa over the
  mountains, and the atmosphere settled into net surface easterlies to balance
  it. ``LottMillerSso`` now takes ``nktopg`` from the model grid
  (``echam_nktopg``), and ``sso_drag`` requires it. The port then reproduces
  the compiled ECHAM ``ssodrag`` on real T63L47 columns to round-off.
- **Changes results** in every configuration that composes ``LottMillerSso``
  (all ``echam_physics`` packages). From the ``ma-t63-l47`` warm state, days
  20-30 against ERA5 January: 200 hPa wind bias −7.8 → −1.8 m/s (RMSE 16.4 →
  9.5), 850 hPa −4.3 → −0.6 m/s (RMSE 6.9 → 4.4), 500 hPa height RMSE 180 →
  73 m; the NH subtropical jet reaches 46 m/s at 30°N (ERA5 44). Warm states
  spun up before this change are out of angular-momentum balance and adjust
  over about four weeks.
- ``sso_drag`` no longer multiplies the drag by the cell's land fraction.
  ``mo_ssortns.f90::ssodrag`` takes no land mask: the sub-grid descriptors are
  statistics over the whole cell with its ocean part at zero elevation, so a
  coastal cell's land fraction is already in them. The ``land_fraction``
  argument is removed. **Changes results** in coastal cells only.

The upper sponge is ECHAM's
"""""""""""""""""""""""""""

- ``UpperSponge`` damped the full (u, v) over the top ten levels of
  ``run=longrun`` (down to 5.2 hPa, 1.5 h at the top) and relaxed T toward an
  absolute 250 K. ECHAM's ``uspnge`` (``mo_upper_sponge.f90``) damps only the
  m ≠ 0 part of vorticity, divergence and temperature, at the top level, with
  ``spdrag = 0.926e-4`` s⁻¹ (3 h) applied implicitly. The term now does exactly
  that in grid-point space (implicit damping of the zonal anomalies of u, v and
  T), ``run=longrun`` uses ECHAM's single-level 3 h profile, and the term's
  defaults are ECHAM's. The absolute target is removed (``run.sponge.target_T_K``
  is no longer a key); ``timescale_h``/``enspodi`` default to null, meaning
  ECHAM's values. The sponge refuses a non-lon-lat grid, and the pySES backend
  refuses a positive ``run.sponge.levels`` (it has ``dycore.lid_sponge``).
- **Changes results** above ~5 hPa in every ``run=longrun`` configuration: the
  zonal-mean stratospheric and mesospheric jets are no longer damped (30 days
  from the ``ma-t63-l47`` warm state: 1 hPa wind at 40°N 21 → 39 m/s, SH
  summer easterlies −9 → −28 m/s; tropospheric scores unchanged). The dry-JW
  cold start, which the absolute target had been added for, runs without it:
  60 days at ``ma-t63-l47``, 150 days at ``t63-echam-1m`` and 30 days at
  ``ma-t63-l95`` finished without a NaN, the top-level temperature settling at
  150-210 K (JAM, L47 and L95) and 230-260 K (1M) instead of being held at
  250 K.

The semi-Lagrangian step conserves water
""""""""""""""""""""""""""""""""""""""""

Every semi-Lagrangian run created water in its dynamics. The global budget of
the ECHAM packages (1M, 2M, JAM) had ``P − E + dPW/dt = +0.22`` to
``+0.24`` mm/day, about 9 % of precipitation, with the physics applying
exactly the evaporation and precipitation it reports. The source was the
transport of ``specific_humidity``:

- Its vertical interpolation at the departure points was linear, which
  over-reads a convex profile whatever the direction of the motion.
- Unlike the nodal tracers, humidity had no mass fixer.

The Dinosaur backend now does two things under semi-Lagrangian transport:

- **Cubic vertical interpolation.** It interpolates in the vertical with
  4-point cubic Lagrange, linear in the first and last cells (the IFS rule;
  ``dycore.sl_vertical_interpolation``, default ``null`` = cubic, linear on
  grids with fewer than four levels).
- **A humidity mass fixer.** It restores the global mass of humidity to its
  post-physics, pre-transport value every step with the proportional fixer
  the nodal tracers already had (``dycore.humidity_mass_fixer``, default
  ``true``).

The water source at T63L47 falls from +0.236 to 0 mm/day (+0.001 from the
cubic interpolation alone). The same interpolation transports temperature;
with linear interpolation the dynamics alone lost about 11 W/m² of energy,
which cubic removes, so expect a warmer atmosphere. Expect precipitation that
no longer exceeds evaporation, and a retune of anything calibrated against
the old balance. The Eulerian core (SPEEDY) is unchanged and bit-identical.
See :doc:`design/tracer_mass_conservation`.

ECHAM surface albedo and frozen-surface saturation
""""""""""""""""""""""""""""""""""""""""""""""""""

- The ECHAM surface albedo now follows ECHAM 6.3 tile by tile instead of fixed
  per-type constants (#672). Land is JSBACH's broadband scheme
  (``update_land_surface_fast``): the snow-free background ``alb``, brightened
  by the prescribed ``snowc`` snow cover towards a temperature-dependent snow
  albedo (0.4 at the melting point to 0.8 five kelvin below), snow masked by
  forest, and a glacier albedo (0.75-0.85) on ice sheets — where the constant
  0.15/0.25 used to put Antarctica and Greenland at ~0.2. Sea ice is
  ``update_albedo_ice``, effectively its cold bare-ice 0.75 while the ice
  temperature is prescribed at ``min(SST, ctfreez)``; open water carries ECHAM's
  zenith-angle-dependent direct-beam albedo with the 0.07 diffuse albedo.
  Over a 5-day January ``t63-echam-1m`` A/B the global planetary albedo rises
  **0.274 → 0.300-0.302** and the absorbed solar radiation at TOA falls by
  **9.1-9.6 W/m²** (ice sheets 0.22 → 0.85 surface albedo; snow-covered NH land
  0.22 → 0.50-0.64). See :doc:`science/surface`.
- The surface saturation humidity of every ECHAM tile (and of the
  lowest-level air in the surface-layer Richardson number) is taken over ice
  at and below the melting point and over water above, as ECHAM's ``tlucua`` table
  does, instead of the Sundqvist mixed-phase blend; the latent heat in the
  surface-layer buoyancy switches with the air temperature. The reported
  latent heat flux is ``alhs·E`` over sea ice and carries the sublimation share
  of the snow-covered fraction over land, and snow-covered land evaporates at
  the potential rate (JSBACH's land wetness ``s + (1 − s)·w``).
- New optional static forcing fields ``forest`` and ``glac``
  (``ForcingData.forest_fraction`` / ``glacier_fraction``) carry the land
  cover the land albedo reads; the bundle builders write them from ERA5
  ``cvh`` and the permanent-snow mask, and every published forcing bundle
  carries them from the pinned mirror commit on. A bundle without them loads
  with both ``None`` (no forest masking; ice sheets keep their ERA5
  background albedo of ≈0.8). ``snowc`` is the snow-covered
  fraction of the non-glacier land, so the snow-covered share of the land
  is ``glac + (1 − glac)·snowc`` (``jcm.forcing.land_snow_cover``, also
  read by the JAM dust snow gate). Forcing files now also carry ``lsm``, the
  land share, and every regrid of the land-surface channels (bundle
  builders, runtime upsampler, pySES column sampler) weights each by the
  part of the cell it describes — ``glac`` and ``stl`` by the land,
  ``forest``, ``snowc``, ``alb`` and the soil wetness by the non-glacier
  land — so
  coastal and ice-margin cells are no longer diluted by their ocean or
  glacier neighbours. On pySES this changes the coastal columns of every
  run on the packaged T63 forcing. SPEEDY reads the same fields through the
  same helpers (``jcm.forcing.land_snow_cover`` / ``land_wetness``): with a
  glacier map its land tile counts the glacier as fully snow covered and
  fully wet; without one (the T30 climatology) nothing changes.
- The radiation solves with the surface albedo and emissivity of its solve
  step and publishes those in ``radiation.surface_*``, held between solves,
  so the published albedo, reflected flux and heating stay one solve's
  under ``radiation_interval`` sub-stepping.
- **Breaking:** ``SurfaceOpticsParameters`` holds the albedo constants in a
  nested ``EchamSurfaceAlbedoParameters`` (``albedo=``) and keeps only the
  three emissivities at the top level; the six ``*_albedo_vis``/``*_albedo_nir``
  fields are gone. The packaged ``jcm/data/bc/t63/forcing.nc`` — the pySES
  backend's default forcing — stores ``snowc`` as the jcm cover fraction
  ``min(1, SWE/sd2sc)`` with a seasonal cycle (the data-mirror ERA5 monthly
  climatology, zero on glaciers) where it held one January ECHAM snow water
  equivalent in metres for every month, and gains ``forest``/``glac`` from
  the ECHAM surface file. Everything reading ``snowc`` from that file sees
  the change: the ECHAM albedo and land latent heat, the JAM dust snow gate,
  and SPEEDY's snow albedo if pointed at it.

ECHAM boundary-layer stability is moist and cloud-weighted
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The TTE-TKE vertical diffusion forms the interior Richardson number and the
  buoyancy term of its TKE budget from ECHAM6.3's one ``zbuoy``
  (``vdiff.f90`` l.658-700, 777-799): the liquid-water potential temperature
  and total water of the adjacent levels, averaged to the interface by layer
  mass and weighted by the interface's cloud cover with the ``ua``-table
  saturation humidity, over the squared shear floored at ECHAM's
  ``zepshr = 1e-5``. A saturated cloudy layer mixes on its moist-adiabatic
  stability; the humidity enters even where there is no cloud. The surface
  layer's bulk Richardson number takes the lowest level's cover into the same
  multipliers on every tile (``precalc_land``/``_ocean``/``_ice``). **Changes
  results** for every ECHAM configuration. A 10-day ``t63-echam-1m`` A/B from
  the spun-up state (``dev`` at 40701518, 40 instantaneous snapshots each):
  the interior stability alone moves most of it, the surface layer's cover
  very little (PBL height 200.3 against 200.5 m, lowest-level condensate 53.7
  against 54.4 mg/kg).

  ==============================================  =============  =============
  global mean, days 0-10                          before         after        
  ==============================================  =============  =============
  PBL height (m)                                  186.7          200.3        
  total cloud cover                               0.566          0.527        
  low cloud cover (below 680 hPa)                 0.332          0.290        
  lowest-level cloud cover                        0.176          0.158        
  lowest-level condensate (mg/kg)                 57.8           53.7         
  latent heat flux (W/m²)                         65.2           67.1         
  sensible heat flux (W/m²)                       23.4           23.4         
  convective / stratiform precipitation (mm/d)    1.96 / 0.65    2.01 / 0.63  
  TOA shortwave reflected (W/m²)                  102.9          98.6         
  outgoing longwave (W/m²)                        246.0          246.6        
  unstable interior interfaces                    3.0 %          1.3 %        
  ==============================================  =============  =============

  The larger mixing leaves the model columns less unstable: in the lowest six
  interfaces the unstable fraction falls from 21.6 % to 9.8 %. The ECHAM 1-day
  T21L16 trajectory moves by 13 % (normalized RMS) in specific humidity, 1.5 %
  in ``v``, 1.1 % in ``u`` and 0.03 % in temperature, almost all of it the
  humidity in the Richardson number rather than the cloud weighting. The
  whole-model RCE column evaporates 1.20 mm/d instead of 1.01 and its TOA net
  falls from 47.9 to 38.4 W/m² (days 40-80); its bounds are re-derived in
  :doc:`design/rce_testbed`. See :doc:`science/vertical_diffusion`.
- **API:** ``TteTkeVerticalDiffusion`` requires the ``clouds`` diagnostic,
  ``VDiffState.cloud_fraction`` is required, ``prepare_vertical_diffusion_state``
  and ``vertical_diffusion_scheme`` take ``cloud_fraction`` after ``qi``, and
  ``compute_richardson_number`` takes the state; see :doc:`v2_to_v3`.

Moist dynamics: condensate loading and one tracer contract
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The hybrid dynamical core's virtual temperature now carries condensate
  loading as well as moisture,
  ``Tv = T (1 + (Rv/Rd - 1) q - sum(q_condensate))``, and the geopotential
  handed to physics is built from that same virtual temperature. This matches
  ECHAM6 (``dyn.f90::ztv`` and ``physc.f90::ztvm1``). The condensate set is
  whatever the active composition declares out of ``qc``/``qi``/``qr``/``qs``;
  including prognostic rain and snow is a deliberate departure from ECHAM6,
  which carries no prognostic precipitation. Pure-sigma (SPEEDY)
  configurations keep a dry dynamics — only their physics geopotential
  changes.
- **Breaking for dycore-native saved state:** every mass mixing-ratio tracer
  (cloud condensate, aerosol mass, gas mass) now crosses the Dinosaur boundary
  as the dimensionless kg/kg value rather than being nondimensionalised as
  g/kg, the same contract specific humidity received above. The dynamics reads
  condensate directly for the loading term, so a scaled store would suppress
  it by 1000x. Values in a checkpoint written before this release are 1000x
  smaller than the new convention; multiply them by 1000, or start from a
  gridpoint ``PhysicsState``, which is unaffected. Tracers declaring
  ``nondimensionalize=False`` (number concentrations, VMRs) are unchanged.
  The rescale is behaviourally neutral on its own — transport, filters and the
  modal round trip are all linear in the tracer.

``set_constants`` reaches the JAM and tropopause modules
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ``jcm.constants.set_constants(...)`` now propagates into the JAM aerosol
  activation, sedimentation, dry-deposition, ice-nucleation and
  aqueous-chemistry schemes, the TTE-TKE vertical-diffusion closure, the
  emissions preparation step and the WMO-tropopause diagnostic. All of these
  captured constants at import time — as a value import, as a reference to the
  singleton object, or by evaluating ``c.<name>`` in a module-level constant or
  a default argument — and so silently kept Earth values while the rest of the
  model used the override (#772). A run with a non-default ``grav``, ``cpd``,
  ``m_air``, ``r_universal`` or ``ak`` composing any of those terms therefore
  **changes results**: it was computing with a mixed constant set before.
  Overrides must still be applied before the model is built (a constant read
  inside a jitted term is fixed when that term is traced), and constants
  internal to ``mam4-jax`` remain outside jcm's control.

ARG activation with local-state transport coefficients
""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ARG droplet activation uses CAM ``ndrop.F90``'s temperature- and
  pressure-dependent vapour diffusivity and air conductivity, and CAM's fixed
  Kelvin coefficient, in place of sea-level constants that over-activated by
  ~4 % at 900 hPa rising to ~18 % at 500 hPa. **Changes results** for every
  JAM configuration (fewer activated droplets aloft) (#679).

Convective-type cloud inhomogeneity
"""""""""""""""""""""""""""""""""""

- The radiation's liquid cloud inhomogeneity now follows the previous step's
  convective type, as in ECHAM ``mo_cloud_optics.f90``: 0.8 without
  convection and for deep/shallow/mid-level convection, 0.4 in shallow
  columns whose liquid sits below the convective cloud top (``ktype = 4``,
  which the 1M cloud scheme sets from ECHAM's ``clwprat`` test). The ice
  factor is separate: 0.8, except 0.7 in the JAM composition (ECHAM-HAM's
  2M + ARG value). **Changes results** for 1M configurations (thinner
  trade-cumulus liquid optically, a weaker SW cloud radiative effect there)
  and for JAM (optically thinner ice cloud); the other 2M configurations are
  unchanged, since ECHAM's 2M scheme never re-types.
  **Breaking for direct callers:** ``RadiationParameters.cloud_inhomogeneity``
  is replaced by ``cloud_inhomogeneity_liquid``,
  ``cloud_inhomogeneity_liquid_convective``,
  ``cloud_inhomogeneity_liquid_shallow`` and ``cloud_inhomogeneity_ice``.
  ``convection.cloud_top``/``cloud_base`` now carry the updraft's level
  indices (top-first physics axis) instead of zeros (#870).

Cloud droplets: effective radii from the current state, one 1M droplet number
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- RRTMGP and the NN emulator form the droplet and crystal effective radii
  inside the radiation call from the step's in-cloud condensate and
  droplet/crystal number, as ECHAM's ``mo_cloud_optics.f90::cloud_optics``
  does, clamped to the RRTMGP table range (2.5-21.5 µm droplets, 5-90 µm
  crystals). The radii used to reach radiation one step late through the
  ``clouds`` carry, with 0 meaning "not provided" and an 11 µm / Moss-Foot
  fallback substituted per cell, so a cell that turned cloudy between steps
  radiated with the fallback (#929). The 2M configurations (SPA and JAM) use
  the prognostic ``qnc``/``qni`` with the Peng & Lohmann breadth factor and
  the Lohmann (2008) plate crystal radius; the 1M configurations the Martin
  et al. continental/maritime breadth factor and the Moss/Foot crystal radius.
- The 1M droplet number, for the radiation and for the 1M autoconversion
  alike, is ECHAM's prescribed ``acdnc`` (80 cm⁻³ over sea, 180 cm⁻³ over
  land below 800 hPa, falling to 20 cm⁻³ aloft) times the MACv2-SP Twomey
  factor, from one shared call (``cloud_utils.prescribed_droplet_number``).
  The 1M autoconversion used a uniform 100 cm⁻³ × Twomey factor (#936). The
  Twomey factor stays on the autoconversion, which MPI-ESM1.2 does not do
  (#932).
- **Changes results.** Over days 5-10 of a ``t63-echam-1m`` A/B the two
  changes together raise the global-mean TOA net by **3.4 W/m²**
  (radiation alone: 2.1): reflected SW −4.1 W/m², SW cloud radiative effect
  +4.1, LW cloud radiative effect −0.6, OLR +0.7 W/m², liquid water path
  **−9.0 g/m² (−16 %; −32 % over ocean)**, cloud cover −0.35 %, precipitation
  unchanged. Fewer droplets make larger droplets for the radiation and a
  faster Beheng autoconversion. On ``t63-echam-2m`` the radius change moves
  TOA net by less than the 0.1 W/m² run-to-run spread (OLR +0.18 W/m², LW
  cloud radiative effect −0.19 W/m²). ``clouds.r_eff_liq`` /
  ``clouds.r_eff_ice`` are now written by the radiation term (the radii it
  used; 0 where the phase is absent) rather than by the microphysics. The
  COSP simulators and the AeroCom cloud diagnostics, which read the
  post-microphysics condensate, form the radii of that condensate with the
  same law (``cloud_optics.post_physics_effective_radii``) instead of
  reading ``clouds.r_eff_*``.
- **Breaking:** ``MicrophysicsParameters.base_cdnc`` and
  ``resolve_effective_radii`` are removed, and ``radiation_scheme_rrtmgp`` /
  ``radiation_scheme_emulated`` require ``r_eff_liq_um`` / ``r_eff_ice_um``
  (form them with ``jcm.physics.radiation.cloud_optics.radiation_effective_radii``).

RCE initial state seeds a mixed sub-cloud layer
"""""""""""""""""""""""""""""""""""""""""""""""

- ``jcm.rce.rce_initial_state`` now seeds a dry-adiabatic, well-mixed
  sub-cloud layer below ``mixed_layer_top_m`` (default 800 m). This changes
  results for any RCE case composing ``TiedtkeConvection``: ECHAM's ``cubase``
  trigger finds no cloud base at all in a sounding running at ``lapse_rate``
  to the surface. Pass ``mixed_layer_top_m=0.0`` to restore the previous
  profile; see :doc:`design/convective_trigger_soundings` for the reasoning.

The whole-model RCE testbed runs RRTMGP
"""""""""""""""""""""""""""""""""""""""

- ``rce_test.py::TestRceWholeModelTiedtke`` now integrates ``echam_physics()``
  with RRTMGP (it ran the idealized grey scheme, whose atmosphere cools by
  7 W/m² and whose lowest level fogs under the ECHAM 1M) and pins the
  column's equilibrium from seven trajectories (six perturbed by 1e-4 K of
  initial noise) and up to four 40-day windows: P/E 0.958-1.001, Tiedtke in
  0.98 of the steps, column water steady to 0.05 mm/d, a clear lowest level. No public API changes; see
  :doc:`design/rce_testbed` for the configuration, the bounds and their
  provenance. The column is finite for the 200 days run under jax-rrtmgp
  0.5.0 and is overcast (total cloud cover 1.0).

Grey two-stream shortwave conserves energy
""""""""""""""""""""""""""""""""""""""""""

- The grey scheme's shortwave now delta-scales its optical properties
  (delta-Eddington, Joseph et al. 1976), scatters the direct beam into the
  diffuse streams with the exact two-stream source solution (Meador & Weaver
  1980; Toon et al. 1989) and joins the layers with the adding method, as the
  RRTMGP solver does. Before, the scattered part of the direct beam was dropped
  wherever a forward-scattering cloud met a high sun, and the missing flux was
  booked as absorption: a thick non-absorbing cloud reflected nothing and heated
  instead. A ``tau = 82`` liquid cloud under an overhead sun now reflects 88 %
  of the incident flux (4e-36 before), and with no absorption the column closes
  its energy budget to float32 round-off (#855). **Changes results** for every
  configuration on the grey radiation (the ``echam_physics()`` default): cloudy
  columns reflect far more and heat far less in the shortwave. The
  ``echam_t21l16_1day`` composable-physics reference is regenerated.
  ``layer_reflectance_transmittance``'s ``T_dir`` is now the diffusely
  transmitted fraction of the beam; the unscattered ``exp(-tau/mu0)`` is no
  longer included.
- New diagnostic ``convection.precip_floor_source`` [kg m-2 s-1]: the water
  the Tiedtke scheme creates where ``cuflx`` floors a negative convective rain
  flux, i.e. where the downdraft takes up more rain than the re-run updraft
  generates. ECHAM behaves the same way; the floor is kept and its source
  tracked (#912). A column water budget closes as
  ``E - P + precip_floor_source``. The grey RCE column reaches this regime
  once its clouds reflect, at ~0.06-0.09 mm/d.

Convective scavenging follows ECHAM-HAM
"""""""""""""""""""""""""""""""""""""""

- The convective tracer transport's in-plume scavenging takes ECHAM-HAM's
  parameters, inputs and processes. Each aerosol mode carries HAMMOZ's
  convective in-droplet fraction ``csr_conv`` of the M7 class it corresponds
  to (accumulation 0.99, coarse 0.99, Aitken 0.60, primary carbon 0.20;
  dust and sea salt, carried in MAM4's soluble modes, take 0.99). That share
  joins the condensate once where the aerosol meets cloud; each cloudy level
  removes from it HAMMOZ's precipitation efficiency ``peff =
  pmrateprecip/pmwc``; and the removed aerosol is released where the
  convective precipitation evaporates (``prevap``), as is the aerosol the
  convective carrier washes out below cloud. HAMMOZ's own bookkeeping
  (removal from the unscavenged updraft concentration, the total-flux
  overwrite and ``xt_conv_massfix``) is not ported: it drops the
  compensating subsidence and drives tracers negative, where the closed
  plume budget used here conserves exactly and stays positive; see
  :doc:`science/aerosol`. ``TiedtkeConvection`` publishes the two new
  interface fields ``convection.precip_efficiency`` and
  ``convection.precip_evap_fraction``. **Changes results** for every JAM
  configuration with convective transport: soluble aerosol reaches the
  convective outflow at about 1 % of its boundary-layer concentration
  instead of ~10⁻⁵ (``scm_check.py`` failed "soluble also lofted but less",
  #923), fresh primary carbon is now scavenged in convective cloud (it was
  not), and Aitken-mode aerosol is scavenged less (#928). **Breaking for
  direct callers:** ``ConvTransportParameters.scav_ratio`` is replaced by
  the per-tracer ``csr_conv``, ``ConvectiveTracerTransport``'s
  ``scav_weights`` by ``csr_conv``, and ``convective_tracer_tendency``
  takes ``csr_conv``, ``precip_efficiency``, ``plume_condensate`` and
  ``evap_fraction``.

Lohmann 2M utility fields are ECHAM's
"""""""""""""""""""""""""""""""""""""

- Four fields of the two-moment cloud scheme now follow ECHAM6.3-HAM2.3
  (#942). The snow Reynolds number of riming divides by the viscosity of air
  ``pviscos`` (``mo_cloud_utils.f90``, line 132) instead of the thermal
  conductivity of air, which was ~1400 times larger and held the collection
  efficiency of droplets by snow at its 0.01 floor; it is now ~0.8. The ice
  fall-speed factor is ``paaa = (p/30000)^-0.178·(T/233)^-0.394`` (line 129)
  instead of ``(1.3/ρ)^0.4``, so cloud ice falls 30-35 % slower aloft. The
  turbulent updraft of the phase and Wegener-Bergeron-Findeisen criteria is
  ``100·fact_tke·√TKE``, zero at the lowest level
  (``mo_cloud_micro_2m.f90``, lines 814-815), instead of ``√(2·TKE)``. The
  threshold's ice radius is ``0.9·r_eff``
  (``effective_2_volmean_radius_param_Schuman_2011``) instead of the plate
  radius ECHAM uses only for aggregation, which was up to three times
  smaller. **Changes results** for every 2M configuration, including JAM.
  Over days 5-10 of ``t63-echam-2m`` runs restarted from a 30-day spin-up of
  the preset, global liquid water path falls from 60.6 to 41.0 g/m² (the
  supercooled part from 42.7 to 23.8), ice water path rises from 2.95 to
  3.4 g/m², large-scale snowfall rises 2.6-fold, total cloud cover falls by
  3.4 points, the shortwave cloud effect weakens by 7.5 W/m² and the
  longwave one by 3.6 W/m², and net TOA radiation rises by 3.7 W/m². The
  riming viscosity accounts for most of the liquid and snowfall change. Ten
  days measure the immediate response, not a new climate; the release-matrix
  bands of the ``echam-2m`` and ``echam-jam`` members shift accordingly. See
  :doc:`science/clouds_microphysics`.

Lohmann 2M detrained ice carries ECHAM's crystal number
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

- Convectively detrained condensate in the two-moment cloud scheme follows
  ECHAM6.3-HAM2.3's section-1 and section-4 rules (#941). Detrained ice
  carries the crystal number ``znidetr`` at the temperature-parameterised
  radius ``zrid`` (``mo_cloud_micro_2m.f90``, lines 945-983) and joins the
  cell after that step's ice sedimentation (lines 1227-1252), instead of
  arriving without crystals and sedimenting at the old number in the same
  step. The whole detrained condensate is split into ice and liquid by the
  Wegener-Bergeron-Findeisen criterion ``lo2``, with the fusion-heat
  correction (lines 1276-1317), instead of the Tiedtke split at the melting
  point. The ICNC diagnosis inverts the ice mass at ``zrid`` instead of the
  radius of the existing ice (line 1511). The number-tracer tendencies are
  taken against the unclipped tracers (lines 1780-1781, 3625-3628), so an
  out-of-range crystal or droplet number no longer persists from step to
  step.
- The DeMott (2010) INP number is converted from standard to ambient air
  density, and mixed-phase freezing creates no more crystals than there are
  droplets. (Under JAM the mixed-phase freezing is now ECHAM-HAM's; see the
  next entry.)
- **New output:** ``clouds.conv_detrainment_qc`` and
  ``clouds.conv_detrainment_qi`` [kg kg⁻¹ s⁻¹], the condensate the Tiedtke
  scheme detrains each step, in its own phase split.

- ``demott2010_inp`` takes the ambient air density as a third positional
  argument.
- **Changes results** for every 2M configuration, including JAM. Over days
  5-10 of ``t63-echam-2m`` runs restarted from a 30-day spin-up, global ice
  water path rises from 3.4 to 27.1 g/m² (observed about 27), the longwave
  cloud effect from 14.1 to 24.4 W/m², the shortwave one strengthens from
  -38.8 to -44.7 W/m², liquid water path falls from 41.2 to 30.5 g/m², and
  net TOA radiation rises by 4.0 W/m². Under JAM (``ma-t63-l47``, ten days
  from a cold-started state) ice water path rises from 1.2 to 6.8 g/m², the
  longwave cloud effect from 14.0 to 17.2 W/m², and the mixed-phase
  condensate stays mostly liquid. Glaciation remains too warm, cold ice
  cloud holds too many crystals, and liquid water path lies below the
  observed range; the retune left these defaults unchanged
  (:doc:`design/jam_aerosol_retune`). The release-matrix bands of the
  ``echam-2m`` and ``echam-jam`` members shift accordingly. See
  :doc:`science/clouds_microphysics`.

ECHAM physics saturation is ECHAM's Sonntag (1990)
""""""""""""""""""""""""""""""""""""""""""""""""""

- Every ECHAM scheme takes its saturation vapour pressure from
  ``jcm.physics.thermodynamics``, which evaluates the Sonntag (1990) fit
  ECHAM's lookup tables hold, with the table ECHAM reads at each site
  (#956): ice at and below the melting point and water above (``ua``) for
  Tiedtke-Nordeng convection, the TTE-TKE vertical diffusion, the surface
  tiles and the 2M ice saturations; water at all temperatures (``uaw``) for
  the 1M rain evaporation and the 2M water saturations; ECHAM's ``lo2``
  choice between the two for the cloud cover and the 2M condensation. The
  1M's condensation takes the same ``lo2`` choice (see the 1M entry
  below). The Tetens forms it replaces were up to 0.15 % off between 273
  and 330 K, 1.2-2.4 % between 238 and 273 K and 8-16 % between 200 and
  238 K. ``qs`` is ECHAM's ``x/(1 − vtmpc1·x)`` with
  ``x = MIN(es·rd/rv/p, 0.5)``, so the ratio is ``rd/rv`` (0.62196),
  consistent with ``vtmpc1``, rather than ``c.eps``.
- Tiedtke-Nordeng's saturation adjustment is ECHAM's ``cuadjtq`` (#957): one
  Newton step clipped by ``kcall``, then one unclipped refinement where the
  first step moved. It matches ECHAM's compiled routine to rounding.
- **Changes results** for every ECHAM configuration. Evaluated term by term
  on one saved T63 state, the diagnosed cloud fraction of levels colder than
  238 K, where the ice fit now sits 1.3-9 % above the old one, falls from
  2.2 % to 0.8 % (1M preset) and from 3.1 % to 2.4 % (2M preset); the cloud
  microphysics tendencies move by 10 % (1M) and 18 % (2M) RMS, 32-48 % below
  238 K, the convection's by 15 % and the vertical diffusion's moisture
  tendency by 0.6 %. Over days 5-10 of ``t63-echam-1m`` / ``t63-echam-2m``
  runs restarted from 30-day spin-ups of each preset, the global net TOA
  radiation goes from −10.26 to −9.71 / 8.62 to 8.79 W/m², the shortwave
  cloud effect from −78.4 to −76.5 / −38.5 to −38.1 W/m², the longwave one
  from 34.6 to 33.1 / 13.7 to 13.3 W/m², liquid water path from 129.9 to
  127.1 / 41.0 to 40.5 g/m², ice water path from 16.6 to 16.1 / 3.51 to
  3.57 g/m², total cloud cover from 71.6 to 71.0 / 66.0 to 65.9 %, and
  precipitation from 2.52 to 2.53 mm/day (1M; the 2M's stays at 2.66);
  humidity at
  200 hPa rises by 2 / 3 % (by 2.4 / 3.3 % in the tropics), and no
  band-mean upper-tropospheric temperature (90-60-30° bands) moves by more
  than 0.07 K. Rebuilds of this change that differ only at the 1e-4 level
  spread by 0.3 / 0.06 W/m² in net TOA radiation over the same window,
  which is the noise of these numbers. Ten days measure the immediate
  response, not a new climate; the release-matrix bands of every ECHAM
  member shift (#943). These numbers compare dev at ``f3780690``, before the
  cloud-droplet (#929, #936) and detrained-ice (#941) entries above, with this
  change at ``b56ceebf`` (term by term, and the 1M runs) and at ``5ad09de1``
  (the 2M runs).
- Unchanged, bit for bit: SPEEDY, Held-Suarez, Betts-Miller and the RCE
  testbed, JAM's ARG activation, MAM4 humidity and ice nucleation, the public
  relative-humidity diagnostic, the AeroCom diagnostics and the initial-state
  injectors.
- **Breaking:** ``jcm.physics.convection.tiedtke_nordeng.adjustment`` is
  removed; ``cuadjtq``, ``cuadjtq_newton`` and ``cuadjtq_newton_evap`` live in
  ``tiedtke_nordeng.cuadjtq``, the last two without ``n_refine``.
  ``convection.saturation`` no longer carries the ``cuadjtq`` helpers or
  ``saturation_specific_humidity_and_derivative``,
  ``tiedtke_nordeng.tiedtke_nordeng`` no longer re-exports a Tetens
  ``saturation_vapor_pressure``, ``thermodynamics`` drops its Tetens
  constants, and ``clouds.sundqvist`` its two ``saturation_vapor_pressure_*``
  functions; ``surface.echam.AtmosphericForcing`` takes a required
  ``surface_pressure``.
  See :doc:`v2_to_v3` and :doc:`science/constants`.

The ECHAM 1M cloud scheme is ECHAM6.3's ``cloud``
"""""""""""""""""""""""""""""""""""""""""""""""""

- ``Echam1MMicrophysics`` runs ECHAM6.3's ``mo_cloud.f90::cloud`` (r7492)
  section by section, in ECHAM's order, at every level of one top-down
  column sweep (#940). Its condensation is driven by the step's increments,
  ``zqcdif = (Δq − Δq_sat)·paclc`` with the cloudy part saturated at the
  anchor (``mo_cloud.f90`` l.696-750), so partial cloud evaporates only where
  the increments dry it, not by a relaxation of the grid mean toward
  saturation. ``lo2`` is ECHAM's binary phase switch, evaluated at
  ``ptm1 + ptte·dt`` with the sedimented ice and again in section 5.4
  (l.647-650, 763-764); it selects the latent heat, the ice or water
  saturation and the phase of new condensate. The sweep carries snow
  sublimation (l.442-507), homogeneous freezing of cloud water at and below
  ``cthomi`` (l.821-828), Bigg and contact freezing (l.832-885, #939), a
  clear cell that gains condensate acting as cloudy for the microphysics
  (l.800-810), the whole-box supersaturation check (l.754-784), the
  precipitating-fraction reset to the local cover (l.1129, 1177) and the
  return of condensate below ``ccwmin`` to vapour with the cover write-back
  (l.1264-1288). Melt, snow sublimation and rain evaporation are evaluated at
  the anchor, before condensation as in ECHAM, with ECHAM's caps, and
  sedimenting ice keeps ECHAM's ``EPSILON`` floor (l.583). The section-5
  bounds and phase split are shared with the 2M
  (``cloud_utils.sundqvist_condensation``) in the 2M's operation order. See
  :doc:`science/clouds_microphysics`.
- **Compared with the compiled Fortran.**
  ``jcm/physics/clouds/echam_fortran_reference_test.py`` runs jcm against
  numbers from ECHAM6.3's own routine, with its code unmodified and its
  saturation lookup tables replaced by the analytic formula they tabulate (the
  two agree to 3e-11 of each column's scale), compiled in a local harness (the
  ECHAM source is not in this repository), under ECHAM's constants. Every
  output passes on 42 designed and sampled columns, at T63 in float64 (41
  columns in float32; one tests exact melting point thresholds) under ECHAM's
  Sonntag saturation and under jcm's Tetens as a variant kept for localising
  disagreements, and at T31, T127 and T255 in float64. The 73 locals the
  harness records are compared under Sonntag at T63, in float64 and float32.
  Everything passes except three float32 locals recorded in
  ``jcm/data/test/echam_cloud_reference/known_gaps.json`` as strict expected
  failures: two Bigg/contact columns whose 4e-10 kg m⁻² s⁻¹ trace snow flux
  carries 0.4 % float32 error, and one 1.7e-16 melt remainder that flips a
  ``zclcpre`` test. Those columns pass in float64, and their float32 outputs
  pass.
- Where ECHAM's value is not smooth inside the range the model visits, it is
  kept exactly and the derivative is that of a named smooth surrogate (see
  *Reference-exact values with surrogate derivatives* under New capabilities):
  ``lo2``, the melt of cloud ice above ``tmelt``, the
  freezing of cloud water at ``cthomi``, the ice fall speed (a C1 parabola
  below 1e-7 kg m⁻³, superseding #887), the contact-freezing radius and the
  KK2000 gate. The widths are static fields of ``MicrophysicsParameters``; a
  width of 0 selects the reference derivative. The clear-cell test, the
  ``zclcpre`` reset, the ``ccwmin`` correction and the ``ub`` branch keep the
  reference derivative.
- ``MicrophysicsParameters`` is a flax dataclass. Its numeric fields, now
  including ``csecfrl`` and ``cthomi``, are differentiable leaves;
  ``cvtfall``, ``csecfrl`` and ``clwprat`` default to their truncation's
  values (entry below); ``autoconversion_scheme``, ``autoconversion_twomey``
  (on, as the droplet entry above describes) and the surrogate widths are
  static. **Breaking:** ``t_mix_min``, ``t_mix_max`` and ``d_epsilon`` are
  removed, ``smooth_ccraut`` and ``autoconversion_scheme`` are static, and
  ``cloud_microphysics_column_sweep`` takes the anchor and the increments as
  separate arguments. See :doc:`v2_to_v3`.
- **Changes results** for every 1M configuration. Over days 5-10 of
  ``t63-echam-1m`` runs restarted from a 30-day spin-up, the changes of this
  entry and of the cover, inputs and overlap entries below together (jcm
  66b0d11d against dev with #965 (7e14ec2f)) move the global net TOA
  radiation from −5.63 to +0.68 W/m²: the shortwave cloud effect weakens from
  −71.12 to −47.22 W/m² and the longwave one from 31.98 to 14.80 W/m²
  (OLR +17.56 W/m²). Liquid water path falls from 111.0 to 69.80 g/m²
  (−48.5 g/m² over ocean) and the liquid held below 273.15 K from 75.71 to
  23.81 g/m²; ice water path rises from 16.12 to 18.70 g/m². The supercooled
  liquid fraction of the condensate falls from 0.817 to 0.400 at 253-258 K
  and from 0.723 to 0.100 at 243-248 K. Total cloud cover
  (``radiation.total_cloud_cover``) falls from 70.53 to 55.25 %,
  precipitation rises from 2.545 to 2.645 mm/day (convective +0.132,
  large-scale −0.032 mm/day), column water vapour falls by 0.81 kg/m², and at
  300 hPa the humidity falls by 16 % and the temperature by 0.89 K. The
  lowest model level does not fog: its mean cover falls from 0.195 to 0.169.
  Runs of identical physics from the same state spread by at most
  0.19 W/m² in net TOA radiation, 0.30 W/m² in either cloud effect,
  0.9 g/m² in liquid water path, 0.26 % in cover and 0.006 mm/day in
  precipitation, which is the noise of these numbers. The branch members ran
  on one devbox host (NVIDIA A100 80GB PCIe, jax 0.10.2) because no Nautilus
  A100 could be scheduled, from the same spin-up checkpoints, overrides and
  package pins as the controls, which ran on Nautilus SXM4 nodes under jax
  0.10.1, so the spread, measured within one node type and jax version, does
  not strictly cover that boundary; the same configuration at the previous
  head (82c2f294, run on Nautilus) differs from these members by less than
  the spread for the 1M (the area with lowest-level cover above 0.9 apart)
  and by a few per cent for the 2M and JAM (2M in-cloud crystal number
  −1.9 %, large-scale precipitation +1.2-1.7 %), which bounds it. Ten days measure the immediate response, not a new
  climate; the release-matrix bands of every ECHAM member are regenerated
  after this change (#943).

The ECHAM cloud cover is ECHAM6.3's ``cover``; radiation masks it by condensate
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ``SundqvistCloudFraction`` computes ECHAM6.3's ``mo_cover.f90::cover``
  (#940). The cover is ECHAM's exact ``1 − sqrt(1 − b0)`` with ``b0`` clipped
  to ``[0, 1]`` (``mo_cover.f90`` l.249-251): exactly 0 at or below the
  critical humidity and exactly 1 at saturation. The stratocumulus
  enhancement uses ECHAM's inversion search, from the lowest level up to
  ``jbmin``, enhancing only at or above ``jbmax`` (l.164-207, 234-247), with
  ``jbmin``/``jbmax`` derived from the model's own levels by ``sucloud``'s
  rule (``mo_echam_cloud_params.f90`` l.132-162; 40/45 on L47, 88/93 on L95).
  There is no stratospheric cutoff: ECHAM computes the cover at every level
  (``ktdia = 1``, ``physc.f90`` l.444). The derivatives of the clip and of the
  inversion stability test are those of named surrogates (``smooth_b0``,
  ``smooth_inv_thr``, static; 0 selects the reference derivative); the
  choice of inversion level keeps its zero derivative. The cover reads the
  state the physics receives, where ECHAM's reads the previous time level
  (``physc.f90`` l.543-548); :doc:`science/clouds_microphysics` records the
  difference.
- **Radiation sees the cover only where there is condensate**, as ECHAM's
  does (``mo_radiation.f90`` l.428-434): ``cloud_data.radiation_cloud_fields``,
  which RRTMGP, the grey two-stream and the emulator read, returns the
  step-start ``qc``/``qi`` clipped at zero and the cover zeroed where neither
  phase is positive. ``clouds.cloud_fraction`` itself is unchanged; COSP masks
  its cover with the condensate it is given.
- Compared with the compiled Fortran on 25 cover columns: all pass at T63 in
  float64 and float32 under Sonntag and Tetens saturation, and at T31, T127
  and T255 in float64.
- **Breaking:** ``CloudParameters`` is a flax dataclass built for a
  truncation (``CloudParameters.default(truncation=...)``,
  ``CloudParameters.for_grid(coords)``); ``inversion_z_max``,
  ``inversion_z_min``, ``cloud_top_pressure_pa``, ``epsilon``, ``t_mix_min``,
  ``t_mix_max``, ``smooth_inv_score`` and ``smooth_inv_depth`` are removed,
  ``csecfrl`` and ``nadd`` are added, and
  ``sundqvist.condensation_evaporation``, which had no production caller, is
  gone. See :doc:`v2_to_v3`.
- **Changes results** for every ECHAM configuration (measured with the 1M
  entry above and the inputs entry below).

The cloud schemes take ECHAM's anchor, increments and detrainment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The 1M and the Lohmann 2M receive the inputs of ECHAM's tendency-driven
  cloud routines (``physc.f90`` l.1073-1081; ``mo_cloud.f90`` l.706-730) from
  one helper, ``jcm.physics.clouds.cloud_inputs.cloud_scheme_inputs``: the
  anchor ``ptm1``/``pqm1``/``pxlm1``/``pxim1`` is the previous step's
  post-physics state; the increment is the dynamics of the last step plus
  ``dt`` times the running tendency of every term composed before the cloud
  scheme (radiation, vertical diffusion with its condensate change, the
  surface, convection); and the convective detrainment arrives by itself, as
  ECHAM's ``pxtecl``/``pxteci``, from ``clouds.conv_detrainment_qc`` /
  ``conv_detrainment_qi``. Large-scale ascent and cloud-top radiative cooling
  therefore condense in a partly cloudy cell in the step they happen. See
  :doc:`design/operator_split_physics`.
- ``DynamicalCore.after_physics_state(state, physics_tendency)`` returns the
  gridpoint state after the physics tendency and before the dynamics.
  ``Model`` records it each step in the physics carry slot
  ``_post_physics_state``, with a ``valid`` flag, when a composed term
  declares ``requires_post_physics_fields``, as the 1M and 2M terms do.
  Dinosaur applies the tendency through the same spectral projection its
  ``step`` uses; pySES adds the lumped forcing of its coupling step, which
  under ``lump_all`` and ``hybrid`` coupling (the shipped ne30 presets) forms
  every field of the anchor, so the GLL projection residual is not counted as
  dynamics in the temperature increment; the base class's default is the
  gridpoint forward-Euler add. Where no valid anchor exists
  (the first step, a checkpoint written before the slot, the single-column
  and RCE hosts) the anchor is the state the physics receives and the
  dynamics increment is zero.
- The 2M's air density is ECHAM's ``papm1/(rd·ptvm1)`` at the anchor, the
  virtual density (``mo_cloud_micro_2m.f90`` l.578), as the 1M's is, instead
  of the dry density of the state the term received; the layer depth that
  goes with it keeps the layer mass. The per-m³ quantities of the 2M and JAM
  presets (water contents, number concentrations, autoconversion,
  sedimentation) change by about 1 % at 15 g/kg humidity.
- Checkpoints written before the slot existed restore with it seeded from
  the fresh carry and ``valid = 0`` (logged at INFO), and the carried anchor
  applies from the next step, with no schema change
  (:doc:`design/checkpoint_compatibility`).
- The term order is unchanged. Gravity-wave and orographic drag run after
  the cloud scheme, so their heating (at most 0.03 K/day in the troposphere
  at T63) reaches it through the anchor one step later; ECHAM runs them
  before ``cloud`` (``physc.f90`` l.835-884). The upper sponge, which has no
  counterpart in ``physc``, also runs after the scheme and is part of the
  anchor.
- **Changes results** for every 2M configuration, including JAM, together
  with the cover entry above and the overlap entry below. Over days 5-10 of
  ``t63-echam-2m`` / ``t63-echam-jam`` runs restarted from 30-day spin-ups
  (jcm 66b0d11d against dev with #965 (7e14ec2f), measured as for the 1M
  above), the global net TOA radiation falls by 0.83 / 0.68 W/m² (shortwave
  cloud effect −0.77 / −0.53, longwave +0.07 / −0.04 W/m²), total cloud cover
  (``radiation.total_cloud_cover``) falls from 66.65 to 60.24 % / 68.42 to
  61.93 %, liquid water path rises from 33.60 to 40.70 g/m² / 44.17 to
  50.04 g/m², ice water path changes by −0.18 / +0.12 g/m² (of 26.9 / 8.8),
  in-cloud droplet number falls by 1.4 / 1.9 cm⁻³, the in-cloud crystal
  number changes by −19 / +28 L⁻¹ (of 1079 / 590), and precipitation moves
  by at most 0.007 mm/day. The supercooled liquid fraction of the 2M rises by
  up to 0.08 (at 253-268 K). The lowest model level does not fog: its mean
  cover falls from 0.162 to 0.145 / 0.159 to 0.143. The 2M's longwave cloud
  effect, OLR and large-scale precipitation and JAM's longwave cloud effect
  and total precipitation are within the spread of identical-physics runs
  (0.13 and 0.16 W/m², 0.005 and 0.006 mm/day); the other changes are
  outside it. The JAM numbers were measured before the JAM mixed-phase
  freezing entry below, which also changes JAM; the combination was not
  re-measured.

Default cloud overlap is maximum-random, sampled by ECHAM's rule
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- ``RadiationParameters.cloud_overlap`` defaults to maximum-random (code 1,
  ``CLOUD_OVERLAP_MAXIMUM_RANDOM``), ECHAM6.3's default (``i_overlap = 1``,
  ``mo_radiation_parameters.f90`` l.71), instead of exponential overlap with
  a 2 km decorrelation length, which ECHAM's sampler does not offer. The
  McICA sampler draws it with the rank rule of ECHAM's
  ``mo_cld_sampling.f90::sample_cld_state`` (l.66-83): a sub-column keeps its
  rank only where it is cloudy in the adjacent level, so adjacent cloudy
  layers overlap maximally and layers separated by clear air overlap
  randomly, including across a layer with cover but no condensate (entry
  above). ECHAM runs the chain top-down on its surface-first column and jcm
  bottom-up on its top-first one; the two give every sub-column cloud
  pattern the same probability. Its expected
  total cover is ECHAM's ``cld_cvr`` (``mo_radiation.f90`` l.436-442), which
  is what the emulator's ``radiation.total_cloud_cover`` reports; RRTMGP
  reports the cover of its drawn sub-columns.
- **Changes results** for every RRTMGP configuration. The grey two-stream
  weights its cloudy beam by the column's largest cover under both
  maximum-random and exponential overlap, so its fluxes do not change with
  this default; the emulator's fluxes carry the overlap of their training
  labels, so only its total-cover diagnostic changes. Exponential overlap
  remains available as ``cloud_overlap=2`` (``CLOUD_OVERLAP_EXPONENTIAL``,
  with ``cloud_decorrelation_km``).
- Measured at fixed branch physics over the same days 5-10 (the branch at
  82c2f294 rerun with ``cloud_overlap=2``; 66b0d11d keeps the same default),
  the switch lowers total cloud cover by
  1.02 / 0.55 / 0.94 points and raises the net TOA radiation by
  0.55 / 0.22 / 0.35 W/m² in ``t63-echam-1m`` / ``t63-echam-2m`` /
  ``t63-echam-jam``, through a weaker shortwave cloud effect (+0.84 / +0.45 /
  +0.63 W/m²); the 2M's TOA change is at the 0.19 W/m² spread of
  identical-physics runs. Liquid water path, the lowest-level cover and
  precipitation stay within that spread, except JAM's large-scale
  precipitation (−0.010 mm/d, 1.8 times the largest spread), and part of the
  cover change is by construction, since ``radiation.total_cloud_cover`` is the sampled cover
  under the rule in use. The switch is a small part of the changes the 1M and
  inputs entries above measure: it is 7 / 9 / 14 % of their fall in total
  cover, so at least 85 % of that fall comes from the rest.
- The packaged NN radiation emulator was trained on RRTMGP fluxes under
  exponential overlap at 2 km. Under the maximum-random default its
  ``radiation.total_cloud_cover`` follows the configured rule while its
  fluxes keep the training overlap, until it is retrained (#881).

ECHAM cloud parameters default to their truncation's values
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The cloud cover's ``crs``, ``crt``, ``nex``, ``csatsc``, ``cinv``,
  ``csecfrl`` and ``nadd``, the 1M's ``cvtfall``, ``csecfrl`` and
  ``clwprat`` and the 2M's ``cvtfall`` take ECHAM6.3's per-truncation values
  (``mo_echam_cloud_params.f90::sucloud`` l.198-237; at T63 the cover's ``crs``,
  ``crt``, ``nex``, ``csatsc`` and ``cinv`` are jcm's calibrated values, next
  entry) for the run's grid,
  chosen when the physics is built: by ``echam_physics(coords=...)``, and by
  both Hydra doors, which give the physics the grid (the factory presets
  receive ``coords``; the term-list presets build each term's parameters with
  ``jcm.physics.resolution_defaults.default_parameters``; the pySES door
  builds the physics the model runs with its dycore's grid). Without a grid the
  defaults are the T63 values. They remain differentiable parameters: an
  explicit ``Parameters`` object is used as given, and a field override
  replaces its field on top of the grid's defaults.
- Between ECHAM's truncations (T31, T63, T127, T255) the values are
  interpolated linearly in the truncation number, and integer fields take the
  nearer truncation's value. T106, which ECHAM does not support, gets the
  untuned blend of the T63 row (with the calibrated cover values) and ECHAM's
  T127 row: ``crs = 0.963156``, ``crt = 0.726708``, ``csatsc = 0.781446``,
  ``cinv = 0.237861``, ``cvtfall = 2.8359375``, ``csecfrl = 8.359375e-6``,
  ECHAM's ``nex = 2`` and T63's ``nadd`` and ``clwprat``. Outside T31-T255 the
  end row is held,
  and a grid without a spectral truncation (pySES) takes the T63 row; both
  warn once. A term whose factory-built defaults were made for another grid
  than the one it runs on warns once, naming both.
- **Changes results** at every truncation but T63 (T63 changes through the
  calibration in the next entry), including the ``t106-echam-1m`` and
  ``t106-echam-2m`` members. See :doc:`design/resolution_defaults`.

The cloud-cover defaults at T63 are calibrated
""""""""""""""""""""""""""""""""""""""""""""""

- The Sundqvist cover's ``crt``, ``crs``, ``nex``, ``csatsc`` and ``cinv`` are
  **0.679016061, 0.9, 1.84856084, 0.948216414 and 0.213005383** at T63
  (ECHAM6.3: 0.75, 0.975, 2, 0.7 and 0.25), for the 1M, 2M and JAM-2M hosts
  alike. They are the interior arm of a 25-arm Gaussian-process search over
  exactly these five parameters on the 2M host (30-day windows against the
  jcm-monitor climatologies), confirmed in a 365-day T63 L47 year. Against the
  same year with ECHAM's values, the radiation cover rises from 0.58 to 0.65
  (observed 0.63), the cover the ``cloud_cover`` release gate measures from 0.46
  (a failure) to 0.51 (a pass; the floor is 0.5), the net TOA flux falls from
  +9.8 to +5.6 W/m² (CERES +1.0), the reflected SW rises from 90.8 to
  96.9 W/m² (99.0) and the LW CRE from 26.1 to 28.2 (27.9). The cost is a SW
  CRE 2.6 W/m² too strong (-48.3 against -45.7), 7.5 g/m² more liquid water
  path (49.6, ESA-CCI 36.4), cloud at the lowest model level that rises from a
  mean cover of 0.10 to 0.17 (about half of the added cloud lies in the lowest
  five levels), a near-surface temperature 0.2 K lower and an
  all-sky OLR bias of -8.7 against -6.7 W/m² (the clear-sky OLR bias, -7.5 to
  -9.5 W/m², does not move). Precipitation and the tropical precipitation
  extremes are unchanged. The sweep's unconstrained optimum sat on the edges of
  its box (``nex`` 3.96 of 4, ``cinv`` 0.5 of 0.5, ``crs`` 0.9) and was not
  adopted; the adopted ``crs`` is itself on its lower bound of 0.9, below every
  ECHAM value. The non-integer ``nex`` is admissible: the critical-humidity
  profile is continuous in it. The 1M host reads the set without a calibration
  of its own; its 365-day year gains cover (offline 0.42 to 0.48, radiation 0.52
  to 0.60), takes the SW CRE from -35.5 to -45.8 W/m² (observed -45.7) and the
  annual TOA flux from +5.8 to -2.7 W/m² (CERES +1.0), and pays with liquid water
  path (61 to 82 g/m², ESA-CCI 36). T106 and the cubed sphere read the set
  without a calibration or year of their own (#1014).
- **On the JAM-2M host the cover set and the aerosol scales are one
  calibration.** On the 2M host the set passes every gate; on the JAM-2M host,
  whose droplet number comes from the interactive aerosol, the same set with the
  aerosol scales fitted on ECHAM's cover parameters (dust 0.379095663, sea salt 2)
  over-brightens and strips aerosol. That year passes the cover gate
  (0.52 offline, 0.65 radiation) and its annual TOA flux is +3.2 W/m² (+7.1
  without the set), but its SW CRE is 3.3 W/m² too strong (-49.0), its liquid water
  path is 57 g/m² (48 without it; ESA-CCI 36), and the larger cloud fraction
  increases wet removal: the sea-salt burden falls from 13.3 to 9.2 mg/m²
  (lifetime 0.65 to 0.40 d), the total AOD from 0.082 to 0.052 (the control's
  0.059) and the dust burden from 15.7 to 14.6 mg/m², while sulphate stays in the
  AeroCom band (4.15 mg/m², ion basis). The loss of the eight-target recipe is
  461.1, worse than the aerosol-only configuration's 377.6 (January better, July
  much worse), and the black-carbon (+0.0032 /day) and sulphate (+0.0026 /day)
  drifts of the year fail the recipe's 0.002 /day limit, black carbon also the
  0.003 /day limit for whole-year records. The shipped aerosol scales are the
  ones fitted with this set in place (dust 0.344, sea salt 4, in the entry on the
  aerosol defaults above); the cover parameters themselves were not re-fitted on
  the JAM host.
- **Changes results** on every ECHAM host at T63 (cloud cover, cloud radiative
  effect, TOA balance, surface temperature), and at T106 and on the cubed
  sphere, which take the T63 row or interpolate from it. ECHAM's constants are
  one override away: ``+physics.clouds.crt=0.75`` (and ``crs``, ``nex``,
  ``csatsc``, ``cinv``) on the factory-built presets, or
  ``++physics.terms.sundqvist_cloud_fraction.params.crt=0.75`` on the
  term-list presets. The release-matrix bands of the ECHAM members shift
  accordingly and are regenerated with the release candidate.

JAM mixed-phase freezing follows ECHAM-HAM
""""""""""""""""""""""""""""""""""""""""""

- Under JAM, supercooled cloud water freezes at ECHAM-HAM's rates:
  Brownian contact freezing on insoluble dust and Lohmann & Diehl (2006)
  immersion freezing of droplets holding dust or black carbon
  (``mo_cloud_micro_2m.f90::het_mxphase_freezing``, lines 2675-2840),
  instead of jcm's closure that raised the crystal number to an INP number.
  The immersion rate follows the cooling of the turbulent updraft; the
  large-scale vertical velocity is not plumbed yet (#705) and in 10-day T63
  runs would change the rate by 0.5 %.
- JAM's ``IceNucleation`` term computes the dust and black-carbon fractions
  of the activated droplets and of the insoluble aerosol that ECHAM-HAM's
  ``ham_IN_setup`` passes to that routine, from the MAM4 modes, and
  publishes them as ``freezing_aerosol``. MAM4 has no insoluble dust, so
  contact freezing is zero under JAM. Both routines are compared with the
  compiled ECHAM6.3-HAM2.3 source on designed columns
  (``cloud2m_frz_T63L47.npz``, ``hamfrz_M7.npz``).
- **Removed:** the Niemand (2012) and jcm Lohmann-Diehl INP schemes of the
  JAM ice term with their ``IceNucleationParameters``, the
  ``jam_ice_scheme`` argument of ``echam_physics`` and the ``ice_scheme`` /
  ``ice_nucleation_params`` arguments of ``jam_aerosol_physics``, and the
  ``ice_nuclei`` / ``ice_nuclei_deposition`` outputs of JAM. The Niemand INP
  followed Niemand et al. (2012) correctly; it sat four orders of magnitude
  below DeMott because most mixed-phase clouds held almost no dust.
- **New output:** ``freezing_aerosol.*`` (the eight HAM freezing inputs);
  ``CloudParams2M`` gains the differentiable ``immersion_coefficient_dust``
  (32.3) and ``immersion_coefficient_bc`` (2.91e-3).
- ``CloudParams2M.default()`` sets the ice aggregation coefficient
  ``ccsaut`` to ECHAM's generic 95 (``mo_echam_cloud_params.f90``) instead of
  ECHAM-HAM's 900, and the JAM members take it. ECHAM-HAM's 900
  (``mo_activ.f90``) is a retune for HAM's own aerosol and
  its insoluble dust mode, which MAM4 does not reproduce. The value is a
  swept lever of the retune and keeps 95 (:doc:`design/jam_aerosol_retune`).
  ``ccraut`` keeps ECHAM-HAM's 10.6: it
  acts on droplets from AR&G activation, which JAM runs. The 2M presets
  already set 95 and 15, so the ``echam-2m`` members are unchanged; a 2M
  scheme built in Python without a preset gets 95 instead of 900 (see
  :doc:`v2_to_v3`).
- **Changes results** for the JAM members. Days 5-10 of 10-day
  ``t63-echam-jam`` runs from a state ten days past a cold start (T63L47,
  January), with dev at ``2dc609fd`` against this change at ``2366def6``:

  ============================================  ==============  ==============
  quantity                                      before          after
  ============================================  ==============  ==============
  ice / liquid water path [g/m²]                7.00 / 44.6     22.3 / 40.7
  supercooled liquid water path [g/m²]          27.9            22.1
  supercooled mass fraction, 238-243 K          0.52            0.19
  supercooled mass fraction, 253-258 K          0.83            0.48
  cloud cover [%]                               57.4            56.5
  longwave / shortwave cloud effect [W/m²]      19.66 / -38.33  23.34 / -44.42
  net TOA radiation [W/m²]                      5.94            3.26
  ============================================  ==============  ==============

  Most of the change is ``ccsaut``. The freezing rates alone, measured with
  ``ccsaut = 900`` before the Sonntag entry above, moved the ice water path
  by +0.2 g/m², the liquid water path by -2.5 g/m² and the supercooled
  fraction at 238-243 K from 0.52 to 0.45. The run-to-run noise of these
  runs is about 0.75 g/m² in liquid water path and 0.3 W/m² in the cloud
  effects. The aerosol-free 2M path is unchanged: bit-identical on CPU at
  fixed parameters, and on the GPU the ``t63-echam-2m`` member differs from
  two runs of the unchanged code by less than the run-to-run noise (ice
  water path 27.8 against 28.0 and 27.5 g/m², before the Sonntag entry).
  The release-matrix bands of the ``echam-jam`` members shift and are
  regenerated with #943.
- JAM's dust is removed about five times faster than ECHAM-HAM's. This is a
  finding for the maintainer and is not changed here. Measured over five
  January days, the burden is 2.8 Tg and the lifetime 1.1 days, with 79 %
  of the removal dry. ECHAM6.3-HAM2.3 has 16.5 Tg, 5.3 days and 39 % dry
  (Tegen et al. 2019). The emission is comparable. This also limits the JAM
  immersion freezing.

  See :doc:`science/clouds_microphysics` and :doc:`science/aerosol`.

The ECHAM land evaporates in JSBACH's form and closes a skin energy balance
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The ECHAM hosts' land tile is a prescribed-moisture land (#979). Its
  moisture flux carries JSBACH's humidity factors,
  ``ρ·C·(csat·q_s − cair·q_a)`` (``mo_soil.f90::update_soil``). Bare soil
  evaporates only while ``h·q_s`` exceeds the air humidity, with ``h`` from the
  upper layer's fill (``calc_relative_humidity_upper`` of the bundle's
  ``soilw_rel``, ECHAM6.3's 5-layer path). The forest fraction transpires
  through ECHAM3's canopy conductance (evaluated every step from the absorbed
  PAR, leaf area index 4), limited by JSBACH's water stress between the wilting
  (0.35) and critical (0.75) fractions of ``soilw_am``. Snow and glacier
  evaporate at the potential rate. The beta form it replaces,
  ``cair = csat = soilw_am``, let a dry 307 K soil evaporate like a wet one.
- The land skin temperature is prognostic. It is solved each step from
  ``C_s·dT_s/dt = Rn − SH − LH − G`` with ECHAM's ``update_surfacetemp``,
  coupled implicitly to the lowest level. ``G = Λ·(T_s − stl_am)`` flows into a
  soil at the prescribed ERA5 temperature, which until now *was* the land
  temperature. ``C_s`` and ``Λ`` are JSBACH's top-layer capacity and
  conductance; snow grades the layer by depth, and snow or glacier holds the
  skin at the melting point. The skin is what the radiation, the land albedo,
  the surface saturation and the surface-layer stability see. Between
  radiation calls the surface longwave is re-emitted at it and the change
  heats the lowest level, which the convection and cloud schemes after it see
  (ECHAM's ``radheat``), and the land absorbs the held downward shortwave
  through the land albedo of the last radiation solve
  (``surface.land_albedo_at_solve``).
- Heat and moisture couple to the surface tile by tile through ECHAM's
  Richtmyer–Morton relations (``richtmyer_land``/``_ocean``/``_ice``, then
  ``blend_zq_zt``). Each tile's flux is taken against its own lowest-level
  value, which changes the fluxes of mixed coastal and sea-ice cells by a few
  per cent; pure cells are unchanged. ``VDiffParameters`` defaults
  ``tpfac2``/``tpfac3`` to ECHAM's exact ``1/tpfac1`` and ``1 − 1/tpfac1``
  (they were 0.667/0.333).
- **Compared with the compiled Fortran.**
  ``jcm/physics/surface/echam/jsbach_land_test.py`` reproduces, in float64 at
  round-off and on 324 land columns plus scans through every switch, the
  outputs of ECHAM6.3 / JSBACH's own ``richtmyer_land``,
  ``update_surfacetemp``, ``update_soiltemp``, ``unstressed_canopy_cond_par``,
  the relative-humidity and stress functions, and ``update_soil``'s
  humidity-factor block (``jcm/data/test/echam_land_reference``, compiled
  unmodified in a local harness).
- The bare-soil, dew, water-stress and melt switches keep ECHAM's values and
  take the derivatives of named smooth surrogates; the widths are static
  fields of ``JsbachLandParameters``.
- **New parameters.** ``JsbachLandParameters`` (wilting and critical
  fractions, leaf area index, canopy constants, PAR fraction, soil, snow and
  ice thermal constants, critical snow depth) are differentiable leaves of
  ``TteTkeVerticalDiffusion(land_params=...)``, set from the CLI as
  ``+physics.terms.tte_tke_vertical_diffusion.land_params.<field>=...``.
  ``land_temperature="prescribed"`` holds the land skin at the forcing's land
  temperature with the same evaporation form and coupling, for the
  fixed-SST and fixed-land-temperature effective-radiative-forcing method of
  Andrews et al. (2021). **New outputs** on ``surface``:
  ``land_surface_temperature``, the land tile's ``land_net_radiation``,
  ``land_sensible_heat_flux``, ``land_latent_heat_flux``,
  ``ground_heat_flux``, ``snow_melt_heat_flux``, ``land_heat_storage`` and
  ``land_evaporation``, which close ``Rn = SH + LH + G + melt + storage``,
  ``land_energy_residual`` (``Rn − SH − LH``, the open budget of a prescribed
  skin), and ``cair``, ``csat``, ``water_stress_factor``,
  ``bare_soil_humidity``, ``canopy_conductance`` and ``land_albedo_at_solve``.
- A checkpoint written before this change restores with the skin seeded from
  ``stl_am`` and the land albedo of the last solve unset, which the land
  balance replaces with the current step's until the radiation next solves
  (the new carry fields migrate by name).
- A run's first step from a cold start has no land evaporation, as in ECHAM
  (``init_surface`` sets ``zcair = zcsat = 0``); a restored carry is
  unaffected.
- **Changes results** for every ECHAM configuration. Over days 5-10 of T63
  members against the prescribed land: net TOA radiation +4.1 W m⁻² (1M and
  2M) and +3.7 (JAM), land latent heat about halved (1M 62 → 33 W m⁻²), land
  precipitation −43 to −46 % and ocean precipitation +8 to +11 %, with less
  land cloud. Over 240 days of ``t63-echam-1m``, the spring surface surplus
  ``H + LH − Rn`` of the Sahel, Mexican plateau and India falls from 157, 102
  and 347 to 17, 33 and 30 W m⁻², land precipitation between 40°S and 40°N
  from 6.83 to 3.66 mm d⁻¹ (GPCP 2.74), and land convective rain moves from a
  02 h to a 14 h local-time maximum as the land heats its boundary layer by
  day. The numbers and box tables are in
  :doc:`design/land_skin_energy_balance`. See :doc:`science/surface`.
- **Retune items.** Net TOA radiation rises by 4-5.5 W m⁻² with the drier,
  less cloudy land (7.47 W m⁻² over days 30-240 of the 1M run, inside the
  release gate's 10 W m⁻²). JAM's dust emission falls by 34 % (1741 → 1154
  Tg yr⁻¹, burden −19 %) because the 10 m wind and friction velocity over the
  sources fall by 11 %, so the dust calibration ``jam_dust_nduscale_scale``,
  set against the prescribed land's winds, needs redoing on this surface, and
  with it the dust-borne ice nuclei. The aerosol retune
  (:doc:`design/jam_aerosol_retune`) redoes the dust scale on this surface;
  nothing is retuned in this entry.


ECHAM surface emissivity is ECHAM's cemiss
""""""""""""""""""""""""""""""""""""""""""

- The ECHAM hosts' surface longwave emissivity is ECHAM6.3's single
  ``cemiss = 0.996`` (``mo_radiation_parameters.f90``) for land, open water
  and sea ice, where jcm had 0.95, 0.98 and 0.95 (a grid-box 0.970 on the
  packaged T63 land-sea mask). ``SurfaceOpticsParameters.land_emissivity`` /
  ``ocean_emissivity`` / ``seaice_emissivity`` and the standalone tile-flux
  reference's ``SurfaceParameters.emissivity`` (0.99; ``EchamSurface``
  discards its results) all default to
  ``surface_types.ECHAM_SURFACE_EMISSIVITY``. The radiation's surface
  boundary, the land skin balance and the longwave re-emission read it, as
  ECHAM's radiation, ``land_rad`` and ``update_surfacetemp`` read ``cemiss``.
  The three per-tile values stay separate differentiable leaves. See
  :doc:`science/surface`.
- **Changes results** for every ECHAM configuration, in the direction of more
  surface longwave cooling: a higher emissivity raises the emission
  ``ε·σ·T⁴`` by more than the absorbed downward longwave ``ε·LW↓``, because
  the surface is warmer than the sky's effective temperature. Days 5-10 of
  10-day ``t63-echam-1m`` runs from a spun-up state (T63L47, January,
  instantaneous 6-hourly snapshots, one run per arm), with dev at
  ``f1f0df1e`` against this change at ``610ebed0``:

  ====================================  ===============  ===============
  quantity [W/m² unless stated]         before           after
  ====================================  ===============  ===============
  grid-box emissivity                   0.9703           0.9960
  surface longwave down / up            327.0 / 389.9    326.7 / 391.3
  surface net longwave (down - up)      -62.9            -64.6
  surface net longwave, land            -60.1            -62.6
  land net radiation                    38.9             36.5
  land sensible / latent heat           28.0 / 32.5      27.3 / 32.2
  land skin temperature [K]             276.15           275.95
  net TOA radiation                     -6.38            -6.98
  outgoing longwave radiation           243.2            244.1
  precipitation, global / land [mm/d]   2.314 / 1.993    2.332 / 1.976
  ====================================  ===============  ===============

  The surface longwave change is the emissivity's: ``Δε·(σT⁴ − LW↓)`` on the
  control's own fields is 1.7 globally and 2.8 over land (``Δε`` is 0.026 and
  0.045), against 1.7 and 2.5 in the runs. The runs were not repeated, so the
  run-to-run noise of the turbulent fluxes, skin temperature and
  precipitation is not measured and their changes (a few tenths of W/m² or of
  a kelvin) are not read as an effect.
- **Retune item.** Net TOA radiation falls by 0.6 W/m² and the outgoing
  longwave rises by 0.9. This is the control of the #682 retune: it is
  adopted first so that no tuning absorbs the old emissivity.


Tiedtke-Nordeng takes ECHAM's decisions
"""""""""""""""""""""""""""""""""""""""

- The Tiedtke-Nordeng scheme decides whether a column convects, which plume
  it carries, where the plume stops and where it rains exactly as ECHAM6.3's
  ``cumastr``/``cuasc`` do (#968). The ascent ends at the first interface
  whose test fails (the ``klab = 0`` latch, ``mo_cuascent.f90:294``) instead
  of letting a sigmoid-weighted fraction of the plume climb on; a plume that
  passes no interface above a cloud base at ``klevm1`` leaves the column
  non-convective; the precipitation onset is ECHAM's ``zdnoprc`` switch, at
  the land depth on land columns (land fraction 0.5 or more, ECHAM's default
  binary land-sea mask). There is no CAPE trigger:
  a surface plume needs ``cumastr``'s ``zlo1`` gate, a sub-cloud layer that
  gains moisture and a cloud-base parcel wetter than its environment, and
  its first-guess flux is ``zdqpbl/(g·zqumqe)`` of the whole pre-convection
  moisture tendency, dynamics included, with no floor at the surface
  evaporation. Deep and shallow are ECHAM's moisture-convergence test. The
  ``cubase`` and ``cubasmc`` seeds carry ECHAM's static energy (the
  ``cubase`` parcel is about 0.13 K colder per g/kg of humidity drop between
  the lowest two levels), a failed first ascent leaves no surface plume for
  the second, and a downdraft whose level of free sinking lies above the
  final plume's top is cancelled, as in ``cuflx``.
- Each decision's derivative is that of a logistic surrogate
  (``tiedtke_nordeng/switches.py``, :doc:`design/surrogate_gradients`); the
  value does not depend on the widths.
- ECHAM6.3's compiled convection, run on 758 columns (states of the
  whole-model RCE column in its earlier grey-radiation configuration,
  and the same states under a synthetic ascent, convergence or divergence
  that exercise the mid-level and deep plumes and the ``zlo1`` gate), is the reference
  (``jcm/data/test/echam_cumastr_reference``): with ECHAM's physical
  constants jcm takes its decision on every column and matches its cloud-base
  flux, precipitation and tendencies to 2.1e-12 or better. With jcm's own
  constants, whose ``rv`` is ECHAM's (next entry), 2 of the 758 decisions
  differ, through the latent heats; on that column's days
  40-80 states the port, in float32, convects in 15.1 % of the steps and
  ECHAM in 15.0 %.
- **Breaking:** ``ConvectionParameters`` loses ``trigger_cape``,
  ``smooth_trigger_j`` and ``smooth_rh``; ``smooth_term_buoy``,
  ``smooth_term_mf``, ``smooth_term_cond``, ``smooth_precip_pa`` and
  ``cu_dqcv_width`` become the static surrogate widths
  ``ascent_buoyancy_width`` (now in K), ``ascent_mass_flux_width``,
  ``ascent_condensate_width``, ``precip_onset_width`` and
  ``deep_convergence_width``, joined by ``sub_cloud_supply_width`` and
  ``cloud_base_excess_width``; ``ConvectionParameters`` is a ``flax.struct``
  dataclass; ``flux_tendencies.mass_flux_closure_blend`` is removed. See
  :ref:`v3-tiedtke-parameters`.
- **Breaking (value):** ``ConvectionParameters.cevapcu`` is the leading
  coefficient of ECHAM's sub-cloud rain-evaporation profile
  (``iniphy.f90:87-89``) and scales the whole profile; its default is ECHAM's
  ``1.93e-6`` (v2: ``2.0e-5``, a linear rate in the downdraft's
  pseudo-evaporation, which ECHAM's ``cuadjtq`` replaces). A value on the old
  scale is 10.4 times ECHAM's. ``convective_precip_fluxes`` takes it as
  ``cevapcu_coefficient``. See :ref:`v3-tiedtke-parameters`. A test now fails
  any ``*Parameters`` field that no code reads
  (``jcm/parameter_fields_test.py``); the twelve other such fields found are
  tracked in #999.
- **Changes results** for every ECHAM configuration. Over days 5-10 of
  ``t63-echam-1m`` / ``t63-echam-2m`` runs restarted from 30-day spin-ups
  (against the same physics with the 2.x decisions), the exact decisions
  move the global net TOA radiation by −1.80 / −0.43 W/m² and the shortwave
  cloud effect by −2.08 / −0.13 W/m², and lower precipitation by 0.040 /
  0.046 mm/day (convective 0.033 / 0.029); the area with more than 1 mm/day
  of convective precipitation shrinks from 26.3 to 24.3 % / 26.5 to 24.8 %,
  the humidity at 300 hPa falls by 2.8 / 2.9 % and the liquid held below
  273.15 K by 2.4 / 0.5 g/m² (table below, with the vapour gas constant of
  the next entry). The run-to-run spread of these numbers is 0.19 W/m² in
  net TOA radiation and 0.006 mm/day in precipitation (1M entry above). Both
  runs stay finite, with the sub-cloud supply carrying the lagged dynamics
  and no evaporation floor. In that grey-radiation column, Tiedtke convects
  in 15.1 % of the days 40-80 steps and its precipitation is 2.4 % of
  0.31 mm/d.

.. list-table:: Global means over days 5-10, ``t63-echam-1m`` / ``t63-echam-2m`` from 30-day spin-ups
   :header-rows: 1
   :stub-columns: 1

   * - Quantity
     - 1M control
     - 1M switches
     - 1M switches + rv
     - 2M control
     - 2M switches
     - 2M switches + rv
   * - net TOA radiation (W/m²)
     - 0.68
     - -1.13
     - -1.25
     - 8.93
     - 8.50
     - 8.51
   * - SW cloud effect (W/m²)
     - -47.22
     - -49.30
     - -49.42
     - -50.88
     - -51.01
     - -51.05
   * - LW cloud effect (W/m²)
     - 14.80
     - 15.04
     - 15.06
     - 26.78
     - 26.36
     - 26.43
   * - OLR (W/m²)
     - 246.80
     - 246.52
     - 246.51
     - 234.87
     - 235.16
     - 235.11
   * - liquid water path (g/m²)
     - 69.80
     - 70.11
     - 70.77
     - 40.70
     - 41.15
     - 41.10
   * - liquid below 273.15 K (g/m²)
     - 23.81
     - 21.40
     - 21.80
     - 19.23
     - 18.76
     - 18.70
   * - ice water path (g/m²)
     - 18.70
     - 18.99
     - 18.99
     - 26.69
     - 26.73
     - 26.87
   * - cloud cover (%)
     - 55.25
     - 55.93
     - 56.03
     - 60.24
     - 60.12
     - 60.30
   * - precipitation (mm/day)
     - 2.645
     - 2.605
     - 2.599
     - 2.586
     - 2.540
     - 2.546
   * - convective (mm/day)
     - 2.045
     - 2.012
     - 2.010
     - 1.932
     - 1.903
     - 1.903
   * - large-scale (mm/day)
     - 0.600
     - 0.593
     - 0.589
     - 0.653
     - 0.637
     - 0.643
   * - area, convective > 1 mm/day (%)
     - 26.3
     - 24.3
     - 24.5
     - 26.5
     - 24.8
     - 24.9
   * - column water vapour (kg/m²)
     - 25.05
     - 24.86
     - 24.84
     - 24.92
     - 24.75
     - 24.73
   * - q at 300 hPa (mg/kg)
     - 314
     - 305
     - 304
     - 351
     - 341
     - 340
   * - T at 300 hPa (K)
     - 239.44
     - 239.18
     - 239.18
     - 240.06
     - 239.84
     - 239.83

The vapour gas constant is ECHAM's
"""""""""""""""""""""""""""""""""""

- ``jcm.constants.rv`` is ECHAM-6.3's 461.51 J/(kg K)
  (``mo_physical_constants``), from 461.0 (#968). ``vtmpc1`` (0.6078),
  ``cvv`` and ``rd/rv`` (0.62196) follow; ``eps`` stays 0.622, a field
  separate from ``rd/rv``. With it Tiedtke-Nordeng takes ECHAM6.3's decision
  on 756 of the 758 reference columns under jcm's constants, against 688 at
  461.0: its trigger and ascent tests are thresholds whose marginal outcomes
  follow the saturation humidity and buoyancy that ``rv`` sets.
- **Changes results**, slightly, for every configuration that reads ``rv``:
  every ECHAM configuration (saturation humidity and virtual temperature in
  the convection, vertical diffusion, surface and cloud schemes; the moist
  dynamics of the hybrid-level dinosaur and of the pySES dycore), and the
  physics geopotential, which every dinosaur configuration builds from the
  virtual temperature. SPEEDY's physics reads its own constants, not ``rv``;
  it sees ``rv`` only through that geopotential (and the surface air density
  it publishes), and its 1-day regression trajectory moves by at most 1.6e-6
  (normalized RMS). Over days 5-10 of the runs above, ``rv`` moves the
  global net TOA radiation by −0.13 / +0.01 W/m² and precipitation by
  −0.006 / +0.006 mm/day (table above), within the run-to-run spread.
  ``set_constants(rv=461.0)`` restores the 2.x value.

Sub-grid orographic drag never accelerates the wind
"""""""""""""""""""""""""""""""""""""""""""""""""""

- ``t63-echam-1m`` went 100 % NaN within three to five 12-minute steps on CPU, from a
  cold or a warm start, while the same commit was healthy on a GPU (#981). The
  Lott-Miller SSO energy cap (``mo_ssortns.f90::orodrag``, ``IF (zdis < 0)``) tested a
  difference of nearly equal squares, which is rounding noise at every level the
  drag does not touch. XLA:CPU evaluated that test differently in each of the
  places it was read, so a level the drag did not touch could be rescaled with a
  placeholder denominator and accelerated by 0.3-2.5 m/s² (a 32 m/s level to over
  400 m/s within one 12-minute step) over the Himalaya and Andes. The cap is now ``u*·min(1, |u|/|u*|)``,
  branch-free, with the kinetic-energy change formed from the wind increment.
  ``jax_enable_x64`` did not cure it, since the same mechanism acts in float64.
- **Changes results** only at round-off: it is ECHAM's rescale in exact arithmetic.
  The frictional heating at a level the drag does not touch is now exactly zero
  rather than ~1e-11 K/s of either sign, and no level can gain kinetic energy. CPU
  runs of ``t63-echam-1m`` that diverged now integrate; a GPU's result is unchanged
  to round-off. See ``JAX_gotchas.md`` for how to recognise this class of
  failure.


Radiation derivatives are finite for cloud in the lowest layer and for negligible condensate
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The derivative of the radiation was non-finite, with finite fluxes, in two
  cloud states: cloud in the lowest model layer (the RRTMGP library's halo cells
  repeated the surface layer's cloud paths, giving a cloudy below-surface cell
  at the library's extrapolated temperature) and cloud of negligible but nonzero
  condensate, whose optical depth is a nonzero float32 below about ``1e-19`` and
  whose reciprocal square, in the divisions by the optical depth that both the
  RRTMGP and the grey scheme make, overflows. The halo cells now hold no cloud,
  and both schemes pass no cloud below an in-cloud path of ``1e-11 kg/m²``
  (``1e-8 g/m²``, far below any radiative relevance). See
  :doc:`science/radiation`.
- **Changes results** only in derivatives: the fluxes and heating rates of the
  replayed single columns of ``term_gradients_test.py`` are bit-identical.

Known limitations
^^^^^^^^^^^^^^^^^

Behaviour that ships as documented rather than fixed. Each is an accepted
limitation recorded against the release tracker (#831); :ref:`the migration
guide <v3-support-matrix>` carries the full list with the evidence behind each
verdict.

Positivity corrections are an explicit water-budget source
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- The final physics-interface positivity cap remains in place for water vapor,
  cloud liquid/ice, rain and snow. When summed operator-split sinks overdraw a
  layer, this safety cap can create a small artificial water source; this is an
  accepted known limitation for this release, not a conservative
  redistribution scheme. ``water_positivity_correction`` diagnostics now
  report the exact stop-gradient ``applied - raw`` tendency for specific
  humidity and every water field declared by the active composition, plus
  their total. ECHAM-family compositions, which publish
  ``pressure_thickness``, additionally report
  ``column_water_source`` in kg m\ :sup:`-2`\  s\ :sup:`-1`; SPEEDY does not
  claim a pressure-weighted source because it has no pressure-thickness
  diagnostic (#806).
- Full-model and single-column drivers now return and integrate the same
  verified tendency, and the cross-step humidity carry records that applied
  value. For release monitoring, cumulative positivity correction should be
  negligible relative to cumulative precipitation, with an informational
  target below 0.1%. This target is not yet a runtime failure threshold;
  conservative vertical redistribution is deferred to a separately validated
  physics change.
- Explicit SCM humidity nudging retains a separate non-negativity guard for
  aggressive ``dt/tau`` configurations. Because nudging is user-configured
  outside the physics tendency, any truncation there is not included in the
  physics positivity-correction diagnostics.
- The ``thermo_run`` and Tiedtke ``clouds.qc``/``qi`` floors keep those
  inter-term condensate views non-negative for the terms that read them
  (JAM's aqueous chemistry, the COSP and AeroCom diagnostics); the cloud
  schemes take their condensate from ``cloud_scheme_inputs`` instead. The
  floors can influence what those terms compute but do not directly update
  the prognostic state, so they are intentionally outside the reported
  interface correction; the diagnostics quantify the final positivity cap
  only.

Accepted limitations (proposed)
"""""""""""""""""""""""""""""""

- **pySES publishes no** ``omega``, so ECHAM's mid-level convection trigger
  cannot run on that backend. Model construction **raises** rather than
  substituting zero; the reference's own ``cu_lmfmid=false`` switch is the
  escape hatch, and its spelling depends on whether the physics group is
  term-list or factory-built. A run with the trigger off has no elevated
  convection above a stable layer at all (#698).
- **Only four PhysicsTerms are verified layout-agnostic.** The dynamic audit
  classifies all 62 shipped terms but behaviourally compares grid against
  column hosts for ``AerocomDiagnostics``, ``MoistAirColumnState``,
  ``NudgingTerm`` and ``UpperSponge`` only. The composability claim is
  narrower than it reads (#626).
- **There is no ne30 dust product.** The HAMMOZ dust inputs are native at
  T63, T127 and T255 (T106 conservatively derived), but nothing is published
  on the cubed sphere, so a shipped ne30 configuration has the dust term
  composed but inert. The emission calibration is a T63 quantity, which is
  the resolution every shipped JAM configuration runs at; online aerosol on
  the cubed sphere is separate work. See :doc:`science/boundary_conditions`.
- **The release-validation matrix has two gaps**: the T106 members' multi-GPU
  mesh configurations have never been run for a full year, and ``echam-jam``
  at L95 needs L95 oxidant and ozone inputs staged (#638).
- **PrescribedStateModel re-diagnoses carried state from a cold start.** Each
  time is evaluated independently, so TTE-TKE turbulence takes its spin-up
  value and JAM's carry-stored cloud-borne aerosol is an empty reservoir at
  every re-diagnosed time; construction warns once, naming the slots. Read a
  saved run's own ``jam_cloud_borne.*`` output instead; threading the carry
  is left for after v3.0 (#623, :ref:`v3-limitation-prescribed-carry`).

Regression fixtures follow the supported matrix
"""""""""""""""""""""""""""""""""""""""""""""""


- The GPU-gated climatology regression is now
  ``test_release_matrix_default_statistics``, one sub-test per member of
  ``tools/release_validation/matrix.yaml``, each built through that member's
  **validated preset** rather than a composition written for the test.
  Per-member bands live in ``jcm/data/test/release_matrix/`` (tens to a few
  hundred KB, so a change is reviewable as a diff); the init state each member resumes from is
  hosted on the data mirror under ``bundles/<grid>_<levels>/init_states/`` and
  fetched cache-first. A member's bands and its state are regenerated together
  by ``jcm.data.test.release_matrix.generate_stats.generate(<member>)`` — they
  describe the same window and are only meaningful as a pair. Set
  ``JCM_FIXTURE_STATE_DIR`` to validate freshly generated states before
  publishing them.
- Every band is an **area-weighted global mean** (per level for 3-D fields),
  weighted by the grid's own Gauss-Legendre quadrature weights through
  :func:`jcm.analysis.global_mean`, not an arithmetic mean over latitude
  rings, which would over-weight the small polar rings. Generation and the
  regression's own check share the one reduction.
- Band widths are floored so a regression band can never be narrower than
  the computation's own noise, and never so wide it cannot fail. Each band is
  ``3 * std`` of the daily global means, widened (widest wins) by four
  floors: a few float32 ULP of the field magnitude (``1e-6 * |mean|``, ~8 ULP,
  so the bit-reproducible pure-a ``pressure_full`` levels get a few-ULP band);
  ``1e-6 * max|mean|`` over the variable's **own** profile, for the near-zero
  tail (humidity and condensate aloft) — scaled per variable because a single
  absolute floor sized for humidity would be wider than every aerosol mass
  signal (global means of 1e-10 to 3e-9 kg/kg); ``1e-30``, which pins a
  species with no source in the window to exactly zero, so a source appearing
  for it fails; and ``3 x <var>.noise``, the measured peak-to-peak spread of
  the same window across independent repeats in separate processes. The JAM
  members measure ``noise`` from six repeats, the others from three
  (``REPRODUCIBILITY_REPEATS``).
- Each member is banded on its defining prognostic state, not just core
  meteorology: ``qc``/``qi`` and the MACv2-SP or JAM aerosol optical depth for
  every ECHAM member, and for the JAM members every interstitial and
  cloud-borne aerosol mass and the precursor gases. Number concentrations are
  banded as mass-weighted **column integrals** (``qnc_column``,
  ``qni_column``, and for JAM ``n_total_column`` summed over modes and both
  phases), each layer weighted by its air mass ``dp/g`` — not by
  ``air_density * layer_thickness``, whose thickness is floored at 10 m for
  the physics that divides by it and so overstates thin layers; per-level numbers are not banded, because
  near-threshold activation and nucleation cells make them jump between runs
  of identical code.
- Every reduction behind a band propagates NaN, on both the generating and the
  checking side: a run that goes non-finite in even one cell fails its member
  as non-finite (and ``generate`` refuses to write bands from it), rather than
  averaging the surviving cells into a plausible mean.
- Each band file records the environment its bands were drawn under
  (``bands_environment``: python, jax, jax-rrtmgp, dinosaur, flax, mam4-jax,
  ...), the one its init state was spun up under
  (``init_state_environment``) and its ``reproducibility_repeats``. Bands must
  be generated in a CI-parity environment: bands drawn under a different
  jax-rrtmgp release fail a correct model across the whole column.

Calibration and capability gaps
"""""""""""""""""""""""""""""""

- **The calibration is two emission scales and one cloud-cover set; what is left
  is structural.** Measured in the retune's 365-day T63 L47 JAM year and, for the
  1M and 2M, its control windows (:doc:`design/jam_aerosol_retune` has the
  evidence; the dust regional balance and the sulphur budget are from the
  review of its earlier-tree sweeps): clear-sky OLR is 8 to
  10 W/m² below CERES on every ECHAM host and no swept lever moves it by more
  than 1 W/m²; with ECHAM's cover parameters total cloud cover is 0.58 to 0.59
  as the radiation sees it and 0.46 on the offline maximum-random overlap,
  against 0.63 observed (the calibrated T63 cover set brings the 2M host to 0.65
  and 0.51, with liquid water path 7.5 g/m² higher, and the JAM-2M host to 0.65
  and 0.52 with a SW CRE 3.1 W/m² too strong and 57 g/m² of liquid water path),
  and the 1M and 2M
  convection and microphysics optima buy cloud radiative effect with liquid
  water path instead of cover; dust lifetime is 1.6 d against AeroCom's 4.1 and the regional dust
  source balance is wrong, so dust AOD stays at about half of the observed
  (0.0109 against 0.0213);
  the sea-salt burden is 12 % above the AeroCom mean (from a source four times
  HAM's) yet its coarse-mode extinction per unit mass is low, and total AOD is
  0.074 against 0.145 (0.072 over the last 195 days of the year); sulphate
  is governed by wet removal and its DMS source, and there is no SO2
  deposition; and global-mean precipitation is 0.7 mm/day below GPCP (0.9 to
  1.1 over the ocean).
  See :doc:`design/jam_aerosol_retune` for the evidence.
- **Cloud-borne aerosol is closed as a cycle but not as a full process set**
  (#602 is closed). Interstitial and cloud-borne mass and number exchange on
  activation and evaporation, wet and dry deposition drain the in-droplet
  phase, and aqueous sulfate is produced into the cloud-borne modes. Still
  outstanding: convective processing of the cloud-borne phase (CAM's
  ``aero_convproc`` analogue), no resolved-scale advection of the carry (a
  trade CAM makes too), no cloud-borne sedimentation, and in-droplet mass is
  invisible to the interstitial-only aerosol optics.
- **Aerosol-convection coupling trails by one step.** The two physics-side
  tracer transport terms read the previous step's published profiles, so
  aerosol transport lags the convection driving it by one ``dt``.
- **Middle-atmosphere memory.** T63L95 fits one 40 GB A100; T106L95 does not
  and needs a 4-GPU mesh there. ne30L95 does not fit a single 80 GB A100
  either and needs a memory reduction rather than a faster backend (#595).
- **Betts-Miller is a Python-only entry point.** It is the default convection
  of the single-column RCE layer (``jcm.rce``), which no ``physics=`` or
  ``+configuration=`` group composes, and its coverage is the ``rce_test.py`` /
  ``betts_miller_test.py`` unit suites rather than the release-validation
  matrix.

:ref:`The migration guide <v3-support-matrix>` carries the support matrix and
the evidence behind each accepted-limitation verdict.


Dependencies
^^^^^^^^^^^^


dinosaur is pinned to a release
"""""""""""""""""""""""""""""""

- ``requirements.txt`` requires ``dinosaur>=1.5.0`` instead of the
  semi-Lagrangian development branch, so jcm can be published to PyPI again.
  1.5.0 also fixes the hybrid-coordinate temperature equation
  (neuralgcm/dinosaur#144), so results on ECHAM hybrid levels differ from
  runs made with earlier dinosaur builds; sigma-level runs are unchanged.
- jcm runs dinosaur's float32 GPU matmuls at ``Precision.HIGHEST`` instead of
  1.5.0's bfloat16-emulation defaults. On jaxlib < 0.11.2, an XLA miscompile
  of the default corrupts the inverse transform of log surface pressure, and
  hybrid-level GPU runs drift mass from the northern to the southern
  hemisphere within weeks (a −116 hPa NH−SH surface-pressure asymmetry on a
  Held-Suarez aquaplanet, ~938 hPa at 40–60°N with ECHAM physics). The
  3-pass default used with ``spmd_mesh`` also visibly alters the solution on
  GPU. **GPU runs made with dinosaur 1.5.0 before this fix should be
  repeated**; CPU runs are unaffected. The cost is +2 % per simulated day
  single-device and +8 % with SPMD on a dycore-only case
  (neuralgcm/dinosaur#147).

Other floors that are floors for a reason:

- ``flax>=0.12.1``. That release added ``nnx.Variable.get_value()``, which jcm
  reads every parameter through; on exactly 0.12.0 the lookup falls through to
  the wrapped object and a ``Model`` fails to build.
- ``pyses>=0.1.3.1`` for the optional CAM-SE backend. Earlier builds lower the
  spectral-element contractions to per-gridpoint GEMMs on GPU (1.4x slower,
  1.8x the device memory at ne30L47) and carry an upstream
  tracer-hyperviscosity bug active on the ``quasi_uniform`` path every
  canonical ne30 configuration selects. **ne30 results produced with an older
  pyses should be treated as provisional** (#599).
- The ``cosp`` extra still installs ``jax-cosp`` from a VCS URL, so
  ``pip install jcm[cosp]`` needs git and cannot be resolved from PyPI alone.
  The core install and every other extra are PyPI-resolvable.

v2.0.0b1
--------

This is the first beta for the v2.0 release line. It is intended for early
users who want the new ECHAM/RRTMGP workflow, composable physics API, and
pluggable dynamical-core interface before the stable v2.0.0 tag.

Install the beta explicitly:

.. code-block:: console

   $ pip install "jcm==2.0.0b1"

Because ``2.0.0b1`` is a Python pre-release, normal ``pip install --upgrade
jcm`` users will continue to receive the latest stable release unless they opt
in with ``--pre`` or an exact version pin.

Highlights
^^^^^^^^^^

- Added the :class:`jcm.dycore.base.DynamicalCore` protocol and moved Dinosaur
  behind the shipped :class:`jcm.dycore.dinosaur.DinosaurDycore` backend.
- Refreshed the v2 documentation around dycore ownership, operator-split
  physics, composable physics, and the ECHAM target configuration.
- Made ECHAM the beta target for climate-quality integrations, especially
  ``physics=echam-rrtmgp grid=echam_t63_l47_hybrid``.
- Added persistent checkpoint/resume support for long and preemptible runs.
- Added ozone climatology forcing for ECHAM-RRTMGP.
- Consolidated shared physical constants behind
  :mod:`jcm.constants`, with runtime overrides via
  :func:`jcm.constants.set_constants` before model construction.
- Stabilized ECHAM cloud, convection, vertical diffusion, gravity-wave,
  aerosol, and surface-process wiring for the T63L47 beta target.
- Updated the Python package version to the canonical PEP 440 pre-release
  string ``2.0.0b1``.

Beta Fixes
^^^^^^^^^^

- ``echam_physics(radiation_scheme="rrtmgp")`` now configures the enclosing
  ``ComposablePhysics.band_config`` for RRTMGP bands, matching the Hydra
  runner path and avoiding broadband aerosol optics in Python-created RRTMGP
  compositions.
- Example notebooks were checked for v2 API drift; the ECHAM demo now uses
  ``predictions.to_xarray()``, ``Model(coords=..., terrain=...)``, and
  ``Parameters.float_zeros()``.

Known Beta Caveats
^^^^^^^^^^^^^^^^^^

- The pluggable dycore interface is present, but the shipped production backend
  remains Dinosaur. The Hydra CLI currently selects Dinosaur explicitly.
- Column-vectorized ECHAM physics still assumes a two-dimensional horizontal
  layout. Non-lat/lon dycores need an adapter or flattening step before using
  the shipped column physics packages.
- The beta is intended for named early users and API feedback. Pin the exact
  beta version in user environments and update deliberately between beta tags.
