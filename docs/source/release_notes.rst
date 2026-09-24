Release Notes
=============

v3.0.0 (unreleased)
-------------------

v3.0 is a deliberate major release. It makes **online interactive aerosol**
(JAM/MAM4) a working configuration end to end, adds the pySES CAM-SE
dynamical-core backend alongside a semi-Lagrangian-only Dinosaur backend, and
settles a set of unit, API and output contracts that were inconsistent in the
2.x line. Several of those corrections change the climate a configuration
produces.

**Read the** :doc:`v2-to-v3 migration guide <v2_to_v3>` **before upgrading.** It carries the before/after
snippets, the checkpoint-compatibility rules, the support matrix and the list
of accepted limitations; this page is the change list.

Breaking changes
^^^^^^^^^^^^^^^^

Every item here requires a change to code, a config, a saved file, or a reader
of the output. :doc:`v2_to_v3` has the migration for each.

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
  ``load_checkpoint(..., unstamped_scale=...)`` / ``init.unstamped_scale``.
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
  survive a quiet application are recorded in the run's provenance instead;
  the parameters-changed-after-compilation warning, the case #735 was about,
  is also the ``live_parameters_differ_from_compiled`` provenance key.
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
  axes + 2-D area, #533).
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
  ``physics.jam_dust_nduscale_scale``), set to **0.5** at T63; the regional
  ratios stay HAM's, and T106 and ne30 take the Fortran's uniform default
  since their inputs are interpolated from T63. A full
  ``echam-jam-t63-l47`` year emits 829 Tg/yr of D < 10 µm dust, against the
  642 Tg/yr that the parent model's published budget becomes once converted
  to this window; the annual budget is a release-validation gate on any T63
  run of 300 days or more (``DUST_EMISSION_TG_PER_YR``, 400-1300 Tg/yr)
  (#808).
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
- **Semi-Lagrangian transport is the Dinosaur backend's only transport.**
  Every extra tracer rides nodally with a Bermejo-Staniforth quasi-monotone
  limiter, so aerosol non-negativity is structural in transport rather than
  imposed afterwards. There is no Eulerian option and no ``+advection``
  switch. ``diffusion.tracer_positivity`` survives, defaulting to ``auto``
  (on for JAM), but only as a mass-conserving hole-filler at the
  dynamics-to-physics boundary — see
  :doc:`design/dinosaur_sl_jam_configuration`.
- **ECHAM6 middle-atmosphere L95 vertical table** (lid ~0.01 hPa) with
  T63/T106/T119 grid presets and matching ECHAM hyperdiffusion profiles, plus
  an ``ne30`` L95 dycore preset.
- **The Hydra** ``dycore`` **group selects the backend**, dispatched by
  ``jcm.runners.build_model``; a whole pySES run is one command.

Diagnostics and output
""""""""""""""""""""""

- **AeroCom phase-4 diagnostic suite** with CMOR post-processing
  (``tools/aerocom_cmor.py``), and the CALIPSO and MODIS satellite simulators
  alongside CloudSat, including COSP joint histograms (``clmodis`` tau/Reff,
  LWP+IWP/Reff, the lidar scattering-ratio CFAD and ISCCP).
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
  It is identically zero under the grey two-stream scheme, and the NN emulator
  publishes the analytic expectation of that draw rather than sampling it.
  This is *not* the same number as ``jcm.analysis.total_cloud_cover``, the
  maximum-random-overlap post-processing function the release-validation gate
  scores — see :doc:`design/cloud_cover_gate`.
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

Public model clock conversion
"""""""""""""""""""""""""""""

- :meth:`jcm.model.Model.date_from_sim_time` is now the public, JIT-safe way
  to convert elapsed simulation seconds into the same :class:`jcm.date.DateData`
  used by forcing and physics. It documents the stop-gradient boundary,
  nearest-second date rounding and day rollover, and the independently
  timestep-derived ``model_step``. ``Model._date_from_sim_time`` remains a
  compatibility alias in 3.0 and is planned for removal in a later release
  (#758).

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
  reports the compiled ones and both a log warning and a
  ``live_parameters_differ_from_compiled`` key say so: that disagreement
  means an in-place parameter change did nothing to the run. Rebuild the
  ``Model`` to change parameters; making the mutation take effect (or
  fail loudly) is tracked in #735.
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
- The native ne30pg3 terrain published as ``bundles/ne30pg3/sso.nc``
  carried a DEM-validity placeholder ``lsm`` (99.8 % land) instead of a
  land-sea mask; it is replaced by the assembled
  ``bundles/ne30pg3/terrain.nc`` (CESM ``LANDFRAC`` land fraction, exact
  GLL orography), and the pySES ``build_terrain`` now rejects any
  terrain file averaging >0.9 land as a placeholder (#596).


Corrected physics
^^^^^^^^^^^^^^^^^

Fixes that change the climate of a configuration you did not otherwise touch.
:doc:`v2_to_v3` quotes the measured direction and magnitude for each, where one
was measured.

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
  below the melting point and over water above, as ECHAM's ``tlucua`` table
  does, instead of the Sundqvist mixed-phase blend; the latent heat in the
  surface-layer buoyancy switches with the air temperature. The reported
  latent heat flux is ``alhs·E`` over sea ice and carries the sublimation share
  of the snow-covered fraction over land, and snow-covered land evaporates at
  the potential rate (JSBACH's land wetness ``s + (1 − s)·w``).
- New optional static forcing fields ``forest`` and ``glac``
  (``ForcingData.forest_fraction`` / ``glacier_fraction``) carry the land
  cover the land albedo reads; the bundle builders write them from ERA5
  ``cvh`` and the permanent-snow mask. Bundles published before this change
  lack them and load with both ``None`` (no forest masking; ice sheets keep
  their ERA5 background albedo of ≈0.8). ``snowc`` is the snow-covered
  fraction of the non-glacier land, so the snow-covered share of the land
  is ``glac + (1 − glac)·snowc`` (``jcm.forcing.land_snow_cover``, also
  read by the JAM dust snow gate). Forcing files now also carry ``lsm``, the
  land share, and every regrid of the land-surface channels (bundle
  builders, runtime upsampler, pySES column sampler) weights each by the
  part of the cell it describes — ``glac``, ``stl`` and the soil fields by
  the land, ``forest``, ``snowc`` and ``alb`` by the non-glacier land — so
  coastal and ice-margin cells are no longer diluted by their ocean or
  glacier neighbours. On pySES this changes the coastal columns of every
  run on the packaged T63 forcing.
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

RCE initial state seeds a mixed sub-cloud layer
"""""""""""""""""""""""""""""""""""""""""""""""

- ``jcm.rce.rce_initial_state`` now seeds a dry-adiabatic, well-mixed
  sub-cloud layer below ``mixed_layer_top_m`` (default 800 m). This changes
  results for any RCE case composing ``TiedtkeConvection``: ECHAM's ``cubase``
  trigger finds no cloud base at all in a sounding running at ``lapse_rate``
  to the surface. Pass ``mixed_layer_top_m=0.0`` to restore the previous
  profile; see :doc:`design/convective_trigger_soundings` for the reasoning.


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
- The existing ``thermo_run`` and Tiedtke qc/qi floors remain as guards on the
  provisional inter-term state consumed by downstream microphysics. They can
  influence those downstream tendencies but do not directly update the
  prognostic state, so they are intentionally outside the reported interface
  correction; the diagnostics quantify the final positivity cap only.

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
- **Native HAMMOZ dust inputs exist only at T63.** T106 is a
  nearest-neighbour refinement, and there is no ne30 product at all — so a
  shipped ne30 configuration has the dust term composed but inert. Both the
  inputs and the emission calibration are T63 quantities, which is the
  resolution every shipped JAM configuration runs at; online aerosol on the
  cubed sphere is separate work. See :doc:`science/boundary_conditions`.
- **The release-validation matrix has three gaps**: the T106 members' multi-GPU
  mesh configurations have never been run for a full year, ``echam-jam`` at
  L95 needs L95 oxidant and ozone inputs staged, and the single-column
  JAM check (``scm_check.py``) composes grey radiation against the matrix's own
  RRTMGP-for-ECHAM pairing policy (#638).

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

- **The aerosol configuration is validated for stability and wiring, not
  calibrated.** Shortwave cloud forcing is too strong (SW CRE −56 against an
  observed −45 W/m²), LW CRE is low because the ice is too thin, and OLR runs
  about 15 W/m² low (an upstream residual, ``jax-rrtmgp#19``). Aerosol
  lifetimes are mixed: BC and sea salt are in the observed range while
  sulfate is long, i.e. wet scavenging is too weak. See
  :doc:`design/dinosaur_sl_jam_configuration` for the current numbers.
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
