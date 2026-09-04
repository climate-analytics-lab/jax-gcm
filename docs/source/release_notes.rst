Release Notes
=============

Unreleased — MACv2-SP removed from JAM; namespaced aerosol output
-----------------------------------------------------------------

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
  than the MACv2-SP SPA floor. The SPA floor remains the ``macv2sp``+2M path's
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

Unreleased — one vertical direction in the output, and CF metadata
------------------------------------------------------------------

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

Unreleased — provenance records the parameters
----------------------------------------------

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
- **``jcm_prov_run_hash`` values change**, because the parameters are now
  folded into the hash. They have to be: every member of a parameter
  sweep shares one code state, config and input set, so without them a
  sweep produced a single run hash for every member.

Unreleased — transient AMIP forcing
-----------------------------------

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

Unreleased — issue-backlog tidy-up
----------------------------------

- **SPEEDY surface fluxes are published as flat 2D maps** (#645, #328,
  #390). ``ustr``, ``vstr``, ``shf``, ``evap`` and ``rlus`` carried land,
  sea and area-weighted values in a trailing channel axis, so output files
  held ``surface_flux.shf.0/.1/.2``. Only the weighted grid mean ever
  reached the atmosphere, and ``hfluxn`` had no channel for it at all —
  ``hfluxn[:, :, 2]`` clamped to the sea value instead of raising, which a
  coupled run consumed as its grid-mean heat flux. Each of these is now a
  single 2D variable holding the grid mean: **``surface_flux.shf.2``
  becomes ``surface_flux.shf``**, and the per-surface ``.0``/``.1``
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
- **Betts-Miller default shallow flavor is now ``SHALLOWER``** (was
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

Unreleased — boundary-condition and emissions data mirror
---------------------------------------------------------

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


