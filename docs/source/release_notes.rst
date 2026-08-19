Release Notes
=============

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


