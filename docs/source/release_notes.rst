Release Notes
=============

Unreleased — issue-backlog tidy-up
----------------------------------

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


