Release Notes
=============

v2.1.0 (unreleased)
-------------------

The v2.1 line makes **online interactive aerosol** (JAM/MAM4) a working
configuration end to end — emissions, microphysics, chemistry, deposition
and radiative effects — and adds the pySES CAM-SE dynamical-core backend.

.. warning::

   **Not releasable yet.** The validated aerosol configuration runs on
   dinosaur's semi-Lagrangian transport (neuralgcm/dinosaur#135), which is
   not in a released dinosaur. ``advection="semi_lagrangian"`` raises a
   clear ``RuntimeError`` on a stock dinosaur; everything else in this
   release line works on the released core with
   ``diffusion.tracer_positivity=true``. Tag v2.1.0 once the upstream
   transport is merged and released, and pin the dinosaur minimum here.

Highlights
^^^^^^^^^^

Interactive aerosol (JAM)
"""""""""""""""""""""""""

- End-to-end online aerosol: prescribed CEDS/biomass emissions plus
  interactive sea salt (Gong 2003), dust and DMS; the MAM4-JAX modal
  microphysics core; gas-phase and aqueous sulfur chemistry; ARG droplet
  activation; heterogeneous ice nucleation on dust/BC; dry deposition,
  sedimentation, and in-cloud/below-cloud wet scavenging.
- Aerosol direct radiative effect from the modal population: per-band Mie
  optics integrated over each mode's lognormal size distribution and fed
  to RRTMGP.
- Wet scavenging is driven by convective as well as stratiform
  precipitation, using the convection scheme's own per-layer precipitation
  formation and in-updraft condensate.

Dynamical cores and grids
"""""""""""""""""""""""""

- New :class:`jcm.dycore.pyses.PysesCamSEDycore` backend (CAM-SE
  spectral elements on the cubed sphere) with Hydra selection, multi-GPU
  element sharding, and a frontogenesis physics-fields provider.
- Optional semi-Lagrangian transport on the dinosaur backend
  (``+advection=semi_lagrangian``) carrying every extra tracer nodally with
  a quasi-monotone limiter — structurally non-negative aerosol transport
  (see the caveat above).
- ECHAM6 middle-atmosphere ``L95`` vertical table (lid ~0.01 hPa) with
  T63/T106/T119 grid presets, plus a ``pyses_ne30l95`` dycore preset.
- **The pySES extra now requires ``pyses >= 0.1.3.1``.** Earlier builds
  lower the spectral-element contractions to one cuBLAS GEMM per grid point
  on GPU (measured 1.4x slower and 1.8x the device memory at ne30L47) and
  carry an upstream tracer-hyperviscosity bug that is active on the
  ``quasi_uniform`` path every canonical ne30 config selects. **ne30 results
  produced with an older pyses are affected and should be regarded as
  provisional** (#599).
- pySES prescribed ozone now uses the flattened column layout
  ``OzoneClimatology`` documents; previously it reached RRTMGP shaped
  ``(1, nlev)`` (#594).

Radiation and clouds
""""""""""""""""""""

- Climatological ozone is now the default (``forcing.ozone_file: auto``,
  packaged per-grid climatology). The previous analytic profile carried
  roughly 7.6x the climatological ozone column and biased clear-sky OLR
  about 12 W/m2 low.
- New ``radiation.total_cloud_cover`` diagnostic: total cloud cover as the
  McICA sub-columns see it, under the same overlap assumption the flux
  solve integrates.
- CAM spectral frontal gravity-wave drag; ``gw_scheme="both"`` to run it
  alongside Hines.
- CloudSat COSP warm-rain hook and per-level precipitation flux profiles.

Diagnostics
"""""""""""

- AeroCom phase-4 diagnostic suite with CMOR post-processing
  (``tools/aerocom_cmor.py``), plus the CALIPSO and MODIS satellite
  simulators alongside CloudSat.
- COSP joint histograms: ``clmodis`` (tau/Reff), LWP+IWP/Reff, the lidar
  scattering-ratio CFAD and ISCCP.
- Per-species/mode/spectral aerosol optics and microphysical process-rate
  and emission-flux diagnostics.

Boundary conditions and emissions: the data mirror
""""""""""""""""""""""""""""""""""""""""""""""""""

All boundary conditions and emissions now come from the Hugging Face
dataset ``climate-analytics-lab/jax-gcm-data`` (issue #515), buildable
end-to-end with ``python -m jcm.data.mirror.build_mirror`` and
reachable from any config path via the ``hf://`` prefix (see
:doc:`design/data_mirror`). **Runs forced from the mirror
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

Infrastructure
""""""""""""""

- Virtual observation operators (stations, tracks, solar-time swaths)
  sampled every model timestep.
- Declared-dependency convention for physics terms, enforced by a static
  audit test.
- ``tools/jam_burden_report.py`` reports column burdens against
  climatological anchors for any dycore/grid, with inferred per-species
  lifetimes from an emissions file.

Documentation
^^^^^^^^^^^^^

- :doc:`design/dinosaur_sl_jam_configuration` — the validated online-aerosol
  configuration, the timestep sweep behind ``run.time_step=15``, middle-
  atmosphere feasibility, and the known biases that remain as calibration
  targets.
- :doc:`design/pyses_cam_se_dycore` and the performance review for the
  CAM-SE backend and its sharding behaviour.

Known Caveats
^^^^^^^^^^^^^

- The semi-Lagrangian dependency above.
- The aerosol configuration is validated for stability and wiring, not
  calibrated: cloud shortwave forcing is too strong, outgoing longwave is
  low, and the sea-salt source under-emits. See the design note for the
  current numbers.
- **The cloud-borne aerosol pathway is not closed** (#602). ARG activation
  supplies droplet number to the cloud scheme, but nothing transfers aerosol
  *mass or number* from the interstitial to the cloud-borne population, and
  there is no resuspension on evaporation and no deposition of cloud-borne
  mass. The 25 ``mc_*``/``nc_*`` tracers are therefore transported but
  essentially empty, and aqueous sulfate production falls back to the
  interstitial accumulation mode everywhere.
- **No physics-side aerosol vertical transport** (#602). Neither vertical
  diffusion nor convection mixes aerosol tracers — both emit only ``qc`` and
  ``qi`` — so the dynamical core's tracer advection is the sole transport for
  every aerosol species. A fidelity gap against ECHAM-HAM and CAM
  (``dropmixnuc`` / ``aero_convproc``).
- Middle-atmosphere runs above T63 need more than one 40 GB GPU. ne30L95
  does not fit on one 80 GB A100 either (#595).

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


