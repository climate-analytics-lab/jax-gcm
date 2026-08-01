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
  T63/T106/T119 grid presets.

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
- Middle-atmosphere runs above T63 need more than one 40 GB GPU.

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


