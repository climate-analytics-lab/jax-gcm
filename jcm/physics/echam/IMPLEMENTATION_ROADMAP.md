# Composable Physics Refactor — Housekeeping

Things to clean up *during or after* the scheme-named-terms refactor.
Living tracker — checked off as commits land.

## What was here before

Replaced an out-of-date "ECHAM Physics Implementation Roadmap" (last
updated 2025-12-09) that catalogued ICON-physics V1/V2/V3 release goals.
Most items were either resolved, marked out-of-scope, or unrelated to
the current composable refactor — see git history for the previous
content.

## In flight

The branch ``refactor/composable-physics-flatten`` is migrating each
ECHAM ``apply_*`` wrapper into a scheme-named ``PhysicsTerm`` living
next to its scheme. See
[``docs/design/composable_physics.md``](../../../docs/design/composable_physics.md)
for the design and migration phases. Bit-exact T31L8 + T21L8 regression
must hold across every commit.

## Known follow-ups (not yet addressed)

### Docs and READMEs that still reference removed code

- [ ] ``docs/source/echam_physics.rst`` mentions ``EchamHines``,
      ``EchamSSO``, ``EchamSimpleGwd``. Rename to ``HinesGwd``,
      ``LottMillerSso``, ``SimpleGwd``. Same file references the
      legacy ``EchamRadiation`` / ``EchamRadiationRRTMGP`` /
      ``EchamRadiationEmulated`` / ``EchamConvection`` /
      ``EchamCloudsAndMicrophysics{,1M,2M}`` / ``EchamSurface`` /
      ``EchamVerticalDiffusion`` etc. — sweep once Phase 3 is done.

- [ ] ``jcm/physics/aerosol/macv2_sp_README.md`` still shows the old
      ``get_simple_aerosol(state, physics_data, parameters, forcing,
      terrain)`` signature; the function now takes direct array inputs.
      Update the example and import path.

- [ ] ``docs/echam_physics_perf_plan.md`` mentions
      ``apply_convection``, ``apply_clouds_and_microphysics``,
      ``apply_radiation`` and the apply_*-layer batching plan. Most of
      its perf observations still apply but the function names will be
      stale once their term migrations land — re-anchor the doc to the
      scheme-named term classes after Phase 5.

- [ ] ``REMAINING_ISSUES.md`` (next to this file) catalogues bugs by
      old-style ``apply_*`` line numbers. Either rebase the line
      numbers onto the new term files in Phase 5 or — better — kill it
      entirely and migrate any still-open bugs to GitHub issues with
      the right ``physics/`` labels.

### Code cleanup that's blocked on later phases

- [ ] ``jcm/physics/echam/echam_physics_data.py`` re-exports
      ``ConvectionData`` from
      ``jcm.physics.convection.tiedtke_nordeng`` for back-compat with
      ``radiation/aerosol_radiation_test.py`` and a few other
      importers. Drop the re-export when the file itself goes away in
      Phase 5.

- [ ] After Phase 3 there are no remaining users of ``EchamTermBase``
      / ``_data_from_diagnostics`` / ``_diagnostics_from_data`` —
      ``echam_terms.py`` shrinks to just the ``echam_physics()``
      factory + the (still required) ``ComposableEchamPhysics``
      subclass. Drop the orphans in Phase 4.

- [ ] ``apply_radiation_emulated`` is referenced as still-present in
      the radiation terms' docs but the function is gone — the
      reference comment in ``echam_physics.py`` needs to follow into
      a deletion in Phase 5.

### Smaller nits

- [ ] ``echam_physics_data.py`` still defines ``DiagnosticData``. After
      ``EchamTermBase`` goes away, nothing constructs it (the
      moist-air diagnostics live in the dict as top-level keys); drop
      it then.

- [ ] ``parameters.with_timestep`` and the
      ``isinstance(physics, ComposableEchamPhysics)`` gate in
      ``model.py`` / ``single_column_model.py`` /
      ``prescribed_state_model.py`` go away in Phase 4. ``dt_conv`` is
      a deletion candidate too — every consumer now reads the model
      dt from ``diagnostics["_date"].dt_seconds``.

- [ ] The ``radiation`` and ``_surface`` keys in the diagnostics dict
      are inconsistently styled. ``radiation`` flipped to public in
      Phase 3; ``_surface`` will flip in the surface migration.

- [ ] ``data/bc/t30/clim/{forcing,terrain}.nc`` get rewritten by the
      pytest runs (xarray re-serialising netCDF metadata). They should
      not be touched by tests. Until then, ``git checkout -- jcm/data/bc/``
      after each test sweep before ``git add``.

## Done

- 2026-05-07: Phase 0 — bit-exact reference trajectory tests.
- 2026-05-07: Phase 1 — ``MoistAirColumnState``, ``EchamBoundaryConditions``.
- 2026-05-07: Phase 2 — ``TiedtkeConvection`` template migration.
- 2026-05-07: Phase 3 thin wrappers — ``SimpleGwd``, ``HinesGwd``,
  ``LottMillerSso``, ``SimpleChemistry``, ``Macv2SpAerosol``,
  ``SundqvistCloudFraction``, ``Echam1MMicrophysics``.
- 2026-05-07: Phase 3 radiation triplet — ``GreyTwoStreamRadiation``,
  ``RRTMGPRadiation``, ``NNEmulatorRadiation``.
- 2026-05-07: Phase 3 vdiff/surface/2m — ``TteTkeVerticalDiffusion``,
  ``EchamSurface``, ``Lohmann2MMicrophysics``. **Phase 3 complete.**

## Tests skipped pending rewrite against new term classes

- ``jcm/physics/echam/exchange_coupling_test.py`` — tested the
  ``apply_vertical_diffusion`` ↔ ``apply_surface`` exchange-coefficient
  flow. Now covered by the regression test; rewrite against
  ``TteTkeVerticalDiffusion`` + ``EchamSurface`` for finer-grained
  unit coverage.
- ``jcm/physics/echam/surface_fraction_test.py`` — tested the
  ``apply_surface`` tile-fraction logic. Logic now lives inline in
  ``EchamSurface.__call__`` and is bit-exact-covered by the
  regression test; rewrite as a per-term unit test.
- ``radiation/grey_two_stream/radiation_scheme_test.py::TestRadiationCaching``
  — tested the removed ``_radiation_with_caching`` helper. The
  replacement helpers (``radiation_should_compute`` /
  ``cached_radiation_tendency``) are exercised through the
  ``GreyTwoStreamRadiation`` / ``RRTMGPRadiation`` /
  ``NNEmulatorRadiation`` term ``__call__`` paths.
- ``clouds/sundqvist_test.py::TestAerosolPrecipitationCoupling`` —
  tested the ``_cloud_and_microphysics_column`` helper, now folded
  into ``Echam1MMicrophysics``.
- ``radiation/aerosol_radiation_test.py::test_aerosol_microphysics_droplet_coupling``
  — tested ``apply_clouds_and_microphysics``; equivalent flow through
  ``Macv2SpAerosol`` → ``Echam1MMicrophysics`` is regression-covered.
