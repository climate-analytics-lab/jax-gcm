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

- [ ] ``EchamCloudsAndMicrophysics`` in ``echam_terms.py`` is the
      pre-split single-term cloud+microphysics variant. Already
      deprecated, not in the default factory. Drop in Phase 5 with
      everything else.

- [ ] ``EchamCloudsAndMicrophysics2M`` and ``apply_microphysics_2m`` /
      ``apply_clouds_and_microphysics`` haven't migrated yet — final
      Phase 3 batch. The 2M migration must preserve the qnc_prev /
      qni_prev state-carry across radiation sub-steps (tracked in
      ``apply_microphysics_2m``).

- [ ] ``EchamSurface`` migration must preserve the implicit-stability
      damping inside ``apply_surface``. That damping is what stabilises
      the T63L47 + real-terrain runs the user is debugging on
      ``debug/echam-2m-micro-stability`` — schedule it last and verify
      against the moist-run reproducer in ``.claude/moist_run_debug_log.md``.

- [ ] ``EchamVerticalDiffusion`` similarly carries the surface-tile
      construction (Charnock, exchange coefficients) and TKE clamp.

- [ ] After ``apply_surface`` + ``apply_microphysics_2m`` migrate, the
      ``EchamTermBase`` base class (echam_terms.py:135) loses all its
      consumers and can be deleted along with the
      ``_data_from_diagnostics`` / ``_diagnostics_from_data`` helpers.
      Until then they're the bridge that keeps the legacy-style terms
      reading data populated by the new public-key writers.

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
