# hamseasalt_long reference

Numbers only, from the UNMODIFIED ECHAM6.3-HAM2.3 r7492
`mo_ham_m7_emi_seasalt.f90` (`start_emi_seasalt`, `seasalt_emissions_long`
[HAM `nseasalt=7`], `seasalt_emissions_gong` [`nseasalt=6`, cross-checking
jcm's existing Gong port]), compiled by a standalone harness
(`fortran_harness/echam_hamseasalt/`, own driver + stubs, source not
committed here or there -- see its own git history for provenance).

## Grid (720 cases, full cross product)

- 10 m wind [m/s]: [0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]
- SST [K]: [271.15, 275.0, 278.15, 290.0, 300.0, 305.0] (brackets the Sofiev correction's 271.15-298.15 K
  fitted window; 305 K exercises the unclamped linear extrapolation beyond
  it -- both optional clamps the source offers are commented out, so
  neither applies)
- Sea-ice fraction: [0.0, 0.5, 1.0]
- (land, lake) fraction pairs: [[0.0, 0.0], [0.0, 0.3], [0.3, 0.0], [0.3, 0.3]]
- Sea-salt density: 2165.0 kg/m3 (HAM's own `mo_ham_species.F90`
  registry value, not jcm's MAM4 value of 1900 kg/m3 -- see
  `hamseasalt_provenance.json`)

## Fields (each shape ``(n_cases,)``)

Inputs: ``wind``, ``sst``, ``seaice``, ``land``, ``lake``, ``ss_density``
(scalar). Outputs, per scheme (``long``/``gong``) and class
(``as``/``cs``): ``<scheme>_massf_<class>`` [kg m-2 s-1],
``<scheme>_numf_<class>`` [m-2 s-1].

## O0 vs O2

Max |O2 - O0| per field is in ``hamseasalt_provenance.json``
(``max_abs_diff_o2_vs_o0``); both builds copy source with identical md5s.

## Regenerate

``python py/build_reference_hamseasalt.py`` from this harness's own git
checkout, with ``ECHAMSRC`` pointing at the read-only ECHAM-HAMMOZ source
tree (default ``/data/dwatsonparris/echam6.3.0-ham2.3-moz1.0.r7492/src``).
