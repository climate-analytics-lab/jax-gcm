# Kaercher-Lohmann cirrus nucleation reference (jax-gcm#1017 task 2 part 2a)

Runs the UNMODIFIED ECHAM6.3-HAM2.3 r7492 `mo_cirrus.f90::xfrzmstr` (which calls `xfrzhom`/`xicehom`) directly on independently designed single-level cells. `xfrzhet`/`xicehet` and the aerosol-size-effect branches of `xicehom` are UNREACHABLE in any supported HAM configuration (`lhetfreeze` is an `em_error` without `-DWITH_LHET`, `nosize` is a compile-time `.true.`) and are therefore not exercised or ported. Numbers only; no ECHAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference zri_m 0.0e+00, znicex_m3 0.0e+00.

24 designed cells (`meta/names`, `meta/description`): temperature 190-238 K, ice supersaturation below/at/above `SCRHOM(T)`, updraft 0.01-2 m/s, aerosol number 1e6-1e10 m-3 plus the orchestrator's own depletion floor (modelling 'pre-existing ice has consumed the available aerosol', since XFRZMSTR itself has no such argument), and both sides of the `ZTHOMI` gate.

## Arrays

`in/t` [K], `in/susati` [S_ice-1], `in/verv_cms` [cm/s], `in/apn_cm3` [1/cm3], `in/apr_cm` [cm, inert on this NOSIZE path], `in/apsig` [inert], `in/p_pa` [Pa]. `out/zri_m` [m], `out/znicex_m3` [1/m3] -- XFRZMSTR's own output units (it converts internally: `ZRI=ZRI1*1e-2`, `ZNICEX=ZNICE*1e6`).

## Regenerate

`python build_cirrus_reference.py <jcm>/jcm/data/test/echam_cloud_reference`.
