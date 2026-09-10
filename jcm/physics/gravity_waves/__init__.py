"""Gravity wave drag parameterisations.

Four coexisting schemes:

- :mod:`jcm.physics.gravity_waves.hines` — Hines (1997) Doppler-spread
  spectral non-orographic GWD. Faithful port of ECHAM ``mo_gw_hines.f90``.
- :mod:`jcm.physics.gravity_waves.sso` — Lott & Miller (1997) + Lott
  (1999) sub-grid orographic drag (blocking + wave drag + mountain lift).
  Port of ECHAM ``mo_ssodrag.f90``.
- :mod:`jcm.physics.gravity_waves.spectral` — CAM spectral non-orographic
  GWD with the frontogenesis-triggered (Charron & Manzini) source.
  Faithful port of CAM ``gw_common.F90`` + ``gw_front.F90``
  (ref ``cam_cesm2_2_rel``).
- :mod:`jcm.physics.gravity_waves.simple` — placeholder monochromatic
  GWD that used to live under ``hines/``. Kept as a cheap option.
"""
