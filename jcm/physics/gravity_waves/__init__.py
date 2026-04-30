"""Gravity wave drag parameterisations.

Three coexisting schemes:

- :mod:`jcm.physics.gravity_waves.hines` — Hines (1997) Doppler-spread
  spectral non-orographic GWD. Faithful port of ECHAM ``mo_gw_hines.f90``.
- :mod:`jcm.physics.gravity_waves.sso` — Lott & Miller (1997) + Lott
  (1999) sub-grid orographic drag (blocking + wave drag + mountain lift).
  Port of ECHAM ``mo_ssodrag.f90``.
- :mod:`jcm.physics.gravity_waves.simple` — placeholder monochromatic
  GWD that used to live under ``hines/``. Kept as a cheap option.
"""
