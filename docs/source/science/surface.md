# Surface

**What we do.** Two surface schemes exist. The **SPEEDY bulk** scheme
(``jcm/physics/surface/speedy_surface_flux.py::get_surface_fluxes``) computes
momentum, heat and moisture exchange between the surface and the lowest model
level with bulk-aerodynamic formulae and a stability correction, evaluated
separately over land and sea fractions and returned as the area-weighted grid
mean ``merged = sea + fmask·(land − sea)``; near-surface air properties are
extrapolated to σ = 0.99 using a lapse rate anchored at a fixed sigma, and land
includes an interactive skin-temperature energy balance and an orographic drag
enhancement. The **ECHAM multi-tile** scheme (``jcm/physics/surface/echam/``)
resolves water/ice/land tiles (``ocean.py``, ``sea_ice.py``, ``land.py``) and
does the albedo/radiative/tile energy-balance bookkeeping in
``surface_physics.py::EchamSurface``. Crucially, the ECHAM turbulent surface
fluxes are *delivered* by the vdiff term (which carries the surface exchange as
the bottom-row Robin BC of its implicit solve, see {doc}`vertical_diffusion`), so
``EchamSurface`` returns zero u/v/T/qᵥ tendencies and republishes the
vdiff-delivered fluxes as the public ``"surface"`` fields.

**What ECHAM/CAM does.** ECHAM6's ``vdiff``/``mo_surface`` scheme couples the
surface into a single tridiagonal spanning the column plus the surface exchange,
with a tiled land/ocean/ice surface (JSBACH land; prescribed or slab SST/ice).
SPEEDY's bulk surface fluxes are ``suflux.f90``.

**Why we differ.** Faithful in structure — the ECHAM implicit-flux delivery
(``pev_vdiff`` identity, reported equals received) and the SPEEDY ``suflux`` port
are reproductions. The main scoping choice is `compute` / status: ocean and
sea-ice tiles return zero prognostic temperature tendencies (SST and ice are
prescribed from boundary forcing); slab / mixed-layer evolution lives outside the
repo.

**Status & known limitations.** The ECHAM land tile is simplified (no
frozen-soil / snow-sublimation model); ocean/sea-ice are diagnostic-flux-only
over prescribed surface temperatures — any prognostic slab-ocean or interactive
sea-ice configuration is out of scope.

**Code pointers.**
- ``jcm/physics/surface/speedy_surface_flux.py`` — ``get_surface_fluxes``, the
  land/sea flux helpers, ``get_orog_land_sfc_drag``.
- ``jcm/physics/surface/echam/`` — ``surface_physics.py`` (``EchamSurface``,
  ``combine_surface_fluxes``), ``ocean.py``, ``sea_ice.py``, ``land.py``,
  ``turbulent_fluxes.py``, ``surface_types.py``.

**Validation evidence.** ``jcm/physics/surface/speedy_surface_flux_test.py``;
``jcm/physics/surface/echam/surface_physics_test.py``, ``ocean_test.py``,
``turbulent_fluxes_test.py``, ``surface_types_test.py``. The ``pev_vdiff``
delivered-equals-reported identity is exercised through the vdiff column tests.
