# Surface

**What we do.** Two surface schemes exist. The **SPEEDY bulk** scheme
(``jcm/physics/surface/speedy_surface_flux.py::get_surface_fluxes``) computes
momentum, heat and moisture exchange between the surface and the lowest model
level with bulk-aerodynamic formulae and a stability correction, evaluated
separately over land and sea fractions and returned as the area-weighted grid
mean ``merged = sea + fmask·(land − sea)``; near-surface air properties are
extrapolated to σ = 0.99 using a lapse rate anchored at a fixed sigma, and land
includes an interactive skin-temperature energy balance and an orographic drag
enhancement. The sea tile blends open water and sea ice the way SPEEDY's
ocean/ice coupler does (``sea_model.f90``): the bulk formulae see one
ice-weighted sea-surface temperature ``tsea = (1 − sice)·SST + sice·T_ice``,
with the ice surface at the saline freezing point (``273.2 − 1.8 K``, jcm
carrying no separate ice temperature) and ``sice = forcing.sice_am`` — in the
SPEEDY convention the ice fraction *of the sea part* of the cell, entering the
sea tile unnormalised exactly as reference ``forcing.f90`` uses it for the sea
albedo, with the ``fmask`` merge supplying the sea-area weighting once (the
ECHAM multi-tile path below instead reads the field under ECHAM's box-tiling
convention, ``clip(sice, 0, 1 − land)``). The
colder, humidity-poor ice surface suppresses the sensible and latent exchange
over ice relative to open water; the sensible flux is linear in ``tsea`` so the
temperature blend equals a flux blend, while for evaporation and the stability
factor blending the temperature is SPEEDY's chosen approximation. The **ECHAM multi-tile** scheme (``jcm/physics/surface/echam/``)
carries water/ice/land tile machinery (``ocean.py``, ``sea_ice.py``,
``land.py``), but its per-step albedo/radiative/tile energy-balance computation
(``surface_physics.py::surface_physics_step``) is currently **diagnostic-only
and discarded**: the tile state is re-initialised from prescribed forcing every
call (no prognostic memory — no precipitation input, snow, or soil moisture;
#672), and the active surface albedos come from ``EchamBoundaryConditions``
instead (see *Surface albedo* below). The ECHAM turbulent surface fluxes are *delivered* by the vdiff term
(which carries the surface exchange as the bottom-row Robin BC of its implicit
solve, see {doc}`vertical_diffusion`), so ``EchamSurface`` returns zero
u/v/T/qᵥ tendencies and republishes the vdiff-delivered fluxes as the public
``"surface"`` fields.

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

**Status & known limitations.** The ECHAM land tile is simplified: snow cover
is prescribed (snow-covered land evaporates at the potential rate and the
latent heat of the land moisture flux takes the sublimation share of the
snow-covered fraction, see {doc}`vertical_diffusion`, but there is no snow
mass, melt or frozen-soil model — #672); ocean/sea-ice are
diagnostic-flux-only over prescribed surface temperatures — any prognostic
slab-ocean or interactive sea-ice configuration is out of scope.

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

## Surface albedo

**What we do.** ``EchamBoundaryConditions``
(``jcm/physics/forcing/echam_boundary_conditions.py``) hands the radiation a visible and a near-IR
surface albedo per column, the land / sea-ice / open-water tile average of three
per-tile schemes in ``jcm/physics/surface/echam/albedo.py`` (fractions
``fmask``, ``clip(sice_am, 0, 1 − fmask)`` and the remainder):

- **Land** — ``land_albedo``, JSBACH's broadband scheme
  ``mo_land_surface.f90::update_land_surface_fast``. The snow-free background
  is ``forcing.alb0``; the snow cover is ``forcing.snowc_am``; snow on the
  ground brightens it towards ``0.4 → 0.8`` as the land temperature
  ``stl_am`` falls from the melting point to 5 K below it; forest
  (``forcing.forest_fraction``) hides the ground snow behind a canopy
  fraction ``forest·(1 − exp(−max(LAI, 2)))``; glacier cells
  (``forcing.glacier_fraction``) take the glacier albedo ``0.75 → 0.85`` over
  the same ramp; the result never falls below the background. It is broadband
  and enters both bands, as JSBACH writes it to ``albedo_vis`` and
  ``albedo_nir``.
- **Sea ice** — ``sea_ice_albedo``, ``mo_surface_ice.f90::update_albedo_ice``:
  bare ice ``calbmni = 0.60`` at the melting point to ``calbmxi = 0.75`` 1 K
  below it (``calbmns``/``calbmxs`` = 0.70/0.85 with more than 1 cm of snow),
  at the ice tile's temperature ``min(SST, ctfreez)``; broadband.
- **Open water** — ``ocean_albedo``,
  ``mo_surface_ocean.f90::update_albedo_ocean``: the direct beam follows the zenith-angle fit
  ``0.026/(μ₀^1.7 + 0.065) + 0.015(μ₀−0.1)(μ₀−0.5)(μ₀−1)`` (+0.0082 visible,
  −0.007 near-IR), diffuse light ``calbsea = 0.07``; the RCE column uses
  ECHAM's ``lrce`` constant 0.07 for the direct beam.

The term evaluates these every step and hands them to the radiation as that
step's input (``jcm/physics/radiation/__init__.py::SURFACE_OPTICS_KEY``); the
radiation reads them only when it solves and publishes the values it solved
with in ``radiation.surface_albedo_*``, holding them between solves. The
published albedo, the reflected flux and the heating therefore always describe
one solve, and the zenith-dependent open-water albedo enters at the solve-time
sun, as in ECHAM, whose radiation reads the surface albedo at a radiation step
(``trigrad``) and replays the transmissivities in between (``radheat``).
The hand-off is step-local: it is dropped before the cross-step carry, so it is
never checkpointed, and a restart replays the held ``radiation.surface_*`` of
the last solve bit for bit.

The land-surface maps follow one convention across products, regrids and
consumers (``jcm/data/regridding.py::CONDITIONAL_FIELDS``): ``lsm`` is the land
share of the cell, ``glac`` the glacier share of the **land**, and ``forest``,
``snowc`` and ``alb`` describe the **non-glacier** land (JSBACH's tiling); the
soil fields and ``stl`` are conditional on the land. Every regrid of these
fields — the bundle builders, the runtime upsampler and the pySES column
sampler — weights each by its own mask
(``jcm/data/regridding.py::regrid_land_surface``), so ocean or glacier
neighbours never dilute a coastal or ice-margin cell. Consumers combine them
once: the snow-covered share of the land is ``glac + (1 − glac)·snowc``
(``jcm/forcing.py::land_snow_cover``, used by the land wetness and sublimation
and by the dust snow gate), the effective forest ``(1 − glac)·forest``, and the
background albedo applies to the non-glacier tile only (``land_albedo``'s tile
average).

Every constant is a differentiable leaf of ``EchamSurfaceAlbedoParameters``
(held in ``SurfaceOpticsParameters``), at the ECHAM 6.3 T63/T127/T255 values;
``EchamSurfaceAlbedoParameters.echam_t31`` carries ECHAM's T31 sea-ice set.

**What ECHAM/CAM does.** ECHAM 6.3 evaluates the same three routines each step
in ``mo_surface.f90`` and averages the tile albedos band by band. Its land
albedo comes from JSBACH, which offers two schemes: the broadband
``update_land_surface_fast`` above (``use_albedo=.FALSE.``, the code
default) and the per-band ``update_albedo_snowage_temp``
(``use_albedo=.TRUE.``, which the standard MPI-M run setup selects), which needs per-band soil and canopy
albedo maps, the land-cover-type composition, LAI, a canopy snow store and a
prognostic snow age. Its default sea-ice path is the melt-pond scheme
(``update_albedo_ice_meltpond``, ``lmeltpond=.TRUE.``), which needs prognostic
ice thickness, pond depth and snow age. Its snow cover is diagnosed from a
prognostic snow depth (Roesch et al. 2002: ``0.95·tanh(100 S)·√(1000 S/(1000 S
+ 0.15 σ_oro))``). The land-cover constants are those of the JSBACH land-cover
library ``lctlib_nlct21.def``, identical for every non-glacier type.

**Why we differ.**
- `science` — jcm's land surface is prescribed: a broadband snow-free
  background ``alb`` (the minimum monthly ERA5 forecast albedo), a cover
  fraction ``snowc = min(1, SWE/60 mm)``, a forest fraction (ERA5 high
  vegetation cover ``cvh``) and a glacier mask (the cells whose ERA5 snow
  never melts), both per unit land, built by ``jcm/data/mirror/bundles.py``
  (the packaged T63 file carries ECHAM's own ``FOREST``/``GLAC`` with the
  same ERA5 snow cover). The broadband ``use_albedo=.FALSE.`` scheme is the
  JSBACH path those inputs support exactly; the per-band scheme would need
  maps that do not exist here. For
  the same reason sea ice uses ``lmeltpond=.FALSE.``. Snow cover is the
  prescribed climatology rather than ECHAM's Roesch fraction of a prognostic
  snow depth: without prognostic snow there is no depth to apply it to. With
  no LAI the canopy fraction uses JSBACH's own ``MAX(LAI, 2)`` floor
  (``forest·0.865``, at most 13 % of the forest fraction below a dense
  canopy's), and with no canopy snow store the canopy is snow-free — so the
  canopy part of a snow-covered forest shows the background albedo instead
  of JSBACH's snowy-canopy 0.20 (lower for the usual dark forest
  background; above 0.20 the ``MAX(…, bg)`` floor makes the two agree).
  Sea ice carries no snow and sits at ``min(SST, ctfreez) = 271.38 K``,
  below the 1 K ramp, so it is effectively a constant 0.75 (``calbmxi``):
  neither the melt-season drop to 0.60 nor the snow-on-ice 0.85 occurs
  until ice temperature and snow on ice are prognostic.
- `compute` — the RRTMGP surface boundary takes one albedo per column for
  both the direct beam and diffuse light (see {doc}`radiation`), so the
  open-water direct and diffuse albedos are merged with equal weights
  (``OCEAN_DIRECT_WEIGHT``) instead of ECHAM's weighting by the previous
  step's direct/diffuse downward irradiance, which the library does not
  return. The even split halves the worst-case error of either pure choice
  (all-direct clear sky, all-diffuse overcast). It is a stopgap until the
  library accepts separate direct/diffuse and per-band surface albedos
  ([jax-rrtmgp issue 38](https://github.com/climate-analytics-lab/jax-rrtmgp/issues/38)). A column whose sun is down
  reports the diffuse value; its albedo never enters a shortwave solve.
- `differentiability` — the temperature ramps are single clipped linear
  interpolations, so every constant and the surface temperature carry
  gradients everywhere except exactly at the ramp ends.

**Status & known limitations.** Prognostic snow — snow mass, the Roesch
cover fraction, snow age, canopy snow and snow on sea ice — is the open half
of #672; until it lands, snow albedo follows the climatological cover. A
forcing bundle built before ``forest``/``glac`` were published loads with both
``None``, which the albedo reads as no forest and no glacier: ice sheets then
take their ERA5 background albedo (≈0.8) instead of the glacier ramp, and snow
under forest is not masked.

**Code pointers.**
- ``jcm/physics/surface/echam/albedo.py`` — ``land_albedo``,
  ``sea_ice_albedo``, ``ocean_albedo``, ``ocean_albedo_per_band``,
  ``EchamSurfaceAlbedoParameters``.
- ``jcm/physics/forcing/echam_boundary_conditions.py`` —
  ``EchamBoundaryConditions``, ``SurfaceOpticsParameters``.
- ``jcm/data/mirror/bundles.py`` — ``land_surface_fields`` (every
  land-surface channel under the convention), ``translate_land`` (the snow
  cover).
- ``jcm/data/regridding.py`` — ``regrid_land_surface``,
  ``regrid_conditional_fraction``, ``CONDITIONAL_FIELDS``.

**Validation evidence.** ``jcm/physics/surface/echam/albedo_test.py`` pins each
scheme at hand-evaluated ECHAM points; ``jcm/physics/forcing/echam_boundary_conditions_test.py`` checks that ``alb0``, ``snowc_am`` and the
land-cover maps reach the radiation input.
