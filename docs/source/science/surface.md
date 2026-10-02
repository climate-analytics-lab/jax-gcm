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
factor blending the temperature is SPEEDY's chosen approximation. The **ECHAM multi-tile** surface has three tiles: open water at the
prescribed SST, sea ice at ``min(SST, ctfreez)``, and land. The land tile is a
*prescribed-moisture land* with a prognostic skin temperature (*The ECHAM land
tile* below). The ECHAM turbulent surface fluxes are *delivered* by the vdiff
term, which couples the surface to its implicit solve tile by tile (see
{doc}`vertical_diffusion`). ``EchamSurface`` therefore returns no turbulent
tendencies and republishes the vdiff-delivered fluxes as the public
``"surface"`` fields; its one tendency is the longwave re-emission at the new
skin temperature. The tile machinery in ``jcm/physics/surface/echam/``
(``ocean.py``, ``sea_ice.py``, ``land.py``,
``surface_physics.py::surface_physics_step``) is diagnostic-only bookkeeping
and is discarded; the active surface albedos come from
``EchamBoundaryConditions`` (see *Surface albedo* below). The 10 m wind is the
vdiff term's per-tile surface-layer reduction (ECHAM ``nsurf_diag``, see
{doc}`vertical_diffusion`), the one 10 m profile the coupling contract's wind
and the AeroCom ``uas``/``vas`` use.

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

**Status & known limitations.** The ECHAM land tile keeps its soil moisture,
snow cover and deep soil temperature prescribed (see *The ECHAM land tile*;
the rest of JSBACH is #672). Ocean and sea ice carry diagnostic fluxes only,
over prescribed surface temperatures; a prognostic slab-ocean or interactive
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

## The ECHAM land tile

**What we do.** The ECHAM hosts' land is a *prescribed-moisture land*
(``jcm/physics/surface/echam/jsbach_land.py``, solved inside the vdiff term).
It applies JSBACH's evaporation form to prescribed soil moisture and snow, and
carries a prognostic skin temperature from the surface energy balance, coupled
implicitly to the lowest model level.

- **Evaporation** carries JSBACH's humidity factors,
  ``E = ρ·C_h|U|·(csat·q_s − cair·q_a)`` (``mo_soil.f90::update_soil``). Bare
  soil evaporates only while ``h·q_s > q_a`` (``qair_fact = 1``), with ``h``
  from ``calc_relative_humidity_upper`` of the upper layer's fill. The
  vegetated fraction transpires through ``1/(1 + C_h|U|·r_c)``, with
  ``r_c = 1/(g_c·β)`` and the water-stress factor ``β`` running between the
  wilting (0.35) and critical (0.75) fractions of the root-zone fill. The
  snow-covered and glacier land evaporates at the potential rate.
- **Skin temperature** follows ``C_s·dT_s/dt = Rn − SH − LH − G``, with
  ``G = Λ·(T_s − stl_am)`` the flux into a soil whose temperature is the
  prescribed ERA5 soil-layer climatology. ECHAM's ``update_surfacetemp`` solves
  it with the lowest level, against the land tile's own Richtmyer–Morton
  coefficients. ``C_s`` (1.46e5 J m⁻²K⁻¹) and ``Λ`` (10.4 W m⁻²K⁻¹) are
  JSBACH's top-layer capacity and the conductance to its second layer
  (``update_soiltemp``). Snow grades the top layer by depth, glaciers take
  ice, and a snow- or glacier-covered surface is held at the melting point,
  with the excess reported as melt.
- The skin temperature is what the longwave emission, the land albedo's
  melting ramp, the surface saturation and the surface-layer stability see.
  Between radiation calls the surface longwave is re-emitted at the current
  skin temperature, and the change heats the lowest level, as ECHAM's
  ``radheat`` does; the convection and cloud schemes after it see that heating.
  The land absorbs the held downward shortwave through the land albedo of the
  last radiation solve, as ECHAM's JSBACH takes the radiation's net shortwave
  and moves its albedo only at a radiation step.

**What ECHAM/CAM does.** ECHAM6.3 couples JSBACH: a five-layer soil-water and
soil-temperature model, prognostic snow and an interception reservoir, BETHY
canopy conductance from LAI and PAR, and precipitation as the land's input. It
uses the same humidity factors and the same implicit surface balance
(``mo_surface_land.f90::richtmyer_land``, ``update_surfacetemp.f90``).

**Why we differ.**
- `science` — the JSBACH state the forcing bundle does not carry has stand-ins:
  - the root-zone fill is ``soilw_am``;
  - the upper-layer fill is ``soilw_rel`` (ERA5 swvl1 over its field
    capacity, the quantity ECHAM6.3's 5-layer soil reads); ``soilw_am``
    stands in for a bundle without it;
  - the vegetated fraction is the forest fraction;
  - the unstressed canopy conductance is ECHAM3's formula
    (``mo_canopy.f90::unstressed_canopy_cond_par``), evaluated every step with
    half the net shortwave as PAR and a leaf area index of 4, where ECHAM6.3
    uses BETHY;
  - the wet-skin fraction is 0;
  - the snow depth is the bundle's ``SWE = 60 mm·snowc``;
  - one soil layer over the prescribed ``stl_am`` stands in for JSBACH's five;
  - the land emissivity is jcm's 0.95, the value the radiation uses, where
    ECHAM has 0.996.
- `differentiability` — the bare-soil, dew, water-stress and melt switches keep
  ECHAM's values and take the derivatives of named smooth surrogates (widths in
  ``JsbachLandParameters``).

The derivations and the measured effect are in
{doc}`../design/land_skin_energy_balance`.

**Fixed land temperature.** `JsbachLandParameters.land_temperature =
"prescribed"` holds the land skin at the forcing's land temperature every step
(`forcing_land_temperature`, today `stl_am`), with the same evaporation form,
humidity factors and per-tile coupling as the prognostic skin. The surface
energy budget is then open by construction, and `surface.land_energy_residual`
(`Rn − SH − LH`) is the heat the prescription supplies or removes. This is the
fixed-SST and fixed-land-temperature configuration Andrews et al. (2021, *J.
Geophys. Res. Atmos.*, doi:10.1029/2020JD033880) use to measure the effective
radiative forcing without the land's warming response; for that method the
forcing carries the model's own control-run land temperature in `stl_am`.
`stl_am` is a monthly climatology with no diurnal cycle, so the prescribed skin
has none either; a sub-daily land-temperature forcing is #984. From the command
line: `+physics.terms.tte_tke_vertical_diffusion.land_params.land_temperature=prescribed`
(`+physics.land_surface.land_temperature=prescribed` for the factory-built
presets).

**Status & known limitations.** Soil moisture, snow cover and soil temperature
stay prescribed. There is no bucket, no interception, no snow mass or melt
water and no runoff, and precipitation does not reach the land (#672).
Prescribed snow cannot run out, so a snow-covered skin stays at the melting
point for as long as the climatology keeps the snow. The soil under the skin is
held at ERA5's climatology, so where the model's skin runs colder than it the
soil keeps supplying heat (about 30 W m⁻² in the semi-arid boxes' monsoon
season): the limitation of one layer over a prescribed soil, which a
prognostic soil temperature removes (#672). The surface-layer exchange coefficients follow ICON's stable branch
rather than ECHAM6.3's (#982).

**Code pointers.**
- ``jcm/physics/surface/echam/jsbach_land.py`` — ``humidity_factors``,
  ``unstressed_canopy_conductance``, ``top_layer_thermal_properties``,
  ``update_surfacetemp``, ``richtmyer_morton``, ``melt_cap``,
  ``JsbachLandParameters``.
- ``jcm/physics/vertical_diffusion/tte_tke/matrix_solver.py`` —
  ``couple_surface_tiles``.
- ``jcm/physics/vertical_diffusion/tte_tke/vertical_diffusion.py`` —
  ``TteTkeVerticalDiffusion`` (the land inputs).
- ``jcm/physics/surface/echam/surface_physics.py`` —
  ``correct_surface_longwave``.
- ``jcm/physics/forcing/echam_boundary_conditions.py`` — the skin seed.

**Validation evidence.**
- ``jcm/physics/surface/echam/jsbach_land_test.py`` checks against the compiled
  ECHAM6.3 / JSBACH Fortran (``jcm/data/test/echam_land_reference``): the
  humidity factors, the canopy conductance, the top-layer capacity and
  conductance, the Richtmyer–Morton coefficients and the energy balance.
- ``jcm/physics/vertical_diffusion/tte_tke/land_coupling_test.py`` closes the
  land budget to round-off, checks that a dry hot soil stops evaporating while
  a wet one does not, and checks the derivatives through the hinge and the
  implicit solve.
- ``jcm/checkpoint_test.py::TestLandSkinTemperatureMigration`` restores a
  checkpoint written before the field existed.

## Surface albedo

**What we do.** ``EchamBoundaryConditions``
(``jcm/physics/forcing/echam_boundary_conditions.py``) hands the radiation a visible and a near-IR
surface albedo per column, the land / sea-ice / open-water tile average of three
per-tile schemes in ``jcm/physics/surface/echam/albedo.py`` (fractions
``fmask``, ``clip(sice_am, 0, 1 − fmask)`` and the remainder):

- **Land** — ``land_albedo``, JSBACH's broadband scheme
  ``mo_land_surface.f90::update_land_surface_fast``. The snow-free background
  is ``forcing.alb0``; the snow cover is ``forcing.snowc_am``; snow on the
  ground brightens it towards ``0.4 → 0.8`` as the land skin
  temperature falls from the melting point to 5 K below it; forest
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
the last solve bit for bit. The land tile's own albedo is held the same way, on
the ``surface`` carry (``land_albedo_at_solve``), for the land energy balance.

The land-surface maps follow one convention across products, regrids and
consumers (``jcm/data/regridding.py::CONDITIONAL_FIELDS``): ``lsm`` is the land
share of the cell, ``glac`` the glacier share of the **land**, and ``forest``,
``snowc``, ``alb`` and the soil wetness describe the **non-glacier** land
(JSBACH's tiling; the glacier counts as fully wet); ``stl`` is conditional on
the land. Every regrid of these
fields — the bundle builders, the runtime upsampler and the pySES column
sampler — weights each by its own mask
(``jcm/data/regridding.py::regrid_land_surface``), so ocean or glacier
neighbours never dilute a coastal or ice-margin cell. Consumers combine them
once: the snow-covered share of the land is ``glac + (1 − glac)·snowc``
(``jcm/forcing.py::land_snow_cover``, used by the land wetness and sublimation,
by the dust snow gate and as SPEEDY's whole-land snow cover), the whole-land
wetness is ``jcm/forcing.py::land_wetness`` (glacier, and for ECHAM the
snow-covered share, at the potential rate; ECHAM and SPEEDY both call it), the
effective forest ``(1 − glac)·forest``, and the background albedo applies to
the non-glacier tile only (``land_albedo``'s tile average; SPEEDY's
``alb0 + S·(albsn − alb0)`` with the whole-land snow cover is the same
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
