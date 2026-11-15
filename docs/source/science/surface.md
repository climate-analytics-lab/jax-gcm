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
with ``T_ice`` the ice-surface temperature
``forcing.sea_ice_temperature`` (SPEEDY's ``tice_am``: the optional
``sea_ice_temperature`` forcing-file variable, or the skin temperature a
coupled thermodynamic sea-ice model hands back) or, when the forcing carries
none, the SST capped at the saline freezing point, ``min(SST, 273.2 − 1.8 K)``
(exact under either SST convention the packaged files use beneath ice), and
``sice = forcing.sice_am`` — in the
SPEEDY convention the ice fraction *of the sea part* of the cell, entering the
sea tile unnormalised exactly as reference ``forcing.f90`` uses it for the sea
albedo, with the ``fmask`` merge supplying the sea-area weighting once (the
ECHAM multi-tile path below instead reads the field under ECHAM's box-tiling
convention, ``clip(sice, 0, 1 − land)``). The
colder, humidity-poor ice surface suppresses the sensible and latent exchange
over ice relative to open water; the sensible flux is linear in ``tsea`` so the
temperature blend equals a flux blend, while for evaporation and the stability
factor blending the temperature is SPEEDY's chosen approximation. The **ECHAM multi-tile** surface has three tiles: open water at the
prescribed SST, sea ice with a prognostic surface temperature, and land. The land tile is a
*prescribed-moisture land* with a prognostic skin temperature (*The ECHAM land
tile* below). The ECHAM turbulent surface fluxes are *delivered* by the vdiff
term, which couples the surface to its implicit solve tile by tile (see
{doc}`vertical_diffusion`). ``EchamSurface`` therefore returns no turbulent
tendencies and republishes the vdiff-delivered fluxes as the public
``"surface"`` fields; its one tendency is the longwave re-emission at the new
skin temperature.
Two sensible-heat definitions are published: ``surface.sensible_heat_flux`` is
ECHAM's own ``ahfs`` (``mo_surface_ocean.f90::postproc_ocean``,
``zcoefw*(ztdif - zcptw)``), a dry-static-energy flux with the *moist* heat
capacity ``cpd(1 + vtmpc2 q)``, so it carries the enthalpy
``(cpv - cpd) T_s E`` of the evaporated vapour (about 8.5 W m⁻² globally);
``surface.sensible_heat_flux_dry`` removes that term tile by tile, with the
skin temperature and evaporation the vdiff solve used
(``matrix_solver.py::couple_surface_tiles``), and is the dry-``cp`` sensible
heat flux of CMIP and reanalyses (``hfss``). The tile machinery in ``jcm/physics/surface/echam/``
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
are reproductions. `differentiability` — SPEEDY's soil-moisture-limited
evaporation hinge, and its ``evap > 0`` activity weight in the land skin
balance, keep their reference values; with ``surface_flux.evap_smoothing`` > 0
they carry a softplus and its sigmoid derivative. Its stability correction
clips the surface-to-air temperature excess to ``±dtheta`` (the stable side
after the ``astab`` damping), a plateau on which the factor is ``1 ± fstab``
and the derivative in ``dtheta`` is exactly zero; the clip keeps its value and
with ``surface_flux.stability_smoothing`` > 0 (K) carries the derivative of a
hyperbolic smooth clip
({doc}`../design/surrogate_gradients`). The main scoping choice is `compute` / status: ocean and
sea-ice tiles return zero prognostic temperature tendencies (SST and ice are
prescribed from boundary forcing); slab / mixed-layer evolution lives outside the
repo.

**Status & known limitations.** The land is SLIM's by default (*The SLIM land*:
water and snow buckets and a soil temperature column); the prescribed-moisture
tile (*The ECHAM land tile*) keeps soil moisture, snow cover and deep soil
temperature prescribed. Neither carries JSBACH's vegetation (LAI, BETHY,
interception). Ocean and sea ice carry diagnostic fluxes only,
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

## The SPEEDY land

**What we do.** SPEEDY runs its slab land (``land_model.f90``): it is
``SpeedySurfaceFlux``'s default, which the ``speedy`` Hydra presets and the
Python ``speedy_physics()`` both inherit, so the two build the same land. The land
surface is one slab — 1 m of soil at 2.5·10⁶ J m⁻³K⁻¹, or 5 m of land ice at
1.93·10⁶ where the background albedo is at least 0.4 — whose temperature is
the prescribed climatology ``stl_am`` plus a prognostic anomaly
``C dT′/dt = hfluxn_land − C T′/tdland``, driven by the land tile's net heat
flux (the conductive flux below the skin when the skin energy balance runs)
and relaxed back on ``tdland`` = 40 days, only where the land fraction is at
least 1/3. Soil moisture and snow stay prescribed. The anomaly is stepped every
physics step, explicitly in the flux and implicitly in the relaxation; a cold
start and a restart from a state without it begin at the climatology. With
``land_model="prescribed"`` the land temperature is the climatology.

**What ECHAM/CAM does.** SPEEDY's Fortran runs this slab by default
(``land_coupling_flag = 1``). Its ``land_model.f90`` states the relaxation in
days and a daily step, but speedy.f90 calls it every 40-minute step with the
per-call relaxation factor unchanged, so there the 40 "days" act in 40 steps
(27 hours) and the land is all but held at the climatology. ECHAM couples
JSBACH (see below).

**Why we differ.** Faithful to SPEEDY, whose slab jcm had dropped as unused code in 2025. `science` —
the prescribed land is an infinite heat reservoir: at T31 it absorbs +9 W m⁻²
over land in the annual mean (2.4 W m⁻² of the global mean), so a coupled
ocean would see a TOA imbalance partly dumped into land that cannot store it,
and land cannot warm under forcing. The slab closes 85 % of that leak and,
at SPEEDY's parameter values, removes the land latent-heat bias
({doc}`../design/speedy_retune`). The
relaxation is itself an energy source/sink of ``C/tdland`` (0.72 W m⁻²K⁻¹ for
the soil slab): ``LandModelParameters.tdland = inf`` gives a free,
energy-conserving slab, which closes the land budget exactly but lets the
uncapped ice-sheet slabs drift up to 20 K warm at T31. `science` — jcm
integrates the stated equation, relaxation in days, with a consistent step
(the per-step call with a per-day factor is an evident quirk of the
refactored Fortran, not a choice). `compute` —
stepping every physics step rather than daily costs nothing and keeps the carry a
plain per-step state.

**Code pointers.** ``jcm/physics/surface/speedy_surface_flux.py``
(``step_slab_land``, ``get_surface_fluxes(land_params=...)``);
``jcm/physics/speedy/params.py`` (``LandModelParameters``);
``jcm/physics/speedy/physics_data.py`` (``LandModelData.stl_anom``).

**Validation evidence.** ``jcm/physics/surface/speedy_surface_flux_test.py``
(``TestSlabLand``: the reference update and its daily coefficients, the land
mask, energy conservation of the free slab, gradients, the coupling to the
fluxes).

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
  and moves its albedo only at a radiation step. The surface temperature the
  radiation solves with, and the re-emission corrects, is ECHAM's own
  ``radtemp``: the tile-weighted T⁴ mean over land, sea ice and ocean
  (``mo_surface.f90`` ~l.1712,
  :func:`~jcm.physics.forcing.echam_boundary_conditions.tile_weighted_radiative_temperature`).

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
  - one soil layer over the prescribed ``stl_am`` stands in for JSBACH's five.
- `differentiability` — the bare-soil, dew, water-stress and melt switches keep
  ECHAM's values and take the derivatives of named smooth surrogates (widths in
  ``JsbachLandParameters``).

The Amazon's dry-season evaporation is limited by that constant-LAI ECHAM3
canopy conductance standing in for BETHY: in 240 days of T63L47 `t63-echam-1m`
its JJA latent heat is 110 W m⁻², 18 % below the prescribed land's (134) and
11 % below MERRA-2's (124), with a wet soil (β ≈ 0.9). An LAI climatology and
the BETHY canopy are not ported: the SLIM land, which replaces this tile as
the default, has no canopy either.

The derivations and the measured effect are in
{doc}`../design/land_skin_energy_balance`.

**Fixed land temperature.** `JsbachLandParameters.land_temperature =
"prescribed"` holds the land skin at the forcing's land temperature every step
(`forcing_land_temperature`), with the same evaporation form,
humidity factors and per-tile coupling as the prognostic skin. The surface
energy budget is then open by construction, and `surface.land_energy_residual`
(`Rn − SH − LH`) is the heat the prescription supplies or removes. This is the
fixed-SST and fixed-land-temperature configuration Andrews et al. (2021, *J.
Geophys. Res. Atmos.*, doi:10.1029/2020JD033880) use to measure the effective
radiative forcing without the land's warming response; for that method the
forcing carries the model's own control-run land skin temperature in
`ForcingData.land_skin_temperature`. Without this optional input, the land
retains the `stl_am` monthly soil-temperature fallback. From the command
line: `+physics.terms.tte_tke_vertical_diffusion.land_params.land_temperature=prescribed`
(`+physics.land_surface.land_temperature=prescribed` for the factory-built
presets).

To build a dated archive, save instantaneous control-run snapshots using
`run.output_averages=true`, `run.snapshot_interval=0.125` (three hours) and
`run.snapshot_variables=[surface.land_surface_temperature]`, then run:

```bash
python tools/build_land_skin_temperature.py control/*_snapshots.nc \
    --output land_skin.nc --control-sha CONTROL_RUN_COMMIT
```

Use `forcing.land_skin_temperature_file=land_skin.nc` in a run with the
prescribed-land switch above and fixed SST. The loader linearly interpolates
absolute timestamps on the matching model grid and checks dated coverage at
run start. The experiment must fit between the archive's actual first and last
snapshots. Snapshots are post-step: the first is one snapshot interval after
the control starts, so start the control early enough to cover the experiment's
initial time. Source timestamps remain unchanged: the recorded local daily cycle
is already present at each longitude, so no additional local-time shift is
applied. This option does not form a repeating month/hour climatology or
regrid between models. The file stores `land_skin_temperature(time, lat, lon)`
in kelvin and records the control commit, source provenance, time span and
conversion recipe. It may also be passed to
`ForcingData.from_bundles(..., land_skin_temperature_file=...)`; in-memory
callers can attach `read_land_skin_temperature(dataset)` with `forcing.copy()`.

Supply snapshots covering the experiment dates. The file loader uses strict
coverage and does not silently repeat a year. An archive supplied to a
prognostic-land run is rejected rather than ignored. The gridded archive
reader targets the spectral backend; pySES requires a separate column-grid
reader and rejects this file option explicitly. The prescribed surface energy
budget remains open by construction; inspect `surface.land_energy_residual`.

**Status & known limitations.** This tile is the ``land_model: prescribed``
alternative to the default SLIM land, which carries the bucket, the snow mass,
the runoff and the soil column this tile lacks. Here soil moisture, snow cover
and soil temperature stay prescribed: there is no bucket, no interception, no
snow mass or melt water and no runoff, and precipitation does not reach the
land.
Prescribed snow cannot run out, so a snow-covered skin stays at the melting
point for as long as the climatology keeps the snow. The soil under the skin is
held at ERA5's climatology, so where the model's skin runs colder than it the
soil keeps supplying heat (about 30 W m⁻² in the semi-arid boxes' monsoon
season): the limitation of one layer over a prescribed soil, which the SLIM
land's soil column removes. The surface-layer exchange
coefficients are ECHAM6.3's own Louis (1979) form on both branches (see
{doc}`vertical_diffusion`).

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

## The SLIM land

**What we do.** The ECHAM presets' land is SLIM (``physics.land_model: slim``,
the default; ``prescribed`` selects the prescribed-moisture tile above, and the
T106, T119 and ne30 configurations keep it: T106 for its calibration and warm
states, T119 and ne30 for lack of a SLIM dataset). SLIM is the Simple Land
Interface Model of Laguë, Bonan & Swann (2019, "Separating the impact of
individual land surface properties on the terrestrial surface energy budget in
both the coupled and uncoupled land–atmosphere system", *J. Climate* 32,
5725–5744, doi:10.1175/JCLI-D-18-0812.1), ported from its source,
[ESCOMP/SimpleLand](https://github.com/ESCOMP/SimpleLand) at tag
``slim0.4.002_cesm2_3_beta15``, ``src/biogeophys/mml_main.F90``
(``jcm/physics/surface/slim/``). SLIM is a land whose behaviour is set by a few
prescribed monthly maps rather than by vegetation physics:

- **A water bucket** ``W`` [kg m⁻²] of prescribed capacity (``bucketdepth``,
  200 kg m⁻² everywhere in the dataset used) fills with rain and snow melt and empties by
  evaporation; water above the capacity runs off. It is the bucket hydrology
  of Manabe (1969, *Mon. Wea. Rev.* 97, 739–774,
  doi:``10.1175/1520-0493(1969)097<0739:CATOC>2.3.CO;2``).
- **A snow bucket** ``S`` fills with snowfall and loses melt and sublimation;
  snow above 5000 kg m⁻² runs off (``snowcap``).
- **Evaporation** ``E = ρ·(q_s − q_a)·β/(r_evap + r_a)`` (``mml_main`` ~1050),
  with ``β = min(W/(0.75·capacity), 1)`` over snow-free land and ``β = 1``
  where there is snow, the evaporative resistance ``r_evap`` a prescribed map
  (60-1000 s m⁻²: about 200 s m⁻² in the Amazon, 750-1000 s m⁻² over semi-arid
  land and deserts) and ``r_a = 1/(C_h|U|)`` the
  aerodynamic resistance. The land does not evaporate more water in a step
  than its buckets hold. The latent heat is that of sublimation where there is
  snow.
- **A ten-layer soil column** (depths ``−0.025·(e^{0.5(i−½)} − 1)`` m, nodes to
  2.87 m, a 1.75 cm top layer) conducts heat implicitly with the prescribed
  conductivity and heat capacity (``soil_tk_1d``, ``soil_cv_1d``; ice values on
  glaciers). Its top layer *is* the land surface: its temperature closes the
  surface energy balance. Each layer below the top holds 300 kg m⁻³ of water
  that freezes and thaws at 273.15 K (``phase_change``).
- **Snow melt** takes the energy that would lift the surface above freezing,
  up to the snow present (``ptl_snow_melt``); the melt water joins the bucket.
- **Roughness** is a tenth of the prescribed vegetation height for momentum
  and a tenth of that for heat (``z0m = 0.1·h``, ``z0h = 0.1·z0m``).
- **Albedo** is the snow-masked blend ``(1 − m)·α_ground + m·α_snow``,
  ``m = S/(S + snowmask)``, of SLIM's ground and deep-snow albedo maps.
- **Glacier** cells (``glc_mask``) take ice's thermal properties and never run
  out of snow.
- **Dust**: the JAM dust emission's snow and soil-wetness gates read the snow
  bucket (as ECHAM's snow cover ``cqsncr·tanh(S/10 mm)``, orography-weighted)
  and the bucket fill ``W/capacity`` (as ``ws/wsmx``), as HAM reads ECHAM's land
  (``mo_ham_dust.f90`` l.1570-1600).

The surface properties are SLIM's CLM5-derived "realistic" maps of Laguë et al.
(2019), the dataset of SLIM's ``realistic_from_1850`` scenario at 1.9°×2.5° in
the CESM input data, ``lnd/slim/surdat/slim_realistic_fromCLM5_alb1850_hc1850_rs1850_f19_20190110.nc``
(albedos, vegetation height and evaporative resistance from an 1850 simulation
of CLM5, Lawrence et al. 2019, *J. Adv. Model. Earth Syst.* 11, 4245–4287,
doi:10.1029/2018MS001583; uniform bucket capacity 200 kg m⁻², snow-masking depth 50 kg m⁻²,
soil conductivity 1.5 W m⁻¹ K⁻¹ and heat capacity 2e6 J m⁻³ K⁻¹), regridded to
the model grid by area-weighted averaging over the CESM 1.9° land
(``forcing.slim_file``; ``auto`` is the mirror's ``bundles/<grid>/slim.nc``,
published for the T63, T106, T127 and T255 Gaussian grids, whose
``provenance`` attribute records the method). Every field steps by calendar
month, as SLIM's ``nc_import`` reads it; in this dataset none varies.

**Calibration levers.** ``SlimParameters.evap_res_scale`` and
``bucket_depth_scale`` (default 1, differentiable) multiply the evaporative
resistance and bucket capacity maps: ``physics.slim_land.evap_res_scale=1.2``.
They are the land's calibration levers, as the emission scales are the
aerosol's.

**What SLIM does.** In CESM, SLIM solves its surface fluxes with its own
Monin–Obukhov surface layer against the coupler's lowest-level state, held
fixed over the step, and one linearised (Newton) solve of the surface and the
soil column together. It reads rain and snow from the atmosphere.

**Why we differ.**
- `science` — the land is solved with the lowest model level, through the land
  tile's Richtmyer–Morton coupling and ECHAM's ``update_surfacetemp`` (the same
  implicit solve as the prescribed tile), so the atmosphere and the land
  exchange one flux. SLIM's soil column enters that solve as the capacity and
  ground flux of its eliminated tridiagonal, which is ECHAM's own coupling of
  JSBACH's soil. The surface-layer exchange coefficients are the host's
  (ECHAM's Louis form with SLIM's roughness), not SLIM's Monin–Obukhov solver,
  so all three tiles use one surface layer; SLIM's canopy displacement height
  is not used, as the host's surface layer has none. The saturation humidity
  is the host's.
- `science` — sublimation is charged where there is snow (the reservoir the
  water leaves), where SLIM charges it below the freezing point; the snow
  bucket evaporates with SLIM's resistance ``r_evap + r_a`` and ``β = 1``.
- `science` — a negative bucket is not clipped to zero: SLIM's clip creates
  water, and the step-start evaporation limit keeps the undershoot to the
  difference between the step-start and the implicit flux.
- `science` — glaciers never run out of snow: melt and sublimation beyond the
  snow present come from the ice (``slim_land.glacier_supply``), so a glacier
  surface stays at the melting point. SLIM's code only gives glaciers ice's
  thermal properties: its glacier snow is an ordinary bucket that can melt
  away, after which the surface warms above 0 °C.
- `science` — the land albedo is broadband, ``0.46·α_vis + 0.54·α_nir`` of
  SLIM's direct-beam albedos, and goes to every band and beam slot, so the
  shortwave the land absorbs is the shortwave the radiation lets it absorb.
- `science` — the land emissivity is the host's (0.996), not SLIM's 1.
- `differentiability` — the bucket clip, the snow switch and the melt hinge
  keep SLIM's values and take the derivatives of named smooth surrogates
  (widths in ``SlimParameters``).

The operator split, the evaporation factor, the energy and water budgets and
the migration of an older state are in {doc}`../design/land_skin_energy_balance`.

**Status & known limitations.** The dataset's bucket capacity and soil
properties are uniform, and its fields do not vary by month. SLIM has no vegetation
physics: no canopy, interception or LAI. A run starting from a state without
the SLIM land seeds it from the forcing (soil at ``stl_am``, the bucket at
``soilw_am`` of its capacity, the snow from the bundle's cover), so the
buckets spin up over the first months. SLIM's dust fluxes are not used.

**Code pointers.**
- ``jcm/physics/surface/slim/slim_land.py`` — the column physics,
  ``SlimParameters``.
- ``jcm/physics/surface/slim/slim_term.py`` — ``SlimLand`` (the buckets at the
  end of the step), ``SlimLandData`` (the carry), ``resolve_state`` (the seed).
- ``jcm/physics/surface/slim/properties.py`` — ``read_slim_properties``.
- ``jcm/physics/vertical_diffusion/tte_tke/matrix_solver.py`` —
  ``couple_surface_tiles`` (the SLIM branch).
- ``jcm/physics/forcing/echam_boundary_conditions.py`` — ``slim_land_albedo``.

**Validation evidence.**
- ``jcm/physics/surface/slim/slim_land_test.py``: the soil grid against
  ``InitCold``, the tridiagonal against its own equations, the long-time
  heating rate and the decay of the gravest mode against the analytic diffusion
  solution, phase change and bucket water budgets to round-off, runoff, the
  snow-first evaporation, melt, albedo and derivatives.
- ``jcm/physics/vertical_diffusion/tte_tke/slim_coupling_test.py``: the land
  energy budget ``d/dt Σ(c_v·dz·T − L_f·ice) = Rn − SH − LH − L_f·melt`` to
  round-off, the column receiving the land's fluxes, a dry bucket that stops
  evaporating, SLIM's resistance form, the evaporation limit, melt at the
  freezing point and the derivatives through β, melt and the soil solve.
- ``jcm/physics/surface/slim/slim_term_test.py``: the property reader, the
  seed, the composition of the presets with ``physics.land_model=slim``, the
  migration of a state without the SLIM land and a T21 model run.

## The ECHAM sea-ice tile

**What we do.** The ice tile's surface temperature is prognostic
(``jcm/physics/surface/echam/sea_ice_temperature.py``, advanced inside the vdiff
term after the implicit solve). ECHAM6.3 with a prescribed ice fraction
integrates it (``mo_surface_ice.f90::s_sicetemp``):

``T_new = (C/Δt·T + F + (k/h)·T_f) / (C/Δt + k/h)``, capped at ``tmelt``,

with ``C = ρ_ice·c_ice·d_ice`` (917 · 2106 · 0.05 m = 9.7e4 J m⁻²K⁻¹, ``hcapice``),
``k = alice = 2.1656 W m⁻¹K⁻¹``, ``T_f = ctfreez = 271.38 K`` (``iniphy.f90``),
and ``F`` the net flux into the ice surface: net shortwave through the ice
albedo the last radiation solve used, ``ε·LW↓ − ε·σ·T⁴``, minus the ice tile's
own sensible and latent heat. The thickness ``h`` is the AMIP boundary's
``siced`` (``clsst.f90:126-128``): 2 m north of the equator and 1 m south of it.
The fluxes are those the implicit solve delivered, evaluated at the step-start
temperature, as in ECHAM, where the lowest-level exchange is solved with the old
``ptsi`` and the temperature is advanced after it; the ice albedo
(``sea_ice_albedo``) and the tile-weighted radiative temperature
(``tile_weighted_radiative_temperature``) read the carried temperature, and the
longwave re-emission of ``EchamSurface`` uses the new one. A cold start begins
at ``min(SST, ctfreez)``. A cell without ice follows the SST, and one that gains
ice starts from that SST, at most ``tmelt`` (ECHAM carries ``ptsi = tmelt`` in
an ice-free cell). The conduction ``alice·(T − T_f)/h`` and the energy of the melt cap
``(C/Δt + k/h)·(T − tmelt)`` are published as ``ice_conductive_heat_flux`` and
``ice_melt_residual_flux``.

- Snow on the ice is not modelled (``psni = 0``): no precipitation is routed to
  the ice tile. The snow terms of ``s_sicetemp`` (snow heat capacity, the
  ``snicecond`` thickness, snow melt) therefore drop out.
- The ice thickness is the hemispheric constant, not a prognostic field, and the
  ice fraction is prescribed: there is no ice growth, melt or lateral transport.
- `differentiability` — the melt cap's value is the exact ``min(T, tmelt)``; its
  derivative is that of ``T − w·softplus((T − tmelt)/w)`` with ``w = 0.5 K``, so a
  surface at the cap still responds to the energy it is given
  ({doc}`../design/surrogate_gradients`).
- `compute` — the explicit-flux form is stable while the flux feedback
  (about 25 W m⁻²K⁻¹ at the sensible-heat exchange of a windy cell) stays below
  ``2(C/Δt + k/h)``, which holds for steps up to about two hours.
- `science` — the ice's own emission is seen at two time levels, as in ECHAM:
  ``s_sicetemp`` charges the ice ``ε·σ·T_n⁴`` (``ice_rad`` runs before it), and
  the same step's re-emission gives the atmosphere the emission at ``T_{n+1}``
  (``radheat`` reads ``pztsnew``, formed from the ice temperature just
  advanced). After ``N`` steps the two ledgers differ by
  ``ε·σ·(T_N⁴ − T_0⁴)·Δt``, which is bounded and returns when the ice does:
  14 kJ m⁻² after 30 days of a ±30 W m⁻² diurnal forcing on a 20-minute step,
  against 690 MJ m⁻² emitted. The re-emission is linearised about the old grid
  temperature for every tile (``correct_surface_longwave``), where ECHAM takes
  the ice tile's ``T⁴`` exactly; the second-order remainder is an accepted
  limitation, −0.009 W m⁻² per unit ice area at the 0.19 K mean step change of
  that forcing and −0.04 W m⁻² at 0.51 K (±80 W m⁻²).

**What ECHAM/CAM does.** Exactly this; ``s_sicetemp``'s uncoupled branch.

**Validation evidence.**
- ``jcm/physics/surface/echam/sea_ice_temperature_test.py`` matches the Fortran
  formula written out in plain floats, and checks the conduction-balance
  equilibrium ``T = T_f + F·h/k``, the melt cap and the derivatives.
- ``jcm/physics/echam/sea_ice_temperature_term_test.py`` runs the composed ECHAM
  stack over ice in both hemispheres.
- ``jcm/checkpoint_test.py::TestIceSurfaceTemperatureMigration`` restores a
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
  at the ice tile's prognostic surface temperature (*The ECHAM sea-ice tile*
  below); broadband.
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
  Sea ice carries no snow. Its temperature is prognostic, so the melt-season
  drop to 0.60 occurs when the ice surface reaches the melting point; the
  snow-on-ice 0.85 occurs only once snow on ice is prognostic.
- `compute` — ``EchamBoundaryConditions`` publishes the open-water direct
  and diffuse albedo separately (``albedo_{vis,nir}_{dir,dif}``)
  for the RRTMGP backend, which maps them to jax-rrtmgp's per-band
  ``sfc_alb_dir``/``sfc_alb_dif`` (see {doc}`radiation`) — the real Fresnel
  direct-beam albedo and the constant diffuse albedo reach the shortwave
  solve as they are, rather than merged into one value. The broadband pair
  (``OCEAN_DIRECT_WEIGHT``-averaged over direct and diffuse, equal weights
  rather than ECHAM's weighting by the previous step's direct/diffuse
  downward irradiance, which jax-rrtmgp does not return) remains for the
  grey two-stream backend, which has one SW band and so never receives a
  direct/diffuse distinction, and for the NN emulator, trained on the
  broadband feature. A column whose sun is down reports the diffuse value in
  every slot; its albedo never enters a shortwave solve.
- `differentiability` — the temperature ramps are single clipped linear
  interpolations, so every constant and the surface temperature carry
  gradients everywhere except exactly at the ramp ends.

**Status & known limitations.** With the default SLIM land the snow mass is
prognostic and the land albedo is SLIM's (*The SLIM land*); this JSBACH albedo
then applies only with ``land_model: prescribed`` (or ``slim_land.albedo:
jsbach``, fed by SLIM's snow bucket). JSBACH's Roesch cover fraction, snow age,
canopy snow and snow on sea ice are not ported; with the prescribed land the
snow albedo follows the climatological cover. A
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

## Surface emissivity

**What we do.** Land, open water and sea ice all have the longwave emissivity
ε = 0.996. Each tile's value is a differentiable leaf of
``SurfaceOpticsParameters`` (``land_emissivity``, ``ocean_emissivity``,
``seaice_emissivity``), defaulting to
``jcm/physics/surface/echam/surface_types.py::ECHAM_SURFACE_EMISSIVITY``.
``EchamBoundaryConditions`` hands the radiation the tile-fraction-weighted
grid-box value together with the albedos, and the radiation solves its
longwave with it; the land tile's own value goes to the land skin balance,
``Rn = SW_net + ε·LW↓ − ε·σ·T⁴``, and to the longwave re-emission at the new
skin temperature (the radiation's value, see *The ECHAM land tile*). The
surface is therefore a near-blackbody that reflects 0.4 % of the downward
longwave, over land, water and ice alike.

**What ECHAM/CAM does.** ECHAM6.3 has one surface emissivity, the parameter
``mo_radiation_parameters.f90::cemiss = 0.996``, shared by every surface. The
radiation uses it as the surface boundary of every longwave band
(``mo_psrad_interface.f90``, ``zsemiss = cemiss``) and as the emission in the
surface-flux bookkeeping of ``radheat.f90``. The surface uses it for the
downward longwave over the tile mix
(``mo_surface_boundary.f90::longwave_down_rad``), the land net longwave
(``mo_surface_land.f90::land_rad``) and the skin temperature's emission
derivative (``update_surfacetemp.f90``); JSBACH's ``Emissivity`` is
``cemiss`` (``mo_jsbach_constants.f90``).

**Why we differ.** The value is ECHAM's. The structure differs in one
respect: `differentiability` — the emissivity is held as three leaves rather
than one constant, so a tile's value can be perturbed or differentiated on
its own (the land balance reads the land leaf, the radiation the blend); the
three are equal by default.
``SurfaceParameters.emissivity``, the standalone tile-flux reference of
``jcm/physics/surface/echam/`` (discarded by ``EchamSurface``, see above),
carries the same value.

**Status & known limitations.** The emissivity is a constant: ECHAM's
``cemiss`` has no spectral, soil-type or snow dependence either, and none is
modelled here.

**Code pointers.**
- ``jcm/physics/surface/echam/surface_types.py`` —
  ``ECHAM_SURFACE_EMISSIVITY``, ``SurfaceParameters``.
- ``jcm/physics/forcing/echam_boundary_conditions.py`` —
  ``SurfaceOpticsParameters``, ``_surface_optical_properties``.
- ``jcm/physics/vertical_diffusion/tte_tke/matrix_solver.py`` — the land
  skin balance's net radiation.

**Validation evidence.**
``jcm/physics/forcing/echam_boundary_conditions_test.py`` pins the defaults at
0.996 for every tile and checks each tile fraction weights its own
emissivity; ``jcm/physics/surface/echam/jsbach_land_test.py`` reproduces the
land balance against the compiled ECHAM6.3 routines at ECHAM's emissivity.

## Surface-layer diagnostics

**What we do.** ``EchamSurface`` publishes the 2 m temperature, 2 m specific
humidity, 2 m dew point, bulk Richardson number and energy-balance residual
``compute_surface_diagnostics`` (``jcm/physics/surface/echam/
turbulent_fluxes.py``) computes every step, as ``surface.temperature_2m`` /
``humidity_2m`` / ``dewpoint_2m`` / ``richardson_number`` /
``energy_balance_residual`` and the per-tile ``surface.temperature_2m_tile``
/ ``humidity_2m_tile`` — serialized as one variable per tile index,
``surface.temperature_2m_tile.0``/``.1``/``.2`` and
``surface.humidity_2m_tile.0``/``.1``/``.2`` (water/sea_ice/land, the same
tile order and ``surface_type``/``tile_index`` attribute convention
{doc}`../design/surface_exchange`'s own wind tiles use), because the output
flattener expands a trailing tile axis into separate variables before CF
attrs are matched by exact name; a bare ``surface.temperature_2m_tile`` is
never a real output name. The 2 m temperature and humidity are a linear blend
between the surface and the lowest MODEL level's values at that level's
actual height above the surface, ``X_2m = X_s + (X_level − X_s)·(2/z_ref)``,
with ``z_ref`` threaded in as ``AtmosphericForcing.reference_height`` (tens
to hundreds of metres, not a bare 10 m — this is the model's lowest full
level, not a literal 10 m observation); the same ``z_ref`` sets the bulk
Richardson number's buoyancy scale. ``z_ref`` is ECHAM's hydrostatic
``pgeom1(klev)/g`` ({doc}`vertical_diffusion`'s "surface layer's reference
height"; the TTE-TKE vertical-diffusion term's own exchange coefficients use
the identical quantity), not half the gap between the two lowest full
levels — 32.5 m vs 57.6 m on the standard L47 levels.

Each tile's surface-humidity anchor is computed FIRST, then fraction-weighted
— the same order every other grid-mean tile diagnostic in this module uses
— rather than interpolating from a fraction-weighted mean surface
temperature: saturation humidity is nonlinear in temperature, so averaging
tile temperatures before taking saturation does not equal averaging the
tile saturation humidities (a mixed warm-water/cold-ice cell diverges
materially). ``temperature_2m`` needs no such ordering, since a linear
blend of a linear per-tile blend commutes with fraction-weighting either
way. Over water and ice the per-tile anchor is ECHAM's own Sonntag (1990)
law (``jcm.physics.thermodynamics``, what the live
``surface_bulk_richardson``/exchange coefficients use); over land it is the
SAME JSBACH boundary value the live implicit surface exchange saw this
step, ``csat·q_sat + (1−cair)·q_air`` (ECHAM ``precalc_land``), not
saturation — a dry land tile (``cair``/``csat`` well below one) is nowhere
near saturated, so anchoring it at saturation made the published land
``humidity_2m_tile`` (and any land-dominated grid cell's ``humidity_2m``)
too humid. The bulk Richardson number's per-tile surface humidity uses the
identical per-tile anchor, for the same two reasons.

This anchor is built in exactly ONE place —
:func:`~jcm.physics.thermodynamics.jsbach_surface_humidity_boundary` — and
``compute_surface_diagnostics`` CONSUMES it as an argument rather than
re-deriving it; ``surface_physics_step`` builds it (water/ice tiles get
``cair = csat = 1``, which collapses the formula to plain saturation) and
the live TTE-TKE exchange's own Richardson number
(``vertical_diffusion/tte_tke/surface_layer.py::surface_bulk_richardson``'s
``qts``) calls the SAME function. This consolidation followed three
separate Codex review rounds, during the work that published these 2 m
and surface Richardson diagnostics (PR 1151), each catching a
different defect in what had been three independent copies of this
formula: saturation at the fraction-weighted mean tile temperature instead
of per-tile, the land tile anchored at saturation instead of this
boundary, and — the last of the three — this boundary evaluated at the
lowest MODEL level's pressure (``AtmosphericForcing.pressure``) instead of
the surface (lowest interface) pressure (``AtmosphericForcing.
surface_pressure``), which the live exchange and
``surface_bulk_richardson`` always used; the 2 m dew point inversion below
inherited the same wrong pressure. Three defects in one quantity, found one
review at a time, is the signature of a quantity with more than one
definition — not three unrelated bugs — which is why the fix is structural
rather than another local patch.

The 2 m dew point inverts the Tetens/Magnus saturation law
``jcm.physics.convection.saturation`` already carries for exactly this
purpose (that module's ``ES0``/``c3les``/``c4les``/``c3ies``/``c4ies`` are
ECHAM's own ``mo_physical_constants.f90`` values, documented there as "the
constants ECHAM itself uses only to invert the 2 m dew point"): given the 2 m
specific humidity, evaluated at the surface pressure (the same pressure the
humidity itself is anchored at — the 2 m level sits a few metres above the
surface, not partway to the lowest model level tens to hundreds of metres
up), ``saturation.dewpoint_temperature`` solves
``qs(T_d, p) = q`` in closed form for ``T_d``, picking the water or ice
branch by whether the water-branch estimate itself falls below the melting
point, and the result is capped at the 2 m temperature
(``MIN(T_2m, T_d)``, exactly as ECHAM's own ``postproc_ocean``/``_land``/
``_ice`` cap their ``zdew2w``/``zdewl``/``zdewi``): the 2 m temperature and
humidity are independently derived quantities with nothing tying them
together (even after the per-tile humidity fix above — fixing HOW
humidity_2m is averaged does not create a saturation constraint between it
and temperature_2m), so the pair can imply a supersaturated (dew point
above air temperature) column — most readily over a cold surface under
warmer, near-saturated air — without this cap.

The bulk Richardson number and the energy-balance residual come from
the SAME explicit bulk-aerodynamic reference pass ``surface.bulk_*`` does
(see {doc}`vertical_diffusion`'s *Water-tile momentum roughness* section) —
not the live Louis Richardson number the delivered fluxes actually use —
so they carry that pass's "reference diagnostic, not the delivered column"
caveat; the 2 m temperature/humidity/dew point do not, since they read only
the step's own forcing and tile temperatures.

Three fields of the underlying diagnostic bundle are deliberately NOT
published, each because a better or the only real version of it is already
public: ``friction_velocity`` would be a second, disagreeing bulk-estimate
u* alongside the live ``vertical_diffusion.surface_friction_velocity`` (the
one the water tile's Charnock roughness actually reads);
``surface_layer_height`` is a bare ``100.0`` constant for every column, never
computed from anything, where the live PBL height is
``vertical_diffusion.pbl_height``; and ``net_radiation``/
``radiation_balance`` are an exact duplicate pair, each exactly recoverable
as ``energy_balance_residual + bulk_sensible_heat_flux +
bulk_latent_heat_flux`` once those three are public, so publishing them
would add no information.

**What ECHAM/CAM does.** ECHAM6.3's ``postproc_land``/``_ocean``/``_ice``
(``mo_surface_land.f90`` and the ocean/ice analogues) diagnose the 2 m
temperature from a stability-corrected profile reduction (the same ``nsurf_diag``
machinery the 10 m wind reduction uses, {doc}`vertical_diffusion`) and the 2 m
dew point from the Tetens form with these same constants, applied to a 2 m
specific humidity built from the lowest level's relative humidity and the
2 m saturation humidity.

**Why we differ.** `science` — the 2 m temperature/humidity use a linear
height interpolation rather than ECHAM's stability-corrected profile
reduction (the one already ported here for the 10 m wind,
``wind_10m_reduction``): a stable near-surface inversion can disagree from
the real profile by several K, where the linear blend under-recovers how
close the air stays to the surface value. Tracked as #1148 (a heat/humidity
analogue of ``wind_10m_reduction`` using the live ``CH``/``CHN`` in place of
``CM``/``CMN``) rather than built here, to keep this PR to the publication
and the two correctness fixes above. `science` — the 2 m dew
point is the exact analytic inverse of jcm's OWN forward saturation law,
not a transcription of ECHAM's ``zdew2w``/``zdewl``/``zdewi`` line: as
transcribed, that line's Magnus argument is short a division by ``eps``
relative to the vapour pressure the Tetens law defines, so feeding it a
specific humidity taken at EXACT saturation over a temperature does not
return that temperature (recovers 282.7 K from 290 K/101325 Pa input, a
measured ~7 K low bias) — reproducing that offset would fail the round-trip
property a dew point must have, so this port inverts jcm's own law instead
(see ``saturation.dewpoint_temperature``'s docstring for the full
derivation).

**Status & known limitations.** The 2 m temperature/humidity blend is linear
in height rather than a stability-corrected profile (#1148).

**Code pointers.**
- ``jcm/physics/surface/echam/turbulent_fluxes.py`` —
  ``compute_surface_diagnostics``.
- ``jcm/physics/surface/echam/surface_physics.py`` — ``EchamSurface.__call__``,
  ``surface_physics_step`` (builds the per-tile ``surface_humidity_tile`` anchor).
- ``jcm/physics/thermodynamics.py`` — ``jsbach_surface_humidity_boundary``,
  the one definition of the per-tile surface-layer humidity boundary,
  shared with ``vertical_diffusion/tte_tke/surface_layer.py::
  surface_bulk_richardson``.
- ``jcm/physics/surface/echam/surface_types.py`` — ``SurfaceData``'s
  ``DIAG_SCALAR_FIELDS``/``DIAG_TILE_FIELDS``/``DIAG_OUTPUT_ATTRS``.
- ``jcm/physics/convection/saturation.py`` — ``dewpoint_temperature``.

**Validation evidence.**
``jcm/physics/convection/saturation_test.py::TestDewpointTemperature`` round-trips
``dewpoint_temperature`` through ``saturation_specific_humidity`` to round-off
over both the water and ice branches and across the melting point;
``jcm/physics/surface/echam/surface_physics_test.py`` checks every published
field is finite and physically sane (2 m temperature between the air and
surface values, dew point never above it) in a composed step, and that the
CF attributes reach the output Dataset.
