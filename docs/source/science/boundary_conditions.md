# Boundary conditions, ozone and forcing

**What we do.** ``jcm/forcing.py::ForcingData`` is the pytree of all
boundary conditions (SST, sea ice, land T, snow/soil, GHG scalars, ozone
climatology, aerosol/emission/oxidant/DMS/dust fields), updated by convention
through ``.copy(...)`` rather than field assignment (``tree_math.struct`` does
not freeze instances). The build engine lives in ``jcm/forcing_assembly.py``
(``build_forcing`` and the attach chain) over the typed input-resolution layer
in ``jcm/data/input_resolution.py``, driven by the Hydra ``forcing`` group
(``jcm/config/forcing/{default,from_file,amip,era5}.yaml``); the CLI's
``jcm/runners.py::build_forcing`` is a thin delegate.

- **Climatological vs transient bundles.** ``jcm/config/forcing/amip.yaml`` and
  ``jcm/config/forcing/era5.yaml`` are
  transient yearly bundles: one file per year, a ``years`` range, and
  ``align: by_date_interp`` (linear interpolation between month-start / mid-month
  boundary samples). Plain single-file paths use ``align: auto``, which chooses
  ``wrap_year`` (climatology, indexed by fraction-of-year) for ≤~1-year spans and
  ``by_date`` otherwise.
- **Per-product coverage clamping.** ``jcm/forcing.py::expand_yearly_files``
  pads the requested range by one year each side, **clipped to the product's
  ``available_years``**, so ``by_date_interp`` has bracketing samples across the
  Jan-1/Dec-31 boundaries. Products with different coverage in one configuration
  are supported via a per-product override (e.g. ``ozone_available_years``). Beyond
  a product's coverage the time lookup clamps to the nearest end sample
  (``jcm/forcing.py::make_time_series``).
- **GHG extrapolation beyond coverage.** CO₂/CH₄/N₂O ride along in the yearly
  files as global-mean annual series. For run dates past the product endpoint
  (PCMDI-AMIP ends 2022; the ERA5 bundle runs later) the time lookup **clamps GHG
  to the last sample** — slowly-evolving-species behaviour, not linear
  extrapolation.
- **Ozone auto-resolve.** ``forcing.ozone_file: auto`` (the shipped default)
  resolves a packaged CMIP6 ozone climatology matching the grid; no match degrades
  to the analytic profile with a warning; an explicit path loads strictly (grid
  lat/lon passed so flipped/shifted grids are caught); ``null`` silently disables
  it (analytic fallback). The grid-aware loader is
  ``jcm/ozone_climatology.py::OzoneClimatology``.

**What ECHAM/CAM does.** The standard prescribed-AMIP protocol: PCMDI-AMIP
mid-month SST/sea-ice boundary values, a CMIP6 ozone climatology, global-mean GHG
series. The ERA5 bundle is a jcm-specific internally-consistent transient
alternative (all fields on one ERA5 land-sea mask).

**Why we differ.**
- `science` — the ``era5`` and ``amip`` SST constructions differ by ~0.16 K RMS
  and must not be mixed within one configuration (documented in
  ``jcm/config/forcing/era5.yaml``).
  GHG and ozone beyond coverage are clamped, not extrapolated.
- `differentiability` — all ``ForcingData`` numeric fields (GHG scalars,
  emission/ozone fields) are pytree leaves, so autodiff traces through them and
  boundary conditions remain calibratable by gradient (their *effect* is
  generally nonlinear — gas optics, reaction integration — so sensitivities are
  state-dependent).

**Dust boundary conditions.** The Tegen scheme (see {doc}`aerosol`) needs five
fields rather than one, all published per grid on the data mirror and picked up
by ``forcing.dust_file`` / ``dust_preferential_file`` / ``dust_soil_types_file``
/ ``dust_regions_file`` / ``dust_roughness_file`` (all ``auto`` by default):

| key | variable(s) | axis | role |
|---|---|---|---|
| ``dust_file`` | ``pot_source`` | 12 monthly records | effective-LAI erodible fraction — both the gate and a linear factor |
| ``dust_preferential_file`` | ``source`` | static | paleolake area fraction, swapped to soil type 10 |
| ``dust_soil_types_file`` | ``type2/3/4/6``, ``type13..17`` | static | nine texture area fractions (type 1 is the residual) |
| ``dust_regions_file`` | ``regions`` | static, integer 1-8 | regional tuning index |
| ``dust_roughness_file`` | ``surfrough`` [cm] | 12 monthly records | satellite roughness, read only when ``ndurough = 0`` |

The first four are mandatory together: ``mo_ham_dust.f90`` aborts without any of
them, and running with the textures or regions missing would silently emit an
untuned, all-coarse-soil flux, so ``_attach_dust`` raises instead. The monthly
climatologies are stepped by **month start** (``WRAP_YEAR``), never interpolated
— the Fortran reads one record per call and marks the region mask
``EF_NOINTER``. The region mask is categorical and is refused at load if it is
not integral in [1, 8], which is what a linear or conservative regrid would
produce. T63 is the native HAMMOZ grid; the T106 products are nearest-neighbour
refinements of it, stamped as such in their file attributes (#802).

**Status & known limitations.** The analytic-ozone fallback is a real
low-fidelity path (loud warning); a run that logs the analytic-ozone warning is
*not* a valid radiation benchmark. AMIP-SST land extrapolation is heuristically
detected and can be rejected at load. Prescribed emission / oxidant / DMS /
dust fields are inert until the data mirror supplies them (online Gong sea
salt needs no file — see {doc}`aerosol`). The GHG clamp beyond coverage is a
simplification for slowly-evolving species.

**Code pointers.**
- ``jcm/forcing.py`` — ``ForcingData``, ``make_time_series`` (end-clamp),
  ``expand_yearly_files`` (per-product coverage padding), the GHG series helpers.
- ``jcm/forcing_assembly.py`` — ``build_forcing`` and the ``_attach_ozone`` /
  ``_attach_emissions`` / ``_attach_dms`` / ``_attach_dust`` / ``_attach_oxidants``
  chain; ``jcm/runners.py::build_forcing`` delegates to it.
- ``jcm/data/input_resolution.py`` — the typed input-resolution engine
  (``resolve_input``, ``resolve_packaged``, ``expand_yearly_files``).
- ``jcm/ozone_climatology.py`` — ``OzoneClimatology`` (grid-aware loader).
- ``jcm/config/forcing/{default,from_file,amip,era5}.yaml``.
- ``jcm/data/mirror/dust.py`` — ``build_dust_product`` (the five dust bundles).
- Data provenance and the mirror: {doc}`../design/data_mirror`.

**Validation evidence.** ``jcm/forcing_test.py`` (incl. year-expansion / start-date
cases), ``jcm/physics/forcing/echam_boundary_conditions_test.py``,
``jcm/data/input_resolution_test.py`` (per-product coverage and auto resolution).
