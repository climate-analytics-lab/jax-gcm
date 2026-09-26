# The surface-exchange coupling contract

`jcm.physics.surface.surface_exchange.SurfaceExchange` is the
package-independent struct an external surface component (ocean, sea ice,
land, wave — e.g. Veros through JAX-ESM) reads **from** the atmosphere
(jax-gcm#754), and the same field list and sign convention drive the inverse
door: **forced mode**, where externally computed fluxes replace jcm's own
surface exchange (jax-gcm#301). One contract, two directions — a coupler can
feed back exactly the numbers it read, with no per-package key names, unit
conversions or sign flips.

## The contract

Every publishing package writes the struct under
`diagnostics["surface_exchange"]` (flattened to `surface_exchange.*`
variables in the netCDF output) and declares it in `provides`;
`ComposablePhysics.require_surface_exchange()` gives a coupler
composition-time failure instead of a mid-run `KeyError`.

**Guaranteed fields** are grid-box means, filled by every publisher
*exactly from the fluxes the atmosphere column actually received that
step* — never from a parallel diagnostic recomputation that could drift
from the delivered values.

| Field | Units | Sign / definition |
|---|---|---|
| `net_heat_flux` | W m⁻² | **positive down** into the surface medium: SW_net + LW_net − SHF − LHF (SPEEDY's `hfluxn` convention) |
| `sensible_heat_flux` | W m⁻² | **positive up** (surface → atmosphere) |
| `latent_heat_flux` | W m⁻² | positive up; the package's own delivered value |
| `evaporation` | kg m⁻² s⁻¹ | positive up |
| `precipitation` | kg m⁻² s⁻¹ | positive down, ≥ 0; total (rain + snow, convective + stratiform) |
| `stress_u`, `stress_v` | N m⁻² | **positive down**: eastward/northward momentum flux into the surface (= −stress on the atmosphere); westerlies give `stress_u > 0` |
| `wind_speed` | m s⁻¹ | near-surface speed at the package's reference (`wind_reference`, below) |
| `wind_u`, `wind_v` | m s⁻¹ | eastward/northward near-surface wind at the same reference, on the unrotated model grid; `hypot(wind_u, wind_v) == wind_speed` |
| `air_density` | kg m⁻³ | moist density at the lowest model level, p/(R_d·T·(1 + vtmpc1·q)) |
| `air_potential_temperature` | K | lowest model level, T·(p₀/p)^κ |

`wind_reference` is static metadata on the struct (not an array leaf, so the
struct stays a valid `jit`/`scan` output), a key of `WIND_REFERENCES`; the
publisher's `output_attrs` stamp the same value, its description and, for
10 m, a `height` onto the netCDF wind variables.

The signs are a maintainer decision on #754: turbulent fluxes positive up,
the net heat flux positive down. The last two thermodynamic fields exist
because external bulk-flux algorithms need them (#301 discussion). The
reference coupler contracts (ESMF/NUOPC, CESM `flux_atmOcn`) follow the
same shape: a fixed named field list with declared units and signs, stated
once.

**Optional fields** default to `None`, and a `None` field is *omitted*
from the output — absence is explicit, never a zero indistinguishable from
data (the #647 lesson):

- `precip_rain`, `precip_snow` — the phase split;
- `tile_fraction` and `*_tile` counterparts of the fluxes and the wind
  (`wind_u_tile`, `wind_v_tile`, `wind_speed_tile`), with the invariant
  `sum(tile_fraction * field_tile, axis=-1) == field`.

`SurfaceExchange.validate()` checks the wind-vector and tile invariants on
concrete values, for couplers and tests.

## The near-surface wind: each package's own reference

The wind fields sit at the reference the package's own surface closure
defines, and the struct says which (maintainer decision on #911):

| `wind_reference` | Package | Definition |
|---|---|---|
| `"10m"` | ECHAM | 10 m wind from the stability-dependent surface-layer profile of each tile (`mo_surface` `nsurf_diag`): per tile `u10_t = zred_t·u_low`, box mean `u_low·Σ f_t·zred_t` |
| `"lowest_level"` | SPEEDY | `fwind0 × (u, v)` at the lowest model level (σ = 0.95 at L8; `fwind0 = 0.95` by default), the wind the SPEEDY bulk formulae use; no gustiness |

A common 10 m height was rejected: SPEEDY has no surface-layer profile, so
its 10 m wind would be invented rather than diagnosed. A consumer that needs
a specific height reads `wind_reference` and adapts. The components are the
wind itself, not a stress-derived direction: a stress-direction proxy
(`wind_speed·stress/|stress|`) is exact for SPEEDY but only approximate for
ECHAM, whose implicit solve rotates the delivered stress, and the real field
costs nothing to publish.

ECHAM-MPIOM itself exchanges per-surface-type stresses and the open-water 10 m
speed (`wind10w`), not `u10`/`v10`, which are output diagnostics; the vector
is published because a coupled component with its own drag law (e.g. JAX-ESM's
Veros bulk stress) or a sea-ice free-drift term needs it.

**Ocean surface currents are reserved, not applied.** ECHAM takes the ocean
stress and `wind10w` relative to the ocean surface current
(`mo_surface_ocean.f90`, `zudif = u − ocu`). `ForcingData.ocean_u`/`ocean_v`
exist so that a coupler's API is stable before this lands, but the TTE-TKE
vertical diffusion does not read them yet: it applies a zero current (the
stress against a surface at rest), and the published 10 m wind is the wind
over a surface at rest. Setting the fields has no effect today; the
implementation is tracked in #915.

## Why grid-mean is guaranteed and tiles are optional

Both current packages deliver **only grid-mean** fluxes to the atmosphere,
and neither has per-tile *delivered* fluxes that satisfy the tile
invariant:

- **SPEEDY** (`jcm/physics/surface/speedy_surface_flux.py`) computes land
  and sea fluxes separately but the atmosphere only ever feels the
  `fmask`-blended mean, and sea ice is handled by blending the sea-surface
  *temperature* (`tsea = (1−sice)·SST + sice·T_ice`, SPEEDY
  `sea_model.f90`), not the fluxes — so an open-water/ice tile split does
  not exist even internally.
- **ECHAM** delivers its surface fluxes through the TTE-TKE vertical
  diffusion's implicit solve, where the tiles collapse into a single Robin
  boundary-condition coefficient per variable
  (`tte_tke/vertical_diffusion.py`); the delivered fluxes are diagnosed
  from the implicit solution (the `pev_vdiff` identity: reported ==
  column-integrated tendency). The per-tile *explicit* bulk fluxes that
  `surface_physics_step` computes are reference diagnostics that do **not**
  aggregate to that delivered mean, so publishing them would violate the
  tile invariant.

A tile-resolved *mandatory* contract would therefore force both packages
to publish numbers that are either fabricated or inconsistent with what
was delivered. The struct instead guarantees the grid mean and reserves
the tile fields: a package may fill them once (and only once) its
fraction-weighted tile fluxes provably reproduce the delivered mean —
an API-stable upgrade path, no break.

The wind tiles are the case that already qualifies. ECHAM's grid-mean 10 m
wind is *defined* as the fraction-weighted sum of the per-tile 10 m winds,
so ECHAM fills `wind_u_tile`/`wind_v_tile`/`wind_speed_tile` and
`tile_fraction` (tile axis: water, sea ice, land), while its flux tiles stay
`None`. SPEEDY has no tiles and leaves all of them `None`.

Similarly the rain/snow split is optional because the Tiedtke-Nordeng port
exposes only total convective precipitation (`convection.precip_conv`);
publishing the stratiform-only split (`clouds.precip_rain/snow`) as the
contract's rain/snow would be wrong as a *total* split, and SPEEDY has no
phase split at all. Both packages therefore guarantee `precipitation`
(total) and leave the split `None`.

## Publishers

- **SPEEDY** publishes inline from `SpeedySurfaceFlux`
  (`jcm/physics/speedy/speedy_terms.py`): at that point of SPEEDY's fixed
  ordering the convective (`_convection.precnv`) and large-scale
  (`_condensation.precls`) precipitation and the radiation terms of
  `hfluxn` already exist, and the struct is filled from the very `merged`
  fluxes the bottom-level tendencies use. Normalisations: g → kg m⁻² s⁻¹
  for evaporation/precipitation, `latent = alhc·evap` (SPEEDY's J/g
  constant against g m⁻² s⁻¹), stress negated from SPEEDY's
  on-the-atmosphere `ustr/vstr`. The wind is the closure's own `(u0, v0)`.
- **ECHAM** publishes from a terminal `EchamSurfaceExchange` term
  (`jcm/physics/surface/echam/surface_exchange_publisher.py`), composed
  after the cloud microphysics because stratiform precipitation only
  exists then — everything published is the same step's. Turbulent fluxes
  come from the vdiff-delivered `"surface"` fields; the ECHAM
  `momentum_flux_u/v` are already positive-down (verified against the
  column-integrated vdiff tendency), so they pass through unnegated. The
  wind (grid mean and tiles, with the tile fractions) is the 10 m wind the
  vertical-diffusion term diagnoses (`vertical_diffusion.wind_10m*`). It is
  the ECHAM family's only 10 m wind: the AeroCom `uas`/`vas` are the same
  fields rather than a separate neutral log-profile interpolation, and the
  surface term's own diagnostics carry no 10 m wind.
- **Held-Suarez opts out**: a bulk relaxation has no surface fluxes,
  precipitation or hydrology to report, and zeros would read as a calm dry
  planet. `require_surface_exchange()` fails loudly with the opt-out named.

## Forced mode (#301): where the seam sits

Forced mode replaces the package's own **turbulent** surface fluxes
(sensible heat, evaporation, momentum) with externally prescribed ones,
grid-mean, land and sea alike; radiation and the rest of the physics stay
interactive. The prescribed values live on `ForcingData.prescribed_*`
fields (2-D maps or `TimeSeries`), in the contract's units and signs.

- **SPEEDY**: the fluxes were already an explicit bottom-layer source, so
  the prescribed values replace the bulk-formula `merged` grid means
  inside `get_surface_fluxes` itself
  (`SpeedySurfaceFlux(prescribed_fluxes=True)`, preset
  `physics=speedy-forced-flux`). Every downstream consumer — bottom-level
  tendencies, published diagnostics, upward longwave — sees prescribed and
  interactive fluxes through the same code path, and `hfluxn` is re-closed
  against the prescribed turbulent terms so the published energy budget
  stays exact. Fed its own published fluxes, the forced step reproduces
  the interactive step bit-for-bit.
- **ECHAM**: the interactive delivery is the implicit solve's surface
  Robin row, which exists to keep a *state-dependent* exchange stable. A
  prescribed flux is state-independent, so the forced configuration
  (`echam_physics(prescribed_surface_fluxes=True)`, preset
  `physics=echam-forced-flux`) runs
  `TteTkeVerticalDiffusion(couple_surface=False)` — interior-only mixing,
  insulating/free-slip bottom — and a `PrescribedSurfaceFlux` term
  (`jcm/physics/surface/prescribed_flux.py`) delivers the fluxes as an
  exact explicit bottom-layer source in the slot where the interactive
  delivery happened, preserving the same-step bookkeeping: the
  `vertical_diffusion` delivered-flux fields (which `EchamSurface`
  republishes and the Tiedtke moisture-budget closure anchors to), the
  vdiff `qv_tendency` profile (ECHAM's `pqte` at `cucall` time), and the
  running `thermo_run` view. Fed its own published fluxes, the forced
  step matches the interactive step's column-integrated water, heat and
  momentum delivery to solver round-off; the level-by-level *placement*
  differs by construction (implicit bottom-row distribution vs a pure
  bottom-layer source), which is the accepted semantic of prescribing a
  flux instead of an exchange law.

Absent `prescribed_*` fields raise a pointed error at composition/trace
time — a forced run never silently applies zero fluxes. The published
`surface_exchange` struct in forced mode echoes the prescribed values
exactly (the closed loop a coupler iterates on).

The wind fields are not prescribed: the coupler supplies stress, and the
published wind is the atmosphere's own in both modes (ECHAM's vdiff diagnoses
the 10 m wind before its surface-coupling branch; SPEEDY's `(u0, v0)` come
from the lowest level whatever the flux source).

Prescribed evaporation is republished with `latent = alhc·E`
(vaporization). A coupler whose evaporation includes sublimation over ice
accounts for the ice enthalpy on its own side of the interface; the
atmosphere's moisture and heat tendencies depend only on E and SHF, so
this bookkeeping choice does not touch the delivered column budgets.

### Doors

Prescribed fluxes require a forced-mode consumer, and the converse holds
too. A composition's consumers are the terms whose
`PhysicsTerm.consumed_forcing_fields()` declares the `prescribed_*`
fields: `SpeedySurfaceFlux(prescribed_fluxes=True)` and
`PrescribedSurfaceFlux`. The interactive schemes (`SpeedySurfaceFlux`,
the surface-coupled `TteTkeVerticalDiffusion`) compute their own fluxes and
never read them. Both directions of that contract are enforced by ONE
helper, `validate_run_forcing(physics, forcing, run_window)`, which every
run entry point applies to its concrete forcing before stepping:

| entry point | window passed |
| --- | --- |
| `Model.run_from_state_with_carry` (so `run`, `resume`, `run_from_state`) | the run's absolute start/end |
| `PrescribedStateModel.run` (`run.mode=prescribed`) | the span of the prescribed state times |
| `SingleColumnModel.run` | none (a column has no absolute date) |
| CLI runners and `jcm.configurations.load`, right after forcing assembly | the configured `[run.start_time, start + total_time]` |

(a) Prescribed fluxes with no consumer are rejected rather than silently
ignored; the error names `forcing.prescribed_surface_flux` and how to
enable forced mode. The check is by declared capability, not class name,
so a composition edited with `replace`/`remove` is judged by what its terms
actually read. (b) With a consumer composed, every term's
`validate_forcing` runs: the forced-mode terms raise if their fields are
absent or, given a window, if a date-aligned archive does not cover it
under its declared persistence. The same helper then checks every other
dated forcing input against the window; that general rule is
{doc}`forcing_time_semantics`.
The SCM CLI (`run.mode=scm`) builds no `ForcingData`, so it refuses both a
`prescribed_surface_flux` block and a forced-mode physics; drive a forced
column from Python instead.

- **CLI**: `forcing.prescribed_surface_flux` with either
  `constants: {sensible_heat_flux, evaporation, stress_u, stress_v}`
  (uniform maps — the smoke-test door) or `file:` (a netCDF on the model
  grid, static or time-resolved; its time alignment is the block's
  `align` key, below). All four fields are required together: a
  partially prescribed surface is not a defined mode.
- **Python/coupler**: set the `prescribed_*` fields on `ForcingData`
  directly (per coupling interval, the JAX-ESM pattern) and compose the
  forced physics via the factory flag / term constructor. A file is read
  with `jcm.forcing.read_prescribed_surface_fluxes(ds, lat, lon,
  align_mode=...)` — the reader the CLI door calls — whose result
  `ForcingData.copy(**...)` attaches; an in-memory time series is a
  `make_time_series(values, times, align_mode)` leaf (`times` exact
  dates, e.g. `datetime64[s]`) with the mode
  chosen explicitly.

### Time alignment of a flux archive

A time-resolved flux file is either a *climatology* (a representative
annual cycle, replayed every model year: `WRAP_YEAR`) or a *transient
archive* (fluxes of particular dates, e.g. a coupler's history file:
`BY_DATE`). The two cannot be told apart from their timestamps: twelve
monthly samples January→December of one year are exactly what a monthly
climatology looks like, and a transient archive wrongly replayed every year
silently recycles that year's fluxes. The alignment is therefore
**declared, never inferred from the samples** — the rule every
time-resolved forcing input shares (`jcm.forcing.resolve_align`, #884):

| `align` | behaviour |
| --- | --- |
| `wrap_year` | replay every model year, the record of the model's calendar month held from the 1st |
| `by_date` | piecewise-constant on the absolute timestamps |
| `by_date_interp` | linear in time between the absolute timestamps |
| `auto` (default) | resolves only a data-mirror or packaged product, from the `alignment` the mirror manifest records; no mirror product carries fluxes, so for a flux file it raises and names the knob |

Both modes validate what they are given, so a wrong declaration fails at
load or run start rather than silently mis-phasing the fluxes:

- The time axis must decode to dates (numeric axes cannot be put on the
  model clock), have no missing or duplicate stamps, and is sorted
  ascending with the samples reordered to match (`BY_DATE`'s
  `searchsorted` needs ascending time; `WRAP_YEAR`'s position 0 is
  January). A length-1 time axis is a static field. Sorting and the
  climatology checks work on the decoded calendar values themselves;
  only a date-aligned axis is placed on the model's exact Gregorian
  clock. So a climatology stamped with idealised calendar dates (CF
  `noleap` year 0, a `360_day` calendar) loads as `wrap_year` (every
  forcing reader gives a `WRAP_YEAR` leaf nominal month/day labels on a
  reference year, read from the decoded calendar fields, never the source
  year), while declaring such an axis `by_date` fails, because it has no
  place on the model's clock.
- `WRAP_YEAR` selects record `month − 1` for the model clock's Gregorian
  month, so a climatology must be exactly twelve samples, one per calendar
  month January→December (month-start or mid-month stamps, any year).
  Anything else — a July→June span, a four-weekly axis, a seasonal
  climatology — raises.
- `BY_DATE` selection clamps to the end samples outside its axis, so at run
  start a date-aligned archive must cover the whole run window, or declare
  `persist: hold` in the block (`read_prescribed_surface_fluxes(...,
  persist="hold")` from Python) to hold its end samples deliberately. This is
  the rule every dated input follows, and the covered span is defined once in
  {doc}`forcing_time_semantics`. What is specific to flux archives is the
  **CF time bounds**: the reader carries them (the variable the `time`
  coordinate's `bounds` attribute names, or `time_bnds`/`time_bounds`;
  validated to bracket each sample) to the check, which makes the span exact
  for any stamp placement. The bounds are kept per interval, so disjoint ones
  are a declared gap the run may not cross. Rejecting such files at read time
  would refuse archives with a declared outage that a run avoiding it can
  use. Without bounds the span is the end samples plus each end's own
  spacing. A run outside it raises and names the remedies: supply covering
  fluxes, declare `forcing.prescribed_surface_flux.persist=hold`, or declare
  the file a climatology.

The same `auto` rule applies to the surface boundary file (`forcing.align`)
and the ozone / emissions / oxidant files (`forcing.ozone_align`,
`forcing.emissions_align`, `forcing.oxidants_align`): the mirror manifest's
recorded kind is the only thing `auto` may consult, because it is the only
place the climatology/transient distinction is actually recorded. The
alternative of stamping the CF `climatology` attribute on files and keying
`auto` on it was rejected: the attribute is essentially absent from real
forcing data, so it would still leave every user file to a guess. What a
dated input does outside its time axis is declared the same way, per input
(`forcing.persist`, `forcing.ozone_persist`, ...; see
{doc}`forcing_time_semantics`).

## What this replaces

JAX-ESM's SPEEDY-only reach into private keys (`_surface_flux.hfluxn`
negated by hand, freshwater from three structs with a hard-coded g → kg
conversion) is superseded by reading the published struct; that block is
tagged `TODO(jax-gcm#754)` in the JAX-ESM source, and its read of SPEEDY's
private `_surface_flux.u0/v0` for the wind vector (which left ECHAM without
one) is superseded by `wind_u`/`wind_v`, published by every package at the
reference the table above defines.
