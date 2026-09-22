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
| `wind_speed` | m s⁻¹ | near-surface speed at the package's reference (SPEEDY: fwind0-scaled σ = 0.99 wind; ECHAM: 10 m) |
| `air_density` | kg m⁻³ | moist density at the lowest model level, p/(R_d·T·(1 + vtmpc1·q)) |
| `air_potential_temperature` | K | lowest model level, T·(p₀/p)^κ |

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
- `tile_fraction` and `*_tile` counterparts of the fluxes, with the
  invariant `sum(tile_fraction * field_tile, axis=-1) == field`.

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
  on-the-atmosphere `ustr/vstr`.
- **ECHAM** publishes from a terminal `EchamSurfaceExchange` term
  (`jcm/physics/surface/echam/surface_exchange_publisher.py`), composed
  after the cloud microphysics because stratiform precipitation only
  exists then — everything published is the same step's. Turbulent fluxes
  come from the vdiff-delivered `"surface"` fields; the ECHAM
  `momentum_flux_u/v` are already positive-down (verified against the
  column-integrated vdiff tendency), so they pass through unnegated.
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

Prescribed evaporation is republished with `latent = alhc·E`
(vaporization). A coupler whose evaporation includes sublimation over ice
accounts for the ice enthalpy on its own side of the interface; the
atmosphere's moisture and heat tendencies depend only on E and SHF, so
this bookkeeping choice does not touch the delivered column budgets.

### Doors

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
  `make_time_series(values, time_seconds, align_mode)` leaf with the mode
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
| `wrap_year` | replay by month position every model year |
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
  only a date-aligned axis is converted to the model's Gregorian epoch
  clock. So a climatology stamped with idealised calendar dates (CF
  `noleap` year 0, a `360_day` calendar) loads as `wrap_year` (every
  forcing reader gives a `WRAP_YEAR` leaf informational time bins rather
  than decoded dates), while declaring such an axis `by_date` fails,
  because it has no place on the model's clock.
- `WRAP_YEAR` selects sample `floor(fraction_of_year × 12)`, a
  January-anchored month position, so a climatology must be exactly twelve
  samples, one per calendar month January→December (month-start or
  mid-month stamps, any year). Anything else — a July→June span, a
  four-weekly axis, a seasonal climatology — raises.
- `BY_DATE` selection clamps to the end samples outside its axis, so at run
  start (`validate_forcing` of the forced-mode terms, which `Model`
  calls with the run window) a date-aligned archive must cover the whole
  run, with one sample interval of slack at each end (its largest sample
  spacing: a sample may be stamped at the start or the middle of the
  interval it represents, so a Jan-1…Dec-1 or a Jan-15…Dec-15 monthly
  archive both cover their calendar year). A run outside it raises and
  names both remedies — supply covering fluxes, or declare the file a
  climatology. The check is skipped only when the window or the series is
  traced (a run inside a JAX transformation).

The same `auto` rule applies to the surface boundary file (`forcing.align`)
and the ozone / emissions / oxidant files (`forcing.ozone_align`,
`forcing.emissions_align`, `forcing.oxidants_align`): the mirror manifest's
recorded kind is the only thing `auto` may consult, because it is the only
place the climatology/transient distinction is actually recorded. The
alternative of stamping the CF `climatology` attribute on files and keying
`auto` on it was rejected: the attribute is essentially absent from real
forcing data, so it would still leave every user file to a guess.

## What this replaces

JAX-ESM's SPEEDY-only reach into private keys (`_surface_flux.hfluxn`
negated by hand, freshwater from three structs with a hard-coded g → kg
conversion) is superseded by reading the published struct; that block is
tagged `TODO(jax-gcm#754)` in the JAX-ESM source. #482 (near-surface wind /
boundary-layer coupling) concerns the `wind_speed` field's reference
height; the contract deliberately states the per-package reference rather
than promising a common height until that lands.
