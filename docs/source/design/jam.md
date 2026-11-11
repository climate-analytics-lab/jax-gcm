# The Jax Aerosol Module (JAM)

The `jcm.physics.aerosol.jam` package is a HAMMOZ-style, microphysics-agnostic
**process harness** for online aerosol, wrapping an interchangeable
**microphysics core**. It provides emissions, an aerosol microphysics core,
ARG cloud-droplet activation, gravitational sedimentation, and dry/wet
deposition, all as composable `PhysicsTerm`s. "HAMMOZ-style" is used here as a
familiar point of comparison — JAM is *inspired by* HAM and currently shares
several of its parameterizations, but the package name is deliberately
decoupled from any specific scheme so the implementations can be swapped
freely (like the other JCM physics packages). See issue #461.

Two aerosol populations ship on the harness, each with its own microphysics
core and physics preset:

- **M7** (`physics=echam-jam-m7`, `+configuration=t63-echam-jam`, the
  default JAM configuration) —
  ECHAM6.3-HAM2.3's seven-mode population and the native M7 microphysics
  (the `m7-jax` package, `jcm[m7]`), with HAM's activation, ion-mediated
  nucleation, per-sector emissions, dust and wet deposition. See
  {doc}`jam_m7_configuration`.
- **MAM4** (`physics=echam-jam`, `+configuration=t63-echam-jam-mam4`) — the
  E3SM/CAM6 four-mode population and the MAM4-JAX core (`jcm[mam4]`) inside
  the same HAMMOZ-lineage process harness.

Both are release-validated at T63 L47 and share every harness term except the
population-specific variants each preset selects.

## Design

JAM mirrors HAMMOZ's split between a microphysics-agnostic process layer and a
swappable microphysics core, expressed in JCM's composable architecture:

- **Population contract** (`population.py`): `ModalAerosolSpec` /
  `AerosolMode` / `AerosolSpecies` are pure-Python frozen dataclasses
  describing the *shape* of the population (classes, per-class species, σ_g,
  density, κ). `ModalAerosolSpec` is the modal realisation (each class a
  log-normal mode); a sectional realisation (each class a size bin) would be a
  sibling spec with the same role (#491). They are static config — never JAX
  pytree leaves — so class counts are known at compose time and no
  data-dependent shapes arise. Field names echo the AMBRS modal vocabulary so
  a future part2pop/AMBRS interop adapter is a thin mapping rather than a
  rewrite.
- **Microphysics core** (`microphysics/`): the single swap point, each core
  paired with a population and selected by `jam_microphysics` (its switches by
  `jam_microphysics_options`):
  - `m7_jax` on **M7** (`M7_SPEC`, `m7_data.py`: HAM's modes NS, KS, AS, CS
    soluble and KI, AI, CI insoluble; no cloud-borne phase) — the `m7-jax`
    package (`jcm[m7]`).
  - `mam4_jax` on **MAM4** (`MAM4_SPEC`, `mam4_data.py`, constants from E3SM
    `rad_constituents.F90` / `modal_aero_data.F90`) — the MAM4-JAX package
    (`jcm[mam4]`, #490); `mam4_jax_astem` is its forward-only adaptive-ASTEM
    variant.
  - `placeholder` / `m7_placeholder` — `PlaceholderMicrophysics`, κ-Köhler
    equilibrium radii/density with **zero tendency** on the MAM4 / M7
    population: the bare `echam_physics()` default and test vehicle, keeping
    the Apache-2.0 harness importable with no optional dependency.
- **Tracers** (`tracer_layout.py`): interstitial aerosol mass and number are
  ordinary `state.tracers` entries (`m_`/`n_`, keyed by aerosol class), so the
  dynamical core transports them and existing diagnostics work. The
  cloud-borne phase (`mc_`/`nc_`) lives in the cross-step physics carry, not
  the tracer set (`cloud_borne_store.py`, #602): MAM4 prognoses it, while M7
  has none (`jam_cloud_borne: false`) and scavenges interstitial aerosol by
  its activated fraction.
- **Inter-term state** (`jam_state.py`): the core writes a typed
  `JamAerosolState` under the `_jam_state` diagnostic; activation, deposition
  and sedimentation read it. Its leading `n_aer` axis is a generic aerosol-
  *class* index (mode or bin), so the struct is representation-agnostic.

Every harness step is differentiable; scheme choice (core, activation
scheme, ARG variant) is a compose-time Python decision with no traced branching.

## Term chain

`jam_aerosol_physics()` returns the ordered list
(`jcm/physics/aerosol/jam/jam_terms.py`):

1. `AerosolCarrySeeder` (owns the shared `aerosol` slot radiation reads), then
   `CloudBorneCarryStore` when the population carries a cloud-borne phase.
2. Emissions — natural `SeaSaltEmissions` (Gong 2003 for MAM4, Long et al.
   2011 for M7), `DmsEmissions` (Nightingale 2000), `DustEmissions` (Tegen et
   al. 2002 / HAM's `ndust` presets), and the prescribed
   `AnthropogenicEmissions` (#498), `PreSpeciatedEmissions` and
   `VolcanicEmissions` (#1190). Emission is computed in gridpoint (nodal)
   space; the split of a source's mass into aerosol classes belongs to the
   population, so the harness works with any dycore.
3. Physics-side transport — `TracerVerticalDiffusion` and
   `ConvectiveTracerTransport` (with in-plume scavenging).
4. Chemistry — `PrescribedOxidants` and `SulfurGasChemistry` (producing the
   H₂SO₄ / SOAG the core condenses the same step).
5. The microphysics core (`m7_jax`, `mam4_jax`, or a placeholder) — writes
   `_jam_state`.
6. Optics — `JamOpticsTerm` (jcm's Mie kernel) or `HamLutOpticsTerm` (HAM's
   tables, `jam_optics_backend: ham_lut`).
7. Activation — `ArgActivation` (Abdul-Razzak & Ghan 2000; the MAM4 presets
   pin the `ghosh2025` variant, Ghosh et al. 2025, GMD 18 4899, Eq. 5) or
   HAM's Köhler A/B + ARG over the West et al. updraft PDF (`ham_arg`,
   M7). Writes `activated_cdnc`, the same key the 2M SPA floor produces
   (#342). See {doc}`../science/aerosol`.
8. `IceNucleation` — aerosol inputs to mixed-phase and cirrus freezing.
9. `StokesSedimentation` — Stokes settling + donor-cell vertical transport.
10. `CAMDryDeposition` — aerodynamic resistance plus CAM's eleven-class
    surface collection, including dry-surface rebound. The former
    `SlinnDryDeposition` ocean-only term remains available explicitly.
11. `CloudBorneExchange` (populations with a cloud-borne phase only),
    `AqueousSulfur`, then `WetScavenging` — in-cloud nucleation and impaction
    plus below-cloud scavenging, from the cloud scheme's per-level
    precipitation formation/evaporation rates, with re-evaporation
    re-injection (#499).

In `echam_physics` the chain is *split*: `CloudBorneExchange`, `AqueousSulfur`
and `WetScavenging` run **after** the cloud microphysics term, against the
current step's clouds and precipitation; everything else runs in the
pre-cloud aerosol block (activation must precede the cloud microphysics term
that consumes `activated_cdnc`).

## Usage

```python
from jcm.physics.echam.echam_terms import echam_physics

physics = echam_physics(
    aerosol_module="jam",      # default "macv2sp"
    cloud_scheme="2m",         # ARG activated_cdnc feeds the 2M scheme
    jam_microphysics="mam4_jax",   # "placeholder" is the bare default
    jam_arg_variant="ghosh2025",   # or "arg2000"
)
```

The M7 preset sets many HAM switches at once (`jam_microphysics="m7_jax"`,
`jam_cloud_borne=False`, `jam_activation_scheme="ham_arg"`, …); compose it with
`physics=echam-jam-m7` rather than by hand.

The JAM path **replaces** MACv2-SP; the two are mutually exclusive aerosol
sources (#640). MACv2-SP was a stopgap providing the shared `aerosol` optics /
Twomey diagnostic before JAM had its own coupling — having both on was
confusing and redundant. JAM now supplies both, so `echam_physics(
aerosol_module="jam")` no longer composes `Macv2SpAerosol`:

- **Optics.** A minimal `AerosolCarrySeeder` runs first and owns the shared
  `aerosol` slot (which radiation hard-requires and the 2M microphysics reads),
  resetting it to an all-zero base each step. `JamOpticsTerm` overwrites the
  per-band SW/LW optics (consumed by RRTMGP) and — for the grey two-stream
  scheme, which reads a single broadband 550 nm profile — the
  `aod_profile`/`ssa_profile`/`asy_profile`/`angstrom` fields. With
  `jam_optics=False` the zero base is left untouched, making the aerosol
  **radiatively passive** — a clean A/B control for the direct effect.
- **Activation.** The 2M scheme uses ARG's `activated_cdnc` where it is
  non-empty and, in the JAM path, falls back where ARG is empty to its own
  ECHAM-HAM minimum-CDNC (`cdnc_min_fixed`, 10 cm⁻³ on the JAM hosts, or the dynamic max-radius
  floor; #674) rather than the MACv2-SP SPA floor (which no longer exists in
  this path — JAM carries no prescribed-plume `Nccn`). The SPA floor remains
  the `macv2sp`+2M path's Twomey link.

**Output namespaces (#640).** MACv2-SP's diagnostics publish under `macsp.*`
(CF/AeroCom names where they exist, e.g. `macsp.od550aer`); JAM's column optics
under `jam_optics.*` (`jam_optics.aod_550` is the band-centre-approx column
AOD — distinct from the Mie-based `od550aer` of the `aerocom_optics` pass). The
internal `aerosol` struct radiation/microphysics read by attribute is
unchanged; only the output keys are namespaced. The old top-level
`aerosol_optical_depth` key (which collided with a per-band `RadiationInput`
field) is gone.

## Prescribed anthropogenic & biomass emissions (#498)

Beyond the online natural sources, JAM can read **prescribed** SO₂/BC/OC
emissions (`AnthropogenicEmissions`, opt-in via
`echam_physics(aerosol_module="jam", jam_anthropogenic=True)`). These cover both
CEDS anthropogenic activity and open biomass burning, organised into four
**super-sectors** distinguished by *injection type and source size* (HAMMOZ's
basis), not economic activity:

| Super-sector | Injection (default) | Source |
|---|---|---|
| `surface_combustion` | surface (~0 m, σ 30 m) | CEDS TRA/RCO/AGR/WST/SLV |
| `elevated_industrial` | ~50 m | CEDS ENE/IND |
| `shipping` | marine surface | CEDS SHP |
| `biomass_burning` | deep FIRE (~1 km, σ 1.5 km) | open burning (GFED/BB4CMIP7) |

Each super-sector's bulk flux is speciated following HAMMOZ — SO₂ → a primary-SO₄
fraction (default 2.5 %, into Aitken+accum sulfate) plus the `g_so2` gas
remainder; BC/OC → the MAM4 primary-carbon mode (OC×1.4 = POA); under M7 the
split instead follows HAM's per-sector sizes and modes (`emissions/ham_sectors.py`,
`M7_SPEC.sector_emission`), with residential and energy as their own channels
({doc}`jam_m7_configuration`) — and distributed
over a **smooth Gaussian vertical profile** (`injection.py`) so the injection
height is differentiable (a hard level pick has no gradient). The injection
height/thickness and primary-SO₄ fraction are per-super-sector differentiable
`EmissionParameters`, defaulting to the HAMMOZ values, so they can be calibrated
by gradient through the model.

### Emissions-file contract

The model is driven by a user-supplied file on (or already interpolated to) the
model horizontal grid, carrying **bulk per-super-sector surface mass fluxes** —
the model does the speciation and injection. Requirements:

- **Variables:** `emis_<super_sector>_<species>` for the super-sectors above and
  `species ∈ {so2, bc, oc}`. Any missing variable is treated as zero, so a file
  need only carry the channels it has.
- **Units:** kg m⁻² s⁻¹ surface flux. `so2` as SO₂ mass (not S); `bc`/`oc` as
  carbon mass — **OC, not OM** (the OM:OC = 1.4 is applied in-model). The
  primary-SO₄ fraction is *not* pre-applied — supply the full SO₂.
- **Dims/time:** `(lon, lat, time)`; `time` may be a 12-month climatology
  (wrap-year) or a multi-year monthly axis (by-date), matching the other forcing
  fields.

Load it onto `ForcingData` via:

```python
import xarray as xr
from jcm.forcing import read_anthropogenic_emissions
emis = read_anthropogenic_emissions(xr.open_dataset(emissions_file))
forcing = forcing.copy(anthropogenic_emissions=emis)
```

### Preparing a file from a source product

`jcm.data.emissions` regrids an arbitrary source onto the model grid and writes
contract variables. The regridder (`regrid.py`) is light and **first-order
conservative**: a regular lat/lon source gets the exact-overlap operator (CDO
`remapcon`'s scheme, exact at any resolution ratio), an unstructured `ncol`
source (e.g. CESM ne30) area-weighted nearest-cell binning, with target cells
no source centre reaches (a target finer than the source) taking the nearest
source value rather than a zero-flux hole. `prepare.py` maps
source variables → contract variables via `Channel` records; shipped adapters
`cesm_cmip_anthro(dir)` and `cesm_bb4cmip7(dir)` consume the CESM CMIP7 CEDS /
biomass-burning files. `downloader.fetch` resolves a local path or arbitrary URL
(host-agnostic — no ESGF coupling).

```python
from jcm.data.emissions import prepare_emissions, cesm_cmip_anthro
ds = prepare_emissions(cesm_cmip_anthro(source_dir), coords, time_index=month)
```

### From the CLI

The `echam-jam` and `echam-jam-m7` presets enable the bulk, pre-speciated and
volcanic emission terms. On supported grids `forcing.emissions_file: auto`
(the `forcing=from_file` default) loads the period-matched mirror bundle; pass
`null` for no prescribed sources, or point it at a model-grid file — it
auto-routes by content (`emis_*` bulk vs `aero_emis_*` pre-speciated), and a wrong-grid file
raises rather than silently zeroing:

```
python -m jcm.main physics=echam-jam grid=echam_t42_l8_sigma \
    physics.land_model=prescribed \
    forcing.emissions_file=/path/to/emissions_on_model_grid.nc
```

`echam-jam` is *factory-built* (`physics.builder: echam_physics`) rather than a
flat term list, because the JAM chain's ordering (split around the cloud term) is
encoded by `echam_physics()` — `build_physics` delegates to it.

### Natural-emission and oxidant climatology hooks

Further forcing-file hooks feed the natural-emission and sulfur-chemistry
terms, which are otherwise inert (DMS/dust fall back to zero; the oxidants fall
back to the analytic interim proxies). All accept the raw HAMMOZ/ECHAM-layout
files (`(time[, mlev], lat, lon)`, *descending* latitude — validated against the
model grid and flipped to model order; a mismatched grid raises):

- `forcing.dms_file` — seawater DMS monthly climatology (`DMS_sea`, nmol/L,
  e.g. `emiss_fields_dms_sea_monthly_T63.nc`). Converted to kg-DMS/m³ at load
  so `DmsEmissions`' `piston_velocity · dms_seawater` product is directly a
  kg/m²/s flux; `_FillValue` land cells → 0.
- `forcing.dust_file` plus `dust_preferential_file`, `dust_soil_types_file`,
  `dust_regions_file` and the optional `dust_roughness_file` — the five HAMMOZ
  inputs the Tegen scheme reads (`dust_potential_sources_T63.nc`,
  `dust_preferential_sources_T63.nc`, `soil_type_all_T63.nc`,
  `dust_regions_T63.nc`, `surface_rough_12m_T63.nc`). The first four are
  mandatory together; see `docs/source/science/boundary_conditions.md`.
- `forcing.volcanic_file` — the grid-free volcanic SO₂ product
  (`products/volcanic_so2_cmip.nc`: CMIP6 degassing + CMIP7 eruptions as
  point sources), mapped onto the columns at load for `VolcanicEmissions`;
  `forcing.volcanic_eruption_year` replays one year's eruptions. See
  {doc}`jam_volcanic_so2`.
- `forcing.oxidants_file` — monthly `OH/NO3/O3/H2O2_VMR_avrg` mole fractions on
  ECHAM hybrid model levels (e.g. `ham_oxidants_monthly_T63L47_macc.nc` with
  `grid=echam_t63_l47_hybrid`). Levels are mapped one-to-one onto the model
  levels (level count asserted; `hyam`/`hybm` cross-checked against the model's
  hybrid coefficients). The forcing carries **VMR** (`forcing.oxidant_vmr`);
  `PrescribedOxidants` converts to molec cm⁻³ in-term, where the instantaneous
  T and p live.

The monthly ones load as wrap-year `TimeSeries` leaves, so `select(date)`
slices them per step like every other forcing field.

The mirror's emissions bundles and their sources are described in
{doc}`data_mirror`.

## Natural emissions: the 10 m wind (#723)

### The emission wind is 10 m, not the lowest model level

Gong sea salt (`u10**3.41`) and Nightingale DMS (`k_w ~ u10**2`) are fitted to
the 10 m wind; HAMMOZ passes `vphysc%velo10m`. Reading the lowest full level
instead — ~33 m at L47 — inflates sea salt by 37-46 % and DMS by 20-25 %.

`TteTkeVerticalDiffusion` now publishes `VerticalDiffusionData.wind_10m`, the
ECHAM/ICON `nsurf_diag` reduction of the lowest-level wind along the same
surface-layer profile that produced the drag:

```
bn  = ln(z1/z0m)                              neutral profile factor
bm  = bn * sqrt(CM_n|U| / CM|U|)              stability-corrected
red = [ln(1 + (e^bn - 1)*10/z1) + merge] / bm
merge = -(bn - bm)*10/z1                      stable   (CM|U| < CM_n|U|)
      = -ln(1 + (e^(bn-bm) - 1)*10/z1)        unstable
```

Building it from the per-tile `CM·|U|` the surface stress already uses means the
10 m wind cannot drift from the momentum exchange, and the stable/unstable
branch needs no separate Richardson number (the two branches meet continuously
at `Ri = 0`, where `CM|U| = CM_n|U|`). The profile factor is built from the
`zepdu2`-floored speed the coefficients themselves used (`max(|U|, 1 m/s)`),
or ECHAM's calm-wind floor would be misread as a stability signal; the
reduction it yields then multiplies the true wind.

Emission terms read the result through `emissions/surface_wind.py`. They run
*before* vdiff in the ECHAM ordering, so the value is one step old. This is
the **declared** cross-step carry — `vertical_diffusion` is one of the slots
`Physics.initial_carry_state` seeds — i.e. the deliberate pattern #673
distinguishes from an accidental stale `.get()`, and the same lag the dust
term's `u*` already carries. When #673's `carry_slots` check lands this read
is one to declare explicitly.

**The fallback to the lowest model level is bootstrap-only, by construction.**
The 10 m wind *cannot* exist on step 1 of a cold start: emissions run first,
and the `vertical_diffusion` carry slot is seeded zero-filled
(`initial_carry_state`; verified — `surface_friction_velocity` and `wind_10m`
are both exactly 0 there), so no surface layer has been diagnosed yet.
(A resumed run carries a real 10 m wind and never takes the fallback; nor does
a composition with no vdiff term, where there is no surface layer at all.)
Because taking it silently would mean emitting 37-46 % too much sea salt, the
emission terms publish a per-column flag `wind_10m_model_level` — 1 where the
model level was used, 0 where the diagnosed wind was — zeroed every step with
the other emission diagnostics. It is 1 on step 1 of a cold start and 0
thereafter; `seasalt_test.py` (`test_fallback_is_taken_on_step_one_and_never_again`
and its companions) asserts exactly that per step, against the zero-filled carry
a cold start begins with and a populated one thereafter. The JAM integration
test adds the end-to-end check that no column is still flagged after six steps.

**The three terms behave differently on step 1 of a cold start, deliberately.**
Dust emits nothing (its carried `u*` is zero, and a saltation threshold has no
defensible value without a surface layer); sea salt and DMS emit from the
lowest model level and flag it
(`|U|` at ~33 m is a defensible approximation of `u10` — it is what the model
did on every step before #723); and `EchamSurface` has no step-1 case, because
it runs after vdiff. Unifying them would make one of the three worse: dust
would have to invent a wind, or sea salt would have to discard a step of a
legitimate flux for symmetry's sake. The asymmetry follows from what each
quantity is, and is stated here rather than in three docstrings.

`check_health` **reports** the chunk's fraction — averaged over every save
interval in the chunk, so the bootstrap step cannot hide in an interval the
report does not look at — and fails the chunk only on a *persistent* fallback.
The distinction matters. Under `output_averages` the saved field is an interval
*mean*, so a cold start's one legitimate fallback step reads `1/N`, the
identical value a single defective step mid-chunk would give: no rule can
separate those, and treating `> 0` as fatal aborted a healthy 30-day validation
run at day 5 on `1/480`, losing the chunk because the bail path skips the
checkpoint. A fallback that never stops is separable, though — no vdiff term,
or `wind_10m` never published, reads `1.0` in *every* chunk and costs +37-46 %
sea salt for the whole run. So the gate is `chunk_idx > 0 and frac > 0.5`:
after the first chunk the flag can only be zero, and the one legitimate way to
read `1.0` (a first chunk that is itself a single step) can only be chunk 0.
That needs no timestep and no cold-start flag — which is what the earlier
one-step exemption needed, and crashed on for configs whose dycore owns the
timestep. A single-chunk run degrades to report-only, and the per-step
invariant stays where it can be checked exactly, in the unit tests.

### Two emission paths: differentiable bulk vs CAM6-faithful pre-speciated

The above is the **bulk / differentiable** path (`jam_anthropogenic=True`). There
is a second, complementary path — `PreSpeciatedEmissions`
(`jam_prescribed_speciated=True`) — that mirrors how **CAM6 actually applies
emissions**: it reads **already-speciated per-tracer** fields and injects them
directly, with no in-model speciation or injection parameters (CAM bakes the
mode/sector split and vertical placement into the files offline;
`mo_srf_emissions` for surface fields, `mo_extfrc` for altitude-resolved ones).

| | bulk (`AnthropogenicEmissions`) | pre-speciated (`PreSpeciatedEmissions`) |
|---|---|---|
| Forcing | `emis_<sector>_<species>` bulk SO₂/BC/OC | `aero_emis_<tracer>` (`m_so4_acc`, `n_pcm`, `g_so2`, …) |
| Speciation | **in-model**, differentiable (SO₄ frac, modes, OM:OC) | pre-baked in the file |
| Injection | smooth Gaussian, differentiable height | surface (bottom layer) or 3-D volume per level |
| Use | calibration of injection/speciation params | bit-faithful reproduction of CESM emissions |

Both are independent flags (enable either, both, or neither) and both remain
differentiable **w.r.t. the emission `ForcingData` fields themselves** — so even
the pre-speciated path supports `∂(aerosol mmr)/∂(emission)` gradients, just not
w.r.t. an injection-height knob it doesn't have.

`prepare.cesm_mam4_speciated(dir)` + `prepare_speciated_emissions(...)` build the
pre-speciated file from the CESM MAM4 files (a1→accum, a2→Aitken, a4→primary
carbon; `SO2`→gas; `num_*`→number; the energy-sector `*_ene_vertical` 3-D
`mo_extfrc` field column-integrated — ≤ ~400 m sits within the lowest model
layer(s) at GCM resolution). This reproduces CESM's global budget, including the
**2.5 % primary-sulfate split recovered to 3 decimals** — the validation
counterpart to the differentiable path.

## Status and caveats

- **Natural source scales** (sea salt, dust) are calibrated per population
  against the host's surface winds (the v3.1 refit, {doc}`jam_aerosol_retune`),
  not inventory-derived.
- **Differentiability**: the fixed-substep `mam4_jax` core is differentiable;
  `mam4_jax_astem` and the float32 core paths (MAM4 and M7) are forward-only.

Out of scope, tracked separately: sectional/bulk families (#491), part2pop
diagnostics adapter (#492), a full SOA volatility basis (#493; CAM6's SOAG
is implemented, {doc}`jam_soa_dust`).
