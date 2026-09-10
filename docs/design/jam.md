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
- **Microphysics core** (`microphysics/`): the single swap point. The default
  `PlaceholderMicrophysics` computes κ-Köhler equilibrium radii/density with
  **zero tendency** on the real MAM4 4-mode population (`mam4_data.py`,
  constants from E3SM `rad_constituents.F90` / `modal_aero_data.F90`,
  cross-checked against MAM4-JAX). Replacing it with the real MAM4-JAX core
  swaps only the per-step microphysics, not the contract (#490).
- **Tracers** (`tracer_layout.py`): aerosol mass and number are ordinary
  `state.tracers` entries (`m_`/`mc_`/`n_`/`nc_` for interstitial/cloud-borne
  mass/number, keyed by aerosol class), so the dynamical core transports them
  and existing diagnostics work. The cloud-borne mirror is carried for cores
  that prognose it (e.g. MAM4; currently inert).
- **Inter-term state** (`jam_state.py`): the core writes a typed
  `JamAerosolState` under the `_jam_state` diagnostic; activation, deposition
  and sedimentation read it. Its leading `n_aer` axis is a generic aerosol-
  *class* index (mode or bin), so the struct is representation-agnostic.

Every harness step is differentiable; scheme choice (placeholder vs real core,
ARG variant) is a compose-time Python decision with no traced branching.

## Term chain

`jam_aerosol_physics()` returns the ordered list:

1. Natural-emission scheme terms — `SeaSaltEmissions` (Gong 2003),
   `DmsEmissions` (Nightingale 2000), `DustEmissions` (Tegen et al. 2002) —
   each a faithful port with calibratable `Parameters`, producing lowest-layer
   tracer tendencies. Emission is computed in gridpoint (nodal) space; the
   split of a source's mass into aerosol classes belongs to the microphysics
   core's population, so the harness works with any dycore (no modal
   representation is assumed on the dynamics side). DMS reads a prescribed
   seawater field and dust a prescribed source field from `ForcingData`
   (zero fallback). Prescribed anthropogenic + biomass emissions
   (`AnthropogenicEmissions`, #498) read bulk per-super-sector fluxes from
   `ForcingData` — see "Prescribed anthropogenic & biomass emissions" below.
2. microphysics core (`PlaceholderMicrophysics`) — writes `_jam_state`.
3. `ArgActivation` — Abdul-Razzak & Ghan (2000); writes `activated_cdnc`
   (the same key the 2M SPA floor produces, so ARG and SPA are
   interchangeable, #342). Optional `ghosh2025` variant (Ghosh et al. 2025,
   GMD 18 4899; coefficients reconstructed from the paper's Table 3, gated
   off by default pending PDF verification).
4. `StokesSedimentation` — Stokes settling + donor-cell vertical transport.
5. `SlinnDryDeposition` — aerodynamic + Slinn & Slinn (1980) over-water
   resistances; reads `surface_friction_velocity` from the
   `vertical_diffusion` diagnostic (previous step).
6. `WetScavenging` — in-cloud nucleation + size-dependent below-cloud
   impaction scavenging, built from the cloud scheme's precip / condensate
   diagnostics.

In `echam_physics` the chain is *split*: terms 1–5 run in the pre-cloud
aerosol block (activation must precede the cloud microphysics term that
consumes `activated_cdnc`), while `WetScavenging` is placed immediately
**after** the cloud microphysics term so it scavenges against the current
step's precipitation and condensate rather than the previous step's.

## Usage

```python
from jcm.physics.echam.echam_terms import echam_physics

physics = echam_physics(
    aerosol_module="jam",      # default "macv2sp"
    cloud_scheme="2m",         # ARG activated_cdnc feeds the 2M scheme
    jam_microphysics="placeholder",
    jam_arg_variant="arg2000", # or "ghosh2025"
)
```

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
  ECHAM-HAM minimum-CDNC (`cdnc_min_fixed`, 40 cm⁻³, or the dynamic max-radius
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
remainder; BC/OC → the MAM4 primary-carbon mode (OC×1.4 = POA) — and distributed
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
conservative** (area-weighted nearest-cell binning), handling both regular
lat/lon and unstructured `ncol` sources (e.g. CESM ne30). `prepare.py` maps
source variables → contract variables via `Channel` records; shipped adapters
`cesm_cmip_anthro(dir)` and `cesm_bb4cmip7(dir)` consume the CESM CMIP7 CEDS /
biomass-burning files. `downloader.fetch` resolves a local path or arbitrary URL
(host-agnostic — no ESGF coupling).

```python
from jcm.data.emissions import prepare_emissions, cesm_cmip_anthro
ds = prepare_emissions(cesm_cmip_anthro(source_dir), coords, time_index=month)
```

### From the CLI

The `echam-jam` physics preset enables JAM with both emission terms (inert until
fed). Point `forcing.emissions_file` at a model-grid file — it auto-routes by
content (`emis_*` bulk vs `aero_emis_*` pre-speciated), and a wrong-grid file
raises rather than silently zeroing:

```
python -m jcm.main physics=echam-jam grid=echam_t42_l8_sigma \
    forcing.emissions_file=/path/to/emissions_on_model_grid.nc
```

`echam-jam` is *factory-built* (`physics.builder: echam_physics`) rather than a
flat term list, because the JAM chain's ordering (split around the cloud term) is
encoded by `echam_physics()` — `build_physics` delegates to it.

### Natural-emission and oxidant climatology hooks

Three further forcing-file hooks feed the natural-emission and sulfur-chemistry
terms, which are otherwise inert (DMS/dust fall back to zero; the oxidants fall
back to the analytic interim proxies). All accept the raw HAMMOZ/ECHAM-layout
files (`(time[, mlev], lat, lon)`, *descending* latitude — validated against the
model grid and flipped to model order; a mismatched grid raises):

- `forcing.dms_file` — seawater DMS monthly climatology (`DMS_sea`, nmol/L,
  e.g. `emiss_fields_dms_sea_monthly_T63.nc`). Converted to kg-DMS/m³ at load
  so `DmsEmissions`' `piston_velocity · dms_seawater` product is directly a
  kg/m²/s flux; `_FillValue` land cells → 0.
- `forcing.dust_file` — dust source map (`pot_source`). Two conventions are
  supported, selected by `physics.jam_dust_source`; only the lower bound is
  imposed at load (the file's `-1` missing marker and NaN → 0). See
  "Dust source gating" below.
- `forcing.oxidants_file` — monthly `OH/NO3/O3/H2O2_VMR_avrg` mole fractions on
  ECHAM hybrid model levels (e.g. `ham_oxidants_monthly_T63L47_macc.nc` with
  `grid=echam_t63_l47_hybrid`). Levels are mapped one-to-one onto the model
  levels (level count asserted; `hyam`/`hybm` cross-checked against the model's
  hybrid coefficients). The forcing carries **VMR** (`forcing.oxidant_vmr`);
  `PrescribedOxidants` converts to molec cm⁻³ in-term, where the instantaneous
  T and p live.

All three load as monthly wrap-year `TimeSeries` leaves, so `select(date)`
slices them per step like every other forcing field.

See `.claude/aerosol_emissions_plan.md` for the full design, the data-source
investigation (the raw 0.5° gridded CEDS is ESGF-only; a self-hosted compressed
mirror is a tracked follow-up), and the CESM adapter's documented approximations.

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

## Natural emissions: source gating, emitted size, and the 10 m wind

Three corrections to the natural-emission terms (#768, #723), all grounded in
CAM/CLM's Zender-2003 dust chain and ECHAM's `vdiff` surface-layer diagnostic.

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
thereafter, and the JAM integration test asserts exactly that per step.

`check_health` **reports** the chunk's fraction rather than failing on it, and
the distinction is the point. Under `output_averages` the saved field is an
interval *mean*, so a cold start's one legitimate fallback step reads `1/N` —
the identical value a single defective step mid-chunk would give. No per-chunk
rule can separate them, and making the fraction fatal aborted a healthy 30-day
validation run at day 5 on `1/480`, losing the chunk because the bail path
skips the checkpoint. A wrong-but-finite emission wind is a bias, not a
blowup, so it belongs in the report the chunk prints (`Emis wind: ...`), where
a persistent fallback shows as ~100 % every chunk, and the per-step invariant
stays where it can be checked exactly — in the unit tests.

### Dust source gating

`forcing.dust_source` is a *prescribed* erodibility; the physics it needs
around it is CLM's, because CAM's `dust_flux_in` arrives from CLM already
masked. The term now applies, per column:

| factor | reference | field used |
| --- | --- | --- |
| erodibility, zeroed below 0.1, **not** bounded above | CAM `dust_model.F90` `soil_erod_threshold` | `forcing.dust_source` |
| land fraction | CLM operates on land columns | `terrain.fmask` |
| snow-free fraction | CLM `lnd_frc_mbl` | `forcing.snowc_am` |
| unfrozen fraction (ramp over the 2 K below `tmelt`) | CLM `liqfrac` | `forcing.stl_am` |
| `u*t` x `sqrt(1 + 1.21*(100*(w - w_thr))^0.68)` | Fecan (1999), CLM `frc_thr_wet_fct` | `forcing.soilw_am` |

`source_kind` (`physics.jam_dust_source`) chooses how the map itself is read:
`cam_erodibility` is CAM's geomorphic basin factor `mbl_bsn_fct_geo`, an
unbounded 0-5.7 weight — clipping it at 1 truncated 15 % of the global source
weight, concentrated in exactly the closed basins that are the world's
strongest sources — while `tegen_potential` is a HAMMOZ potential-source
fraction in [0, 1] that needs no threshold because its own preprocessing
embeds the land-cover mask. The default is `cam_erodibility`, matching the map
the data mirror ships.

**Known gap.** CLM's vegetation gate `1 - VAI/0.3` has no counterpart: no LAI,
VAI or land-cover field exists on `ForcingData`/`TerrainData`. It matters
because the basin factor is purely topographic — the Amazon (7.2 % of global
source weight, peak 4.40) and Congo (3.5 %, peak 2.16) carry *larger* values
than the Sahara, and in CAM they emit nothing only because CLM's LAI gate
zeroes them. Here the Fecan moisture factor suppresses them (~3.4x higher
`u*t` at 0.95 relative wetness) but does not eliminate them. Adding the field
is issue #777; a monthly VAI-masked map would also be the natural
`tegen_potential` product.

### Emitted number comes from the emitted size, not the mode's size

Freshly emitted particles are not at their mode's equilibrium size, so
converting an emitted mass flux to number with the mode geometry
(`rho/number_factor` at `dgnum`) is wrong by the cube of the size ratio. CAM
and the CESM emission-file generator both use the volume-mean diameter of the
*emission* size distribution, `x_mton = 6/(pi rho D^3)`:

| species / class | D [um] | source |
| --- | --- | --- |
| dust accumulation (0.1-1 um bin) | 0.7806 | CAM `dust_common::dust_set_params` |
| dust coarse (1-10 um bin) | 3.8983 | same |
| primary carbon (BC, POA) | 0.134 | CMIP7 `num_bc_a4`/`num_pom_a4` `mapping_equation` |
| accumulation sulfate (surface, biomass) | 0.134 | `num_so4_a1_ag` |
| accumulation sulfate (energy/industry, shipping) | 0.261 | `num_so4_a1_ene_vertical`, `num_so4_a1_ship_slv` |
| Aitken sulfate | 0.0504 | `num_so4_a2_res_trs` |

For accumulation dust this is 1.54e15 #/kg against the 1.17e17 #/kg the mode's
0.11 um `dgnum` gives — 75x. Every diameter is a differentiable parameter
(`DustParameters.emission_diameter`, `EmissionParameters.emission_diameter`),
not static config, so the emitted number stays calibratable like the rest of
the physics.

The dust mass split also follows CAM's `dust_emis_sclfctr`: 2.1 %
accumulation / 97.9 % coarse over those two bins, replacing an assumed
10/90. (CAM's third, 1.65e-5 Aitken share has no home in a population whose
Aitken mode carries no dust.) Sea salt needs neither correction: it already
partitions the Gong spectrum across the modes and derives its number from the
same spectrum.

## Status and caveats

- **Natural source magnitudes** are order-of-magnitude defaults, not
  inventory-calibrated. Prescribed CEDS/biomass emissions with HAMMOZ-grounded,
  differentiable per-super-sector characteristics are now available (#498; see
  above) and supersede the placeholders where a contract-conforming emissions
  file is supplied.
- **Wet scavenging** currently reconstructs the per-level precip-formation
  rate from column precip; exposing the true per-level formation/evaporation
  rates from the cloud schemes and adding re-evaporation re-injection is
  tracked in #499.
- **Real MAM4-JAX core** (#490) is not yet differentiable upstream; the
  harness itself is fully differentiable.

Out of scope, tracked separately: sectional/bulk families (#491), part2pop
diagnostics adapter (#492), SOA volatility basis (#493), heterogeneous
freezing (#494), aerosol optics in radiation (#495), gas-phase chemistry
coupling (#496).
