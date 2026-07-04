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

For now the JAM path **augments** MACv2-SP rather than replacing it: MACv2-SP
is kept for the aerosol radiative optics and Twomey factor that radiation and
the cloud schemes currently read — a temporary fudge in lieu of proper JAM
aerosol↔radiation and aerosol↔microphysics coupling (#495). Once JAM supplies
those, MACv2-SP need not be included.

The 2M scheme uses ARG's `activated_cdnc` where it is non-empty and falls back
to the MACv2-SP SPA floor wherever the online source is ≈0 (e.g. before the
prognostic JAM tracers spin up from a zero-seeded initial state), so the
default JAM+2M run always activates droplets.

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
- `forcing.dust_file` — potential-dust-source map (`pot_source`, 0–1,
  e.g. `dust_potential_sources_T63.nc`), clipped to `DustEmissions`' [0, 1]
  erodibility contract (the file's `-1` missing marker → 0).
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
