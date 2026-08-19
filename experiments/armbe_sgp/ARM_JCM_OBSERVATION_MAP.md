# ARM-to-JCM Observational Data Map

## Purpose

This document is the human-oriented starting point for building a general
ML-ready ARM observational dataset for JAX-GCM. The objective is broader than
cloud cover: provide observational inputs, targets, and evaluation diagnostics
for symbolic regression or other ML work on multiple JCM parameterizations.

The full ARM catalog is not itself a dataset design. It contains 20,440
site-facility datastream variants and many engineering, calibration, image, and
superseded products. This map groups a small number of scientifically relevant
product families by the JCM process they could constrain.

Product codes below are catalog candidates, not automatic endorsements. Before
ingestion, each product still needs a variable, units, dimensions, QC, cadence,
retrieval-definition, and temporal-coverage audit.

## Coverage Policy

SGP, NSA, and ENA are the durable reference observatories for the first dataset
design. Mobile facilities and short field campaigns remain in scope because
they add regimes and measurements unavailable at the anchors.

Each product receives one support tier from the generated catalog:

| Tier | Meaning | Dataset role |
| --- | --- | --- |
| `anchor_supported` | Present at SGP, NSA, or ENA | Eligible for the reusable backbone |
| `other_fixed_support` | No anchor support, but some non-mobile/non-campaign support | Secondary extension requiring review |
| `mobile_or_campaign_only` | Available only from mobile facilities or off-site campaigns | Specialized process/regime dataset, never silently required by the backbone |

An anchor-supported family may still have inconsistent instruments or retrieval
versions across the three anchors. Support tier establishes availability, not
cross-site comparability.

## Priority Map

### 1. Atmospheric State And Humidity

**JCM use:** initial/state variables, humidity diagnostics, large-scale
condensation, convection, radiation, and vertical diffusion.

**Quantities:** pressure, temperature, dew point or specific humidity, relative
humidity, horizontal wind, surface pressure, and derived stability.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | ARMBE atmospheric state | `armbeatm`, `armbeatmhires` | Harmonized state profiles and surface context |
| Backbone | Radiosondes | `sonde`, `interpolatedsonde`, `griddedsonde` | Independent profile source and ARMBE provenance check |
| Extension | Merged soundings | `mergesonde1mace`, `mergesonde2mace` | Higher-cadence thermodynamic profiles where retrieval assumptions are acceptable |
| Extension | Microwave radiometer retrievals | `mwrret1liljclou`, `mwrret2turn` | Continuous thermodynamic and column-water constraints |
| Extension | AERI profiles | `aeriprof`, `aeri01prof3feltz` | Boundary-layer thermodynamic structure |
| Surface context | Meteorology | `met`, `mettwr`, `towermet` | Near-surface state and forcing |

**Main caveat:** sondes are sparse in time, while MWR/AERI products are
retrievals with instrument-specific vertical resolution and priors. They should
not be treated as interchangeable samples without uncertainty metadata.

### 2. Clouds And Large-Scale Condensation

**JCM use:** cloud fraction, condensation, cloud optical properties, and
cloud-radiative coupling.

**Quantities:** layer cloud fraction, total cloud cover, cloud base/top,
condensate, cloud phase, liquid/ice water path, and radar/lidar cloud occurrence.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | ARMBE cloud/radiation | `armbecldrad`, `armbecldradhires` | Existing column and layer cloud-fraction targets |
| First extension | MICROBASE | `microbase`, `microbasekaplus` | Audit condensate-aware layer targets/features |
| Validation | ARSCL | `arscl1cloth`, `arsclkazr1kollias` | Cloud boundaries and occurrence from combined active sensors |
| Validation | Corrected cloud radar | `kazrcorge`, `kazrcormd`, `kazrcorpr` | Radar vertical structure and cloud occurrence |
| Validation | MPL cloud masks | `30smplcmask1zwang`, `mplcmaskml` | Independent lidar-sensitive cloud occurrence |
| Context | Ceilometer | `ceil`, `ceilpol` | Cloud base and boundary-layer context |

**Main caveat:** radar, lidar, ARMBE cloud fraction, and MICROBASE condensate
have different sampling volumes and cloud definitions. MICROBASE must be audited
for units, grid-mean versus in-cloud meaning, phase representation, and QC before
use as `qc` or `qi`.

### 3. Shortwave And Longwave Radiation

**JCM use:** shortwave and longwave radiation schemes, cloud optical effects,
surface albedo, and radiative evaluation of cloud closures.

**Quantities:** downwelling/upwelling SW and LW fluxes, direct/diffuse SW,
clear-sky estimates, spectral irradiance, surface skin temperature, and albedo.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | Quality-controlled radiation | `qcrad1long`, `tqcrad1long` | Primary surface broadband radiative targets |
| Backbone | Radiation flux analysis | `radflux1long`, `radfluxbrs1long` | Evaluated flux products and clear/cloud context |
| Instrument source | SIRS/SKYRAD | `sirs`, `sirs60s`, `skyrad`, `skyrad60s` | Native broadband measurements and provenance checks |
| SW diagnostics | SW flux analysis | `1swfanalsirs1long`, `15swfanalsirs1long` | SW and cloud-fraction diagnostics |
| Spectral/context | MFRSR | `mfrsr`, `mfrsrcldod1min` | Spectral SW, optical-depth, and cloud-screen context |
| Surface property | Spectral albedo | `surfspecalb1mlawer`, `surfspecalbmfrsr` | Surface albedo constraints |
| Thermal context | Infrared thermometer | `irt`, `gndirt` | Surface/sky brightness-temperature context |

**Main caveat:** most ARM radiation products constrain surface fluxes, whereas
JCM also predicts top-of-atmosphere quantities. Do not label surface-only
training as complete radiation closure validation.

### 4. Surface Exchange And Land State

**JCM use:** sensible and latent heat fluxes, momentum exchange, surface
temperature, soil state, and land-atmosphere coupling.

**Quantities:** sensible/latent heat flux, friction velocity or turbulent wind
statistics, soil temperature/moisture, ground heat flux, albedo, and near-surface
meteorology.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | Eddy covariance | `30ecor`, `30qcecor` | Turbulent sensible/latent heat and momentum-related diagnostics |
| Backbone at SGP | Energy-balance Bowen ratio | `30ebbr`, `15ebbr`, `5ebbr` | Independent surface energy flux method |
| Extension | Surface energy balance system | `sebs` | Mobile and anchor surface-energy measurements |
| Context | Surface/tower meteorology | `met`, `towermet`, `mettwr` | Wind, temperature, humidity, pressure, precipitation |
| Land state | Soil products | `soil`, `soilvue`, `okmsoil` | Soil moisture/temperature context where available |
| Surface optics | Spectral albedo | `surfspecalb1mlawer`, `surfspecalbmfrsr` | Radiation and land-surface coupling |

**Main caveat:** point flux footprints and heterogeneous coarse GCM grid cells
represent different spatial scales. Preserve terrain, land cover, instrument
height, and footprint metadata instead of treating observations as direct T31
grid-cell truth.

### 5. Boundary Layer And Vertical Diffusion

**JCM use:** turbulent diffusion, PBL height, surface-layer exchange, stability
functions, and mixing timescales.

**Quantities:** PBL height, wind and thermodynamic profiles, turbulence moments,
friction velocity, sensible heat flux, and vertical gradients.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | PBL-height products | `pblhtrl1zhang`, `pblhtsonde1mcfarl`, `ceilpblht` | Multi-method PBL-height targets |
| Backbone | ECOR/tower | `30ecor`, `towermet` | Surface turbulence and forcing |
| Profile | Radiosonde/AERI | `sonde`, `aeriprof` | Stability and vertical-gradient predictors |
| Profile | Radar wind profiler | `1290rwp`, `1290rwpwindmom` | Wind profile and boundary-layer dynamics |
| Specialized | Tethered balloon systems | `tbs`, `tbsmerged`, `tbsmergedincloud` | High-resolution campaign/anchor profiles |

**Main caveat:** diagnosed PBL height is method dependent. Keep the algorithm or
instrument family in the target identity rather than blending all PBL-height
products into one apparently objective label.

### 6. Convection And Precipitation

**JCM use:** convective triggering and adjustment, large-scale precipitation,
rain/snow partitioning, and precipitation efficiency.

**Quantities:** precipitation rate and accumulation, drop-size distribution,
hydrometeor phase, radar reflectivity/velocity structure, and thermodynamic
preconditioning.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | Surface precipitation | `rain`, `precipmet`, `disdrometer` | Rate/accumulation targets and basic microphysical context |
| Backbone | Video disdrometer | `vdis`, `vdisdrops` | Drop-size and precipitation-type information |
| Profile | Cloud/precipitation radar | `kazr`, `kazrcorge`, `radarcfad` | Vertical hydrometeor structure and regime diagnostics |
| Extension | Best-estimate precipitation | `precipbestats`, `precipbetseries` | Harmonized precipitation statistics where available |
| Specialized | Scanning precipitation radar | `xprecipradar`, `xprecipradarcmacppi` | Campaign-only spatial precipitation structure |

**Main caveat:** several scanning-radar products are mobile/campaign-only. They
are valuable for process discovery but should not become mandatory inputs to a
parameterization intended for evaluation at SGP, NSA, and ENA.

### 7. Aerosol Activation And Optical Properties

**JCM use:** aerosol optics, CCN activation, cloud-aerosol interactions, and
future aerosol parameterizations.

**Quantities:** CCN spectra, number/size distributions, composition, scattering,
absorption, hygroscopic growth, aerosol optical depth, and trace gases.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Backbone | CCN | `aosccn`, `aosccn1col`, `aosccn2cola` | Activation constraints |
| Backbone | Number concentration | `aoscpc`, `aoscpcf`, `aoscpcu` | Total particle-number context |
| Backbone | Size distributions | `aossmps`, `aosuhsas`, `aosaps` | Aerosol size-spectrum constraints |
| Composition | ACSM | `aosacsm`, `aosacsmtof` | Non-refractory aerosol composition |
| Optical | Nephelometer/PSAP/CLAP | `aosnephdry`, `aosnephwet`, `aospsap3w`, `aosclap3w` | Scattering, absorption, and humidity response |
| Column optical | MFRSR AOD | `mfrsraod1mich` | Column aerosol optical-depth evaluation |
| Harmonized | Aerosol best estimate | `aerosolbe1turn` | Candidate compact aerosol dataset |

**Main caveat:** inlet cuts, drying, instrument size ranges, supersaturation,
and STP/ambient conventions must be features of the schema. A single generic
`aerosol_concentration` variable would erase physically essential distinctions.

### 8. Atmospheric Composition And Simple Chemistry

**JCM use:** chemistry evaluation and boundary conditions for simple or future
chemistry schemes. This is less mature than the state/radiation/cloud pathways.

**Quantities:** ozone, CO, SO2, NOx, greenhouse gases, and aerosol composition.

| Priority | ARM family | Candidate codes | Recommended role |
| --- | --- | --- | --- |
| Candidate | Ozone | `aoso3`, `o3` | Surface composition target/context |
| Candidate | Carbon monoxide | `aosco` | Combustion/transport regime indicator |
| Candidate | Sulfur dioxide | `aosso2` | Aerosol precursor context |
| Specialized | NOx | `aosnox` | Campaign-focused chemistry extension |
| Candidate | Greenhouse gases | `aosghg`, `amcmethane` | Boundary/context data, not yet a JCM process target |

**Main caveat:** many composition measurements are surface-only. They cannot by
themselves constrain a vertically resolved chemistry scheme or transport error.

### 9. Processes ARM Does Not Directly Constrain Well

ARM is not a complete observing system for every JCM equation. The following
areas require reanalysis, satellite, aircraft, or specialized campaign data in
addition to ARM:

- large-scale dynamics and global forecast evolution;
- top-of-atmosphere radiation;
- non-orographic and orographic gravity-wave drag;
- global ocean, sea-ice, snow, and land-surface boundary conditions;
- globally representative trace-gas and aerosol vertical distributions.

ARM can still provide local state and process diagnostics for these areas, but a
column-site dataset should not be presented as globally identifying them.

## Missing Information And Candidate Sources

The remaining gaps do not have equal importance for SPEEDY and ECHAM. They also
cannot all be filled by ARM observations alone.

| Missing information | SPEEDY | ECHAM/JCM | Priority | Candidate data sources |
| --- | --- | --- | --- | --- |
| Cloud liquid/ice profiles | Useful extension, but not native to the standard closure | Essential for microphysics and radiation | Very high | ARM MICROBASE for retrieved condensate profiles; ARM MWR for LWP; CloudSat/CALIPSO or CERES CCCM for global phase and vertical context; CloudBench LES for resolved total condensate; dedicated radar-lidar retrievals may be needed to separate `qc` and `qi` |
| Turbulence and PBL profiles | Important | Essential | Very high | ARM ECOR, tower, PBL-height, Doppler-lidar, radar-wind-profiler, tethered-balloon, and sounding products; LASSO LES at ARM sites; CloudBench LES for tropical marine regimes; ERA5 for global PBL context, but not observational turbulence truth |
| Convective mass flux and entrainment | Important | Essential | Very high, difficult | LASSO LES and CloudBench LES as process-resolved sources; ARM KAZR, scanning radar, Doppler lidar, soundings, and precipitation products as observational proxies; field-campaign aircraft retrievals where available |
| Process tendencies and heating rates | Essential for equation discovery | Essential for equation discovery | Very high | LASSO or CloudBench LES process diagnostics; ARM radiative-flux profiles for estimated radiative heating; ERA5 tendency diagnostics if available; observed budget residuals from collocated ARM state and flux products; JCM diagnostics for identical-twin development only |
| Detailed precipitation microphysics | Limited value for standard SPEEDY | Important for ECHAM microphysics | Medium-high | ARM disdrometers, video disdrometers, KAZR, precipitation radar, and scanning radar; CloudBench `q_r`, `q_s`, microphysics source terms, and fluxes; GPM satellite products for global precipitation structure |
| Aerosol properties and CCN | Usually unnecessary | Important with aerosol-cloud or aerosol-radiation schemes | Medium-high | ARM AOS, CCN, CPC, SMPS, UHSAS, APS, ACSM, nephelometer, PSAP/CLAP, and MFRSR AOD; CERES SYN1deg for global AOD and aerosol radiative effects; MODIS/VIIRS aerosol products for broader coverage |
| Gravity-wave diagnostics | Relevant to global circulation | Relevant to global and upper-atmospheric circulation | Medium, poorly covered by ARM | ARM radiosondes and radar wind profilers for local proxies; ERA5 for resolved-wave context; AIRS, MLS, SABER, and GNSS radio occultation for global temperature-wave structure; specialized superpressure-balloon datasets for momentum-flux constraints |
| Chemistry observations | Not needed | Relevant only when chemistry is enabled | Low currently | ARM ozone, CO, SO2, NOx, and greenhouse-gas products for surface context; TROPOMI, OMI, MLS, MOPITT, and AIRS for global columns or profiles; aircraft campaigns for vertically resolved evaluation |

### Source Roles

| Role | Recommended source |
| --- | --- |
| Harmonized observational backbone | ARMBE |
| Detailed ARM process observations | Raw and derived ARM products beyond ARMBE |
| Cloud condensate retrievals | MICROBASE plus radar/lidar products |
| Process-resolved turbulence, convection, and tendencies | LASSO LES and CloudBench LES |
| Global radiation and bulk cloud properties | CERES SYN1deg |
| Global state and dynamical context | ERA5 |
| Global precipitation | GPM |
| Global aerosol/cloud context | MODIS/VIIRS and CERES |
| Gravity waves and upper atmosphere | ERA5 plus AIRS, MLS, SABER, and GNSS radio occultation |
| Chemistry | TROPOMI, OMI, MLS, and MOPITT plus ARM surface products |

These sources represent different kinds of evidence. ARM and satellite products
provide observational constraints but often rely on retrieval assumptions.
ERA5 is a dynamically complete model-assimilated analysis, not a direct
observation. LES datasets provide process-resolved supervision but remain
simulated and conditioned on their forcing and model assumptions. JCM diagnostics
are useful for development and identical-twin recovery, but are not independent
truth.

A practical first release should combine ARMBE, selected ARM products, CERES
SYN1deg, and ERA5. LASSO or CloudBench should remain a separate process-learning
module for quantities that observations cannot identify directly.

## Recommended Dataset Architecture

Do not flatten every ARM product into one universal table. Build three linked
layers instead:

1. **Native observation layer:** minimally transformed variables, original QC,
   instrument/retrieval identity, site/facility, cadence, and uncertainty.
2. **Harmonized physical layer:** canonical units and names, explicit temporal
   averaging windows, vertical coordinates, collocation distances, and masks.
3. **Process-example layer:** JCM-specific inputs and targets generated from a
   versioned recipe for a particular equation or closure.

Every process example should retain pointers to its source records. Splits must
operate on deployment/site and time blocks, not randomly on individual rows, to
avoid leakage from dense time series.

## First Build Sequence

1. Inventory variables and overlap for the backbone state products: ARMBE,
   radiosondes, surface meteorology, ECOR, QCRAD, and precipitation.
2. Define canonical names, units, QC policy, time bounds, and vertical-coordinate
   conventions without yet creating JCM-specific predictors.
3. Add cloud/radar/MICROBASE and aerosol groups as independently versioned
   modules because their retrieval conventions are more complex.
4. Generate one process-example dataset at a time and compare anchor-only
   results against mobile/campaign transfer.
5. Promote a mobile/campaign-only variable into a model-development experiment
   only when its deployment limitation is explicitly accepted.

## Generated References

- `outputs/arm_catalog_all.json`: complete machine-readable catalog with support
  tiers and deployment provenance.
- `outputs/arm_catalog_classes.csv`: technical instrument-class index, not the
  recommended human entry point.
- `inventory_arm_datastreams.py`: metadata-only catalog generator.
