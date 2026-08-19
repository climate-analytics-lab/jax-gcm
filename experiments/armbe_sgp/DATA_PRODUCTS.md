# Data Inventory and Future Products

This note distinguishes fields already available in the downloaded ARMBE
collections from products that would support future validation or a different
single-column-model experiment. It does not imply that every listed product
should be downloaded or used. `LOCAL_DATA_AUDIT.md` records the exact local
filesystem inventory and canonical-release policy.

## Original 2018 Experiment Input

The original SGP experiment uses annual 2018 files for:

- `sgparmbeatmC1.c1`: atmospheric state, surface meteorology, turbulent fluxes,
  and pressure- and height-grid sounding fields.
- `sgparmbecldradC1.c1`: cloud, radiation, precipitable-water, and
  liquid-water-path fields.

The current ARMBE runner uses pressure-grid temperature, dewpoint, horizontal
winds, surface pressure, and 2 m temperature to construct SPEEDY states. It
scores precipitation, BAEBBR sensible/latent heat, surface SW/LW fluxes, TOA
SW fluxes, total cloud fraction, and LWP where a model diagnostic exists.

## Downloaded Multi-Site Coverage

ARM order `267892` contains the following nominal coordinate-time coverage.
Coverage does not imply a complete usable atmospheric profile at every listed
timestamp; standard ARMBEATM sounding-derived states are substantially sparser
than their hourly time coordinate.

`ARMBE_STANDARD_VARIABLE_CADENCE.md` gives a complete per-variable breakdown of
the one-hour temporal operator, empirical finite-value availability, and spacing
between available cells across order `267892`. Its source CSV can be regenerated
with `summarize_armbe_cadence.py`.

The order is complete relative to its 298-file order manifest, but it is not a
complete copy of every ARMBE family in the current ARM catalog. In particular,
local data lack `sgparmbe2dgridX1.c1` and `sgparmbestnsX1.c1`. See
`LOCAL_DATA_AUDIT.md` for the catalog comparison and duplicate-collection audit.

| Site | Standard ATM coverage | Standard CLDRAD coverage | High-resolution coverage |
|---|---|---|---|
| ANX, Andenes/COMBLE | 2019-12-01 to 2020-05-31 | Same | None |
| AWR, West Antarctica/AWARE | 2016-01-01 to 2016-12-31 | Same | None |
| COR, Cordoba/CACTI | 2018-09-23 to 2019-04-30 | Same | None |
| ENA, Azores | 2014-01-01 to 2023-12-31 | Same | ATMHIRES and CLDRADHIRES: 2023-01-01 to 2024-12-31 |
| EPC, La Jolla/EPCAPE | 2023-02-15 to 2024-02-14 | Same | None |
| GUC, Gunnison/SAIL | 2021-09-01 to 2023-06-15 | Same | None |
| HOU, Houston/TRACER | 2021-10-01 to 2022-09-30 | Same | None |
| MAO, Manacapuru | 2014-01-01 to 2015-12-31 | 2014-01-01 to 2015-12-01 | None |
| MOS, MOSAiC | None in this order | 2019-11-01 to 2020-10-31 | None |
| NIM, Niamey | 2006-01-01 to 2006-12-30 | Same | None |
| NSA, Barrow | 2001-01-01 to 2023-12-31 | 1998-01-01 to 2023-12-31 | ATMHIRES and CLDRADHIRES: 2023-01-01 to 2024-12-31 |
| OLI, Oliktok Point | 2013-10-01 to 2021-06-14 | Same | None |
| SGP, Lamont | 1994-01-01 to 2023-12-31 | 1996-01-01 to 2023-12-31 | ATMHIRES and CLDRADHIRES: 2023-01-01 to 2024-12-31 |
| TWP C1, Manus | 1996-01-01 to 2010-12-31 | 1996-01-01 to 2011-12-31 | None |
| TWP C2, Nauru | 1998-01-01 to 2010-12-31 | Same | None |
| TWP C3, Darwin | 2002-01-01 to 2010-12-31 | 2002-01-01 to 2011-12-31 | None |

For SGP, standard ATM and CLDRAD overlap from 1996 through 2023. The separate
SGP ARMBELAND stream covers 1994 through 2016. MOSAiC has no ARMBEATM file in
this order and therefore cannot initialize the current SCM experiment.

## Downloaded but Unused Fields

### Quality Control and Observational Spread

ARMBECLDRAD provides `qc_*` flags and hourly `stdev_*` fields for radiation,
PWV, LWP, total cloud fraction, and height-resolved cloud fraction. These are
the highest-priority additions to the analysis.

- Apply product QC before calculating comparison scores.
- Report within-window spread with every six-hour interval mean.
- Treat flagged periods as missing rather than as model error.

### Humidity and Surface-State Checks

- `pwv`, `stdev_pwv`, and `qc_pwv`: an independent column-integrated check on
  humidity derived from sounding dewpoint.
- `relative_humidity_sfc`, `u_wind_sfc`, and `v_wind_sfc`: 2 m RH and 10 m
  winds for boundary-layer validation.
- `sensible_heat_flux_qcecor` and `latent_heat_flux_qcecor`: alternative flux
  products to quantify the uncertainty relative to the BAEBBR values currently
  used as primary targets.

### Radiation and Cloud Diagnostics

- `swdif`, `swdir`, `totswfluxdn`, `clrswfluxdn`, and `skycover`: direct,
  diffuse, whole-sky, clear-sky, and sky-cover radiation context.
- `swup / swdn`: observed effective surface albedo for a future fixed-albedo
  sensitivity; it is not yet a time-varying SPEEDY forcing.
- Height-resolved `cld_frac`, `cld_frac_radar`, and `cld_frac_MPL`: cloud
  structure beyond total cloud fraction.
- Satellite `cld_low`, `cld_mid`, `cld_high`, `cld_tot`, `cld_thick`, and
  `cld_top`: cloud-regime diagnostics.
- `lw_net_TOA`: present, but no directly comparable SPEEDY TOA-longwave
  diagnostic has been confirmed.

### NWP Context Only

The ARMBEATM files include `u_wind_nwp_p`, `v_wind_nwp_p`,
`temperature_nwp_p`, `relative_humidity_nwp_p`, and `omega_nwp_p`. These can
provide meteorological context, but isolated NWP profiles or vertical velocity
are not a budget-consistent large-scale SCM forcing data set.

## Constraints Missing From the Downloaded Pair

- Verified land skin temperature. `temperature_sfc` is explicitly a 2 m air
  temperature and is currently used only as a `stl_am` proxy.
- Soil moisture, soil temperature, soil heat flux, snow, vegetation state, and
  a direct surface-albedo forcing field.
- Large-scale temperature and moisture advection tendencies and a consistent
  forcing budget.
- Direct cloud liquid/ice profiles. ARMBE LWP is available, but the current
  SPEEDY diagnostic archive has no directly comparable LWP output.

`armbeland` is a separate ARMBE product family, but the SGP C1 stream
`sgparmbelandC1.c1` covers 1994-2016 only. It cannot provide land constraints
for the current 2018 experiments.

## SPEEDY Forcing and Terrain Inputs

The following are boundary conditions rather than atmospheric profile-state
inputs. Current values and potential upgrades are:

| SPEEDY input | Current value | Better data possibility |
|---|---|---|
| Latitude / longitude | SGP C1 constants | Already exact from ARMBE metadata. |
| Surface elevation | Fixed 315 m | Read ARMBE `alt` directly; expected effect is minor. |
| Land fraction | 1.0 | Correct for the SGP land column. |
| Land surface temperature `stl_am` | ARMBE `temperature_sfc`, a 2 m air-temperature proxy | Highest-priority upgrade: a verified land skin temperature. |
| Soil moisture `soilw_am` | Fixed 0.30 | Not in the downloaded ATM/CLDRAD pair; obtain a defensible land/soil source. |
| Bare-land albedo `alb0` | Fixed 0.20 | Diagnose effective observed albedo from QC-passed daytime `swup / swdn`; test a fixed-value sensitivity before adding time variation. |
| Snow cover | Fixed 0 | Requires a separate snow/land source. |
| Sea ice / SST | Fixed; irrelevant at `fmask=1` | Not relevant for this land column. |
| CO2 | Fixed 407 ppmv | Reasonable for 2018; unimportant for six-hour forecasts. |
| Solar geometry | Calendar date and SGP latitude | Used, but SPEEDY resolves only seasonal daily-mean insolation, not the local solar diurnal cycle. |

Generic `ForcingData` also carries CH4, aerosol, and ozone-climatology fields,
but the configured SPEEDY path does not currently use them as observational SGP
forcing inputs.

## Interpretation Limits

- ARMBE is a single-point observation product. ARM recommends statistics when
  comparing it with a model grid-box representation.
- ARMBE inherits quality issues from contributing VAPs, including ARSCL, QCRAD,
  and MWRRET. QA masking and uncertainty reporting precede strong conclusions.
- SPEEDY shortwave uses daily-mean, zonally averaged insolation. Six-hour
  radiation and surface-flux values remain process diagnostics, not diurnal
  radiation skill scores.
- Better observations alone cannot restore the absent SPEEDY solar diurnal
  cycle.

## Candidate External ARM Products

### VARANAL: First Choice for a Constrained SCM

The Constrained Variational Analysis product provides continuous one-hour,
25-mb large-scale forcing. ARM describes its use for SCM, CRM, and LES studies.
It includes advective tendencies, vertical velocity, surface skin temperature,
cloud/radiation fields, and large-scale conditions.

Use VARANAL only for a separately labelled forcing-constrained experiment. Do
not treat it as an incremental input to the current physics-only hindcasts, and
do not substitute `omega_nwp_p` alone for its consistent forcing budget.

### QCRAD: Radiation Quality Assurance

QCRAD provides one-minute surface-radiation measurements with detailed QC and
corrections, including an operational downwelling-shortwave correction. Use it
to audit the ARMBE radiation targets and to identify questionable periods.

### MWRRET: Independent PWV and LWP

MWRRET provides approximately 28-second retrievals. ARM recommends
best-estimate `be_pwv` and `be_lwp` variables for general use. Use PWV to check
the humidity column and LWP to characterize uncertainty in the ARMBE LWP target.
Thin-cloud LWP below roughly 30 g/m2 remains challenging for this retrieval.

### ARSCL: Cloud Structure

ARSCL combines radar and lidar observations into cloud location/base,
hydrometeor-height distributions, reflectivity, Doppler quantities, and
vertical velocity. It is appropriate if cloud regime and vertical structure
become evaluation objectives. It is not needed for the current total-cloud
comparison alone.

### Surface and Sounding Source Products

Consider source surface-meteorology, BAEBBR, QCECOR/ECOR, and radiosonde
products only for targeted audits: flux-product disagreement, surface/skin
temperature selection, and validation of ARMBE profile conversion. ARMBE remains
the preferred common hourly comparison product.

## Recommended Sequence

1. Use existing ARMBE QC flags, standard deviations, PWV, and QCECOR fields.
2. Obtain QCRAD and MWRRET for radiation and humidity/LWP quality assurance.
3. Obtain VARANAL only when beginning a distinct constrained-forcing SCM study.
4. Add ARSCL if evaluating cloud structure or cloud-regime dependence.
5. Query ARM Data Discovery for the exact SGP datastream/version availability
   before ordering any source product; do not hardcode stream names from this
   note.

## Official References

- [ARMBE](https://www.arm.gov/capabilities/vaps/armbe)
- [VARANAL](https://www.arm.gov/capabilities/vaps/varanal)
- [QCRAD](https://www.arm.gov/capabilities/vaps/qcrad)
- [MWRRET](https://www.arm.gov/capabilities/instruments/mwrret)
- [ARSCL](https://www.arm.gov/capabilities/vaps/arscl)
- [ARM Data Discovery](https://adc.arm.gov/discovery/)
