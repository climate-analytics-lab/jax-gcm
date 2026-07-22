# Data Inventory and Future Products

This note distinguishes fields already available in the downloaded 2018 SGP
ARMBE archive from products that would support future validation or a different
single-column-model experiment. It does not imply that every listed product
should be downloaded or used.

## Current Download

The local ARM order contains annual 2018 files for:

- `sgparmbeatmC1.c1`: atmospheric state, surface meteorology, turbulent fluxes,
  and pressure- and height-grid sounding fields.
- `sgparmbecldradC1.c1`: cloud, radiation, precipitable-water, and
  liquid-water-path fields.

The current ARMBE runner uses pressure-grid temperature, dewpoint, horizontal
winds, surface pressure, and 2 m temperature to construct SPEEDY states. It
scores precipitation, BAEBBR sensible/latent heat, surface SW/LW fluxes, TOA
SW fluxes, total cloud fraction, and LWP where a model diagnostic exists.

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
