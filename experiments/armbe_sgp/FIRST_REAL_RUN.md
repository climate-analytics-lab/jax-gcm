# First Real ARMBE SGP SCM Run

## Purpose

This is the first execution of the ARMBE SGP experiment using downloaded ARM
observations rather than the synthetic fixture. It is a diagnostic
single-column calculation: the dynamical core does not run, and ARMBE
atmospheric profiles are prescribed at every retained timestep.

The result is a plumbing and scientific-sanity baseline, not a definitive model
validation.

## Input Window

- Site: SGP Central Facility, Lamont, Oklahoma (C1)
- Source products: `sgparmbeatmC1.c1` and `sgparmbecldradC1.c1`
- Requested interval: 2018-06-03 11:30 through 2018-06-05 17:30 UTC
- Input cadence: hourly product timestamps, with usable atmospheric profiles at
  a six-hour cadence
- Retained states: 10 of 55 timestamps
- Integration timestep: 21,600 seconds (6 hours)

The window was selected because it is the longest contiguous six-hour sequence
in the initial June 2018 order. The ATM product contains profile gaps, so the
runner deliberately rejects a regular scan that would cross them.

## Prescribed And Evaluated Fields

The state profiles are `temperature_p`, `dewpoint_p`, `u_wind_p`, `v_wind_p`,
and `pressure_sfc`, using the `pressure` coordinate. Dewpoint is converted to
specific humidity, then converted from the source kg/kg convention to SPEEDY's
physics-facing g/kg convention and interpolated to eight SPEEDY sigma levels.

`temperature_sfc` is used as a time-varying land-temperature proxy. It is a 2 m
air-temperature measurement, not a confirmed skin-temperature field, and is a
known limitation for surface-flux interpretation.

Evaluation targets are:

- `precip_rate_sfc`
- `swdn` and `lwdn`
- `sensible_heat_flux_baebbr` and `latent_heat_flux_baebbr`
- `tot_cld` and `lwp` as saved diagnostic targets

BAEBBR is the initial surface-flux selection; QCECOR remains the fallback if
BAEBBR coverage is unsuitable for a later window. The run manifest records the
resolved names.

## Fixed Forcing And Terrain

- Land fraction: 1.0; land fluxes enabled
- Orography: 315 m
- Bare-land albedo: 0.20
- Soil moisture: 0.30
- CO2: 407 ppmv
- Snow and sea ice: zero
- Calendar: Gregorian

Only the land-temperature proxy varies with time. Soil moisture and albedo are
still fixed.

## Preliminary Daily-Mean Comparison

The ten retained states span three calendar dates. Daily means are shown below;
partial days and the short window mean these statistics are descriptive only.

| Field | Observed | SPEEDY SCM | Bias |
|---|---:|---:|---:|
| Surface shortwave down | 329.4 W/m2 | 287.6 W/m2 | -41.8 W/m2 |
| Surface longwave down | 363.1 W/m2 | 358.2 W/m2 | -4.9 W/m2 |
| Sensible heat | 38.0 W/m2 | 67.7 W/m2 | +29.6 W/m2 |
| Latent heat | 148.6 W/m2 | 15.3 W/m2 | -133.3 W/m2 |
| Precipitation | 0.000 mm/hr | 0.692 mm/hr | +0.692 mm/hr |

The close longwave mean is an important sanity check after correcting the
ARMBE-to-SPEEDY humidity-unit boundary. Precipitation remains diagnostic rather
than a headline score because prescribed-state mode does not retain convective
thermodynamic feedback between observations.

## Artifacts And Reproduction

Ignored local artifacts:

- Archive: `outputs/real_2018-06-03_05.npz`
- Manifest: `outputs/real_2018-06-03_05.manifest.json`

Command:

```bash
JAX_PLATFORMS=cpu python run_scm.py \
  --atm data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbeatmC1.c1 \
  --cldrad data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbecldradC1.c1 \
  --start 2018-06-03T11:30:00 --end 2018-06-05T17:30:00 \
  --dt 21600 --output outputs/real_2018-06-03_05.npz

python evaluate.py --run outputs/real_2018-06-03_05.npz
```

## Next Steps

- Find longer contiguous six-hour profile sequences in the downloaded annual
  files.
- Identify a true land skin-temperature forcing.
- Compare BAEBBR and QCECOR flux targets and document the final choice.
- Run a temperature/humidity relaxation sensitivity before interpreting
  precipitation behavior.
