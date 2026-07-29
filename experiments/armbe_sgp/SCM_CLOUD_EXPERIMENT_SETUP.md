# SGP Single-Column Cloud Experiment

## Definitions

Each forecast begins from one ARMBE atmospheric profile, interpolated onto the
eight SPEEDY sigma levels. The initialized atmospheric fields are temperature,
specific humidity, zonal wind, meridional wind, surface pressure, and derived
geopotential. 

Temperature, humidity, and wind then evolve under local SPEEDY physical
parameterizations, including radiation, convection, condensation, turbulence,
and surface fluxes. There is no modelled transport of air from outside the column,
nudging, relaxation, or prescribed large-scale forcing.

The locally available SGP ARMBEATM product has no direct
specific-humidity profile, so specific humidity is derived from its dew-point
profile and pressure. (The generic input reader can instead use direct specific
humidity or relative humidity where another ARMBE product supplies it.)

The observed ARMBE `temperature_sfc` record is prescribed throughout each
forecast as SPEEDY land surface temperature (`stl_am`). It is a 2 m air
temperature proxy, not a verified skin-temperature observation. Thus this is a
boundary-forced atmospheric forecast, not a fully free land-surface forecast.

The evaluated model quantity is SPEEDY shortwave cloud fraction, `cloudc`. The
observational target is ARMBE total cloud fraction, `tot_cld`, represented in
the cache as `cloud_fraction`. A comparison is retained only when
`qc_tot_cld == 0` and the value is finite. RMSE is calculated independently at
each observation lead time.

This cloud comparison is provisional. ARMBE `tot_cld` is a narrow-field-of-view
total cloud fraction, whereas `cloudc` is a SPEEDY cloud diagnostic. SPEEDY
also has `cloudstr`, a low-level stratiform diagnostic. The experiment does not
add these quantities because their physical overlap is unspecified; `cloudc`
versus `tot_cld` must therefore be reported as an imperfect observation
operator.

## Fixed Defaults

| Choice | Value | Scientific role |
| --- | --- | --- |
| Vertical grid | 8 SPEEDY sigma levels | Common vertical representation for ARMBE profiles and SPEEDY physics. |
| Physics timestep | 30 minutes | Local physical-tendency integration interval. |
| Default forecast horizon | 6 hours | Configurable evaluation horizon. |
| Default start stride | 6 hours | Separation between independent forecast initializations. |
| Default observation cadence | 6 hours | Valid target lead-time spacing. |
| Model parameters | SPEEDY defaults | No parameter overrides or optimizer in the current evaluation. |
| Terrain elevation | 315 m | Fixed SGP single-column terrain. |
| Land fraction | 1.0 | Entire column is land. |
| Land surface fluxes | enabled | SPEEDY land exchange is active. |
| Surface albedo | 0.20 | Constant lower-boundary albedo. |
| Soil wetness | 0.30 | Constant lower-boundary soil moisture. |
| Snow cover and sea ice | 0 | No snow or sea-ice contribution. |
| Carbon dioxide | 407 ppmv | Constant radiative composition. |
| Sea-surface temperature | Mean ARMBE `temperature_sfc` over the loaded record | Fixed background value; the column is land. |

If ARMBE surface temperature is unavailable, the forcing falls back to 295 K.
Missing values in an otherwise available surface-temperature record are filled
with that record's mean. These fallbacks have not been used as an alternate
scientific experiment and should be reported if triggered.

## September 2018 Configuration

The current SGP run uses 24-hour free physics rollouts evaluated every six hours (horizon 24hr, stride + obs cadence 6hr). It
therefore evaluates the same model at 6, 12, 18, and 24 hour leads while using
overlapping forecast windows. The configured ARMBE range spans 1 September to
1 October 2018; starts without a complete 24-hour surface-temperature forcing
record are excluded.

```yaml
atm: "../data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbeatmC1.c1/sgparmbeatmC1.c1.20180101.003000.nc"
cldrad: "../data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbecldradC1.c1/sgparmbecldradC1.c1.20180101.003000.nc"
start: "2018-09-01"
end: "2018-10-01"
nlev: 8
physics_dt_minutes: 30
horizon_minutes: 1440
stride_minutes: 360
observation_cadence_minutes: 360
target:
  observation: cloud_fraction
  model: shortwave_rad.cloudc
  reduction: trajectory
```

The evaluation currently uses all cached windows and an execution batch size of
eight, which changes computational throughput but not the scientific result.

```yaml
cache: "../outputs/cache_sgp_2018_september_24h_6h"
out_dir: "../outputs/evaluation_sgp_2018_september_24h_6h"
split: all
batch_size: 8
```
