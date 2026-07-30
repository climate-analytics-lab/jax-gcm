# SGP Instantaneous Cloud-Diagnostic Experiment

## Purpose

This experiment tests the SPEEDY cloud-cover diagnostic against simultaneous
ARM Southern Great Plains (SGP) observations. It is not a weather forecast.
For every valid observation time, the model receives the observed atmospheric
profile and diagnoses cloud cover once; the result is compared with observed
cloud cover at that same time.

## Scientific Definition

Each independent sample begins from one ARMBE atmospheric profile, interpolated
onto eight SPEEDY sigma levels. The input fields are temperature, specific
humidity, zonal wind, meridional wind, surface pressure, and derived
geopotential. SGP ARMBEATM has no direct specific-humidity profile, so humidity
is derived from the ARMBE dew-point profile and pressure.

SPEEDY physics is evaluated once to obtain the shortwave cloud diagnostic,
`cloudc`. No state is advanced from one ARMBE timestamp to the next: every
sample has fresh empty tracer and physics carry. There is no dynamical core,
horizontal transport, nudging, relaxation, or forecast integration. The
observed ARMBE `temperature_sfc` at the same timestamp is prescribed as SPEEDY
land surface temperature (`stl_am`). It is a 2 m air-temperature proxy, not a
verified skin-temperature observation.

The target is ARMBE total cloud fraction, `tot_cld`, represented as
`cloud_fraction`. A sample is evaluated only if `tot_cld` is finite and
`qc_tot_cld == 0`. The reported metric is the root mean square error over these
QC-passed same-time pairs.

This remains an imperfect observation operator: ARMBE `tot_cld` is a
narrow-field-of-view total cloud fraction, while SPEEDY `cloudc` is a cloud
diagnostic. SPEEDY `cloudstr` is retained as a separate diagnostic and is not
added to `cloudc` without an explicit cloud-overlap assumption.

The diagnostic evaluator selects reviewed model operators by name. The
exploratory `cloudc_plus_cloudstr_raw` operator returns the literal sum without
clipping or an overlap assumption, so it can exceed one and must not be
interpreted as a physical cloud fraction.

## Fixed Defaults

| Choice | Value |
| --- | --- |
| Vertical grid | 8 SPEEDY sigma levels |
| Model parameters | SPEEDY defaults, with no optimizer or overrides |
| Terrain elevation | 315 m |
| Land fraction | 1.0 |
| Land surface fluxes | enabled |
| Surface albedo | 0.20 |
| Soil wetness | 0.30 |
| Snow cover and sea ice | 0 |
| Carbon dioxide | 407 ppmv |
| Sea-surface temperature | Mean ARMBE `temperature_sfc` over the loaded record |
| Physics carry | Reset for every observation time |

If ARMBE surface temperature is unavailable, the forcing falls back to 295 K.
Missing values in an otherwise available surface-temperature record are filled
with that record's mean. These fallbacks should be reported if triggered.

## September 2018 Configuration

The initial diagnostic evaluation uses all valid SGP atmospheric profiles from
the configured September 2018 range, without forecast horizons, strides, or
lead times.

```yaml
atm: "../data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbeatmC1.c1/sgparmbeatmC1.c1.20180101.003000.nc"
cldrad: "../data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbecldradC1.c1/sgparmbecldradC1.c1.20180101.003000.nc"
start: "2018-09-01"
end: "2018-10-01"
nlev: 8
batch_size: 8
target:
  observation: cloud_fraction
  model: shortwave_rad.cloudc
```

```bash
python diagnose_cloud_cover.py \
  --config configs/sgp_2018_september_diagnostic.yaml \
  --out-dir outputs/diagnostic_sgp_2018_september
```

The output directory contains `cloud_pairs.nc`, with one predicted/observed
pair per profile, plus `metrics.json` and a resolved `manifest.json`.

The former free-rollout forecast setup is retained separately for later online
validation of a calibrated equation. It is not the current equation-accuracy
experiment.
