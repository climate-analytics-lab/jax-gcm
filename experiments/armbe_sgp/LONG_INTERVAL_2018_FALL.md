# September-October 2018 ARMBE SGP SCM Experiment

## Configuration

- Products: `sgparmbeatmC1.c1` and `sgparmbecldradC1.c1`
- Window: 2018-09-03 11:30 through 2018-10-02 23:30 UTC
- Model: diagnostic SPEEDY single-column model, 8 sigma levels
- Physics timestep: 21,600 seconds (6 hours)
- Retained profiles: 119
- Cadence policy: `--regular-cadence` retained the dominant six-hour timestamp
  phase and removed 10 otherwise valid off-cadence profiles. It did not bridge
  any missing six-hour profile times.

The source window has 709 hourly timestamps. Of these, 580 do not contain a
complete atmospheric state profile and 10 are complete profiles outside the
dominant six-hour cadence. The resulting 119-profile sequence is contiguous.

## Expanded Comparisons

All values are daily means over 30 calendar dates. They are diagnostic
comparisons on prescribed atmospheric profiles, not free-running forecast skill.

| Field | Observed | SPEEDY SCM | Bias | Correlation |
|---|---:|---:|---:|---:|
| Surface SW down | 148.081 W/m2 | 175.525 | +27.443 | 0.25 |
| Surface SW net | 117.554 W/m2 | 140.420 | +22.866 | 0.24 |
| Surface LW down | 390.160 W/m2 | 376.165 | -13.995 | 0.80 |
| Surface LW up | 440.402 W/m2 | 441.403 | +1.001 | 0.94 |
| TOA SW down | 347.302 W/m2 | 360.632 | +13.330 | 0.81 |
| TOA SW net | 211.650 W/m2 | 210.911 | -0.739 | 0.25 |
| Cloud fraction | 0.575 | 0.709 | +0.134 | 0.27 |
| Sensible heat | 25.709 W/m2 | 57.783 | +32.074 | 0.08 |
| Latent heat | 48.265 W/m2 | -0.067 | -48.332 | -0.05 |
| Precipitation | 0.237 mm/hr | 1.641 | +1.404 | 0.43 |

Surface LW-up compares the model surface emission with ARMBE's 10 m upwelling
measurement, so it is a near-surface consistency check rather than an exact
collocation. Precipitation remains diagnostic because prescribed temperature
and humidity profiles do not retain convective feedback between observations.

LWP is available from ARMBECLDRAD but is not scored: the current SPEEDY archive
does not expose a directly comparable liquid-water-path diagnostic.

## Reproduction

```bash
JAX_PLATFORMS=cpu python run_scm.py \
  --atm data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbeatmC1.c1 \
  --cldrad data/order-267737/ftp.archive.arm.gov/fisherm1/267737/sgparmbecldradC1.c1 \
  --start 2018-09-03T11:30:00 --end 2018-10-02T23:30:00 \
  --dt 21600 --regular-cadence \
  --output outputs/real_2018-09-03_10-02.npz

python evaluate.py --run outputs/real_2018-09-03_10-02.npz --plot
```

The local archive, manifest, and plot are ignored under `outputs/`.
