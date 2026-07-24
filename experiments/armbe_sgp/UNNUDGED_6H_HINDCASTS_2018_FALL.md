# Independent Six-Hour ARMBE SGP Hindcasts

## Configuration

- Products: `sgparmbeatmC1.c1` and `sgparmbecldradC1.c1`
- Window: 2018-09-03 11:30 through 2018-10-02 23:30 UTC
- Model: SPEEDY single-column model, 8 sigma levels
- Physics timestep: 1,800 seconds (30 minutes)
- Forecast windows: 118 independent six-hour windows
- Prognostic fields: temperature, specific humidity, zonal wind, and meridional
  wind
- Surface pressure and geopotential: held at each window's initial observed
  state
- Physics carry, tracers, and atmospheric state: reset before each window

Each window starts from the observed profile at its initial timestamp, advances
with physics only for twelve 30-minute steps, and compares the final model
profile with the following six-hour observed profile. There is no state
nudging, dynamics, advection, or large-scale forcing.

## Interval-Mean Physics Diagnostics

These values are means over each six-hour forecast window, compared to hourly
ARMBE observations over the same half-open interval. They are not directly
comparable to the daily prescribed-state metrics in
`LONG_INTERVAL_2018_FALL.md`.

| Field | Observed | SPEEDY SCM | Bias | RMSE | Correlation |
|---|---:|---:|---:|---:|---:|
| Precipitation | 0.093 mm/hr | 0.364 | +0.271 | 0.556 | 0.39 |
| Surface SW down | 165.187 W/m2 | 200.264 | +35.077 | 199.924 | 0.23 |
| Surface SW net | 130.926 W/m2 | 160.211 | +29.285 | 159.160 | 0.23 |
| Surface LW down | 389.408 W/m2 | 371.038 | -18.371 | 36.373 | 0.70 |
| Surface LW up | 440.244 W/m2 | 447.926 | +7.682 | 26.528 | 0.67 |
| TOA SW down | 367.766 W/m2 | 360.377 | -7.389 | 374.109 | 0.05 |
| TOA SW net | 234.740 W/m2 | 230.362 | -4.378 | 250.216 | 0.17 |
| Cloud fraction | 0.575 | 0.534 | -0.041 | 0.458 | 0.29 |
| Sensible heat | 36.243 W/m2 | 57.554 | +21.310 | 57.596 | 0.63 |
| Latent heat | 50.619 W/m2 | -0.024 | -50.643 | 87.966 | -0.01 |

## Final Profile Error

Each row pools all forecast-window final states and vertical levels.

| Field | Observed | SPEEDY SCM | Bias | RMSE | Correlation |
|---|---:|---:|---:|---:|---:|
| Temperature | 250.435 K | 250.775 | +0.340 | 5.313 | 0.99 |
| Specific humidity | 3.461 g/kg | 3.239 | -0.223 | 1.144 | 0.98 |
| Zonal wind | 6.103 m/s | 6.069 | -0.034 | 3.800 | 0.90 |
| Meridional wind | 2.686 m/s | 2.450 | -0.236 | 4.299 | 0.76 |

## Historical Artifacts

This result was produced by the superseded `run_6h_hindcasts.py` prototype. Its
semantics are retained here for provenance, but new work must use the
config-driven cache and evaluator documented in `ML_FORECASTING_ROADMAP.md`.

The local archive, manifest, and plots are ignored under `outputs/`.
`hindcast_compare.png` aggregates the six-hour interval means to calendar-day
means, matching the temporal resolution of `compare.png`. The metrics above
remain six-hour interval metrics.

`hindcast_compare_complete_days.png` duration-weights interval means into full
UTC days and excludes incomplete boundary days.

`hindcast_compare_6h.png` plots every six-hour interval mean without daily
aggregation.
