# The dinosaur-SL JAM science configuration

The locked reference configuration for online-aerosol (JAM/MAM4) climate
runs on the semi-Lagrangian dinosaur backend, and the measurements behind
its settings. Numbers are from 215-day T63L47 real-terrain campaigns on a
single A100-40GB (July 2026).

## The configuration

```bash
python -m jcm.main \
    physics=echam-jam grid=echam_t63_l47_hybrid \
    physics.jam_microphysics=mam4_jax \
    terrain=from_file terrain.file=$JCM/data/bc/t63/terrain.nc \
    forcing=from_file forcing.file=$JCM/data/bc/t63/forcing.nc \
    init=jw init.rh=0.0 \
    run=longrun diffusion.tracer_positivity=true \
    +sl_off_centering=0.2 \
    run.time_step=15
```

Load-bearing pieces:

- **Semi-Lagrangian transport (now unconditional)** — nodal tracer transport (no spectral
  round-trip for the ~40 JAM tracers) with Bermejo–Staniforth monotone
  limiting: positivity by construction. The Eulerian path requires
  `tracer_positivity` clipping to survive at all and loses ~20 % of
  near-source tracer mass to it by day 10.
- **`sl_off_centering=0.2`** — required over real orography (`off=0` is
  unstable even from a good state); validated over 215 days. This is now
  the default everywhere (`DEFAULT_OFF_CENTERING` in the dinosaur dycore,
  shared by direct construction and the runner), so the explicit override
  above is documentation rather than a requirement.
- **`run=longrun`** — carries the calibrated upper sponge (10 levels,
  1.5 h, `target_T_K=250`). Without it the model top refrigerates
  (T_min < 100 K by day ~135).
- **`forcing.ozone_file: auto`** (default) — the packaged climatological
  ozone. The analytic fallback biases clear-sky OLR ~12 W/m² low.

## Timestep: why dt = 15 min

All of dt = 15/20/30/45 min complete 215 days through the SH-winter
window without incident — stability does not choose the timestep. Fidelity
and economics do:

| dt [min] | days/hr | ms/step | TOA net | SW CRE | IWP [g/m²] | dust [mg/m²] | sea salt |
|---|---|---|---|---|---|---|---|
| 15 | 115 | 326 | −5.8 | −56.3 | 12.1 | 15.7 | 1.18 |
| 20 | 121 | 413 | −4.1 | −55.2 | 13.7 | 18.3 | 1.22 |
| 30 | 129 | 581 | +1.3 | −51.5 | 17.4 | 24.4 | 1.51 |
| 45 | 150 | 750 | +6.0 | −48.7 | 22.5 | 34.2 | 2.11 |

(days 155–215 means, W/m² unless noted.)

1. **The speedup saturates.** Per-step cost *rises* with dt because
   radiation fires every 8/6/4/3 steps at the fixed 2 h interval — at long
   dt the model is radiation-bound. Tripling the step buys +30 %
   throughput. The real speed lever is the radiation solve itself, not dt.
2. **The climate drifts monotonically with dt**: SW CRE weakens ~8 W/m²
   across the sweep, IWP nearly doubles, TOA swings 12 W/m², and the
   wind-driven interactive sources are strongly dt-sensitive (dust ×2.2,
   sea salt +80 %). Calibrating at a long dt would bake step-size
   dependence into the physics parameters.

**dt = 15 min is the science target**; dt = 20 is within noise if a free
5 % is ever needed; the headroom to dt = 45 is stability insurance, not a
setting to use.

## Vertical extensions (middle atmosphere, L95)

The ECHAM6 MA table (`layers: 95`, lid ~0.01 hPa) works with the
production physics unchanged:

| grid | GPUs (A100-40GB) | throughput |
|---|---|---|
| T63L95, full science | 1 | 52 days/hr |
| T106L95 | 1 | out of memory |
| T106L95 | 4, `spmd_mesh [2,2,1]` | 42 days/hr |

Level-resolved aux forcing must be regenerated for L95
(`jcm.data.bc.interpolate_ozone --nlevels 95` works out of the box; the
oxidant climatology needs the same treatment). 80 GB cards are expected to
fit T106L95 on one or two GPUs.

## Known biases at this baseline (calibration targets)

- SW side too bright (SW CRE −56 vs −45, LWP ~120 g/m²); LW CRE low
  (ice too thin) — cloud microphysics parameters.
- OLR ~15 W/m² low after the ozone fix: upper-troposphere moisture
  (physics, shared with the pySES backend — not a transport artifact) plus
  a residual under investigation upstream
  (climate-analytics-lab/jax-rrtmgp#19).
- Aerosol lifetimes vs observations (from
  `tools/jam_burden_report.py --emissions-file`): BC 7.8 d (obs 5–8 ✓),
  SO4 10.1 d (obs ~4–5 — wet scavenging), sea-salt source under-emitting.
