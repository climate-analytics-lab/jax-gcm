# Sampling the aerosol-free radiation call (AeroCom ERFari)

AeroCom's aerosol effective radiative forcing from aerosol–radiation
interactions (ERFari) is diagnosed from the difference between the normal
top-of-atmosphere fluxes and a set of *aerosol-free* companions
(`rsutnoa`, `rlutnoa` and their clear-sky variants). Producing those
companions means solving the radiative transfer **twice** per
radiation step — once with the aerosol optics and once with them zeroed.

Radiation is ~87 % of a T63L47 ECHAM+JAM timestep, so that second solve is
not a rounding error: it costs **+64 %** wall-clock. This document records
what we measured when trying to make it cheaper, and why the API ended up
shaped the way it is.

## The knob

```yaml
aerosol_free_interval: 1      # null = off, 1 = exact, N > 1 = subsampled
```

| setting | what it does | s/sim-day | cost |
|---|---|---|---|
| `null` | no `*noa` fluxes at all (default) | 30.9 | +0 % |
| `1` | companion solve every radiation step — exact | 50.8 | +64 % |
| `N > 1` | companion every Nth step, effect held between | 36.2 at N=4 | +17 % |

This one integer provides a fine-tuned control which is monotonic in 
both directions: cost falls and error grows with N. 
The costs were measured at T63L47 on an A100 with semi-Lagrangian transport,
steady-state chunks only. The simulation itself is bit-identical at 
every N. The model always sees aerosol; only the reported `*noa` fluxes 
are extrapolated between calls. 

Between companion solves the aerosol effect is held as a **fraction** of
the all-sky flux and re-applied to the fresh all-sky value. That fraction
is not a stylistic choice. Holding the effect as an *absolute* W m⁻², which
was the first implementation, works in the longwave and fails in the
shortwave: the SW effect tracks the solar cycle, so on the night side the
scheme subtracts a stale daytime effect from a zero flux — reconstructing a
non-zero aerosol-free flux for a dark column, which is not merely
inaccurate but impossible. 


## What we measured

Four runs, **same commit**, T63L47 with semi-Lagrangian transport, ERA5
wind-nudged, 180 days with the first 25 discarded as spin-up. Nudging is
what makes the comparison possible at all: it holds the large-scale
trajectory roughly fixed so that differences in the diagnostic are not
swamped by chaotic divergence.

The fourth run is the important one: a **second run of the `exact` scheme**,
on a different node and a different A100 variant (PCIe rather than SXM4). It
measures the floor — how far apart two runs that *should* be identical
actually land. Because the runs are nudged, this bounds run-to-run
reproducibility (hardware non-determinism plus what little trajectory
freedom nudging leaves), not free-running climate noise, which would be
larger.

| run | SW | LW | total ERFari (W m⁻²) |
|---|---|---|---|
| N=1 | −0.8202 | +0.0547 | −0.7655 |
| N=1 (twin) | −0.8177 | +0.0545 | −0.7633 |
| N=4 | −0.9150 | +0.0544 | −0.8607 |

| difference from N=1 | total | vs floor |
|---|---|---|
| twin | +0.0023 | 1× (this *is* the floor) |
| N=4 | −0.0952 | ~40× |

Read those multiples as an order-of-magnitude yardstick, not a significance
test: the floor is a single pair of runs, so "~40×" moves between 41 and 43
depending on how the inputs are rounded.

**Both approximations are far outside the floor.** At 12 % and 9 % of a
−0.766 W m⁻² signal, neither is a free lunch; if the ERFari number is the
point of the run, `exact` is the only mode that earns the name.


## Practical guidance

- Diagnosing ERFari as a headline number: use `aerosol_free_interval: 1`
  and pay the +64 %.
- Long runs where ERFari is a secondary output and a ~10 % bias is
  tolerable: N=4 is defensible *because it does not touch the simulation* —
  every other field remains reference-quality.
- The untested middle ground is N=2, which should halve the extrapolation
  gap for a derived ~+30 %. Check it in the offline harness before spending
  cluster time; jax-gcm#648 adds it as an arm.

See jax-gcm#583 (the diagnostic), #630 (the sampling experiment), #648
(the unresolved magnitude) and #647 (the CMOR hazard).
