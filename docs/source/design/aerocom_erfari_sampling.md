# Sampling the aerosol-free radiation call (AeroCom ERFari)

AeroCom's aerosol effective radiative forcing from aerosol–radiation
interactions (ERFari) is diagnosed from the difference between the normal
top-of-atmosphere fluxes and a set of *aerosol-free* companions
(`rsutnoa`, `rlutnoa` and their clear-sky variants). Those companions are
the all-sky and clear-sky problems solved once more with no aerosol optics
mixed in. Since jax-rrtmgp 0.7.0 they are solved **inside the one radiation
call**, in the same g-point loop as the fluxes with aerosol
(`compute_heating_rate(aerosol_free_fluxes=True)`): the gas optics, Planck
sources and McICA cloud field are shared and only the cell properties and
the transport are repeated, so the companion costs a fraction of a solve.
On an A100 at T63L47 (`t63-echam-jam-aerocom`) the radiation call goes from
142.7 ms without the diagnostic to 211.7 ms with it
(+48 %); as a second solve with the aerosol optics zeroed, which is
what it was until then, the same call cost 268.3 ms (+88 %).
Because the twins share the cloud draw by construction, the difference to
the fluxes with aerosol is the instantaneous aerosol effect alone.

When the companion was a second full solve it cost **+64 %** of the whole
run, which is what the sampling knob below was built to trade against. The
knob stays (the companion still costs transports) and the numbers in the
rest of this page are the ones that shaped it; they predate the shared
solve.

## The knob

```yaml
aerosol_free_interval: 1      # null = off, 1 = exact, N > 1 = subsampled
```

| setting | what it does | s/sim-day (second solve) | ms / radiation call (shared solve) |
|---|---|---|---|
| `null` | no `*noa` fluxes at all (default) | 30.9 | 142.7 |
| `1` | companion every radiation step — exact | 50.8 (+64 %) | 211.7 (+48 %) |
| `N > 1` | companion every Nth step, effect held between | 36.2 at N=4 (+17 %) | 160 on average at N=4 (every 4th call pays 211.7) |

The s/sim-day column is the second-solve implementation the sampling
experiment was run on; the per-call column is the shared solve, measured
on an A100 with `tools/profile_terms.py --preset t63-echam-jam-aerocom`.

This one integer provides a fine-tuned control which is monotonic in 
both directions: cost falls and error grows with N. 
The costs were measured at T63L47 on an A100 with semi-Lagrangian transport,
steady-state chunks only. The model always sees aerosol; only the reported
`*noa` fluxes are extrapolated between calls. With the companion solved
inside the radiation call the simulation is no longer bit-identical at
every N, as it was when the companion was a second solve: the fluxes with
aerosol come out of a larger program that XLA fuses differently and differ
from a run without the diagnostic at float32 rounding (3e-4 W/m² on a
shortwave flux of ~500 W/m², 1e-7 K/s in the heating, on one CPU column),
and with N > 1 the companion calls run that program while the others run
the plain one. An `optimization_barrier` around the twins does not restore
bit-identity (XLA's CPU backend drops barriers before fusion), so this is
accepted as a rounding-level perturbation of the trajectory, of the same
kind as any change of the compiled radiation program.

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
