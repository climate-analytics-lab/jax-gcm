# Sampling the aerosol-free radiation call (AeroCom ERFari)

AeroCom's aerosol effective radiative forcing from aerosol–radiation
interactions (ERFari) is diagnosed from the difference between the normal
top-of-atmosphere fluxes and a set of *aerosol-free* companions
(`rsutnoa`, `rlutnoa` and their clear-sky variants). Producing those
companions honestly means solving the radiative transfer **twice** per
radiation step — once with the aerosol optics and once with them zeroed.

Radiation is ~87 % of a T63L47 ECHAM+JAM timestep, so that second solve is
not a rounding error: it costs **+64 %** wall-clock. This document records
what we measured when trying to make it cheaper, and why the API ended up
shaped the way it is.

## The knob

```yaml
aerosol_free: off | exact | paired | alternating
aerosol_free_interval: 4        # `paired` only
```

| mode | what it does | s/sim-day | cost |
|---|---|---|---|
| `off` | no `*noa` fluxes at all (default) | 30.9 | +0 % |
| `exact` | companion solve every radiation step | 50.8 | +64 % |
| `paired` | companion every Nth step, effect held in between | 36.2 | +17 % at N=4 |
| `alternating` | no companion; alternate the single solve | 30.9 | +0 % |

Measured at T63L47 on an A100 with semi-Lagrangian transport, steady-state
chunks only — chunk 0 *and* chunk 1 are compile-contaminated and give
figures several times too high. Earlier revisions of this work quoted
+55 % and +7 %; those were measured on the now-removed Eulerian dycore and
should be ignored.

`aerosol_free_interval` is the parameter of `paired` and is rejected with
any other mode, so neither "paired but no N" nor "N set on a mode that
ignores it" can reach a run silently. `N=1` is likewise rejected: it is
`exact` under a second name, and one scheme with two spellings invites
people to compare them as though they were different.

## The two approximations differ in kind, not just in size

This distinction matters more than either error bar.

**`paired` approximates only the diagnostic.** The model always sees
aerosol, so the simulation is bit-identical to `exact`; only the reported
`*noa` fluxes are extrapolated. Between companion solves the aerosol
effect is held as a **fraction** of the all-sky flux and re-applied to the
fresh all-sky value.

That fraction is not a stylistic choice. Holding the effect as an
*absolute* W m⁻², which was the first implementation, works in the longwave
and fails in the shortwave: the SW effect tracks the solar cycle, so on the
night side the scheme subtracts a stale daytime effect from a zero flux —
reconstructing a non-zero aerosol-free flux for a dark column, which is not
merely inaccurate but impossible. The first version matched the reference
*exactly* in the LW and was 0.077 W m⁻² wrong in the SW; the perfect half is
what hid the broken half for a whole year-long run.

Note that 0.077 W m⁻² is **not** comparable with the 0.095 W m⁻² quoted
below for the fraction hold: they come from different runs on different
dynamical cores. The fraction hold was adopted because it makes the dark
column reconstruct to zero by construction — a property now pinned by a
regression test — not because it scored a smaller annual mean.

**`alternating` changes the simulation.** It steals every other radiation
call rather than adding one, so the model feels aerosol-free heating half
the time and the aerosol direct effect enters with a 50 % duty cycle. Every
output is affected, not just the `*noa` fluxes. Its ERFari therefore cannot
be compared against a reference without an ensemble, because the two runs
have genuinely diverged.

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
| `exact` | −0.8202 | +0.0547 | −0.7655 |
| `exact` (twin) | −0.8177 | +0.0545 | −0.7633 |
| `paired` N=4 | −0.9150 | +0.0544 | −0.8607 |
| `alternating` | −0.7562 | +0.0575 | −0.6987 |

| difference from `exact` | total | vs floor |
|---|---|---|
| twin | +0.0023 | 1× (this *is* the floor) |
| `paired` N=4 | −0.0952 | ~40× |
| `alternating` | +0.0668 | ~30× |

Read those multiples as an order-of-magnitude yardstick, not a significance
test: the floor is a single pair of runs, so "~40×" moves between 41 and 43
depending on how the inputs are rounded.

**Both approximations are far outside the floor.** At 12 % and 9 % of a
−0.766 W m⁻² signal, neither is a free lunch; if the ERFari number is the
point of the run, `exact` is the only mode that earns the name.

One caveat on `alternating`'s entry: because it perturbs the physics, its
difference from the reference is not cleanly separable from trajectory
divergence even under nudging, so its 0.0668 should be read as an upper
bound on a scheme error rather than a measurement of one. `paired`'s entry
carries no such caveat.

## The tension that turned out to be three bugs

`paired` leaves the physics bit-identical, so its error should be purely
the extrapolation between companions — and yet it measured *larger* than
the physics-perturbing `alternating` scheme's. Two observations sharpened
the puzzle:

- An offline, un-nudged, same-node test found `paired` N=4
  **bit-identical** to `exact` over 6 days, with an ERFari difference of
  exactly zero.
- That test ran in a window where SW ERFari was only −0.11 W m⁻²; here it
  is −0.82. The same *relative* extrapolation error would be ~7× larger in
  absolute terms, which is the right order of magnitude but not a
  demonstration.

Chasing that mismatch found **three real bugs in the hold**, all of which
pushed the error in the measured direction:

1. **A dark companion erased the held fraction.** The fraction used to be
   re-derived from the two flux slots on the carry each step. That
   round-trip is exact only while the all-sky flux is non-zero, so a
   companion landing on a night-side column returned "no aerosol effect",
   wrote `noa == allsky` into the carry, and re-derived zero from it for
   every remaining skip step — *including after sunrise*.
2. **A twilight companion could fabricate a huge effect.** Near the
   terminator the TOA upward SW can be ~10⁻³ W m⁻², where the aerosol
   slant path dominates and the ratio approaches 1. Real for that flux,
   catastrophic re-applied to a sunlit one: a held ratio of 0.9 turns a
   +1.4 W m⁻² effect into +150.
3. **The division NaN'd reverse-mode gradients.** A single `where` around
   a division leaves a NaN primal in the masked branch whose VJP returns
   0/0 — the jax-gcm#558/#559 pattern. Dark columns always exist, so every
   gradient through `paired` was poisoned.

The fraction is now carried explicitly on `RadiationData` rather than
re-derived, is only updated from a companion whose flux is large enough for
the ratio to mean anything, and uses a safe denominator. Each fix has a
regression test verified to fail against the old form.

**Consequently the −0.095 W m⁻² above is stale** — a plausible upper bound
rather than a measurement of the current scheme. A further confound
remains: three of the four runs resumed from day-20 checkpoints while the
twin ran fresh. Re-measurement with all runs started fresh, plus an N=2
arm, is jax-gcm#648.

What survives: `alternating` is genuinely outside the floor, and `paired`
is structurally the safer approximation because it does not touch the
simulation. What does not survive: any claim about `paired`'s numerical
accuracy, or the ranking of the two.

## Practical guidance

- Diagnosing ERFari as a headline number: use `exact` and pay the +64 %.
- Long runs where ERFari is a secondary output and a ~10 % bias is
  tolerable: `paired` is defensible *because it does not touch the
  simulation* — every other field remains reference-quality.
- `alternating` is hard to recommend: it perturbs the model for a saving
  over `paired` of ~17 percentage points.
- The untested middle ground is `paired` with N=2, which halves the
  extrapolation gap for roughly +30 %. Check it in the offline harness
  before spending cluster time.

Because none of this is recoverable from the saved fields, both approximate
modes emit a startup **warning** quoting their measured error, and the
chosen mode is stamped into every output file as the global attribute
`jcm_prov_aerosol_free` (`exact`, `paired:N=4`, `alternating`). A log line
alone is gone by the time anyone reads the netCDF; an approximate ERFari
would otherwise be indistinguishable from the real one three months later.

The attribute is absent for `aerosol_free: off` — which is also how a
consumer should tell that the all-zero `*noa` fields in such a file are
placeholders rather than data (jax-gcm#647).

See jax-gcm#583 (the diagnostic), #630 (the sampling experiment), #648
(the unresolved ranking) and #647 (the CMOR hazard).
