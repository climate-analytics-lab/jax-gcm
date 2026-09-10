# Per-species and per-mode aerosol optics diagnostics

AeroCom asks models to report aerosol optical depth broken down by
component (`od550so4`, `od550bc`, `od550oa`, `od550dust`, `od550ss`,
`od550aerh2o`) and absorption likewise (`abs550bc`, `abs550dust`, ...).
JAM is an **internally mixed** modal scheme, and that makes the request
harder to answer than it looks. This note records what the diagnostic
actually computes and why, so the numbers are not over-read.

## The problem: internal mixing has no per-species extinction

Within a mode, `JamOpticsTerm` volume-mixes every species into **one**
effective refractive index before the Mie call:

```
m = (Σ_s V_s n_s) / ΣV_s  +  i (Σ_s V_s k_s) / ΣV_s
```

and then evaluates a single Mie efficiency `Q_ext(x, m)` for the whole
particle. There is only one particle. Extinction is a property of that
mixed particle, not a sum of per-species extinctions — a sulfate coating
on a soot core changes the soot's absorption, which is the entire point
of treating the mixture internally. So there is no per-species extinction
sitting inside the calculation waiting to be read out.

The honest alternatives were:

1. **Re-run Mie once per species**, each time pretending the mode
   contains only that species. This gives a well-defined *external
   mixture* answer, but it is a different optical model from the one the
   radiation actually used, the parts do **not** sum to the total, and it
   costs one extra Mie sweep per species (≈7× the optics).
2. **Apportion the mode's computed extinction** among its species. The
   parts sum to the total the radiation saw, at negligible cost, but the
   split is a bookkeeping convention rather than a measurement.

JAM does **(2)**. The reported fields are an *apportionment*, and they
are labelled as such throughout the code.

## The apportionment rule

Extinction and absorption are apportioned differently, because they are
driven by different parts of the refractive index:

- **Extinction** is split by **volume fraction**, `V_s / ΣV`. In the
  Rayleigh limit extinction is proportional to volume, so this is exact
  for the fine mode and a reasonable interpolation elsewhere.
- **Absorption** is split by the species' **contribution to the
  imaginary index**, `V_s k_s / Σ V_s k_s`. Absorption is driven by `k`,
  so a volume split would hand most of the absorption to whatever species
  is most abundant rather than to the one actually absorbing — soot is a
  small volume fraction and nearly all of the absorption.

Hygroscopic water is treated as a species (`od550aerh2o`), with its own
volume in the extinction split and its own `k` in the absorption split.
Both sets of fractions therefore sum to one over species-plus-water, so
**the components close on the total exactly** — this is enforced by
`test_species_apportionment_closes`, and it is the property that makes
the diagnostic safe to use in a budget.

The absorption weight is not an ad-hoc choice. Under the volume mixing rule,
`Σ_s V_s k_s` **is** `V_tot · k_eff` — the very effective imaginary index the
Mie call used. So `abs_s / abs_mode = V_s k_s / (V_tot k_eff)` is exactly the
linear decomposition of `k_eff` into per-species contributions, i.e. the
first-order attribution consistent with the model's own mixing rule.

Both rules reduce to the correct answer when a mode carries a single
species.

**Per-mode** fields (`od550_mode_acc`, ...) are a genuine decomposition,
not an apportionment: each mode gets its own Mie call, so its extinction
is separately computed and the modes simply add.

## Cross-check: ECHAM-HAM uses the same convention

This was verified against the ECHAM-HAM (HAMMOZ) Fortran rather than
assumed. `ham_rad_diag` in `mo_ham_rad.f90:1645-1972` resolves the same
tension the same way:

- It **apportions**; there is no per-species Mie call anywhere in HAM.
- Its comment at `mo_ham_rad.f90:1839-1841` states the convention outright:
  *"Split up according to compounds (based on volume average for optical
  thickness, additionally weighted with ni for absorption)"*.
- The weights, at `mo_ham_rad.f90:1926-1936`, are `V_s / ΣV` for
  `TAU_COMP_*` and `V_s·k_s / Σ(V_s·k_s)` for `ABS_COMP_*` — identical to
  the pair above, with aerosol water carried as its own species exactly as
  `od550aerh2o` is here.
- HAM likewise emits the component split at **550 nm only**.

So these fields are directly comparable to HAM's `TAU_COMP_*` /
`ABS_COMP_*`. Two differences worth knowing: HAM's Ångström diagnostic is
over 550/865 nm, so `ang550865aer` is published alongside AeroCom's
440-based `ang4487aer`; and HAM applies the volume-based apportionment even
when its radiation ran Maxwell-Garnett or Bruggeman mixing (`nradmix=2/3`),
which makes the split inconsistent with its own optics in those
configurations. JAM has only the volume rule, so that inconsistency cannot
arise here.

## Caveat: coating enhancement is credited to the absorber

`abs_mode` comes out of the full coated-particle Mie calculation, so any
lensing — a sulfate or water shell focusing light onto a soot core, which
can enhance absorption substantially — is real in the total and is then
apportioned by `V_s k_s`. Because soot's `k` dwarfs everything else
(BC ≈ 0.7 at 550 nm against ≈1e-9 for sulfate), **essentially all of that
enhancement is credited to black carbon**, not to the coating that caused
it. That is the physically sensible attribution and it matches HAM, but it
means `abs550_bc` is "BC including its coating enhancement", not "what bare
BC would absorb". Do not difference it against an external-mixture
calculation and call the residual an error.

## Diagnostic wavelengths, not radiation bands

The diagnostics are evaluated at 355, 440, 550, 670 and 865 nm rather
than at whichever band centres the radiation configuration happens to
provide. `refractive_index_at` interpolates in `log10(λ)`, so arbitrary
wavelengths cost nothing extra to set up, and it means `od550aer` is at
550 nm — not "the RRTMGP band nearest 550 nm", which moves if the band
configuration changes and is a single broad band under grey radiation.

`ang4487aer` is computed from the 440/865 nm pair. AeroCom's definition
is nominally 440/870 nm; 865 nm is used because it is a jax-rrtmgp band
centre, a 0.6 % difference in the lever arm. `ang550865aer` uses the
550/865 pair that ECHAM-HAM reports.

## What is deliberately *not* applied

The radiative optics carry two numerical guards — the `_AER_RAD_PMIN`
mask that zeroes aerosol above ~2 hPa, and the `_MAX_LAYER_TAU` cap.
Both exist because a heating rate divides absorbed flux by a near-zero
lid air mass; neither is physics. The diagnostics skip them, because the
diagnostic is meant to be the column a satellite retrieval would see. The
mass above 2 hPa is radiatively negligible, so the two agree in practice.

## Cost

The diagnostic is a second Mie sweep, so it is **off by default**
(`aerocom_optics: false`). When enabled it rides the same radiation gate
as the radiative optics — it is computed only on radiation-compute steps
and replayed from the scan carry in between — so its cost is
`len(_DIAG_WAVELENGTHS_NM) / n_sw_band` of the already-gated aerosol
optics, not a per-step Mie evaluation.

Everything except the 355 nm extinction profile is reduced to a column
integral inside the optics term rather than downstream. These fields live
in the `lax.scan` carry, and keeping `n_species × n_wavelength × nlev`
three-dimensional arrays alive there would cost hundreds of megabytes at
T63L47.

## Reading the output

Raw per-species fields are written as `od550_<species>` /
`abs550_<species>` using jcm's own species names (`so4`, `bc`, `poa`,
`soa`, `moa`, `du`, `ss`, `wat`). `tools/aerocom_cmor.py` groups them
into the protocol's components — in particular the three organic species
are **summed** into `od550oa`, which is what AeroCom asks for; the raw
split stays in the model output if a finer breakdown is wanted.
