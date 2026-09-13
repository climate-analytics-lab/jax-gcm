# JAM optics backends: Mie LUT and NeuralMie

`JamOpticsTerm` supports two Mie pathways, selected with
`jam_optics_backend` (`echam_physics(...)`) or the
`physics=echam-jam-neuralmie` preset. The default is `mie_lut`, so enabling
NeuralMie is opt-in and default answers are untouched.

## The two backends compute the same quantity

This is worth stating precisely, because it is what makes a direct comparison
meaningful rather than merely suggestive. Write `⟨·⟩` for the number-weighted
lognormal moment and `n_A` for the column number per area.

The **LUT** path integrates per-particle Bohren–Huffman efficiencies over the
mode's lognormal with an 8-node Gauss–Hermite quadrature in `ln r`:

```
τ = n_A · sec · π r_g²,      sec = ⟨Qe·(r/r_g)²⟩
  = π · n_A · ⟨Qe r²⟩
```

The **NeuralMie** path predicts the mode-integrated extinction cross-section
per unit particle volume, `ke_ρ = 0.75·⟨Qe r²⟩/⟨r³⟩` [1/m], and multiplies by
a particle volume per area:

```
τ = ke_ρ · V_A,      V_A = n_A·(4/3)π⟨r³⟩
  = π · n_A · ⟨Qe r²⟩
```

The two are therefore the same integral reached by different routes. The LUT
discretises it with 8 quadrature nodes over a 64×24×24 trilinear table;
NeuralMie reproduces the 1024-point quadrature it was trained on.

## Accuracy

The emulator sits within **0.18%** (max; 0.023% median) of an exact
1024-point quadrature on extinction, with `ssa` within 9.2e-4 and `g` within
1.4e-3, over the range modal schemes occupy (λ 200 nm–30 µm, r_g 10 nm–2 µm,
σ_g 1.2–2.0, m_r 1.3–2.6, m_i 1e-8–1). That was measured against a
Bohren–Huffman kernel independent of the TAMie code the network was trained
on, so it is not self-referential.

For scale, the LUT path's own interpolation tolerance in `mie_test.py` is 15%
on `q_ext` and 0.05 on `ssa` off-grid. Where the two backends disagree, the
LUT is the less accurate of the two.

The accuracy figures cap the size parameter at 100 to keep the reference
affordable, so they do not characterise the large-`x` corner of the emulator's
domain (which reaches ~1570).

## Two physics changes come with the backend

It is not a pure numerical swap, and both changes are deliberate.

### Core-shell mixing for BC-bearing modes

Modes carrying black carbon (accumulation, coarse and primary carbon in MAM4)
use the core-shell network: BC as a concentric core, every other species plus
aerosol water as the coating, with

```
f = clip((V_bc/V_tot)^(1/3), 0, 0.98)
```

`f` is a *radius* ratio, hence the cube root. Where the coating vanishes the
shell index falls back to the core index, which is the exact homogeneous limit
and keeps both indices inside the trained domain. Mode membership is static
Python config, so the sphere/core-shell choice is made at trace time with no
`lax.cond`.

**The direction of the effect is easy to get backwards.** The familiar
"coating enhances BC absorption 1.2–2×" compares coated BC to *bare* BC. The
comparison here is against the volume-average-of-refractive-index rule, and it
runs the other way: averaging the index smears BC's large imaginary part over
the whole particle, which over-absorbs relative to confining it to a core.
Verified against exact TAMie + 1024-point quadrature at 550 nm (r_g = 100 nm,
σ_g = 1.8, BC core in a sulfate shell), single-scattering albedo:

| V_bc/V | core-shell | volume-mixed |
|---|---|---|
| 0.008 | 0.9676 | 0.9661 |
| 0.064 | 0.8182 | 0.7887 |
| 0.125 | 0.7001 | 0.6710 |
| 0.216 | 0.5736 | 0.5698 |
| 0.512 | 0.4323 | 0.4585 |

So BC-bearing modes become **less** absorbing, with the sign reversing only
near `V_bc/V ≈ 0.5`, far above realistic loadings. This moves ERFari and is
not retuned for.

Applying core-shell to the primary-carbon mode is the weakest part of the
choice: `_PCARBON_SPECIES` is `soluble=False`, "hydrophobic until it ages",
which is exactly where a concentric coated sphere is least physical. It is
also the mode most likely to reach the `f = 0.98` cap. Restricting core-shell
to soluble modes is the alternative; see jax-gcm#791.

### Number-free hygroscopic water volume

On the NeuralMie path the water volume is

```python
v_water = vol_dry * ((r_wet / r_dry) ** 3 - 1.0)
```

rather than `N·(4/3)π·(r_wet³ − r_dry³)`. `PlaceholderMicrophysics` defines
`r_dry` from `V = N·(π/6)·Dg³·exp(4.5 ln²σ)`, so `N·(4/3)π·r_dry³` is the dry
volume *divided by* `exp(4.5 ln²σ)` — meaning the original form understates
the water volume by 2.70× (Aitken, σ=1.6) to 4.73× (σ=1.8 modes). The form
above is exactly consistent with the third moment the term already
accumulates, and it removes the `aer.number` dependence from the optics
entirely, which also removes the cold-start ringing pathway the `vol_dry`
mass gate exists to catch.

This is fixed **only** on the NeuralMie path here. Correcting it on the
default path changes default answers and is tracked separately in
jax-gcm#790.

## Diagnostics

The backend branch lives inside the per-mode block of `one_band`, not at
`_band_optics`, so the AeroCom diagnostic pass
(`_optics_diagnostics_fields`, at 355/440/550/670/865 nm) switches with it.
Branching one level up would let the published `od550aer` describe a
different optical model from the one radiation used, and the existing closure
tests would not catch it because both sides would remain internally
consistent.

One caveat follows from core-shell mixing: the per-species apportionment is
justified by volume-mixing into a single effective index, which no longer
holds exactly once extinction is not a function of one mixed `m`. It remains
a defensible apportionment — and the only one an internally-mixed model can
report — but it is no longer exact. See
`aerosol_optics_diagnostics.md`.

## Implementation notes

- **Weights** are two `.npz` files under `jcm/data/neuralmie/` (218 KB), held
  in `nnx.Param` and read with `.get_value()`, so they stay reachable by
  `jax.grad` rather than becoming static aux data.
- **`neuralmie.py` is a leaf module**, importing only the standard library,
  `numpy` and `jax`. It is vendored from
  [`reflective-org/neuralmie-jax`](https://github.com/reflective-org/neuralmie-jax),
  which holds the NumPy Mie ground truth and the validation suite behind the
  accuracy figures above. Resyncing is a file copy plus the weights path in
  `default_weights`; a test enforces the import restriction so that cannot rot.
- **The Rayleigh branch** (analytic small-particle limit, `g` exactly 0) is
  selected with `jnp.where` over both evaluated arms. The size parameter is
  clamped up to the switch boundary before it reaches the network: the
  networks are untrained below it, and unclamped the core-shell network's raw
  output reaches +2142 there, so `exp()` would overflow across ~1.4% of the
  training box and poison the gradients of the selecting `where`. The clamp is
  bitwise identity on every non-Rayleigh point, so it cannot perturb a
  returned value. In practice the branch is unreachable for realistic aerosol
  in the shortwave — at 550 nm it fires only below r_g ≈ 1.3 nm — and becomes
  reachable only in the longwave.
- **Domain clipping** uses a straight-through estimator, so an out-of-domain
  input reports the emulator's sensitivity at the boundary rather than an
  exactly-zero gradient (cf. jax-gcm#664).
- The `mie_lut` table build (~4 s) is skipped entirely when the NeuralMie
  backend is selected.

## References

Geiss, A. and Ma, P.-L.: *NeuralMie (v1.0): An Aerosol Optics Emulator*,
Geoscientific Model Development, doi:10.5194/gmd-2024-30.

Upstream implementation: [`pnnl/NEURALMIE`](https://github.com/pnnl/NEURALMIE),
BSD-2-Clause, Copyright 2024 Battelle Memorial Institute.
