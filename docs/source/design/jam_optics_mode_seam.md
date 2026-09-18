# JAM aerosol optics: the per-mode backend seam

`JamOpticsTerm` turns the JAM modal population into per-band optical depth,
single-scattering albedo and asymmetry. Most of what it does is bookkeeping
that every Mie pathway needs identically — summing species volumes, adding
hygroscopic water, gating empty modes, weighting SSA and asymmetry by
extinction, apportioning the result across species for the AeroCom
diagnostics, and reducing to the 550 nm column observables. Only one step is
actually about optics: *given this mode's lognormal and refractive index at
this wavelength, how much extinction, scattering and forward-scattering does
it carry?*

That one step is a hook. `JamOpticsTerm._mode_optics` answers it, and an
alternative Mie pathway is a subclass that overrides it.

## The seam

Three overridable methods, all on `jcm/physics/aerosol/jam/optics/optics_term.py`:

| method | default | a backend overrides it to… |
|---|---|---|
| `_mode_optics(inputs)` | 8-node Gauss–Hermite quadrature over the Bohren–Huffman LUT | compute the mode-integrated optics its own way |
| `_map_bands(fn, lam, ri)` | `jax.vmap` over the band axis | use `jax.lax.map` when its per-band intermediates are large |
| `_build_mie_lut()` | `default_mie_lut()` | return `None` and skip the ~4 s table build it never reads |

`_mode_optics` receives a `ModeOpticsInputs` record and returns
`(tau, tau_scat, tau_scat_g)` — extinction, scattering, and
scattering-weighted-asymmetry optical depths for that mode alone, **ungated**.

### Why optical depths and not `(k_ext, ssa, g)`

Backends disagree about normalisation, and the seam refuses to pick a winner.
A quadrature over Mie efficiencies naturally produces an extinction per
particle **cross-section**, so it wants the mode's column number per area; an
emulator of the mode-integrated integral naturally produces an extinction per
unit particle **volume**, so it wants a volume per area. `ModeOpticsInputs`
supplies both normalisations — `num_per_area`, and `vol_total` with the
`col_factor = air_density · dz` that converts it — and each backend uses the
one it is expressed in. For the same reason it carries both radii and the
dry/wet volume split rather than the minimum set: every quantity a backend
would otherwise reconstruct from the others goes through a cube root or a
division that is singular on an empty mode, and the record holds finite
values for all of them. With `⟨·⟩` the number-weighted lognormal moment and
`n_A` the column number per area, both routes reduce to `π·n_A·⟨Q_e r²⟩`, so
the two are directly comparable rather than merely similar.

Returning an optical depth also means the base class never has to divide by a
number or a volume that may be zero, which is where the reverse-mode NaNs in
this term have historically come from.

### What a backend does *not* have to reproduce

The dry-mass gate (`vol_dry > 1e-24`, which keeps cold-start ringing out of
the lid layers), the `[0,1]`/`[-1,1]` clamps on the weighted SSA and
asymmetry, the `_AER_RAD_PMIN` mask, the `_MAX_LAYER_TAU` cap, the AeroCom
per-species and per-mode apportionment, and the 550 nm/Ångström column
diagnostics all stay with the base class and apply to any backend. A backend
that answered the optical question but forked the term would have to carry
duplicates of all of it, and they would drift.

### The contract

An override must be:

* **Finite everywhere, in value *and* in derivative.** The gate and the cap
  downstream select and bound *values*; neither can repair a `NaN`, an `inf`
  or a singular derivative produced inside the hook, because reverse mode
  multiplies a zero cotangent into whatever came back. Empty modes, a single
  species, zero aerosol water and a degenerate wet radius all occur in a real
  column. Build reciprocals and fractional powers on a substituted argument —
  the `jnp.where`-before-divide pattern used throughout the term. A cube root
  of a species volume fraction (the natural way to get a core *radius* ratio
  from a core *volume* fraction) is singular at zero and must be guarded even
  though the mode it applies to may look statically safe.
* **Non-negative in extinction, with scattering no larger**, so the SSA the
  caller derives stays inside `[0,1]`: RRTMGP's two-stream solver returns
  `NaN` outside it.
* **Traceable** — no Python branch on a traced value. Branching on
  `inputs.mode` is fine; the population spec is static config, so a backend
  can select a different network per mode at trace time with no `lax.cond`.

## Attaching a backend

Backends are attached in Python, on an assembled package, by category:

```python
from jcm.physics.echam.echam_terms import echam_physics
from some_optics_package import SomeOpticsTerm     # subclasses JamOpticsTerm

physics = echam_physics(aerosol_module="jam").replace(
    "aerosol_optics", SomeOpticsTerm(optics_diagnostics=True))
model = jcm.Model(coords=coords, terrain=terrain, physics=physics)
```

`ComposablePhysics.replace` inserts at the position of the term it removes, so
the JAM chain's validated ordering — optics after the microphysics core that
writes `_jam_state` — is preserved, and `ComposablePhysics` raises if it is
not.

It also hands the displaced term to the replacement's
`adopt_runtime_configuration`, which matters more than it looks. Some settings
are applied by the factory *after* the package is composed, because they come
from a sibling term rather than from a constructor argument: the optics term's
radiation cadence is read off the radiation term, so a replacement has no way
to know it. Without the handover a swapped-in optics term would run with the
gate unset and recompute all 30 bands on every step rather than every eighth —
correct, roughly 8x the optics cost, and completely silent. A backend that
holds extra post-compose state of its own overrides the same hook.

There is deliberately **no config key, registry or entry point** for
selecting a backend. Nothing in this repository implements one, so a string
selector would have nothing to resolve to; and a user who has installed and
imported a third-party optics package is already writing Python. If a backend
is ever vendored in-tree, it gets a config group entry then, like any other
in-repo scheme.

## Notes for backend authors

Three properties are easy to assume and wrong, and none of them is the seam's
to enforce:

* **Which integral the backend actually computes.** An emulator trained on a
  lognormal truncated to a fixed fraction of the *number* distribution does
  not predict the full-distribution moment ratio, because the `r³` volume
  weight shifts the density by `3 ln σ_g` and so loses far more of the volume
  than of the number — at `σ_g = 1.8` a 99.9 %-of-number interval retains
  93.7 % of the volume. Multiplying such a prediction by JAM's full third
  moment (`vol_total` here *is* the full moment) leaves a σ_g-dependent bias
  of several percent. State the convention; do not assume it matches.
* **The representable range of `g`.** A backend whose asymmetry comes from a
  sigmoid cannot produce `g < 0`, so it cannot represent back-scattering,
  which the default path (clamped to `[-1, 1]`) can.
* **The per-species apportionment premise.** The AeroCom decomposition the
  base class computes weights extinction by species volume fraction and
  absorption by `V_s k_s / Σ V_s k_s`. Those weights are *exact* under the
  volume mixing rule the default path uses — `Σ V_s k_s` is then literally
  `V_tot k_eff`. A backend that does not volume-mix, a core-shell treatment
  for instance, still gets a defensible apportionment (and the only one an
  internally-mixed model can report), but not an exact one.

## One thing that is not free

`_map_bands` is the only hook whose two forms are not bit-identical: XLA
associates a batched and a scanned band axis differently, worth ≈2 ulp in
float32. Switching band maps is a memory decision, not a costless one for a
configuration whose answers are pinned bitwise.
