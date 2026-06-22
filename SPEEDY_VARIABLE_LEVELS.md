# Generalizing SPEEDY physics to an arbitrary number of vertical levels

Status: implemented and verified for `nlev ∈ {7, 8, 16, 30}` (see "Verification").

## 1. Goal and summary

jcm's SPEEDY physics port was restricted to exactly 7 or 8 vertical levels:
`jcm/physics/speedy/physical_constants.py` defined the σ half-level boundaries
only as a hand-tuned lookup table keyed by 7 and 8, and
`speedy_coords.get_speedy_coords` / `compute_speedy_vertical_coords` raised
`ValueError` for any other level count.

The key finding of this work is that **the physics schemes themselves were
already written generically** — every scheme derives `kx = shape[0]` at runtime
and indexes relative to it (`kx-1`, `kx-2`, `2:kx`, `jnp.arange(2, kx-3)`,
`dhs`, `fsg`, `hsg`, `wvi`, `grdsig`, `grdscp` arrays sized to `kx`). The
`SpeedyCoords` struct and all `PhysicsData` sub-structs are parameterized by
`num_levels`/`nodal_shape` and allocate no fixed-size vertical arrays. So the
**only** hard blocker to arbitrary `nlev` is the σ-boundary table lookup.

The fix is therefore small and surgical:

1. Add a **generalized σ half-level generator** (`compute_sigma_boundaries`)
   that returns the hand-tuned SPEEDY tables exactly for `nlev ∈ {7, 8}` and a
   principled stretched profile for any other `nlev`.
2. Route `get_speedy_coords`, `compute_speedy_vertical_coords` and
   `SpeedyCoords.single_column_coords` through it, removing the `ValueError`
   guards.

No per-scheme physics code needed changing (Phase D confirmed by running the
schemes end-to-end at `nlev = 16` and `30`). Section 4 documents the one
structural assumption that is *retained by design* (top-2-levels = stratosphere)
and contrasts it with SpeedyWeather's approach.

## 2. The σ-level generation formula

### What SPEEDY and SpeedyWeather.jl do

* **Fortran SPEEDY** uses hand-tuned, non-analytic σ half-level tables. jcm's
  existing tables are:
  * `nlev=7`: `[0, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]`
  * `nlev=8`: `[0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]`
  Both are stretched: fine near the surface (small Δσ at σ→1), fine near the
  model top (small Δσ at σ→0), coarse in the mid-troposphere.

* **SpeedyWeather.jl** (`src/dynamics/vertical_coordinates.jl`,
  `default_sigma_coordinates(nlayers)`) defaults to **equidistant**
  `σ_half = range(0, 1, nlayers+1)`, but ships a commented-out **Frierson
  (2006)** stretch that it documents as giving "higher resolution in surface
  boundary layer and in stratosphere":

  ```julia
  z = collect(range(1, 0, nlayers+1))
  σ_half = @. exp(-5*(0.05*z + 0.95*z^3))
  σ_half[1] = 0
  ```

### What we adopted and why

We do **not** adopt SpeedyWeather's *equidistant* default: equidistant spacing
puts almost no resolution in the boundary layer or near the model top, which is
exactly where SPEEDY's physics concentrates structure (the top two levels act as
the stratosphere; the surface flux / PBL schemes need a thin lowest layer). The
hand-tuned 7/8 tables are strongly stretched, and an equidistant profile would
be a poor and surprising default for a SPEEDY-physics column.

Instead we adopt SpeedyWeather's **Frierson (2006) cubic stretch** as the
generator for `nlev ∉ {7, 8}`, because it is the spacing SpeedyWeather itself
offers precisely "for higher resolution in the surface boundary layer and
stratosphere" — i.e. it reproduces the *qualitative* shape of SPEEDY's tables:

```
z      = linspace(1, 0, nlev+1)          # 1 at the surface end, 0 at the top
σ_half = exp(-5 * (0.05*z + 0.95*z**3))  # Frierson (2006) stretch
σ_half[0]  = 0.0                          # enforce exact TOA
σ_half[-1] = 1.0                          # enforce exact surface
```

This is strictly increasing for all `nlev ≥ 1`, satisfies `σ_half[0]=0`,
`σ_half[-1]=1`, and is qualitatively close to the SPEEDY tables (e.g. at
`nlev=8` it gives `[0, 0.033, 0.112, 0.268, 0.487, 0.709, 0.872, 0.960, 1]`
vs the table `[0, 0.05, 0.14, 0.26, 0.42, 0.60, 0.77, 0.90, 1]`).

**The 7- and 8-level cases continue to use the exact hand-tuned tables**, so all
existing behaviour and regression tests are bit-for-bit unchanged. The Frierson
generator only activates for other level counts. The numerical difference
between the Frierson stretch and the hand-tuned tables at 7/8 is real (tens of a
σ-unit mid-column) and is the reason we keep the tables as exact special cases
rather than replacing them.

## 3. Files changed

| File | Change |
|---|---|
| `jcm/physics/speedy/physical_constants.py` | Add `compute_sigma_boundaries(nlev)`: returns the hand-tuned `SIGMA_LAYER_BOUNDARIES` table for `nlev ∈ {7,8}`, else the Frierson (2006) cubic-stretch generator. Keep the table dict for backward compatibility and as the exact 7/8 special cases. |
| `jcm/physics/speedy/speedy_coords.py` | `get_speedy_coords` and `compute_speedy_vertical_coords` now call `compute_sigma_boundaries(nlev)` and accept any `nlev ≥ 2` (drop the `ValueError`/table-membership guards). Docstrings updated. |

No physics-scheme file required modification. The schemes in
`convection/speedy_convection.py`, `clouds/speedy_condensation.py`,
`clouds/speedy_humidity.py`, `radiation/speedy_shortwave.py`,
`radiation/speedy_longwave.py`, `vertical_diffusion/speedy_vdiff.py`,
`surface/speedy_surface_flux.py`, and `forcing/speedy_forcing.py` already derive
all vertical structure from `kx = shape[0]` and the σ-coordinate arrays.

## 4. Retained design assumption: top-2-levels = stratosphere

Several SPEEDY schemes treat the **top two model levels (indices 0 and 1) as
the stratosphere**, with distinct physics:

* `radiation/speedy_shortwave.py`: ozone absorption is applied only at `k=0`
  (`ozupp`) and `k=1` (`ozone`); the troposphere is `k = 2:kx`. The
  stratospheric correction uses `dhs[0] + dhs[1]`.
* `radiation/speedy_longwave.py`: levels 0,1 get isothermal/mean-temperature
  blackbody treatment and the `corlw1`/`corlw2` stratospheric cooling
  corrections use `dhs[0]`, `dhs[1]`.
* `vertical_diffusion/speedy_vdiff.py` and the `clouds` RH search skip the top
  two levels (`jnp.arange(2, kx-2)`, `rh[2:kx-2]`).

This is hardcoded as **level indices 0 and 1**, which means "the top two
levels" for *any* `nlev` — the assumption travels with the grid and remains
self-consistent. This matches Fortran SPEEDY, which always treats its top two
σ-levels as the stratosphere regardless of the total count. It is therefore
retained deliberately and is correct for arbitrary `nlev`, but note:

* It is a **fixed-count** stratosphere (always 2 levels), not a σ- or
  temperature-thresholded one. At large `nlev` (e.g. 30) only the top two of the
  many levels carry ozone / get the stratospheric correction, so the modelled
  stratosphere is physically thin. This is faithful to SPEEDY but is *not* how
  SpeedyWeather.jl does it.
* **SpeedyWeather.jl by contrast identifies the stratosphere by physical
  threshold**, not by index: shortwave ozone uses a σ-distribution
  `50·max(0, 1/5 − σ)` (ozone where `σ < 0.2`); longwave `UniformCooling`
  diagnoses the stratosphere where `T < 207.5 K`; the PBL top is found by a bulk
  Richardson-number threshold; convective cloud top is the level of zero
  buoyancy. These are more physical at high `nlev` but are a *different scheme*,
  not a generalization of SPEEDY's formulae.

Migrating SPEEDY's fixed-2 stratosphere to a σ-threshold would change results at
7/8 levels and is out of scope here; it is the natural next step if a thicker,
nlev-scaled stratosphere is desired. This is the main physical caveat of the
current generalization and is flagged as uncertain/incomplete.

## 5. Verification

See the run log in the task report. Summary:

* `get_speedy_coords(layers=L)` and a short forward integration of
  `speedy_physics()` through `Model` run with finite outputs and correct shapes
  for `L ∈ {8, 16, 30}`.
* `speedy_terms_test.py` (which derives `nlev` from `coords.nodal_shape[0]`)
  passes at `L = 8` (unchanged) and, parameterized, at `L = 16` and `30`.
* 7/8-level results are bit-for-bit unchanged because those cases still use the
  hand-tuned tables.
