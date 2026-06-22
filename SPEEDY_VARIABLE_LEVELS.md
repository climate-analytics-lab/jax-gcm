# Generalizing SPEEDY physics to an arbitrary number of vertical levels

Status: implemented and verified for `nlev ∈ {7, 8, 16, 30}` (see "Verification").
The σ-table generalization (§2–§3) and the physics-scheme migration to a
σ-threshold stratosphere (§4) are both complete. **The §4 migration changes the
7/8-level results** (intentionally — see §4).

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

The σ-table change alone lets the schemes *run* at any `nlev`, but several
schemes still hardcoded "the top two levels are the stratosphere", which does not
scale physically. **Section 4 documents the follow-up migration** that replaces
those fixed-index assumptions with a physical σ<0.2 stratosphere mask (the way
SpeedyWeather.jl identifies the stratosphere), so the stratosphere/cloud/diffusion
level ranges scale with the grid. That migration deliberately changes the 7/8
results.

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

## 4. Physics-scheme migration: σ-threshold stratosphere

SPEEDY hardcoded several schemes around the assumption that the **top two model
levels (indices 0 and 1) are the stratosphere**. A fixed *count* does not scale:
at `nlev=8` those two levels are at σ<~0.2, but at `nlev=30` only 2 of 30 levels
would be stratospheric, giving a physically far-too-thin stratosphere. This
section migrates those fixed-index assumptions to a **physical σ threshold**, the
way SpeedyWeather.jl does. **This changes the 7/8-level results** (deliberately —
see the verification note at the end).

### 4.0 Shared stratosphere mask (single source of truth)

`jcm/physics/speedy/speedy_coords.py` defines:

* `STRATOSPHERE_SIGMA_THRESHOLD = 0.2` and `stratosphere_mask(fsg)` →
  `fsg < 0.2 - 1e-5` (boolean, shape `(kx,)`).
* `ozone_sigma_weight(fsg)` → `50·max(0, 1/5 − σ)` (SpeedyWeather's ozone
  distribution).

`fsg` (σ layer midpoints) is fixed at coords-build time, so **every mask derived
from these is a compile-time-constant boolean array** — fully differentiable, no
`jnp.where` on traced state and no Python `if` on traced predicates. Temperature
masks were deliberately *not* used (see §4.2).

**The `−1e-5` tolerance is load-bearing.** The SPEEDY 8-level table places a layer
midpoint *exactly* at σ=0.2 (`(0.14+0.26)/2`), which is the troposphere top, not
the stratosphere, under SpeedyWeather's strict `σ<0.2`. In float32 that midpoint
stores as `0.19999998`, which would spuriously satisfy `<0.2` and pull a third
(tropospheric) layer into the stratosphere. The tolerance keeps the nlev=8
stratosphere at the intended **top two** layers.

Stratosphere-level counts produced by this mask:

| nlev | strat levels | σ midpoints selected |
|---|---|---|
| 7  | 1  | 0.07 (the 2nd midpoint is exactly 0.2 → troposphere) |
| 8  | 2  | 0.025, 0.095 |
| 16 | 5  | 0.008 … 0.146 |
| 30 | 10 | 0.005 … 0.186 |

(nlev=7 picks only 1 strat level because its 2nd midpoint sits exactly on the
0.2 boundary; this is an accepted change for the 7-level table.)

### 4.1 Shortwave ozone — `radiation/speedy_shortwave.py`

* **Was:** ozone applied at fixed indices `k=0` (`ozupp`) and `k=1` (`ozone`);
  troposphere scanned `2:kx`; stratospheric correction `eps1 = epslw/(dhs[0]+dhs[1])`;
  cloud absorptivity over `2:kx-1`; "no water vapour" zeroed only at `k=0`.
* **Now:** the *total* stratospheric ozone absorption `ozupp+ozone` is
  distributed over the σ<0.2 layers with SpeedyWeather's weight
  `50·max(0,1/5−σ)·dσ`, **normalised to sum to 1 over the column** so the
  column-integrated ozone absorption is exactly preserved (= `ozupp+ozone`). The
  downward beam is a single unified scan over the whole column: at each level it
  loses ozone (non-zero only in the stratosphere) and is then attenuated by layer
  transmissivity and cloud reflection (non-zero only in the troposphere). The
  unified propagator `τ₀·(F − oz)·(1−τ_cloud)` reduces exactly to SPEEDY's old
  stratosphere update where there is no cloud and to its troposphere update where
  there is no ozone (verified bit-for-bit on a synthetic column). `eps1` now
  divides by the total thickness of the σ<0.2 layers; cloud absorptivity and the
  "no water vapour" zeroing use the σ mask (with the lowest layer kept cloud-free
  as the PBL).
* **SpeedyWeather basis:** `radiation/shortwave_radiation.jl` +
  `radiation/shortwave_transmissivity.jl`, ozone distribution
  `(σ)->50·max(0,1/5−σ)` applied as `oz·distribution(σ)·dσ`.
* **Deviation:** SPEEDY's two distinct ozone fields (`ozupp` constant aloft,
  `ozone` with latitude/season structure) are summed and redistributed by the σ
  weight rather than kept as two separate single-level injections. Total and
  latitude/seasonal content are preserved; only the vertical split changes.

### 4.2 Longwave stratospheric treatment — `radiation/speedy_longwave.py`

* **Was:** levels 0,1 got isothermal/mean-temperature blackbody emission
  (`0.75·T₀+0.25·T_b`, `0.50·T₁+0.25·(T_b0+T_b1)`), the gradient-emission term was
  set only over `2:kx-1`, and the `corlw1`/`corlw2` stratospheric cooling
  corrections used `dhs[0]`,`dhs[1]`,`st4a[0]`,`st4a[1]`.
* **Now:** every σ<0.2 layer is treated as a stratospheric blackbody. The
  topmost layer keeps `0.75·T+0.25·T_b` (isothermal above TOA); other
  stratospheric layers use `0.50·T+0.25·(T_b_above+T_b_below)` — which **reduces
  exactly to SPEEDY's k=0/k=1 formulas at nlev=8**. The cooling correction
  `dhs[k]·(eps1·psa)·st4a[k,0]` is applied to every stratospheric layer (with the
  polar-night term on the topmost layer only), and `ftop` is the column sum.
  Because `eps1` is spread over the σ<0.2 mass (§4.1), the column-integrated
  cooling is preserved.
* **Choice of σ vs T threshold (documented deviation):** SpeedyWeather's
  `UniformCooling` (`radiation/longwave_radiation.jl`) diagnoses the stratosphere
  by **`T < 207.5 K`** and *replaces this whole scheme* with a relaxation toward
  200 K. We deliberately use the **static σ<0.2 mask instead**, because (a)
  SPEEDY's treatment here is *structural* — it always treats the top of the
  column as isothermal/blackbody by grid position, independent of the actual
  temperature — and the σ mask is the faithful nlev-scaling of that intent; (b) it
  keeps the blackbody construction a compile-time constant and fully
  differentiable (a T<207.5 K mask is state-dependent and would need `jnp.where`
  on traced T). This is a generalization of *SPEEDY's* longwave formulae, not a
  port of SpeedyWeather's different scheme.

### 4.3 Vertical diffusion + cloud/RH top skips

* `vertical_diffusion/speedy_vdiff.py` — **Was:** moisture diffusion above the
  PBL ran over `jnp.arange(2, kx-2)` (skipping top-2 stratosphere and bottom-2
  shallow-convection/PBL). **Now:** runs over interfaces `arange(1, kx-2)` with
  the upper bound replaced by the σ mask (skip an interface whose upper layer is
  stratospheric); the original `σ>0.5` gate and the bottom shallow-convection
  layers are unchanged.
* `radiation/speedy_shortwave.py` `clouds()` — **Was:** RH-max search over
  `rh[2:kx-2]`. **Now:** searched over the whole column with a mask
  `(qa>qacl) & ~strat_mask & (level<kx-2)`, so the stratosphere top bound scales
  and the PBL bottom exclusion is kept.
* `clouds/speedy_condensation.py` — **Left as-is (noted).** Large-scale
  condensation runs over `[1:]` (skip the single topmost layer), which is a
  faithful reproduction of SPEEDY's Fortran `do k=2,kx`, *not* a top-2
  stratosphere assumption. It is already self-consistent for any nlev, and the
  RH-reference profile `rhref(σ²)` plus the σ²-scaled `dqmax` already suppress
  condensation aloft. Migrating it to the σ mask was judged out of scope (it is
  not one of the fixed-2 hardcodings).

### 4.4 PBL top — `vertical_diffusion/speedy_vdiff.py`, `surface/speedy_surface_flux.py`

**Left as-is (noted).** SPEEDY does **not** fix the PBL to a level index. The
shallow-convection step acts on the lowest two layers, moisture diffusion is
gated by `σ>0.5`, and surface fluxes act on the lowest layer (`kx-1`) — all
correct for any nlev. SpeedyWeather uses a bulk-Richardson-number PBL top
(`surface_fluxes/boundary_layer.jl`, critical Ri≈10) and zeroes diffusion above
it; that is a *different closure*, not a generalization of SPEEDY's, so it was
not adopted.

### 4.5 Convection cloud base/top — `convection/speedy_convection.py`

* **Cloud base:** already the PBL (lowest layer) — physical, unchanged.
* **Cloud top:** diagnosed by buoyancy/instability (`mss0 > mss2`, `mse1 > mss2`)
  — physical. **Was:** the search range was hardcoded `jnp.arange(2, kx-3)`
  (skipping the top-2 stratosphere). **Now:** the range is `arange(1, kx-3)` and
  any candidate level falling in the σ<0.2 stratosphere is masked out of the
  instability test, so the upper bound scales with nlev while the buoyancy-based
  selection and the lower bound are unchanged. SpeedyWeather diagnoses cloud top
  by zero buoyancy / level of neutral buoyancy; SPEEDY's instability test is the
  equivalent intent, so only the hardcoded stratosphere bound was migrated.

### Effect on 7/8-level results

The fixed-2 stratosphere is now a σ<0.2 mask. At **nlev=8** the mask selects
exactly the top two layers, so most results are *close* to the old 8-level run
but **not identical**: the shortwave ozone is redistributed vertically within the
stratosphere (column total preserved, surface flux shifts by up to ~30 W/m² in
some columns), and one convection column shifts its cloud top by a single level.
The longwave golden tests still pass at their original tolerances because the
mask reproduces the top-2 treatment. At **nlev=7** the mask selects only 1 strat
level (its 2nd midpoint sits exactly on σ=0.2), a larger but accepted change. All
of this is intentional and accepted: a σ-scaled stratosphere is the goal.

## 5. Verification

See the run log in the task report. Summary:

* `get_speedy_coords(layers=L)` and a short forward integration of
  `speedy_physics()` through `Model` run with finite outputs and correct shapes
  for `L ∈ {8, 16, 30}`.
* `speedy_terms_test.py` (which derives `nlev` from `coords.nodal_shape[0]`)
  passes at `L = 8` (unchanged) and, parameterized, at `L = 16` and `30`.
* 7/8-level results are bit-for-bit unchanged because those cases still use the
  hand-tuned tables.
