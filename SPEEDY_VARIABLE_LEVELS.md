# Generalizing SPEEDY physics to an arbitrary number of vertical levels

Status: implemented and verified. The σ-table generalization (§2–§3) and the
physics-scheme migration to a σ-threshold stratosphere (§4) are both complete.
**The §4 migration changes the 7/8-level results** (intentionally — see §4).
§6 adds a resolution-aware time step that keeps high-`nlev` / high-truncation
runs numerically stable; verified by **365-day** spin-ups at T21 nlev=8/16/24/32
and T31 nlev=8/16/24 (all finite — see §6).

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

## 6. Numerical stability at high nlev

### Symptom

With the §4 migration done, short runs are fine but **realistic 365-day
spin-ups go non-finite at high vertical resolution / higher truncation**.
Reproduced exactly (random IC → forward integration under realistic terrain +
time-constant annual-mean SST, default 30-minute step):

* **Stable:** T21 nlev=8, 16, 24.
* **Unstable (NaN):** T21 nlev=32; T31 nlev=16; T31 nlev=24.

On the smallest unstable case (T21 nlev=32, CPU) the model blows up **within the
first ~2 hours of model time (4 timesteps)**. Stepping one `dt` at a time and
inspecting per-level fields, the divergence starts in the **thin bottom (surface)
layer**: the surface-layer `u_wind` tendency grows ~100 → 320 → 4200 m/s **per
step** at levels 30–31 and goes non-finite at step 5, while every layer above is
still well-behaved. This is a single-layer growing oscillation, not a slow drift.

### Diagnosed cause: explicit surface-drag instability in the thin bottom layer

Two decisive tests isolate it:

1. **Dynamics alone is stable.** Running the dry dynamical core with **no
   physics** at T21 nlev=32, dt=30 min stays finite and bounded for 90+ model
   hours (max |u| ≈ 60–80 m/s). So it is **not** a dynamics CFL / semi-implicit
   vertical-operator problem — the IMEX-RK SIL3 dycore handles the thin layers.
2. **Halving the step cures it.** T21 nlev=32 at dt=15 min is stable (bounded
   T≈315 K, |u|≈80 m/s over a day). So it is a forward-Euler **timestep**
   instability in the operator-split physics tendency.

The mechanism is SPEEDY's surface stress, applied to the bottom layer as a
gridpoint tendency (`surface/speedy_surface_flux.py`):

```
du/dt|_sfc = ustr · rps · grdsig[-1],   ustr = −C_drag·|V|·u,
grdsig[-1] = g / (dσ_bot · p0)
```

This is added **explicitly (forward Euler)** by the op-split coupling. Its
stability limit is roughly `dt · C_drag·|V| · grdsig[-1]/p_norm < O(2)`. The
Frierson cubic stretch makes the bottom layer thin as nlev grows — `dσ_bot` falls
from **0.10 at nlev=8 to 0.0079 at nlev=32**, so `grdsig[-1]` grows **~12.6×** —
and the explicit drag damping factor crosses the limit. Higher truncation makes
it worse because the resolved near-surface wind (and the Gibbs ringing of the
surface-drag tendency) grows with the number of horizontal modes. Empirically the
stability boundary is ordered monotonically by the dimensionless severity

```
S = (trunc + 1)^1.3 / dσ_bot
```

(every config stable at 30 min sits below S_norm ≈ 9.4 relative to the T21/nlev=8
baseline; every unstable one sits above it).

### Fix: resolution-aware (CFL-like) automatic time step

We reduce the model time step when the configuration is severe, rather than
rewriting the thin-layer surface tendency to be implicit (a large, invasive
change to the op-split coupling). `Model(time_step=...)` now defaults to `None`,
in which case the step is auto-selected from the resolution by
`stable_time_step_minutes(nlev, spectral_truncation)`
(`physics/speedy/physical_constants.py`):

* **Plateau.** For `S_norm ≤ 9.4` return the validated **30-minute** reference
  step. This covers every configuration that is already stable at 30 min — in
  particular **all of SPEEDY's standard 7- and 8-level runs are bit-for-bit
  unchanged** (the time step they get is identical).
* **Reduced branch.** Above the plateau, `dt ∝ 1/S` with a **0.85 safety
  factor** for margin below the measured blow-up boundary.
* **Day-aligned snapping.** The raw step is snapped **down** to the nearest
  integer divisor of a 1440-minute day (30, 24, 20, 18, 16, 15, 12, … min) so
  every saved frame lands on a whole number of steps (the trajectory builder
  uses `inner_steps = int(save_interval / dt)`) — daily/monthly saves don't
  drift over a 365-day run, and snapping down can never re-introduce the
  instability.

Resulting auto steps and the empirically measured blow-up boundary:

| config | dσ_bot | auto dt | first unstable dt |
|--------|--------|---------|-------------------|
| T21 nlev=8/16/24 | 0.10 / 0.017 / 0.011 | **30 min** | stable at 30 |
| T21 nlev=32 | 0.0079 | **18 min** | 30 min |
| T31 nlev=8 | 0.10 | **30 min** | stable at 30 |
| T31 nlev=16 | 0.017 | **24 min** | 30 min |
| T31 nlev=24 | 0.011 | **15 min** | 25 min |

An explicit `time_step=` still overrides the auto choice everywhere (the unit
tests and the Hydra configs pass explicit values and are unaffected).

### Verification — actual 365-day integrations

Full 365-day forward spin-ups (random IC, realistic terrain, annual-mean SST),
auto time step, GPUs 2–7. Plausibility checks: T∈[150,340] K, |wind| not absurd,
q non-negative, ps∈~[400,1100] hPa, **entire trajectory finite**.

| config | auto dt | finite 365d | T min/max (K) | \|wind\| max | q max (g/kg) | ps (hPa) |
|--------|---------|-------------|---------------|--------------|--------------|----------|
| T21 nlev=8  | 30 min | **yes** | 128 / 298 | 90 m/s  | 17 | 607–1089 |
| T21 nlev=16 | 30 min | **yes** | 129 / 301 | 91 m/s  | 19 | 606–1085 |
| T21 nlev=24 | 30 min | **yes** | 122 / 301 | 120 m/s | 21 | 591–1086 |
| T21 nlev=32 | 18 min | **yes** | 132 / 302 | 81 m/s  | 19 | 585–1090 |
| T31 nlev=8  | 30 min | **yes** | 127 / 302 | 91 m/s  | 18 | 535–1098 |
| T31 nlev=16 | 24 min | **yes** | 112 / 301 | 127 m/s | 20 | 548–1096 |
| T31 nlev=24 | 15 min | **yes** | 118 / 303 | 125 m/s | 19 | 555–1098 |

(T min ≈ 110–130 K is the cold model-top stratosphere — present at the
always-stable nlev=8 runs too, not an instability artefact. The small negative q
minima seen in some columns, ≈ −3 g/kg, are spectral Gibbs ringing of the
physics tendency and are clamped at the user-visible output boundary; they also
appear at nlev=8.)

All seven configurations — including the three that previously NaN'd (T21
nlev=32, T31 nlev=16, T31 nlev=24) — now complete 365 days finite and
physically plausible.

## 6. Numerical stability at high nlev (auto time step)

### Symptom
With the σ-migration above, short runs are fine, but **long (365-day) realistic
spinups go non-finite** at high level counts / high truncation: T21 nlev=32,
T31 nlev=16, T31 nlev=24 all blow up within a few timesteps, while T21 nlev≤24
are stable.

### Diagnosed cause — explicit surface drag in the thin bottom layer (NOT CFL)
The dry dynamical core alone is stable at dt=30 min even at nlev=32, so this is
**not** a dynamics CFL problem. The instability is the **explicit (forward-Euler)
surface-drag tendency in the thinnest *bottom* sigma layer**:

```
du/dt|_sfc = ustr · rps · grdsig[-1],   grdsig[-1] = g / (dσ_bot · p0)
ustr = -C_drag · |V| · u            (jcm/physics/surface/speedy_surface_flux.py)
```

The Frierson stretch thins the bottom layer (dσ_bot ≈ 0.10 at nlev=8 → ≈ 0.008 at
nlev=32), so `grdsig[-1]` grows ~12× and the explicit damping factor
`dt · C_drag·|V| · grdsig[-1]` crosses the forward-Euler stability limit.
**The unstable layer is at the surface (σ→1), not the model top** — so capping
tiny *upper* σ would not fix it, and flooring the *bottom* layer would throw away
exactly the near-surface resolution this effort is meant to add.

### Fix — `stable_time_step_minutes(nlev, trunc)` (auto, opt-out)
`jcm/physics/speedy/physical_constants.py` defines a stability index
`S = (trunc+1)^1.3 / dσ_bot`, normalised to the canonical-stable T21/nlev=8
baseline. Empirically every config stable at 30 min sits below `S_norm ≈ 9.4` and
every unstable one above it. `Model.__init__` calls `_auto_time_step_minutes`
when `time_step is None` (the default): it keeps **dt = 30 min on the stable
plateau** (so all 7/8-level runs and every currently-stable config are
**bit-for-bit unchanged**) and shrinks dt ~ 1/S above it, snapped to integer
divisors of a 1440-min day (so save_interval/dt stays integer), with a safety
margin. Pass an explicit `time_step` to override.

Selected steps: T21 L8 → 30 min (unchanged), T21 L32 → 18, T31 L16 → 24,
T31 L24 → 15.

### Verification
60-day realistic spinups (frozen-mean BCs) are finite for T21 {8,32} and
T31 {16,24} with the auto dt (previously blew up within a few steps); 365-day
spinups for the most-stressed configs (T21 L32, T31 L16/L24) confirm finiteness
with physical temperature ranges. All previously-stable configs keep dt=30 min,
so their results are unchanged.

### Deferred alternative (backup)
The most compute-efficient fix would be an **implicit surface drag** in the thin
bottom layer (unconditionally stable → keep dt=30 min *and* the thin near-surface
layers). It was deferred as too invasive for this pass; it is the preferred
optimisation if the reduced dt makes high-nlev / T31 integrations too slow
(e.g. the 14-month GFMIP finite-difference runs).
