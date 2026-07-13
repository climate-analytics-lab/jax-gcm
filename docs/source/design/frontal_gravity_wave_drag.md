# Frontal spectral gravity-wave drag (CAM port)

`jcm/physics/gravity_waves/spectral/` is a faithful JAX port of CAM's
spectral non-orographic gravity-wave drag with the frontogenesis-triggered
(Charron & Manzini 2002) source, from **ESCOMP/CAM, ref `cam_cesm2_2_rel`,
`src/physics/cam/`**:

| jcm module          | CAM source                                | what |
|---------------------|-------------------------------------------|------|
| `solver.py`         | `gw_common.F90` (`GWBand`, `gw_prof`, `gw_drag_prof`, `calc_taucd`, `momentum_flux`, `momentum_fixer`, `energy_change`, `energy_fixer`) + the `alpha0` Newtonian-cooling table from `gw_drag.F90::gw_init` | wave propagation (Lindzen saturation + molecular/Newtonian damping), momentum deposition, tendency limiters, momentum/energy bookkeeping |
| `frontal.py`        | `gw_front.F90` (`flat_cm_desc`, `gaussian_cm_desc`, `gw_cm_src`) | launch mask `frontgf > frontgfc`, flat/Gaussian source spectra, phase speeds `c = cref + |u_src|` |
| `frontogenesis.py`  | `src/dynamics/se/gravity_waves_sources.F90::compute_frontogenesis` (formula only) | lat-lon provider for the trigger field |
| `term.py`           | `gw_drag.F90` (`use_gw_front` wiring in `gw_drag_init`/`gw_tend`) | `FrontalGravityWaveDrag` PhysicsTerm |

The pure-NumPy loop transliteration of `gw_prof`/`gw_drag_prof` kept in
`solver_test.py` is the validation reference (float64, rtol 1e-5), per the
repo's Fortran-port pattern.

## Axis conventions

- Vertical on **axis 0, top-first** (identical to CAM's `k = 1..pver` and
  the physics-internal frame): midpoints `(nlev, *horiz)`, interfaces
  `(nlev+1, *horiz)`; midpoint `k` lies between interfaces `k` and `k+1`.
- The **phase-speed spectrum is its own axis in front of the horizontal
  axes**: `c`, `tau_src` are `(nspec, *horiz)`; `tau` is
  `(nlev+1, nspec, *horiz)`; `gwut` is `(nlev, nspec, *horiz)`, with
  `nspec = 2*ngwv + 1` and CAM's `l ∈ [-ngwv, ngwv]` at index `l + ngwv`.
  Because horizontal axes trail, `(*horiz)` fields broadcast against
  `(nspec, *horiz)` ones with no reshapes, and the same code runs on a
  single column, a `(nlev, ncols)` block, or a `(nlev, ix, il)` grid.

## Frontogenesis-provider contract

`FrontalGravityWaveDrag` reads the trigger from the diagnostics dict key
**`"frontogenesis"`** — a midpoint field `(nlev, *horiz)` in K²/m²/s (CAM's
`FRONTGF`), i.e.

F = −∇ₕθ · [(∇ₕθ · ∇ₕ) **u**ₕ]

evaluated by the *dycore or a provider term*, exactly as CAM's dynamical
cores hand `frontgf` to the physics through the physics buffer. The term
samples it at the static trigger level `kfront` (deepest midpoint whose
upper interface is above 600 hPa, from reference pressures `a + b·p0`).

- **Without a provider the term is inert**: the trigger falls back to the
  constant `params.fallback_frontogenesis` (default 0.0 < any positive
  `frontgfc`), so no waves launch and all tendencies are **exactly zero**
  (tested). Setting the fallback above `frontgfc` forces a uniform launch —
  a testing aid with no CAM counterpart.
- `frontogenesis.py` provides the field for **separable lat-lon grids**
  (dinosaur backend): centered differences with spherical metric factors
  (`1/(a cos φ) ∂λ`, `1/a ∂φ`), periodic in longitude, one-sided at the
  first/last latitude rows. Unstructured-grid (pySES pg2 / SE-GLL)
  stencils are **out of scope** — those backends must supply their own
  `"frontogenesis"` diagnostic. Wiring a provider into each backend (and
  composing the term into `echam_physics()`) is a follow-up.

## Deviations from CAM (complete list)

Solver (`gw_common.F90`):

1. `src_level`/`tend_level` are one static Python int, uniform over
   columns and equal to each other (always true for the frontal source,
   which derives them from reference pressures). CAM's per-column arrays
   are not supported.
2. `gw_diffusion.F90` is **not ported**: no effective diffusivity
   `egwdffi`, no constituent tendencies `qtgw`, no dry-static-energy
   diffusion `dttdf`. This matches CAM's own `lapply_vdiff=.false.` path;
   the returned heating is the kinetic-energy conversion `dttke` (plus
   the energy fixer at term level).
3. Only the `lapply_effgw=.true.` branch exists (CESM2.2 CAM6 default
   `gw_apply_tndmax=.true.`); the WACCM legacy ordering is not ported.
4. Optional arguments `kwvrdg`, `ro_adjust`, `vramp` (ridge scheme,
   inertial-gravity-wave adjustment, top-of-model taper) are not ported —
   none is used by `use_gw_front` with CAM6 defaults. Likewise
   `coriolis_speed`/`adjust_inertial` (only used by `use_gw_front_igw`).
5. `kvtt` (molecular thermal diffusivity) defaults to zero, CAM's value
   whenever WACCM's `do_molec_diff` is off; the `dback = 0.05` background
   diffusivity is retained.
5b. **Heating-bounded stability limiter** (`limit_tendency_sum`, default
   True). CAM's tndmax limiter caps only the **net** per-level tendency
   `|Σ_l gwut_l|`; the frictional heating `dttke = Σ_l |u−c||gwut_l|`
   has **no bound anywhere in CESM2.2** (`vramp`/`gw_top_taper` only
   activates above 0.6 Pa, i.e. WACCM-X grids, and is off by default).
   On grids whose lid layer is a few Pa thick with p_top = 0 (ECHAM
   L47), `rho → 0` drives `tausat → 0` for *every* wave, the whole
   surviving flux deposits in the lid layer, the two spectrum halves
   cancel in the net (observed: net 353 vs Σ|gwut| 4136 m/s/day), and
   the heating reached 123 K/day in real ne30 winter-jet columns —
   blowing up the run while |du/dt| sat innocently at tndmax. CAM never
   operates this scheme with a lid layer thinner than O(100 Pa). The
   port's default therefore applies the *same* limiter to
   `Σ_l |gwut_l|` instead: identical to CAM whenever deposition is
   one-signed (the usual single-critical-level case), still caps the
   net, and bounds the heating by `max|u−c|·tndmax` (~tens of K/day
   worst case; 22 K/day on the offending columns). The stress
   re-adjustment machinery is untouched, so the momentum/flux
   bookkeeping stays exactly conservative. `limit_tendency_sum=False`
   restores the exact CAM limiter (validated against the NumPy
   reference in both modes).
6. Degenerate-state gradient guards: every masked division/sqrt keeps its
   safe operand *inside* `jnp.where` (no 0·inf in reverse mode). Callers
   must pass finite `piln`; the term floors the top interface pressure at
   1e-4 Pa before the log (CAM grids have p_top > 0 so its `piln` is
   always finite; pure-sigma grids here can have p_top = 0).

Source (`gw_front.F90`):

7. `gaussian_spectrum` adds an optional `center` (CAM's Gaussian is
   hard-centered on c = 0); default 0.0 reproduces CAM bit-for-bit.
8. `CMSourceDesc` collapses to plain arguments/static ints; `gw_cm_src`
   fills `ubm` at every level (CAM leaves levels below `ksrc` undefined;
   the solver masks them either way).

Term wiring (`gw_drag.F90`):

9. No polar taper (`gw_polar_taper`; off for the SE dycore this port
   follows) — `effgw` is a spatially uniform differentiable scalar.
10. The frontogenesis *angle* field (`FRONTGA`) and the history/diagnostic
    output plumbing (`gw_spec_outflds` etc.) are not ported. Instead the
    term publishes its applied tendencies as user-facing diagnostics —
    `gw_frontal_dudt` / `gw_frontal_dvdt` [m/s²] and `gw_frontal_dtdt`
    [K/s] (the analogue of CAM's `UTGW_CM`/`VTGW_CM`/`TTGW` history
    fields) — so runs can see the frontal drag in the output stream.
11. CAM adds `qtgw`/`egwdffi` into the host model's diffusion; with (2)
    there is nothing to add. The energy fixer receives `de` computed from
    this term's own tendencies only (CAM passes the accumulated `ptend`,
    but at this point in CAM's sequence those are exactly the frontal-GW
    increments too); orographic `flx_heat` does not exist here (0).

Frontogenesis provider:

12. The vector term `(∇θ·∇)u` is evaluated component-wise on (u, v); the
    spherical-curvature (Christoffel) corrections included by HOMME's
    covariant `ugradv_sphere` are neglected (`O(tan φ |u|/a)` next to the
    deformation terms; the trigger is thresholded, not dynamical).
    Discretization is centered finite differences, not spectral elements.

## Defaults (CESM2.2 CAM6 ne30, `use_gw_front`)

| parameter | value | CAM origin |
|---|---|---|
| `taubgnd` | 1.25e-3 Pa | `namelist_defaults_cam.xml` (`ne30np4`) |
| `frontgfc` | 3.0e-15 K²/m²/s | `namelist_defaults_cam.xml` (`ne30np4`) |
| `effgw` (`effgw_cm`) | 1.0 | `namelist_defaults_cam.xml` |
| `ngwv` (`pgwv`) | 32 | `namelist_defaults_cam.xml` |
| `dc` (`gw_dc`) | 2.5 m/s | `build-namelist` |
| `fcrit2` | 1.0 | `gw_drag.F90` (`band_mid`) |
| `wavelength` | 1e5 m | `gw_drag.F90` (`wavelength_mid`) |
| `gaussian_width` | 30 m/s | `gw_drag.F90` (`front_gaussian_width`) |
| `tndmax` | 400 m/s/day | `gw_common.F90` |
| `umcfac` / `satfac` | 0.5 / 2.0 | `gw_common.F90` / `gw_drag_prof` |
| source level | interface just above 500 hPa | `gw_drag.F90` (`kbot_front`) |
| trigger level | midpoint just above 600 hPa | `gw_drag.F90` (`kfront`) |
| spectrum | bin-averaged Gaussian | `gw_drag.F90::gw_init` |
| `tau_0_ubc` | False | CAM6 non-WACCM default |

All numeric parameters are differentiable leaves of the
`flax.struct.dataclass` `FrontalGWParameters` (held in an `nnx.Param`);
`ngwv`, the spectrum shape, the level-selection pressures and the static
flags are `pytree_node=False` aux data.


## The frontogenesis-provider contract (dycore physics-fields hook)

The trigger field is a horizontal-gradient quantity, so it belongs to
whoever owns the horizontal discretization — exactly CAM's architecture,
where the SE dycore computes `frontgf` and hands it to physics through
the physics buffer. jcm mirrors this with a `DynamicalCore` hook:

- `DynamicalCore.physics_field_names()` declares the fields a backend
  supplies; `physics_fields(state, physics_state)` computes them each
  step (it receives the already-projected gridpoint state so backends
  need not redo the native->gridpoint conversion).
- `Model` injects the dict into the physics diagnostics under
  `"_dycore_fields"` every step. The key is in
  `ComposablePhysics._INTERNAL_DIAGNOSTIC_KEYS`, so it is stripped from
  the cross-step scan carry and from saved output (structure-stable
  carry; no output bloat).
- Terms declare `requires_dycore_fields`; `Model.__init__` validates the
  requirement against the backend's declared names (or an upstream
  term's `provides`) and fails at construction, not at trace time.
- `DinosaurDycore(compute_frontogenesis=True)` enables the lat-lon
  provider (theta from the core's own hybrid/sigma coefficients, then
  `frontogenesis_function`; spectral-gradient upgrade possible).
  The pySES/pg2 provider (weak-form GLL gradients + DSS + gll_to_fv,
  i.e. CAM-SE's own compute_frontogenesis) is a follow-up.
- `echam_physics(gw_scheme="hines" | "frontal" | "none")` selects the
  non-orographic GW scheme; the two schemes are exclusive alternatives
  (same physical role — running both would double-count mid-atmosphere
  drag unless retuned together).
