# The TTE-TKE closure: ECHAM6.3's `vdiff`

`jcm/physics/vertical_diffusion/tte_tke/` is ECHAM6.3's vertical diffusion
(`vdiff.f90`, r7492): the moist, cloud-weighted buoyancy of every interface
(`moist_buoyancy.py`), the TKE closure that turns it into exchange
coefficients (`closure.py`), the surface layer (`surface_layer.py`) and the
implicit column solve (`matrix_solver.py`). This page records the closure and
the solve: what they are, the line of `vdiff.f90` each piece is, the choices
the port makes where ECHAM's structure and jcm's differ, and the checkpoint
consequence of the interface staggering. The science summary is
{doc}`../science/vertical_diffusion`.

## The closure, line by line

All arrays are vertical-on-axis-0, model top first, and broadcast over any
trailing horizontal axes (a `(nlev,)` column and a `(nlev, ncols)` block run the
same code).

| piece | `vdiff.f90` | jcm |
|---|---|---|
| constants `zshn = zh1·zh2·√2`, `zsmn = zshn·zm1·zm2/zm4`, `zda1 = 1/zsmn³`, `zcons2/3/5/8/9/18/25` | l.560-604 | `closure_constants`, inline |
| geopotential above the surface `pgeom1` | `physc.f90` (`geopot` with `φ_s = 0`) | `geopotential_above_surface` |
| PBL extension `ihpbl`: dynamic `zhdyn = min(z_top, 0.3·u*/max(|f|, 5e-5))`, convective `ihpblc` (lowest level whose `zcptgz` exceeds the lowest level's), `ihpbl = min(ihpblc, ihpbld)`, or `ihpblc` alone under ECHAM's RCE switch `lrce`, `zghabl = min(50000, pgeom1(ihpbl))` | l.737-765 | `pbl_extension` (`VDiffParameters.lrce`) |
| mixing length: `zmix = κz/(1 + κz/λ)`, `λ = clam` at and below `ihpbl`, `λ = 1 + (clam − 1)·exp(1 − φ_k/φ_ihpbl)` above | l.801-817 | `mixing_length` |
| Louis (1979) `zsm`, `zsh`: unstable `zucf` branch, stable branch | l.819-832 | `louis_stability` |
| implicit TKE source and dissipation `zzb`, `zdisl`, `zktest`, `ztkevn` | l.837-844 | `tke_source_update` |
| first-step substitution (`lstart`) | l.845-848 | `interior_closure` (carried 0) |
| exchange coefficients `zcfm`, `zcfh` with `zalf = p_{k+1/2}/(T̄_v·Δφ·R_d)`; TKE and variance transport `zcdum`, `zcthv` | l.849-856 | `interior_closure` |
| θ_v variance `zthvprod`, `zthvdiss`, floor | l.857-860, 864-866 | `thv_variance_source_update` |
| surface TKE (Mailhot-Benoit) and the surface rows `cdum`, `cduh`, `tkevn` | `mo_surface_*.f90::precalc_*`, l.1099-1102 | `surface_tke_boundary`, `vertical_diffusion_column` |
| solve of TKE and variance on the interfaces | l.1235-1314 | `matrix_solver._interface_rows` |
| solve of u, v, s, q, ql, qi on full levels | l.877-941, 1212-1229, 1347-1403 | `matrix_solver._full_level_rows` |
| dissipation heating `zdis`, temperature from `s` | l.1412-1456 | `compute_tendencies_from_solution` |

The closure constants (`closure_h1 … closure_m4`, `mixing_length_asymptote`
= `clam`, `pbl_dynamic_height_factor` = `zchneu`,
`surface_tke_convective_factor` = `zwstf`, and the surface layer's `louis_cb`,
`louis_cc`, which ECHAM shares between the two) are differentiable
`VDiffParameters` leaves; `totte_min` is ECHAM's `ztkemin = 1e-10`, the floor
of TKE and of the variance, and must be positive (`VDiffParameters.validate`):
a carried 0 marks a cold start, and the closure takes `√e` at the floor.

## Interface staggering

ECHAM keeps TKE, the θ_v variance and the exchange coefficients on the
interfaces, in `klev`-row arrays whose row `jk` is the interface *below* full
level `jk` and whose last row is the surface layer's (`zcfm(klev) = cdum`,
`ztkevn(klev) = tkevn`). jcm keeps the same layout and the same row count:
`VDiffState.tke`, `VerticalDiffusionData.tke`/`thv_variance`/`km`/`kh` have
`nlev` rows, row `k` the interface between full levels `k` and `k + 1`, the
last row the surface layer. Two things follow from keeping ECHAM's layout
rather than averaging to full levels:

* **Consumers read what ECHAM's consumers read.** ECHAM-HAM's activation
  (`mo_activ.f90`) and the two-moment cloud microphysics
  (`mo_cloud_micro_2m.f90`) take `ptkem1(jl,jk)` as "the TKE of level `jk`".
  jcm's activation, Lohmann 2M and AeroCom updraft diagnostics read row `k` of
  the carry at level `k`, so they get the same value with no change of their
  own.
* **The solve uses ECHAM's coefficients directly.** The full-level rows couple
  through the interior interfaces' conductances `G = g·ρ·K/Δz` (ECHAM's
  `zcf/(cvdifts·Δt)`), with ECHAM's `ρ/Δz` of the interface; no averaging of a
  full-level K and no thickness floor is involved. The TKE and variance rows
  have the pressure thickness between full levels and the mean coefficient of
  the two rows, and the surface row is a fixed value the row above reads.

`km`/`kh` carry 0 in the surface row (the surface exchange is
`surface_exchange_*`). `TracerVerticalDiffusion` and the JAM cloud-borne store
take `kh[k]` as the interface below level `k`.

## Heat is ECHAM's dry static energy

The solve diffuses `s = φ + c_pd·T·(1 + vtmpc2·q)` (`zcptgz`, l.669-670, 890),
carried as `s/c_pd`, with `φ` above the surface, and the surface exchanges
`s_s − s_K` with the tiles' `zcptw`/`zcpti`/`zcptl = c_pd·T_s·(1 + vtmpc2·q_s)`
(the land's `q_s` is `csat·q_sat + (1 − cair)·q_a`). The temperature follows as
`T = (s + zdis − φ)/(c_pd·(1 + vtmpc2·q))` with the step-start `q`, where
`zdis = ½(|u|² − |u_new|²)` is the kinetic energy the momentum diffusion
removed (l.1418-1445). A well-mixed layer therefore has uniform `s` (a dry
adiabat), the column's dry static energy changes by exactly the surface
sensible heat flux, and its `c_p·T` by that plus the dissipated kinetic energy.
The land energy balance solves for `ŝ` with the tile's heat capacity `zcpq`
(`mo_surface_land.f90` l.329-331).

## Choices where the structures differ

* **Geopotential above the surface.** ECHAM hands `vdiff` `φ − φ_s`; jcm's
  physics state carries `φ` with the orography. The closure subtracts the
  surface value the hydrostatic discretisation implies for the lowest level,
  `φ_K − α_K·R_d·T_v,K` (`geopot.f90`, `auxhyb.f90`, the same discretisation
  the dynamical core uses), which recovers ECHAM's field exactly and keeps
  every level difference the model's own.
* **The dynamic PBL height's top level.** `zhdyn` never exceeds the top
  level's height, so ECHAM's top level always qualifies (`zdz >= 0` at
  `jk = 1` holds with equality when `zhdyn` is capped); `pbl_extension` states
  that outright instead of relying on two float32 roundings of the same
  quantity agreeing.
* **Previous-step friction velocity.** `ustarm` is the previous step's
  grid-mean `u*` (ECHAM's own JSBACH note, l.732-735); jcm's is the carried
  `surface_friction_velocity`, the fraction-weighted surface stress's. With
  none (a cold start) it is ECHAM's `lstart` value of 1 m/s.
* **Cold start.** ECHAM starts from `tke = thvvar = 1e-4` (`ioinitial.f90`
  l.418-420) and on its first step forms the coefficients from the
  post-source TKE (`lstart`, l.845-848). jcm's carry starts at zero, which the
  closure reads as an interface with no previous step and treats the same
  way: a stepped value is never zero (`ztkemin > 0`).
* **No time filter.** ECHAM's leapfrog applies an Asselin filter to `ptkem1`
  (l.1318-1332); jcm's physics is two-level, so the carried TKE is the new
  value. The TKE after transport is floored at `ztkemin`, where ECHAM aborts on
  a negative value (l.1313) — the implicit transport of a non-negative field
  stays non-negative, so the floor is a round-off guard.
* **`zcfm2m`.** The surface TKE's `u*` uses ECHAM's `zcfm2m`, which differs
  from the tile's `C_m|U|` only for a roughness above `zepz0o = 2` m; no jcm
  tile is that rough.
* **The surface layer's reference height.** The surface layer
  (`surface_layer.py::surface_bulk_richardson`,
  `compute_surface_exchange_coefficients_echam_louis` and
  `echam_louis_neutral_drag`) takes the lowest level's height from the
  moist-air column's `height_full − height_half`, where ECHAM's
  `precalc_*` take `pgeom1(klev)/g`, the hydrostatic height the closure forms
  (`geopotential_above_surface`); 57.6 m against 32.5 m on L47. Tracked in
  jax-gcm#1171.

## Checkpoints

The carry's `tke`, `thv_variance`, `km` and `kh` changed meaning with the
staggering, so the checkpoint schema is 4 ({doc}`checkpoint_compatibility`).
A file of an earlier schema stored row `k` as the interface *above* full level
`k`; `load_checkpoint` moves its rows up one (`_migrate_tte_tke_to_interface_rows`)
and gives the surface row the lowest interior interface's value, which the
scheme replaces on its first step. A resumed or warm-started run from an older
file therefore starts from its own turbulence, one level in the right place,
rather than from a cold start.

## Differentiability

The PBL extension is a level index and the closure has ECHAM's branch points
(`Ri = 0`, `zktest = 1`, the floors). Each is differentiated on the branch it
selects; the mixing length jumps when `ihpbl` moves a level, so a parameter
that moves it (`zchneu`) has a derivative of the selected level only. The
Louis unstable branch sees `Ri` only below zero and the TKE square root only
above it, so the derivative at `Ri = 0` and at a zero TKE is finite.

In float32 the derivative of the temperature tendency is limited by the dry
static energy, not by the closure. The temperature is the difference of `s`
before and after the solve less `pgeom1`, so every direction that moves
`pgeom1` (the geopotential, the lowest level's virtual temperature) carries
terms of the size of `s` that cancel to the size of the tendency. On a
cold-started, cloud-topped column whose exchange coefficients reach
10⁴-10⁵ m²/s, the jvp and vjp projections of `term_gradients_test.py` differ
by up to 1.3e-2 in float32, and both are as far from their float64 values; in
float64 the adjoint identity holds to 2e-11, so the term's identity is checked
there. A smooth surrogate of `√e` at the floor `ztkemin` (where `d√e/de` is
5e4) leaves the float32 gap unchanged.

## Validation

`closure_test.py` compares every piece above with the compiled statements of
`vdiff.f90` and `precalc_ocean` on the 384 columns of the #962 reference set
(`jcm/data/test/echam_vdiff_reference/vdiff_closure_T63L47.npz`, provenance
there), in float64 with ECHAM's constants:

| quantity | ECHAM | agreement (fraction of the field's largest value) |
|---|---|---|
| `zshn`, `zsmn`, `zda1` | 0.690702, 0.550091, 6.007545 | exact |
| `ihpblc`, `ihpbld`, `ihpbl` | l.737-765 | identical in all 384 columns |
| `ihpbl`, `zghabl`, `zmix` under `lrce` (299 columns change level) | l.756-760 | identical; 0, 3e-16 |
| `zghabl`, `zhdyn` | | 4e-16, 0 |
| `zmix` | l.801-817 | 4e-15 |
| `zsm`, `zsh` | l.819-832 | 2e-14 |
| `ztkevn` | l.837-844 | 2e-14 |
| `zthvvar` | l.857-860 | 7e-15 |
| `zcfm`, `zcfh`, `zcdum`, `zcthv` | l.849-856 | 3e-14 |
| `ptke`, `pthvvar` after transport | l.1235-1314 | 7e-14 |
| `pvom`, `pvol` | l.1347-1425 | 3e-14 |
| `ptte`, `pqte`, `pxlte`, `pxite`, `zcpten` (the dry static energy solve, with the surface's `ztdif_new`/`zqdif_new` as inputs) | l.877-941, 1212-1219, 1439-1457 | 7e-14 |
| surface `ztkevw` (192 cells) | `precalc_ocean` | 2e-16 |

The boundary-layer cases (`boundary_layer_cases_test.py`,
`scm_boundary_layer_cases_test.py`) check the regimes: TKE spinning up to a
steady state under neutral shear with `l → κz` near the ground, stable
stratification suppressing it, buoyancy sustaining it without shear, the
neutral Prandtl number `zsmn/zshn`, and an Ekman jet inside the PBL extension.
