# The ECHAM land tile: JSBACH's evaporation form and a skin energy balance

The ECHAM hosts (`echam-1m`, `echam-2m`, `echam-jam`) run a *prescribed-moisture
land*: soil moisture, snow cover and the deep soil temperature are prescribed
climatologies, and the land skin temperature is prognostic. It is solved each
step from the surface energy balance, coupled implicitly to the lowest model
level the way ECHAM6.3 couples JSBACH. The land's moisture flux carries
JSBACH's humidity factors. This page derives the formulation, states every
stand-in for the JSBACH state the forcing bundle does not carry, and records
how it was checked against the compiled ECHAM6.3 / JSBACH Fortran (r7492).
What the land tile is, as a scientific choice, is in {doc}`../science/surface`.
The rest of JSBACH (#672) is out of scope: the bucket, interception, snow mass
and the multi-layer soil.

## 1. Evaporation: JSBACH's humidity factors

ECHAM writes every tile's moisture flux with two factors
(`mo_soil.f90::update_soil` 1900-1902, `mo_surface_land.f90::richtmyer_land`):

```
E     = ρ·C_h|U|·(csat·q̂_s − cair·q̂_K)          positive up
E_pot = ρ·C_h|U|·(q̂_s − q̂_K)
LH    = L_v·E + (L_s − L_v)·s·E_pot              (update_soil 1915-1916)
```

Over open water and sea ice `cair = csat = 1`. Over land they are built as
`update_soil` builds them: the canopy-resistance block (1741-1761) and the
humidity-factor block (2503-2574), with JSBACH's tiles collapsed to one
non-glacier and one glacier tile:

```
β        = clip((w − w_wilt)/(w_crit − w_wilt), 0, 1)           calc_water_stress_factor
r_c      = 1/(g_c·β)   if β > ε, g_c > ε and q_a ≤ q_s, else 1e20
qsat_veg = s + (1 − s)/(1 + C_h|U|·r_c)   above wilting,  s below
bare:      h > q_a/q_s  →  qsat_fact = s + (1 − s)·h,  qair_fact = 1
           otherwise    →  qsat_fact = qair_fact = s
           q_a > q_s    →  qsat_fact = qair_fact = 1   (deposition at the potential rate)
csat     = v·qsat_veg + (1 − v)·qsat_fact,   cair = v·qsat_veg + (1 − v)·qair_fact
glacier share g: csat = cair = 1;   land = (1 − g)·non-glacier + g
```

The relative-humidity form of the bare soil is what bounds the moisture source.
A dry soil evaporates only while `h·q_s` exceeds the air humidity, and then only
the excess. The beta form it replaces, `cair = csat = w`, evaporated
`w·(q_s − q_a)` whatever `h` was. The same factors also set the surface-layer
humidity, `qts = csat·q_s + (1 − cair)·q_a` (`precalc_land` 200-232), which
reduces to the port's earlier `w·q_s + (1 − w)·q_a` when `cair = csat = w`.

### Stand-ins for the JSBACH state

| JSBACH quantity | jcm stand-in | why |
|---|---|---|
| root-zone fill `ws/wsmx` (β) | `soilw_am`, the bundle's root-zone availability | no bucket (#672) |
| bare-soil `h` | `calc_relative_humidity_upper(soilw_rel)` | see below |
| vegetation ratio `v` | `forest_fraction` (ERA5 high-vegetation cover) | no LAI or veg ratio in the bundle |
| unstressed canopy conductance `g_c` | ECHAM3's formula at LAI 4, every step | no BETHY |
| wet-skin fraction | 0 | no interception reservoir (#672) |
| snow fraction `s` | `snowc_am` of the non-glacier land | no snow mass (#672) |

**Bare-soil humidity.** ECHAM6.3 runs the 5-layer soil (`nsoil = 5` in the
AMIP control's `namelist.jsbach`). For that soil `update_soil` (2492-2499)
calls `calc_relative_humidity_upper` (2754-2777),
`h = ½(1 − cos(π·min(ws₁, fc₁)/fc₁))`, on the upper layer's water against its
field capacity. The bundle carries exactly that ratio as `soilw_rel` (ERA5
swvl1 over the HTESSEL field capacity of the 0-7 cm layer), so this form needs
no invented constant. A bundle without `soilw_rel` falls back to `soilw_am`,
logged once. The single-bucket alternative,
`calc_relative_humidity` (`nsoil = 1`, 2705-2725), needs an absolute bucket
capacity the bundle does not have. With ECHAM's own T63 `WSMX` (land median
0.25 m) it gives `h > 0` only for fills above 0.6. Box-mean March-May `h` for
the two forms, from the bundle's fields:

| box | soilw_am | soilw_rel | h, 5-layer (used) | h, bucket, wsmx 0.25 m | h, bucket, wsmx 0.1 m |
|---|---|---|---|---|---|
| Sahel | 0.18 | 0.24 | 0.19 | 0.02 | 0.12 |
| Mexican plateau | 0.21 | 0.29 | 0.24 | 0.05 | 0.12 |
| India | 0.41 | 0.45 | 0.44 | 0.09 | 0.31 |
| Amazon | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Canopy conductance.** ECHAM6.3 takes the unstressed canopy conductance from
BETHY (photosynthesis, LAI, PAR). JSBACH's path without BETHY
(`mo_jsbach_interface.f90` 770-785) evaluates ECHAM3's formula
(`mo_canopy.f90::unstressed_canopy_cond_par`, ECHAM3 manual eq. 3.3.2.12) and
then overwrites it with a placeholder `1.e-5`, which would switch
transpiration off. jcm evaluates ECHAM3's formula every step:

```
d = (a + b·c)/(c·PAR),   g_c = [ln((d·e^{kL} + 1)/(d + 1))·b/(d·PAR) − ln((d + e^{−kL})/(d + 1))]/(k·c)
```

with `k = 0.9`, `a = 5000`, `b = 10`, `c = 100`, the absorbed PAR taken as
half the land's net shortwave (`par_fraction`, a stand-in for ECHAM's net
visible band, since the radiation here publishes broadband surface fluxes) and
`L = 4` (`leaf_area_index`, a closed canopy for a vegetated fraction that is
the forest fraction). The conductance is 0.019 m/s at PAR 200 W/m² and tends
to `L/600` m/s at night, so the stomata narrow but do not close, as in ECHAM3.
The land albedo keeps its own `MAX(LAI, 2)` floor (`land_albedo`); the two
LAIs are separate stand-ins.

**Timing.** ECHAM computes the factors at the end of step N−1 (JSBACH runs
after the surface solve) and uses them in step N. jcm computes them at the
start of step N from step N's state, so they are one step fresher. The
canopy's exchange velocity `C_h|U|` is the previous step's, which is ECHAM's
lag for `zchl`. A cold start has none, and it takes ECHAM's cold-start
factors: a run that is not a restart sets `zcair = zcsat = 0`
(`mo_surface.f90::init_surface`), so the land does not evaporate on its
first step and `update_soil` forms the factors at the end of it. A carry
without a land exchange velocity is that first step.

## 2. The skin energy balance

### Heat capacity and conductance

JSBACH's surface temperature is its top soil layer's
(`update_soiltemp.f90:142`). Five layers lie below it, and the implicit soil
solve gives the surface the effective capacity
`pgrndcapc = zdz2(1)·Δt + Δt·(1 − D₁)·zdz1(1)` and the flux
`pgrndhflx = zdz1(1)·(C₁ + (D₁ − 1)·T₁)` (213-222), with
`zdz2(1) = ρc₁·cdel(1)/Δt` and `zdz1(1) = κ₁/(cmid(2) − cmid(1))`. The
stand-in holds the second layer at the prescribed ERA5 soil temperature
`stl_am` (`C₁ = stl_am`, `D₁ = 0`):

```
C_s = ρc·cdel(1)                    = 2.25e6 J m⁻³K⁻¹ × 0.065 m      = 1.46e5 J m⁻² K⁻¹
Λ   = ρc·D_soil/(cmid(2) − cmid(1))  = 2.25e6 × 7.4e-7 / 0.1595 m     = 10.4 W m⁻² K⁻¹
G   = Λ·(T_s − stl_am)              (into the ground)
```

`cdel`/`cmid` are JSBACH's 5-layer grid (`mo_soil.f90` 485-497). `ρc` and
`D_soil` are the FAO row-0 values of `VolHeatCap`/`ThermalDiff` (996-997);
FAO classes 1-5 span 1.93-2.48e6 and 6.7-8.7e-7, and the bundle has no FAO
map. SPEEDY's skin uses `clambda = 7 W m⁻² K⁻¹` with no heat capacity, because
its skin is a diagnostic Newton balance; Λ = 10.4 is ECHAM's own top-layer
conductance, of the same order. The `stl_am` reservoir is ERA5's 0-7 cm layer,
so the skin relaxes to the observed monthly soil climatology and its daily
mean stays near it.

**Snow and glacier.** ECHAM grades the top layer between soil and snow by snow
depth, in series (`update_soiltemp.f90` 157-170):
`x = min(h_sn/cmid(2), 1)`, `ρc = x·ρc_sn + (1 − x)·ρc_soil`,
`κ = 1/(x/κ_sn + (1 − x)/κ_soil)`, with `ρc_sn = 634500` and `κ_sn = 0.31`.
The depth comes from the prescribed cover through the bundle's own definition
`snowc = min(1, SWE/60 mm)`: `h_sn = 0.06 m·snowc·ρ_w/330`. At `snowc = 1`
this gives 0.18 m, below `cmid(2) = 0.192` m, so the full-snow branch is never
reached. Glacier land takes ice (`ρc = 2.09e6`, `D = 1.2e-6`), and the two
tiles average by cover as `update_soil` 1834-1840 does. At `snowc = 0.5` the
conductance is 3.4 W m⁻² K⁻¹.

### The balance

`update_surfacetemp.f90` is ported verbatim. With `α = tpfac1 = 1.5` it solves
for the α-weighted implicit surface dry static energy `ŝ = α·s_new + (1 − α)·s_old`.
Emission and saturation are linearised about the step-start value, and the
lowest level is eliminated through its Richtmyer–Morton relations (section 3):

```
zca = L_s·s + L_v·(cair − s),   zcs = L_s·s + L_v·(csat − s)
zcolin = (C_s + Δt·Λ)/c_p + αΔt·[4εσ(s_old/c_p)³/c_p − ρC·(zca·E_q − zcs)·(dq_s/dT)/c_p]
zcohfl = −αΔt·ρC·(E_s − 1)
zcoind =  αΔt·[Rn_old + ρC·F_s + ρC·((zca·E_q − zcs)·q_s,old + zca·F_q) + Λ·(stl − T_old)]
ŝ = (zcolin·s_old + zcoind)/(zcolin + zcohfl)
```

with `Rn_old = SW_net + ε·LW↓ − εσT_old⁴` (`update_soil` 1828-1829). The step
it implies, with `T̂ = ŝ/c_p` and `T_new = tpfac2·T̂ + tpfac3·T_old`, is

```
C_s·(T_new − T_old)/Δt = Rn(T̂) − SH(T̂) − LH(T̂) − Λ·(T_new − stl_am)
```

The atmospheric fluxes are evaluated at the implicit `T̂` and the ground flux
at `T_new`. Every term on the right is the flux the column then receives, so
the balance closes to round-off (`land_coupling_test.py`).

**Melt.** Where the land holds more than ECHAM's critical snow depth
(5.85 mm SWE, from the bundle's cover) or any glacier, ECHAM keeps the surface
at the melting point and spends the excess on melt (`update_soil` 1859-1863,
`update_surf_down.f90` 246-263). jcm does the same: the fluxes are evaluated
at `min(T̂, tmelt)` and the new temperature is `min(T_new, tmelt)`. Prescribed
snow has no mass to lose, so the excess is published as the melt heat flux,
the term that closes the budget.

**What the skin feeds.** The step-start skin temperature is what the land
albedo's melting ramp, the grid surface temperature the radiation solves with
(snapped by `fmask > 0.5`), the surface-layer saturation, buoyancy and
Richardson number, and the Richtmyer–Morton coefficients all see.

**Longwave between radiation calls.** ECHAM holds the absorbed downward
longwave between radiation calls and emits from the current surface
temperature every step, the emission change heating the lowest layer with the
top of atmosphere held (`radheat.f90` 404-410). jcm's radiation runs before the
vertical diffusion, so `EchamSurface` applies the change after the balance:

```
Δ = εσ·T_old³·(4·T_new − 3·T_old) − εσ·T_old⁴      (land_rad's zteffl4, mo_surface_land.f90:612)
```

`Δ` is added to `radiation.surface_lw_up` and to the lowest-level longwave
heating `Δ·g/(c_pd·Δp_K)`. Both are written back into the radiation carry, so
successive steps build on one another the way the shortwave zenith rescale
does.

**Emissivity.** The balance uses jcm's land emissivity (0.95,
`SurfaceOpticsParameters.land_emissivity`), the value the radiation solves
with; ECHAM uses 0.996 for every surface.

## 3. Implicit coupling: Richtmyer–Morton per tile

After the top-down elimination of the column (`vdiff.f90` 887-931), the lowest
level obeys, for each tile `t` with its own exchange coefficient
`k_t = Δt·α·ρ_s·C_t·g/Δp_K` (`richtmyer_land`, `_ocean`, `_ice`):

```
X̂_K,t = E_t·X̂_s,t + F_t,   E_s = k/(D + k),  F_s = α·R/(D + k)
                           E_q = csat·k/(D + cair·k),  F_q = α·R_q/(D + cair·k)
D = 1 + zfac·(1 − zebsh_{K−1}),   R = ztdif_K + zfac·ztdif_{K−1}
```

The prescribed tiles give `X̂_K,t` directly, and the land tile first solves
its balance against its own `E`/`F`. The column's bottom value is the
fraction-weighted blend `bb_K = tpfac2·Σ_t f_t·X̂_K,t` (`blend_zq_zt`), and
back-substitution completes the solve. Each tile's flux is taken against its
own lowest-level value (`postproc_ocean`/`_ice`, `update_soil` 1892-1911).
Since the eliminated bottom row then reads
`D·bb_K = R + tpfac2·Σ_t f_t·k_t·(X̂_s,t − X̂_K,t)`, the grid-mean flux is
exactly what the column receives (`Σ dm·ΔX/Δt = Σ_t f_t·F_t`, the `pev_vdiff`
identity). Momentum keeps one Robin row with the fraction-weighted drag, as
ECHAM's box-averaged coefficient does (`mo_surface.f90` 1205-1219,
`vdiff.f90` 1099-1100).

The port diffuses `T` rather than the dry static energy, with surface value
`T_s − φ_K/c_pd` (`φ_K` the lowest level's geopotential above the surface). In
dry static energy, `s_s = c_pd·T̂_s` and
`s_K = E·s_s + c_pd·F + (1 − E)·φ_K`, which is what the balance reads. The
port's sensible flux `ρ·c_pd·C·(T̂_s − φ_K/c_pd − T̂_K)` is ECHAM's
`ρ·C·(s_s − s_K)` exactly.

**A fidelity gain over the earlier collapsed row.** Before this change the
three tiles shared one Robin row, `(R + Σ f k X)/(D + Σ f k)`. That equals
ECHAM's blend only in single-tile cells, or when the tiles have the same `k`.
In a mixed cell ECHAM's flux differs, and the per-tile form is ECHAM's.
Measured on the reference columns' eliminated bottom rows (`k/D`: median 0.19,
p90 0.27), the difference in the grid-mean sensible flux is:

| mixed cell | collapsed − per-tile, median (p90) |
|---|---|
| sea ice / open water 60/40, water 2 K warmer, `k_water = 2·k_ice` | 3.1 % (4.7 %) |
| sea ice / open water 60/40, water 10 K warmer, `k_water = 3·k_ice` | 11 % (15 %) |
| land / ocean 50/50, `k_ocean = k_land/2`, SST 5 K above the skin | 0.6 % (1.6 %) |
| land / ocean 50/50, `k_ocean = k_land/2`, SST 8 K below the skin | 4.6 % (31 %, where the grid flux nearly cancels) |

Pure-ocean and pure-land cells are unchanged (`land_coupling_test.py`
re-derives the old row there to round-off).

**Time weights.** ECHAM's `ztpfac2 = 1/cvdifts` and `ztpfac3 = 1 − ztpfac2`
(`mo_soil.f90:1684`, `mo_surface_boundary.f90:90`) are exact.
`VDiffParameters.default` derives them from `tpfac1`. With the earlier rounded
0.667/0.333, `α·tpfac2 = 1.0005`, and the balance and the column disagreed on
the sensible flux by 5e-4·ρ·c_p·C·T (median 2.8, p90 3.9 W/m² on the reference
columns).

## 4. Smoothness

The values are the reference's exactly. The derivatives come from named smooth
surrogates ({doc}`surrogate_gradients`) at the switches inside the active
range:

| switch | surrogate | static width |
|---|---|---|
| bare-soil `h > q_a/q_s` | `σ((h − q_a/q_s)/w)` | `hinge_width = 0.02` |
| dew `q_a > q_s`, and the canopy's `q_a ≤ q_s` | `σ((q_a/q_s − 1)/w)` | `hinge_width` |
| stress `clip(·, 0, 1)` | softplus clip | `stress_width = 0.02` |
| melt `min(T, tmelt)` | `T − w·softplus((T − tmelt)/w)` | `melt_width = 0.5 K` |

The canopy factor is written as `g_c·β/(g_c·β + C_h|U|)`, so its slope stays
finite as `β → 0`. JSBACH's separate `w > w_wilt` gate on the canopy is kept in
the value. The surrogate drops it, because the smooth stress already takes the
canopy to zero there and a hard gate would hide the wilting point from the
derivative. The bare-soil `h` is C¹ (flat at both ends) and needs no surrogate.
A width of 0 selects the reference derivative.

## 5. Carry and checkpoints

`SurfaceData.land_surface_temperature` is the prognostic. A value ≤ 0 means
unset: `EchamBoundaryConditions` then seeds the skin from `stl_am`, the value
the land had when it was prescribed. One rule covers a cold start, a carry
restored from a checkpoint written before the field existed (the name-matched
migration fills it with the bootstrapped zeros, {doc}`checkpoint_compatibility`)
and `init=from_state`. No schema bump is needed: only the field set changed.
`jcm/checkpoint_test.py::TestLandSkinTemperatureMigration` strips the fields
from a payload, restores it and checks that the first step seeds from
`stl_am`. In forced-flux mode, and in a composition without radiation, the
land keeps the prescribed `stl_am`.

New outputs on the `surface` namespace: `land_surface_temperature`,
`land_net_radiation`, `land_sensible_heat_flux`, `land_latent_heat_flux`,
`ground_heat_flux`, `snow_melt_heat_flux`, `land_heat_storage`,
`land_evaporation`, `cair`, `csat`, `water_stress_factor`,
`bare_soil_humidity` and `canopy_conductance`. The land budget closes from
output alone:
`land_net_radiation = sensible + latent + ground + melt + storage`.

## 6. Checked against the compiled Fortran

`jcm/data/test/echam_land_reference/` holds inputs and outputs of the
unmodified ECHAM6.3 / JSBACH routines, compiled in a local harness (the ECHAM
source is not in this repository; `provenance.json` records the checksums,
flags and stubs):

- `precalc_land`, `richtmyer_land`, `update_land`, `update_surfacetemp`,
  `update_soiltemp`, `atm_conditions` and `mo_canopy::unstressed_canopy_cond_par`;
- verbatim line ranges of `mo_soil.f90`: the four humidity and stress
  functions and `update_soil`'s canopy and humidity-factor blocks.

The inputs are 324 land columns of the #979 diagnosis control run (12 boxes,
January, April and July), with scans through every switch. `-O0` and `-O2`
builds, and each column run alone or in the batch, are bit-identical.
`jcm/physics/surface/echam/jsbach_land_test.py` reproduces every output in
float64 at round-off. The capacity and conductance extracted from
`update_soiltemp`'s outputs are the stand-in's to 1e-10: 10.4389 W m⁻²K⁻¹ and
146250 J m⁻²K⁻¹ for soil, 15.7241 and 135850 for ice, 1.9436 and 41242.5 for
full snow, and 9.1732 and 6.1774 at 2 and 10 mm of graded snow.

The same set shows the surface-layer exchange coefficients agree with
`precalc_land`'s to 0.1 % in unstable air but not in stable air, where jcm
carries ICON's Mauritsen (2007) functions (#982).

## 7. Measured effect

*(Filled in from the acceptance runs: the 240-day T63 1M run's box budgets
against the control and the evaporation-only arm, and the 10-day A/B per
preset.)*
