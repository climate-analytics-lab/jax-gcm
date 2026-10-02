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

**Shortwave between radiation calls.** The radiation holds the downward and
upward surface shortwave between solves (rescaled to the sun,
`rescale_cached_radiation`), so the land absorbs the held downward flux
through the land albedo of the same solve: the radiation term writes it to
`surface.land_albedo_at_solve` when it solves (`hold_land_albedo`, from the
`land_albedo` that `EchamBoundaryConditions` publishes every step for the
next solve) and the vertical diffusion reads it in place of the step's own
albedo. ECHAM does the same: JSBACH takes the radiation's net shortwave
(`mo_jsbach_interface.f90`, `swnet`) and moves its interactive albedo only at
a radiation step. A skin crossing the snow-albedo ramp between solves therefore
absorbs `SW↓·(1 − α_solve)`, and in an all-land cell the land's net equals the
held `SW↓ − SW↑`. A value `<= 0` is unset (a cold start before the first
solve, a checkpoint from before the slot existed, a radiation term that does
not hold it) and the step's own albedo is used.

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
does. The heating advances the running thermodynamic state (`thermo_run`) like
every other temperature tendency, so the convection and cloud schemes after
`EchamSurface` see it, as they see `radheat`'s tendency in ECHAM (`physc.f90`
runs `radheat` before `cucall`, which forms its environment from `ptm1 +
ptte·dt`).

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
`land_evaporation`, `land_energy_residual`, `cair`, `csat`,
`water_stress_factor`, `bare_soil_humidity`, `canopy_conductance` and
`land_albedo_at_solve`. The land
budget closes from output alone:
`land_net_radiation = sensible + latent + ground + melt + storage`, and
`land_energy_residual = land_net_radiation − sensible − latent` is
`ground + melt + storage`. With `land_temperature = "prescribed"` the skin is
the forcing's every step, nothing is solved, ground, melt and storage are 0,
and the residual is the heat the prescription supplies or removes (the
fixed-land-temperature configuration, {doc}`../science/surface`).

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

**240 days of `t63-echam-1m`.** T63L47 from the 1M spin-up, 1 January to
28 August, 5-day means, against the same model with the prescribed land
temperature and the beta evaporation form (dev 80699ac8), and against an arm
with JSBACH's bare-soil form alone and the prescribed temperature. Box means
over land (land fraction ≥ 0.5, cos-latitude weights); observations are
MERRA-2 latent heat and GPCP precipitation (2001-2020). `S = H + LH − Rn` is
the grid surface surplus: the energy the surface hands the atmosphere beyond
what it absorbs, which a real land closes with its ground-heat flux.

| box, season | LH: prescribed / form only / this land / MERRA-2 [W m⁻²] | P: … / GPCP [mm d⁻¹] | S: prescribed / form only / this land [W m⁻²] |
|---|---|---|---|
| Sahel, MAM | 149 / 63 / 37 / 14 | 7.6 / 2.7 / 2.0 / 0.6 | 157 / 38 / 17 |
| Mexican plateau, MAM | 64 / 1 / 27 / 20 | 9.5 / 3.8 / 3.0 / 0.6 | 102 / 38 / 33 |
| India, MAM | 308 / 138 / 75 / 24 | 26.8 / 12.6 / 8.0 / 0.7 | 347 / 132 / 30 |
| Amazon, MAM | 113 / 137 / 108 / 128 | 6.2 / 9.1 / 7.3 / 9.1 | −19 / 37 / 3 |
| Congo, MAM | 166 / 170 / 101 / 114 | 13.8 / 14.8 / 8.8 / 5.0 | 111 / 121 / 12 |
| Sahel, JJA | 179 / 109 / 68 / 55 | 16.3 / 9.5 / 6.7 / 3.9 | 207 / 99 / 32 |
| Mexican plateau, JJA | 134 / 22 / 60 / 50 | 10.9 / 6.1 / 7.9 / 2.9 | 129 / −26 / 10 |
| India, JJA | 290 / 261 / 142 / 93 | 26.4 / 20.4 / 10.5 / 8.0 | 239 / 211 / 25 |
| Amazon, JJA | 134 / 130 / 110 / 124 | 9.1 / 7.4 / 3.5 / 3.2 | 48 / 50 / 15 |
| Congo, JJA | 123 / 122 / 95 / 98 | 7.1 / 7.5 / 4.7 / 3.4 | 60 / 57 / 10 |
| Gobi, MAM | 12 / 1 / 7 / 12 | 3.0 / 1.9 / 2.0 / 0.4 | −43 / −49 / −21 |

- The land tile's own budget closes from output to round-off in every box
  (`Rn = SH + LH + G + melt + storage`). What remains of the grid surplus is
  the ground-heat flux from the prescribed soil: the skin runs 0.5-3.6 K
  colder than ERA5's soil-temperature climatology `stl_am` in these boxes, and
  the soil held at `stl_am` supplies the difference (Sahel JJA 32, India JJA
  26, Amazon JJA 16 W m⁻²). A prognostic soil temperature (#672) would let it
  run down.
- The Gobi's deficit has other sources and does not close with the land
  surface: in MAM, 13 W m⁻² melts a prescribed snow cover that never thins,
  and the skin is warmer than `stl_am` (8 W m⁻² into the soil).
- The wet tropics evaporate 3-15 % below MERRA-2: Amazon 108 against 128
  (MAM) and 110 against 124 (JJA), Congo 101 against 114 and 95 against
  98 W m⁻². The Amazon's dry-season (JJA) evaporation is 18 % below the
  prescribed land's, which was 8 % above MERRA-2; its soil is wet (β ≈ 0.9),
  so the canopy conductance limits it. Congo, where the prescribed land
  evaporated 25-45 % above MERRA-2, comes down to it, and its precipitation
  falls with its surplus.
- Days 30-240, global means: precipitation 2.96 → 2.67 mm d⁻¹ (land
  4.31 → 2.51, ocean 2.41 → 2.74; land 40°S-40°N 6.83 → 3.66 against GPCP's
  2.74), land latent heat 77 → 47 W m⁻², land sensible heat 36 → 42 W m⁻²,
  OLR 242.6 → 239.6 W m⁻² and net TOA radiation 1.95 → 7.47 W m⁻². The
  tropical Atlantic's MAM rain rises from 0.63 to 1.37 mm d⁻¹ (GPCP 6.24) and
  its JJA rain from 1.78 to 4.29 (7.61).
- The water-positivity correction the physics applies after the transport
  stays at 0.0002-0.0003 mm d⁻¹ over land.

**The diurnal cycle.** Composites by local solar time over days 30-240 from
the 3-hourly snapshots (eight bins; first harmonic by vector averaging,
Covey et al. 2016):

| region | sensible heat: mean / 1st-harmonic amplitude / hour of max | convective rain: mean [mm d⁻¹] / amplitude / hour of max |
|---|---|---|
| tropical land, prescribed | 47 / 2.7 / 05 | 8.6 / 9 % / 02 |
| tropical land, this land | 50 / 79 / 13 | 4.4 / 136 % / 14 |
| Sahel, this land | 66 / 100 / 13 | 3.6 / 122 % / 14 |
| Amazon, this land | 29 / 52 / 13 | 5.1 / 162 % / 14 |
| tropical ocean, this land | 21 / 1.1 / 08 | 3.9 / 14 % / 03 |

The land now heats its boundary layer by day (Sahel sensible heat 211 W m⁻²
in the 12-15 h bin) and its convection follows: deep convective rain peaks
in the early afternoon, the near-noon timing known for ECHAM's Tiedtke
convection over land (Bechtold et al. 2004), earlier than the observed late
afternoon (Dai 2006). The ocean keeps its nocturnal maximum.

**Ten days of each preset.** Days 5-10 of T63L47 members from the presets'
spin-ups (January), this land against the prescribed one (dev 80699ac8);
land and ocean split by the land tile's fraction (> 0.5). Values are the
prescribed land's, then the change:

| quantity | 1M | 2M | JAM |
|---|---|---|---|
| net TOA [W m⁻²] | 0.68, +4.10 | 8.93, +4.12 | 9.20, +3.69 |
| … over land | −69.7, +8.0 | −63.7, +11.0 | −64.1, +12.0 |
| SW CRE [W m⁻²] | −47.2, +0.9 | −50.9, +1.0 | −50.3, +0.7 |
| LW CRE [W m⁻²] | 14.8, +0.3 | 26.8, +0.6 | 26.9, +0.6 |
| OLR [W m⁻²] | 246.8, −3.3 | 234.9, −3.2 | 234.5, −3.2 |
| cloud cover [%] | 55.3, +0.0 | 60.2, −1.2 | 61.9, −0.9 |
| … over land | 52.5, −4.5 | 56.4, −7.9 | 57.0, −8.3 |
| LWP [g m⁻²] | 69.8, −3.7 | 40.7, −3.4 | 45.1, −4.2 |
| IWP [g m⁻²] | 18.7, −1.1 | 26.7, −0.7 | 28.7, −1.8 |
| precipitation [mm d⁻¹] | 2.65, −0.28 | 2.59, −0.27 | 2.70, −0.31 |
| … convective | 2.05, −0.21 | 1.93, −0.22 | 1.99, −0.22 |
| … large-scale | 0.60, −0.07 | 0.65, −0.05 | 0.71, −0.09 |
| … over land | 3.70, −1.58 | 3.48, −1.50 | 3.36, −1.55 |
| … over ocean | 2.22, +0.25 | 2.22, +0.23 | 2.43, +0.19 |
| column water vapour [kg m⁻²] | 25.05, −0.06 | 24.92, −0.27 | 24.80, −0.20 |
| land latent heat [W m⁻²] | 62.2, −29.5 | 63.2, −30.7 | 64.1, −30.0 |
| land sensible heat [W m⁻²] | 32.0, −2.4 | 28.9, −2.6 | 28.8, −2.7 |

In every preset the land evaporates about half as much, its rain falls by
43-46 %, the ocean rains 8-11 % more, and the land loses cloud, which
raises net TOA radiation by about 4 W m⁻² globally. The step costs the same:
the steady 5-day chunk of the 1M member takes 77.8 s against 77.6 s on one
A100.

**JAM's dust.** The dust scheme reads the surface layer's 10 m wind and
friction velocity, so the land surface reaches it through the wind over the
source regions. In the JAM member (days 5-10, global; source columns are the
prescribed-land member's columns with dust emission):

| quantity | prescribed land | this land | change |
|---|---|---|---|
| dust emission [Tg yr⁻¹] | 1741 | 1154 | −34 % |
| … North Africa / Arabia / East Asia / Australia | 416 / 177 / 9 / 1001 | 254 / 257 / 15 / 487 | −39 / +45 / +60 / −51 % |
| dust dry / wet sink [Tg yr⁻¹] | 1277 / 366 | 933 / 130 | −27 / −64 % |
| dust burden [Tg] | 5.7 | 4.6 | −19 % |
| 10 m wind over the sources, land tile [m s⁻¹] | 5.10 | 4.55 | −11 % |
| friction velocity over the sources [m s⁻¹] | 0.296 | 0.263 | −11 % |
| saltation gate, land mean | 10.4 % | 8.4 % | −19 % |
| soil-wetness gate, land mean | 0.673 | 0.673 | prescribed |
| SO₄ / BC burden [Tg] | 2.34 / 0.115 | 2.55 / 0.122 | +9 / +6 % |

The source winds weaken by about a tenth, and the emission, a threshold
function of the friction velocity, falls by a third. The dust calibration
(`jam_dust_nduscale_scale`, set in #808 against the prescribed land's wind
distribution) therefore needs redoing on this surface, as does everything
calibrated against JAM's dust, including the immersion-freezing ice nuclei
that follow it.
