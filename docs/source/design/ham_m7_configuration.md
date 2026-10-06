# ECHAM-HAM M7 configuration

The `echam-ham-m7` preset (provisional name) runs jcm's ECHAM physics with an
aerosol population and microphysics that follow ECHAM6.3-HAM2.3 M7 rather than
CAM's MAM4. It is built on the same JAM harness as `echam-jam`: the harness is
core-agnostic (#461), so the configuration is a second population
(`M7_SPEC`), a second core (an adapter over the `m7-jax` package, installed as
the `jcm[m7]` extra), and a set of HAM-faithful process variants selected by
the preset. Nothing in the MAM4 configuration changes: every HAM variant is
either a new term or a switch whose default reproduces the current result bit
for bit.

This page records the design and the reasons for it. Status and validation
results live on jax-gcm#1017.

## Reference configuration

The reference is the ECHAM6.3-HAM2.3 r7492 example run
`setup_templates/run_examples/run_transient_echam6-hamm7`, with every switch
the template does not set taken from its namelist default in the source.

| switch | value | source |
|---|---|---|
| `nham_subm` | 2 (M7) | `mo_ham.f90` default |
| `nwater` | 1 (κ-Köhler lookup, `lut_kappa.nc`) | `mo_ham_m7ctl.f90:84` |
| `nsnucl` | 2 (Kazil–Lovejoy ion-mediated + neutral H2SO4/H2O) | `mo_ham_m7ctl.f90:89` |
| `nonucl` | 1 (organic activation nucleation, forest-scaled, in the PBL) | `mo_ham_m7ctl.f90:95` |
| `lscond`, `lscoag`, `lgcr` | on | `mo_ham.f90:208-212` |
| `nsolact` | from the calendar date | `mo_ham.f90:214` |
| `nsoa` | 0 (no SOA scheme; biogenic OC emitted as primary) | `mo_ham.f90:255` |
| `nseasalt` | 7 (Long et al. 2011 with the Sofiev 2011 SST correction) | template |
| `nwetdep` | 3 (size-dependent in-cloud nucleation + impaction, Croft below-cloud) | template |
| `ndrydep` | 2 (Ganzeveld/Slinn, interactive) | `mo_ham.f90:177` |
| `ndust` | 5 | `mo_ham.f90:189` |
| `npist` | 3 (Nightingale DMS piston velocity) | `mo_ham.f90:164` |
| `naerorad`, `nraddiag` | 1, 1 | template |
| `lcdnc_progn`, `nauto` | true, 2 | template |
| `ncd_activ` | 2 (ARG) | `setphys.f90:78` |
| `nactivpdf` | **1** (West et al. 2013 20-bin updraft PDF) — maintainer decision | template leaves the code default 0 (`setphys.f90:79`) |
| `nic_cirrus` | 2 (Kärcher–Lohmann) | `setphys.f90:80` |

The preset runs ARG over the West et al. (2013) 20-bin updraft PDF
(`nactivpdf = 1`). This deviates from the reference-run template, which leaves
`nactivpdf` at its code default 0 (a single characteristic updraft
`w_large + w_turb`); the maintainer chose the PDF on 2026-10-05.
`jam_nactivpdf: 0` selects the template's single updraft.

The template's run settings that belong to an ECHAM experiment rather than to
the aerosol model (L31, nudging, AMIP SST) are not part of the reference: the
preset runs jcm's T63L47 free-running configuration.

## The M7 population (`M7_SPEC`)

Seven log-normal modes (`mo_ham_m7ctl.f90:150-172`, `mo_ham.f90:320-321`,
`mo_ham_init.f90`). Tracer keys use HAM's two-letter class names in lower case:
`m_so4_ks`, `n_ki`, and so on.

| short | HAM class | σ_g | soluble | species | sediments (`lsed`) | activates (`lactivation`) | `csr_conv` | `caccso4` |
|---|---|---|---|---|---|---|---|---|
| `ns` | NS nucleation | 1.59 | yes | so4 | no | no | 0.20 | 1.0 |
| `ks` | KS Aitken | 1.59 | yes | so4 bc oc | no | yes | 0.60 | 1.0 |
| `as` | AS accumulation | 1.59 | yes | so4 bc oc ss du | yes | yes | 0.99 | 1.0 |
| `cs` | CS coarse | 2.0 | yes | so4 bc oc ss du | yes | yes | 0.99 | 1.0 |
| `ki` | KI Aitken | 1.59 | no | bc oc | no | no | 0.20 | 0.3 |
| `ai` | AI accumulation | 1.59 | no | du | yes | no | 0.40 | 0.3 |
| `ci` | CI coarse | 2.0 | no | du | yes | no | 0.40 | 0.3 |

Stratiform in-cloud fractions `csr_strat_wat/mix/ice` and the below-cloud
coefficients `cbcr`/`cbcs` (`mo_ham_m7ctl.f90:515-526`) are carried on the
spec for the `nwetdep = 1/2` variants.

**Size bounds.** M7's mode boundaries are dry *radii* `crdiv` = 0.0005, 0.005,
0.05, 0.5 µm (`mo_ham_m7ctl.f90:164`); the spec stores diameters, so
`dgnum_lo/dgnum_hi` are 1 nm–10 nm (NS), 10–100 nm (KS, KI), 0.1–1 µm (AS, AI)
and 1–10 µm (CS, CI). M7 has no coarse upper bound; 10 µm is the edge of HAM's
sea-salt and dust size integrations. `dgnum` is the geometric midpoint of the
bounds. These fields are harness geometry (emission windows, the placeholder
core, initial `_jam_state`), never M7 state: the core computes every mode's
radius from its own mass and number each step, and every M7 emission passes its
own HAM size (below), so `dgnum` is a fallback that the M7 preset never reaches.

**Species.** `M7_SPEC` carries its own `AerosolSpecies` with HAM's properties
(`mo_ham_species.f90:154-430`), so the global MAM4 `SPECIES` table is untouched:

| token | MW [g/mol] | ρ [kg/m³] | κ | electrolyte (`nion`) |
|---|---|---|---|---|
| `so4` | 96.0631 | 1841 | 0.60 | 2 |
| `bc` | 12.01 | 2000 | 0.0 | 0 |
| `oc` | 180.0 | 2000 | 0.06 | 0 |
| `ss` | 58.443 | 2165 | 1.0 | 2 |
| `du` | 250.0 | 2650 | 0.0 | 0 |

`oc` is a new token rather than a reuse of MAM4's `poa`: HAM's OC tracer holds
organic matter from primary sources and from biogenic emission alike
(`nsoa = 0`), and the M7 reference data and diagnostics name it `oc`. Every
harness lookup that needs a molar mass (aqueous sulfate, primary sulfate from
SO₂) reads it from the population, so the M7 sulfate is SO₄ at 96.06 g/mol while
the MAM4 value (115 g/mol, ammonium bisulfate) is unchanged.

**Cloud-borne phase.** `cloud_borne = False`: HAM scavenges interstitial aerosol
by its activated fraction; there is no `mc_*`/`nc_*` phase.

## Core: `m7-jax` and the jcm adapter

The microphysics lives upstream in `reflective-org/M7-JAX` (BSD-3, alongside the
HAMMOZ Consortium's BSD-3 M7 Fortran it is validated against) and is pinned
exactly as the `jcm[m7]` extra, as `jcm[mam4]` pins `mam4-jax`. Version 0.2
widens the package from pure sulfate to the full HAM M7 state.

### Native state (per cell)

| field | shape | units | content |
|---|---|---|---|
| `mass` | (18,) | SO₄: molecules cm⁻³; others: µg m⁻³ | HAM `aerocomp` order: SO₄ NS KS AS CS; BC KS AS CS KI; OC KS AS CS KI; SS AS CS; DU AS CS AI CI |
| `number` | (7,) | cm⁻³ | NS KS AS CS KI AI CI |
| `h2so4` | () | molecules cm⁻³ | gas at the start of the step (`zgso4`) |

The units are M7's own (`mo_ham_subm.f90` `ham_subm_interface`): the core is a
direct port of `m7`, and keeping its units keeps every threshold (`cmin_aernl`,
`cmin_aerml`, `cminvol`, …) at its native value.

### Inputs the host supplies (per cell)

| input | units | jcm source |
|---|---|---|
| temperature, pressure | K, Pa | step-start state, `pressure_full` |
| relative humidity, **clear-sky** | 0–1 | `(q − q_s·c)/(1 − c)`, `c = min(c_cloud, 1 − 1e-10)`, `q_s` from ECHAM's Sonntag water table (`jcm.physics.thermodynamics`), clipped to [0, 1] (`mo_ham_subm.f90:~250`) |
| H2SO4 production | molecules cm⁻³ s⁻¹ | the running H2SO4 tendency of the step so far (gas chemistry + transport), `_tendency_run["tracers"]["g_h2so4"]` |
| cloud fraction | 0–1 | previous step's `clouds.cloud_fraction` (ECHAM's `paclc` at the M7 call is the previous cover) |
| ion-pair production rate | cm⁻³ s⁻¹ | GCR ionisation (`mo_ham_gcrion.f90`) computed in jcm, host-side as ECHAM computes it (`ham_subm_interface` calls `gcr_ionization` before `m7`), from the geomagnetic cut-off rigidity at the column's position, pressure, temperature and the solar activity of the model date |
| forest fraction | 0–1 | `ForcingData.forest_fraction` |
| in-PBL mask | bool | level index ≥ the PBL-top level, from the previous step's TTE-TKE PBL height (`nucl_activation`'s `jk ≥ int(ppbl)`) |
| time step | s | `_dt_seconds` |

Static options (compile-time): `nwater`, `nsnucl`, `nonucl`, `lscond`,
`lscoag`, selected from the factory via
`jam_microphysics_options` (e.g. `{nucleation_scheme: 2}`). Tables passed
in, never read inside a traced function: the κ lookup (`lut_kappa.nc`,
shipped with `m7-jax`) and, for `nsnucl = 2` (the preset's default), the
Kazil–Lovejoy table and the GCR ion-pair tables — see the Data table below.
`nsnucl = 2` needs `$HAM_INPUT_DIR` to hold them; construction raises,
naming it, rather than silently running `nsnucl = 1` if it is unset.

### Outputs

The updated state plus, per mode, the wet count-median radius (all seven), the
dry count-median radius (soluble four; an insoluble mode's dry radius is its
radius), the particle density and the aerosol water (`mo_ham_subm.f90`
post-call block).

### Operator-split mapping

jcm sums every term's tendency computed against the step-start state; ECHAM
calls M7 on `pxtm1 + pxtte·Δt`, the state with every earlier process of the step
applied, and then overwrites `pxtte` so that the step ends at M7's result. The
adapter reproduces this exactly: it builds the M7 input from
`state + _tendency_run·Δt` (aerosol and number), passes the H2SO4 gas as its
step-start value with the accumulated H2SO4 tendency as the production rate
(ECHAM's `zgso4m1`/`zdgso4`), and returns `(x_M7 − (x₀ + run·Δt))/Δt`, so the
summed tendency over the step is `(x_M7 − x₀)/Δt`.

### Mass conversion

Sulfur is counted in molecules across the boundary: the H2SO4 gas tracer
(kg/kg of H2SO4, 98.08 g/mol) and the aerosol sulfate tracers (kg/kg of SO₄,
96.06 g/mol) are both converted to molecules cm⁻³ with their own molar mass, so
the conversion conserves sulfur exactly; other species convert to µg m⁻³ with
the air density, number to cm⁻³.

### `_jam_state`

`r_dry` = M7 dry radius for soluble modes and the radius for insoluble ones (as
`ham_subm_interface` fills `rdry`), `r_wet`, `rho` = particle density × 10³,
`mass` and `number` from the post-call state, and `kappa` = the volume-weighted
species κ of each mode, so the existing κ-based consumers (CAM ARG, the 2M
scheme's diagnostics) work unchanged on the M7 population.

### Precision

`m7-jax` passes its full test suite on jax 0.10.2, the line jcm pins
(`jax>=0.10,<0.11`), so the extra does not need a newer jax. The core runs in
float64 by default, as `mam4-jax` does; a `core_dtype="float32"` forward-only
option runs it in float32 under a scoped `jax.enable_x64(False)` with boundary
casts.

## HAM-faithful variants

Each is selectable and off by default; the M7 preset turns them on.

| process | HAM | jcm default (unchanged) | variant |
|---|---|---|---|
| activation | Köhler A/B from electrolyte species (`nion > 0`: SO₄, SS), ARG over KS AS CS, single updraft `w_large + w_turb` (`nactivpdf = 0`) or the 20-bin PDF; Lin–Leaitch as `ncd_activ = 1` | CAM ARG on κ | `ham_arg`, `ham_lin_leaitch` |
| optics | four Mie tables (SW/LW × fine/coarse σ), nearest-neighbour on HAM's axes, volume-mixed RI incl. water | Gauss–Hermite over jcm's Mie LUT | HAM-LUT `_mode_optics` backend, tables built with jcm's Mie kernel on HAM's axes |
| wet deposition | `nwetdep = 3` | CAM Slinn impaction + ledger | HAM nucleation + impaction scavenging, Croft below-cloud tables |
| aqueous sulfate | new SO₄ to AS/CS by number; CS with new number if both empty | single mode | number-weighted AS/CS split |
| emissions | sector-dependent sizes and modes (`mo_ham_m7_emissions.f90`) | per-species split at class geometry | per-sector policy on the population |
| sea salt | `nseasalt = 7` (Long + SST), AS/CS cut at 0.551 µm dry diameter | Gong (`nseasalt = 6`) | Long scheme |
| dust | BGC bin 1 → AI, bins 2–4 → CI, 16.1 µm cut | MAM4 windows | windows from the population |
| cirrus | `nic_cirrus = 2` Kärcher–Lohmann | `nic_cirrus = 1` | ported, fed by M7 `pascs/papnx/paprx/papsigx` |

## Reference numbers

Faithful means agreeing with compiled reference Fortran. The core is compared
against the native `m7` built from the BSD-3 sources that `m7-jax` ships. The
harness variants are compared against the unmodified r7492 routines compiled in
a scratch harness outside the repository (the `echam_hamfrz` precedent); only
the resulting numbers, with provenance, are committed under `jcm/data/test/`.

The SALSA-box copy of `m7_averageproperties` that `m7-jax` ships differs from
r7492 in one place: it takes the cube root of the insoluble-mode *density* as if
it were the mean particle volume, giving every insoluble mode a radius of about
0.57 cm. r7492 computes the radius from the volume. The `m7-jax` oracle restores
the r7492 text at build time, leaving the shipped file byte-identical to its
pinned hash, so the reference the port is held to is the reference
configuration's.

## Data

| input | status |
|---|---|
| κ lookup `lut_kappa.nc` | shipped with `m7-jax` |
| Kazil–Lovejoy table `parnuc.15H2SO4.A0.total.nc` | HAMMOZ input pool, staged under `$HAM_INPUT_DIR`; read outside any traced function and passed to the core like the κ table, stored as float64 numpy and rebuilt at the core's own working dtype per step (so it composes with the float32 forward core too, #1017 task 6). The preset defaults to `nsnucl = 2` (`jam_microphysics_options: {nucleation_scheme: 2}`); construction raises, naming `HAM_INPUT_DIR`, if this table is absent — there is no silent fallback to `nsnucl = 1`. `+physics.jam_microphysics_options.nucleation_scheme=1` opts into binary nucleation explicitly where this data is unavailable |
| GCR ion-pair tables `solmin.txt`/`solmax.txt` (ECHAM's own `gcr_ipr_solmin.txt`/`gcr_ipr_solmax.txt`, or README_GCR.txt's `SOLMIN.txt`/`SOLMAX.txt`) | HAMMOZ input pool, staged under `$HAM_INPUT_DIR` alongside the Kazil table; read by jcm's GCR ionisation term (`gcr_ionisation.py`), dtype-generic like the Kazil table, same reason |
| Mie tables | built by jcm's Mie kernel on HAM's axes (the SALSA repository's `lut_optical_properties*_M7.nc` are header-only stubs) |
| anthropogenic, biomass-burning emissions | jcm's CEDS/BB4CMIP bundle, sized by HAM's per-sector rules; the residential (`DOM`) and energy (`ENE`) sectors need separate channels, added alongside the existing super-sector channels so the MAM4 inputs do not change |
| biogenic OC | HAM's own AeroCom II climatology (`emiss_aerocom_OC_monthly_2000`), converted into the emissions bundle as `emis_biogenic_oc`; HAM's split: 35 % KI at 0.03 µm, 32.5 % KS and 32.5 % AS without number, no OM:OC factor (`nsoa = 0`) |
| oxidants, dust sources, DMS | jcm's existing inputs |

## Default-path invariance

The MAM4 presets are calibrated and frozen for v3.0. Every change to a shared
harness module is behind the population (a field whose default reproduces the
MAM4 value) or a new term. Each pull request is checked three ways: the existing
test suite unchanged; a step-level comparison of the full saved state of the
placeholder and MAM4 JAM compositions (with and without cloud-borne aerosol,
with synthetic emissions on every sector) and the MACv2-SP 2M stack against the
pre-change code, which must agree in every bit; and the release-matrix
statistics.

## Secondary organic aerosol (`nsoa = 1`, selectable)

The preset keeps the reference template's `nsoa = 0`. With that setting, biogenic
SOA enters as a prescribed primary organic source (the AeroCom II biogenic OC
above). The interactive scheme of O'Donnell et al. (2011) is a selectable variant
(`mo_ham_soa.f90`, `mo_ham_soa_processes.f90`). `nsoa = 2` (volatility basis set)
exists only with SALSA and is out of scope.

| part | HAM | jcm |
|---|---|---|
| precursors | 11 gas tracers: α-pinene, t-β-ocimene, β-pinene, limonene, sabinene, myrcene, 3-carene, isoprene, toluene, xylene, benzene | gas tracers `g_<token>` |
| products | with `nsoalumping = 0` (default): 7 semi-volatile species (monoterpene SOA 1/2, isoprene SOA 1/2, toluene, xylene, benzene SOA). Each has a gas-phase tracer and an aerosol tracer in every class with `lsoainclass` (KS AS CS KI), plus 4 monoterpene/isoprene "total" tracers | the same tracers; `M7_SOA_SPEC` is the M7 population with the SOA species added to KS AS CS KI |
| chemistry | `soa2prod`: precursor oxidation by OH/O₃/NO₃ into products with fixed yields | prescribed oxidants (`PrescribedOxidants`) |
| partitioning | `soa_equi0`/`soa_equi1`, `soa_part`: absorptive gas–particle equilibrium (temperature-dependent Kp) over the organic mass of each class | a gas-chemistry-side term ahead of the core |
| microphysics | the `nsoa == 1` branches of `m7` (`m7_dconc`, `m7_delcoa`: SOA moves with its class) | `m7-jax`, a second upstream release |
| emissions | biogenic precursors online from MEGAN (`mo_hammoz_emi_biogenic.f90`: PFT emission factors, LAI, temperature, PAR; inputs `megan_*_T63.nc`); anthropogenic and fire aromatics, isoprene and terpenes from the inventory (`emi_spec_hammoz_default+isoa_transient_ham.txt`) | MEGAN term on jcm's land-surface fields; the inventory species as extra channels in the emissions bundle |

**Cost.** Tracer count is about 11 + 7 + 4 + 7 × 4 = 50 on top of the M7
preset's 25 aerosol and 4 gas tracers. Tracer transport dominates the dycore step
at T63L47, so the variant runs at roughly half the preset's speed.
