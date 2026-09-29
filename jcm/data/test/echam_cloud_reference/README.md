# ECHAM6.3 cloud cover and 1-moment cloud reference data

Inputs, outputs and intermediate quantities of the **unmodified** ECHAM6.3
routines `mo_cover.f90::cover` and `mo_cloud.f90::cloud` (the 1-moment branch
that `physc.f90:1067` calls when `.NOT. lcdnc_progn`), run in double precision
on single columns by a standalone harness. They are the numerical reference
for `jcm/physics/clouds/sundqvist.py` (cover) and
`jcm/physics/clouds/echam_1m.py` (1M), compared by
`jcm/physics/clouds/echam_fortran_reference_test.py`.

These files hold **data only**. ECHAM6 is under the MPI-M Software Licence
Agreement; no ECHAM source is part of this repository. The harness compiles
ECHAM from a local ECHAM6.3 tree at build time. `provenance.json` records the
ECHAM revision and file checksums, compiler and flags, the constants, every
stub and patch, the integrity checks and how to regenerate.

## Files

| file | content |
|---|---|
| `cover_T63L47.npz` | 25 cover columns (14 synthetic, 11 realistic); inputs, `paclc`/`knvb`/`printop` for three saturation variants, cover intermediates for all three |
| `cloud_T63L47.npz` | 42 1M columns (31 synthetic, 11 realistic); inputs, every INOUT/OUT argument for three variants, 73 intermediates for the `sonntag` variant |
| `resolution_T31_T127_T255.npz` | the same columns run with the T31, T127 and T255 constants of `sucloud` (`sonntag` variant) |
| `known_gaps.json` | comparisons that fail against today's jcm (the test marks them `xfail(strict=True)`) |
| `provenance.json` | how the data were produced |

## Conventions

* **Layout.** Every level field is a 2-D array shaped `(nlev, ncol)` =
  `(47, ncol)`; half-level fields are `(48, ncol)`; per-column fields are
  `(ncol,)`. Column `j` is `meta/names[j]`.
* **Vertical ordering: top-first**, as in ECHAM's argument arrays. Index 0 is
  the model top (full level about 1 Pa), index 46 the lowest full level;
  `paphm1[0]` = 0 Pa and `paphm1[47]` = surface pressure. This is the ECHAM
  argument order. jcm's model *output* files are surface-first, and jcm's
  physics works top-first. Select levels by pressure (`in/papm1`), not by index.
* **Grid.** ECHAM6.3 / ICON L47 hybrid levels (`in/vct` = ECHAM `vct`, a in
  Pa then b; same values as `jcm.physics.echam.echam_levels.get_echam_levels(47)`),
  full level pressure = mean of the bounding half levels.
* **Time step.** `in/ptime_step_len` = 1200 s (ECHAM T63 leapfrog
  `time_step_len` = 2 x `delta_time`); `in/pdelta_time` = 600 s is used only
  for ECHAM's accumulated diagnostics.
* **Truncation.** `in/nn` = 63: `sucloud` sets the T63 constants
  (`param/<variant>/*`). `jbmin` = 40, `jbmax` = 45 (1-based level indices)
  come from `sucloud`'s own search on `in/vct`.
* Integers (`ktype`, `kctop`, `knvb`, `jbmin`, ...) are ECHAM's 1-based
  Fortran values.

## Saturation variants

ECHAM takes saturation vapour pressure from lookup tables
(`mo_echam_convect_tables.f90`). The tables tabulate **Sonntag (1990)**: over
water in `uaw` everywhere, and over ice for T <= tmelt in `ua`. Values are
`e_s * rd/rv`, cubic Hermite splines on 0.025 K knots with analytic knot
derivatives (lines 42-52, 262-309, 394-437). Evaluated every 1e-4 K from
150 K to 330 K, the tables reproduce the formula to 4.0e-12 (relative) in
`e_s` and 1.8e-9 in `de_s/dT`.

| variant | what it is |
|---|---|
| `table` | ECHAM as it runs (the spline tables) |
| `sonntag` | the tables replaced by the analytic Sonntag formula (harness module). This is ECHAM's formulation without table interpolation, and the **primary reference**. Its outputs differ from `table` by < 3e-11 of each column's scale. |
| `tetens` | **not ECHAM**: jcm's Tetens `e_s` (water 17.27/237.3, ice 21.87/265.5) inside the otherwise unmodified routines. It lets a comparison separate formulation errors from the saturation formula. Tetens differs from Sonntag by up to 0.14 % above 273 K, 1.2 % (ice) / 2.4 % (water) between 238 and 273 K, and 8 % (ice) / 16 % (water) between 200 and 238 K. |

The realistic cloud columns take `paclc`/`knvb` from each variant's own cover
output (the ECHAM chain), so they are stored per variant in
`in_variant/<variant>/paclc|knvb`. For the synthetic columns all three are
identical and equal the designed inputs.

## Keys

### Inputs (`in/*`): ECHAM argument names, units as in ECHAM

| key | shape | units | meaning |
|---|---|---|---|
| `paphm1` | (48, ncol) | Pa | half-level pressure (n-1) |
| `paphp1` | (48, ncol) | Pa | half-level pressure (n+1); = `paphm1` (read only by the disabled submodel hook) |
| `papm1` | (47, ncol) | Pa | full-level pressure (n-1) |
| `papp1` | (47, ncol) | Pa | full-level pressure (n+1); = `papm1` (submodel hook only) |
| `ptm1` | (47, ncol) | K | temperature at step start |
| `pqm1` | (47, ncol) | kg/kg | specific humidity at step start |
| `pxlm1`, `pxim1` | (47, ncol) | kg/kg | cloud liquid / ice at step start |
| `ptvm1` | (47, ncol) | K | virtual temperature `T(1 + vtmpc1 q - (xl + xi))` (physc.f90:267) |
| `pcair` | (47, ncol) | J/kg/K | moist heat capacity `cpd + cpd vtmpc2 max(q,0)` (physc.f90:289) |
| `pgeo` | (47, ncol) | m2/s2 | full-level geopotential above the surface (geopot.f90 with auxhyb.f90 alpha/lnpr, from `ptvm1`) |
| `pacdnc` | (47, ncol) | 1/m3 | droplet number: the physc.f90 section 3.12 profile (land 180, sea 80 cm-3 below 800 hPa, 20 cm-3 aloft) |
| `paclc` | (47, ncol) | 1 | cover input to `cloud` (synthetic: designed; realistic: see `in_variant`). In `cover_T63L47.npz`: cover's INOUT input (overwritten by cover) |
| `ptte` | (47, ncol) | K/s | accumulated temperature tendency before `cloud` |
| `pqte` | (47, ncol) | kg/kg/s | accumulated humidity tendency |
| `pxlte`, `pxite` | (47, ncol) | kg/kg/s | accumulated liquid / ice tendency (advective + vdiff) |
| `pxtecl`, `pxteci` | (47, ncol) | kg/kg/s | convective detrainment of liquid / ice |
| `pqtec` | (47, ncol) | kg/kg/s | convective humidity tendency (only clipped by `cloud`, not used) |
| `pvervel` | (47, ncol) | Pa/s | vertical velocity (only used when `cauloc > 0`; ECHAM6.3 has `cauloc = 0`) |
| `ktype` | (ncol,) | - | convection type (0 none, 1 deep, 2 shallow, 3 mid-level) |
| `kctop` | (ncol,) | - | convective cloud-top level index (1-based, top-first); for realistic columns the highest level with updraft mass flux |
| `knvb` | (ncol,) | - | inversion level from `cover` (cloud file; 1 for synthetic columns) |
| `pfrw`, `pfri` | (ncol,) | 1 | open-water and sea-ice fractions (cover file; physc.f90:402-403, land = 1 - pfrw - pfri) |
| `land` | (ncol,) | bool | land (land fraction >= 0.5), used for `pacdnc` |
| `paclcov`, `paprl`, `pqvi`, `pxlvi`, `pxivi`, `paprs`, `pch_concloud`, `pcw_concloud` | (ncol,) | | accumulated INOUT diagnostics, 0 on input |
| `vct` | (96,) | Pa, 1 | hybrid coefficients a(1:48), b(1:48) |
| `nn`, `ptime_step_len`, `pdelta_time` | scalar | -, s, s | truncation and time steps |

### Outputs (`out/<variant>/*`)

`cover`: `paclc` (47, ncol) cloud cover; `knvb` (ncol,) inversion level
(1 = none found); `printop` (ncol,) 100 where the inversion enhancement was
applied, else 0.

`cloud`: every INOUT/OUT argument after the call. The tendencies are the
input plus the routine's contribution. The contribution to `pxlte`/`pxite`
also includes the detrainment `pxtecl`/`pxteci`, which `cloud` adds
(mo_cloud.f90:1254-1255). So the routine's own increments are
`ptte - in/ptte`, `pqte - in/pqte`, `pxlte - in/pxlte - in/pxtecl` and
`pxite - in/pxite - in/pxteci`.

| key | units | meaning |
|---|---|---|
| `ptte`, `pqte`, `pxlte`, `pxite` | per s | tendencies after `cloud` |
| `pxtecl`, `pxteci`, `pqtec` | kg/kg/s | inputs after `cloud`'s `MAX(., 0)` clip |
| `paclc` | 1 | cover after the ccwmin write-back (mo_cloud.f90:1280) |
| `prsfl`, `pssfl` | kg/m2/s | stratiform surface rain / snow |
| `prelhum` | 1 | relative humidity diagnostic |
| `ktype` | - | after the 2 -> 4 re-typing (mo_cloud.f90:1440-1455) |
| `paclcac`, `paclcov`, `paprl`, `paprs`, `pqvi`, `pxlvi`, `pxivi` | | **accumulated** (`+= pdelta_time * value`); use the `*_na` fields for instantaneous values |
| `aclcov_na`, `aprl_na`, `aprs_na` | 1, kg/m2/s | total cover, total and snow surface precipitation (not accumulated) |
| `xlvi_na`, `xivi_na`, `qvi_na` | kg/m2 | vertical integrals of the step-start fields |
| `pch_concloud`, `pcw_concloud` | W/m2, kg/m2/s | ECHAM's heat and water budget checks |

`param/<variant>/*`: the constants the routine ran with, from
`mo_echam_cloud_params`: `crs crt cvtfall csecfrl clwprat csatsc cinv nex
nadd jbmin jbmax ncctop nccbot`.

### Intermediates (`diag/<variant>/*`, (47, ncol))

These are local variables of the routines, copied out by an output-only patch.
The patched build is bit-identical to the pristine one on every column. Names
are ECHAM's locals, with a suffix where the same local is recorded at more
than one point. All are grid-box values unless marked in-cloud. "Per step"
means the change over `ptime_step_len`, in kg/kg unless stated.

`cover` (all three variants): `ua` (`e_s*rd/rv` used, Pa), `zqsm1` (qsat),
`zrhc` (critical RH), `zsat` (inversion factor), `zqr` (`q/(qsat*zsat)`),
`lao` (1 where the enhancement applied).

`cloud` (`sonntag` variant):

| name | section | meaning |
|---|---|---|
| `zclcpre_in`, `zrfl_in`, `zsfl_in`, `zxiflux_in` | level entry | precipitating fraction, rain / snow / sedimenting-ice flux (kg/m2/s) arriving from above |
| `ua_m1`, `dua_m1`, `uaw_m1`, `duaw_m1` | 1.3 | saturation lookups at `ptm1` (mixed / water) |
| `zsmlt`, `zimlt` | 3.1 | snow melt (kg/kg per step, as `zsnmlt/(zcons2 zdp)`), cloud-ice melt |
| `zsub`, `zevp` | 3.2, 3.3 | snow sublimation, rain evaporation (per step) |
| `zrfl_melt`, `zsfl_melt` | 3 | fluxes after melting |
| `zqsed`, `zxised`, `zxiflux_sed` | 4 | sedimentation change, ice after sedimentation, ice flux leaving the level |
| `lo2` | 4 | phase switch at `ptm1+ptte*dt` (1 = ice) |
| `zclcaux_in`, `zxlevap`, `zxievap`, `zxlb_in`, `zxib_in` | 4 | cover, clear-cell evaporation, in-cloud liquid / ice |
| `zqsm1`, `zdtdt`, `zqp1`, `ztp1`, `zdqsat1` | 5 | qsat, T change, provisional q / T, `dqs/dT / (1 + clc L dqs/dT)` |
| `zqcdif` | 5 | condensation in the cloudy part (per step) |
| `zcnd_pre54`, `zdep_pre54` | 5 | condensation / deposition before the 5.4 check |
| `ztp1tmp_pre54`, `zqp1tmp_pre54`, `ua_54`, `dua_54`, `ub_54`, `lo2_54`, `zqsp1tmp` | 5.4 | state, lookups and phase at the post-condensation temperature |
| `zcnd`, `zdep` | 5.4 | final condensation / deposition (per step) |
| `zclcaux`, `zxlb_55`, `zxib_55`, `ztp1tmp` | 5.5 | cover after clear-cell promotion, in-cloud values, temperature |
| `zfrl_hom`, `zfrl` | 6.1, 6.2 | freezing: homogeneous only, total (grid box, per step) |
| `zxlb_6`, `zxib_6` | 6 | in-cloud liquid / ice after freezing |
| `zauloc`, `zxrp1`, `zxsp1` | 7 | local-rain factor (0 at `cauloc = 0`), rain / snow content in the precipitating area |
| `zraut`, `zrac1`, `zrac2` | 7.1 | autoconversion, accretion by falling rain, by local rain (in-cloud, per step) |
| `zrieff`, `zsaut`, `zcolleffi` | 7.2 | ice effective radius (um), ice autoconversion (in-cloud), collection efficiency |
| `zsaci1`, `zsaci2` | 7.2 | aggregation by falling / local snow (in-cloud, per step) |
| `zsacl1`, `zsacl2` | 7.2 | riming by falling / local snow, **already weighted** by `min(clc, clcpre)` / `clc` (mo_cloud.f90:1069, 1079) |
| `zrpr`, `zspr`, `zsacl` | 7 | grid-box rain / snow production and riming (per step) |
| `zxlb_7`, `zxib_7` | 7 | in-cloud liquid / ice after section 7 |
| `zclcpre_out`, `zrfl_out`, `zsfl_out`, `zsmlt_final`, `zxiflux_final` | 7.3 | precipitating fraction and fluxes leaving the level (at the lowest level `zsfl_out` already contains the ice flux) |
| `zdxlcor`, `zdxicor` | 8.4 | ccwmin correction (kg/kg/s) |

Zeros mean "not reached" as well as "zero". For example the section 7 rates
are only computed for cells with cover and condensate, and section 3 only
below a precipitating level.

## Resolution file keys

`T<nn>/cover/out/{paclc,knvb,printop}`, `T<nn>/cloud/out/<as above>`,
`T<nn>/cloud/in/{paclc,knvb}` (realistic columns chained through that
truncation's cover), `T<nn>/param/*`. All other inputs are those of the T63
files. T31 uses `nadd = 1`, which also enhances the level below the inversion.

## Numerical floors of the reference

ECHAM floors the sedimenting ice at `EPSILON(1.0)` = 2.2e-16 kg/kg
(mo_cloud.f90:583) and the in-cloud condensate at 1e-20 (:911-912). Even
cloud-free columns therefore show surface fluxes of O(1e-18) kg/m2/s and
heating of O(1e-17) K/s. The EPSILON floor also gives an ice-free level a
small fall speed, so part of an incoming ice flux passes through instead of
being stored.

## Regenerating

On branch `harness/echam-cloud-cover` (not pushed), in
`fortran_harness/echam_cloud`, with the ECHAM tree at the path in
`provenance.json`:

```bash
python py/make_diag_patches.py <echam_src> patches
for s in table sonntag tetens; do make SAT=$s DIAG=1; done
make SAT=table DIAG=0; make SAT=sonntag DIAG=0; make SAT=table DIAG=1 OPT=-O0
cd py && python build_reference.py <jcm>/jcm/data/test/echam_cloud_reference
```

`build_reference.py` aborts unless the patched and pristine builds agree
bit-for-bit, and unless every column run alone agrees with the batch run.
It also records the -O0 vs -O2 difference, which is zero.

## Column catalogue

### `cover` columns (25)

| # | name | process (section of mo_cover.f90) | description |
|---|---|---|---|
| 0 | `sc_inversion_in_window` | 1.3 inversion + zsat | marine column (Tsfc 292 K), 6 K inversion above ~900 hPa, humid boundary layer (RH 0.9): inversion level found between jbmin and jbmax; zsat = csatsc at knvb (mo_cover.f90:179-247). |
| 1 | `no_inversion` | 1.3 no stable layer | standard 6.5 K/km lapse everywhere: no level is more stable than -cinv*g/cpd, knvb stays 1, no enhancement. |
| 2 | `weak_stability_in_window` | 1.3 zgam > 0 | a layer with lapse 2 K/km (stable but not an inversion) near 930 hPa: zsat = min(1, csatsc + zgam) with zgam > 0. |
| 3 | `inversion_below_jbmax` | 1.3 knvb > jbmax | inversion at the lowest interface (below ~500 m): knvb is found but lies below jbmax, so no enhancement (mo_cover.f90:237). |
| 4 | `inversion_above_jbmin` | 1.3 search window | inversion at ~720 hPa (above 2000 m) and a standard lapse below: the search from klev up to jbmin does not reach it (mo_cover.f90:188). |
| 5 | `sc_land` | 1.3 gate (pfrw, pfri, ktype) | same profile over land: no inversion search (pfrw <= 0.5). |
| 6 | `sc_frac_land_045` | 1.3 gate (pfrw, pfri, ktype) | same profile, land fraction 0.45: searched (pfrw > 0.5). |
| 7 | `sc_frac_land_055` | 1.3 gate (pfrw, pfri, ktype) | same profile, land fraction 0.55: not searched. |
| 8 | `sc_sea_ice` | 1.3 gate (pfrw, pfri, ktype) | same profile over partial sea ice (pfri > 1e-12): not searched. |
| 9 | `sc_convective` | 1.3 gate (pfrw, pfri, ktype) | same profile with ktype = 1: not searched. |
| 10 | `two_inversions` | 1.3 tie-break | two identical inversions inside the window: the scan from the surface keeps the lowest (strict improvement only, mo_cover.f90:202). |
| 11 | `rh_thresholds` | 2 closure | levels at RH = rhc - 1e-3, rhc + 1e-3, rhc + 0.1, exactly 1 (cover 1), and 1.2 (super-saturated, cover 1), water phase, no inversion. |
| 12 | `ice_phase_cover` | 1 lo2 qsat | cold cells at RH_w 0.95: T = 250 K with ice 1e-5 (> csecfrl, ice qsat), T = 250 K ice-free (water qsat), T = 230 K ice-free (< cthomi, ice qsat), T = 250 K with ice exactly csecfrl (water qsat). |
| 13 | `strat_humid` | no stratospheric cutoff in mo_cover | saturated levels above 10 hPa: ECHAM's cover has no pressure cutoff, so they get cover (jcm cuts at 1000 Pa). |
| 14 | `real_tropical_deep` | realistic | tropical ocean, deep convection (ktype 1, max CAPE, LWP < 1 kg/m2); lat 4.66, lon 0.00, 2005-04-02, frl 0.02, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 15 | `real_sc_east_pacific` | realistic | subtropical marine inversion, eastern Pacific (180-290E); lat -34.51, lon 268.12, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 16 | `real_sc_atlantic` | realistic | subtropical marine inversion, eastern Atlantic (330-20E); lat -12.12, lon 350.62, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 17 | `real_trade_cumulus` | realistic | trade-wind shallow convection (ktype 2), ocean; lat 17.72, lon 91.88, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 18 | `real_midlat_nh_mixed` | realistic | NH mid-latitude ocean, mixed-phase cloud; lat 58.76, lon 178.12, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 19 | `real_southern_ocean` | realistic | Southern Ocean, mixed-phase cloud; lat -58.76, lon 232.50, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 20 | `real_arctic_sea_ice` | realistic | Arctic, sea-ice point with cloud; lat 77.41, lon 82.50, 2005-04-02, frl 0.00, sice 0.98, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 21 | `real_arctic_cold` | realistic | Arctic, coldest near-surface air; lat 81.13, lon 146.25, 2005-04-02, frl 0.00, sice 1.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 22 | `real_tropical_land` | realistic | tropical land convection; lat 6.53, lon 1.87, 2005-04-02, frl 0.58, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 23 | `real_cirrus` | realistic | tropical upper-tropospheric ice with little liquid; lat -8.39, lon 120.00, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 24 | `real_nh_land_cold` | realistic | NH high-latitude land, cold cloudy column; lat 66.22, lon 318.75, 2005-04-02, frl 1.00, sice 0.77, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |

### `cloud` columns (42)

| # | name | process (section of mo_cloud.f90) | description |
|---|---|---|---|
| 0 | `null_clear` | none | Sub-saturated, cloud-free, no tendencies: every output increment must be zero. |
| 1 | `clear_condensate_evap` | 4 zxlevap/zxievap | paclc=0 cell holding liquid and ice at 600 hPa: all condensate is evaporated unconditionally (mo_cloud.f90:661-671). |
| 2 | `clear_new_condensate_warm` | 5.4 + 5.5 promotion | paclc=0, 5% supersaturated w.r.t. water at 850 hPa: 5.4 condenses the excess over 1% and 5.5 promotes the cell to zclcaux=1 so section 7 acts on it (mo_cloud.f90:754-810). |
| 3 | `clear_new_condensate_ice` | 5.4 + 5.5 promotion (ice) | paclc=0, 5% supersaturated w.r.t. ice at 300 hPa, T < cthomi: deposition via 5.4, promotion to zclcaux=1. |
| 4 | `partial_cloud_zero_tend` | 5 zqcdif (no tendencies) | cover 0.3, RH 0.9, liquid 1e-4, zero tendencies at 850 hPa: ECHAM zqcdif = 0 (tendency-driven condensation; issue #940 headline 1). |
| 5 | `cond_growth_warm` | 5 zqcdif > 0, zcnd | cover 0.6, liquid 1e-4, moistening 2e-7 kg/kg/s and cooling 5e-5 K/s at 800-900 hPa: condensational growth in the cloudy part (mo_cloud.f90:726-750). |
| 6 | `cloud_dissipation_mixed` | 5 zqcdif < 0, zifrac split | cover 0.5, liquid 8e-5 + ice 4e-5 at ~700 hPa, drying and warming tendencies: dissipation split between zcnd and zdep by the ice fraction (mo_cloud.f90:739-744). |
| 7 | `lo2_ice_memory` | 4 lo2 / csecfrl | T = 255 K with ice 2e-5 > csecfrl: lo2 true, growth goes to ice (zdep) with ice saturation and Ls (mo_cloud.f90:647-650, 697-699, 747-748). |
| 8 | `lo2_liquid_no_ice` | 4 lo2 / csecfrl | T = 255 K with ice 1e-6 < csecfrl: lo2 false, growth goes to supercooled liquid (zcnd), which section 6.2 then partly freezes. |
| 9 | `lo2_prov_temperature` | 3.1 zimlt (ptm1) vs lo2 (ptm1+ptte*dt) | ptm1 = 273.6 K but ptm1 + ptte*ztmst = 272.4 K, ice 2e-5, cover 0.4 at 800 hPa: section 3.1 melts all cloud ice from ptm1 (mo_cloud.f90:433-439) while lo2 reads the provisional temperature (:647-650). |
| 10 | `homogeneous_freezing` | 6.1 | cover 0.5, advected liquid 5e-5 at T ~ 232 K (< cthomi): all in-cloud liquid freezes (mo_cloud.f90:821-828). |
| 11 | `bigg_contact_sea` | 6.2 Bigg + contact freezing | cover 0.7, supercooled liquid 2e-4, no ice, T = 258 K at 650 hPa over sea (acdnc 80 cm-3 profile): Bigg and contact freezing (mo_cloud.f90:832-885), both read pacdnc. |
| 12 | `bigg_contact_land` | 6.2 Bigg + contact freezing | cover 0.7, supercooled liquid 2e-4, no ice, T = 258 K at 650 hPa over land (acdnc 180 cm-3 profile): Bigg and contact freezing (mo_cloud.f90:832-885), both read pacdnc. |
| 13 | `supersat_54_warm` | 5.4 (water) | cover 0.4, liquid 5e-5, 3% supersaturated w.r.t. water at 900 hPa, no tendencies: zqcdif = 0 and the whole-box 1% check condenses the rest (mo_cloud.f90:769-784). |
| 14 | `supersat_54_ice` | 5.4 (ice) | cover 0.3, ice 1e-5, 10% supersaturated w.r.t. ice at 300 hPa (T < cthomi). |
| 15 | `supersat_ub_branch` | 5.4 zes >= 0.4 branch (unphysical edge case) | UNPHYSICAL edge case: T = 292 K at ~30 hPa so that es/p >= 0.4 and 5.4 uses zlcdqsdt = zqsp1tmp*zcor*ub (mo_cloud.f90:777) and the 0.5 cap on zes; q = 1.02 qsat. |
| 16 | `snow_melt` | 3.1 snow melt + zimlt | ice cloud (cover 0.8, ice 1e-4) at 420-480 hPa makes snow that falls into T > tmelt below ~700 hPa and melts (mo_cloud.f90:430-437); a level at 800 hPa also holds cloud ice at T > tmelt (zimlt, :438-439). |
| 17 | `snow_sublimation` | 3.2 snow sublimation | ice cloud (cover 0.9, ice 1.5e-4) at 380-420 hPa over dry cold air (RH_ice 0.5, T < tmelt) between 500 and 750 hPa: Lin (1983) sublimation (mo_cloud.f90:449-506). |
| 18 | `rain_evaporation` | 3.3 rain evaporation | warm cloud (cover 0.8, liquid 8e-4) at 750-800 hPa over RH 0.5 air below 850 hPa: Rotstayn (1997) evaporation (mo_cloud.f90:518-549). |
| 19 | `autoconv_accretion` | 7.1 zraut + zrac1 | thick warm cloud 700-900 hPa, cover 0.9, liquid 3e-4 to 8e-4: Beheng autoconversion at every level, accretion (zrac1) of the rain from above (mo_cloud.f90:968-1017). |
| 20 | `zclcpre_reset` | 7.3 zclcpre reset | weakly precipitating wide cloud (cover 0.9, liquid 1.5e-4) at 600 hPa above a narrow strongly precipitating cloud (cover 0.25, liquid 1.5e-3) at 850 hPa: the precipitating fraction is reset to the local cover where local production exceeds the incoming flux (mo_cloud.f90:1177), then max-weighted. |
| 21 | `ice_sedimentation_cirrus` | 4 ice sedimentation | cirrus (cover 0.4, ice 3e-5) at 200-250 hPa above dry air: sedimentation (mo_cloud.f90:580-615) carries ice into the levels below. |
| 22 | `ice_sed_to_surface` | 4 + 7.3 bottom-level ice flux into snow | polar column (Tsfc 245 K): ice 2e-5 with cover 0.5 in the lowest levels; the ice flux leaving the bottom level becomes surface snow (mo_cloud.f90:1119-1121). |
| 23 | `riming_aggregation` | 7.2 zsacl1 + zsaci1 | snow from an ice cloud (cover 0.9, ice 1.5e-4) at 450 hPa falls through a mixed-phase cloud at 650 hPa (T 262 K, liquid 1.5e-4, ice 1e-5): riming and aggregation by the incoming snow (mo_cloud.f90:1064-1073). |
| 24 | `below_ccwmin` | 8.4 ccwmin correction + cover write-back | cover 0.4 with liquid 5e-8 and ice 3e-8 (both < ccwmin) at 700 hPa: condensate returned to vapour and paclc set to 0 (mo_cloud.f90:1264-1288); at 600 hPa only the liquid is below ccwmin, so the cover survives. |
| 25 | `negative_condensate` | 4/8.4 negative input condensate | spectral-ringing inputs: liquid -2e-9 in a cloudy cell (cover 0.2) at 750 hPa and ice -1e-9 in a clear cell at 500 hPa. |
| 26 | `detrainment` | 1/8.3 convective detrainment pxtecl/pxteci | deep convection (ktype 1, top at 250 hPa): liquid detrainment 5e-8 kg/kg/s at 450-600 hPa and ice 5e-8 at 250-350 hPa, into cloudy (cover 0.5) and clear cells. |
| 27 | `ktype2_to_4` | 10 ktype 2 -> 4 | ktype 2 with top at 800 hPa and liquid 1e-4 at/below the top only: re-typed to 4 (mo_cloud.f90:1440-1455). |
| 28 | `ktype2_stays` | 10 ktype 2 -> 4 | ktype 2 with top at 800 hPa and much more liquid above the top (500 hPa): stays 2. |
| 29 | `tmelt_boundary` | 3.1 / 4 exact tmelt thresholds | levels at exactly ptm1 = tmelt and tmelt + 1e-3 K receiving snow and holding cloud ice: melting uses MAX(0, ptm1 - tmelt) and FSEL(-ztdif) (mo_cloud.f90:433-439). float64 only (tmelt is not representable in float32). |
| 30 | `time_level_rain_evap` | 3.3 m1 state vs provisional | rain falls into a layer that is sub-saturated at step start (RH 0.6) but is moistened strongly within the step (pqte 1e-6 kg/kg/s): section 3.3 evaporates against pqm1/ptm1 (mo_cloud.f90:524, 546). |
| 31 | `real_tropical_deep` | realistic | tropical ocean, deep convection (ktype 1, max CAPE, LWP < 1 kg/m2); lat 4.66, lon 0.00, 2005-04-02, frl 0.02, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 32 | `real_sc_east_pacific` | realistic | subtropical marine inversion, eastern Pacific (180-290E); lat -34.51, lon 268.12, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 33 | `real_sc_atlantic` | realistic | subtropical marine inversion, eastern Atlantic (330-20E); lat -12.12, lon 350.62, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 34 | `real_trade_cumulus` | realistic | trade-wind shallow convection (ktype 2), ocean; lat 17.72, lon 91.88, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 35 | `real_midlat_nh_mixed` | realistic | NH mid-latitude ocean, mixed-phase cloud; lat 58.76, lon 178.12, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 36 | `real_southern_ocean` | realistic | Southern Ocean, mixed-phase cloud; lat -58.76, lon 232.50, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 37 | `real_arctic_sea_ice` | realistic | Arctic, sea-ice point with cloud; lat 77.41, lon 82.50, 2005-04-02, frl 0.00, sice 0.98, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 38 | `real_arctic_cold` | realistic | Arctic, coldest near-surface air; lat 81.13, lon 146.25, 2005-04-02, frl 0.00, sice 1.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 39 | `real_tropical_land` | realistic | tropical land convection; lat 6.53, lon 1.87, 2005-04-02, frl 0.58, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 40 | `real_cirrus` | realistic | tropical upper-tropospheric ice with little liquid; lat -8.39, lon 120.00, 2005-04-02, frl 0.00, sice 0.00, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
| 41 | `real_nh_land_cold` | realistic | NH high-latitude land, cold cloudy column; lat 66.22, lon 318.75, 2005-04-02, frl 1.00, sice 0.77, from B_fix2.nc (ERA5 init 2005-04-01, instantaneous) |
