# ECHAM6.3-HAM2.3 two-moment cloud microphysics reference data (#941)

Inputs, outputs and intermediates of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_cloud_micro_2m.f90::cloud_micro_interface` (the whole routine; `physc.f90:1091`
calls it when `lcdnc_progn`), run in double precision on designed single columns by a
standalone harness. They are the numerical reference for the #941 ice-source pieces
of `jcm/physics/clouds/lohmann_2m/`, compared by
`jcm/physics/clouds/lohmann_2m_fortran_reference_test.py`. `F <n>` is a line of that
`mo_cloud_micro_2m.f90` (md5 4e0dbb96328930e934a959fa608af6db).

These files hold **data only**. ECHAM is under the MPI-M Software Licence Agreement;
no ECHAM source is part of this repository. The harness copies ECHAM from a local
tree into its build directory at build time. `cloud2m_provenance.json` records the
file checksums, compiler and flags, every stub, the patch, the integrity checks
(patched = pristine bit for bit, -O0 = -O2 exactly, each column alone = all
together) and how to regenerate. (The files carry a `cloud2m_` prefix so they sit
beside the 1M reference data of the same directory without colliding.)

## Files

| file | content |
|---|---|
| `cloud2m_T63L47.npz` | 8 designed columns: inputs, every INOUT/OUT argument and 111 intermediates, at two time steps |
| `cloud2m_provenance.json` | how the data were produced |

## Conventions

* **Layout.** Level fields are `(nlev, ncol)` = `(47, ncol)`, half-level fields
  `(48, ncol)`, per-column fields `(ncol,)`. Column `j` is `meta/names[j]`.
* **Vertical ordering: TOP-FIRST**, as ECHAM's argument arrays: index 0 is the model
  top (about 1 Pa), index 46 the lowest full level; `in/paphm1[47]` is the surface
  pressure. jcm's model *output* files are surface-first; jcm's physics works
  top-first. Select levels by pressure (`in/papm1`), not by index.
* **Grid.** ECHAM6.3 L47 hybrid levels (`in/vct`, the values of
  `jcm.physics.echam.echam_levels.get_echam_levels(47)`), surface pressure 1010 hPa.
* **Time steps.** `timestep/dt1200/*`: `delta_time` 600 s, `time_step_len` (ECHAM's
  `ztmst`) 1200 s, ECHAM's T63 leapfrog. `timestep/dt720/*`: 720 s / 720 s, jcm's T63
  step. Inputs are the same for both; `out/<step>/` and `diag/<step>/` differ.
* **Tendencies** are ECHAM's: relative to the step-start (m1) state, including
  whatever tendency came in. End-of-step state = `X_m1 + ztmst*out/<step>/pXte`. The
  number tracers: end tracer = `xtm1 + ztmst*pxtte` = n/rho (0 where the ccwmin repair
  acted).
* **Saturation.** ECHAM's own 0.001 K lookup tables; the 2M reads them at
  `NINT(1000*T)` (F 3932-3933). Every column temperature is on the 1 mK grid, so
  `diag/*/esw` and `esi` are the Sonntag (1990) formula at T.
* Integers are ECHAM's 1-based Fortran values.

## Configuration

`nic_cirrus = 1`, `nauto = 2`, `lsecprod = .FALSE.`, `lconv = .TRUE.`, fixed minimum
CDNC 40 cm-3, T63 constants (`param/*`). **Aerosol-free** (the `cloud_subm_1` stub):
no activation (`zcdncact = 0`), no cirrus nucleation (`zascs = 0` makes `zninucl = 0`),
no dust/BC (no heterogeneous mixed-phase freezing). `pvervel = 0` (jcm has no omega in
the updraft, #705); `zcvcbot = 0` (no convective cloud-base droplets). Built without
`-DHAMMOZ`, which removes only the orographic-cirrus hooks (they need `nic_cirrus = 2`).

## Arrays

### `in/*` (ECHAM's arguments and the boundary conditions from convection)

| name | meaning |
|---|---|
| `in/papm1` | full-level pressure [Pa] (= papp1) |
| `in/paphm1` | half-level pressure [Pa] (48, ncol) (= paphp1) |
| `in/ptm1` | step-start temperature [K] (1 mK grid) |
| `in/pqm1` | step-start specific humidity [kg/kg] |
| `in/pxlm1` | step-start cloud liquid [kg/kg] |
| `in/pxim1` | step-start cloud ice [kg/kg] |
| `in/paclc` | cloud cover (from cover; the 2M's input) |
| `in/ptkem1` | TKE [m2/s2] |
| `in/pqte` | accumulated q tendency before the call [kg/kg/s] |
| `in/ptte` | accumulated T tendency [K/s] |
| `in/pxlte` | accumulated liquid tendency, WITHOUT detrainment [kg/kg/s] |
| `in/pxite` | accumulated ice tendency, WITHOUT detrainment [kg/kg/s] |
| `in/pqtec` | convective moisture tendency (only clamped >= 0 by the 2M) |
| `in/xtm1_cdnc` | CDNC tracer pxtm1(:,:,idt_cdnc) [1/kg], RAW (may be negative / huge) |
| `in/xtm1_icnc` | ICNC tracer pxtm1(:,:,idt_icnc) [1/kg], RAW |
| `in/xtte_cdnc` | CDNC tracer tendency before the call [1/kg/s] |
| `in/xtte_icnc` | ICNC tracer tendency before the call [1/kg/s] |
| `in/zxtec` | detrained condensate, BOTH phases [kg/kg/s] (bc 'Detrained condensate') |
| `in/ztconv` | convection temperature [K] (bc 'Temperature in convective scheme'; = ptm1 unless stated) |
| `in/ptvm1` | virtual temperature T(1+vtmpc1 q-(xl+xi)) (physc.f90:267-268) |
| `in/pgeo` | full-level geopotential above the surface [m2/s2] (geopot) |
| `in/papp1` | = papm1 |
| `in/paphp1` | = paphm1 |
| `in/pvervel` | large-scale omega [Pa/s], 0 everywhere |
| `in/cdncact` | cloud_subm_1 activated CDNC [1/m3], 0 |
| `in/ascs` | cloud_subm_1 soluble aerosol [1/m3], 0 |
| `in/cdncact_cv` | convective-base CDNC [1/m3], 0 |
| `in/aclc_tm1` | cloud_cover_duplic (= paclc) |
| `in/zcvcbot` | (ncol,) convective cloud-base index, 0 |
| `in/zwcape` | (ncol,) CAPE updraft, 0 |
| `in/knvb` | (ncol,) int, 1 (inert: pvervel = 0) |
| `in/vct` | (96,) ECHAM L47 vct (a [Pa] then b) |
| `in/nn` | 63 |
| `in/nic_cirrus` | 1 |
| `in/nauto` | 2 |

### `out/<step>/*` (every INOUT/OUT argument after the call)

| name | meaning |
|---|---|
| `out/<step>/pqtec` | max(pqtec, 0) |
| `out/<step>/paclc` | cloud cover after the call (written back) |
| `out/<step>/paclcac` | accumulated cover (x delta_time) |
| `out/<step>/pqte` | q tendency relative to pqm1 [kg/kg/s] |
| `out/<step>/ptte` | T tendency relative to ptm1 [K/s] (includes the F 1305 fix) |
| `out/<step>/pxlte` | liquid tendency relative to pxlm1 [kg/kg/s] (includes the lo2 liquid part of zxtec) |
| `out/<step>/pxite` | ice tendency relative to pxim1 [kg/kg/s] (includes the lo2 ice part of zxtec) |
| `out/<step>/pxtte_cdnc` | (cdnc/rho - xtm1_cdnc)/ztmst, after the ccwmin repair [1/kg/s] |
| `out/<step>/pxtte_icnc` | (icnc/rho - xtm1_icnc)/ztmst, after the ccwmin repair [1/kg/s] |
| `out/<step>/pacdnc` | CDNC [1/m3] after section 7 |
| `out/<step>/picnc` | ICNC [1/m3] after section 7 |
| `out/<step>/prelhum` | RH |
| `out/<step>/sice` | max(q/qsi - 1, 0) |
| `out/<step>/reffl` | liquid r_eff [um] |
| `out/<step>/reffi` | ice r_eff [um] |
| `out/<step>/paclcov` | (ncol,) accumulated total cover |
| `out/<step>/paprl` | (ncol,) accumulated precipitation |
| `out/<step>/pqvi` | (ncol,) accumulated |
| `out/<step>/pxlvi` | (ncol,) accumulated |
| `out/<step>/pxivi` | (ncol,) accumulated |
| `out/<step>/paprs` | (ncol,) accumulated snow |
| `out/<step>/pssfl` | (ncol,) surface snow flux [kg/m2/s] |
| `out/<step>/prsfl` | (ncol,) surface rain flux [kg/m2/s] |

### `diag/<step>/*` (intermediates, named after ECHAM locals)

Written by assignment statements the harness patch adds to a build-time copy of
the source; `(1/0)` fields are logicals. Per level `(47, ncol)`.

| name | meaning |
|---|---|
| `picnc_in` | ICNC [1/m3] after loop 122 (rho*(pxtm1+ztmst*pxtte), max cqtmin, zcd2ic/zic2cd), F 597-630 |
| `cdnc_in` | CDNC [1/m3] after loop 122, F 600-627 |
| `cdnc_min_in` | minimum CDNC [1/m3] (cdnc_min_fixed), F 609-610 |
| `icncq_detr` | zicncq [1/m3] after += znidetr, F 982 |
| `s1_picnc` | section-1 ICNC [1/m3] (after the second consistency, F 794): the ICNC lo2_2d uses |
| `s1_xip1` | section-1 ice max(pxim1+ztmst*pxite,0) [kg/kg], F 860-861 |
| `s1_rice` | section-1 Schumann volume-mean radius [m], F 866-877 |
| `s1_vervmax` | section-1 threshold updraft [m/s], F 880-882 |
| `lo2_2d` | section-1 WBF criterion (1/0), F 885 |
| `ll_cv` | detrained-ice gate (1/0), F 958-963 |
| `vervx` | updraft velocity [cm/s], F 814-816 |
| `rid` | temperature-parameterised volume-mean radius [m], F 945-956 |
| `nidetr` | crystal number of detrained ice [1/m3] (floored at cqtmin), F 970-980 |
| `ninucl` | nic_cirrus=1 nucleation [1/m3] (0 with zascs=0), F 988-999 |
| `icncq` | zicncq [1/m3] after the F 1127-1131 floors (melting's picncq) |
| `s1_cdnc` | CDNC [1/m3] at the end of section 1 |
| `esw` | saturation vapour pressure over water at ptm1 [Pa] (table), F 668 |
| `esi` | saturation vapour pressure over ice at ptm1 [Pa] (table), F 690 |
| `eta` | WBF growth factor zeta, F 850-856 |
| `rho` | air density papm1/(rd*ptvm1) [kg/m3], F 578 |
| `lsdcp` | als/cp_moist [K], F 843-846 |
| `lvdcp` | alv/cp_moist [K], F 843-845 |
| `qsw` | qsat water at ptm1 [kg/kg] |
| `qsi` | qsat ice at ptm1 [kg/kg] |
| `qswp1` | qsat water at ptm1+1 mK [kg/kg] |
| `qsip1` | qsat ice at ptm1+1 mK [kg/kg] |
| `sice` | ice supersaturation max(q/qsi-1,0), F 692-693 |
| `dp` | layer pressure thickness [Pa] |
| `dz` | layer thickness [m] |
| `aaa` | ice fall-speed density factor |
| `viscos` | dynamic viscosity of air |
| `xtec` | detrained condensate rate [kg/kg/s] (bc 'Detrained condensate') |
| `tconv` | convection temperature [K] (bc 'Temperature in convective scheme') |
| `imlt` | in-cloud ice melted [kg/kg], F 2001-2003 |
| `smlt` | snow melt [kg/kg] |
| `ximlt` | falling-ice melt [kg/kg] |
| `xisub` | falling-ice sublimation [kg/kg] |
| `sub` | snow sublimation [kg/kg] |
| `evp` | rain evaporation [kg/kg] |
| `mlt_xite` | pxite [kg/kg/s] after melting (before sedimentation) |
| `mlt_icnc` | ICNC [1/m3] after melting (before sedimentation) |
| `sed_xip1_in` | sedimentation input ice max(pxim1+ztmst*pxite, EPSILON) [kg/kg], F 1227-1228 |
| `sed_icnc_in` | sedimentation input ICNC [1/m3] (no znidetr yet) |
| `sed_xiflux_in` | ice mass flux entering the level [kg/m2/s] |
| `sed_xifluxn_in` | ice number flux entering the level [1/m2/s] |
| `sed_clcfi_in` | falling-ice cover entering the level |
| `sed_xip1` | ice after sedimentation [kg/kg] |
| `sed_icnc` | ICNC after sedimentation [1/m3] |
| `sed_xite` | pxite = (zxip1-pxim1)/ztmst after sedimentation [kg/kg/s], F 1248 |
| `sed_xiflux` | ice mass flux leaving the level [kg/m2/s] |
| `sed_xifluxn` | ice number flux leaving the level [1/m2/s] |
| `sed_clcfi` | falling-ice cover leaving the level |
| `sed_mrateps` | sedimented ice (in-cloud where cf>clc_min) [kg/kg] |
| `icnc_add` | ICNC after += znidetr + zninucl, MIN icemax, MAX icemin [1/m3], F 1251-1253 |
| `icnc_s4` | ICNC after zic2cd (T>tmelt to CDNC) [1/m3], F 1255-1263 |
| `s4_rice` | section-4 Schumann radius [m], F 1281-1289 |
| `s4_vervmax` | section-4 threshold updraft [m/s], F 1291-1293 |
| `lo2` | section-4 phase criterion (1/0), F 1295-1298 |
| `ptte_pre` | ptte before the #368 energy fix [K/s] |
| `ptte_fix` | ptte after the #368 energy fix [K/s], F 1300-1307 |
| `xite` | ice part of zxtec by lo2 [kg/kg/s], F 1310 |
| `xlte` | liquid part of zxtec by lo2 [kg/kg/s], F 1311 |
| `xidt` | ztmst*(pxite+zxite2) [kg/kg], F 1316 |
| `xldt` | ztmst*(pxlte+zxlte2)+zximlt+zimlt [kg/kg], F 1317 |
| `xib_4` | in-cloud ice after section 4 [kg/kg] |
| `xlb_4` | in-cloud liquid after section 4 [kg/kg] |
| `xievap` | clear-sky ice evaporation [kg/kg], F 1399 |
| `xlevap` | clear-sky liquid evaporation [kg/kg], F 1402 |
| `qcdif` | condensation source [kg/kg] |
| `cnd0` | zcnd before 5.4 [kg/kg] |
| `dep0` | zdep before 5.4 [kg/kg] |
| `cnd` | zcnd after 5.4 [kg/kg] |
| `dep` | zdep after 5.4 [kg/kg] |
| `tp1tmp` | temperature after condensation [K] |
| `qp1tmp` | humidity after condensation [kg/kg] |
| `qsp1tmp` | qsat at ztp1tmp [kg/kg] |
| `uicw_icnc_in` | ICNC entering update_in_cloud_water [1/m3] |
| `uicw_xib_in` | in-cloud ice entering update_in_cloud_water [kg/kg] |
| `uicw_xlb_in` | in-cloud liquid entering update_in_cloud_water [kg/kg] |
| `uicw_cdnc_in` | CDNC entering update_in_cloud_water [1/m3] |
| `uicw_aclc_in` | cloud cover entering update_in_cloud_water |
| `ll_cc` | cloud flag paclc > clc_min (1/0), F 1276 |
| `uicw_icnc` | ICNC after update_in_cloud_water [1/m3], F 2610-2624 |
| `uicw_xib` | in-cloud ice after update_in_cloud_water [kg/kg] |
| `uicw_xlb` | in-cloud liquid after update_in_cloud_water [kg/kg] |
| `uicw_aclc` | cloud cover after update_in_cloud_water |
| `uicw_cdnc` | CDNC after update_in_cloud_water [1/m3] |
| `frl_hom` | homogeneous freezing [kg/kg] |
| `icnc_61` | ICNC after 6.1 [1/m3] |
| `frl` | freezing 6.1+6.2 [kg/kg] |
| `icnc_6` | ICNC after 6.2 and WBF [1/m3] |
| `xib_6` | in-cloud ice after section 6 [kg/kg] |
| `xlb_6` | in-cloud liquid after section 6 [kg/kg] |
| `icnc_7` | ICNC after section 7 [1/m3] (the final picnc) |
| `rpr` | rain formation [kg/kg] |
| `spr` | snow formation [kg/kg] |
| `sacl` | riming [kg/kg] |
| `xib_7` | in-cloud ice after section 7 [kg/kg] |
| `xlb_7` | in-cloud liquid after section 7 [kg/kg] |
| `clcpre` | precipitating fraction leaving the level |
| `rsfl_lev` | rain flux leaving the level [kg/m2/s] |
| `ssfl_lev` | snow flux leaving the level [kg/m2/s] |
| `icnc_cand` | ICNC diagnosis candidate 0.75/(pi*rhoice)*rho*pxib/prid**3 [1/m3], F 2616 |
| `icnc_dmask` | diagnosis applied (cloud, pxib>cqtmin, picnc<=icemin) (1/0), F 2611-2612 |
| `prid` | prid passed to update_in_cloud_water [m] (= zrid) |
| `end_xlp1` | end-of-step grid-mean liquid before the ccwmin repair [kg/kg], F 3609 |
| `end_xip1` | end-of-step grid-mean ice before the ccwmin repair [kg/kg], F 3617 |
| `xtte_cdnc_raw` | (cdnc/rho - pxtm1_cdnc)/ztmst before the repair [1/kg/s], F 3625 |
| `xtte_icnc_raw` | (icnc/rho - pxtm1_icnc)/ztmst before the repair [1/kg/s], F 3628 |
| `dxlcor` | liquid ccwmin repair [kg/kg/s] |
| `dxicor` | ice ccwmin repair [kg/kg/s] |

### `param/*`, `meta/*`, `timestep/*`

`param/*`: ECHAM's constants as compiled (mo_echam_cloud_params, mo_cloud_utils): `jbmin` = 40, `jbmax` = 45, `cqtmin` = 1e-12, `cthomi` = 238.15, `ceffmin` = 10, `ceffmax` = 150, `crhoi` = 500, `ccsaut` = 95, `ccraut` = 15, `cvtfall` = 2.5, `ccwmin` = 1e-07, `crhosno` = 100, `cn0s` = 3e+06, `icemin` = 10, `icemax` = 1e+07, `conv_effr2mvr` = 0.9, `clc_min` = 0.01, `fact_PK` = 0.008253, `pow_PK` = 2.475, `rhoice` = 925, `fact_tke` = 0.7, `epsec` = 1e-12, `eps` = 2.22045e-16.
`meta/names`, `meta/process`, `meta/description`: the column catalogue below.

## Columns

Each starts from a quiescent, sub-saturated, cloud-free base (no condensate, numbers,
tendencies, TKE or detrainment) and modifies only the levels named.

| # | name | process | description |
|---|---|---|---|
| 0 | `null_clear` | none | Sub-saturated, cloud-free, no tendencies, no detrainment: the scheme only sets the number concentrations to their floors (cqtmin), and the ccwmin repair returns the number tracers to 0. |
| 1 | `detrainment_cold` | 1 znidetr/ll_cv (T < cthomi) + 4 lo2 split | T < cthomi: detrained condensate zxtec into a clear cell (cf 0, 250 hPa, 222 K; the top detraining level, so no precipitation reaches it and all of its detrainment sublimates), a cloudy cell (cf 0.5, 300 hPa, 225 K, no ice, ICNC at cqtmin), a cloudy cell with large detrainment whose znidetr exceeds icemax (cf 0.5, 400 hPa, 233 K; the F 1252 cap), and a cloudy turbulent cell (cf 0.3, 450 hPa, 236 K, TKE 1 m2/s2: below cthomi ll_cv needs no lo2_2d). ll_cv is true where cf > clc_min, lo2 true, all detrainment goes to ice. |
| 2 | `detrainment_mixed_lo2_true` | 1 lo2_2d true -> znidetr > 0; 4 lo2 true -> ice | cthomi < T < tmelt with existing ice at high ICNC and weak turbulence, so the Korolev-Mazin threshold exceeds the updraft in both sections: 600 hPa, 255 K, cf 0.5, ice 2e-5, ICNC 1e6 m-3, TKE 1e-4 (updraft 0.7 cm/s); 650 hPa, 262 K, cf 0.4, ice 1e-5, ICNC 5e5, TKE 0. Detrained condensate becomes ice with znidetr > 0. Also 700 hPa, 272 K, cf 0.5, ice 1e-5, ICNC 1e6, TKE 0, with the CONVECTION temperature ztconv = 274 K > tmelt: convection counted the condensate as liquid, lo2 puts it in ice, and ECHAM applies no latent-heat correction (F 1302 tests ztconv <= tmelt only). |
| 3 | `detrainment_mixed_lo2_false` | 1 lo2_2d false -> znidetr = cqtmin; 4 lo2 false -> liquid + ptte fix | cthomi < T < tmelt with tiny ICNC and strong turbulence (TKE 0.5, updraft 49.5 cm/s): lo2_2d and lo2 false, the detrained condensate becomes liquid, znidetr = cqtmin, and since ztconv <= tmelt ECHAM subtracts (als-alv)*zxtec/cpd from ptte (F 1300-1307). Levels: 550 hPa 252 K cf 0 (clear and the top detraining level: all of it evaporates, no precipitation from above); 600 hPa 255 K cf 0.5 no ice; 700 hPa 268 K cf 0.5 with ice 1e-7 and ICNC 1e3. |
| 4 | `detrainment_warm` | 1 ll_cv false (T > tmelt); 4 lo2 false -> liquid, no number | T > tmelt: detrained condensate into a clear cell (cf 0, 800 hPa, 277 K; top detraining level) and a cloudy cell (cf 0.5, 850 hPa, 280 K): liquid, znidetr = cqtmin, no energy fix (ztconv > tmelt). |
| 5 | `sediment_then_detrain` | 4 sedimentation of pxim1+ztmst*pxite only, then += znidetr | Cirrus (300 hPa, 225 K, cf 0.6) holding ice 5e-5 at a low ICNC (2e4 m-3, large fast crystals) with an upstream ice tendency pxite 1e-9 AND detrained condensate 1e-7 at the same level: sedimentation acts on max(pxim1+ztmst*pxite, EPSILON) only (F 1227-1228), the detrained ice joins through zxidt afterwards (F 1316) and znidetr joins the ICNC after sedimentation (F 1251). The ice falls into a cloudy level below (350 hPa, 230 K, cf 0.3, ice-saturated so the falling ice does not sublimate). |
| 6 | `number_clamp` | 8 pxtte_cdnc/icnc against the RAW pxtm1 (F 1781, 3625-3652) | Raw number tracers outside the physical range, negative and above icemax/rho. Clear ice-free cells (500 hPa: icnc -1e3/rho, cdnc -1e6/rho; 550 hPa: icnc 5e7/rho, cdnc 1e12/rho): the scheme's numbers are cqtmin and the ccwmin repair sets the end-of-step tracers to 0, so pxtte = -pxtm1/ztmst. A cold cloudy cell with ice (300 hPa, 225 K, cf 0.5, ice 3e-5, icnc 5e7/rho, cdnc -1e6/rho, plus an upstream ICNC tendency pxtte 1e3/rho/s): the ICNC is capped at icemax after sedimentation (F 1252). A warm liquid cloud (850 hPa, 280 K, cf 0.5, liquid 1e-4, icnc -1e3/rho, cdnc 1e12/rho). |
| 7 | `icnc_diagnosis_zrid` | 5.5 ICNC diagnosis at prid = zrid (F 1511, 2610-2624) | Cold cloudy cells holding ice with no crystal number (tracer 0, so picnc reaches update_in_cloud_water at icemin and the diagnosis 0.75*rho*zxib/(pi*rhoice*zrid**3) replaces it): 150 hPa 200 K, 250 hPa 220 K, 400 hPa 233 K (cf 0.5, ice 1e-5), and a mixed-phase cell 600 hPa 250 K (cf 0.5, ice 1e-5, TKE 0, lo2 true). |

## Regenerate

On the local harness branch `harness/echam-cloud-2m`
(`fortran_harness/echam_cloud2m`): `cd py && python build_reference_2m.py
<jcm>/jcm/data/test/echam_cloud_reference <table_dir>` (numpy; it runs `make`, which
copies the ECHAM files, generates and applies the diagnostic patch, and builds).
