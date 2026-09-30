# ECHAM6.3-HAM2.3 heterogeneous mixed-phase freezing reference data (#953)

Inputs, outputs and intermediates of the **unmodified** ECHAM6.3-HAM2.3 r7492 routine
`mo_cloud_micro_2m.f90::cloud_micro_interface`, run in double precision on designed
supercooled cloud columns with the HAM freezing inputs of `het_mxphase_freezing`
(F 2675-2840; the fractions and wet radii `mo_ham_freezing.f90::ham_IN_setup`
returns) set per level through the harness's `cloud_subm_1` stub. They are the
numerical reference for `jcm/physics/clouds/lohmann_2m/deposition_freezing.py::het_mxphase_freezing`
and its section-6.2 wiring, compared by
`jcm/physics/clouds/lohmann_2m_freezing_reference_test.py`. Same conventions, grid,
time steps, constants and harness as `cloud2m_README.md` (TOP-FIRST levels, `(nlev, ncol)`
layout, `timestep/dt1200` = ECHAM's T63 leapfrog, `timestep/dt720` = jcm's step); only
the aerosol inputs differ. **Data only**: no ECHAM source is part of this repository.

Integrity (`cloud2m_frz_provenance.json`): patched = pristine outputs bit for bit,
-O0 vs -O2 max abs difference dt1200 0.0e+00, dt720 0.0e+00,
each column alone = all together (outputs), and the extended harness reproduces the #941 fixture
`cloud2m_T63L47.npz` bit for bit (all outputs and intermediates) with the freezing inputs 0.

## Arrays

`in/*` as in `cloud2m_README.md`, plus the HAM freezing inputs:

| name | meaning |
|---|---|
| `in/fracdusol` | HAM pfracdusol: dust fraction of the activated droplets (immersion) |
| `in/fracduai` | HAM pfracduai: insoluble accumulation dust over all insoluble aerosol (contact) |
| `in/fracduci` | HAM pfracduci: insoluble coarse dust over all insoluble aerosol (contact) |
| `in/fracbcsol` | HAM pfracbcsol: BC fraction of the activated droplets (immersion) |
| `in/fracbcinsol` | HAM pfracbcinsol: insoluble BC over all insoluble aerosol (contact; disabled in ECHAM) |
| `in/rwetki` | wet radius of the insoluble Aitken mode [m] |
| `in/rwetai` | wet radius of the insoluble accumulation mode [m] |
| `in/rwetci` | wet radius of the insoluble coarse mode [m] |
| `in/pvervel` | large-scale omega [Pa/s] (nonzero only in frz_omega) |

`out/<step>/*`: every INOUT/OUT argument, as in `cloud2m_README.md`.

`diag/<step>/*`: the subset of intermediates the freezing tests read:

| name | meaning |
|---|---|
| `hf_mask` | ll_mxphase_frz, the 6.2 gate (1/0), F 1541-1545 |
| `hf_cdnc_in` | CDNC entering het_mxphase_freezing [1/m3] |
| `hf_icnc_in` | ICNC entering het_mxphase_freezing [1/m3] |
| `hf_xlb_in` | in-cloud liquid entering het_mxphase_freezing [kg/kg] |
| `hf_xib_in` | in-cloud ice entering het_mxphase_freezing [kg/kg] |
| `hf_tp1tmp` | temperature het_mxphase_freezing sees (ptp1tmp) [K] |
| `hf_cdncmin` | minimum CDNC het_mxphase_freezing sees [1/m3] |
| `hf_icnc` | ICNC after het_mxphase_freezing [1/m3], F 2827-2829 |
| `hf_cdnc` | CDNC after het_mxphase_freezing [1/m3], F 2823-2825 |
| `hf_xib` | in-cloud ice after het_mxphase_freezing [kg/kg], F 2834-2835 |
| `hf_xlb` | in-cloud liquid after het_mxphase_freezing [kg/kg], F 2831-2832 |
| `hf_frl` | grid-mean freezing after het_mxphase_freezing (pfrl*paclc) [kg/kg], F 2837-2838 |
| `hf_dfbcki` | Brownian diffusivity, insoluble Aitken (BC) [m2/s], F 2757-2759 |
| `hf_dfduai` | Brownian diffusivity, insoluble accumulation dust [m2/s], F 2763-2765 |
| `hf_dfduci` | Brownian diffusivity, insoluble coarse dust [m2/s], F 2769-2771 |
| `hf_frzcnt` | contact-frozen in-cloud liquid this step [kg/kg], F 2786-2792 |
| `hf_ztte` | cooling rate zomega/(cpd*rho) [K/s], F 2800-2802 |
| `hf_frzimm` | immersion-frozen in-cloud liquid this step [kg/kg], F 2804-2805 |
| `hf_frl_ic` | in-cloud frozen mass MAX(0,MIN(contact+immersion,pxlb)) where the gate holds [kg/kg], F 2807-2817 |
| `hf_frln` | frozen number MAX(MIN(pcdnc*frl/(pxlb+eps), pcdnc-pcdnc_min),0) [1/m3], F 2811-2821 |
| `rho` | air density papm1/(rd*ptvm1) [kg/m3], F 578 |
| `frl_hom` | homogeneous freezing [kg/kg] |
| `icnc_61` | ICNC after 6.1 [1/m3] |
| `uicw_aclc` | cloud cover after update_in_cloud_water |
| `uicw_xlb` | in-cloud liquid after update_in_cloud_water [kg/kg] |
| `uicw_xib` | in-cloud ice after update_in_cloud_water [kg/kg] |
| `uicw_cdnc` | CDNC after update_in_cloud_water [1/m3] |
| `uicw_icnc` | ICNC after update_in_cloud_water [1/m3], F 2610-2624 |
| `tp1tmp` | temperature after condensation [K] |
| `cdnc_min_in` | minimum CDNC [1/m3] (cdnc_min_fixed), F 609-610 |
| `vervx` | updraft velocity [cm/s], F 814-816 |
| `frl` | freezing 6.1+6.2 [kg/kg] |
| `icnc_6` | ICNC after 6.2 and WBF [1/m3] |
| `xib_6` | in-cloud ice after section 6 [kg/kg] |
| `xlb_6` | in-cloud liquid after section 6 [kg/kg] |

The `hf_*` values after the call (`hf_icnc`, `hf_cdnc`, `hf_xib`, `hf_xlb`, `hf_frl`)
and inside it (`hf_df*`, `hf_frzcnt`, `hf_ztte`, `hf_frzimm`, `hf_frl_ic`, `hf_frln`) are
recorded only on levels where some column passes the 6.2 gate (ECHAM calls the routine
under `IF (ANY(ll_mxphase_frz))`, F 1547); elsewhere they are 0.

## Columns

| # | name | process | description |
|---|---|---|---|
| 0 | `frz_none` | 6.2 het_mxphase_freezing (F 2675-2840) | Supercooled liquid clouds (cf 0.5, in-cloud liquid 2e-4 kg/kg, CDNC 8e7 m-3, saturated over water) at 700/600/550/500/450 hPa and 268/258/250/243/239 K, TKE 0.1 m2/s2; no dust or BC: nothing freezes heterogeneously. The 600 hPa level also holds ice 1e-6 kg/kg at ICNC 1e4 m-3. |
| 1 | `frz_dust` | 6.2 het_mxphase_freezing (F 2675-2840) | Supercooled liquid clouds (cf 0.5, in-cloud liquid 2e-4 kg/kg, CDNC 8e7 m-3, saturated over water) at 700/600/550/500/450 hPa and 268/258/250/243/239 K, TKE 0.1 m2/s2; dust only: immersion (fracdusol 0.3) and Brownian contact on insoluble accumulation (fracduai 0.2, rwetai 0.3 um) and coarse (fracduci 0.1, rwetci 1.5 um) dust. The 600 hPa level also holds ice 1e-6 kg/kg at ICNC 1e4 m-3. |
| 2 | `frz_bc` | 6.2 het_mxphase_freezing (F 2675-2840) | Supercooled liquid clouds (cf 0.5, in-cloud liquid 2e-4 kg/kg, CDNC 8e7 m-3, saturated over water) at 700/600/550/500/450 hPa and 268/258/250/243/239 K, TKE 0.1 m2/s2; black carbon only: immersion (fracbcsol 0.5; coefficient 2.91e-3) and the insoluble-Aitken inputs (fracbcinsol 0.4, rwetki 0.05 um) of the BC contact term, which ECHAM disables (zfrzcntbc = 0, F 2784). The 600 hPa level also holds ice 1e-6 kg/kg at ICNC 1e4 m-3. |
| 3 | `frz_both` | 6.2 het_mxphase_freezing (F 2675-2840) | Supercooled liquid clouds (cf 0.5, in-cloud liquid 2e-4 kg/kg, CDNC 8e7 m-3, saturated over water) at 700/600/550/500/450 hPa and 268/258/250/243/239 K, TKE 0.1 m2/s2; dust and black carbon together. The 600 hPa level also holds ice 1e-6 kg/kg at ICNC 1e4 m-3. |
| 4 | `frz_dust_nocool` | 6.2 het_mxphase_freezing (F 2675-2840) | Supercooled liquid clouds (cf 0.5, in-cloud liquid 2e-4 kg/kg, CDNC 8e7 m-3, saturated over water) at 700/600/550/500/450 hPa and 268/258/250/243/239 K, TKE 0 m2/s2; dust with TKE 0 and no large-scale omega: ztte = 0 switches immersion off (MIN(ztte, 0), F 2804), contact still acts. The 600 hPa level also holds ice 1e-6 kg/kg at ICNC 1e4 m-3. |
| 5 | `frz_outside_window` | 6.2 gate ll_mxphase_frz false (F 1541-1545) | Dust and BC present, outside or at the edge of the 6.2 gate: a warm liquid cloud (850 hPa, 278 K: no gate), a cold cloud below cthomi (300 hPa, 230 K: homogeneous freezing, 6.1), and a supercooled cloud (600 hPa, 255 K) whose CDNC tracer (2e7 m-3) is raised to the 40 cm-3 floor on entry (F 1124-1125): the gate holds, liquid freezes, but the frozen number MIN(., pcdnc - pcdnc_min) is 0 (F 2819-2821). |
| 6 | `frz_omega` | 6.2 het_mxphase_freezing with large-scale omega (F 2800) | Dust, TKE 0.01 m2/s2 and a large-scale pressure velocity: ascent -0.5 Pa/s at 600 and 500 hPa, subsidence +1.5 Pa/s at 550 hPa (ztte > 0: no immersion). jcm has no omega in the scheme (#705), so its comparison is a strict xfail. |

## Regenerate

From the harness copy (`fortran_harness/echam_cloud2m/py`):
`python build_reference_frz.py <jcm>/jcm/data/test/echam_cloud_reference`.
