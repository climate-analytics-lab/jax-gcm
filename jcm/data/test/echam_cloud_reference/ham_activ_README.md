# ECHAM6.3-HAM2.3 M7 activation reference data (#1017)

Outputs of the **unmodified** ECHAM6.3-HAM2.3 r7492 activation routines (`mo_activ.f90::activ_updraft/activ_lin_leaitch`, `mo_ham_activ.f90::ham_activ_koehler_ab/ham_activ_abdulrazzak_ghan/ham_avail_activ_lin_leaitch`, `mo_ham_tools.f90::ham_m7_logtail`, `mo_ham_m7.f90::m7_cumulative_normal`), extracted verbatim (see `ham_activ_provenance.json` for the exact line ranges and why full-file compilation was not tractable) and compiled standalone in double precision with declaration-only stubs of the HAM modules they USE, on designed single-level M7 aerosol cells. They are the numerical reference for `jcm/physics/aerosol/jam/activation/ham_activation.py`, compared by `ham_activation_reference_test.py`. **Data only**: no ECHAM or HAM source is part of this repository.

Integrity: -O0 vs -O2 max abs difference 0.0e+00 (`ham_activ_provenance.json`).

## Arrays (one entry per cell, `meta/names` order)

| name | meaning |
|---|---|
| `in/mass/<mode>/<species>` | tracer mass mixing ratio pxtm1 [kg/kg] |
| `in/number/<mode>` | mode number pxtm1 [1/kg] |
| `in/rdry/<mode>`, `in/rwet/<mode>` | dry / wet radius [m] (M7-core output, fed in directly here since only the activation routines are under test) |
| `in/t`, `in/p`, `in/q`, `in/esw` | temperature [K], pressure [Pa], specific humidity [kg/kg], saturation vapour pressure [Pa] (exogenous input to ARG) |
| `in/tke`, `in/omega`, `in/rho` | TKE [m2/s2], large-scale omega [Pa/s], air density [kg/m3] |
| `in/density/<sp>`, `in/moleweight/<sp>`, `in/nion/<sp>`, `in/osm/<sp>` | `mo_ham_species.f90` electrolyte properties |
| `out/a/<mode>`, `out/b/<mode>` | Koehler A [m], B [-] per mode |
| `out/sc/<mode>` | critical supersaturation Sm per mode [0-1] (nw-independent) |
| `out/cdncact0`, `out/cdncact1` | total activated CDNC [m-3], nactivpdf=0 / =1 (PDF) |
| `out/nact0/<mode>`, `out/nact1/<mode>` | per-mode activated number [m-3] |
| `out/fracn0/<mode>`, `out/fracn1/<mode>` | per-mode activated fraction [-] |
| `out/rc0/<mode>` | per-mode critical radius [m], nactivpdf=0 (single bin) |
| `out/rc1/<mode>` | per-mode critical radius [m] per PDF bin, shape (ncol,20) |
| `out/smax0` | maximum supersaturation [-], nactivpdf=0 |
| `out/smax1` | maximum supersaturation [-] per PDF bin, shape (ncol,20) |
| `out/na`, `out/na_cv` | Lin-Leaitch available number, stratiform/convective cut [m-3] |
| `out/cdncact_ll`, `out/cdncact_cv` | Lin-Leaitch activated CDNC, stratiform/convective [m-3] |
| `out/w0` | the ARG (ncd_activ=2) nactivpdf=0 updraft [m/s] |
| `out/wll` | the Lin-Leaitch (ncd_activ=1) updraft [m/s] -- its own activ_updraft call: mo_activ.f90:122-126's w_turb prefactor is 1.33, not ARG's 0.7, so this is NOT `out/w0` |
| `out/w_large`, `out/w_turb` | w_large/w_turb as left by the LAST activ_updraft call (the Lin-Leaitch one, so w_turb pairs with `out/wll`, not `out/w0`) [m/s] |

## Cells

| # | name | description |
|---|---|---|
| 0 | `pure_so4_small_n` | pure SO4 in AS, small number -- near-complete activation |
| 1 | `pure_so4_large_n` | pure SO4 in AS, large number (polluted) -- suppressed fraction |
| 2 | `ss_rich_coarse` | sea-salt-rich CS (coarse soluble), kappa~1 via nion/osm |
| 3 | `bc_oc_rich_ks` | BC/OC-rich KS (Aitken soluble) internally mixed with SO4 |
| 4 | `dust_laden_as` | dust-laden AS (accumulation soluble), SO4 + DU |
| 5 | `dust_laden_cs` | dust-laden CS (coarse soluble), SS + DU |
| 6 | `all_modes_populated` | every M7 mode carries mass and number |
| 7 | `empty_ks_mode` | KS (activatable Aitken) mode entirely empty: AS/CS carry all activation |
| 8 | `cold_high_alt` | T=230K below cthomi=238.15K: ARG returns zero activation (gate) |
| 9 | `warm_surface` | T=300K, p=1000hPa: warm boundary-layer cell |
| 10 | `mid_t_mid_p` | T=270K, p=700hPa mid-troposphere cell |
| 11 | `strong_updraft` | omega<<0 (ascending), strong TKE: w_large>0 and large w_turb |
| 12 | `descending_weak_tke_wmin_binds` | omega>0 (descending) + weak TKE: w_large+w_turb=-0.25<0, MAX(w_min,.) clips the nactivpdf=0 updraft to 0. (omega/tke are tuned so |w_large|~3*sigma_pdf, not so extreme that the PDF run's Gaussian underflows to an all-zero bin weight -- see provenance notes.) |
| 13 | `descending_strong_tke` | omega>0 (descending) but TKE keeps w_large+w_turb>0 |
| 14 | `near_zero_w` | omega~0, tiny TKE: w_large~0 and w_turb near its floor |
| 15 | `w_turb_high` | very large TKE: w_turb at the top of the designed range (2 m/s) |
| 16 | `w_turb_low` | very small TKE: w_turb at the bottom of the designed range (0.05 m/s) |
| 17 | `insoluble_dominant` | insoluble AI/CI/KI dust+BC/OC dominate; AS carries little SO4 |
| 18 | `high_number_polluted_ks_as_cs` | KS+AS+CS all near 1e10 m^-3: strongly number-limited |
| 19 | `low_number_pristine` | KS+AS+CS all at 1e6 m^-3: pristine, near-total activation |

## Regenerate

`python build_reference_hamactiv.py <jcm>/jcm/data/test/echam_cloud_reference`
from the harness copy `fortran_harness/echam_hamactiv/py`.
