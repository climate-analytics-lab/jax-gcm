# Standard ARMBE Variable Cadence

This inventory covers every time-dependent variable in the downloaded standard
`armbeatm`, `armbecldrad`, and `armbeland` files from order `267892`.
High-resolution products are excluded. All products use one-hour time cells;
`dominant interval` describes spacing between finite cells, not an averaging
window. `Availability` is the finite-cell fraction within files containing that
variable, pooled across sites and years. For profile variables, a cell is
available when at least one vertical level is finite.

Temporal semantics:

- `one_hour_mean`: a one-hour aggregate, documented by the variable or product.
- `sounding_associated_hourly_cell`: a radiosonde profile placed in its one-hour
  cell; sparse availability does not imply a multi-hour mean.
- `nwp_analysis_on_hourly_grid`: an NWP analysis field, not an observed hourly mean.
- `within_hour_standard_deviation`: spread of native samples within the hour.
- `quality_flag_for_hourly_cell` and `source_or_provenance_flag`: ancillary values.
- `hourly_product_value_operator_not_explicit`: hourly-grid field whose exact
  averaging operator is not stated consistently across all downloaded versions.

## ARMBEATM

| Variable | Temporal semantics | Availability | Dominant interval | Common intervals | Sites |
|---|---|---:|---:|---|---|
| `LH_baebbr` | `one_hour_mean` | 80.7% | 1h | 1h:132979; 2h:897; 3h:148; 4h:59; 5h:46 | sgp |
| `LH_qcecor` | `one_hour_mean` | 6.8% | 1h | 1h:11147; 2h:91; 3h:29; 5h:17; 6h:10 | sgp |
| `SH_baebbr` | `one_hour_mean` | 79.8% | 1h | 1h:131046; 2h:1049; 3h:227; 4h:107; 5h:85 | sgp |
| `SH_qcecor` | `one_hour_mean` | 7.5% | 1h | 1h:12350; 2h:132; 3h:38; 4h:17; 5h:17 | sgp |
| `T_nwp_p` | `nwp_analysis_on_hourly_grid` | 72.4% | 1h | 1h:119027; 2h:1366; 3h:122; 4h:18; 5h:16 | sgp |
| `T_p` | `sounding_associated_hourly_cell` | 9.2% | 12h | 12h:19146; 6h:16789; 3h:6436; 9h:1818; 11h:1652 | nsa; sgp; twp |
| `T_sfc` | `one_hour_mean` | 87.3% | 1h | 1h:504966; 2h:86; 3h:37; 4h:17; 25h:10 | nsa; sgp; twp |
| `T_z` | `sounding_associated_hourly_cell` | 9.2% | 12h | 12h:19150; 6h:16789; 3h:6437; 9h:1818; 11h:1652 | nsa; sgp; twp |
| `Td_p` | `sounding_associated_hourly_cell` | 9.2% | 12h | 12h:19051; 6h:16780; 3h:6419; 9h:1815; 1d:1648 | nsa; sgp; twp |
| `Td_z` | `sounding_associated_hourly_cell` | 9.2% | 12h | 12h:19055; 6h:16780; 3h:6419; 9h:1815; 1d:1648 | nsa; sgp; twp |
| `dewpoint_h` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25006; 12h:14799; 3h:2261; 18h:1926; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `dewpoint_p` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25006; 12h:14795; 3h:2261; 18h:1926; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `latent_heat_flux_baebbr` | `one_hour_mean` | 20.0% | 1h | 1h:87669; 2h:726; 3h:85; 4h:27; 5h:26 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `latent_heat_flux_qcecor` | `one_hour_mean` | 53.6% | 1h | 1h:227829; 2h:3475; 3h:1417; 4h:1230; 5h:599 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `omega_nwp_p` | `nwp_analysis_on_hourly_grid` | 34.6% | 1h | 1h:208233; 2h:2091; 3h:179; 6h:89; 4h:51 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `p_sfc` | `one_hour_mean` | 87.2% | 1h | 1h:504302; 2h:52; 3h:32; 4h:16; 25h:11 | nsa; sgp; twp |
| `prec_sfc` | `one_hour_mean` | 73.6% | 1h | 1h:425346; 2h:60; 3h:12; 4h:12; 25h:9 | nsa; sgp; twp |
| `precip_rate_sfc` | `one_hour_mean` | 97.7% | 1h | 1h:432442; 2h:65; 3h:55; 7h:36; 4h:35 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `pressure_sfc` | `one_hour_mean` | 99.4% | 1h | 1h:440338; 2h:23; 9h:22; 4h:19; 6h:17 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `qc_precip_sfc` | `quality_flag_for_hourly_cell` | 0.0% | none | none | sgp; twp |
| `relative_humidity_h` | `sounding_associated_hourly_cell` | 9.8% | 6h | 6h:24465; 12h:11084; 3h:2144; 18h:1935; 4h:758 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `relative_humidity_nwp_p` | `nwp_analysis_on_hourly_grid` | 20.3% | 1h | 1h:89222; 2h:721; 6h:78; 3h:55; 4h:33 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `relative_humidity_p` | `sounding_associated_hourly_cell` | 9.8% | 6h | 6h:24465; 12h:11082; 3h:2144; 18h:1935; 4h:758 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `relative_humidity_sfc` | `one_hour_mean` | 98.9% | 1h | 1h:438081; 2h:83; 3h:42; 4h:41; 9h:27 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `rh_nwp_p` | `nwp_analysis_on_hourly_grid` | 72.4% | 1h | 1h:119018; 2h:1370; 3h:122; 4h:18; 5h:16 | sgp |
| `rh_p` | `sounding_associated_hourly_cell` | 8.4% | 12h | 12h:18411; 6h:15653; 3h:4353; 1d:1627; 11h:1607 | nsa; sgp; twp |
| `rh_sfc` | `one_hour_mean` | 87.3% | 1h | 1h:504856; 2h:84; 3h:39; 4h:21; 5h:11 | nsa; sgp; twp |
| `rh_z` | `sounding_associated_hourly_cell` | 8.4% | 12h | 12h:18412; 6h:15653; 3h:4354; 1d:1627; 11h:1607 | nsa; sgp; twp |
| `sensible_heat_flux_baebbr` | `one_hour_mean` | 20.0% | 1h | 1h:87425; 2h:739; 3h:120; 4h:45; 5h:25 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `sensible_heat_flux_qcecor` | `one_hour_mean` | 71.2% | 1h | 1h:309163; 2h:2998; 3h:1064; 4h:858; 5h:608 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `temperature_h` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25030; 12h:14795; 3h:2261; 18h:1927; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `temperature_nwp_p` | `nwp_analysis_on_hourly_grid` | 20.3% | 1h | 1h:89222; 2h:721; 6h:78; 3h:55; 4h:33 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `temperature_p` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25030; 12h:14795; 3h:2261; 18h:1927; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `temperature_sfc` | `one_hour_mean` | 97.9% | 1h | 1h:433806; 2h:51; 4h:28; 9h:25; 3h:21 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `time` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:1021748 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `time_bounds` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:1021748 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `time_frac` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:666092 | nsa; sgp; twp |
| `time_offset` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:1021748 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `u_nwp_p` | `nwp_analysis_on_hourly_grid` | 72.4% | 1h | 1h:119011; 2h:1370; 3h:124; 4h:18; 5h:16 | sgp |
| `u_p` | `sounding_associated_hourly_cell` | 9.8% | 6h | 6h:17889; 12h:13313; 3h:6347; 7h:4964; 5h:4934 | nsa; sgp; twp |
| `u_sfc` | `one_hour_mean` | 87.0% | 1h | 1h:503027; 2h:70; 3h:20; 4h:17; 5h:9 | nsa; sgp; twp |
| `u_wind_h` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25067; 12h:14824; 3h:2262; 18h:1936; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `u_wind_nwp_p` | `nwp_analysis_on_hourly_grid` | 20.3% | 1h | 1h:89222; 2h:721; 6h:78; 3h:55; 4h:33 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `u_wind_p` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25050; 12h:14796; 3h:2262; 18h:1921; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `u_wind_sfc` | `one_hour_mean` | 98.2% | 1h | 1h:434662; 2h:134; 3h:57; 4h:38; 5h:35 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `u_z` | `sounding_associated_hourly_cell` | 10.0% | 6h | 6h:17895; 12h:13404; 3h:6380; 7h:4969; 5h:4958 | nsa; sgp; twp |
| `v_nwp_p` | `nwp_analysis_on_hourly_grid` | 72.4% | 1h | 1h:119009; 2h:1371; 3h:124; 4h:18; 5h:16 | sgp |
| `v_p` | `sounding_associated_hourly_cell` | 9.8% | 6h | 6h:17889; 12h:13312; 3h:6347; 7h:4964; 5h:4934 | nsa; sgp; twp |
| `v_sfc` | `one_hour_mean` | 87.0% | 1h | 1h:503029; 2h:69; 3h:20; 4h:16; 5h:9 | nsa; sgp; twp |
| `v_wind_h` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25067; 12h:14824; 3h:2262; 18h:1936; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `v_wind_nwp_p` | `nwp_analysis_on_hourly_grid` | 20.3% | 1h | 1h:89222; 2h:721; 6h:78; 3h:55; 4h:33 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `v_wind_p` | `sounding_associated_hourly_cell` | 11.0% | 6h | 6h:25050; 12h:14796; 3h:2262; 18h:1921; 4h:966 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `v_wind_sfc` | `one_hour_mean` | 98.2% | 1h | 1h:434658; 2h:136; 3h:57; 4h:41; 5h:36 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `v_z` | `sounding_associated_hourly_cell` | 10.0% | 6h | 6h:17895; 12h:13404; 3h:6380; 7h:4969; 5h:4958 | nsa; sgp; twp |

## ARMBECLDRAD

| Variable | Temporal semantics | Availability | Dominant interval | Common intervals | Sites |
|---|---|---:|---:|---|---|
| `alt` | `hourly_product_value_operator_not_explicit` | 75.4% | 1h | 1h:6604; 2h:10; 3h:2; 6h:2; 9h:2 | mos |
| `cld_base_source_status` | `hourly_product_value_operator_not_explicit` | 92.1% | 1h | 1h:488196; 25h:8; 73h:3; 145h:2; 121h:2 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp |
| `cld_frac` | `one_hour_mean` | 79.4% | 1h | 1h:900414; 2h:174; 25h:63; 3h:61; 4h:46 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `cld_frac_MMCR` | `one_hour_mean` | 59.3% | 1h | 1h:357273; 2h:173; 25h:153; 3h:126; 4h:83 | nsa; sgp; twp |
| `cld_frac_MPL` | `one_hour_mean` | 71.8% | 1h | 1h:812016; 2h:644; 3h:253; 6h:167; 4h:128 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `cld_frac_radar` | `one_hour_mean` | 87.2% | 1h | 1h:462069; 2h:68; 4h:53; 3h:53; 5h:25 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp |
| `cld_high` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.6% | 1h | 1h:348207; 3h:1648; 2h:1552; 4h:219; 11h:89 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `cld_high_sat_CERES` | `hourly_product_value_operator_not_explicit` | 8.8% | 3h | 3h:6201; 9h:2698; 10h:2157; 2h:1590; 12h:1435 | nsa; twp |
| `cld_high_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `cld_low` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.6% | 1h | 1h:348207; 3h:1648; 2h:1552; 4h:219; 11h:89 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `cld_low_sat_CERES` | `hourly_product_value_operator_not_explicit` | 15.3% | 2h | 2h:13424; 1h:10244; 3h:3486; 4h:1332; 12h:971 | nsa; twp |
| `cld_low_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `cld_mid` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.6% | 1h | 1h:348207; 3h:1648; 2h:1552; 4h:219; 11h:89 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `cld_mid_sat_CERES` | `hourly_product_value_operator_not_explicit` | 11.1% | 2h | 2h:6563; 1h:4988; 3h:3818; 4h:893; 12h:888 | nsa; twp |
| `cld_mid_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `cld_thick` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.7% | 1h | 1h:349886; 3h:1551; 2h:1468; 11h:77; 9h:75 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `cld_thick_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `cld_top` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 33.3% | 1h | 1h:281256; 2h:4578; 3h:2975; 4h:1431; 5h:964 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `cld_top_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `cld_tot` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.6% | 1h | 1h:348207; 3h:1648; 2h:1552; 4h:219; 11h:89 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `clrswfluxdn` | `one_hour_mean` | 84.2% | 1h | 1h:526989; 2h:7508; 3h:408; 12h:55; 11h:54 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `lat` | `hourly_product_value_operator_not_explicit` | 75.4% | 1h | 1h:6604; 2h:10; 3h:2; 6h:2; 9h:2 | mos |
| `lon` | `hourly_product_value_operator_not_explicit` | 75.4% | 1h | 1h:6604; 2h:10; 3h:2; 6h:2; 9h:2 | mos |
| `lw_net_TOA` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.6% | 1h | 1h:348181; 3h:1654; 2h:1554; 4h:220; 11h:89 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `lwdn` | `one_hour_mean` | 90.0% | 1h | 1h:1019798; 2h:703; 3h:301; 4h:184; 5h:101 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `lwnet_TOA_sat_CERES` | `hourly_product_value_operator_not_explicit` | 24.7% | 2h | 2h:19167; 1h:14288; 3h:9044; 9h:3863; 10h:3100 | nsa; twp |
| `lwnet_TOA_sat_VISST` | `hourly_product_value_operator_not_explicit` | 23.0% | 1h | 1h:51221; 2h:1660; 4h:982; 3h:559; 5h:28 | nsa; twp |
| `lwnet_clr_TOA_sat_CERES` | `hourly_product_value_operator_not_explicit` | 24.7% | 2h | 2h:19167; 1h:14288; 3h:9044; 9h:3863; 10h:3100 | nsa; twp |
| `lwnet_clr_TOA_sat_VISST` | `hourly_product_value_operator_not_explicit` | 53.5% | 1h | 1h:37808; 2h:1784; 4h:786; 3h:640; 5h:174 | twp |
| `lwp` | `one_hour_mean` | 67.2% | 1h | 1h:723639; 2h:15686; 3h:6829; 4h:3803; 5h:2546 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `lwup` | `one_hour_mean` | 89.5% | 1h | 1h:1015014; 2h:487; 3h:193; 4h:128; 5h:52 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `num_samples_sat_CERES` | `hourly_product_value_operator_not_explicit` | 100.0% | 1h | 1h:236637 | nsa; twp |
| `pwv` | `one_hour_mean` | 67.6% | 1h | 1h:729546; 2h:15130; 3h:6480; 4h:3599; 5h:2426 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_cld_base_source` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:604755 | nsa; sgp; twp |
| `qc_cld_frac` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_cld_frac_source` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:604755 | nsa; sgp; twp |
| `qc_clrswfluxdn` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:635511 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `qc_lwdn` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_lwp` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_lwup` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_pwv` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_skycover` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:635511 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `qc_swdif` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1126327 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `qc_swdir` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1126327 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `qc_swdn` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_swup` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_tot_cld` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_tot_cld_tsi` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `qc_totswfluxdn` | `quality_flag_for_hourly_cell` | 100.0% | 1h | 1h:635511 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `skycover` | `one_hour_mean` | 31.9% | 1h | 1h:182296; 15h:4670; 16h:3168; 12h:2414; 14h:1835 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `source_cld_frac` | `source_or_provenance_flag` | 100.0% | 1h | 1h:530354 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp |
| `source_sat_CERES` | `source_or_provenance_flag` | 100.0% | 1h | 1h:236637 | nsa; twp |
| `stdev_clrswfluxdn` | `within_hour_standard_deviation` | 84.2% | 1h | 1h:526962; 2h:7509; 3h:407; 12h:55; 11h:53 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `stdev_lwdn` | `within_hour_standard_deviation` | 90.0% | 1h | 1h:1019798; 2h:703; 3h:301; 4h:184; 5h:101 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `stdev_lwp` | `within_hour_standard_deviation` | 67.2% | 1h | 1h:723639; 2h:15686; 3h:6829; 4h:3803; 5h:2546 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `stdev_lwup` | `within_hour_standard_deviation` | 89.5% | 1h | 1h:1015014; 2h:487; 3h:193; 4h:128; 5h:52 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `stdev_pwv` | `within_hour_standard_deviation` | 67.6% | 1h | 1h:729546; 2h:15130; 3h:6480; 4h:3599; 5h:2426 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `stdev_skycover` | `within_hour_standard_deviation` | 31.8% | 1h | 1h:181752; 15h:4666; 16h:3141; 12h:2380; 14h:1898 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `stdev_swdif` | `within_hour_standard_deviation` | 91.4% | 1h | 1h:1027597; 2h:761; 3h:323; 4h:130; 5h:67 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `stdev_swdir` | `within_hour_standard_deviation` | 91.4% | 1h | 1h:1027597; 2h:761; 3h:323; 4h:130; 5h:67 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `stdev_swdn` | `within_hour_standard_deviation` | 91.4% | 1h | 1h:1036476; 2h:322; 3h:175; 4h:52; 5h:33 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `stdev_swup` | `within_hour_standard_deviation` | 88.9% | 1h | 1h:1005240; 2h:1401; 3h:664; 4h:341; 5h:216 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `stdev_totswfluxdn` | `within_hour_standard_deviation` | 84.0% | 1h | 1h:525481; 2h:7499; 3h:570; 4h:65; 12h:60 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |
| `sw_albedo_allSky_sat_VISST` | `hourly_product_value_operator_not_explicit` | 73.9% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | twp |
| `sw_dn_TOA` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 40.9% | 1h | 1h:363640; 25h:13; 49h:4; 97h:3; 289h:3 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `sw_net_TOA` | `hourly_product_value_operator_not_explicit; one_hour_mean` | 39.6% | 1h | 1h:348914; 2h:1245; 3h:628; 4h:408; 5h:284 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `swdif` | `one_hour_mean` | 91.4% | 1h | 1h:1027597; 2h:761; 3h:323; 4h:130; 5h:67 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `swdir` | `one_hour_mean` | 91.4% | 1h | 1h:1027597; 2h:761; 3h:323; 4h:130; 5h:67 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp; twp |
| `swdn` | `one_hour_mean` | 91.4% | 1h | 1h:1036476; 2h:322; 3h:175; 4h:52; 5h:33 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `swdn_TOA_sat_CERES` | `hourly_product_value_operator_not_explicit` | 13.2% | 2h | 2h:9368; 1h:8285; 3h:4045; 20h:2505; 22h:1339 | nsa; twp |
| `swdn_TOA_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `swnet_TOA_sat_CERES` | `hourly_product_value_operator_not_explicit` | 13.2% | 2h | 2h:9368; 1h:8285; 3h:4045; 20h:2505; 22h:1339 | nsa; twp |
| `swnet_TOA_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `swnet_clr_TOA_sat_CERES` | `hourly_product_value_operator_not_explicit` | 13.2% | 2h | 2h:9368; 1h:8285; 3h:4045; 20h:2505; 22h:1339 | nsa; twp |
| `swnet_clr_TOA_sat_VISST` | `hourly_product_value_operator_not_explicit` | 73.9% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | twp |
| `swup` | `one_hour_mean` | 88.8% | 1h | 1h:1004078; 2h:1555; 3h:735; 4h:360; 5h:223 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `time` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `time_bounds` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `time_frac` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:673651 | nsa; sgp; twp |
| `time_offset` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:1135109 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `tot_cld` | `one_hour_mean` | 79.4% | 1h | 1h:900414; 2h:174; 25h:63; 3h:61; 4h:46 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `tot_cld_sat_CERES` | `hourly_product_value_operator_not_explicit` | 24.7% | 2h | 2h:19247; 1h:14264; 3h:8956; 9h:3803; 10h:3144 | nsa; twp |
| `tot_cld_sat_VISST` | `hourly_product_value_operator_not_explicit` | 24.6% | 1h | 1h:54813; 2h:1655; 4h:1020; 3h:732; 5h:34 | nsa; twp |
| `tot_cld_tsi` | `one_hour_mean` | 33.0% | 1h | 1h:347115; 13h:6488; 12h:4735; 10h:3023; 14h:2941 | anx; awr; cor; ena; epc; guc; hou; mao; mos; nim; nsa; oli; sgp; twp |
| `totswfluxdn` | `one_hour_mean` | 84.1% | 1h | 1h:525746; 2h:7539; 3h:482; 12h:60; 10h:58 | anx; awr; cor; ena; epc; guc; hou; mao; nim; nsa; oli; sgp |

## ARMBELAND

| Variable | Temporal semantics | Availability | Dominant interval | Common intervals | Sites |
|---|---|---:|---:|---|---|
| `co2_density` | `one_hour_mean` | 49.3% | 1h | 1h:96809; 2h:949; 3h:433; 4h:254; 5h:153 | sgp |
| `co2_flux` | `one_hour_mean` | 41.4% | 1h | 1h:76887; 2h:2353; 3h:1131; 4h:679; 5h:450 | sgp |
| `friction_velocity` | `one_hour_mean` | 52.0% | 1h | 1h:103116; 2h:951; 3h:110; 4h:22; 5h:12 | sgp |
| `ppfd` | `one_hour_mean` | 51.7% | 1h | 1h:103659; 2h:19; 4h:13; 25h:12; 5h:12 | sgp |
| `ppfd_up` | `one_hour_mean` | 42.9% | 1h | 1h:85931; 4h:20; 2h:20; 9h:15; 5h:15 | sgp |
| `soil_heat_flux_CO2FLX` | `one_hour_mean` | 46.5% | 1h | 1h:93093; 2h:34; 4h:30; 5h:17; 3h:16 | sgp |
| `soil_moisture_content_EBBR` | `one_hour_mean` | 90.8% | 1h | 1h:181801; 2h:225; 3h:70; 4h:21; 5h:16 | sgp |
| `soil_moisture_content_east_SWATS` | `one_hour_mean` | 90.4% | 1h | 1h:181291; 2h:23; 3h:20; 25h:10; 4h:5 | sgp |
| `soil_moisture_content_lower_CO2FLX` | `one_hour_mean` | 40.1% | 1h | 1h:80111; 2h:54; 3h:30; 5h:21; 4h:18 | sgp |
| `soil_moisture_content_upper_CO2FLX` | `one_hour_mean` | 35.0% | 1h | 1h:69208; 2h:220; 3h:130; 4h:74; 5h:64 | sgp |
| `soil_moisture_content_west_SWATS` | `one_hour_mean` | 90.4% | 1h | 1h:181314; 3h:19; 2h:18; 25h:10; 49h:5 | sgp |
| `soil_temperature_EBBR` | `one_hour_mean` | 97.8% | 1h | 1h:196023; 2h:114; 3h:34; 4h:15; 7h:13 | sgp |
| `soil_temperature_east_SWATS` | `one_hour_mean` | 90.4% | 1h | 1h:181292; 2h:23; 3h:21; 25h:10; 49h:5 | sgp |
| `soil_temperature_lower_CO2FLX` | `one_hour_mean` | 43.5% | 1h | 1h:87114; 2h:128; 3h:38; 4h:18; 25h:9 | sgp |
| `soil_temperature_middle_CO2FLX` | `one_hour_mean` | 43.8% | 1h | 1h:87544; 2h:134; 3h:31; 4h:22; 5h:12 | sgp |
| `soil_temperature_upper_CO2FLX` | `one_hour_mean` | 43.5% | 1h | 1h:87009; 2h:118; 3h:41; 4h:18; 5h:14 | sgp |
| `soil_temperature_west_SWATS` | `one_hour_mean` | 90.4% | 1h | 1h:181318; 3h:19; 2h:16; 25h:10; 49h:5 | sgp |
| `surface_soil_heat_flux_EBBR` | `one_hour_mean` | 85.0% | 1h | 1h:167695; 2h:1696; 3h:471; 4h:214; 5h:103 | sgp |
| `time` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:200693; 25h:13; 49h:4; 73h:1; 217h:1 | sgp |
| `time_bounds` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:200693; 25h:13; 49h:4; 73h:1; 217h:1 | sgp |
| `time_frac` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:200693; 25h:13; 49h:4; 73h:1; 217h:1 | sgp |
| `time_offset` | `time_coordinate_auxiliary` | 100.0% | 1h | 1h:200693; 25h:13; 49h:4; 73h:1; 217h:1 | sgp |
