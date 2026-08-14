# Mirror source data (all on NCAR Glade unless noted)

Destination: Hugging Face dataset `climate-analytics-lab/jax-gcm-data`.

Regridding always starts from the highest-resolution product available.

| product | path | resolution |
|---|---|---|
| CEDS anthropogenic (CMIP7 2025-04-18) | `/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18` | 0.5°, 8 sectors, monthly |
| BB4CMIP7 biomass (DRES 2-0) | `.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0` | native (~0.25°) |
| AMIP SST + sea ice | `.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10` (`tos`, `tosbcs`, `siconc`) | 1°, 1870–2022 |
| Ozone (CMIP7) | `.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/FZJ/FZJ-CMIP-ozone-1-0` (`vmro3`) | 1.9×2.5°, 66 plev to ~1e-4 hPa |
| ERA5 land climatology | `/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc` (`stl1`, `swvl1`, `sd`, `skt`, `fal`) | 0.25° monthly means |
| Dust erodibility | `/glade/campaign/cesm/cesmdata/inputdata/atm/cam/dst/dst_0.23x0.31_c130710.nc` | 0.23×0.31° |
| DMS seawater (Lana 2011) | `.../inputdata/atm/cam/chem/ocnexch/Csw_DMS_Lana2011_f09f09_1750_2100_20200717a.nc` | 0.9×1.25° |
| Oxidants OH/HO2/NO3/H2O2/O3, full lid (BUNDLED) | `/glade/p/cesmdata/cseg/inputdata/atm/cam/ozone/oxid_ozone_WACCM_CCMI_REFC1_f.e11.FWTREFC1.<decade>.f19_f19.ccmi34.001_monthly.nc` (decades 1850s–2000s) | 1.9×2.5°, L66 to ~6e-6 hPa |
| Oxidants, year-specific fallback | `.../inputdata/atm/cam/chem/trop_mozart_aero/oxid/oxid_1.9x2.5_L26_1850-2015_c20181106.nc` (`--oxid-source cam`) | 1.9×2.5°, L26 |
| GMTED2010 mean 30″ (SSO statistics) | downloaded once to `$SCRATCH/hf_mirror/sources/gmted` from USGS EROS | 30 arc-seconds |
| CESM ne30 topography | `.../inputdata/atm/cam/topo/se/ne30np4_gmted2010_modis_bedmachine_nc3000_Laplace0100_noleak_greenlndantarcsgh30fac2.50_20250825.nc` | native ne30np4 |
| CEDS ne30 (MAM4-processed, native) | `/glade/campaign/cesm/cesmdata/cseg/inputdata/atm/cam/chem/emis/cmip7/ne30/CEDS-CMIP-2025-04-18_20251030` | native ne30 |
| ECHAM T127 boundary files | `/glade/derecho/scratch/duncanwp/ECHAM_T127` (`T127GR15_jan_surf.nc` has the full SSO + soil/veg set; `T127L95_jan_spec.nc` A/B table; CMIP5 ozone decades) | T127 (384×192) — SSO cross-check only; GMTED remains the primary source |
Surveyed and ruled out first: `/glade/campaign/acom/acom-climate/cesm2/inputdata/atm/waccm/`
— the SC-WACCM forcing files carry O3/O/O2/H/NO/CO2 + heating but no
OH/NO3/H2O2, and `sulf/` is stratospheric sulfate SAD. The WACCM CCMI
REFC1 decade files above (under `atm/cam/ozone/`) are the real full-lid
oxidant product.

Emissions coverage: 1850–2023 monthly, plus PI (1850–1859) and PD
(2005–2014) 12-month climatologies.

Yearly transient AMIP bundles (`--stage amip`, issue #610) additionally
read `tosbcs`/`siconcbcs` from the PCMDI-AMIP-1-1-10 tree above
(mid-month boundary values, 1870–2022), the FZJ transient `vmro3`
chunks (1850–2022), and CR-CMIP-1-0-0 global-annual-mean GHGs:
`.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/CR/CR-CMIP-1-0-0/atmos/yr/
{co2,ch4,n2o}/gm/v20250228/` (1750–2022, ppm/ppb).

Yearly transient ERA5 bundles (`--stage era5-transient`, issue #629)
read the ERA5 6-hourly surface analyses
`/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc` (`sstk`,
`ci`; 1940–present, also the land-field fallback for years outside the
monthly-mean product), the d633001 monthly means above (1979–2022), and
the CR-CMIP GHGs (trend-extrapolated past 2022, stamped in file attrs).
ERA5 is Copernicus-licensed; derived redistributions carry attribution
in the file attributes.
