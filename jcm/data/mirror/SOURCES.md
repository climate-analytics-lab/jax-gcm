# Mirror source data (NCAR Glade paths unless noted)

Destination: Hugging Face dataset `climate-analytics-lab/jax-gcm-data`.

Regridding always starts from the highest-resolution product available.

The builder resolves every root through `jcm/data/mirror/sites.py`. On DKRZ
Levante the input4MIPs products below live under
`/pool/data/INPUT4MIP/data/input4MIPs/CMIP7/CMIP/...` (same tree below `CMIP7`),
the HAMMOZ and ECHAM6 input pools under `/pool/data/ECHAM6-HAMMOZ` and
`/pool/data/ECHAM6`; there is no RDA ERA5 archive, so Tier A is pulled from the
published mirror (`--stage pull`) instead of rebuilt. Roots can be overridden
per build: `JCM_HAMMOZ_DIR` (HAMMOZ pool layout), `JCM_CESM_INPUTDATA` (a CESM
inputdata tree; the WACCM oxidants then resolve under its `atm/cam/ozone`) and
`JCM_WACCM_OXIDANTS_DIR` (the oxidant directory alone).

| product | path | resolution |
|---|---|---|
| CEDS anthropogenic (CMIP7 2025-04-18) | `/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18` | 0.5°, 8 sectors, monthly |
| BB4CMIP7 biomass (DRES 2-0) | `.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0` | native (~0.25°) |
| AMIP SST + sea ice | `.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10` (`tos`, `tosbcs`, `siconc`) | 1°, 1870–2022 |
| Ozone (CMIP7) | `.../input4MIPs_raw/input4MIPs/CMIP7/CMIP/FZJ/FZJ-CMIP-ozone-1-0` (`vmro3`) | 1.9×2.5°, 66 plev to ~1e-4 hPa |
| ERA5 land climatology | `/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc` (`stl1`, `swvl1`, `swvl2`, `swvl3`, `sd`, `skt`, `fal`) plus the invariants `/glade/campaign/collections/rda/data/d633000/e5.oper.invariant/197901` (`lsm`, `cvl`, `cvh`, `slt` — `slt` is the soil-type code that selects each cell's HTESSEL field capacity for `soilw_rel`, #787; `cvh` is also the `forest` fraction of the ECHAM land albedo, #672) | 0.25° monthly means + invariant |
| Dust inputs, T63 (HAMMOZ/Tegen) | Levante `/pool/data/ECHAM6-HAMMOZ/v0007/hammoz/T63/`: `dust_potential_sources_T63.nc` (`pot_source`, 12 monthly records), `dust_preferential_sources_T63.nc` (`source`, paleolakes), `soil_type_all_T63.nc` (`type2/3/4/6` global Zobler + `type13..17` East-Asian), `dust_regions_T63.nc` (`regions`, integers 1-8), `surface_rough_12m_T63.nc` (`surfrough`, **centimetres**, 12 monthly records) | T63 Gaussian |
| Dust inputs, T127 | `/pool/data/ECHAM6-HAMMOZ/v0003/hammoz/T127/` — the same five files `*_T127.nc` (same 2016 processing; identical copies in `v0002`) | T127 Gaussian |
| Dust inputs, T255 | `/pool/data/ECHAM6-HAMMOZ/v01_001/hammoz/T255/` — older names for the same products: `ndvi_lai_eff.12m.T255.nc` (`laieff` = `pot_source`), `pot_sources.T255.nc` (`source` = preferential sources), `soil_type{2,3,4,6,13..17}.T255.nc` (`type`). No roughness or region file | T255 Gaussian |
| DMS seawater (Lana 2011) | `.../inputdata/atm/cam/chem/ocnexch/Csw_DMS_Lana2011_f09f09_1750_2100_20200717a.nc` (also public: `https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/`, same path; reproduces the Glade-built t63 `dms.nc` exactly) | 0.9×1.25° |
| Oxidants OH/HO2/NO3/H2O2/O3, full lid (BUNDLED) | `/glade/p/cesmdata/cseg/inputdata/atm/cam/ozone/oxid_ozone_WACCM_CCMI_REFC1_f.e11.FWTREFC1.<decade>.f19_f19.ccmi34.001_monthly.nc` (decades 1850s–2000s; the builder uses 1850-1859 and 2000-2009). **Not on the public CESM server** — for a Levante build copy those two into `$JCM_MIRROR_ROOT/sources/cesm_inputdata/atm/cam/ozone/`. The public `oxid_ozone_WACCM_CCMI_{1850,REFC1_2000}_cycle_3D_monthly_c20160531.nc` are a different run (ccmi30, 1995-2004 mean) and must not be substituted | 1.9×2.5°, L66 to ~6e-6 hPa |
| Oxidants, year-specific fallback | `.../inputdata/atm/cam/chem/trop_mozart_aero/oxid/oxid_1.9x2.5_L26_1850-2015_c20181106.nc` (`--oxid-source cam`) | 1.9×2.5°, L26 |
| GMTED2010 mean 30″ (SSO statistics) | downloaded once to `$JCM_MIRROR_ROOT/sources/gmted` from USGS EROS (`https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Grid_ZipFiles/mn30_grd.zip`, unzipped to `mn30_grd/`) | 30 arc-seconds |
| CESM ne30 topography | `.../inputdata/atm/cam/topo/se/ne30np4_gmted2010_modis_bedmachine_nc3000_Laplace0100_noleak_greenlndantarcsgh30fac2.50_20250825.nc` | native ne30np4 |
| CEDS ne30 (MAM4-processed, native) | `/glade/campaign/cesm/cesmdata/cseg/inputdata/atm/cam/chem/emis/cmip7/ne30/CEDS-CMIP-2025-04-18_20251030` | native ne30 |
| ECHAM T127/T255 boundary files | `/glade/derecho/scratch/duncanwp/ECHAM_T127`; on Levante `/pool/data/ECHAM6/T127` and `/pool/data/ECHAM6/T255` (`T127GR15_jan_surf.nc` / `T255_jan_surf.nc` carry the full SSO + soil/veg set; `T127L95_jan_spec.nc` A/B table; CMIP5 ozone decades) | T127 (384×192), T255 (768×384) — SSO cross-check only; GMTED remains the primary source on every grid |
| MACv2-SP simple plumes (SPv2.1, CMIP7) — **repo-packaged, NOT mirrored** | Zenodo https://zenodo.org/records/15283189 (`SPv2.1_18502023_CMIP7.nc`; Fiedler & Azoulay, Univ. Heidelberg 2025 — CEDS-scaled 1850–2023 successor to Stevens et al. 2017 v1). sha256 `1070a0c47b9a4a7417c112fc2e395e89f9306454225dde9a62565f76327810dd` | grid/level-free single file (~19 KB: plume geometry + `year_weight`/`ann_cycle`). Resolution-invariant, so it ships in the wheel at `jcm/data/bc/` (see `jcm.forcing.packaged_macv2_path`) rather than on the HF mirror |
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

The five dust files (`--stage dust`, issue #802) are the HAMMOZ input set for
the MPI-BGC/Tegen emission scheme from the ECHAM-HAMMOZ input pool, supplied by
the maintainer and confirmed to carry no licence restriction for this use (unlike
the `mo_ham_dust.f90` source they drive, which is reference-only and never enters
this repository); the T127/T255 files are the same products under the same
clearance. Override the pool root with `JCM_HAMMOZ_DIR` (a directory laid out
like `/pool/data/ECHAM6-HAMMOZ`). T63, T127 and T255 are native and copied
through with a latitude flip. Every other grid (T106) is remapped from the
finest native file with the exact-overlap first-order conservative scheme
(`jcm.data.regridding.conservative_overlap`, CDO `remapcon`'s scheme), and the
T255 roughness map — the one product with no T255 file — is refined from T127
with the same scheme and stamped as an approximation in its attributes. The
region mask is regenerated on every grid from the `cdo setclonlatbox` recipe in
the HAMMOZ file's `history`, and checked cell-for-cell against every native mask
the builder can see. The same sets also ship `soilpHfrac_*.nc` and
`xtsurf_v2_*.nc`, which the emission scheme never reads and which are therefore
not mirrored. They replace the CAM geomorphic erodibility map
(`dst_0.23x0.31_c130710.nc`) that used to be published as
`bundles/<grid>/dust.nc`; that product and that path are retired.
