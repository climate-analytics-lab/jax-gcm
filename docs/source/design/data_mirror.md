# Boundary-condition and emissions data mirror

All jcm input data beyond the packaged T30/T63 starter files lives in the
Hugging Face dataset
[`climate-analytics-lab/jax-gcm-data`](https://huggingface.co/datasets/climate-analytics-lab/jax-gcm-data)
(issue #515). The mirror has two tiers:

**Tier A — grid-independent products** (`products/`), kept at each
source's native resolution so any future grid regrids from the highest
resolution available:

| product | resolution | source |
|---|---|---|
| `ceds_anthro.zarr` | 0.5°, monthly 1850–2023 | CEDS-CMIP-2025-04-18, summed over the 8 CEDS sectors |
| `bb4cmip7.zarr` | 0.25°, monthly 1850–2023 | DRES BB4CMIP7-2-0 open burning |
| `era5_land_climo_2005-2014_0p25.nc` | 0.25°, 12-month | ERA5 monthly means (skt, stl1, swvl1-3, sd, fal, lsm) |
| `sso/sso_gmted2010_*.nc` | per grid | GMTED2010 30″ DEM, Lott & Miller (1997) statistics |

Both emissions stores carry PI (1850–1859) and PD (2005–2014) 12-month
climatology arrays alongside the transient series.

**Tier B — per-grid bundles** (`bundles/<grid>/`), the files the model
reads directly: `terrain.nc`, `forcing_{pi,pd}.nc`,
`emissions_{pi,pd}.nc`, `dms.nc`, `dust.nc`, and per level count
(`<grid>_l{47,95}/`) `ozone_{pi,pd}.nc` and `oxidants_{pi,pd}.nc`.

**Yearly transient AMIP bundles** (issue #610) sit alongside the era
climatologies as one file per calendar year — download only the years
you run; a new year appends without rewriting history:
`forcing_amip/<year>.nc` (PCMDI-AMIP `tosbcs`/`siconcbcs` mid-month
SST/ice + repeated ERA5 land climatology + CR-CMIP global-annual-mean
CO2/CH4/N2O in ppmv), `emissions_amip/<year>.nc` (transient monthly
CEDS + BB4CMIP7), and `<grid>_l{47,95}/ozone_amip/<year>.nc` (FZJ
monthly ozone on model levels). Config uses a `{year}` pattern plus an
inclusive range, and mid-month boundary values need linear time
interpolation and an on-calendar start date:

```bash
python -m jcm.main forcing=amip forcing.years=[1979,1983] \
    run.start_date=1979-01-01 grid=echam_t63_l47_hybrid ...
```

Built with `python -m jcm.data.mirror.build_mirror --stage amip
--years 1950,2022` (source coverage 1870–2022; excluded from
`--stage all`).

**Yearly transient ERA5 bundles** (issue #629): `forcing_era5/<year>.nc`
prescribes *every* surface field — SST, sea ice, land temperature, soil
moisture, snow cover — from ERA5 on one land-sea mask, so land carries
real interannual variability and trend instead of the repeated
climatology in `forcing_amip`. Use `forcing=era5` for internally
consistent transient runs (land-aware calibration, AIMIP-style
integrations) and for years past PCMDI's 2022 endpoint; keep
`forcing=amip` where a protocol mandates PCMDI SSTs. SST/ice are
month-start boundary values built from 6-hourly analyses with the AIMIP
centred-window construction (which linear interpolation does *not*
reconstruct into exact monthly means — unlike PCMDI `tosbcs`; the
construction is stamped in the file attrs). Land monthly means are
blended `0.5·(prev+cur)` onto the same month-start axis; the `snowc`
ice-sheet mask and background albedo stay climatological so ice sheets
cannot flicker year-to-year. Built with `--stage era5-transient
--years 1979,2024` (buildable from 1941; land monthly means are reduced
from 6-hourly analyses outside the 1979–2022 pre-computed range; GHGs
are trend-extrapolated past 2022, stamped in the attrs).
Supported grids: `t63`, `t106` (Gaussian) and `ne30pg3` (native columns,
`terrain.nc` only — the pySES path interpolates the Gaussian forcing
files and uses the native CESM CEDS emissions product). The ne30pg3
`terrain.nc` is fully assembled: GMTED2010 SSO statistics, land fraction
from the CESM topo `LANDFRAC` (SSO zeroed below 10% land), and exact
GLL-node orography (`orog_gll` = `PHIS_gll`/g).

## Fetching at runtime

Any boundary-file path in the Hydra config accepts an `hf://` prefix,
resolved through the local HF cache by `jcm.runners._resolve_data_path`.

**The online-aerosol emission inputs resolve themselves.** The four
prescribed-emission keys (`emissions_file`, `dms_file`, `dust_file`,
`oxidants_file`) default to `auto` (issue #640): when a prognostic-aerosol
(JAM) physics package is active, `auto` composes the per-grid present-day
bundle for the model grid at build time. So the documented-canonical run is
just the physics preset and the grid —

```bash
python -m jcm.main physics=echam-jam-aerocom grid=echam_t63_l47_hybrid
```

— which auto-resolves `bundles/t63/{emissions_pd,dms,dust}.nc`,
`bundles/t63_l47/oxidants_pd.nc` and (via `ozone_file: auto`)
`bundles/t63_l47/ozone_pd.nc`, exactly as the explicit nine-line form below
did. Prefetch the bundles on a node with internet first; a cold cache fails
loudly at build time, naming the missing `hf://` path and the fixes (prefetch,
a local path, or `forcing.<key>=null` to opt out). `auto` is the only
grid-portable mechanism — it composes the concrete per-grid bundle path itself,
so one config follows the grid without any user-facing path template. An
explicit path is taken verbatim (a `{year}` pattern is still expanded per year).

The equivalent explicit form (any `*_file` still accepts an `hf://` path, and a
real-world SST/land file needs `terrain=from_file`/`terrain=auto` to match its
land-sea mask):

```bash
python -m jcm.main physics=echam-jam grid=echam_t63_l47_hybrid \
    terrain=from_file terrain.file=hf://bundles/t63/terrain.nc \
    forcing=from_file forcing.file=hf://bundles/t63/forcing_pd.nc \
    forcing.emissions_file=hf://bundles/t63/emissions_pd.nc \
    forcing.dms_file=hf://bundles/t63/dms.nc \
    forcing.dust_file=hf://bundles/t63/dust.nc \
    forcing.oxidants_file=hf://bundles/t63_l47/oxidants_pd.nc \
    forcing.ozone_file=hf://bundles/t63_l47/ozone_pd.nc
```

The pySES backend takes the native bundle directly —
`dycore.terrain_file=hf://bundles/ne30pg3/terrain.nc` samples the file
cells onto the physics columns by unit-sphere nearest neighbor and
takes GLL-node orography from the file's `orog_gll` (CESM topo
`PHIS_gll`), replacing the old packaged-T63 downscale. The bundle keeps
full pg3 resolution (48,600 cells) while pySES physics runs on pg2
columns (21,600), so the dycore's column-count warning is expected and
benign for this pairing. (`build_terrain`
refuses a file whose mean land fraction exceeds 0.9 — that is the
signature of a raw SSO product's DEM-validity placeholder `lsm`, which
would silently produce an all-land planet; see #596.) The same `hf://` forcing files
work there too (the column loader interpolates from any regular lon/lat
grid).

Programmatic access:

```python
from jcm.data.remote import bundle_file
terrain = bundle_file("t63", "terrain.nc")     # cached HF download
```

Fetch once on a node with internet; compute nodes then hit the cache.
`registry.json` at the dataset root records sha256 + size for every file.

## Rebuilding the mirror

The builders live in `jcm/data/mirror/` and run on NCAR Glade, where all
sources are on disk (`jcm/data/mirror/SOURCES.md` is the verified path
inventory):

- `sso.py` — streams the GMTED2010 DEM in latitude strips, accumulating
  Lott–Miller gradient-tensor statistics onto Gaussian bins or, for
  ne30pg3, the Voronoi cells of the column centers (unit-sphere KDTree).
- `era5_land.py` — 12-month means over 2005–2014 of the RDA ERA5 monthly
  products.
- `ozone.py` + `jcm.data.bc.interpolate_ozone` — FZJ CMIP7 `vmro3`
  regridded horizontally then log-p interpolated to the ECHAM hybrid
  levels.
- `emissions.py` — CEDS sector sums and BB4CMIP7 fluxes streamed to zarr.
- `bundles.py` — per-grid assembly: bilinear for smooth fields,
  cos-lat-weighted conservative binning for emissions fluxes,
  nearest-ocean fill for AMIP SST under land.
- `registry.py` — hashes the upload tree.
- `build_mirror.py --stage upload` — pushes to the HF dataset with
  retries (the xet backend has aborted 44k-file pushes with transient
  timeouts; uploads resume, committed files are skipped). Deliberately
  excluded from `--stage all` — publishing is explicit. Needs
  `hf auth login` with write access; run `python -m` from the repo
  checkout's own directory.

## Known caveats

- **PI SST/sea-ice is the 1870–1879 AMIP mean** — the earliest observed
  decade; no observational 1850 state exists.
- **Bundled oxidants come from the WACCM CCMI REFC1 decade
  climatologies** (`oxid_ozone_WACCM_CCMI_REFC1_*.f19_f19` under
  `atm/cam/ozone/`): all species incl. H2O2 on L66 with the full WACCM
  lid, decades 1850–2009 — so L95 mesospheric levels carry real values.
  PI uses the 1850s decade; PD uses 2000–2009 (the newest available).
  The CAM L26 transient remains available via
  `prep_jam_aux_inputs.py --oxid-source cam` when a specific year
  matters more than the lid.
- `soilw_am` is the SPEEDY soil-availability **fraction** in [0, 1],
  computed from ERA5 volumetric layers with the `jcm.data.bc.compile`
  formula (vegetation-gated deep layer, wilting/capacity thresholds);
  `snowc` is likewise the snow-cover fraction `min(1, sd/sd2sc)`. Both
  follow the packaged files' conventions exactly (see the `bundles.py`
  docstring).
- The packaged T63 `orosig` was ≈0 everywhere; the GMTED-derived bundles
  supply a real mean-slope field, so SSO gravity-wave drag will behave
  differently (more drag) than with the packaged terrain. The gradient
  tensor is computed on 10′ block-mean topography (Lott & Miller's
  source resolution) — calibrated against the ECHAM T127 reference,
  where 30″ gradients give ~5× the reference σ and 10′ gives 1.16×
  with the best structural agreement (r = 0.84).
