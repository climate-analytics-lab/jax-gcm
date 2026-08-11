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
Supported grids: `t63`, `t106` (Gaussian) and `ne30pg3` (native columns,
`terrain.nc` only — the pySES path interpolates the Gaussian forcing
files and uses the native CESM CEDS emissions product). The ne30pg3
`terrain.nc` is fully assembled: GMTED2010 SSO statistics, land fraction
from the CESM topo `LANDFRAC` (SSO zeroed below 10% land), and exact
GLL-node orography (`orog_gll` = `PHIS_gll`/g).

## Fetching at runtime

Any boundary-file path in the Hydra config accepts an `hf://` prefix,
resolved through the local HF cache by `jcm.runners._resolve_data_path`:

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
