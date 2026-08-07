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
(`<grid>_l{47,95}/`) `ozone_{pi,pd}.nc` and `oxidants_{1850,2014}.nc`.
Supported grids: `t63`, `t106` (Gaussian) and `ne30pg3` (native columns,
SSO only — the pySES path interpolates the Gaussian forcing files and
uses the native CESM CEDS emissions product).

## Fetching at runtime

```python
from jcm.data.remote import bundle_file
terrain = bundle_file("t63", "terrain.nc")     # cached HF download
```

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

## Known caveats

- **PI SST/sea-ice is the 1870–1879 AMIP mean** — the earliest observed
  decade; no observational 1850 state exists.
- **Oxidants remain CAM L26** (clamped above the ~3.5 hPa CAM lid). No
  higher-lid OH/NO3/H2O2 product exists on Glade — the WACCM
  `waccm_forcing` files carry O3/O/O2/H/NO but not the sulfur-oxidant
  set.
- `soilw_am` is an evaporation-availability factor derived from ERA5
  volumetric soil water against SPEEDY's field capacity (see
  `bundles.py` docstring); it is a modelling choice, not an observed
  field.
- The packaged T63 `orosig` was ≈0 everywhere; the GMTED-derived bundles
  supply a real mean-slope field, so SSO gravity-wave drag will behave
  differently (more drag) than with the packaged terrain.
