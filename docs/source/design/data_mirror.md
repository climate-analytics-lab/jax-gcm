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
`dycore.terrain_file=hf://bundles/ne30pg3/sso.nc` maps file columns onto
the physics columns one-for-one (unit-sphere nearest neighbor) and takes
GLL-node orography from the file's `orog_gll` (CESM topo `PHIS_gll`),
replacing the old packaged-T63 downscale. The same `hf://` forcing files
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
- `soilw_am` is an evaporation-availability factor derived from ERA5
  volumetric soil water against SPEEDY's field capacity (see
  `bundles.py` docstring); it is a modelling choice, not an observed
  field.
- The packaged T63 `orosig` was ≈0 everywhere; the GMTED-derived bundles
  supply a real mean-slope field, so SSO gravity-wave drag will behave
  differently (more drag) than with the packaged terrain.
