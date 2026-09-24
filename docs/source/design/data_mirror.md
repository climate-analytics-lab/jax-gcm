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
`emissions_{pi,pd}.nc`, `dms.nc`, the five `dust_*.nc` Tegen inputs, and per level count
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
Supported grids: `t63`, `t106`, `t127`, `t255` (Gaussian) and `ne30pg3`
(native columns, `terrain.nc` only — the pySES path interpolates the Gaussian
forcing files and uses the native CESM CEDS emissions product). `t127` and
`t255` are ECHAM's own T127/T255 grids (384×192 and 768×384) and are
**supported, not validated**: every climatological and static bundle above
exists for them, so `grid=echam_t{127,255}_l{47,95}_hybrid` resolves all of its
inputs, but they are not release-matrix members and nothing is tuned for them
(see {doc}`../science/configurations`). The yearly transient series
(`forcing_amip`, `emissions_amip`, `ozone_amip`, `forcing_era5`) are published
for `t63` and `t106` only — `TRANSIENT_GRIDS` in `build_mirror.py`, tracked
for the new grids in #888. The ne30pg3
`terrain.nc` is fully assembled: GMTED2010 SSO statistics, land fraction
from the CESM topo `LANDFRAC` (SSO zeroed below 10% land), and exact
GLL-node orography (`orog_gll` = `PHIS_gll`/g).

**MACv2-SP simple plumes are NOT on the mirror.** The Simple-Plumes
parameter file (SPv2.1, CMIP7; Fiedler & Azoulay 2025 — the CEDS-scaled
1850–2023 successor to Stevens et al. 2017 v1) is a single ~19 KB,
resolution-invariant file (plume geometry + `year_weight`/`ann_cycle`
scalings), so it ships **in the wheel** at
`jcm/data/bc/SPv2.1_18502023_CMIP7.nc` rather than being mirrored.
`forcing=macv2_sp` (i.e. `forcing.macv2_file=auto`) and
`ForcingData.from_bundles(aerosol="macv2sp")` both resolve it through
`jcm.forcing.packaged_macv2_path`; an explicit `forcing.macv2_file=/path`
overrides. Provenance + sha256 are in `jcm/data/mirror/SOURCES.md`
(source Zenodo <https://zenodo.org/records/15283189>).

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
explicit path is taken verbatim; a `{year}` pattern is expanded per year for
`emissions_file` / `oxidants_file` (and the surface `file` / `ozone_file`), but
`dms_file` / `dust_file` are climatology-only single files — no transient
product is mirrored — and reject a `{year}` at build time.

The equivalent explicit form (any `*_file` still accepts an `hf://` path, and a
real-world SST/land file needs `terrain=from_file`/`terrain=auto` to match its
land-sea mask):

```bash
python -m jcm.main physics=echam-jam grid=echam_t63_l47_hybrid \
    terrain=from_file terrain.file=hf://bundles/t63/terrain.nc \
    forcing=from_file forcing.file=hf://bundles/t63/forcing_pd.nc \
    forcing.emissions_file=hf://bundles/t63/emissions_pd.nc \
    forcing.dms_file=hf://bundles/t63/dms.nc \
    forcing.dust_file=hf://bundles/t63/dust_potential_sources.nc \
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

## Hosted initial states

`bundles/<grid>_<levels>/init_states/` holds model states rather than
boundary conditions: `jcm.checkpoint.save_checkpoint` msgpack files a run can
warm-start from with `init=from_state init.file=hf://...`. They are hosted
rather than committed because they run from a few MB to several GB and are
regenerated whenever the physics they describe moves.

Two kinds live there today:

- **Equilibrated states** from the #638 validation campaign
  (`echam_{1m,2m}_macsp_year2.msgpack`, `speedy_year1.msgpack`). These are
  **unreadable by current jcm** and are kept only as provenance: they predate
  the checkpoint schema stamp, so they carry no field names, *and* they are
  structurally stale — an ECHAM T63L47 donor stores 118 physics-carry arrays
  where the current model expects 146 (51 vs 56 for SPEEDY). `load_checkpoint`
  refuses a structural difference it cannot name rather than guessing, which
  is the correct behaviour and not a bug to work around. Replacing them is
  issue #762.
- **Regression-fixture states**, `<member>_fixture_<digest>.msgpack`, one per
  supported-matrix member, written by
  `jcm.data.test.release_matrix.generate_stats` and consumed by the
  GPU-gated regression in `jcm/model_test.py`. Their *bands* stay in the
  repo (tens to a few hundred KB, so a change is reviewable as a diff);
  only the state is hosted. A member's band file and its state are a matched pair — the bands
  describe the window that follows that exact state — so they are regenerated
  together, one command per member:

  ```bash
  CUDA_VISIBLE_DEVICES=<idx> python -c "import os; os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'; from jcm.data.test.release_matrix.generate_stats import generate; generate('echam-1m-t63', out_dir='/scr/$USER/fixtures')"
  ```

  in a CI-parity environment (a fresh venv with `pip install -e ".[mam4]"`
  and the pinned CUDA jax — never a shared or long-lived one: bands drawn
  under a different jax-rrtmgp release fail a correct model across the whole
  column). The preallocation setting must precede the jcm import, because
  importing jcm initialises the CUDA backend (#859) and the default would
  hand 75 % of the card to the orchestrator; `generate` refuses to run
  without it. See `tools/release_validation/README.md` for re-deriving bands
  on an already-published state.

  and the resulting `<member>_fixture_<digest>.msgpack` is uploaded
  additively under the member's `init_states/` prefix. The digest is in the
  name because `fetch` resolves cache-first and never revalidates a hit: a
  stable name could not be republished without leaving every already-warm
  cache pairing an old state with new bands. The band file records the exact
  path it was generated against.

`jcm.data.remote.fetch` resolves these cache-first like any other mirror file,
so a warm cache needs no network and a cold cache on an internet-less node
fails with the prefetch instructions rather than a bare error.

## Rebuilding the mirror

The builders live in `jcm/data/mirror/` (`jcm/data/mirror/SOURCES.md` is the
verified path inventory). They run on either of two sites, whose source roots
are declared once in `jcm/data/mirror/sites.py` (auto-detected, or
`JCM_MIRROR_SITE`):

- **NCAR Glade** holds every source except the ECHAM-HAMMOZ pool, so it can
  rebuild Tier A and every bundle except the dust inputs.
- **DKRZ Levante** holds the CMIP7 input4MIPs tree and the ECHAM-HAMMOZ and
  ECHAM6 pools under `/pool/data`, but not the RDA ERA5 archive. It therefore
  does not rebuild Tier A: `--stage pull` fetches the published Tier A products
  (only the PI/PD climatology arrays of the emissions stores) and
  `registry.json`, so a new grid regrids from exactly the data the published
  grids were built from. The Lana DMS file and GMTED are downloaded once into
  `$JCM_MIRROR_ROOT/sources/`; the two WACCM CCMI REFC1 decade oxidant files
  are not on the public CESM inputdata server and are copied from Glade (the
  public `oxid_ozone_WACCM_CCMI_*_cycle` files are a different run, ccmi30
  1995–2004, and are not substitutes).

`--grids` restricts every stage to a subset of the published grids, which is how
a grid is added without rebuilding or re-uploading the others. The registry
stage then merges the new hashes onto the pulled `registry.json` rather than
rewriting it from the partial upload tree. The t127/t255 bundles were built on
Levante with

```bash
python -m jcm.data.mirror.build_mirror --grids t127,t255 \
    --stage pull,sso,ozone,aux,dust,bundles
python -m jcm.data.mirror.build_mirror --grids t106 --stage dust
python -m jcm.data.mirror.build_mirror --grids t106,t127,t255 --stage registry
```

and the port was checked by rebuilding the t63 SSO, ozone, oxidant, DMS,
terrain, forcing and emissions bundles the same way and comparing them with the
published Glade-built files (identical up to ~3e-8 relative, float round-off).
The t63/t106 emission bundles were then rebuilt with the exact conservative
remap — `--stage emissions` from the Levante input4MIPs tree, then
`--grids t63,t106 --products emissions --stage bundles,amip`; `--products`
limits those stages to the named bundle products so unchanged files are not
republished.

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
  exact-overlap first-order conservative remapping for emission fluxes
  (`jcm.data.regridding.conservative_to_gaussian`; area means conserved to
  round-off on every grid), nearest-ocean fill for AMIP SST under land.
  Nearest-centre binning, used before, left whole latitude rows of T255
  empty (its 0.47° cells are finer than the 0.5° CEDS source) and carried
  1-3 % global-mean errors on t63/t106; those emission bundles were rebuilt
  with the exact scheme.
- `dust.py` — the five Tegen inputs from the ECHAM-HAMMOZ pool: native at
  T63, T127 and T255 (the T255 files are the older `v01_001` lineage, verified
  to be the same products where both lineages exist), exact-overlap
  conservative from the finest native file elsewhere (T106 from T255;
  T255 roughness is refined from T127, the one product HAMMOZ never shipped at
  T255, and says so in its attributes), and the region mask regenerated on every
  grid from the `setclonlatbox` recipe in the HAMMOZ file history, which
  reproduces the native T63/T127 masks cell for cell.
- `registry.py` — hashes the upload tree (merged onto the published registry
  for a `--grids` build).
- `build_mirror.py --stage upload` — pushes to the HF dataset with
  retries (the xet backend has aborted 44k-file pushes with transient
  timeouts; uploads resume, committed files are skipped). Deliberately
  excluded from `--stage all` — publishing is explicit. Needs
  `hf auth login` with write access; run `python -m` from the repo
  checkout's own directory.

## Known caveats

- **T127/T255 terrain is GMTED-derived like every other grid.** ECHAM's own
  `T127GR15_jan_surf.nc` / `T255_jan_surf.nc` (in `/pool/data/ECHAM6`) carry
  a full SSO set too; they are used only as a cross-check of the GMTED
  statistics, so that every published grid derives its orography the same way.
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
- `soilw_rel` is a **second, independent** soil-moisture channel in the
  same forcing bundle, not a refinement of `soilw_am`: ECHAM's relative
  soil wetness `ws/wsmx = min(1, swvl1/θ_cap(slt))` — the ERA5 0–7 cm
  volumetric content over the HTESSEL field capacity of that cell's own
  soil type (Balsamo et al. 2009). That is the layer and the
  normalisation the Tegen dust saturation cut-off is defined against, so
  `DustEmissions` reads it and SPEEDY's land evaporation keeps
  `soilw_am`. It needs one extra ERA5 invariant, the soil-type code
  `slt` (`128_043_slt`), so a build tree whose Tier A `era5` product
  predates the channel must re-run `--stage era5` before `--stage
  bundles`. Forcing files without the channel still load — the dust term
  warns and falls back.
- The packaged T63 `orosig` was ≈0 everywhere; the GMTED-derived bundles
  supply a real mean-slope field, so SSO gravity-wave drag will behave
  differently (more drag) than with the packaged terrain. The gradient
  tensor is computed on 10′ block-mean topography (Lott & Miller's
  source resolution) — calibrated against the ECHAM T127 reference,
  where 30″ gradients give ~5× the reference σ and 10′ gives 1.16×
  with the best structural agreement (r = 0.84).
