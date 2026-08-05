# Boundary conditions, emissions and aux forcing on Derecho

Three tiers: **packaged** (in the repo, always available), **prepared**
(derived files that must exist before a JAM run), and **upstream sources**
(the CESM `inputdata` tree the prepared files come from).

Set `JAM_INPUTS` to wherever the prepared files live; `mkjob.py` reads it
(default `/glade/derecho/scratch/$USER/jam_inputs`).

## 1. Packaged in the repo — nothing to prepare

| file | contents |
|---|---|
| `jcm/data/bc/t63/terrain.nc` | ECHAM T63GR15 orography, land-sea mask, subgrid-orography stats (`orostd/orosig/orogam/orothe/oropic/oroval`) |
| `jcm/data/bc/t63/forcing.nc` | AMIP monthly SST, sea ice, land T, soil moisture, snow, albedo |
| `jcm/data/bc/t63/ozone.nc` | CAM6chem 2005–2014 monthly ozone climatology on T63L47 |
| `jcm/data/bc/t30/…` | same set at T30 |

`forcing.ozone_file: auto` resolves the packaged ozone by grid shape — leave
it alone unless you need a different product.

## 2. Prepared files — required for online JAM

**These live on scratch and are purge-eligible.** Check they exist before
submitting; regenerate with the commands below if not.

| file | used for | regenerate with |
|---|---|---|
| `emissions_echam_t63_l47_hybrid_2014.nc` | anthropogenic + biomass emissions | `runs/prep_emissions.py` |
| `dms_lana2011_climo_t63.nc` | interactive DMS | `tools/prep_jam_aux_inputs.py --target-truncation 63` |
| `dust_erodibility_cam_f19_t63.nc` | interactive dust | idem |
| `oxidants_cam_echam_l47_2014_t63.nc` | sulfur chemistry | idem |
| `ozone_cam6chem_2005-2014_t63_l95.nc` | L95 runs only | `jcm.data.bc.interpolate_ozone --nlevels 95` |

### L95 ozone — already prepared, NOT packaged

Only **T63L47** ozone ships in the repo (`jcm/data/bc/t63/ozone.nc`), so
`forcing.ozone_file: auto` cannot serve an L95 or T106/T119 run and silently
falls back to the analytic profile. `mkjob.py` refuses those grids without an
explicit `--ozone` for that reason.

The files exist, generated 2026-08-01 for the MA L95 benchmark and verified
NaN-free with latitudes already S->N:

| grid | file | shape |
|---|---|---|
| T63 L95 | `t63_ozone_l95.nc` | 12 x 95 x 96 x 192 |
| T106 L95 | `t106_ozone_l95.nc` | 12 x 95 x 160 x 320 |
| T119 L95 | `t119_ozone_l95.nc` | 12 x 95 x 180 x 360 |

Currently under `/scr/dwatsonparris/bc_l95/` on the dev workstation — **which
is purge-eligible**; copy them somewhere durable before relying on them, and
re-verify before use (`np.isnan(ds.O3).sum() == 0`).

They are deliberately **not** committed: the interpolated field is
near-maximum entropy (17.7M unique values of 21M), so it does not compress —
84 MB raw becomes 48 MB with zlib-9, against 0.99 MB for the packaged L47
file, and T106/T119 are several times larger again. Regenerating is cheap and
scripted; storing them in git is not.

**The polar trap** (worth knowing if regenerating): T106/T119 Gaussian
latitudes fall *outside* the T63 source range, and `xarray.interp` with
`fill_value=None` returns NaN rather than extrapolating — which silently
NaN'd 730k of 820k ozone points and killed every T106/T119 run until the
source was extended to +/-90 by nearest-neighbour first.

Current locations on this machine:

```bash
JAM_INPUTS=/glade/derecho/scratch/$USER/jam_inputs      # dms/dust/oxidants/ozone
EMISSIONS=$HOME/jax-gcm/runs/emissions_echam_t63_l47_hybrid_2014.nc
```

A T42 emissions file (`emissions_echam_t42_l8_sigma_2014.nc`) sits beside it
for cheap tests.

**Grid variants matter.** The spectral (`jcm.main`) path validates grids
strictly and needs the `_t63` files — exact Gaussian latitudes. The pySES
column path interpolates from any regular lon/lat grid, so it can use the
native-grid variants. Level-resolved files (oxidants, ozone) must match the
run's layer count: an L47 oxidant file will not work in an L95 run.

**Portable copy:** `~/jam_input_bundle_20260720.tar.gz` (164 MB, md5
`4e15803b1bed6e01e67e708cef11e47a`) holds all eight files plus a README —
scp it to provision another machine rather than re-preparing.

## 3. Upstream sources (CESM `inputdata`, read-only)

```
/glade/campaign/cesm/cesmdata/inputdata
  atm/cam/chem/ocnexch/Csw_DMS_Lana2011_f09f09_1750_2100_20200717a.nc   # DMS, nmol/L
  atm/cam/dst/dst_1.9x2.5_c090203.nc                                    # dust: mbl_bsn_fct_geo
  atm/cam/chem/trop_mozart_aero/oxid/oxid_1.9x2.5_L26_1850-2015_c20181106.nc
  atm/cam/ozone/ozone_strataero_CAM6chem_1849-2014_zm_5day_c170924.nc   # ozone (zonal mean)
/glade/campaign/cesm/cesmdata/cseg/inputdata/atm/cam/chem/emis/cmip7/ne30  # CEDS + BB4CMIP7
```

Note the emissions live under `cesmdata/cseg/inputdata`, the rest under
`cesmdata/inputdata`.

## 4. Preparation gotchas

- Regrid to the **model's Gaussian latitudes**, not a uniform lat/lon grid.
  `prep_jam_aux_inputs.py` uses the model latitude helper for this; a
  sin-latitude mesh silently produces a ±57° pseudo-grid.
- Oxidant files need `hyam` in **Pa from the raw ECHAM table with no `p0`
  variable** — the runner's validator treats a present `p0` as "normalized
  hyam" and double-scales the pressures.
- Ozone must be pre-interpolated to the model's levels
  (`jcm.data.bc.interpolate_ozone`); the online loader does not regrid
  vertically.
- Chained `&&` prep commands that short-circuit leave stale outputs behind;
  regenerate into a clean directory when a step fails.
