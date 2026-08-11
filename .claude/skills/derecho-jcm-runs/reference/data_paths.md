# Boundary conditions, emissions and aux forcing on Derecho

## 1. The HF data mirror — canonical source

The dataset `climate-analytics-lab/jax-gcm-data` carries per-grid bundles
for every supported grid, built by `jcm/data/mirror/` (see
`docs/source/design/data_mirror.md`). `mkjob.py` uses it by default
(`--data mirror --era pd|pi`): bundle paths are derived from `--grid`,
prefetched on the login node at generation time, and the local cache
paths are baked into the job script — compute nodes need no internet.

Per Gaussian grid `<g>` (t63, t106):

    bundles/<g>/terrain.nc                 GMTED2010 SSO + fractional ERA5 land mask
    bundles/<g>/forcing_{pd,pi}.nc         PCMDI-AMIP SST/ice + ERA5 land climatology
    bundles/<g>/emissions_{pd,pi}.nc       CEDS+BB4CMIP7, 4 super-sectors x 3 species
    bundles/<g>/dms.nc, dust.nc            Lana 2011; 0.23x0.31 deg erodibility
    bundles/<g>_<l>/ozone_{pd,pi}.nc       FZJ CMIP7, pre-interpolated per level count
    bundles/<g>_<l>/oxidants_{pd,pi}.nc    WACCM CCMI full-lid, per level count

Ozone and oxidants are **level-resolved from level-independent sources**
(FZJ vmro3 on 66 plev; WACCM L66): the `<g>_l47` and `<g>_l95` dirs both
exist for every grid — there is no "L95-only" special case. Any config
path also accepts `hf://bundles/...` directly (cache-first resolver;
prefetch on a login node). `bundles/ne30pg3/sso.nc` serves the pySES
native-terrain path. `registry.json` has sha256 for every file.

Era semantics: `pd` = 2005–2014 climatologies; `pi` = 1850s emissions/
ozone/oxidants with 1870–1879 SST/ice (earliest observed decade).

## 2. Packaged in the repo — offline fallback

| file | contents |
|---|---|
| `jcm/data/bc/t63/terrain.nc` | ECHAM T63GR15 orography + SSO stats (NB `orosig`≈0 — defective; mirror supersedes) |
| `jcm/data/bc/t63/forcing.nc` | AMIP monthly climatology (NB systematically dry soil / low snow vs the mirror — see release notes) |
| `jcm/data/bc/t63/ozone.nc` | CAM6chem 2005–2014 on T63L47 (`forcing.ozone_file: auto` resolves it by grid shape) |
| `jcm/data/bc/t30/…` | same set at T30 |

## 3. Legacy prepared files (`--data local`)

Pre-mirror scratch files, kept for reproducing earlier campaigns. Purge-
eligible; regenerate with `runs/prep_emissions.py` /
`tools/prep_jam_aux_inputs.py` (see script headers). `JAM_INPUTS` /
`JCM_EMISSIONS` point at them. The old portable tarball
`~/jam_input_bundle_20260720.tar.gz` predates the mirror — prefer the
mirror for provisioning new machines.

## 4. Upstream sources (read-only, for rebuilding the mirror)

`jcm/data/mirror/SOURCES.md` is the authoritative inventory (CEDS/BB
input4MIPs, PCMDI-AMIP, FZJ ozone, ERA5 RDA, GMTED2010, WACCM CCMI
oxidants, 0.23×0.31° dust). `python -m jcm.data.mirror.build_mirror`
rebuilds everything, with fail-fast guards for wrong-machine / missing
sources.

## 5. Preparation gotchas (legacy path)

- Regrid to the **model's Gaussian latitudes**, not a uniform lat/lon
  grid (a sin-latitude mesh silently produces a ±57° pseudo-grid).
- Oxidant files need `hyam` in **Pa with no `p0` variable** — the
  runner's validator treats a present `p0` as "normalized hyam" and
  double-scales the pressures.
- Ozone must be pre-interpolated to the model's levels
  (`jcm.data.bc.interpolate_ozone`); the online loader does not regrid
  vertically. Level-resolved files must match the run's layer count.
- Chained `&&` prep commands that short-circuit leave stale outputs
  behind; regenerate into a clean directory when a step fails.
