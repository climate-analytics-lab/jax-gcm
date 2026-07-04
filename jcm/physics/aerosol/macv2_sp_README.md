# MACv2-SP (Simple Plumes) Aerosol Scheme

JAX implementation of the MACv2-SP simple-plumes aerosol parameterization
(Stevens et al., 2017, *GMD* 10, 433-452; authoritative reference
`mo_simple_plumes_v1.f90` from the paper's supplement). Nine anthropogenic
Gaussian plumes plus a natural background provide aerosol optical properties
and a Twomey effect without prognostic aerosol tracers.

## Entry points

- **Composable term:** `Macv2SpAerosol` (a `PhysicsTerm` in `macv2_sp.py`),
  wired into the ECHAM package by the `echam_physics()` factory in
  `jcm/physics/echam/echam_terms.py`. Requires the `height_full` and
  `layer_thickness` diagnostics and reads `terrain.orog` for the plume
  orography truncation.
- **Functional core:** `get_simple_aerosol(height_full, layer_thickness,
  orography, lats_deg, lons_deg, aerosol_data, parameters, forcing,
  sw_band_centers_nm)` returns an updated `AerosolData`. It follows the
  reference `sp_aop_profile` structure:
  - `_per_feature_plume_gaussians` — rotated anisotropic Gaussians per
    feature and plume (the 260° longitudinal-wrap case belongs to plume 1,
    Europe)
  - `get_plume_column_weights` — per-plume 550 nm column weights `cw_an` /
    `cw_bg`, with each feature's own time weight
    (`forcing.aerosol_year_weight × aerosol_ann_cycle`) applied to that
    feature's own Gaussian *before* the feature sum
  - `get_vertical_profiles` — dz-weighted beta-function profiles,
    normalized then truncated at the orography (mass below ground is
    removed, not redistributed)
  - per-band optics accumulated per plume weighted by anthropogenic AOD,
    with the reference zero-AOD limits (ssa → 1, asy → 0)
  - `get_dNovrN` — Stevens (2017) Twomey factor from the column plume AOD
    against the natural background
- Parameters live in `macv2_sp_params.py`. **`AerosolParameters.default()`
  is the real `MACv2.0-SP_v1.nc` static geometry** (verbatim transcription,
  verified against the file); `AerosolParameters.from_dataset(ds)` loads a
  (modified) parameter file, owning the `(plume, feature)` transposes.
- Real time-varying `year_weight` / `ann_cycle` forcing: see
  `notebooks/06_macv2_aerosols.py` (BY_DATE / WRAP_YEAR TimeSeries; the v1
  file's `year_weight` is `_FillValue`-masked beyond 2016 and must be
  forward-filled). The `ForcingData` all-ones defaults mean *perpetual
  year-2005 amplitude with no seasonal cycle*.

## Outputs consumed downstream

- Per-SW-band optics (`aod_sw_per_band`, `ssa_sw_per_band`, ...) → RRTMGP
  aerosol radiative effects (`jcm/physics/radiation/rrtmgp.py`).
- `cdnc_factor` (dNovrN) → Twomey scaling of droplet number in the 1M
  microphysics (`jcm/physics/clouds/echam_1m.py`).
- `Nccn` → SPA activation floor for the 2M scheme (`jcm/physics/aerosol/spa.py`).

## Tests

- `jcm/physics/aerosol/macv2_sp_test.py`
- `jcm/physics/aerosol/per_band_optics_test.py`

## Documented deviations from the reference

- The natural background is the reference's column scalar
  (`0.02 + fine-mode plume background`) feeding only `dNovrN`; it never
  enters the radiative AOD/SSA/ASY profiles. `aod_total` is therefore the
  **anthropogenic** column AOD (`caod_sp`).
- `Nccn`/`get_CDNC` (absolute CCN for the SPA activation path) is a jcm
  extension using the AEROCOM-P1 fit — the reference only provides the
  relative factor `dNovrN`.
- No wavelength cutoff is applied beyond the reference's own
  `min(1, 700/λ)` and Ångström scalings — neither the v1 supplement nor
  the ECHAM-HAM host wrapper has one.
