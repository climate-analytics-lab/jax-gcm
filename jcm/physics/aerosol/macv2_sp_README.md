# MACv2-SP (Simple Plumes) Aerosol Scheme

JAX implementation of the MACv2-SP simple-plumes aerosol parameterization
(Stevens et al., 2017, *GMD* 10, 433-452; ICON reference
`mo_bc_aeropt_splumes.f90`). Nine anthropogenic Gaussian plumes plus a natural
background provide aerosol optical properties and a Twomey effect without
prognostic aerosol tracers.

## Entry points

- **Composable term:** `Macv2SpAerosol` (a `PhysicsTerm` in `macv2_sp.py`),
  wired into the ECHAM package by the `echam_physics()` factory in
  `jcm/physics/echam/echam_terms.py`.
- **Functional core:** `get_simple_aerosol(height_full, lats_deg, lons_deg,
  aerosol_data, parameters, forcing, sw_band_centers_nm)` returns an updated
  `AerosolData`. It composes:
  - `get_plume_spatial_distribution` — Gaussian plume weights per column
  - `get_anthropogenic_aod` / `get_background_aod` — 550 nm column AOD
    (time-varying via `forcing.aerosol_year_weight` / `aerosol_ann_cycle`)
  - `get_vertical_profiles` / `get_background_vertical_profile` — beta-function
    vertical AOD distribution
  - `get_optical_properties` — plume-weighted SSA, asymmetry, Angstrom exponent
  - `per_band_optical_properties` — Angstrom scaling to the active SW bands
  - `get_CDNC` — Twomey-style CCN from column AOD
- Parameters live in `macv2_sp_params.py` (`AerosolParameters`, tree_math struct).

## Outputs consumed downstream

- Per-SW-band optics (`aod_sw_per_band`, `ssa_sw_per_band`, ...) → RRTMGP
  aerosol radiative effects (`jcm/physics/radiation/rrtmgp.py`).
- `cdnc_factor` → Twomey scaling of droplet number in the 1M microphysics
  (`jcm/physics/clouds/echam_1m.py`).
- `Nccn` → SPA activation floor for the 2M scheme (`jcm/physics/aerosol/spa.py`).

## Tests

- `jcm/physics/aerosol/macv2_sp_test.py`
- `jcm/physics/aerosol/per_band_optics_test.py`

## Current limitations

- The default plume parameters (`AerosolParameters.default()`) are
  **illustrative placeholders** — representative emission-region values, not
  the MACv2.0-SP_v1.nc dataset used by ECHAM/ICON.
- Known fidelity gaps versus the Fortran reference are catalogued in
  `docs/echam_rrtmgp_physics_review.md` §2.30-2.35.
