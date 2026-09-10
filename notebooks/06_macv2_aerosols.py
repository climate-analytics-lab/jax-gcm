"""Time-varying anthropogenic aerosols from MACv2.0-SP.

End-to-end recipe for piping the Stevens et al. (2017) "Simple Plumes"
file `MACv2.0-SP_v1.nc` through `ForcingData` so the model sees real
year-varying and seasonally-varying plume amplitudes (#437 follow-up).

What the netCDF contains:
  - Static plume geometry: `plume_lat`, `plume_lon`, `sig_lat_W/E`,
    `sig_lon_W/E`, `theta`, `ftr_weight`, `beta_a`, `beta_b`,
    `aod_spmx`, `aod_fmbg`, `ssa550`, `asy550`, `angstrom`. These go
    into `AerosolParameters` and replace the placeholder defaults the
    JAX port has been using.
  - Time-varying scaling: `year_weight(plume, year)` of shape (9, 251)
    over years 1850..2100, and `ann_cycle(plume, week, feature)` of
    shape (9, 52, 2) — the seasonal cycle.

What the model needs:
  - `forcing.aerosol_year_weight`: per-step shape `(nplumes,)` — i.e.
    the year_weight for the current model year. We give it as a
    `TimeSeries` of shape (251, 9) indexed `BY_DATE`.
  - `forcing.aerosol_ann_cycle`: per-step shape `(nfeatures, nplumes)`
    — i.e. the ann_cycle for the current model week. We give it as a
    `TimeSeries` of shape (52, 2, 9) indexed `WRAP_YEAR`.

`Model._get_step_fn_factory` calls `forcing.select(date)` once per
step, which collapses each `TimeSeries` to its current-step slice. So
nothing extra is needed at run time.

Run with:
    python notebooks/06_macv2_aerosols.py /path/to/MACv2.0-SP_v1.nc
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import jax_datetime as jdt

from jcm.forcing import (
    ForcingData,
    read_macv2_weights,
)
from jcm.model import Model
from jcm.physics.aerosol.macv2_sp_params import AerosolParameters
from jcm.physics.echam.echam_terms import echam_physics
from jcm.terrain import TerrainData
from jcm.utils import get_coords


# ---------------------------------------------------------------------------
# Step 1 — read MACv2.0-SP_v1.nc into Python
# ---------------------------------------------------------------------------

def load_macv2_sp(path: Path) -> xr.Dataset:
    """Open the MACv2.0-SP netCDF; return as an xarray Dataset."""
    return xr.open_dataset(path)


# ---------------------------------------------------------------------------
# Step 2 — build an AerosolParameters from the static plume geometry
# ---------------------------------------------------------------------------

def aerosol_parameters_from_macv2(ds: xr.Dataset) -> AerosolParameters:
    """Pull the static plume geometry out of the file and pack it into
    `AerosolParameters`.

    Thin wrapper over `AerosolParameters.from_dataset`, which owns the
    `(plume, feature)` -> `(feature, plume)` transposes and the
    jcm-specific extension fields (`spa_prefactor`/`spa_exponent` — a
    hand-rolled constructor that omits them raises TypeError, since
    tree_math structs have no field defaults). Note the struct defaults
    are already these same values: `AerosolParameters.default()` IS the
    MACv2.0-SP_v1.nc static geometry, so this loader is only needed to
    consume a modified parameter file.
    """
    return AerosolParameters.from_dataset(ds)


# ---------------------------------------------------------------------------
# Step 3 — wrap year_weight and ann_cycle as TimeSeries leaves
# ---------------------------------------------------------------------------

# `year_weight` on the file is `(plume, year)` over 1850..2100; the model
# wants it as a `TimeSeries` with the time axis at index 0 and `BY_DATE`
# alignment so it picks the right year from the calendar clock. The v1 file
# only carries valid data for 1850-2016 (2017-2100 are `_FillValue`, i.e.
# NaN); `read_macv2_weights` forward-fills the last valid year rather than
# inject NaN AOD into a post-2016 run.
#
# `ann_cycle` is `(plume, week, feature)`; the model consumes `(feature,
# plume)` per step, so the loader arranges it `(week, feature, plume)` under
# `WRAP_YEAR` alignment (the seasonal cycle repeats every year). Both leaves
# now come from the single reusable loader in `jcm.forcing` (issue #680).


# ---------------------------------------------------------------------------
# Step 4 — pack everything into a ForcingData
# ---------------------------------------------------------------------------

def build_forcing(ds: xr.Dataset, nodal_shape: tuple[int, int]) -> ForcingData:
    """Build a complete `ForcingData` for an aquaplanet-like run with
    real time-varying MACv2-SP aerosols.

    For a run with realistic SST/sea-ice, replace the bare-array fields
    here with `ForcingData.from_dataset(your_era5_ds, coords=...)`.
    """
    year_weight, ann_cycle = read_macv2_weights(ds)
    base = ForcingData.zeros(nodal_shape)
    return base.copy(
        aerosol_year_weight=year_weight,
        aerosol_ann_cycle=ann_cycle,
    )


# ---------------------------------------------------------------------------
# Step 5 — wire AerosolParameters into the ECHAM physics
# ---------------------------------------------------------------------------

def build_physics_with_real_aerosols(ds: xr.Dataset):
    """Construct ECHAM physics with the AerosolParameters from the file."""
    return echam_physics(aerosol=aerosol_parameters_from_macv2(ds))


# ---------------------------------------------------------------------------
# Step 6 — put it all together and run
# ---------------------------------------------------------------------------

def main(macv2_path: Path) -> None:
    ds = load_macv2_sp(macv2_path)

    # Small aquaplanet grid for the demo — substitute whatever
    # resolution your study uses.
    sigma_boundaries = np.linspace(0, 1, 21)  # 20 vertical layers
    coords = get_coords(sigma_boundaries, spectral_truncation=21)
    terrain = TerrainData.aquaplanet(coords)

    forcing = build_forcing(ds, coords.horizontal.nodal_shape)
    physics = build_physics_with_real_aerosols(ds)

    # Run a single month in 1900 (low aerosol) and a single month in 2000
    # (post-WWII industrial peak), and compare anthropogenic AOD.
    for label, start in [("1900", "1900-06-01"), ("2000", "2000-06-01")]:
        model = Model(
            coords=coords,
            terrain=terrain,
            physics=physics,
            time_step=20.0,                              # minutes
            start_date=jdt.to_datetime(start),
            calendar="gregorian",                        # MACv2 years are gregorian
        )
        preds = model.run(forcing=forcing,
                          save_interval="1 day",
                          total_time="30 days")

        # Extract anthropogenic AOD from the diagnostic output.
        diag_ds = preds.to_xarray()
        aod_anth = diag_ds["macsp.aod_anthropogenic"].mean(dim="time")
        print(f"{label}: mean column AOD_anth = "
              f"{float(aod_anth.mean()):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))
