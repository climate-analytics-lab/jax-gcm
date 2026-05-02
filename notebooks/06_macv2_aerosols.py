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

import jax.numpy as jnp
import numpy as np
import xarray as xr
import jax_datetime as jdt

from jcm.forcing import (
    BY_DATE,
    WRAP_YEAR,
    ForcingData,
    make_time_series,
)
from jcm.model import Model
from jcm.physics.aerosol.macv2_sp_params import AerosolParameters
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics.echam.parameters import Parameters
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
    `AerosolParameters`. These replace the placeholder defaults that the
    JAX port has been using.

    Note on shape conventions: the netCDF stores `(plume, feature)` for
    feature-indexed fields; the JAX port expects `(feature, plume)`.
    We transpose at load time.
    """
    def to_jax(name: str) -> jnp.ndarray:
        return jnp.asarray(ds[name].values)

    def to_jax_T(name: str) -> jnp.ndarray:
        # netCDF: (plume, feature). JAX struct: (feature, plume).
        return jnp.asarray(ds[name].values.T)

    return AerosolParameters(
        nplumes=int(ds.sizes["plume_number"]),
        nfeatures=int(ds.sizes["plume_feature"]),
        plume_lat=to_jax("plume_lat"),
        plume_lon=to_jax("plume_lon"),
        beta_a=to_jax("beta_a"),
        beta_b=to_jax("beta_b"),
        aod_spmx=to_jax("aod_spmx"),
        aod_fmbg=to_jax("aod_fmbg"),
        asy550=to_jax("asy550"),
        ssa550=to_jax("ssa550"),
        angstrom=to_jax("angstrom"),
        sig_lon_E=to_jax_T("sig_lon_E"),
        sig_lon_W=to_jax_T("sig_lon_W"),
        sig_lat_E=to_jax_T("sig_lat_E"),
        sig_lat_W=to_jax_T("sig_lat_W"),
        theta=to_jax_T("theta"),
        ftr_weight=to_jax_T("ftr_weight"),
        background_aod=jnp.asarray(0.02),
    )


# ---------------------------------------------------------------------------
# Step 3 — wrap year_weight and ann_cycle as TimeSeries leaves
# ---------------------------------------------------------------------------

def macv2_year_weight_timeseries(ds: xr.Dataset) :
    """`year_weight` on the file is `(plume, year)` over 1850..2100.

    For the model we want it as a `TimeSeries` with the time axis at
    index 0 and `BY_DATE` alignment so the model picks the right year
    based on its calendar clock.
    """
    # Transpose to (year, plume).
    yw = jnp.asarray(ds["year_weight"].values.T)  # (251, 9)

    # Build a time axis in seconds-since-1970 for each year-start.
    # The MACv2 file labels year `Y` as the integer Y; we treat that as
    # `Y-01-01 00:00 UTC`.
    years = ds["years"].values.astype(int)
    epoch_seconds = []
    epoch = jdt.Datetime.from_pydatetime(jdt.to_datetime("1970-01-01"))
    for y in years:
        when = jdt.Datetime.from_pydatetime(jdt.to_datetime(f"{int(y)}-01-01"))
        delta = when - epoch
        epoch_seconds.append(float(delta.days) * 86400.0)
    time_seconds = jnp.asarray(epoch_seconds)

    return make_time_series(yw, time_seconds, align_mode=BY_DATE)


def macv2_ann_cycle_timeseries(ds: xr.Dataset) :
    """`ann_cycle` on the file is `(plume, week, feature)`. The model
    consumes a per-step shape of `(nfeatures, nplumes)`.

    We arrange the stored array as `(week, feature, plume)` so that
    `select(date)` slicing axis 0 produces `(feature, plume)` at the
    current week. We use `WRAP_YEAR` alignment because the seasonal
    cycle repeats every year.
    """
    # netCDF: (plume, week, feature) → (week, feature, plume)
    ac = jnp.asarray(np.transpose(ds["ann_cycle"].values, (1, 2, 0)))
    # `time_seconds` is unused by `WRAP_YEAR` indexing, but we still need
    # a 1-D coord of the right length; pass the week index for clarity.
    weeks = jnp.arange(ac.shape[0])
    return make_time_series(ac, weeks, align_mode=WRAP_YEAR)


# ---------------------------------------------------------------------------
# Step 4 — pack everything into a ForcingData
# ---------------------------------------------------------------------------

def build_forcing(ds: xr.Dataset, nodal_shape: tuple[int, int]) -> ForcingData:
    """Build a complete `ForcingData` for an aquaplanet-like run with
    real time-varying MACv2-SP aerosols.

    For a run with realistic SST/sea-ice, replace the bare-array fields
    here with `ForcingData.from_dataset(your_era5_ds, coords=...)`.
    """
    base = ForcingData.zeros(nodal_shape)
    return base.copy(
        aerosol_year_weight=macv2_year_weight_timeseries(ds),
        aerosol_ann_cycle=macv2_ann_cycle_timeseries(ds),
    )


# ---------------------------------------------------------------------------
# Step 5 — wire AerosolParameters into the ECHAM physics
# ---------------------------------------------------------------------------

def build_physics_with_real_aerosols(ds: xr.Dataset):
    """Construct ECHAM physics with the AerosolParameters from the file."""
    params = Parameters.default()._replace(
        aerosol=aerosol_parameters_from_macv2(ds),
    )
    return echam_physics(parameters=params)


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
        aod_anth = diag_ds["aerosol.aod_anthropogenic"].mean(dim="time")
        print(f"{label}: mean column AOD_anth = "
              f"{float(aod_anth.mean()):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))
