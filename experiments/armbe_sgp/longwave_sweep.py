"""Probe SPEEDY SCM downward-longwave sensitivity to humidity.

This is a controlled diagnostic, not an ARMBE evaluation. It holds temperature,
wind, terrain, and date fixed while scaling a representative g/kg humidity
profile. The output exercises the same composable SPEEDY path used by
``run_scm.py`` and makes the humidity unit explicit.

    JAX_PLATFORMS=cpu python longwave_sweep.py
"""

from __future__ import annotations

import argparse

import numpy as np
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.single_column_model import SingleColumnModel
from jcm.terrain import TerrainData

from armbe_io import (
    SGP_LAT_DEG,
    SGP_LON_DEG,
    SGP_OROG_M,
    speedy_column_state,
    speedy_sigma_levels,
)
from run_scm import unwrap


def _state_for_humidity(surface_q_gkg: float, nlev: int):
    """Return a stable column with a fixed-shape g/kg humidity profile."""
    sigma = speedy_sigma_levels(nlev)
    temperature = np.linspace(220.0, 298.0, nlev)
    q_gkg = surface_q_gkg * sigma ** 2
    return speedy_column_state(
        temperature_k=temperature,
        specific_humidity=q_gkg,
        u_wind=np.full(nlev, 5.0),
        v_wind=np.zeros(nlev),
        surface_pressure_pa=97500.0,
        sigma=sigma,
    )


def sweep(surface_q_gkg: list[float], nlev: int = 8) -> np.ndarray:
    """Return one `rlds` value for each requested surface humidity in g/kg."""
    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(nlev),
        lat_deg=SGP_LAT_DEG,
        lon_deg=SGP_LON_DEG,
        terrain=TerrainData.single_column(orog=SGP_OROG_M, fmask=1.0, lfluxland=True),
        forcing=ForcingData.zeros((1, 1)),
        dt_seconds=3600.0,
    )
    predictions = scm.run([_state_for_humidity(q, nlev) for q in surface_q_gkg])
    surface_flux = unwrap(predictions.physics_data["_surface_flux"])
    return np.asarray(surface_flux.rlds)[:, 0, 0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface-q-gkg",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0, 30.0],
        help="surface specific-humidity values in g/kg",
    )
    args = parser.parse_args(argv)
    if any(q < 0.0 for q in args.surface_q_gkg):
        parser.error("--surface-q-gkg values must be non-negative")

    rlds = sweep(args.surface_q_gkg)
    print("SPEEDY SCM downward-longwave humidity sweep")
    print("surface q [g/kg]  rlds [W/m2]")
    for q, flux in zip(args.surface_q_gkg, rlds):
        print(f"{q:16.3f}  {flux:11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
