"""Drive SPEEDY physics as a single column at SGP with ARMBE state.

Diagnostic mode: the observed profiles are prescribed at every step and the
physics diagnoses radiation, convection, condensation and surface fluxes on
them. The dynamical core never runs. Only tracers evolve.

    source env.sh
    python run_scm.py                       # synthetic fixture
    python run_scm.py --atm data/sgparmbeatmC1.c1 --cldrad data/sgparmbecldradC1.c1

Writes ``outputs/scm_run.npz`` for ``evaluate.py``.

Two things worth knowing about this setup:

* ``start_date`` is taken from the data and matters a lot. The SCM selects
  forcing per step from it, and SPEEDY's insolation keys off fraction-of-year;
  getting it wrong silently gives you the wrong season's sun (see
  ``SCM_FORCING_PATCH_NOTE.md`` at the repo root).
* ``ForcingData`` is otherwise static across the run — surface temperature, soil
  moisture and albedo are set once from the record's means rather than following
  the observations hour by hour. Fine for a first look; it is the obvious next
  thing to improve if surface fluxes look off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax_datetime as jdt
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
    load_armbe,
    pick,
    to_obs_targets,
    to_state_series,
)

# physics_data term -> fields we pull out as diagnostics.
DIAGNOSTICS = {
    "_condensation": ("precls",),
    "_convection": ("precnv",),
    "_shortwave_rad": ("rsds", "rsns", "fsol", "cloudc", "ftop"),
    "_surface_flux": ("rlds", "rlus", "shf", "evap", "tsfc"),
}


def unwrap(o):
    """physics_data stores each term as a 0-d object array around a dataclass."""
    if getattr(o, "dtype", None) == object and getattr(o, "shape", None) == ():
        return o.item()
    return o


def extract(physics_data) -> dict[str, np.ndarray]:
    out = {}
    for term, fields in DIAGNOSTICS.items():
        obj = unwrap(physics_data[term])
        d = vars(obj) if hasattr(obj, "__dict__") else obj
        for f in fields:
            if f in d:
                out[f"{term.lstrip('_')}.{f}"] = np.asarray(d[f])
    return out


def build_forcing(ds, nodal_shape=(1, 1)) -> ForcingData:
    """Static land-surface forcing for SGP from the record's own statistics."""
    t_name = pick(ds, "surface_temperature", required=False)
    if t_name is not None:
        t_sfc = float(np.nanmean(np.asarray(ds[t_name].values, dtype=float)))
        if t_sfc < 100.0:
            t_sfc += 273.15
    else:
        t_sfc = 295.0
    ones = jnp.ones(nodal_shape)
    return ForcingData.zeros(
        nodal_shape,
        alb0=0.20 * ones,           # bare-land albedo, typical for SGP pasture
        stl_am=t_sfc * ones,        # land surface temperature
        sea_surface_temperature=t_sfc * ones,   # unused at a land point
        soilw_am=0.30 * ones,
        snowc_am=jnp.zeros(nodal_shape),
        sice_am=jnp.zeros(nodal_shape),
        co2_vmr=jnp.asarray(407.0),   # ~2018 global mean, ppmv
    ), t_sfc


def main(argv=None) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atm", default=str(here / "data/synthetic/sgparmbeatmC1.c1.synthetic.nc"))
    ap.add_argument("--cldrad", default=str(here / "data/synthetic/sgparmbecldradC1.c1.synthetic.nc"))
    ap.add_argument("--nlev", type=int, default=8)
    ap.add_argument("--dt", type=float, default=3600.0, help="seconds; match obs cadence")
    ap.add_argument("--start", default=None, help="window start, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="window end, YYYY-MM-DD")
    ap.add_argument("--output", type=Path, default=here / "outputs" / "scm_run.npz")
    args = ap.parse_args(argv)

    ds = load_armbe(args.atm, args.cldrad, args.start, args.end)
    states, times, meta = to_state_series(ds, nlev=args.nlev)
    if not states:
        raise SystemExit("no usable states built from the input — check the loader "
                         "report above for unresolved variables.")
    obs = to_obs_targets(ds)

    # The date the states actually correspond to. This drives insolation.
    t0 = np.asarray(times)[0]
    start_date = jdt.to_datetime(str(np.datetime64(t0, "D")))

    forcing, t_sfc = build_forcing(ds)
    terrain = TerrainData.single_column(orog=SGP_OROG_M, fmask=1.0)  # land

    scm = SingleColumnModel(
        physics=speedy_physics(),
        vertical=SigmaCoordinates.equidistant(args.nlev),
        lat_deg=SGP_LAT_DEG,
        lon_deg=SGP_LON_DEG,
        terrain=terrain,
        forcing=forcing,
        dt_seconds=args.dt,
        start_date=start_date,
        calendar="gregorian",
    )

    print(f"site      : SGP C1  ({SGP_LAT_DEG}N, {SGP_LON_DEG}E)  orog={SGP_OROG_M}m  land")
    print(f"physics   : SPEEDY, {args.nlev} sigma levels, dt={args.dt:.0f}s")
    print(f"start_date: {np.datetime64(t0, 's')}  (drives insolation)")
    print(f"steps     : {len(states)}  ({len(states)*args.dt/86400:.1f} days)")
    print(f"sfc temp  : {t_sfc:.1f} K (static forcing)")
    print(f"dropped   : {meta['n_dropped']} of {meta['n_input_times']} input times")

    pred = scm.run(states)
    diags = extract(pred.physics_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        times=np.asarray(times).astype("datetime64[s]").astype(np.int64),
        dt_seconds=args.dt,
        **{f"model.{k}": v for k, v in diags.items()},
        **{f"obs.{k}": v for k, v in obs.items()},
    )
    print(f"\nwrote {args.output}")
    print("\nmodel diagnostics:")
    for k, v in sorted(diags.items()):
        flat = np.asarray(v).reshape(v.shape[0], -1)
        print(f"  {k:28s} shape={str(v.shape):16s} "
              f"mean={np.nanmean(flat):10.4g}  finite={np.all(np.isfinite(v))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
