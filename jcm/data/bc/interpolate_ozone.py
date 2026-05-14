r"""Pre-interpolate a CMIP6/ECHAM-style ozone climatology onto the model's
hybrid-level grid so the online code can drop the file straight into the
forcing time slicer with no per-step interpolation.

Input file shape:  ``(time=12, plev, lat, lon)`` mole/mole netCDF
Output file shape: ``(time=12, level=nlevels, lat, lon)`` mole/mole netCDF

The output's vertical axis is the model's hybrid pressure-level *centers*
evaluated at a reference surface pressure of 1013.25 hPa. The error vs
evaluating at each column's actual surface pressure is well under 1% for
typical surface-pressure variations and saves a per-step ``vmap`` of
``jnp.interp`` over every column.

Example::

    python -m jcm.data.bc.interpolate_ozone \\
        --in /path/T63_ozone_picontrol.nc \\
        --out jcm/data/bc/T63L47_ozone_picontrol.nc \\
        --nlevels 47

Loaded online by ``jcm.ozone_climatology.OzoneClimatology.from_file``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.physics.echam.echam_levels import get_echam_levels


REFERENCE_SURFACE_PRESSURE_PA = 101325.0


def _vertical_interp_log_p(
    o3_source: np.ndarray, plev_source: np.ndarray, plev_target: np.ndarray,
) -> np.ndarray:
    """Vertical-interp ``o3_source`` from ``plev_source`` to ``plev_target``.

    Args:
        o3_source: ``(..., nplev_source, ..., ...)`` ozone field; the
            ``plev_source`` axis is assumed to be the second one
            (``(time, plev, lat, lon)`` files).
        plev_source: ``(nplev_source,)`` source pressure (Pa).
        plev_target: ``(nplev_target,)`` target pressure (Pa).

    Returns:
        ``(time, nplev_target, lat, lon)`` interpolated array.

    """
    log_src = np.log(plev_source)
    log_tgt = np.log(plev_target)
    if log_src[0] > log_src[-1]:
        log_src = log_src[::-1]
        o3_source = o3_source[:, ::-1]
    out = np.empty(
        (o3_source.shape[0], plev_target.size, *o3_source.shape[2:]),
        dtype=o3_source.dtype,
    )
    # numpy.interp is 1-D — loop over horizontal+time. Once-only offline,
    # so the explicit loop is fine.
    ntime, _, nlat, nlon = o3_source.shape
    for t in range(ntime):
        for j in range(nlat):
            for i in range(nlon):
                out[t, :, j, i] = np.interp(log_tgt, log_src, o3_source[t, :, j, i])
    return out


def interpolate_ozone(
    input_path: str | Path,
    output_path: str | Path,
    nlevels: int,
    var_name: str = "O3",
    reference_ps_pa: float = REFERENCE_SURFACE_PRESSURE_PA,
) -> None:
    """Vertical-interp an ozone climatology onto the ECHAM hybrid grid.

    Args:
        input_path: Source netCDF (``(time, plev, lat, lon)`` mole/mole).
        output_path: Destination netCDF (``(time, level=nlevels, lat, lon)``).
        nlevels: Number of model vertical levels.
        var_name: Source variable name (default ``"O3"``).
        reference_ps_pa: Reference surface pressure used to evaluate the
            hybrid-level centers (default 1013.25 hPa).

    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    ds_in = xr.open_dataset(input_path, decode_times=False)
    if var_name not in ds_in.data_vars:
        raise ValueError(
            f"{input_path} missing '{var_name}' variable; have "
            f"{list(ds_in.data_vars)}"
        )
    o3_in = ds_in[var_name].values
    if o3_in.ndim != 4:
        raise ValueError(
            f"Expected '{var_name}' shape (time, plev, lat, lon); got {o3_in.shape}"
        )
    plev_source = np.asarray(ds_in[var_name].plev.values)

    # Build the model's hybrid-level center pressures at the reference ps.
    # ``a_centers`` is in Pa; ``b_centers`` is dimensionless. Together
    # they give p_k = a_k + b_k * ps for any column with surface pressure ps.
    vertical = get_echam_levels(nlevels)
    a = np.asarray(vertical.a_centers)
    b = np.asarray(vertical.b_centers)
    plev_target = a + b * reference_ps_pa

    o3_out = _vertical_interp_log_p(o3_in, plev_source, plev_target)

    # Build output dataset preserving lat/lon/time, replacing plev with level.
    ds_out = xr.Dataset(
        {
            var_name: (
                ("time", "level", "lat", "lon"),
                o3_out.astype(np.float32),
                {
                    "units": "mole mole-1",
                    "long_name": (
                        f"Ozone climatology vertically-interpolated to ECHAM "
                        f"L{nlevels} hybrid-center pressures at reference "
                        f"surface pressure {reference_ps_pa:g} Pa"
                    ),
                },
            ),
            "level_pressure_pa": (
                ("level",),
                plev_target.astype(np.float32),
                {
                    "units": "Pa",
                    "long_name": (
                        f"Hybrid-level center pressures at reference ps "
                        f"= {reference_ps_pa:g} Pa"
                    ),
                },
            ),
            "level_a_pa": (
                ("level",), a.astype(np.float32),
                {"units": "Pa", "long_name": "Hybrid-coordinate a_centers"},
            ),
            "level_b": (
                ("level",), b.astype(np.float32),
                {"units": "1", "long_name": "Hybrid-coordinate b_centers"},
            ),
        },
        coords={
            "time": ds_in.time,
            "level": np.arange(nlevels, dtype=np.int32),
            "lat": ds_in.lat,
            "lon": ds_in.lon,
        },
    )
    ds_out.attrs.update({
        "history": (
            f"Vertically interpolated from {input_path.name} to ECHAM "
            f"L{nlevels} via jcm.data.bc.interpolate_ozone "
            f"(reference ps = {reference_ps_pa:g} Pa)."
        ),
        "source_file": str(input_path),
    })
    ds_out.to_netcdf(output_path)
    print(
        f"Wrote {output_path}: shape {tuple(ds_out[var_name].shape)} "
        f"(time × L{nlevels} × lat × lon)"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Pre-interpolate a CMIP6 ozone climatology to the "
                    "model's hybrid-level grid.",
    )
    parser.add_argument("--in", dest="input_path", required=True,
                        help="Source netCDF (time, plev, lat, lon).")
    parser.add_argument("--out", dest="output_path", required=True,
                        help="Destination netCDF (time, level, lat, lon).")
    parser.add_argument("--nlevels", type=int, required=True,
                        help="Number of model vertical levels.")
    parser.add_argument("--var", default="O3",
                        help="Source variable name (default 'O3').")
    parser.add_argument(
        "--reference-ps-pa", type=float, default=REFERENCE_SURFACE_PRESSURE_PA,
        help=f"Reference surface pressure for hybrid-level evaluation "
             f"(default {REFERENCE_SURFACE_PRESSURE_PA} Pa).",
    )
    args = parser.parse_args(argv)

    try:
        interpolate_ozone(
            input_path=args.input_path,
            output_path=args.output_path,
            nlevels=args.nlevels,
            var_name=args.var,
            reference_ps_pa=args.reference_ps_pa,
        )
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
