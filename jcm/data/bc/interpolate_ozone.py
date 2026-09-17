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


def vertical_interp_log_p(
    source: np.ndarray, plev_source: np.ndarray, plev_target: np.ndarray,
) -> np.ndarray:
    """Vertical-interp ``source`` from ``plev_source`` to ``plev_target``.

    Public because nothing here is ozone-specific: the same log-pressure
    interpolation applies to any field given on pressure levels, temperature,
    humidity and winds included. Only *axis 1* is fixed as the pressure axis;
    every other axis is carried through untouched whatever it means, so a
    ``(time, plev, lat, lon)`` file, a ``(time, plev, column)`` set of columns
    and an ``(ensemble, plev, time, lat, lon)`` stack all work unchanged
    (#830 — the earlier version unpacked exactly four dimensions, which
    contradicted this documented contract).

    Args:
        source: ``(d0, nplev_source, ...)`` field with the source pressure on
            axis 1 and at least two dimensions. Any number of leading and
            trailing axes is allowed and their order is preserved.
        plev_source: ``(nplev_source,)`` source pressure (Pa), strictly
            monotonic in either direction.
        plev_target: ``(nplev_target,)`` target pressure (Pa).

    Returns:
        ``source``'s shape with axis 1 replaced by ``plev_target``, in
        ``source``'s dtype.

    Raises:
        ValueError: if ``source`` has fewer than two dimensions, if
            ``plev_source`` is not 1-D, if its length does not match
            ``source.shape[1]``, or if it is not strictly monotonic.

    """
    source = np.asarray(source)
    plev_source = np.asarray(plev_source)
    plev_target = np.asarray(plev_target)

    if source.ndim < 2:
        raise ValueError(
            f"source carries the pressure on axis 1, so it needs at least 2 "
            f"dimensions; got shape {source.shape}"
        )
    if plev_source.ndim != 1:
        raise ValueError(
            f"plev_source must be 1-D; got shape {plev_source.shape}"
        )
    if plev_source.size != source.shape[1]:
        raise ValueError(
            f"plev_source has {plev_source.size} levels but source axis 1 has "
            f"{source.shape[1]} (source shape {source.shape}); the pressure "
            f"axis of source must be axis 1"
        )

    log_src = np.log(plev_source)
    # ``np.interp`` needs increasing ``xp`` and returns silent nonsense
    # otherwise, so a descending source — the common top-first file
    # convention, which the shipped ozone climatologies use — is flipped, and
    # a non-monotonic one is rejected rather than quietly interpolated.
    if plev_source.size > 1 and log_src[0] > log_src[-1]:
        log_src = log_src[::-1]
        source = np.flip(source, axis=1)
    if plev_source.size > 1 and not np.all(np.diff(log_src) > 0):
        raise ValueError(
            "plev_source must be strictly monotonic (ascending or "
            f"descending); got {plev_source}"
        )

    log_tgt = np.log(plev_target)

    # numpy.interp is 1-D, so the field is reshaped to (ncolumns, nplev) and
    # looped over. Moving axis 1 to the end and flattening everything else is
    # what makes the rank arbitrary: no axis but the pressure one is ever
    # named. This runs once, offline, so the explicit loop is fine — and it
    # keeps the result bit-identical to the per-column loop this replaced.
    moved = np.moveaxis(source, 1, -1)
    lead_shape = moved.shape[:-1]
    flat = moved.reshape(-1, plev_source.size)
    out = np.empty((flat.shape[0], plev_target.size), dtype=source.dtype)
    for k in range(flat.shape[0]):
        out[k] = np.interp(log_tgt, log_src, flat[k])
    return np.moveaxis(out.reshape(*lead_shape, plev_target.size), -1, 1)


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

    o3_out = vertical_interp_log_p(o3_in, plev_source, plev_target)

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
