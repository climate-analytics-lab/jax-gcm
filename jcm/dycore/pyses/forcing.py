"""Prescribed-forcing builder for the pyses CAM-SE backend.

:func:`build_forcing` interpolates a jcm-canonical monthly climatological
forcing file (regular lon/lat grid, e.g. ``jcm/data/bc/t63/forcing.nc``) onto
the backend's scattered pg2 physics columns and wraps the result as a normal
:class:`jcm.forcing.ForcingData` in the ``(1, ncol)`` layout. Because the
monthly fields become ordinary ``TimeSeries`` leaves (``WRAP_YEAR``
climatology alignment), ``Model``'s existing per-step
``forcing.select(date)`` machinery — month selection, solar-geometry
population, GHG scalars — works unchanged; nothing downstream knows the
horizontal layout is a column list.

Interpolation is host-side numpy bilinear at build time (see
:mod:`jcm.dycore.pyses.interp` — note there about interpolating offline for
higher-resolution production forcing). ``ForcingData.from_file`` is *not*
reused because its interpolation pipeline is spectral-grid specific
(Gaussian-grid upsampling + monthly→daily interpolation); here the monthly
resolution is kept and ``WRAP_YEAR`` indexing selects the current month —
matching the developer prototype's forcing cadence.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jcm.dycore.pyses.interp import interp_grid_to_points
from jcm.forcing import ForcingData, TimeSeries, WRAP_YEAR, make_time_series


# Time-varying monthly fields expected in the jcm-canonical forcing file,
# mapped to their ForcingData field names.
_TIME_FIELDS = {
    "sst": "sea_surface_temperature",
    "icec": "sice_am",
    "stl": "stl_am",
    "soilw_am": "soilw_am",
    "snowc": "snowc_am",
}


def build_forcing(forcing_file: str, dycore, *, validate: bool = True,
                  emissions_file=None, dms_file=None, dust_file=None,
                  oxidants_file=None, ozone_file=None) -> ForcingData:
    """Interpolate a monthly lon/lat forcing climatology onto the physics columns.

    Args:
        forcing_file: jcm-canonical forcing netCDF: monthly ``sst`` /
            ``icec`` / ``stl`` / ``soilw_am`` / ``snowc`` shaped
            ``(lon, lat, time)`` plus static ``alb`` ``(lon, lat)``.
        dycore: A :class:`~jcm.dycore.pyses.dycore.PysesCamSEDycore` (only
            its ``colmap`` column coordinates are read).
        validate: Run the host-side physical-range sanity check jcm applies
            to boundary data (``jcm.forcing._validate_bc_fields``). Disable
            only for synthetic test fixtures.
        emissions_file: Optional path (or list of paths) to jcm-contract
            emissions netCDF(s) (``emis_<sector>_<species>`` and/or
            ``aero_emis_<tracer>`` on a regular ``(lon, lat)`` grid — see
            ``docs/design/jam.md``). Fields are bilinearly sampled onto the
            columns; grids need not match the met forcing file's.
        dms_file: Optional seawater-DMS climatology
            (``DMS_sea (time, lat, lon)``; :func:`jcm.forcing.read_dms_seawater`).
        dust_file: Optional dust-source/erodibility map (``pot_source``;
            :func:`jcm.forcing.read_dust_source`, static or monthly).
        oxidants_file: Optional oxidant climatology
            (``*_VMR_avrg (time, mlev, lat, lon)`` on the model's ``nlev``
            hybrid levels; :func:`jcm.forcing.read_oxidant_vmr`). Levels map
            one-to-one; only the horizontal is interpolated.

    Returns:
        :class:`ForcingData` whose spatial leaves are ``(1, ncol)`` (static
        albedo) or ``TimeSeries`` with values ``(12, 1, ncol)`` in
        ``WRAP_YEAR`` (climatology) alignment — the current month is
        selected by fraction-of-year, so the ``time_seconds`` axis (month
        starts, 365-day year) is informational.

    """
    import xarray as xr

    from jcm.forcing import _validate_bc_fields

    ds = xr.open_dataset(forcing_file)
    if validate:
        _validate_bc_fields(ds)

    lon = np.asarray(ds["lon"].values)
    lat = np.asarray(ds["lat"].values)
    cm = dycore.colmap
    ncol = cm.num_cols
    col_lon = np.degrees(cm.longitudes)
    col_lat = np.degrees(cm.latitudes)

    n_time = int(ds.sizes["time"])
    if n_time != 12:
        raise ValueError(
            f"build_forcing expects a 12-month climatology; {forcing_file} "
            f"has {n_time} time slices."
        )
    # Month-start seconds in a 365-day year: informational under WRAP_YEAR
    # (selection is by fraction-of-year), but kept physically labelled.
    month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    time_seconds = month_days * 86400.0

    def monthly_to_columns(name):
        arr = np.asarray(ds[name].transpose("lon", "lat", "time").values)
        months = np.stack(
            [interp_grid_to_points(lon, lat, arr[:, :, m], col_lon, col_lat)
             for m in range(n_time)],
            axis=0,
        )                                                  # (12, ncol)
        return months.reshape(n_time, 1, ncol)

    fields = {dest: monthly_to_columns(src) for src, dest in _TIME_FIELDS.items()}
    # Fraction fields pick up interpolation noise at coast/ice edges; clip.
    fields["sice_am"] = np.clip(fields["sice_am"], 0.0, 1.0)
    fields["snowc_am"] = np.clip(fields["snowc_am"], 0.0, 20000.0)

    alb0 = np.clip(
        interp_grid_to_points(
            lon, lat, np.asarray(ds["alb"].transpose("lon", "lat").values),
            col_lon, col_lat,
        ),
        0.0, 1.0,
    ).reshape(1, ncol)

    def ts(values):
        return make_time_series(jnp.asarray(values), time_seconds,
                                align_mode=WRAP_YEAR)

    forcing = ForcingData.zeros(
        nodal_shape=(1, ncol),
        alb0=jnp.asarray(alb0),
        sea_surface_temperature=ts(fields["sea_surface_temperature"]),
        sice_am=ts(fields["sice_am"]),
        stl_am=ts(fields["stl_am"]),
        soilw_am=ts(fields["soilw_am"]),
        snowc_am=ts(fields["snowc_am"]),
    )
    return attach_jam_forcing(
        forcing, col_lon, col_lat, nlev=dycore.nlev,
        emissions_file=emissions_file, dms_file=dms_file,
        dust_file=dust_file, oxidants_file=oxidants_file,
        ozone_file=ozone_file,
    )


# ---------------------------------------------------------------------------
# JAM aerosol / oxidant forcing (emissions, DMS, dust, oxidants)
# ---------------------------------------------------------------------------

def _leaf_to_columns(leaf, lon, lat, col_lon, col_lat):
    """Sample a ``(..., lon, lat)`` forcing leaf onto the physics columns.

    ``leaf`` is either a :class:`TimeSeries` (values ``(time[, lev], lon,
    lat)``) or a bare array ``([lev,] lon, lat)`` — the two leaf kinds the
    ``jcm.forcing`` readers produce. Every leading axis (time, level) is
    sampled independently with the same bilinear kernel used for the met
    forcing, and the horizontal is returned in the backend's ``(1, ncol)``
    layout so the physics terms' ``size == ncols`` ravel checks pass.
    """
    values = leaf.values if isinstance(leaf, TimeSeries) else leaf
    arr = np.asarray(values)
    lead = arr.shape[:-2]
    flat = arr.reshape((-1,) + arr.shape[-2:])
    cols = np.stack(
        [interp_grid_to_points(lon, lat, f, col_lon, col_lat) for f in flat],
        axis=0,
    ).reshape(lead + (1, col_lon.size))
    if isinstance(leaf, TimeSeries):
        return TimeSeries(values=jnp.asarray(cols),
                          time_seconds=leaf.time_seconds,
                          align_mode=leaf.align_mode)
    return jnp.asarray(cols)


def _reader_grid(ds):
    """Return the ``(lon, lat)`` axes matching the readers' output orientation.

    The ``jcm.forcing`` readers normalise fields to ``(..., lon, lat)`` with
    *ascending* latitude (a descending-latitude file is flipped), so the
    interpolation source axes are the file's longitudes as-is and its
    latitudes sorted ascending.
    """
    lon = np.asarray(ds["lon"].values, dtype=float)
    lat = np.sort(np.asarray(ds["lat"].values, dtype=float))
    return lon, lat


def attach_jam_forcing(forcing, col_lon, col_lat, *, nlev,
                       emissions_file=None, dms_file=None, dust_file=None,
                       oxidants_file=None, ozone_file=None) -> ForcingData:
    """Attach JAM emission/oxidant fields to a column-layout ``ForcingData``.

    The column analogue of ``jcm.runners``' ``_attach_emissions`` /
    ``_attach_dms`` / ``_attach_dust`` / ``_attach_oxidants``: each file is
    parsed by the shared ``jcm.forcing`` reader (which owns the variable
    contracts, unit conversions and orientation), then every leaf is
    bilinearly sampled onto the ``(col_lon, col_lat)`` points. Files may
    each live on their own regular lon/lat grid — unlike the spectral
    runner path there is no exact-grid requirement, because interpolation
    onto scattered columns happens here anyway (same rationale as the met
    forcing downscale). All-``None`` files make this a no-op.
    """
    import xarray as xr

    from jcm.forcing import (
        read_anthropogenic_emissions,
        read_dms_seawater,
        read_dust_source,
        read_oxidant_vmr,
        read_prescribed_aerosol_emissions,
    )

    col_lon = np.asarray(col_lon, dtype=float)
    col_lat = np.asarray(col_lat, dtype=float)

    def to_cols(leaf, lon, lat):
        return _leaf_to_columns(leaf, lon, lat, col_lon, col_lat)

    if emissions_file is not None:
        paths = ([str(p) for p in emissions_file]
                 if isinstance(emissions_file, (list, tuple))
                 else [str(emissions_file)])
        ds = (xr.open_mfdataset(paths, combine="by_coords")
              if len(paths) > 1 else xr.open_dataset(paths[0]))
        with ds:
            # The emissions readers keep horizontal axes in file order; the
            # jcm emissions-prep contract is canonical ``(time, lon, lat)``.
            # Assert rather than guess — a (lat, lon) file interpolated with
            # swapped axes would silently misplace every source region.
            for name in ds.data_vars:
                if str(name).startswith(("emis_", "aero_emis_")):
                    if tuple(ds[name].dims[-2:]) != ("lon", "lat"):
                        raise ValueError(
                            f"emissions field {name!r} has dims {ds[name].dims}; "
                            "expected trailing ('lon', 'lat') (jcm emissions "
                            "contract — regenerate with jcm.data.emissions.prepare)."
                        )
            lon = np.asarray(ds["lon"].values, dtype=float)
            lat = np.asarray(ds["lat"].values, dtype=float)
            anthro = read_anthropogenic_emissions(ds)
            speciated = read_prescribed_aerosol_emissions(ds)
        if anthro is None and speciated is None:
            raise ValueError(
                f"emissions_file {emissions_file!r} has no emissions variables "
                "(expected ``emis_<sector>_<species>`` or ``aero_emis_<tracer>``)."
            )
        if anthro is not None:
            anthro = {k: to_cols(v, lon, lat) for k, v in anthro.items()}
        if speciated is not None:
            speciated = {k: to_cols(v, lon, lat) for k, v in speciated.items()}
        forcing = forcing.copy(anthropogenic_emissions=anthro,
                               prescribed_aerosol_emissions=speciated)

    if dms_file is not None:
        with xr.open_dataset(str(dms_file)) as ds:
            lon, lat = _reader_grid(ds)
            forcing = forcing.copy(
                dms_seawater=to_cols(read_dms_seawater(ds), lon, lat))

    if dust_file is not None:
        with xr.open_dataset(str(dust_file)) as ds:
            lon, lat = _reader_grid(ds)
            forcing = forcing.copy(
                dust_source=to_cols(read_dust_source(ds), lon, lat))

    if oxidants_file is not None:
        with xr.open_dataset(str(oxidants_file)) as ds:
            lon, lat = _reader_grid(ds)
            vmr = read_oxidant_vmr(ds, nlev=nlev)
            forcing = forcing.copy(
                oxidant_vmr={k: to_cols(v, lon, lat) for k, v in vmr.items()})

    if ozone_file is not None:
        # Column analogue of ``jcm.runners._attach_ozone`` (Codex review of
        # #575: without this the pySES path silently kept the analytic ozone
        # profile and its ~12 W/m2 clear-sky OLR bias). The file follows the
        # ``jcm.data.bc.interpolate_ozone`` contract — ``O3 (time, level,
        # lat, lon)`` mole/mole already on the model's vertical levels — so
        # only the horizontal is sampled onto the columns; unlike the
        # spectral runner there is no exact-grid requirement.
        from jcm.ozone_climatology import OzoneClimatology

        with xr.open_dataset(str(ozone_file)) as ds:
            file_nlev = int(ds.sizes.get("level", -1))
            if file_nlev != int(nlev):
                raise ValueError(
                    f"ozone file {ozone_file!r} has {file_nlev} levels but the "
                    f"model has {nlev}; regenerate with "
                    "jcm.data.bc.interpolate_ozone --nlevels matching the run."
                )
            if int(ds.sizes.get("time", 0)) != 12:
                raise ValueError(
                    f"ozone file {ozone_file!r} is not a 12-month climatology "
                    "(transient files are not supported on the pySES path)."
                )
            lat = np.asarray(ds["lat"].values, dtype=float)
            o3 = np.asarray(
                ds["O3"].transpose("time", "level", "lon", "lat").values,
                dtype=float,
            ) * 1.0e6                                    # mole/mole -> ppmv
            if lat[0] > lat[-1]:                         # ascending-lat kernel
                lat = lat[::-1]
                o3 = o3[..., ::-1]
            lon = np.asarray(ds["lon"].values, dtype=float)
        cols = _leaf_to_columns(o3, lon, lat, col_lon, col_lat)
        seconds_per_month = 30.4375 * 86400.0            # match from_file
        ts = make_time_series(
            jnp.asarray(cols, dtype=jnp.float32),
            jnp.asarray((np.arange(12) + 0.5) * seconds_per_month,
                        dtype=jnp.float32),
            WRAP_YEAR,
        )
        forcing = forcing.copy(ozone_climatology=OzoneClimatology(o3_ppmv=ts))

    return forcing
