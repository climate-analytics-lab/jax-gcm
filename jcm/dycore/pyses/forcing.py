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


def sample_forcing_to_columns(ds, lon, lat, col_lon, col_lat):
    """Sample a jcm-canonical forcing climatology onto physics columns.

    Returns ``(monthly, static)``: ``monthly`` maps each time-varying source
    variable to a ``(12, ncol)`` array, ``static`` maps ``alb`` / ``forest``
    / ``glac`` to ``(ncol,)``. Bilinear (:func:`interp_grid_to_points`); a
    file that carries its land share ``lsm`` has its land-conditional
    channels (``jcm.data.regridding.CONDITIONAL_FIELDS``) sampled with their
    land / non-glacier-land weights, so an ocean or glacier neighbour does
    not dilute a coastal or ice-margin column (#672) — the same helper the
    bundle builders and the spectral upsampler use. Files without ``lsm``
    keep the plain bilinear sample.
    """
    from jcm.data.regridding import CONDITIONAL_FIELDS, regrid_land_surface

    n_time = int(ds.sizes["time"])

    def lonlat(name):
        dims = ("lon", "lat") + (("time",) if "time" in ds[name].dims else ())
        return np.asarray(ds[name].transpose(*dims).values, dtype=np.float64)

    def regrid(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            return interp_grid_to_points(lon, lat, arr, col_lon, col_lat)
        return np.stack([interp_grid_to_points(lon, lat, arr[:, :, m],
                                               col_lon, col_lat)
                         for m in range(arr.shape[-1])], axis=-1)

    monthly_names = [name for name in (*_TIME_FIELDS, "soilw_rel")
                     if name in ds.data_vars]
    static_names = [name for name in ("alb", "forest", "glac")
                    if name in ds.data_vars]
    raw = {name: lonlat(name) for name in (*monthly_names, *static_names)}
    if "lsm" in ds.data_vars:
        glac = raw["glac"] if "glac" in raw else None
        out = regrid_land_surface(raw, np.clip(lonlat("lsm"), 0.0, 1.0),
                                  regrid, glac=glac)
        out = {k: (v if k in CONDITIONAL_FIELDS else regrid(v))
               for k, v in out.items()}
    else:
        out = {name: regrid(value) for name, value in raw.items()}

    # Fraction fields pick up interpolation noise at coast/ice edges; clip.
    bounds = {"icec": (0.0, 1.0),
              "snowc": (0.0, 20000.0), "soilw_rel": (0.0, 1.0),
              "alb": (0.0, 1.0), "forest": (0.0, 1.0), "glac": (0.0, 1.0)}
    for name, (lo, hi) in bounds.items():
        if name in out:
            out[name] = np.clip(out[name], lo, hi)
    monthly = {name: np.moveaxis(out[name], -1, 0).reshape(n_time, -1)
               for name in monthly_names}
    static = {name: out[name].reshape(-1) for name in static_names}
    return monthly, static


def build_forcing(forcing_file: str, dycore, *, validate: bool = True,
                  emissions_file=None, dms_file=None, dust_file=None,
                  dust_preferential_file=None, dust_soil_types_file=None,
                  dust_regions_file=None, dust_roughness_file=None,
                  oxidants_file=None, ozone_file=None, align_mode="auto",
                  emissions_align="auto", oxidants_align="auto",
                  ozone_align="auto") -> ForcingData:
    """Interpolate a monthly lon/lat forcing climatology onto the physics columns.

    Args:
        forcing_file: jcm-canonical forcing netCDF: monthly ``sst`` /
            ``icec`` / ``stl`` / ``soilw_am`` / ``snowc`` shaped
            ``(lon, lat, time)`` plus static ``alb`` ``(lon, lat)``, and the
            optional ``soilw_rel`` relative soil wetness (#787) and static
            ``forest`` / ``glac`` land-cover fractions (#672).
        dycore: A :class:`~jcm.dycore.pyses.dycore.PysesCamSEDycore` (only
            its ``colmap`` column coordinates are read).
        align_mode: ``forcing.align`` for ``forcing_file``. The column reader
            supports only a climatology, so it must resolve to ``wrap_year``
            — explicitly, or via ``auto`` for a data-mirror/packaged
            climatology (:func:`jcm.forcing.resolve_align`, #884); anything
            else raises.
        emissions_align, oxidants_align, ozone_align: the per-input
            alignment specs forwarded to :func:`attach_jam_forcing`.
        validate: Run the host-side physical-range sanity check jcm applies
            to boundary data (``jcm.forcing._validate_bc_fields``). Disable
            only for synthetic test fixtures.
        emissions_file: Optional path (or list of paths) to jcm-contract
            emissions netCDF(s) (``emis_<sector>_<species>`` and/or
            ``aero_emis_<tracer>`` on a regular ``(lon, lat)`` grid — see
            ``docs/source/design/jam.md``). Fields are bilinearly sampled onto the
            columns; grids need not match the met forcing file's.
        dms_file: Optional seawater-DMS climatology
            (``DMS_sea (time, lat, lon)``; :func:`jcm.forcing.read_dms_seawater`).
        dust_file: Optional monthly potential-dust-source climatology
            (``pot_source``; :func:`jcm.forcing.read_dust_source`).
        dust_preferential_file, dust_soil_types_file, dust_regions_file:
            The static companions the Tegen scheme needs alongside it
            (paleolake fraction, nine soil textures, 1-8 tuning regions).
        dust_roughness_file: Optional monthly satellite roughness map [cm],
            read only on the ``ndurough = 0`` sensitivity path.
        oxidants_file: Optional oxidant climatology — a single path or the
            yearly file list of ONE transient product (a ``{year}`` expansion,
            opened together along a shared time axis)
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

    from jcm.forcing import _validate_bc_fields, resolve_align

    if resolve_align(align_mode, paths=forcing_file,
                     config_key="forcing.align") != "wrap_year":
        raise ValueError(
            f"forcing.align={align_mode!r}: the pySES column forcing reader "
            "supports only a 12-month climatology (wrap_year); transient "
            "surface forcing needs the spectral dinosaur backend.")
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

    fields, static = sample_forcing_to_columns(ds, lon, lat, col_lon, col_lat)
    fields = {dest: fields[src].reshape(n_time, 1, ncol)
              for src, dest in {**_TIME_FIELDS, "soilw_rel": "soilw_rel"}.items()
              if src in fields}
    alb0 = static["alb"].reshape(1, ncol)

    def static_fraction(name):
        # Optional static land cover (#672); absent -> ``None`` = none.
        if name not in static:
            return None
        return jnp.asarray(static[name].reshape(1, ncol))

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
        soilw_rel=(ts(fields["soilw_rel"]) if "soilw_rel" in fields else None),
        forest_fraction=static_fraction("forest"),
        glacier_fraction=static_fraction("glac"),
    )
    return attach_jam_forcing(
        forcing, col_lon, col_lat, nlev=dycore.nlev,
        emissions_file=emissions_file, dms_file=dms_file,
        dust_file=dust_file,
        dust_preferential_file=dust_preferential_file,
        dust_soil_types_file=dust_soil_types_file,
        dust_regions_file=dust_regions_file,
        dust_roughness_file=dust_roughness_file,
        oxidants_file=oxidants_file,
        ozone_file=ozone_file,
        emissions_align=emissions_align,
        oxidants_align=oxidants_align,
        ozone_align=ozone_align,
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


def _mask_to_columns(leaf, lon, lat, col_lon, col_lat):
    """Sample a categorical ``(lon, lat)`` mask onto the columns, nearest-neighbour."""
    from jcm.data.regridding import nearest_index

    arr = np.asarray(leaf)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat, indexing="ij")
    idx = nearest_index(mesh_lat.ravel(), mesh_lon.ravel(), col_lat, col_lon)
    return jnp.asarray(arr.reshape(-1)[idx].reshape(1, col_lon.size))


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
                       dust_preferential_file=None, dust_soil_types_file=None,
                       dust_regions_file=None, dust_roughness_file=None,
                       oxidants_file=None, ozone_file=None,
                       emissions_align="auto", oxidants_align="auto",
                       ozone_align="auto") -> ForcingData:
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

    The ``*_align`` specs follow the one rule every forcing input shares
    (:func:`jcm.forcing.resolve_align`, #884): explicit modes as given,
    ``auto`` only for a data-mirror/packaged product (from its manifest kind),
    an error for any other file. Ozone on this path is climatology-only.
    """
    import xarray as xr

    from jcm.forcing import (
        emissions_have_time,
        resolve_align,
        read_anthropogenic_emissions,
        read_dms_seawater,
        read_dust_preferential,
        read_dust_regions,
        read_dust_roughness,
        read_dust_soil_types,
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
            # One combined open means one time axis, so a per-product list of
            # modes must agree (the runner already rejects mixed axes).
            spec = emissions_align
            if isinstance(spec, (list, tuple)) or (
                    hasattr(spec, "__iter__") and not isinstance(spec, str)):
                modes = {str(v) for v in spec}
                if len(modes) != 1:
                    raise ValueError(
                        f"forcing.emissions_align={list(spec)!r}: the pySES "
                        "path opens every emission product as ONE dataset "
                        "along a shared time axis, so they need one mode.")
                spec = modes.pop()
            # Only a timed product needs (or may ask for) an alignment; an
            # all-static user file loads under ``auto`` (#884).
            align = (resolve_align(spec, paths=paths,
                                   config_key="forcing.emissions_align")
                     if emissions_have_time(ds) else spec)
            anthro = read_anthropogenic_emissions(ds, align_mode=align)
            speciated = read_prescribed_aerosol_emissions(ds, align_mode=align)
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
        missing = [name for name, path in
                   (("dust_preferential_file", dust_preferential_file),
                    ("dust_soil_types_file", dust_soil_types_file),
                    ("dust_regions_file", dust_regions_file)) if path is None]
        if missing:
            raise ValueError(
                f"dust_file is set but {missing} are not. The Tegen scheme "
                "needs the preferential sources, soil textures and tuning "
                "regions alongside the potential-source map; without them it "
                "would emit an untuned, all-coarse-soil flux.")
        with xr.open_dataset(str(dust_file)) as ds:
            lon, lat = _reader_grid(ds)
            forcing = forcing.copy(
                dust_source=to_cols(read_dust_source(ds), lon, lat))
        with xr.open_dataset(str(dust_preferential_file)) as ds:
            lon, lat = _reader_grid(ds)
            forcing = forcing.copy(
                dust_preferential=to_cols(read_dust_preferential(ds), lon, lat))
        with xr.open_dataset(str(dust_soil_types_file)) as ds:
            lon, lat = _reader_grid(ds)
            types = read_dust_soil_types(ds)
            forcing = forcing.copy(
                dust_soil_types={k: to_cols(v, lon, lat)
                                 for k, v in types.items()})
        with xr.open_dataset(str(dust_regions_file)) as ds:
            lon, lat = _reader_grid(ds)
            # Categorical: bilinear sampling would invent "region 3.7", so the
            # column takes the value of the nearest source cell.
            forcing = forcing.copy(
                dust_regions=_mask_to_columns(read_dust_regions(ds), lon, lat,
                                              col_lon, col_lat))
        if dust_roughness_file is not None:
            with xr.open_dataset(str(dust_roughness_file)) as ds:
                lon, lat = _reader_grid(ds)
                forcing = forcing.copy(
                    dust_roughness=to_cols(read_dust_roughness(ds), lon, lat))

    if oxidants_file is not None:
        # ``oxidants_file`` may be a single climatology path or the yearly file
        # list of ONE transient product (a ``{year}`` expansion). The runner
        # (:func:`jcm.runners._resolve_oxidant_paths`) has already expanded,
        # resolved and uniform-time-axis-checked the set, so here it is simply
        # opened together along the shared time axis (``open_mfdataset``,
        # by-coords) — mirroring the emissions handling above and the spectral
        # ``_attach_oxidants``. A scalar path opens directly.
        paths = ([str(p) for p in oxidants_file]
                 if isinstance(oxidants_file, (list, tuple))
                 else [str(oxidants_file)])
        # ``data_vars="minimal"`` so only the time-dependent VMR fields are
        # concatenated: the static ``hyam``/``hybm`` hybrid coefficients carry
        # no time axis and must stay 1-D (the default ``data_vars="all"`` would
        # stack them to a spurious ``(nfiles, mlev)`` that breaks
        # ``read_oxidant_vmr``'s level checks).
        ds = (xr.open_mfdataset(paths, combine="by_coords", data_vars="minimal")
              if len(paths) > 1 else xr.open_dataset(paths[0]))
        with ds:
            lon, lat = _reader_grid(ds)
            # Same resolution as the spectral ``_attach_oxidants``.
            vmr = read_oxidant_vmr(
                ds, nlev=nlev,
                align_mode=resolve_align(oxidants_align, paths=paths,
                                         config_key="forcing.oxidants_align"))
            forcing = forcing.copy(
                oxidant_vmr={k: to_cols(v, lon, lat) for k, v in vmr.items()})

    if ozone_file is not None:
        # Column analogue of ``jcm.runners._attach_ozone``. The file follows
        # the ``jcm.data.bc.interpolate_ozone`` contract (``O3 (time, level,
        # lat, lon)`` mole/mole on the model's levels), so only the
        # horizontal is sampled onto the columns — no exact-grid
        # requirement.
        from jcm.ozone_climatology import OzoneClimatology

        if resolve_align(ozone_align, paths=str(ozone_file),
                         config_key="forcing.ozone_align") != "wrap_year":
            raise ValueError(
                f"forcing.ozone_align={ozone_align!r}: transient ozone is not "
                "supported on the pySES path (the column ozone leaf is a "
                "12-month climatology); declare wrap_year for a climatology.")
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
        # Collapse the backend's (1, ncol) horizontal pair into the single
        # flattened column axis ``OzoneClimatology`` documents:
        # ``(ntime, nlev, ncols)``. Ozone is the exception among the leaves
        # attached here. The others keep (1, ncol) to match the pySES state's
        # horizontal layout, and their consumers ravel or reshape to the
        # term's view -- ``oxidant_field_from_vmr`` explicitly reshapes to
        # ``temperature.shape``, for instance. Ozone's consumers do NOT: the
        # spectral loader already flattens to (ntime, nlev, nlon*nlat), so
        # RRTMGP takes ``chemistry.ozone_vmr`` straight to ``lev_to_col``
        # (a plain ``.T``). ``forcing`` is handed to terms UNFLATTENED by
        # ``_compute_tendencies_columns``, so a leaf that keeps the extra
        # axis reaches radiation as (ncol, 1, nlev) and the per-column slice
        # is (1, nlev) -- which is how an ne30 run died in
        # ``_to_3d_with_filled_halo`` with "Cannot broadcast to shape with
        # fewer dimensions: arr_shape=(1, 47) shape=(47,)".
        cols = np.asarray(cols).reshape(cols.shape[:-2] + (-1,))
        seconds_per_month = 30.4375 * 86400.0            # match from_file
        ts = make_time_series(
            jnp.asarray(cols, dtype=jnp.float32),
            jnp.asarray((np.arange(12) + 0.5) * seconds_per_month,
                        dtype=jnp.float32),
            WRAP_YEAR,
        )
        forcing = forcing.copy(ozone_climatology=OzoneClimatology(o3_ppmv=ts))

    return forcing
