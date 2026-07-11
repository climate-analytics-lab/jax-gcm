"""Virtual observation operators: sample model fields at obs locations/times.

For comparison with in-situ or satellite measurements the model can sample any
diagnostic at a set of (latitude, longitude, altitude, time) points **every
model timestep** — far finer than the ``save_interval`` cadence of the regular
gridded output. Three samplers are provided:

- :class:`TrackObserver` — a moving platform (ship, aircraft) described by a
  time-ordered table of positions, or a set of fixed stations
  (:meth:`TrackObserver.stations`).
- :class:`LocalSolarTimeObserver` — a sun-synchronous-satellite proxy that
  samples the longitude at which the local solar time equals a fixed overpass
  hour (the band sweeps westward at 15°/hour), at a set of latitudes.
- :class:`Observer` — the base class; subclass and override
  :meth:`Observer._positions_for_times` for other sampling geometries.

Design (see ``docs/source/design/observers.md``):

- **Time**: platform positions are resampled onto the model's ``dt`` grid
  offline (numpy, at ``prepare`` time), so the scan samples one position per
  platform per timestep. Interpolating the resulting dt-resolution series onto
  exact observation times is cheap xarray post-processing.
- **Horizontal**: interpolation weights are precomputed offline and cached —
  true bilinear on separable (Gaussian lat × uniform lon) grids, k-nearest
  inverse-distance on unstructured column grids. Only the vertical
  interpolation is state-dependent and runs inside the step.
- **Vertical**: linear in geometric height against the sampled ``z_full``
  profile (``vertical="altitude"``), linear in log-pressure
  (``vertical="pressure"``), no interpolation (``vertical="profile"`` returns
  whole columns), or 2-D fields only (``vertical="surface"``).
- **Variables** come from the physics diagnostics dict: top-level diagnostic
  keys (e.g. ``"cloud_fraction"``), dotted sub-struct fields (e.g.
  ``"radiation.tsr"``), and — via the :class:`~jcm.physics.diagnostics.
  state_sampler.StateSampler` term that :class:`jcm.model.Model` appends
  automatically — the state fields ``temperature``, ``u_wind``, ``v_wind``,
  ``specific_humidity``, ``surface_pressure``, ``z_full``, ``p_full`` and
  every entry of ``state.tracers`` by name.

The whole operator is differentiable: gradients flow from sampled values back
through the interpolation weights into the model state and physics parameters,
so an observer doubles as the observation operator H(x) for data assimilation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import jax
import jax.numpy as jnp

_EPOCH = np.datetime64("1970-01-01", "ns")
_DAY = np.timedelta64(1, "D") / np.timedelta64(1, "ns")

_VERTICAL_MODES = ("altitude", "pressure", "surface", "profile")


def _times_to_days(times) -> np.ndarray:
    """Convert datetime64-like or float-day times to float days since 1970.

    Float inputs are passed through unchanged (interpreted as days since
    1970-01-01, the same axis :meth:`ModelPredictions.to_xarray` uses).
    """
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float64)
    arr = arr.astype("datetime64[ns]")
    return (arr - _EPOCH) / np.timedelta64(1, "D")


def _days_to_datetime64(t_days: np.ndarray) -> np.ndarray:
    return (np.asarray(t_days) * _DAY).astype("int64").view("datetime64[ns]")


def _to_degrees(values: np.ndarray, max_radians: float) -> np.ndarray:
    """Return ``values`` in degrees, sniffing radians vs degrees.

    Grid latitude/longitude arrays are radians in the dinosaur backend but
    degrees in some others. Any real grid has coordinates whose magnitude
    exceeds the radian bound (π/2 for latitudes, 2π for longitudes), so the
    unit is decidable from the data.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.max(np.abs(values)) > max_radians * (1.0 + 1e-6):
        return values
    return np.degrees(values)


class Observer:
    """Base class: samples ``variables`` at per-timestep points.

    Subclasses provide the sampling geometry by overriding
    :meth:`_positions_for_times`; everything else (weight construction,
    in-scan sampling, xarray assembly) lives here.

    Args:
        name: Key for this observer in the output (must be unique among the
            observers attached to one model).
        variables: Diagnostic names to sample (see module docstring for the
            name resolution rules).
        vertical: One of ``"altitude"`` (interpolate to a target height above
            the geoid, in m), ``"pressure"`` (target in Pa, linear in log-p),
            ``"surface"`` (2-D fields only), ``"profile"`` (whole columns,
            no vertical interpolation).
        k_neighbors: Number of nearest columns for the unstructured-grid
            inverse-distance path. Ignored on separable grids (bilinear, 4
            corners).

    """

    def __init__(self, name: str, variables, vertical: str = "altitude",
                 k_neighbors: int = 4):
        """Validate and store the observer configuration (see class docstring)."""
        if vertical not in _VERTICAL_MODES:
            raise ValueError(
                f"vertical={vertical!r} not in {_VERTICAL_MODES}")
        self.name = str(name)
        self.variables = tuple(variables)
        if not self.variables:
            raise ValueError("Observer needs at least one variable to sample.")
        self.vertical = vertical
        self.k_neighbors = int(k_neighbors)
        self._grid_cached = False

    # ------------------------------------------------------------------
    # Grid caching (once, at Model construction)
    # ------------------------------------------------------------------

    def cache_grid(self, coords) -> None:
        """Cache static horizontal-grid metadata used to build weights.

        Two layouts are supported:

        - *Separable* (dinosaur): 1-D ``horizontal.latitudes`` ×
          ``horizontal.longitudes``; true bilinear weights.
        - *Unstructured* (e.g. pySES physics columns): per-column
          ``horizontal.column_latitudes`` / ``column_longitudes``;
          k-nearest-neighbour inverse-great-circle-distance weights.
        """
        horizontal = coords.horizontal
        self._nodal_shape = tuple(int(s) for s in horizontal.nodal_shape)
        self._ncols = int(np.prod(self._nodal_shape))

        col_lat = getattr(horizontal, "column_latitudes", None)
        col_lon = getattr(horizontal, "column_longitudes", None)
        if col_lat is not None and col_lon is not None:
            self._separable = False
            self._col_lat = _to_degrees(np.ravel(col_lat), np.pi / 2)
            self._col_lon = _to_degrees(np.ravel(col_lon), 2 * np.pi) % 360.0
        else:
            self._separable = True
            self._grid_lat = _to_degrees(horizontal.latitudes, np.pi / 2)
            self._grid_lon = (
                _to_degrees(horizontal.longitudes, 2 * np.pi) % 360.0
            )
            # nodal_shape is (nlon, nlat); the physics column flatten is
            # lon-major (C-order over the trailing two axes), so the flat
            # column index of (ilon, ilat) is ilon * nlat + ilat.
            nlon, nlat = self._nodal_shape
            if len(self._grid_lon) != nlon or len(self._grid_lat) != nlat:
                raise ValueError(
                    f"Grid shape mismatch: nodal_shape={self._nodal_shape} vs "
                    f"{len(self._grid_lon)} longitudes / "
                    f"{len(self._grid_lat)} latitudes.")
            # Latitudes may run either south→north or north→south; keep a
            # sorting permutation so searchsorted sees an ascending array.
            self._lat_order = np.argsort(self._grid_lat)
            self._lat_sorted = self._grid_lat[self._lat_order]
        self._grid_cached = True

    # ------------------------------------------------------------------
    # Geometry (override in subclasses)
    # ------------------------------------------------------------------

    def _positions_for_times(self, t_days: np.ndarray):
        """Return per-step sampling positions.

        Args:
            t_days: (n_steps,) absolute times in days since 1970 at which the
                physics sees the state (start of each ``dt``).

        Returns:
            Tuple ``(lat_deg, lon_deg, target, valid)`` of numpy arrays, each
            shaped ``(n_steps, n_points)``. ``target`` is the vertical target
            (height in m or pressure in Pa, per ``self.vertical``; NaN/ignored
            for surface/profile modes). ``valid`` marks samples inside the
            observation window; invalid samples are NaN in the output.

        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight construction (offline numpy, cached per run window)
    # ------------------------------------------------------------------

    def _horizontal_weights(self, lat_deg: np.ndarray, lon_deg: np.ndarray):
        """Flat column indices + weights for points ``(lat_deg, lon_deg)``.

        Input arrays share any shape ``(...,)``; returns int/float arrays of
        shape ``(..., k)`` such that ``sum_k w[..., k] * field[idx[..., k]]``
        is the horizontally interpolated field value.
        """
        if not self._grid_cached:
            raise RuntimeError(
                "Observer.cache_grid was never called — attach the observer "
                "to a Model (observers=[...]) before preparing it.")
        if self._separable:
            return self._bilinear_weights(lat_deg, lon_deg)
        return self._knn_weights(lat_deg, lon_deg)

    def _bilinear_weights(self, lat_deg, lon_deg):
        nlon, nlat = self._nodal_shape
        lon = np.asarray(lon_deg, dtype=np.float64) % 360.0
        lat = np.asarray(lat_deg, dtype=np.float64)

        # Longitude: uniform spacing with periodic wrap.
        lon0 = self._grid_lon[0]
        dlon = 360.0 / nlon
        x = (lon - lon0) / dlon
        i0 = np.floor(x).astype(np.int64)
        f = x - i0
        i0 = i0 % nlon
        i1 = (i0 + 1) % nlon

        # Latitude: non-uniform (Gaussian); poleward of the first/last ring
        # collapses onto that ring (weight 1) — constant extrapolation.
        j = np.searchsorted(self._lat_sorted, lat)
        j0s = np.clip(j - 1, 0, nlat - 1)
        j1s = np.clip(j, 0, nlat - 1)
        denom = self._lat_sorted[j1s] - self._lat_sorted[j0s]
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(denom > 0,
                         (lat - self._lat_sorted[j0s]) / np.where(denom > 0, denom, 1.0),
                         0.0)
        g = np.clip(g, 0.0, 1.0)
        j0 = self._lat_order[j0s]
        j1 = self._lat_order[j1s]

        idx = np.stack([i0 * nlat + j0, i0 * nlat + j1,
                        i1 * nlat + j0, i1 * nlat + j1], axis=-1)
        w = np.stack([(1 - f) * (1 - g), (1 - f) * g,
                      f * (1 - g), f * g], axis=-1)
        return idx, w

    def _knn_weights(self, lat_deg, lon_deg):
        k = self.k_neighbors
        shape = np.shape(lat_deg)
        lat = np.deg2rad(np.ravel(lat_deg))
        lon = np.deg2rad(np.ravel(lon_deg))
        glat = np.deg2rad(self._col_lat)
        glon = np.deg2rad(self._col_lon)

        idx = np.empty((lat.size, k), dtype=np.int64)
        w = np.empty((lat.size, k), dtype=np.float64)
        # Chunk over points: the (npts, ncols) distance matrix for a long
        # swath × ne30 grid would otherwise be several GB.
        chunk = max(1, int(2**24 // max(glat.size, 1)))
        for start in range(0, lat.size, chunk):
            sl = slice(start, min(start + chunk, lat.size))
            # Great-circle central angle via the haversine form (stable for
            # small separations, which is all that matters for neighbours).
            dphi = lat[sl, None] - glat[None, :]
            dlam = lon[sl, None] - glon[None, :]
            h = (np.sin(dphi / 2) ** 2
                 + np.cos(lat[sl, None]) * np.cos(glat[None, :])
                 * np.sin(dlam / 2) ** 2)
            d = 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
            nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
            dk = np.take_along_axis(d, nearest, axis=1)
            # Inverse-distance weights; an exact hit gets all the weight.
            eps = 1e-12
            inv = 1.0 / (dk + eps)
            wk = inv / inv.sum(axis=1, keepdims=True)
            hit = dk < 1e-10
            wk = np.where(hit.any(axis=1, keepdims=True),
                          hit.astype(np.float64), wk)
            idx[sl] = nearest
            w[sl] = wk
        return idx.reshape(shape + (k,)), w.reshape(shape + (k,))

    def prepare(self, t0_days: float, dt_seconds: float, n_steps: int) -> dict:
        """Build the per-timestep sampling tables for one integration window.

        Called by the :class:`~jcm.model.Model` (outside jit) before each
        ``run``/``resume`` window; the result is fed through the integration
        scan as ``xs``. All the horizontal geometry is resolved here, offline;
        only the vertical interpolation remains state-dependent.

        Args:
            t0_days: Absolute start time (days since 1970) of the window.
            dt_seconds: Model timestep.
            n_steps: Number of ``dt`` steps in the window.

        Returns:
            Dict of jnp arrays: ``idx``/``w`` ``(n_steps, npts, k)``,
            ``target`` ``(n_steps, npts)``, ``valid`` ``(n_steps, npts)``.

        """
        t_days = t0_days + np.arange(n_steps) * (dt_seconds / 86400.0)
        lat, lon, target, valid = self._positions_for_times(t_days)
        idx, w = self._horizontal_weights(lat, lon)
        # Fold the validity mask into the weights so masked-out steps gather
        # finite zeros (the NaN marker is applied to the sampled value, not
        # the weights, keeping gradients clean).
        w = w * valid[..., None]
        return {
            "idx": jnp.asarray(idx, dtype=jnp.int32),
            "w": jnp.asarray(w, dtype=jnp.float32),
            "target": jnp.asarray(np.nan_to_num(target), dtype=jnp.float32),
            "valid": jnp.asarray(valid),
        }

    # ------------------------------------------------------------------
    # In-scan sampling (traced)
    # ------------------------------------------------------------------

    def _columnize(self, value: jnp.ndarray, name: str) -> jnp.ndarray:
        """Reshape a diagnostic to the flat column layout.

        Accepts both the 3-D gridpoint layout ``(..., nlon, nlat)`` and the
        column-vectorized layout ``(..., ncols)``; returns ``(ncols,)`` for
        surface fields or ``(nlev, ncols)`` for level fields.
        """
        nlon, nlat = self._nodal_shape
        if value.ndim >= 2 and value.shape[-2:] == (nlon, nlat):
            value = value.reshape(value.shape[:-2] + (self._ncols,))
        if value.ndim in (1, 2) and value.shape[-1] == self._ncols:
            return value
        raise ValueError(
            f"Observer {self.name!r}: variable {name!r} has shape "
            f"{value.shape}, which is neither (…, {nlon}, {nlat}) nor "
            f"(…, {self._ncols}); fields with extra trailing axes (e.g. "
            "per-band optics) are not sampleable.")

    def _resolve(self, diagnostics: dict, name: str) -> jnp.ndarray:
        """Look up ``name`` in the physics diagnostics dict.

        Resolution order: the ``_sampler_state`` dict written by the
        ``StateSampler`` term (state fields, then tracers by name), top-level
        diagnostic keys, then dotted ``struct.field`` sub-struct access.
        """
        sampler_state = diagnostics.get("_sampler_state")
        if isinstance(sampler_state, dict):
            if name in sampler_state and name != "tracers":
                return sampler_state[name]
            tracers = sampler_state.get("tracers")
            if isinstance(tracers, dict) and name in tracers:
                return tracers[name]
        value = diagnostics.get(name)
        if isinstance(value, jax.Array):
            return value
        if "." in name:
            root, field = name.split(".", 1)
            struct = diagnostics.get("_" + root, diagnostics.get(root))
            if struct is not None and hasattr(struct, field):
                return getattr(struct, field)
        available = sorted(
            k for k, v in diagnostics.items() if isinstance(v, jax.Array))
        raise KeyError(
            f"Observer {self.name!r}: variable {name!r} not found in the "
            f"physics diagnostics. Top-level array keys: {available}. State "
            "fields (temperature, u_wind, …, z_full, p_full, tracer names) "
            "are available when the StateSampler term is active; sub-struct "
            "fields via dotted names like 'radiation.tsr'.")

    @staticmethod
    def _gather(columns: jnp.ndarray, idx: jnp.ndarray,
                w: jnp.ndarray) -> jnp.ndarray:
        """Horizontal interpolation: weighted gather of neighbour columns."""
        neighbours = columns[..., idx]  # (npts, k) or (nlev, npts, k)
        return (neighbours * w.astype(columns.dtype)).sum(axis=-1)

    def sample(self, diagnostics: dict, xs: dict) -> dict:
        """Sample all variables for one timestep. Runs inside the scan.

        Args:
            diagnostics: The post-step physics diagnostics dict.
            xs: One timestep's slice of the :meth:`prepare` tables.

        Returns:
            Dict ``{variable: (npts,)}`` (or ``(nlev, npts)`` in profile
            mode). Samples outside the observation window are NaN.

        """
        idx, w, valid = xs["idx"], xs["w"], xs["valid"]
        coord_profile, target = None, None
        if self.vertical == "altitude":
            z = self._columnize(self._resolve(diagnostics, "z_full"), "z_full")
            coord_profile = self._gather(z, idx, w)
            target = xs["target"]
        elif self.vertical == "pressure":
            p = self._columnize(self._resolve(diagnostics, "p_full"), "p_full")
            # Log-pressure interpolation; the floor guards log(0) at a
            # pure-sigma model top where the target may sit above the grid.
            coord_profile = jnp.log(
                jnp.maximum(self._gather(p, idx, w), 1e-3))
            target = jnp.log(jnp.maximum(xs["target"], 1e-3))

        out = {}
        for name in self.variables:
            columns = self._columnize(self._resolve(diagnostics, name), name)
            value = self._gather(columns, idx, w)
            if columns.ndim == 2:
                if self.vertical == "surface":
                    raise ValueError(
                        f"Observer {self.name!r}: {name!r} is a level field "
                        "but vertical='surface' only samples 2-D fields. Use "
                        "vertical='altitude', 'pressure' or 'profile'.")
                if self.vertical in ("altitude", "pressure"):
                    value = _interpolate_columns(
                        target, coord_profile, value,
                        increasing=(self.vertical == "pressure"))
            mask = valid if value.ndim == 1 else valid[None, :]
            out[name] = jnp.where(mask, value, jnp.nan)
        return out

    # ------------------------------------------------------------------
    # Output assembly (post-run, numpy/xarray)
    # ------------------------------------------------------------------

    def to_dataset(self, samples: dict, t0_days: float,
                   dt_seconds: float) -> xr.Dataset:
        """Convert stacked per-step samples into an ``xarray.Dataset``.

        Args:
            samples: Dict ``{variable: (n_steps, npts)}`` (profile mode:
                ``(n_steps, nlev, npts)``) as returned by the run.
            t0_days / dt_seconds: The window the samples were taken over
                (recorded on :class:`~jcm.predictions.ModelPredictions`).

        """
        first = np.asarray(next(iter(samples.values())))
        n_steps = first.shape[0]
        t_days = t0_days + np.arange(n_steps) * (dt_seconds / 86400.0)
        lat, lon, target, valid = self._positions_for_times(t_days)

        data_vars = {}
        for name, arr in samples.items():
            arr = np.asarray(arr)
            dims = (("time", "point") if arr.ndim == 2
                    else ("time", "level", "point"))
            # Dots in dotted sub-struct names are xarray-hostile; flatten.
            data_vars[name.replace(".", "_")] = (dims, arr)

        coords = {
            "time": ("time", _days_to_datetime64(t_days)),
            "latitude": (("time", "point"), lat),
            "longitude": (("time", "point"), lon),
            "valid": (("time", "point"), valid),
        }
        if self.vertical == "altitude":
            coords["altitude"] = (("time", "point"), target)
        elif self.vertical == "pressure":
            coords["pressure"] = (("time", "point"), target)
        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["observer"] = self.name
        ds.attrs["vertical"] = self.vertical
        return ds


def _interpolate_columns(target, coord_profile, field_profile, increasing):
    """Per-point linear vertical interpolation.

    The physics-internal level axis 0 is **top-first** (index 0 = model top),
    so height decreases and pressure increases along axis 0. ``jnp.interp``
    needs ascending abscissae; flip for height. Out-of-range targets clamp to
    the end values (constant extrapolation).

    Args:
        target: (npts,) vertical targets (m, or log-Pa for pressure mode).
        coord_profile: (nlev, npts) vertical coordinate at the point.
        field_profile: (nlev, npts) field values at the point.
        increasing: Whether ``coord_profile`` ascends along axis 0.

    """
    if not increasing:
        coord_profile = coord_profile[::-1]
        field_profile = field_profile[::-1]
    return jax.vmap(jnp.interp, in_axes=(0, 1, 1))(
        target.astype(field_profile.dtype),
        coord_profile.astype(field_profile.dtype),
        field_profile,
    )


class TrackObserver(Observer):
    """Sample along a moving-platform track, or at fixed stations.

    Positions between the supplied track waypoints are linearly interpolated
    onto the model's timestep grid (longitudes are unwrapped first, so tracks
    crossing the dateline interpolate the short way round). Timesteps outside
    the track's time span are masked (NaN in the output).
    """

    def __init__(self, times, latitudes, longitudes, *, variables,
                 altitudes=None, pressures=None, name: str = "track",
                 vertical: str | None = None, k_neighbors: int = 4):
        """Build a moving-platform observer from waypoint arrays.

        Args:
            times: (n,) waypoint times (datetime64-like, pandas timestamps,
                or float days since 1970). Must be increasing.
            latitudes / longitudes: (n,) waypoint positions in degrees.
            variables: Diagnostic names to sample.
            altitudes: (n,) heights above the geoid in m (implies
                ``vertical="altitude"``).
            pressures: (n,) pressures in Pa (implies ``vertical="pressure"``).
            name: Output key for this observer.
            vertical: Override the vertical mode (e.g. ``"profile"`` to keep
                whole columns along the track).
            k_neighbors: Unstructured-grid neighbour count.

        """
        if altitudes is not None and pressures is not None:
            raise ValueError("Pass altitudes or pressures, not both.")
        if vertical is None:
            vertical = ("pressure" if pressures is not None
                        else "altitude" if altitudes is not None
                        else "surface")
        super().__init__(name=name, variables=variables, vertical=vertical,
                         k_neighbors=k_neighbors)

        t = _times_to_days(times)
        if t.ndim != 1 or t.size < 1:
            raise ValueError("times must be a 1-D array of waypoints.")
        if np.any(np.diff(t) < 0):
            raise ValueError("Track times must be non-decreasing.")
        self._t = t
        self._lat = np.asarray(latitudes, dtype=np.float64)
        self._lon = np.asarray(longitudes, dtype=np.float64) % 360.0
        target = altitudes if altitudes is not None else pressures
        self._target = (np.asarray(target, dtype=np.float64)
                        if target is not None else np.full_like(t, np.nan))
        if not (self._lat.shape == self._lon.shape == t.shape
                == self._target.shape):
            raise ValueError("times/latitudes/longitudes/altitudes must all "
                             "have the same length.")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, variables,
                       time="time", latitude="latitude",
                       longitude="longitude", altitude="altitude",
                       pressure="pressure", **kwargs) -> "TrackObserver":
        """Build from a pandas DataFrame with time/lat/lon(/alt) columns."""
        alt = df[altitude].to_numpy() if altitude in df else None
        pres = (df[pressure].to_numpy()
                if alt is None and pressure in df else None)
        return cls(df[time].to_numpy(), df[latitude].to_numpy(),
                   df[longitude].to_numpy(), variables=variables,
                   altitudes=alt, pressures=pres, **kwargs)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, *, variables, **kwargs) -> "TrackObserver":
        """Build from an xarray Dataset with 1-D time/latitude/longitude."""
        return cls.from_dataframe(
            ds.to_dataframe().reset_index(), variables=variables, **kwargs)

    @classmethod
    def stations(cls, latitudes, longitudes, *, variables, altitudes=None,
                 pressures=None, name: str = "stations",
                 vertical: str | None = None,
                 k_neighbors: int = 4) -> "TrackObserver":
        """Build an observer for fixed stations, sampled at every timestep.

        A separate fast path: one observer holds all the stations (the point
        axis), positions are constant, and every timestep is valid.
        """
        if altitudes is not None and pressures is not None:
            raise ValueError("Pass altitudes or pressures, not both.")
        obs = cls.__new__(cls)
        if vertical is None:
            vertical = ("pressure" if pressures is not None
                        else "altitude" if altitudes is not None
                        else "surface")
        Observer.__init__(obs, name=name, variables=variables,
                          vertical=vertical, k_neighbors=k_neighbors)
        obs._t = None  # marks station mode
        obs._lat = np.atleast_1d(np.asarray(latitudes, dtype=np.float64))
        obs._lon = np.atleast_1d(
            np.asarray(longitudes, dtype=np.float64)) % 360.0
        target = altitudes if altitudes is not None else pressures
        obs._target = (np.atleast_1d(np.asarray(target, dtype=np.float64))
                       if target is not None
                       else np.full_like(obs._lat, np.nan))
        if not (obs._lat.shape == obs._lon.shape == obs._target.shape):
            raise ValueError("latitudes/longitudes/altitudes must all have "
                             "the same length.")
        return obs

    def _positions_for_times(self, t_days: np.ndarray):
        n_steps = t_days.size
        if self._t is None:  # fixed stations
            npts = self._lat.size
            tile = lambda a: np.broadcast_to(a, (n_steps, npts)).copy()
            return (tile(self._lat), tile(self._lon), tile(self._target),
                    np.ones((n_steps, npts), dtype=bool))
        # Moving platform: one point, positions interpolated to step times.
        lon_unwrapped = np.unwrap(self._lon, period=360.0)
        lat = np.interp(t_days, self._t, self._lat)
        lon = np.interp(t_days, self._t, lon_unwrapped) % 360.0
        target = np.interp(t_days, self._t, self._target)
        valid = (t_days >= self._t[0]) & (t_days <= self._t[-1])
        return (lat[:, None], lon[:, None], target[:, None], valid[:, None])


class LocalSolarTimeObserver(Observer):
    """Sun-synchronous-overpass proxy: sample where local solar time is fixed.

    At every model timestep the longitude whose mean local solar time equals
    ``local_solar_hour`` is sampled at each of ``latitudes`` — the band sweeps
    westward through the day exactly like a sun-synchronous satellite's
    ascending (or descending) node, giving an A-Train-style curtain through
    the model at the overpass hour.
    """

    def __init__(self, *, variables, latitudes=None,
                 local_solar_hour: float = 13.5, altitude: float = None,
                 pressure: float = None, name: str = "solar_swath",
                 vertical: str | None = None, k_neighbors: int = 4):
        """Configure the swath.

        Args:
            variables: Diagnostic names to sample.
            latitudes: (npts,) latitudes of the curtain in degrees. Default:
                every grid latitude ring (separable grids only).
            local_solar_hour: Overpass mean local solar time in hours (13.5 ≈
                the A-Train ascending node).
            altitude / pressure: Optional fixed vertical target for scalar
                sampling; default is ``vertical="profile"`` (whole columns)
                or ``"surface"`` when only 2-D fields are requested.
            name: Output key.
            vertical: Explicit vertical-mode override.
            k_neighbors: Unstructured-grid neighbour count.

        """
        if vertical is None:
            if pressure is not None:
                vertical = "pressure"
            elif altitude is not None:
                vertical = "altitude"
            else:
                vertical = "profile"
        super().__init__(name=name, variables=variables, vertical=vertical,
                         k_neighbors=k_neighbors)
        self.local_solar_hour = float(local_solar_hour)
        self._target_value = (pressure if pressure is not None
                              else altitude if altitude is not None
                              else np.nan)
        self._latitudes = (np.atleast_1d(np.asarray(latitudes, np.float64))
                           if latitudes is not None else None)

    def cache_grid(self, coords) -> None:
        """Cache grid metadata; default the curtain latitudes to grid rings."""
        super().cache_grid(coords)
        if self._latitudes is None:
            if not self._separable:
                raise ValueError(
                    "LocalSolarTimeObserver on an unstructured grid needs "
                    "explicit latitudes=[...].")
            self._latitudes = np.sort(self._grid_lat)

    def _positions_for_times(self, t_days: np.ndarray):
        utc_hours = (t_days % 1.0) * 24.0
        # Mean local solar time at longitude L (deg east): LST = UTC + L/15.
        lon = ((self.local_solar_hour - utc_hours) * 15.0) % 360.0
        npts = self._latitudes.size
        n_steps = t_days.size
        lat = np.broadcast_to(self._latitudes, (n_steps, npts)).copy()
        lon = np.broadcast_to(lon[:, None], (n_steps, npts)).copy()
        target = np.full((n_steps, npts), self._target_value)
        valid = np.ones((n_steps, npts), dtype=bool)
        return lat, lon, target, valid
