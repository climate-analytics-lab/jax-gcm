"""TerrainData struct for boundary conditions that vary per simulation.

Date: 2026-01-26
"""
import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
# Attribute access (not ``from ... import grav``) so a jcm.constants.set_constants
# override is honoured at call time rather than captured at import.
import jcm.constants as _constants
from jcm.utils import VALID_NODAL_SHAPES, VALID_TRUNCATIONS, validate_ds, spectral_truncation


# ---------------------------------------------------------------------------
# SSO descriptors
# ---------------------------------------------------------------------------
#
# The Lott & Miller (1997) SSO drag scheme requires six per-column statistics
# of the sub-grid orography:
#
#   orostd  — standard deviation of the elevation within the grid cell (m)
#   orosig  — mean slope of the sub-grid orography (dimensionless)
#   orogam  — anisotropy factor (ratio of minor/major axis of the orographic
#             stress ellipse; 0 = pure ridge, 1 = isotropic)
#   orothe  — orientation angle of the principal axis (degrees, measured
#             from east — i.e. ``theta=0`` is a ridge oriented north-south
#             facing zonal flow)
#   oropic  — characteristic peak elevation in the cell (m, above sea level)
#   oroval  — characteristic valley elevation in the cell (m, above sea level)
#
# These come from a high-resolution topography product (GMTED2010 or similar)
# processed onto the model grid by an offline orography preprocessor. When
# real preprocessed data is not available, two fallbacks live in this
# module: :func:`derive_sso_descriptors` computes the Baines-Palmer
# statistics from a high-resolution orography array per target cell, and
# :func:`get_simplified_sso_descriptors` generates rough defaults from
# only the mean orography on the target grid.


def get_simplified_sso_descriptors(orog: jnp.ndarray) -> dict:
    """Generate placeholder SSO descriptors from the mean orography field.

    These are *educated guesses*, not real preprocessed values. The intent
    is to give the Lott-Miller scheme reasonable inputs over land (so the
    scheme exercises its full code path during testing) while letting the
    activation gate (``ppic-pmea > min_peak_minus_mean_elevation`` AND
    ``pstd > min_orog_std``) automatically disable the scheme over ocean
    (``orog == 0``).

    Heuristic:

    - ``orostd ≈ 0.25 * orog`` (sub-grid std-dev as ~25% of mean elevation,
      typical of GMTED2010 in continental regions)
    - ``orosig = 0.1`` over land (typical mid-latitude continental mean
      slope; the scheme uses this as a flat scalar within active columns)
    - ``orogam = 0.5`` (mild anisotropy, no preferred direction)
    - ``orothe = 0`` (zonal-aligned principal axis)
    - ``oropic = orog + 2 * orostd`` (~2σ above mean)
    - ``oroval = max(0, orog - 2 * orostd)`` (~2σ below mean, clamped at 0)

    Use :func:`derive_sso_descriptors` instead when high-resolution
    orography data is available — it computes the Baines-Palmer
    statistics properly per target cell.
    """
    has_orog = orog > 1.0
    orostd = jnp.where(has_orog, 0.25 * orog, 0.0)
    orosig = jnp.where(has_orog, 0.1, 0.0)
    orogam = jnp.where(has_orog, 0.5, 0.0)
    orothe = jnp.zeros_like(orog)
    oropic = jnp.where(has_orog, orog + 2.0 * orostd, 0.0)
    oroval = jnp.where(has_orog, jnp.maximum(orog - 2.0 * orostd, 0.0), 0.0)
    return dict(orostd=orostd, orosig=orosig, orogam=orogam, orothe=orothe,
                oropic=oropic, oroval=oroval)


def derive_sso_descriptors(
    highres_orog: jnp.ndarray,
    highres_lat: jnp.ndarray,
    highres_lon: jnp.ndarray,
    target_lat: jnp.ndarray,
    target_lon: jnp.ndarray,
) -> dict:
    """Compute SSO descriptors per target grid cell from high-res orography.

    Implements the Baines & Palmer (1990) preprocessing: for each
    coarse target cell, gather all high-resolution orography points
    that fall inside it and compute the six sub-grid statistics needed
    by the Lott-Miller drag scheme.

    For each target cell:

    - ``orostd`` — standard deviation of high-res elevation in the cell
    - ``oropic`` — characteristic peak (max elevation in the cell)
    - ``oroval`` — characteristic valley (min elevation in the cell)
    - ``orosig`` — RMS slope ``sqrt(mean(K + L))`` from the high-res
      gradient field, where ``K = mean((dH/dx)^2)`` and
      ``L = mean((dH/dy)^2)``
    - ``orogam`` — gradient-tensor anisotropy
      ``λ_minus / λ_plus``, where ``λ_±`` are the eigenvalues of the
      symmetric tensor ``[[K, M], [M, L]]`` with ``M = mean(dH/dx · dH/dy)``
    - ``orothe`` — orientation of the principal axis (degrees from east),
      ``0.5 * atan2(2*M, K-L)``

    Args:
        highres_orog: high-resolution orography, shape ``(nx_hr, ny_hr)``
            with the lat axis last (matches the convention of the t30
            terrain file). Units: metres above sea level.
        highres_lat: high-resolution latitudes (degrees), shape
            ``(ny_hr,)``.
        highres_lon: high-resolution longitudes (degrees), shape
            ``(nx_hr,)``.
        target_lat: target-grid latitudes (degrees), shape ``(ny,)``.
        target_lon: target-grid longitudes (degrees), shape ``(nx,)``.

    Returns:
        dict with the six SSO descriptor arrays, each of shape
        ``(nx, ny)`` matching the target grid.

    """
    import numpy as np
    H = np.asarray(highres_orog)
    hr_lat = np.asarray(highres_lat)
    hr_lon = np.asarray(highres_lon)
    tg_lat = np.asarray(target_lat)
    tg_lon = np.asarray(target_lon)

    # Build target-cell edges by bisecting between adjacent centres.
    def _edges(centres):
        midpoints = 0.5 * (centres[1:] + centres[:-1])
        return np.concatenate([
            [centres[0] - 0.5 * (centres[1] - centres[0])],
            midpoints,
            [centres[-1] + 0.5 * (centres[-1] - centres[-2])],
        ])

    lat_edges = _edges(tg_lat)
    lon_edges = _edges(tg_lon)

    # Assign each high-res point to a target cell (-1 = outside grid).
    lat_bin = np.searchsorted(lat_edges, hr_lat, side="right") - 1
    lon_bin = np.searchsorted(lon_edges, hr_lon, side="right") - 1
    lat_bin = np.where((lat_bin < 0) | (lat_bin >= len(tg_lat)),
                       -1, lat_bin)
    lon_bin = np.where((lon_bin < 0) | (lon_bin >= len(tg_lon)),
                       -1, lon_bin)

    # High-res gradient field. dh/dx in m/m using metres-per-degree at
    # the equator divided by cos(lat). H is shaped (nx_lon, ny_lat).
    R_earth = 6.371e6
    deg_to_m = R_earth * np.pi / 180.0
    dH_dlon, dH_dlat = np.gradient(H, hr_lon, hr_lat)
    cos_lat = np.cos(np.deg2rad(hr_lat))[None, :]
    dH_dy = dH_dlat / deg_to_m
    dH_dx = dH_dlon / (deg_to_m * np.maximum(cos_lat, 1e-3))

    nx, ny = len(tg_lon), len(tg_lat)
    orostd = np.zeros((nx, ny))
    oropic = np.zeros((nx, ny))
    oroval = np.zeros((nx, ny))
    orosig = np.zeros((nx, ny))
    orogam = np.zeros((nx, ny))
    orothe = np.zeros((nx, ny))

    for j in range(ny):
        in_lat = lat_bin == j
        if not in_lat.any():
            continue
        for i in range(nx):
            mask = in_lat[None, :] & (lon_bin == i)[:, None]
            if not mask.any():
                continue
            elev = H[mask]
            orostd[i, j] = float(np.std(elev))
            oropic[i, j] = float(np.max(elev))
            oroval[i, j] = float(np.min(elev))

            gx = dH_dx[mask]
            gy = dH_dy[mask]
            K = float(np.mean(gx * gx))
            L = float(np.mean(gy * gy))
            M = float(np.mean(gx * gy))
            orosig[i, j] = float(np.sqrt(max(K + L, 0.0)))
            disc = np.sqrt(max(0.25 * (K - L) ** 2 + M * M, 0.0))
            lam_plus = 0.5 * (K + L) + disc
            lam_minus = 0.5 * (K + L) - disc
            orogam[i, j] = float(lam_minus / lam_plus) if lam_plus > 1e-30 else 0.0
            orothe[i, j] = float(0.5 * np.degrees(np.arctan2(2.0 * M, K - L)))

    # Clip to physically-meaningful ranges.
    orogam = np.clip(orogam, 0.0, 1.0)

    return dict(
        orostd=jnp.asarray(orostd), orosig=jnp.asarray(orosig),
        orogam=jnp.asarray(orogam), orothe=jnp.asarray(orothe),
        oropic=jnp.asarray(oropic), oroval=jnp.asarray(oroval),
    )


def get_terrain(orography: jnp.ndarray = None, fmask: jnp.ndarray = None, nodal_shape=None,
                terrain_file=None, fmask_threshold=0.1, grid: HorizontalGridTypes = None):
    """Get the orography data for the model grid. If fmask and/or orography are provided, use them directly
    (defaulting the other to zeros if only one is provided). If terrain_file is provided, load both from file.
    Otherwise, default both to zeros with shape nodal_shape.

    Args:
        orography: Orography height (m) (ix, il). If None but fmask is provided, defaults to zeros (flat).
        fmask: Fractional land-sea mask (ix, il). If None but orography is provided, defaults to zeros (all ocean).
        nodal_shape: Shape of the nodal grid (ix, il). Used when neither fmask, orography, nor terrain_file are provided.
        terrain_file: Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
        target_resolution: Spectral truncation to interpolate the terrain data to, default None (no interpolation).
        fmask_threshold: Threshold for rounding fmask values that are close to 0 or 1.

    Returns:
        Orography height (m) (ix, il)
        Land-sea mask (ix, il)

    """
    # the spectral resolution is total wavenumbers - 2
    target_resolution = grid.total_wavenumbers - 2 if grid is not None else None

    if fmask is None and orography is None:
        if terrain_file is None:
            # if only nodal shape is provided, return zeros of that shape
            if nodal_shape is None:
                raise ValueError("Must provide at least one of: fmask, orography, terrain_file, or nodal_shape.")
            return jnp.zeros(nodal_shape), jnp.zeros(nodal_shape)

        # if only terrain file is provided, set orography and fmask from terrain file
        import xarray as xr
        from jcm.data.bc.interpolate import upsample_terrain_ds
        ds = xr.open_dataset(terrain_file)
        validate_ds(ds, expected_structure={"lsm": ("lon", "lat"), "orog": ("lon", "lat")})
        if target_resolution is not None:
            if target_resolution not in VALID_TRUNCATIONS:
                raise ValueError(f"Invalid target resolution: {target_resolution}. Must be one of: {VALID_TRUNCATIONS}.")
            ds = upsample_terrain_ds(ds, grid=grid)
        else:
            file_shape = (ds.sizes["lon"], ds.sizes["lat"])
            if file_shape not in VALID_NODAL_SHAPES:
                raise ValueError(f"Invalid terrain data shape: {file_shape}. Must be one of: {VALID_NODAL_SHAPES}.")

        # set orography and fmask after upsampling happens
        orography, fmask = jnp.asarray(ds['orog']), jnp.asarray(ds['lsm'])

    elif fmask is None:
        # If orography provided but fmask not, default fmask to any orography > 0
        fmask = (orography > 0.0).astype(float)

    elif orography is None:
        # If fmask provided but orography not, default orography to zeros (flat)
        orography = jnp.zeros_like(fmask)

    # Set values close to 0 or 1 to exactly 0 or 1
    fmask = jnp.where(fmask <= fmask_threshold, 0.0, jnp.where(fmask >= 1.0 - fmask_threshold, 1.0, fmask))

    return orography, fmask


_SSO_NAMES = ("orostd", "orosig", "orogam", "orothe", "oropic", "oroval")


def _load_sso_from_file(terrain_file):
    """Load SSO descriptor fields from a JCM-canonical terrain file if present.

    Returns a dict of the six SSO arrays, or ``None`` if the file lacks any
    of them. Use ``utils/convert_echam_bc.py`` to translate ECHAM-style
    boundary files (uppercase ``OROSTD``/…, ``(lat, lon)``-ordered) into
    the canonical JCM layout that this loader expects.
    """
    import xarray as xr
    ds = xr.open_dataset(terrain_file)
    if not all(name in ds for name in _SSO_NAMES):
        return None
    return {name: jnp.asarray(ds[name]) for name in _SSO_NAMES}


@tree_math.struct
class TerrainData:
    """Boundary conditions that vary per simulation.

    Attributes:
        orog: Mean orography height (m), shape (ix, il)
        phis0: Spectrally truncated surface geopotential, shape (ix, il)
        fmask: Fractional land-sea mask, shape (ix, il)
        lfluxland: Whether to compute land surface fluxes (bool)
        orostd: SSO standard deviation (m), shape (ix, il)
        orosig: SSO mean slope (dimensionless), shape (ix, il)
        orogam: SSO anisotropy factor (dimensionless, 0..1), shape (ix, il)
        orothe: SSO orientation angle (degrees from east), shape (ix, il)
        oropic: SSO peak elevation (m above sea level), shape (ix, il)
        oroval: SSO valley elevation (m above sea level), shape (ix, il)

    The six ``oro*`` fields drive the Lott & Miller (1997) sub-grid
    orographic gravity-wave drag scheme. They normally come from an
    offline preprocessing of high-resolution topography (GMTED2010 etc.);
    when only the mean orography is available,
    :func:`get_simplified_sso_descriptors` generates placeholder values.

    """

    orog: jnp.ndarray
    phis0: jnp.ndarray
    fmask: jnp.ndarray
    lfluxland: jnp.bool_
    orostd: jnp.ndarray
    orosig: jnp.ndarray
    orogam: jnp.ndarray
    orothe: jnp.ndarray
    oropic: jnp.ndarray
    oroval: jnp.ndarray

    def copy(self, orog=None, fmask=None, phis0=None, lfluxland=None,
             orostd=None, orosig=None, orogam=None, orothe=None,
             oropic=None, oroval=None):
        """Copy an instance of TerrainData, replacing the named fields."""
        return TerrainData(
            orog=orog if orog is not None else self.orog,
            phis0=phis0 if phis0 is not None else self.phis0,
            fmask=fmask if fmask is not None else self.fmask,
            lfluxland=lfluxland if lfluxland is not None else self.lfluxland,
            orostd=orostd if orostd is not None else self.orostd,
            orosig=orosig if orosig is not None else self.orosig,
            orogam=orogam if orogam is not None else self.orogam,
            orothe=orothe if orothe is not None else self.orothe,
            oropic=oropic if oropic is not None else self.oropic,
            oroval=oroval if oroval is not None else self.oroval,
        )

    @classmethod
    def from_coords(cls, coords: CoordinateSystem, orography=None, fmask=None, lfluxland=None,
                    terrain_file=None, interpolate=False,
                    orostd=None, orosig=None, orogam=None, orothe=None,
                    oropic=None, oroval=None):
        """Initialize TerrainData from a dinosaur CoordinateSystem.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object.
            orography (optional): Orography height (m), shape (ix, il). If None, defaults to zeros.
            fmask (optional): Fractional land-sea mask, shape (ix, il). If None, defaults to zeros (all ocean).
            lfluxland (optional): Whether to compute land surface fluxes (defaults to False if not provided).
            terrain_file (optional): Path to a file containing orog/lsm (and optionally SSO descriptors).
            interpolate (optional): Whether to interpolate the terrain data (default False).
            orostd, orosig, orogam, orothe, oropic, oroval (optional): SSO
                descriptor arrays, shape (ix, il). If any is provided all
                six should be; missing ones are derived by
                :func:`get_simplified_sso_descriptors`. If none provided and
                ``terrain_file`` includes them, those are used; otherwise
                all six are derived from the mean orography.

        Returns:
            TerrainData object

        """
        # Orography and surface geopotential
        orog, fmask = get_terrain(
            fmask=fmask,
            orography=orography,
            nodal_shape=coords.horizontal.nodal_shape,
            terrain_file=terrain_file,
            grid=coords.horizontal if interpolate else None
        )

        # if the user did not specify lfluxland, and fmask is > 0 anywhere (i.e. there is some land),
        # set lfluxland to True, otherwise set to False if not specified
        if jnp.sum(fmask) > 0 and lfluxland is None:
            lfluxland = True
        elif lfluxland is None:
            lfluxland = False

        phi0 = _constants.grav * orog
        phis0 = spectral_truncation(coords.horizontal, phi0)

        # Resolve SSO descriptors. Order of precedence:
        #   1. Explicit kwargs (any subset; missing ones derived).
        #   2. Fields in the terrain file (if all six present).
        #   3. Derived from the mean orography.
        sso_user = dict(orostd=orostd, orosig=orosig, orogam=orogam,
                        orothe=orothe, oropic=oropic, oroval=oroval)
        sso_user_provided = {k: v for k, v in sso_user.items() if v is not None}

        sso_from_file = (_load_sso_from_file(terrain_file)
                         if terrain_file is not None else None)
        sso_derived = get_simplified_sso_descriptors(orog)

        sso = dict(sso_derived)
        if sso_from_file is not None:
            sso.update(sso_from_file)
        sso.update(sso_user_provided)

        return cls(orog=orog, phis0=phis0, fmask=fmask,
                   lfluxland=jnp.bool_(lfluxland), **sso)

    @classmethod
    def from_file(cls, terrain_file, coords: CoordinateSystem, lfluxland=True,
                  orog_envelope_wavenumber: int = None):
        """Initialize TerrainData from a JCM-canonical terrain file.

        Expects the canonical layout: lowercase variables (``orog``,
        ``lsm``, optional ``orostd``/…), ``(lon, lat)`` axis order,
        ascending latitudes. ECHAM-style files (uppercase ``OROMEA``,
        ``(lat, lon)``-ordered) must be pre-converted with
        ``utils/convert_echam_bc.py``.

        SSO descriptor handling, in order of precedence:

        1. If the file contains all six SSO fields (``orostd``,
           ``orosig``, ``orogam``, ``orothe``, ``oropic``, ``oroval``),
           those are loaded directly.
        2. Else, if the file's orography is at higher resolution than
           the target grid, the descriptors are derived from the
           high-resolution orography using :func:`derive_sso_descriptors`
           (Baines-Palmer statistics per target cell).
        3. Otherwise the simplified heuristic
           :func:`get_simplified_sso_descriptors` is applied to the mean
           orography on the target grid.

        Args:
            terrain_file: Path to a file containing ``orog`` (orography)
                and ``lsm`` (land-sea mask). May optionally contain the
                six SSO descriptor fields.
            coords: dinosaur.coordinate_systems.CoordinateSystem object.
            lfluxland: Whether to compute land surface fluxes
                (default True).

        Returns:
            TerrainData object.

        """
        import xarray as xr
        target_grid = coords.horizontal
        target_shape = target_grid.nodal_shape

        # Read raw orography first to inspect its source resolution.
        with xr.open_dataset(terrain_file) as raw_ds:
            src_lat_n = raw_ds.sizes.get("lat", 0)
            src_lon_n = raw_ds.sizes.get("lon", 0)
            src_lat = jnp.asarray(raw_ds["lat"].values)
            src_lon = jnp.asarray(raw_ds["lon"].values)
            src_orog = jnp.asarray(raw_ds["orog"].values)
            src_has_sso = all(name in raw_ds for name in _SSO_NAMES)

        # Load + interpolate orog/lsm onto the target grid (existing path).
        orography, fmask = get_terrain(terrain_file=terrain_file, grid=target_grid)
        if orography.shape != target_shape:
            raise ValueError(
                f"Terrain shape {orography.shape} does not match coords "
                f"horizontal shape {target_shape}"
            )
        phi0 = _constants.grav * orography
        if orog_envelope_wavenumber is not None:
            # Envelope orography: low-pass filter the orographic geopotential
            # to a lower spectral wavenumber than the model truncation, to
            # suppress Gibbs oscillations near sharp coast/mountain edges
            # that would otherwise produce negative effective elevations.
            modal = target_grid.to_modal(phi0)
            nx, mx = modal.shape
            n_idx, m_idx = jnp.meshgrid(jnp.arange(nx), jnp.arange(mx),
                                        indexing='ij')
            total_wn = m_idx + n_idx
            modal = jnp.where(total_wn > orog_envelope_wavenumber, 0.0, modal)
            phis0 = target_grid.to_nodal(modal)
        else:
            phis0 = spectral_truncation(target_grid, phi0)

        # Pick the best available SSO source.
        if src_has_sso:
            sso = _load_sso_from_file(terrain_file)
        elif src_lat_n > target_shape[1] and src_lon_n > target_shape[0]:
            target_lat_deg = target_grid.latitudes * 180.0 / jnp.pi
            target_lon_deg = target_grid.longitudes * 180.0 / jnp.pi
            sso = derive_sso_descriptors(
                src_orog, src_lat, src_lon,
                target_lat_deg, target_lon_deg,
            )
        else:
            sso = get_simplified_sso_descriptors(orography)

        return cls(orog=orography, phis0=phis0, fmask=fmask,
                   lfluxland=jnp.bool_(lfluxland), **sso)

    @classmethod
    def aquaplanet(cls, coords: CoordinateSystem):
        """Initialize an aquaplanet TerrainData (flat, all ocean, no land fluxes).

        All SSO descriptors are zero — the Lott-Miller activation gate
        (``ppic-pmea > gpicmea`` AND ``pstd > gstd``) leaves the SSO scheme
        inactive throughout.

        Args:
            coords: dinosaur.coordinate_systems.CoordinateSystem object.

        Returns:
            TerrainData object with all zeros for orography and fmask.

        """
        nodal_shape = coords.horizontal.nodal_shape
        zero = jnp.zeros(nodal_shape)
        return cls(
            orog=zero, phis0=zero, fmask=zero,
            lfluxland=jnp.bool_(False),
            orostd=zero, orosig=zero, orogam=zero,
            orothe=zero, oropic=zero, oroval=zero,
        )

    @classmethod
    def single_column(cls, orog=0., fmask=0., phis0=None, lfluxland=False,
                      orostd=None, orosig=None, orogam=None, orothe=None,
                      oropic=None, oroval=None):
        """Initialize a TerrainData instance for a single column model.

        Any SSO descriptor not explicitly provided is derived from
        ``orog`` via :func:`get_simplified_sso_descriptors`.

        Args:
            orog (optional): Orography height in meters (default 0).
            fmask (optional): Fractional land-sea mask (default 0, all ocean).
            phis0 (optional): Spectrally truncated surface geopotential (default grav * orog).
            lfluxland (optional): Whether to compute land surface fluxes (default False).
            orostd, orosig, orogam, orothe, oropic, oroval (optional):
                Scalar SSO descriptors. Each defaults to the heuristic
                derivation from ``orog``.

        Returns:
            TerrainData object

        """
        if phis0 is None:
            phis0 = _constants.grav * orog

        orog_arr = jnp.array([[orog]])
        sso_derived = get_simplified_sso_descriptors(orog_arr)
        def _pick(name, value):
            if value is None:
                return sso_derived[name]
            return jnp.array([[value]])

        return cls(
            orog=orog_arr,
            phis0=jnp.array([[phis0]]),
            fmask=jnp.array([[fmask]]),
            lfluxland=jnp.bool_(lfluxland),
            orostd=_pick("orostd", orostd),
            orosig=_pick("orosig", orosig),
            orogam=_pick("orogam", orogam),
            orothe=_pick("orothe", orothe),
            oropic=_pick("oropic", oropic),
            oroval=_pick("oroval", oroval),
        )
