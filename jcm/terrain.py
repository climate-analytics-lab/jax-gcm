"""TerrainData struct for boundary conditions that vary per simulation.

Date: 2026-01-26
"""
import jax.numpy as jnp
import tree_math
from dinosaur.coordinate_systems import CoordinateSystem, HorizontalGridTypes
from jcm.constants import grav
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
# real preprocessed data is not available, :func:`derive_sso_descriptors`
# generates physically-sensible defaults from the mean orography alone
# (intended as a placeholder until the user supplies a real SSO dataset).


def derive_sso_descriptors(orog: jnp.ndarray) -> dict:
    """Generate placeholder SSO descriptors from the mean orography field.

    These are *educated guesses*, not real preprocessed values. The intent
    is to give the Lott-Miller scheme reasonable inputs over land
    (so the column-stable testing works and the scheme exercises its full
    code path) while letting the activation gate (``ppic-pmea > gpicmea``
    AND ``pstd > gstd``) automatically disable the scheme over ocean
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

    Replace this with real preprocessed SSO data once available — pass the
    fields directly to :meth:`TerrainData.from_coords` or load them from a
    terrain file.
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
        elif orography.shape not in VALID_NODAL_SHAPES:
            raise ValueError(f"Invalid terrain data shape: {orography.shape}. Must be one of: {VALID_NODAL_SHAPES}.")

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


def _load_sso_from_file(terrain_file):
    """Load SSO descriptor fields from a terrain file if present.

    Returns a dict of the six SSO arrays, or ``None`` if the file lacks
    any of them. Accepts the standard ECHAM-style names
    (``orostd``/``orosig``/``orogam``/``orothe``/``oropic``/``oroval``).
    """
    import xarray as xr
    ds = xr.open_dataset(terrain_file)
    sso_names = ("orostd", "orosig", "orogam", "orothe", "oropic", "oroval")
    if not all(name in ds for name in sso_names):
        return None
    return {name: jnp.asarray(ds[name]) for name in sso_names}


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
    when only the mean orography is available, :func:`derive_sso_descriptors`
    generates placeholder values.

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
                :func:`derive_sso_descriptors`. If none provided and
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

        phi0 = grav * orog
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
        sso_derived = derive_sso_descriptors(orog)

        sso = dict(sso_derived)
        if sso_from_file is not None:
            sso.update(sso_from_file)
        sso.update(sso_user_provided)

        return cls(orog=orog, phis0=phis0, fmask=fmask,
                   lfluxland=jnp.bool_(lfluxland), **sso)

    @classmethod
    def from_file(cls, terrain_file, coords: CoordinateSystem, lfluxland=True):
        """Initialize TerrainData from a given terrain file containing orog and lsm.

        SSO descriptor fields (``orostd``, ``orosig``, ``orogam``, ``orothe``,
        ``oropic``, ``oroval``) are read from the file if all are present;
        otherwise they are derived from the mean orography via
        :func:`derive_sso_descriptors`.

        Args:
            terrain_file: Path to a file containing a dataset of orog (orography) and lsm (land-sea mask).
            coords: dinosaur.coordinate_systems.CoordinateSystem object.
            lfluxland (optional): Whether to compute land surface fluxes (default True).

        Returns:
            TerrainData object

        """
        orography, fmask = get_terrain(terrain_file=terrain_file, grid=coords.horizontal)

        # Validate that terrain matches coords
        if orography.shape != coords.horizontal.nodal_shape:
            raise ValueError(
                f"Terrain shape {orography.shape} does not match coords horizontal shape {coords.horizontal.nodal_shape}"
            )

        phi0 = grav * orography
        phis0 = spectral_truncation(coords.horizontal, phi0)

        sso_from_file = _load_sso_from_file(terrain_file)
        sso = sso_from_file if sso_from_file is not None else derive_sso_descriptors(orography)

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
        ``orog`` via :func:`derive_sso_descriptors`.

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
            phis0 = grav * orog

        orog_arr = jnp.array([[orog]])
        sso_derived = derive_sso_descriptors(orog_arr)
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
