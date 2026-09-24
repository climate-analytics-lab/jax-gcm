"""ECHAM 6.3 surface albedo: land (with snow and glaciers), sea ice, ocean.

Port of the three albedo paths ECHAM 6.3 evaluates each step in
``mo_surface.f90::update_surface`` before handing the grid-box tile average to
the radiation:

* **Land** — JSBACH ``mo_land_surface.f90::update_land_surface_fast``, the
  ECHAM5-compatible broadband scheme JSBACH runs when ``use_albedo=.FALSE.``.
  It needs only a broadband snow-free background albedo, a snow-cover
  fraction, the surface temperature, a forest fraction and a leaf area index,
  which is what a prescribed-surface model can supply. Snow albedo over the
  ground ramps linearly from ``AlbedoSnowMin`` at the melting point to
  ``AlbedoSnowMax`` at ``tmelt - 5 K``; glacier cells take the glacier
  column of the same table instead of the background-plus-snow blend.
  Constants are the ``lctlib_nlct21.def`` land-cover library values
  (``AlbedoSnowMin/Max`` = 0.4/0.8, glacier 0.75/0.85 — identical for every
  non-glacier land-cover type, so no land-cover map is needed). The
  resulting albedo is broadband and JSBACH writes it to both
  ``albedo_vis`` and ``albedo_nir`` (``mo_jsbach_interface.f90:696-697``).

  JSBACH's other path (``use_albedo=.TRUE.``,
  ``update_albedo_snowage_temp``) needs per-band soil and canopy albedo maps,
  per-PFT cover fractions and a prognostic snow age; none exist here, so it
  is not the one ported. See ``docs/source/science/surface.md`` for the data
  mapping and the known deviations.

* **Sea ice** — ``mo_surface_ice.f90::update_albedo_ice`` (the
  ``lmeltpond=.FALSE.`` path): bare-ice or snow-on-ice albedo ramping over
  the 1 K below the melting point, with the ``init_albedo_ice`` constants for
  T63/T127/T255 (``calbmni``/``calbmxi`` = 0.60/0.75, ``calbmns``/``calbmxs`` =
  0.70/0.85). The melt-pond scheme ECHAM runs by default needs prognostic
  ice thickness, pond depth and snow age, which the prescribed sea-ice tile
  does not carry.

* **Ocean** — ``mo_surface_ocean.f90::update_albedo_ocean``: a zenith-angle
  dependent direct-beam albedo (Taylor et al. 1996 fit) with a small
  visible/near-IR offset, and a constant diffuse albedo ``calbsea = 0.07``.
  Under ECHAM's ``lrce`` switch the direct-beam fit is replaced by the
  constant 0.07.

Every numeric constant is a differentiable leaf of
:class:`EchamSurfaceAlbedoParameters`; only the RCE switch is static. All
functions are broadcasting-native: inputs of any (matching) horizontal shape.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct

import jcm.constants as c

#: Saline freezing point of sea water [K] (ECHAM ``iniphy.f90``: ``ctfreez``).
#: The prescribed sea-ice tile has no prognostic skin temperature; its
#: temperature is ``min(SST, ctfreez)`` in every ECHAM-surface consumer.
CTFREEZ = 271.38


@struct.dataclass
class EchamSurfaceAlbedoParameters:
    """ECHAM 6.3 surface albedo constants (all differentiable leaves).

    Defaults are the values ECHAM 6.3 uses at T63/T127/T255; see
    :meth:`echam_t31` for the T31 sea-ice set.
    """

    # --- Land: JSBACH lctlib_nlct21.def + update_land_surface_fast --------
    #: ``AlbedoSnowMin``: snow on ground at/above the melting point.
    snow_albedo_min: jnp.ndarray = 0.4
    #: ``AlbedoSnowMax``: snow on ground below ``tmelt - snow_ramp_width``.
    snow_albedo_max: jnp.ndarray = 0.8
    #: Glacier column of the same table (``AlbedoSnowMin/Max`` for LCT 1).
    glacier_albedo_min: jnp.ndarray = 0.75
    glacier_albedo_max: jnp.ndarray = 0.85
    #: ``tmelt - min_temp_snow_albedo`` [K]: width of the melting ramp.
    snow_ramp_width: jnp.ndarray = 5.0
    #: ``AlbedoCanopySnow``: albedo of a snow-covered canopy.
    canopy_snow_albedo: jnp.ndarray = 0.20
    #: ``SkyViewFactor`` in ``exp(-SkyViewFactor * max(lai, lai_floor))``.
    sky_view_factor: jnp.ndarray = 1.0
    #: The ``2.0`` in ``MAX(lai, 2.0)`` (a stand-in for stem area in JSBACH).
    leaf_area_index_floor: jnp.ndarray = 2.0

    # --- Sea ice: mo_surface_ice.f90 init_albedo_ice / update_albedo_ice ---
    seaice_albedo_bare_min: jnp.ndarray = 0.60   # calbmni
    seaice_albedo_bare_max: jnp.ndarray = 0.75   # calbmxi
    seaice_albedo_snow_min: jnp.ndarray = 0.70   # calbmns
    seaice_albedo_snow_max: jnp.ndarray = 0.85   # calbmxs
    #: ``tmelt - temp_upper_limit`` [K]: width of the sea-ice melting ramp.
    seaice_ramp_width: jnp.ndarray = 1.0
    #: Snow water equivalent [m] above which the ice counts as snow covered.
    seaice_snow_threshold: jnp.ndarray = 0.01

    # --- Ocean: mo_surface_ocean.f90 update_albedo_ocean ------------------
    #: ``calbsea``: diffuse albedo of sea water (both bands).
    ocean_albedo_diffuse: jnp.ndarray = 0.07
    #: Direct-beam fit ``a / (mu0**b + d) + e (mu0-0.1)(mu0-0.5)(mu0-1)``.
    ocean_direct_a: jnp.ndarray = 0.026
    ocean_direct_b: jnp.ndarray = 1.7
    ocean_direct_d: jnp.ndarray = 0.065
    ocean_direct_e: jnp.ndarray = 0.015
    #: Band offsets on the direct-beam albedo (``palw1dir``/``palw2dir``).
    ocean_direct_vis_offset: jnp.ndarray = 0.0082
    ocean_direct_nir_offset: jnp.ndarray = -0.007
    #: ECHAM ``lrce``: the direct-beam albedo is the constant 0.07 instead
    #: of the zenith fit. Static — it selects a code path.
    rce: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def echam_t31(cls, **overrides) -> "EchamSurfaceAlbedoParameters":
        """Return the T31 sea-ice constants of ``init_albedo_ice`` (``nn == 31``)."""
        values = dict(seaice_albedo_bare_min=0.55,
                      seaice_albedo_snow_min=0.65,
                      seaice_albedo_snow_max=0.80)
        values.update(overrides)
        return cls(**values)


def _melting_ramp(temperature, value_min, value_max, width):
    """``value_min`` at/above tmelt, ``value_max`` below ``tmelt - width``.

    Linear in between: ECHAM's three-branch ``IF`` written as one clipped
    interpolation (``min + (max - min) * (tmelt - T) / width``).
    """
    weight = jnp.clip((c.tmelt - temperature) / width, 0.0, 1.0)
    return value_min + (value_max - value_min) * weight


def land_albedo(
    background_albedo: jnp.ndarray,
    snow_fraction: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    params: EchamSurfaceAlbedoParameters,
    forest_fraction: jnp.ndarray | float = 0.0,
    glacier_fraction: jnp.ndarray | float = 0.0,
    leaf_area_index: jnp.ndarray | float = 0.0,
    canopy_snow_fraction: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Broadband land albedo, JSBACH ``update_land_surface_fast``.

    Non-glacier land::

        ff  = forest * (1 - exp(-SkyViewFactor * max(lai, 2)))
        a_s = ramp(T; AlbedoSnowMin, AlbedoSnowMax)
        alb = max((1-ff)(sf a_s + (1-sf) bg) + ff (csf a_cs + (1-csf) bg), bg)

    Glacier land takes ``ramp(T)`` of the glacier constants. A fractional
    ``glacier_fraction`` blends the two linearly, which is JSBACH's
    cover-fraction tile average (``average_tiles``) when a glacier tile and a
    non-glacier tile share the cell.

    Parameters
    ----------
    background_albedo : jnp.ndarray
        Snow-free broadband background albedo (JSBACH ``Soil%albedo``).
    snow_fraction : jnp.ndarray
        Fraction of the ground covered by snow, in [0, 1].
    surface_temperature : jnp.ndarray
        Land surface temperature [K].
    params : EchamSurfaceAlbedoParameters
        Albedo constants.
    forest_fraction : jnp.ndarray or float
        Forest fraction of the land (JSBACH ``forest_fract``).
    glacier_fraction : jnp.ndarray or float
        Glacier fraction of the land (JSBACH ``is_glacier`` tile).
    leaf_area_index : jnp.ndarray or float
        Leaf area index; ECHAM floors it at ``leaf_area_index_floor``.
    canopy_snow_fraction : jnp.ndarray or float
        Snow-covered fraction of the canopy (JSBACH ``snow_fract_canopy``).

    Returns
    -------
    jnp.ndarray
        Broadband land albedo, used for both the visible and near-IR bands.

    """
    p = params
    snow_albedo = _melting_ramp(surface_temperature, p.snow_albedo_min,
                                p.snow_albedo_max, p.snow_ramp_width)
    glacier_albedo = _melting_ramp(surface_temperature, p.glacier_albedo_min,
                                   p.glacier_albedo_max, p.snow_ramp_width)
    sky_view = jnp.exp(-p.sky_view_factor * jnp.maximum(
        leaf_area_index, p.leaf_area_index_floor))
    canopy = forest_fraction * (1.0 - sky_view)
    ground = (snow_fraction * snow_albedo
              + (1.0 - snow_fraction) * background_albedo)
    crown = (canopy_snow_fraction * p.canopy_snow_albedo
             + (1.0 - canopy_snow_fraction) * background_albedo)
    non_glacier = jnp.maximum((1.0 - canopy) * ground + canopy * crown,
                              background_albedo)
    return (glacier_fraction * glacier_albedo
            + (1.0 - glacier_fraction) * non_glacier)


def sea_ice_albedo(
    surface_temperature: jnp.ndarray,
    params: EchamSurfaceAlbedoParameters,
    snow_depth: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Broadband sea-ice albedo, ``mo_surface_ice.f90::update_albedo_ice``.

    Bare ice ramps from ``calbmni`` (at/above tmelt) to ``calbmxi`` (below
    ``tmelt - 1 K``); ice carrying more than 0.01 m snow water equivalent uses
    ``calbmns``/``calbmxs`` instead. ECHAM copies the result to every band and
    to both the direct and diffuse albedo (``mo_surface.f90:733-738``).

    Parameters
    ----------
    surface_temperature : jnp.ndarray
        Sea-ice surface temperature [K].
    params : EchamSurfaceAlbedoParameters
        Albedo constants.
    snow_depth : jnp.ndarray or float
        Snow water equivalent on the ice [m].

    Returns
    -------
    jnp.ndarray
        Sea-ice albedo, all bands.

    """
    p = params
    snowy = snow_depth > p.seaice_snow_threshold
    albedo_min = jnp.where(snowy, p.seaice_albedo_snow_min,
                           p.seaice_albedo_bare_min)
    albedo_max = jnp.where(snowy, p.seaice_albedo_snow_max,
                           p.seaice_albedo_bare_max)
    return _melting_ramp(surface_temperature, albedo_min, albedo_max,
                         p.seaice_ramp_width)


def ocean_albedo(
    cos_zenith: jnp.ndarray,
    params: EchamSurfaceAlbedoParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Open-water albedo, ``mo_surface_ocean.f90::update_albedo_ocean``.

    Returns ``(vis_direct, nir_direct, diffuse)``. The direct-beam albedo is
    ``zalw = 0.026 / (mu0**1.7 + 0.065) + 0.015 (mu0-0.1)(mu0-0.5)(mu0-1)``
    plus ``+0.0082`` (visible) / ``-0.007`` (near-IR); the diffuse albedo is
    ``calbsea`` in both bands. With ``params.rce`` the fit is replaced by
    ``zalw = 0.07`` (ECHAM ``lrce``).

    ECHAM evaluates the fit only where sunlight reaches the surface and keeps
    the previous value otherwise. A dark column's albedo never enters a
    shortwave calculation, so for ``mu0 <= 0`` the diffuse value is returned
    for the direct beam too rather than carrying state for it (the fit
    extrapolates to ~0.4 at ``mu0 = 0``, which would otherwise show up in the
    night-side albedo diagnostic).

    Parameters
    ----------
    cos_zenith : jnp.ndarray
        Cosine of the solar zenith angle.
    params : EchamSurfaceAlbedoParameters
        Albedo constants.

    Returns
    -------
    tuple of jnp.ndarray
        ``(vis_direct, nir_direct, diffuse)``.

    """
    p = params
    diffuse = p.ocean_albedo_diffuse * jnp.ones_like(cos_zenith)
    sunlit = cos_zenith > 0.0
    if p.rce:
        zalw = 0.07 * jnp.ones_like(cos_zenith)
    else:
        # Dark columns evaluate the fit at mu0 = 1 and are then discarded:
        # at mu0 = 0 the exponent's derivative ``mu0**b · ln(mu0)`` is
        # 0·(-inf), which the ``where`` below would turn into a NaN gradient
        # for ``ocean_direct_b`` rather than drop.
        mu0 = jnp.where(sunlit, jnp.clip(cos_zenith, 0.0, 1.0), 1.0)
        zalw = (p.ocean_direct_a / (mu0 ** p.ocean_direct_b + p.ocean_direct_d)
                + p.ocean_direct_e * (mu0 - 0.1) * (mu0 - 0.5) * (mu0 - 1.0))
    vis_direct = jnp.where(sunlit, zalw + p.ocean_direct_vis_offset, diffuse)
    nir_direct = jnp.where(sunlit, zalw + p.ocean_direct_nir_offset, diffuse)
    return vis_direct, nir_direct, diffuse


#: Weight of the direct-beam albedo when the ocean's direct and diffuse
#: albedos are merged into the single per-band value the radiation accepts.
#:
#: STOPGAP, to be removed when the radiation takes separate direct and
#: diffuse surface albedos. ECHAM weights each by that band's downward
#: direct/diffuse irradiance from the previous radiation call. jax-rrtmgp's
#: surface boundary applies ONE albedo to both the direct beam
#: (``sw_cell_source``) and the diffuse field (``sw_transport``) and does not
#: return the direct/diffuse partition, so neither the ECHAM weights nor a
#: place to use them exist. An even split is the minimax choice under an
#: unknown partition: its error in either limit (all-direct clear sky,
#: all-diffuse overcast) is half the direct/diffuse contrast.
OCEAN_DIRECT_WEIGHT = 0.5


def ocean_albedo_per_band(
    cos_zenith: jnp.ndarray,
    params: EchamSurfaceAlbedoParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ocean ``(vis, nir)`` albedo merged over direct and diffuse light.

    See :data:`OCEAN_DIRECT_WEIGHT` for why the merge is not ECHAM's
    irradiance-weighted one.
    """
    vis_direct, nir_direct, diffuse = ocean_albedo(cos_zenith, params)
    w = OCEAN_DIRECT_WEIGHT
    return (w * vis_direct + (1.0 - w) * diffuse,
            w * nir_direct + (1.0 - w) * diffuse)
