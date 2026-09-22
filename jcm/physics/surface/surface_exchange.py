"""Package-independent surface-exchange contract (jax-gcm#754 / #301).

:class:`SurfaceExchange` is the one struct an external surface component
(ocean, sea ice, land, wave — e.g. Veros through JAX-ESM) reads FROM the
atmosphere, published identically by every physics package that resolves a
surface. It is the *publishing* half of the coupling contract; the inverse
door — prescribing externally computed fluxes INTO jcm — shares this module's
field list and sign convention (see
:class:`jcm.physics.surface.prescribed_flux.PrescribedSurfaceFlux` and the
``prescribed_*`` fields on :class:`jcm.forcing.ForcingData`).

The full design, with the evidence for the grid-mean-guaranteed /
tile-optional split, lives in ``docs/source/design/surface_exchange.md``.

Contract summary
----------------
Every publisher writes the struct under ``diagnostics["surface_exchange"]``
(:data:`SURFACE_EXCHANGE_KEY`) and declares it in ``provides``, so
composition fails loudly (``ComposablePhysics`` ordering validation /
:meth:`~jcm.physics.composable_physics.ComposablePhysics.require_surface_exchange`)
rather than a coupler discovering a missing key at run time.

**Guaranteed fields** are grid-box means filled by every publishing package
*exactly* from the fluxes it actually delivered to the atmosphere this step
— never from a parallel diagnostic recomputation that could drift from the
delivered values. Signs (maintainer decision on #754):

- turbulent fluxes (``sensible_heat_flux``, ``latent_heat_flux``,
  ``evaporation``) are **positive up** (surface → atmosphere);
- ``net_heat_flux`` is **positive down** (into the surface medium — the
  SPEEDY ``hfluxn`` convention);
- ``stress_u`` / ``stress_v`` are **positive down**: the downward flux of
  eastward/northward momentum into the surface, i.e. the stress the
  atmosphere exerts ON the surface (= minus the stress on the atmosphere).
  With surface westerlies ``stress_u > 0``.
- ``precipitation`` is positive down and non-negative.

**Optional fields** default ``None`` — absence is explicit, never a zero
that could be mistaken for data (the #647 lesson). They cover the per-tile
resolution and the rain/snow split, which no current package can fill
faithfully (see the design doc: SPEEDY blends the sea-ice *temperature*
rather than tile fluxes, and ECHAM's per-tile explicit fluxes are computed
but inconsistent with the delivered grid mean from the vdiff implicit
solve). A ``None`` field is simply omitted from the netCDF output.
"""

from typing import Any

import jax.numpy as jnp
import tree_math

#: Diagnostics key every publisher writes the struct under (and lists in
#: ``provides``). No leading underscore: the struct is user-facing output,
#: flattened to ``surface_exchange.<field>`` variables in the Dataset.
SURFACE_EXCHANGE_KEY = "surface_exchange"


@tree_math.struct
class SurfaceExchange:
    """What an external surface component reads from the atmosphere.

    All guaranteed fields are grid-box means on the publisher's horizontal
    layout (``(ix, il)`` for grid-hosted physics, ``(ncols,)`` for
    column-vectorized physics). Optional fields are ``None`` unless the
    package can fill them faithfully; ``*_tile`` fields carry a trailing
    tile axis whose meaning (and fractions) ``tile_fraction`` defines, with
    the guarantee ``sum(tile_fraction * field_tile, axis=-1) == field``.
    """

    # --- Guaranteed grid-mean fields -----------------------------------
    #: Net downward heat flux into the surface medium [W m-2], positive
    #: down: SW_net + LW_net - SHF - LHF at the surface.
    net_heat_flux: jnp.ndarray
    #: Sensible heat flux [W m-2], positive up (surface -> atmosphere).
    sensible_heat_flux: jnp.ndarray
    #: Latent heat flux [W m-2], positive up. The package's own delivered
    #: value (which may include sublimation weighting); not necessarily
    #: alhv * evaporation.
    latent_heat_flux: jnp.ndarray
    #: Evaporation / upward moisture flux [kg m-2 s-1], positive up.
    evaporation: jnp.ndarray
    #: Total precipitation (rain + snow, convective + stratiform)
    #: [kg m-2 s-1], positive down, >= 0.
    precipitation: jnp.ndarray
    #: Downward flux of eastward momentum into the surface [N m-2]
    #: (= minus the stress on the atmosphere).
    stress_u: jnp.ndarray
    #: Downward flux of northward momentum into the surface [N m-2].
    stress_v: jnp.ndarray
    #: Near-surface wind speed [m s-1] at the package's reference height
    #: (SPEEDY: the fwind0-scaled sigma=0.99 wind; ECHAM: 10 m).
    wind_speed: jnp.ndarray
    #: Moist air density at the lowest model level [kg m-3],
    #: p / (Rd * T * (1 + vtmpc1 q)). Requested by external bulk-flux
    #: algorithms (#301 discussion).
    air_density: jnp.ndarray
    #: Potential temperature at the lowest model level [K],
    #: T * (p0 / p)**kappa.
    air_potential_temperature: jnp.ndarray

    # --- Optional fields (None = not provided by this package) ---------
    #: Rain-only precipitation [kg m-2 s-1]. None until a package can split
    #: its TOTAL precipitation by phase (ECHAM's convective precip is
    #: currently unsplit, so stratiform-only rain would be wrong here).
    precip_rain: Any = None
    #: Snow-only precipitation [kg m-2 s-1]. Same caveat as precip_rain.
    precip_snow: Any = None
    #: Tile fractions [(..., ntile)], summing to 1 where provided.
    tile_fraction: Any = None
    #: Per-tile counterparts of the guaranteed fluxes [(..., ntile)]. A
    #: package may only fill these when the fraction-weighted tile sum
    #: reproduces the delivered grid mean — see the design doc.
    net_heat_flux_tile: Any = None
    sensible_heat_flux_tile: Any = None
    latent_heat_flux_tile: Any = None
    evaporation_tile: Any = None
    stress_u_tile: Any = None
    stress_v_tile: Any = None

    @classmethod
    def zeros(cls, nodal_shape) -> "SurfaceExchange":
        """All-zero guaranteed fields (optional fields stay ``None``)."""
        z = jnp.zeros(nodal_shape)
        return cls(
            net_heat_flux=z, sensible_heat_flux=z, latent_heat_flux=z,
            evaporation=z, precipitation=z, stress_u=z, stress_v=z,
            wind_speed=z, air_density=z, air_potential_temperature=z,
        )


#: CF/units metadata for the flattened ``surface_exchange.*`` output
#: variables (PhysicsTerm.output_attrs contract, #740). Shared by every
#: publisher so the netCDF metadata cannot drift between packages.
SURFACE_EXCHANGE_OUTPUT_ATTRS = {
    "surface_exchange.net_heat_flux": {
        "units": "W m-2",
        "long_name": "net downward heat flux into the surface (positive down)",
    },
    "surface_exchange.sensible_heat_flux": {
        "units": "W m-2",
        "standard_name": "surface_upward_sensible_heat_flux",
        "long_name": "surface sensible heat flux (positive up)",
    },
    "surface_exchange.latent_heat_flux": {
        "units": "W m-2",
        "standard_name": "surface_upward_latent_heat_flux",
        "long_name": "surface latent heat flux (positive up)",
    },
    "surface_exchange.evaporation": {
        "units": "kg m-2 s-1",
        "standard_name": "water_evapotranspiration_flux",
        "long_name": "surface evaporation (positive up)",
    },
    "surface_exchange.precipitation": {
        "units": "kg m-2 s-1",
        "standard_name": "precipitation_flux",
        "long_name": "total precipitation (rain + snow, positive down)",
    },
    "surface_exchange.stress_u": {
        "units": "N m-2",
        "standard_name": "surface_downward_eastward_stress",
        "long_name": "downward eastward momentum flux into the surface",
    },
    "surface_exchange.stress_v": {
        "units": "N m-2",
        "standard_name": "surface_downward_northward_stress",
        "long_name": "downward northward momentum flux into the surface",
    },
    "surface_exchange.wind_speed": {
        "units": "m s-1",
        "long_name": "near-surface wind speed at the package reference height",
    },
    "surface_exchange.air_density": {
        "units": "kg m-3",
        "long_name": "moist air density at the lowest model level",
    },
    "surface_exchange.air_potential_temperature": {
        "units": "K",
        "long_name": "potential temperature at the lowest model level",
    },
}


def surface_exchange_from(diagnostics: dict) -> SurfaceExchange:
    """Read the published :class:`SurfaceExchange` from a diagnostics dict.

    The coupler-side accessor: raises a pointed error (rather than a bare
    ``KeyError``) when the composed physics package publishes no surface
    exchange — e.g. Held-Suarez, which resolves no surface fluxes at all
    and explicitly opts out. Prefer
    :meth:`ComposablePhysics.require_surface_exchange` at composition time
    so a non-publishing package fails before the first step.
    """
    try:
        return diagnostics[SURFACE_EXCHANGE_KEY]
    except KeyError:
        raise KeyError(
            f"diagnostics has no {SURFACE_EXCHANGE_KEY!r} entry: the "
            "composed physics package publishes no surface-exchange "
            "struct. SPEEDY and ECHAM packages publish it; Held-Suarez "
            "opts out (it resolves no surface fluxes). See "
            "docs/source/design/surface_exchange.md."
        ) from None
