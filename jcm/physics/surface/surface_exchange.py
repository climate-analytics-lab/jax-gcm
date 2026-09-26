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

The near-surface wind vector ``(wind_u, wind_v)`` is the eastward/northward
wind at the SAME reference as ``wind_speed``, which is each package's own
(maintainer decision on #911): ECHAM's stability-corrected 10 m wind,
SPEEDY's ``fwind0``-scaled lowest-level wind. The struct names its reference
in the static ``wind_reference`` field (a key of :data:`WIND_REFERENCES`), so
a coupler can check it rather than assume a height.

**Optional fields** default ``None`` — absence is explicit, never a zero
that could be mistaken for data (the #647 lesson). They cover the per-tile
resolution and the rain/snow split. No current package can fill the flux
tiles or the split faithfully (see the design doc: SPEEDY blends the sea-ice
*temperature* rather than tile fluxes, and ECHAM's per-tile explicit fluxes
are computed but inconsistent with the delivered grid mean from the vdiff
implicit solve). ECHAM does fill the wind tiles: its grid-mean 10 m wind is
by construction the fraction-weighted sum of the per-tile 10 m winds. A
``None`` field is simply omitted from the netCDF output.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import struct

#: Diagnostics key every publisher writes the struct under (and lists in
#: ``provides``). No leading underscore: the struct is user-facing output,
#: flattened to ``surface_exchange.<field>`` variables in the Dataset.
SURFACE_EXCHANGE_KEY = "surface_exchange"

#: The reference each package's near-surface wind (``wind_speed``,
#: ``wind_u``, ``wind_v`` and their tiles) is defined at, keyed by the value
#: of :attr:`SurfaceExchange.wind_reference`. The packages keep their own
#: reference rather than a common height (#911): SPEEDY has no surface-layer
#: profile to reduce its wind to 10 m, so a common height would be invented.
WIND_REFERENCES = {
    "10m": "10 m above the surface, reduced from the lowest model level with "
           "the stability-dependent surface-layer profile of each tile "
           "(ECHAM mo_surface nsurf_diag)",
    "lowest_level": "the lowest-model-level wind scaled by fwind0 (SPEEDY "
                    "surface-flux closure; fwind0 = 0.95 by default)",
}


@struct.dataclass
class SurfaceExchange:
    """What an external surface component reads from the atmosphere.

    All guaranteed fields are grid-box means on the publisher's horizontal
    layout (``(ix, il)`` for grid-hosted physics, ``(ncols,)`` for
    column-vectorized physics). Optional fields are ``None`` unless the
    package can fill them faithfully; ``*_tile`` fields carry a trailing
    tile axis whose meaning (and fractions) ``tile_fraction`` defines, with
    the guarantee ``sum(tile_fraction * field_tile, axis=-1) == field``.
    :meth:`validate` checks these invariants on concrete values.

    ``wind_reference`` is static pytree metadata (not an array leaf), so the
    struct stays a valid ``jit``/``scan`` output and the value is readable
    on the host without tracing.
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
    #: Near-surface wind speed [m s-1] at the package's reference
    #: (``wind_reference``; ECHAM: 10 m, SPEEDY: fwind0 x lowest level).
    wind_speed: jnp.ndarray
    #: Eastward near-surface wind [m s-1] at the same reference, on the
    #: unrotated model grid; ``hypot(wind_u, wind_v) == wind_speed``.
    wind_u: jnp.ndarray
    #: Northward near-surface wind [m s-1] at the same reference.
    wind_v: jnp.ndarray
    #: Moist air density at the lowest model level [kg m-3],
    #: p / (Rd * T * (1 + vtmpc1 q)). Requested by external bulk-flux
    #: algorithms (#301 discussion).
    air_density: jnp.ndarray
    #: Potential temperature at the lowest model level [K],
    #: T * (p0 / p)**kappa.
    air_potential_temperature: jnp.ndarray
    #: Which reference the wind fields sit at: a key of
    #: :data:`WIND_REFERENCES`. Static (``pytree_node=False``) and a
    #: required keyword, so no publisher can leave its wind undescribed.
    wind_reference: str = struct.field(pytree_node=False, kw_only=True)

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
    #: Per-tile near-surface wind at the same reference as the grid-mean
    #: wind [(..., ntile), m s-1], same tile invariant. Optional: ECHAM
    #: fills them from its per-tile 10 m reduction; SPEEDY has no tiles.
    wind_u_tile: Any = None
    wind_v_tile: Any = None
    wind_speed_tile: Any = None

    @classmethod
    def zeros(cls, nodal_shape, wind_reference: str) -> "SurfaceExchange":
        """All-zero guaranteed fields (optional fields stay ``None``)."""
        z = jnp.zeros(nodal_shape)
        return cls(
            net_heat_flux=z, sensible_heat_flux=z, latent_heat_flux=z,
            evaporation=z, precipitation=z, stress_u=z, stress_v=z,
            wind_speed=z, wind_u=z, wind_v=z, air_density=z,
            air_potential_temperature=z, wind_reference=wind_reference,
        )

    def validate(self, rtol: float = 1e-5, atol: float = 1e-6) -> None:
        """Check the contract invariants on concrete (host) values.

        Raises ``ValueError`` if ``wind_reference`` is not a key of
        :data:`WIND_REFERENCES`, if ``hypot(wind_u, wind_v)`` differs from
        ``wind_speed`` (grid mean, and per tile when the wind tiles are
        present), or if any provided ``*_tile`` field does not reproduce its
        grid mean through ``tile_fraction``. For couplers and tests; it
        reads values, so call it outside ``jit``.
        """
        if self.wind_reference not in WIND_REFERENCES:
            raise ValueError(
                f"wind_reference={self.wind_reference!r}; expected one of "
                f"{sorted(WIND_REFERENCES)}.")

        def _check(name, got, want):
            if not np.allclose(np.asarray(got), np.asarray(want),
                               rtol=rtol, atol=atol):
                err = np.max(np.abs(np.asarray(got) - np.asarray(want)))
                raise ValueError(
                    f"SurfaceExchange invariant violated: {name} "
                    f"(max abs difference {err:.3g}).")

        _check("hypot(wind_u, wind_v) == wind_speed",
               np.hypot(self.wind_u, self.wind_v), self.wind_speed)
        tiles = [(f[:-len("_tile")], getattr(self, f))
                 for f in self.__dataclass_fields__
                 if f.endswith("_tile") and getattr(self, f) is not None]
        if not tiles:
            return
        if self.tile_fraction is None:
            raise ValueError(
                "SurfaceExchange has per-tile fields "
                f"{[n + '_tile' for n, _ in tiles]} but no tile_fraction.")
        frac = np.asarray(self.tile_fraction)
        _check("sum(tile_fraction) == 1", frac.sum(axis=-1), 1.0)
        for name, tile in tiles:
            _check(f"sum(tile_fraction * {name}_tile) == {name}",
                   (frac * np.asarray(tile)).sum(axis=-1), getattr(self, name))
        if all(t is not None for t in (self.wind_u_tile, self.wind_v_tile,
                                       self.wind_speed_tile)):
            _check("hypot(wind_u_tile, wind_v_tile) == wind_speed_tile",
                   np.hypot(self.wind_u_tile, self.wind_v_tile),
                   self.wind_speed_tile)


#: CF/units metadata for the flattened ``surface_exchange.*`` output
#: variables (PhysicsTerm.output_attrs contract, #740). Shared by every
#: publisher so the netCDF metadata cannot drift between packages.
SURFACE_EXCHANGE_OUTPUT_ATTRS = {
    "surface_exchange.net_heat_flux": {
        "units": "W m-2",
        "long_name": "net downward heat flux into the surface (positive down)",
        "description": "net downward heat flux into the surface medium "
                       "(SW_net + LW_net - SHF - LHF), positive down",
    },
    "surface_exchange.sensible_heat_flux": {
        "units": "W m-2",
        "standard_name": "surface_upward_sensible_heat_flux",
        "long_name": "surface sensible heat flux (positive up)",
        "description": "surface sensible heat flux, positive up "
                       "(surface to atmosphere)",
    },
    "surface_exchange.latent_heat_flux": {
        "units": "W m-2",
        "standard_name": "surface_upward_latent_heat_flux",
        "long_name": "surface latent heat flux (positive up)",
        "description": "surface latent heat flux, positive up "
                       "(surface to atmosphere)",
    },
    "surface_exchange.evaporation": {
        "units": "kg m-2 s-1",
        "standard_name": "water_evapotranspiration_flux",
        "long_name": "surface evaporation (positive up)",
        "description": "surface evaporation / moisture flux, positive up",
    },
    "surface_exchange.precipitation": {
        "units": "kg m-2 s-1",
        "standard_name": "precipitation_flux",
        "long_name": "total precipitation (rain + snow, positive down)",
        "description": "total precipitation reaching the surface "
                       "(rain + snow, convective + stratiform), positive down",
    },
    "surface_exchange.stress_u": {
        "units": "N m-2",
        "standard_name": "surface_downward_eastward_stress",
        "long_name": "downward eastward momentum flux into the surface",
        "description": "downward flux of eastward momentum into the surface "
                       "(= minus the stress on the atmosphere)",
    },
    "surface_exchange.stress_v": {
        "units": "N m-2",
        "standard_name": "surface_downward_northward_stress",
        "long_name": "downward northward momentum flux into the surface",
        "description": "downward flux of northward momentum into the surface "
                       "(= minus the stress on the atmosphere)",
    },
    "surface_exchange.wind_speed": {
        "units": "m s-1",
        "standard_name": "wind_speed",
        "long_name": "near-surface wind speed at the package reference",
        "description": "near-surface wind speed at the package's reference "
                       "(see the wind_reference attribute)",
    },
    "surface_exchange.wind_u": {
        "units": "m s-1",
        "standard_name": "eastward_wind",
        "long_name": "near-surface eastward wind at the package reference",
        "description": "near-surface eastward wind at the same reference as "
                       "wind_speed (see the wind_reference attribute)",
    },
    "surface_exchange.wind_v": {
        "units": "m s-1",
        "standard_name": "northward_wind",
        "long_name": "near-surface northward wind at the package reference",
        "description": "near-surface northward wind at the same reference as "
                       "wind_speed (see the wind_reference attribute)",
    },
    "surface_exchange.air_density": {
        "units": "kg m-3",
        "long_name": "moist air density at the lowest model level",
        "description": "moist air density at the lowest model level",
    },
    "surface_exchange.air_potential_temperature": {
        "units": "K",
        "long_name": "potential temperature at the lowest model level",
        "description": "potential temperature at the lowest model level",
    },
}


_WIND_OUTPUT_FIELDS = ("wind_speed", "wind_u", "wind_v")


def surface_exchange_output_attrs(wind_reference: str,
                                  tile_names: tuple[str, ...] = ()) -> dict:
    """:data:`SURFACE_EXCHANGE_OUTPUT_ATTRS` stamped with a wind reference.

    Each publisher declares its ``output_attrs`` through this, so the netCDF
    wind variables carry the same ``wind_reference`` as the in-memory struct
    plus its :data:`WIND_REFERENCES` description (and a CF-style ``height``
    for the 10 m reference).

    ``tile_names`` (tile index order) is given by a publisher that fills the
    wind tiles. The output flattener writes a trailing tile axis as one
    variable per tile (``surface_exchange.wind_u_tile.0``, ...), and output
    attributes match exact names, so each expanded tile variable, and each
    ``tile_fraction.N``, gets its own entry: the grid-mean wind attributes
    plus ``surface_type`` (the tile's name) and ``tile_index``.
    """
    if wind_reference not in WIND_REFERENCES:
        raise ValueError(f"wind_reference={wind_reference!r}; expected one "
                         f"of {sorted(WIND_REFERENCES)}.")
    attrs = {k: dict(v) for k, v in SURFACE_EXCHANGE_OUTPUT_ATTRS.items()}
    for field in _WIND_OUTPUT_FIELDS:
        entry = attrs[f"surface_exchange.{field}"]
        entry["wind_reference"] = wind_reference
        entry["wind_reference_description"] = WIND_REFERENCES[wind_reference]
        if wind_reference == "10m":
            entry["height"] = "10 m"
    for index, name in enumerate(tile_names):
        tile = {"surface_type": name, "tile_index": index}
        for field in _WIND_OUTPUT_FIELDS:
            grid = attrs[f"surface_exchange.{field}"]
            attrs[f"surface_exchange.{field}_tile.{index}"] = {
                **grid, **tile,
                "long_name": f"{grid['long_name']} over the {name} tile",
                "description": f"{grid['description']}; {name} tile "
                               f"(index {index}) of the tile axis",
            }
        attrs[f"surface_exchange.tile_fraction.{index}"] = {
            **tile, "units": "1", "standard_name": "area_fraction",
            "long_name": f"{name} tile fraction of the grid box",
            "description": f"fraction of the grid box covered by the {name} "
                           f"tile (index {index}); the tiles sum to 1",
        }
    return attrs


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
