"""SLIM's carried land state and its end-of-step water budget.

The SLIM land (``land_model: slim``) is split over the step the way ECHAM's
``physc`` orders its surface: the surface energy balance and the soil
temperature column are solved with the lowest model level inside the
vertical diffusion (:func:`~jcm.physics.vertical_diffusion.tte_tke.
matrix_solver.couple_surface_tiles`), and the buckets are updated here, after
the convection and the cloud microphysics, from the evaporation and the melt
that solve delivered and the rain and snow this step's schemes deliver to the
surface. SLIM's own step (``mml_main.F90``) has the same order: the surface
fluxes and the soil column first, then "rain into buckets", evaporation and
runoff.

:class:`SlimLandData` is the carry slot ``slim_land``. Its state fields are
PROGNOSTIC but are not declared as ``prognostic_carry_slots``: a carry
without them (a cold start, a checkpoint written before ``land_model: slim``
existed) is seeded from the forcing on the first step, which is SLIM's own
initialisation stood in by the observed climatology (:func:`resolve_state`).
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.forcing import ForcingData
from jcm.physics.physics_term import PhysicsTerm
from jcm.physics.surface.slim import slim_land
from jcm.physics.surface.slim.properties import MONTHLY_FIELDS
from jcm.physics.surface.slim.slim_land import NSOIL, SlimParameters
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.terrain import TerrainData

#: The diagnostics/carry key of the SLIM land.
SLIM_LAND_KEY = "slim_land"

#: Snow water equivalent [kg/m2] the bundle's cover ``snowc = min(1, SWE/60 mm)``
#: stands for at full cover (``jcm.data.mirror.bundles``); used to seed the
#: snow bucket from the cover.
SNOW_COVER_SWE = 60.0

_STATE_FIELDS = ("soil_temperature", "soil_ice", "water", "snow")
_STEP_FIELDS = (
    "beta", "evaporation_factor", "evaporation", "snow_evaporation",
    "water_evaporation", "rain", "snowfall", "snow_melt", "melt_heat_flux",
    "runoff", "glacier_supply", "phase_change_heat_flux", "soil_heat_flux",
    "bucket_capacity", "roughness_length_momentum",
)


@tree_math.struct
class SlimLandData:
    """The SLIM land's carry: its state and this step's land-water budget.

    Per land column, on the land tile's own area (multiply by the land
    fraction for a grid mean). Rates are kg m-2 s-1 and positive into the
    named reservoir unless noted.
    """

    # --- PROGNOSTIC state. soil_temperature[0] <= 0 means unset: the step
    # seeds the column from the forcing (resolve_state). -------------------
    soil_temperature: jnp.ndarray      # [K] (NSOIL, ncols), layer 0 = surface
    soil_ice: jnp.ndarray              # frozen thermodynamic water [kg/m2] (NSOIL, ncols)
    water: jnp.ndarray                 # soil-water bucket W [kg/m2]
    snow: jnp.ndarray                  # snow bucket S [kg/m2]
    # --- this step --------------------------------------------------------
    beta: jnp.ndarray                  # bucket factor min(W/(0.75 cap), 1) [1]
    evaporation_factor: jnp.ndarray    # cair = csat of the land tile [1]
    evaporation: jnp.ndarray           # land-tile evaporation, positive up
    snow_evaporation: jnp.ndarray      # taken from the snow (and glacier ice), positive up
    water_evaporation: jnp.ndarray     # taken from the water bucket, positive up
    rain: jnp.ndarray                  # rain reaching the land
    snowfall: jnp.ndarray              # snow reaching the land
    snow_melt: jnp.ndarray             # melt, snow -> water bucket
    melt_heat_flux: jnp.ndarray        # L_f x melt [W/m2]
    runoff: jnp.ndarray                # bucket and snow-cap overflow
    glacier_supply: jnp.ndarray        # melt/sublimation a glacier supplies beyond its snow
    phase_change_heat_flux: jnp.ndarray  # soil freezing heat release [W/m2]
    soil_heat_flux: jnp.ndarray        # conduction from the top layer down [W/m2]
    bucket_capacity: jnp.ndarray       # [kg/m2], the step's bucketdepth
    roughness_length_momentum: jnp.ndarray  # land z0m = 0.1 x vegetation height [m]

    @classmethod
    def zeros(cls, nodal_shape, nlev=None):
        """Return an unset carry (seeded from the forcing on its first step)."""
        return cls(
            soil_temperature=jnp.zeros((NSOIL,) + tuple(nodal_shape)),
            soil_ice=jnp.zeros((NSOIL,) + tuple(nodal_shape)),
            **{name: jnp.zeros(nodal_shape) for name in ("water", "snow") + _STEP_FIELDS},
        )

    def copy(self, **kwargs):
        """Return a copy with ``kwargs`` replaced."""
        fields = {name: getattr(self, name) for name in _STATE_FIELDS + _STEP_FIELDS}
        fields.update(kwargs)
        return SlimLandData(**fields)


#: CF attributes of the outputs (``slim_land.<field>``).
SLIM_OUTPUT_ATTRS = {
    "slim_land.soil_temperature": {
        "units": "K", "long_name": "SLIM soil temperature (layer 1 is the land surface)"},
    "slim_land.soil_ice": {
        "units": "kg m-2", "long_name": "frozen part of the soil layer's thermodynamic water"},
    "slim_land.water": {"units": "kg m-2", "long_name": "SLIM soil-water bucket"},
    "slim_land.snow": {"units": "kg m-2", "long_name": "SLIM snow bucket (water equivalent)"},
    "slim_land.beta": {"units": "1", "long_name": "bucket evaporation factor min(W/(0.75 cap), 1)"},
    "slim_land.evaporation_factor": {
        "units": "1", "long_name": "land humidity factor beta_eff/(1 + C_h|U| r_evap)"},
    "slim_land.evaporation": {
        "units": "kg m-2 s-1", "long_name": "land-tile evaporation, positive up"},
    "slim_land.snow_evaporation": {
        "units": "kg m-2 s-1", "long_name": "sublimation from snow and glacier ice, positive up"},
    "slim_land.water_evaporation": {
        "units": "kg m-2 s-1", "long_name": "evaporation from the water bucket, positive up"},
    "slim_land.rain": {"units": "kg m-2 s-1", "long_name": "rain reaching the land"},
    "slim_land.snowfall": {"units": "kg m-2 s-1", "long_name": "snow reaching the land"},
    "slim_land.snow_melt": {"units": "kg m-2 s-1", "long_name": "snow melt"},
    "slim_land.melt_heat_flux": {"units": "W m-2", "long_name": "energy taken up by snow melt"},
    "slim_land.runoff": {
        "units": "kg m-2 s-1", "long_name": "runoff (bucket and snow-cap overflow)"},
    "slim_land.glacier_supply": {
        "units": "kg m-2 s-1",
        "long_name": "glacier ice melted or sublimated beyond the snow present"},
    "slim_land.phase_change_heat_flux": {
        "units": "W m-2", "long_name": "latent heat released by soil freezing (negative: thaw)"},
    "slim_land.soil_heat_flux": {
        "units": "W m-2", "long_name": "conduction from the top soil layer into the column below"},
    "slim_land.bucket_capacity": {"units": "kg m-2", "long_name": "bucket capacity"},
    "slim_land.roughness_length_momentum": {
        "units": "m", "long_name": "land momentum roughness length (0.1 x vegetation height)"},
}


def column_properties(forcing: ForcingData, ncols: int) -> dict:
    """SLIM's property fields for this step, each ``(ncols,)``."""
    props = getattr(forcing, "slim_properties", None)
    if props is None:
        raise ValueError(
            "land_model='slim' needs SLIM's surface properties: set "
            "forcing.slim_file to the regridded SLIM surface dataset "
            "(e.g. slim_t63.nc), or ForcingData.slim_properties.")
    return {name: jnp.asarray(value).reshape(ncols) for name, value in props.items()}


def resolve_state(data: SlimLandData, forcing: ForcingData, props: dict, ncols: int,
                  params: SlimParameters) -> SlimLandData:
    """Return the carried state, seeded from the forcing where it is unset.

    A column is unset when its surface temperature is ``<= 0`` (a cold start,
    or a carry restored from a checkpoint without the SLIM fields; the
    checkpoint logs the seeded fields). It then starts from the observed
    climatology: every soil layer at the forcing's land temperature
    ``stl_am``, the soil ice frozen where that is below the freezing point,
    the water bucket at ``clip(soilw_am, 0, 1)·bucketdepth`` (the bundle's
    root-zone availability as a fill) and the snow bucket at the bundle's
    snow cover times its full-cover 60 kg/m2. SLIM's own cold start (150 kg/m2
    of water, no snow, the soil at 275 K) needs years of spin-up; this one
    starts each column near its observed state.
    """
    t_soil = data.soil_temperature.reshape(NSOIL, ncols)
    unset = t_soil[0] <= 0.0
    stl = jnp.asarray(forcing.stl_am).reshape(ncols).astype(t_soil.dtype)
    t_seed = jnp.broadcast_to(stl, (NSOIL, ncols))
    _, _, _, max_water = slim_land.soil_layer_geometry(params)
    ice_seed = slim_land.seed_soil_ice(t_seed, max_water.astype(t_soil.dtype))
    fill = jnp.clip(jnp.asarray(forcing.soilw_am).reshape(ncols), 0.0, 1.0)
    water_seed = fill * props["bucketdepth"]
    snow_seed = jnp.clip(jnp.asarray(forcing.snowc_am).reshape(ncols), 0.0, 1.0) * SNOW_COVER_SWE
    return data.copy(
        soil_temperature=jnp.where(unset[None], t_seed, t_soil),
        soil_ice=jnp.where(unset[None], ice_seed, data.soil_ice.reshape(NSOIL, ncols)),
        water=jnp.where(unset, water_seed, data.water.reshape(ncols)).astype(t_soil.dtype),
        snow=jnp.where(unset, snow_seed, data.snow.reshape(ncols)).astype(t_soil.dtype),
    )


def surface_rain_and_snow(diagnostics: dict, ncols: int):
    """Rain and snow reaching the surface this step [kg m-2 s-1]."""
    from jcm.physics.surface.echam.surface_exchange_publisher import surface_rain_and_snow
    return surface_rain_and_snow(diagnostics.get("clouds"), diagnostics.get("convection"),
                                 ncols)


class SlimLand(PhysicsTerm):
    """SLIM's water budget at the end of the step, and owner of its carry.

    Composed after the convection and the cloud microphysics (the step's rain
    and snow at the surface), it moves this step's melt from the snow to the
    water bucket, adds rain and snowfall, takes the land-tile evaporation the
    vertical diffusion delivered from the snow and the water, and spills the
    overflow as runoff (:func:`~jcm.physics.surface.slim.slim_land.update_buckets`).
    The soil temperatures, the melt and the evaporation of the step are the
    vertical diffusion's (it reads and writes the same slot). Zero tendencies.
    """

    name: ClassVar[str] = "slim_land"
    category: ClassVar[str] = "land_surface"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = (SLIM_LAND_KEY,)
    carry_slots: ClassVar[dict[str, type]] = {SLIM_LAND_KEY: SlimLandData}
    # Same-step rain and snow: this term runs after both producers in
    # ``echam_physics``; a trimmed composition without them rains nothing.
    optional_requires: ClassVar[tuple[str, ...]] = ("clouds", "convection")
    output_attrs: ClassVar = SLIM_OUTPUT_ATTRS

    def __init__(self, params: SlimParameters | None = None):
        """Hold SLIM's constants (differentiable leaves)."""
        params = params or SlimParameters.default()
        slim_land.check_albedo_mode(params)
        self.params = nnx.Param(params)

    def augment_probe_forcing(self, forcing: ForcingData) -> ForcingData:
        """Give the shape probe SLIM properties, so it traces the SLIM path."""
        if forcing.slim_properties is not None:
            return forcing
        shape = jnp.shape(forcing.stl_am)
        props = {name: jnp.ones(shape) for name in MONTHLY_FIELDS}
        props["glc_mask"] = jnp.zeros(shape)
        return forcing.copy(slim_properties=props)

    def validate_forcing(self, forcing: ForcingData, run_window=None) -> None:
        """Refuse a run without SLIM's surface properties."""
        if forcing is not None and forcing.slim_properties is None:
            raise ValueError(
                "land_model='slim' needs SLIM's surface properties: set "
                "forcing.slim_file to the regridded SLIM surface dataset "
                "(e.g. slim_t63.nc), or ForcingData.slim_properties.")

    def __call__(self, state: PhysicsState, diagnostics: dict, forcing: ForcingData,
                 terrain: TerrainData) -> tuple[PhysicsTendency, dict]:
        """Update the buckets with this step's precipitation, melt and evaporation."""
        _nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()
        props = column_properties(forcing, ncols)
        # Its own carry; absent only in the structural probe (an unset carry).
        carry = diagnostics.get(SLIM_LAND_KEY)
        if carry is None:
            carry = SlimLandData.zeros((ncols,))
        data = resolve_state(carry, forcing, props, ncols, params)
        rain, snowfall = surface_rain_and_snow(diagnostics, ncols)
        dtype = data.water.dtype
        out = slim_land.update_buckets(
            data.water, data.snow, rain.astype(dtype), snowfall.astype(dtype),
            data.snow_melt.reshape(ncols), data.evaporation.reshape(ncols),
            props["bucketdepth"].astype(dtype), props["glc_mask"], dt, params)
        new = data.copy(
            water=out["water"], snow=out["snow"], runoff=out["runoff"],
            snow_evaporation=out["snow_evaporation"],
            water_evaporation=out["water_evaporation"],
            glacier_supply=out["glacier_supply"], rain=rain, snowfall=snowfall,
            bucket_capacity=props["bucketdepth"])
        return (PhysicsTendency.zeros(state.temperature.shape),
                {**diagnostics, SLIM_LAND_KEY: new})
