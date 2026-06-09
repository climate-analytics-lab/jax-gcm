"""Emulated radiation scheme using bidirectional GRU neural networks.

Drop-in replacement for ``radiation_scheme_rrtmgp`` that uses trained neural
networks to predict shortwave and longwave fluxes. The NN weights are passed
as JAX arrays through the ``emulator_weights`` argument, making them
fully differentiable for gradient-based optimization.

Reference architecture: Ukkonen (2024), https://github.com/peterukk/rte-rrtmgp-nn

Date: 2026-04-11
"""

from typing import Tuple, Optional

import jax.numpy as jnp

from jcm.physics.radiation.radiation_types import (
    RadiationParameters,
    RadiationTendencies,
    RadiationData,
)
from jcm.physics.radiation.nn_emulator import (
    EmulatorWeights,
    InputScaling,
    preprocess_sw_inputs,
    preprocess_lw_inputs,
    sw_emulator_column,
    lw_emulator_column,
    reconstruct_sw_fluxes,
    reconstruct_lw_fluxes,
    flux_to_heating_rate,
)
import jcm.constants as c


def radiation_scheme_emulated(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_albedo_vis: jnp.ndarray,
    surface_albedo_nir: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    solar,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6,
    emulator_weights: Optional[EmulatorWeights] = None,
    sw_scaling: Optional[InputScaling] = None,
    lw_scaling: Optional[InputScaling] = None,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Emulated radiation scheme — drop-in replacement for ``radiation_scheme_rrtmgp``.

    Uses bidirectional GRU neural networks to predict shortwave and longwave
    fluxes, then derives heating rates from flux divergence. The call
    signature matches the other radiation schemes so it can be used
    interchangeably.

    Additional Args:
        emulator_weights: Trained NN weights (``EmulatorWeights``). Must be
            provided; passed through the parameters mechanism in EchamPhysics.
        sw_scaling: Input normalization for SW network.
        lw_scaling: Input normalization for LW network.
    """
    from jax_solar import OrbitalTime, radiation_flux, get_solar_sin_altitude

    nlev = temperature.shape[0]

    # --- Solar geometry ---
    # `solar` is a `jcm.forcing.SolarGeometry` precomputed by the Model;
    # the radiation scheme stays date-free.
    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    toa_flux = radiation_flux(
        orbital_time, longitude, latitude, parameters.solar_constant
    )
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = jnp.maximum(sin_altitude, parameters.min_cos_zenith)

    # --- Prepare inputs common to SW and LW ---
    # Water vapour mixing ratio
    eps = c.eps  # Mv/Md ≈ 0.622
    h2o_vmr = specific_humidity / (eps * (1.0 - specific_humidity) + specific_humidity)

    # Ozone
    if ozone_vmr is None:
        ozone_vmr = jnp.full(nlev, 5e-6)

    # Cloud water/ice paths (kg/m^2)
    cwp = cloud_water * air_density * layer_thickness * cloud_fraction
    cip = cloud_ice * air_density * layer_thickness * cloud_fraction

    # Default scaling if not provided
    if sw_scaling is None:
        sw_scaling = InputScaling(x_max=jnp.ones(7))
    if lw_scaling is None:
        lw_scaling = InputScaling(x_max=jnp.ones(7))

    # --- Shortwave ---
    sw_input = preprocess_sw_inputs(
        temperature, pressure_levels, h2o_vmr, ozone_vmr,
        cwp, cip, cos_zenith, sw_scaling,
    )
    surface_albedo = 0.5 * (surface_albedo_vis + surface_albedo_nir)
    sw_nn_output = sw_emulator_column(
        sw_input, jnp.atleast_1d(surface_albedo), emulator_weights.sw,
    )
    toa_sw_down = jnp.maximum(toa_flux, 0.0)
    sw_flux_down, sw_flux_up = reconstruct_sw_fluxes(
        sw_nn_output, toa_sw_down, surface_albedo,
    )

    # --- Longwave ---
    lw_input = preprocess_lw_inputs(
        temperature, pressure_levels, h2o_vmr, ozone_vmr,
        cwp, cip, co2_vmr, lw_scaling,
    )
    lw_nn_output = lw_emulator_column(
        lw_input, jnp.atleast_1d(surface_emissivity), emulator_weights.lw,
    )
    lw_flux_down, lw_flux_up = reconstruct_lw_fluxes(
        lw_nn_output, surface_temperature, surface_emissivity,
    )

    # --- Heating rates ---
    sw_heating = flux_to_heating_rate(sw_flux_down, sw_flux_up, pressure_interfaces)
    lw_heating = flux_to_heating_rate(lw_flux_down, lw_flux_up, pressure_interfaces)
    total_heating = sw_heating + lw_heating

    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating,
    )

    diagnostics = RadiationData(
        cos_zenith=cos_zenith,
        surface_albedo_vis=jnp.atleast_1d(surface_albedo_vis),
        surface_albedo_nir=jnp.atleast_1d(surface_albedo_nir),
        surface_emissivity=jnp.atleast_1d(surface_emissivity),
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
        lw_heating_rate=lw_heating,
        surface_sw_down=sw_flux_down[-1],
        surface_lw_down=lw_flux_down[-1],
        surface_sw_up=sw_flux_up[-1],
        surface_lw_up=lw_flux_up[-1],
        toa_sw_up=sw_flux_up[0],
        toa_lw_up=lw_flux_up[0],
        toa_sw_down=toa_sw_down,
        # NN emulator returns only all-sky fluxes; running it twice
        # (with and without cloud condensate) for clear-sky CRE values
        # is a follow-up. Zeros for now so downstream consumers don't
        # see stale data in the diagnostic key.
        toa_sw_up_clear=jnp.zeros_like(sw_flux_up[0]),
        toa_lw_up_clear=jnp.zeros_like(lw_flux_up[0]),
        # ``step`` is owned by the enclosing ``NNEmulatorRadiation``
        # carry — the standalone scheme emits 0 and the term bumps it
        # after its compute-vs-cache cond.
        step=jnp.int32(0),
    )

    return tendencies, diagnostics


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.clouds.cloud_data import radiation_cloud_fields  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics.radiation import (  # noqa: E402
    cached_radiation_tendency,
    radiation_should_compute,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


def _column_vector_emulated(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))


class NNEmulatorRadiation(PhysicsTerm):
    """Bidirectional-GRU neural network radiation emulator as a PhysicsTerm.

    Drop-in replacement for :class:`GreyTwoStreamRadiation` /
    :class:`RRTMGPRadiation` that uses a pre-trained NN to predict
    SW + LW fluxes per column, then derives heating rates from flux
    divergence. Cheap and differentiable. Reads the same diagnostics
    set as the other radiation terms; the emulator weights / scaling
    live on ``parameters.radiation``.
    """

    name: ClassVar[str] = "nn_emulator_radiation"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "layer_thickness",
        "air_density", "chemistry", "aerosol",
        "radiation", "surface", "clouds",
    )
    provides: ClassVar[tuple[str, ...]] = ("radiation", "clouds")

    def __init__(self, params: RadiationParameters | None = None):
        """Hold the scheme-native :class:`RadiationParameters` (with NN weights)."""
        self.params = nnx.Param(params or RadiationParameters.default())
        self._coords_cached = False

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (deg) for the radiation scheme."""
        lat_deg = jnp.asarray(coords.horizontal.latitudes) * 180.0 / jnp.pi
        lon_deg = jnp.asarray(coords.horizontal.longitudes) * 180.0 / jnp.pi
        lat_2d, lon_2d = jnp.meshgrid(lat_deg, lon_deg)
        self._lats = nnx.Variable(lat_2d.reshape(-1))
        self._lons = nnx.Variable(lon_2d.reshape(-1))
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute or reuse cached NN-emulated heating rates."""
        params = self.params.get_value()
        radiation = diagnostics["radiation"]

        def _compute():
            return self._compute_full(state, diagnostics, forcing, params)

        def _use_cached():
            return cached_radiation_tendency(
                radiation, state.temperature.shape,
            ), radiation

        tendency, new_radiation = jax.lax.cond(
            radiation_should_compute(diagnostics, params),
            _compute, _use_cached,
        )
        # Advance the radiation-local step counter on every call (both
        # compute and cached paths). Mirrors the carry-side step bump
        # in the grey two-stream / RRTMGP radiation terms so the
        # sub-stepping gate sees the same cadence regardless of scheme.
        new_radiation = new_radiation.copy(step=radiation.step + 1)
        # Mirror TOA fluxes onto the clouds sub-struct for CRE
        # diagnostics. The emulator only produces all-sky values, so
        # the clear-sky fields stay at zero until the 2-call clear-sky
        # extension is wired (follow-up).
        clouds = diagnostics["clouds"].copy(
            toa_sw_up_all=new_radiation.toa_sw_up,
            toa_sw_up_clear=new_radiation.toa_sw_up_clear,
            toa_lw_up_all=new_radiation.toa_lw_up,
            toa_lw_up_clear=new_radiation.toa_lw_up_clear,
        )
        return tendency, {
            **diagnostics, "radiation": new_radiation, "clouds": clouds,
        }

    def _compute_full(
        self, state, diagnostics, forcing, params,
    ):
        """Run the full NN-emulator scheme, return (tendency, RadiationData)."""
        nlev, ncols = state.temperature.shape

        latitudes = self._lats.get_value()
        longitudes = self._lons.get_value()
        solar = forcing.solar

        cloud_water, cloud_ice, cloud_fraction = radiation_cloud_fields(
            state, diagnostics,
        )

        chemistry = diagnostics["chemistry"]
        ozone_vmr = chemistry.ozone_vmr * 1e-6
        co2_vmr = jnp.mean(chemistry.co2_vmr) * 1e-6

        surface_temperature_col = (
            diagnostics["surface"].surface_temperature.reshape(ncols)
        )
        radiation_in = diagnostics["radiation"]
        surface_albedo_vis_col = radiation_in.surface_albedo_vis.reshape(ncols)
        surface_albedo_nir_col = radiation_in.surface_albedo_nir.reshape(ncols)
        surface_emissivity_col = radiation_in.surface_emissivity.reshape(ncols)

        aerosol_in = diagnostics["aerosol"]
        aerosol_for_vmap = aerosol_in.copy(
            aod_profile=aerosol_in.aod_profile.reshape(nlev, ncols).T,
            ssa_profile=aerosol_in.ssa_profile.reshape(nlev, ncols).T,
            asy_profile=aerosol_in.asy_profile.reshape(nlev, ncols).T,
            cdnc_factor=aerosol_in.cdnc_factor.reshape(ncols),
            aod_total=aerosol_in.aod_total.reshape(ncols),
            aod_anthropogenic=aerosol_in.aod_anthropogenic.reshape(ncols),
            aod_background=aerosol_in.aod_background.reshape(ncols),
            angstrom=aerosol_in.angstrom.reshape(ncols),
        )

        emulator_weights = params.emulator_weights
        sw_scaling = params.sw_scaling
        lw_scaling = params.lw_scaling

        tendencies_vmapped, diagnostics_vmapped = jax.vmap(
            radiation_scheme_emulated,
            in_axes=(
                1, 1, 1, 1, 1,
                1, 1, 1, 1,
                0, 0, 0, 0,
                None, 0, 0,
                None, 0, 1, None,
                None, None, None,
            ),
            out_axes=(0, 0),
            axis_size=ncols,
        )(
            state.temperature, state.specific_humidity,
            diagnostics["pressure_full"], diagnostics["pressure_half"],
            diagnostics["layer_thickness"], diagnostics["air_density"],
            cloud_water, cloud_ice, cloud_fraction,
            surface_temperature_col, surface_albedo_vis_col,
            surface_albedo_nir_col, surface_emissivity_col,
            solar, latitudes, longitudes,
            params, aerosol_for_vmap, ozone_vmr, co2_vmr,
            emulator_weights, sw_scaling, lw_scaling,
        )

        rad_out = RadiationData(
            cos_zenith=_column_vector_emulated(
                diagnostics_vmapped.cos_zenith, ncols,
            ),
            surface_albedo_vis=_column_vector_emulated(
                diagnostics_vmapped.surface_albedo_vis, ncols,
            ),
            surface_albedo_nir=_column_vector_emulated(
                diagnostics_vmapped.surface_albedo_nir, ncols,
            ),
            surface_emissivity=_column_vector_emulated(
                diagnostics_vmapped.surface_emissivity, ncols,
            ),
            sw_flux_up=diagnostics_vmapped.sw_flux_up.T,
            sw_flux_down=diagnostics_vmapped.sw_flux_down.T,
            sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
            lw_flux_up=diagnostics_vmapped.lw_flux_up.T,
            lw_flux_down=diagnostics_vmapped.lw_flux_down.T,
            lw_heating_rate=tendencies_vmapped.longwave_heating.T,
            surface_sw_down=_column_vector_emulated(
                diagnostics_vmapped.surface_sw_down, ncols,
            ),
            surface_lw_down=_column_vector_emulated(
                diagnostics_vmapped.surface_lw_down, ncols,
            ),
            surface_sw_up=_column_vector_emulated(
                diagnostics_vmapped.surface_sw_up, ncols,
            ),
            surface_lw_up=_column_vector_emulated(
                diagnostics_vmapped.surface_lw_up, ncols,
            ),
            toa_sw_up=_column_vector_emulated(
                diagnostics_vmapped.toa_sw_up, ncols,
            ),
            toa_lw_up=_column_vector_emulated(
                diagnostics_vmapped.toa_lw_up, ncols,
            ),
            toa_sw_down=_column_vector_emulated(
                diagnostics_vmapped.toa_sw_down, ncols,
            ),
            toa_sw_up_clear=_column_vector_emulated(
                diagnostics_vmapped.toa_sw_up_clear, ncols,
            ),
            toa_lw_up_clear=_column_vector_emulated(
                diagnostics_vmapped.toa_lw_up_clear, ncols,
            ),
            # Placeholder — the enclosing ``__call__`` overwrites
            # ``step`` after the compute-vs-cache cond.
            step=jnp.int32(0),
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=tendencies_vmapped.temperature_tendency.T,
            specific_humidity=jnp.zeros((nlev, ncols)),
            tracers={},
        )
        return tendency, rad_out
