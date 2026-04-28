"""RRTMGP-based radiation scheme for ICON physics.

This module integrates jax-rrtmgp with ICON's radiation interface, handling:
- Location-specific solar geometry via jax_solar (OrbitalTime, radiation_flux,
  get_solar_sin_altitude)
- ICON vertical ordering (TOA->surface) vs RRTMGP (surface->TOA) conversion
- Halo management (temperature NaN-padded for RRTMGP fill; others edge-filled)
- Stretched grid mapping for non-uniform vertical coordinates
- Unit conversions and cloud effective radii from ICON functions
- Output conversion to ICON's RadiationTendencies and RadiationData formats

Key entry point: ``radiation_scheme_rrtmgp`` -- ICON-signature drop-in
replacement for the grey ``radiation_scheme``.

Date: 2025-08-01
"""

from pathlib import Path
from typing import Tuple, Optional
import warnings

import jax.numpy as jnp
from jax import lax

from jax_solar import OrbitalTime, radiation_flux, get_solar_sin_altitude
from jcm.physics.radiation.radiation_types import (
    RadiationParameters,
    RadiationTendencies,
)
from jcm.physics.icon.icon_physics_data import RadiationData
from jcm.physics.radiation.grey_two_stream.radiation_scheme import prepare_radiation_state
from jcm.physics.radiation.cloud_optics import (
    effective_radius_liquid,
    effective_radius_ice,
)
from jcm.constants import PhysicalConstants

import rrtmgp
from rrtmgp.config import radiative_transfer
from rrtmgp import stretched_grid_util
from rrtmgp.rrtmgp import RRTMGP

# ---------------------------------------------------------------------------
# Module-level RRTMGP instance (created once at import time)
# ---------------------------------------------------------------------------
_GLOBAL_RRTMGP_INSTANCE = None


def _ensure_rrtmgp():
    """Lazily initialise the global RRTMGP instance on first use."""
    global _GLOBAL_RRTMGP_INSTANCE
    if _GLOBAL_RRTMGP_INSTANCE is not None:
        return _GLOBAL_RRTMGP_INSTANCE

    rrtmgp_root = Path(rrtmgp.__path__[0])
    rrtmgp_data_path = rrtmgp_root / "optics" / "rrtmgp_data"
    test_data_path = rrtmgp_root / "optics" / "test_data"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _GLOBAL_RRTMGP_INSTANCE = RRTMGP(
            radiative_transfer_cfg=radiative_transfer.RadiativeTransfer(
                optics=radiative_transfer.OpticsParameters(
                    optics=radiative_transfer.RRTMOptics(
                        longwave_nc_filepath=str(
                            rrtmgp_data_path / "rrtmgp-gas-lw-g128.nc"
                        ),
                        shortwave_nc_filepath=str(
                            rrtmgp_data_path / "rrtmgp-gas-sw-g112.nc"
                        ),
                        cloud_longwave_nc_filepath=str(
                            rrtmgp_data_path / "cloudysky_lw.nc"
                        ),
                        cloud_shortwave_nc_filepath=str(
                            rrtmgp_data_path / "cloudysky_sw.nc"
                        ),
                    )
                ),
                atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
                    sfc_emis=0.98,
                    sfc_alb=0.07,
                    zenith=1.0,
                    irrad=1361.0,
                    toa_flux_lw=0.0,
                    vmr_global_mean_filepath=str(
                        test_data_path / "vmr_global_means.json"
                    ),
                ),
                save_lw_sw_heating_rates=True,
            ),
            dz=1.0,  # placeholder -- actual dz comes via stretched-grid map
            diagnostic_fields=(
                "surf_lw_flux_down_2d_xy",
                "surf_lw_flux_up_2d_xy",
                "surf_sw_flux_down_2d_xy",
                "surf_sw_flux_up_2d_xy",
                "toa_sw_flux_incoming_2d_xy",
                "toa_sw_flux_outgoing_2d_xy",
                "toa_lw_flux_outgoing_2d_xy",
            ),
        )
    return _GLOBAL_RRTMGP_INSTANCE


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_3d_with_nan_halo(
    arr_1d: jnp.ndarray, nlev: int, halo: int = 1
) -> jnp.ndarray:
    """Convert 1D profile to 3D (1,1,nz+2*halo) with NaN halos (for temperature)."""
    nzh = nlev + 2 * halo
    arr_3d = jnp.full((1, 1, nzh), jnp.nan)
    arr_3d = arr_3d.at[0, 0, halo : halo + nlev].set(arr_1d)
    return arr_3d


def _to_3d_with_filled_halo(
    arr_1d: jnp.ndarray, nlev: int, halo: int = 1
) -> jnp.ndarray:
    """Convert 1D profile to 3D (1,1,nz+2*halo) with edge-filled halos."""
    nzh = nlev + 2 * halo
    arr_3d = jnp.zeros((1, 1, nzh), dtype=arr_1d.dtype)
    arr_3d = arr_3d.at[0, 0, halo : halo + nlev].set(arr_1d)
    arr_3d = arr_3d.at[0, 0, 0].set(arr_1d[0])  # bottom halo
    arr_3d = arr_3d.at[0, 0, -1].set(arr_1d[-1])  # top halo
    return arr_3d


def _reverse_if_needed(pressure: jnp.ndarray) -> jnp.ndarray:
    """Return True if pressure increases with index (needs reversal for RRTMGP)."""
    return pressure[0] < pressure[-1]


# ---------------------------------------------------------------------------
# Data conversion: ICON -> RRTMGP
# ---------------------------------------------------------------------------

def prepare_rrtmgp_data(
    icon_data,
    layer_thickness: jnp.ndarray,
    cdnc_factor: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    land_fraction: float = 0.5,
) -> dict:
    """Convert ICON RadiationState to RRTMGP input dict.

    Handles vertical ordering, halo padding, stretched-grid mapping,
    water-variable conversions, and cloud effective radii.
    """
    nlev = icon_data.temperature.shape[0]
    halo = 1

    to3d_nan = lambda a: _to_3d_with_nan_halo(a, nlev, halo)  # noqa: E731
    to3d_fill = lambda a: _to_3d_with_filled_halo(a, nlev, halo)  # noqa: E731

    phys = PhysicalConstants()
    rho = icon_data.pressure / (phys.rgas * icon_data.temperature)

    # Vertical ordering: ICON is TOA->surface, RRTMGP expects surface->TOA
    needs_reversal = _reverse_if_needed(icon_data.pressure)
    flip = lambda a: a[::-1]  # noqa: E731
    identity = lambda a: a  # noqa: E731

    layer_thickness = lax.cond(needs_reversal, flip, identity, layer_thickness)
    rho = lax.cond(needs_reversal, flip, identity, rho)
    temperature_1d = lax.cond(needs_reversal, flip, identity, icon_data.temperature)
    pressure_1d = lax.cond(needs_reversal, flip, identity, icon_data.pressure)
    cwp_1d = lax.cond(needs_reversal, flip, identity, icon_data.cloud_water_path)
    cip_1d = lax.cond(needs_reversal, flip, identity, icon_data.cloud_ice_path)

    # Stretched-grid mapping for non-uniform vertical coordinates
    layer_thickness_3d = to3d_fill(layer_thickness)
    sg_map = {
        stretched_grid_util.hc_key(2): layer_thickness_3d,
        stretched_grid_util.hf_key(2): layer_thickness_3d,
    }

    # Cloud paths -> mixing ratios
    cloud_water_mixing = cwp_1d / (rho * layer_thickness)
    cloud_ice_mixing = cip_1d / (rho * layer_thickness)
    total_condensate = cloud_water_mixing + cloud_ice_mixing

    # Water vapour VMR -> mass mixing ratio: q = VMR * eps
    h2o_mass_mixing = icon_data.h2o_vmr * phys.eps
    h2o_mass_mixing = lax.cond(needs_reversal, flip, identity, h2o_mass_mixing)
    total_water = h2o_mass_mixing + total_condensate

    # Cloud effective radii (ICON parameterisations, microns -> metres)
    r_eff_liq = effective_radius_liquid(cdnc_factor, land_fraction)
    r_eff_ice = effective_radius_ice(
        temperature_1d,
        cip_1d / jnp.maximum(1.0, cwp_1d + cip_1d),
    )
    if jnp.asarray(r_eff_liq).ndim == 0:
        cloud_r_eff_liq = jnp.full((nlev,), r_eff_liq) * 1e-6
    else:
        r_liq_1d = jnp.asarray(r_eff_liq).reshape(-1)
        cloud_r_eff_liq = (
            jnp.full((nlev,), r_liq_1d[0])
            if r_liq_1d.shape[0] != nlev
            else r_liq_1d
        ) * 1e-6
    cloud_r_eff_ice = jnp.asarray(r_eff_ice).reshape(-1) * 1e-6

    return {
        "rho_xxc": to3d_fill(rho),
        "q_t": to3d_fill(total_water),
        "q_liq": to3d_fill(cloud_water_mixing),
        "q_ice": to3d_fill(cloud_ice_mixing),
        "q_c": to3d_fill(total_condensate),
        "cloud_r_eff_liq": to3d_fill(cloud_r_eff_liq),
        "cloud_r_eff_ice": to3d_fill(cloud_r_eff_ice),
        "temperature": to3d_nan(temperature_1d),
        "sfc_temperature": jnp.reshape(surface_temperature, (1, 1)),
        "p_ref_xxc": to3d_fill(pressure_1d),
        "sg_map": sg_map,
        "use_scan": True,
    }


# ---------------------------------------------------------------------------
# Data conversion: RRTMGP -> ICON
# ---------------------------------------------------------------------------

def prepare_icon_data(
    rrtmgp_data: dict,
    icon_data,
    surface_albedo_vis: jnp.ndarray,
    surface_albedo_nir: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Convert RRTMGP output dict back to ICON RadiationTendencies/RadiationData."""
    halo = 1
    nlev = icon_data.temperature.shape[0]
    cos_zenith = icon_data.cos_zenith[0]

    # Extract heating rates (strip halos)
    total_heating = rrtmgp_data["rad_heat_src"][0, 0, halo : halo + nlev]
    lw_heating = rrtmgp_data["rad_heat_lw_3d"][0, 0, halo : halo + nlev]
    sw_heating = rrtmgp_data["rad_heat_sw_3d"][0, 0, halo : halo + nlev]

    # Reverse back to ICON order if we reversed going in
    needs_reversal = _reverse_if_needed(icon_data.pressure)
    flip = lambda a: a[::-1]  # noqa: E731
    identity = lambda a: a  # noqa: E731

    total_heating = lax.cond(needs_reversal, flip, identity, total_heating)
    lw_heating = lax.cond(needs_reversal, flip, identity, lw_heating)
    sw_heating = lax.cond(needs_reversal, flip, identity, sw_heating)

    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating,
    )

    # Surface / TOA flux diagnostics
    surf_sw_down = rrtmgp_data["surf_sw_flux_down_2d_xy"][0, 0]
    surf_sw_up = rrtmgp_data["surf_sw_flux_up_2d_xy"][0, 0]
    surf_lw_down = rrtmgp_data["surf_lw_flux_down_2d_xy"][0, 0]
    surf_lw_up = rrtmgp_data["surf_lw_flux_up_2d_xy"][0, 0]
    toa_sw_down = rrtmgp_data["toa_sw_flux_incoming_2d_xy"][0, 0]
    toa_sw_up = rrtmgp_data["toa_sw_flux_outgoing_2d_xy"][0, 0]
    toa_lw_up = rrtmgp_data["toa_lw_flux_outgoing_2d_xy"][0, 0]

    # Full flux profiles (nlev+1 interfaces, ngpts bands)
    sw_flux_up = rrtmgp_data["sw_flux_up_full"][0, :, :].transpose(1, 0)
    sw_flux_down = rrtmgp_data["sw_flux_down_full"][0, :, :].transpose(1, 0)
    lw_flux_up = rrtmgp_data["lw_flux_up_full"][0, :, :].transpose(1, 0)
    lw_flux_down = rrtmgp_data["lw_flux_down_full"][0, :, :].transpose(1, 0)

    sw_flux_up = lax.cond(needs_reversal, flip, identity, sw_flux_up)
    sw_flux_down = lax.cond(needs_reversal, flip, identity, sw_flux_down)
    lw_flux_up = lax.cond(needs_reversal, flip, identity, lw_flux_up)
    lw_flux_down = lax.cond(needs_reversal, flip, identity, lw_flux_down)

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
        surface_sw_down=surf_sw_down,
        surface_lw_down=surf_lw_down,
        surface_sw_up=surf_sw_up,
        surface_lw_up=surf_lw_up,
        toa_sw_up=toa_sw_up,
        toa_lw_up=toa_lw_up,
        toa_sw_down=toa_sw_down,
    )
    return tendencies, diagnostics


# ---------------------------------------------------------------------------
# Core compute function
# ---------------------------------------------------------------------------

def radiation_scheme_rrtmgp_fn(
    rrtmgp_input: dict,
    toa_flux: jnp.ndarray,
    cos_zenith: jnp.ndarray,
) -> dict:
    """Call the global RRTMGP instance with per-column solar parameters."""
    rrtmgp_instance = _ensure_rrtmgp()
    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0))
    irrad_val = jnp.maximum(toa_flux, 0.0)
    return rrtmgp_instance.compute_heating_rate(
        zenith=zenith_angle, irrad=irrad_val, **rrtmgp_input
    )


# ---------------------------------------------------------------------------
# Main entry point (ICON-compatible signature)
# ---------------------------------------------------------------------------

def radiation_scheme_rrtmgp(
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
    date,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6,
) -> Tuple[RadiationTendencies, RadiationData]:
    """RRTMGP radiation scheme -- drop-in replacement for ``radiation_scheme``.

    Has the identical call signature so it can be used interchangeably with
    the grey/simplified radiation scheme.
    """
    # CDNC factor from aerosol data
    if aerosol_data.cdnc_factor.ndim == 0:
        cdnc_factor = jnp.array(aerosol_data.cdnc_factor)
    else:
        cdnc_factor = aerosol_data.cdnc_factor

    # Solar geometry via jax_solar
    actual_date = getattr(date, "dt", date)
    orbital_time = OrbitalTime.from_datetime(actual_date)
    toa_flux = radiation_flux(orbital_time, longitude, latitude, parameters.solar_constant)
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = sin_altitude  # cos(zenith) = sin(altitude)

    # Prepare ICON radiation state (shared with grey scheme)
    icon_state = prepare_radiation_state(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        cos_zenith=cos_zenith,
        ozone_vmr=ozone_vmr,
    )

    # Convert to RRTMGP input format
    rrtmgp_input = prepare_rrtmgp_data(
        icon_state,
        layer_thickness,
        cdnc_factor,
        surface_temperature,
    )

    # Run RRTMGP radiative transfer
    rrtmgp_output = radiation_scheme_rrtmgp_fn(rrtmgp_input, toa_flux, cos_zenith)

    # Convert outputs back to ICON format
    return prepare_icon_data(
        rrtmgp_output,
        icon_state,
        surface_albedo_vis,
        surface_albedo_nir,
        surface_emissivity,
    )
