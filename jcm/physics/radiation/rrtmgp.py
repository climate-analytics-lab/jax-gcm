"""RRTMGP-based radiation scheme for ECHAM physics.

This module integrates jax-rrtmgp with ICON's radiation interface, handling:
- Location-specific solar geometry via jax_solar (OrbitalTime,
  direct_solar_irradiance, get_solar_sin_altitude)
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

import jax
import jax.numpy as jnp

from jcm.physics.coords_util import column_lat_lon
from jax import lax

from jax_solar import OrbitalTime, direct_solar_irradiance, get_solar_sin_altitude
from jcm.physics.clouds.cloud_data import radiation_cloud_fields
from jcm.physics.radiation.radiation_types import (
    RadiationParameters,
    RadiationTendencies,
    RadiationData,
)
from jcm.physics.radiation.grey_two_stream.radiation_scheme import prepare_radiation_state
from jcm.physics.radiation.mcica import (
    column_key,
    generate_subcolumns,
    in_cloud_path,
)
from jcm.physics.radiation.radiation_types import cloud_overlap_name
from jcm.physics.radiation.cloud_optics import (
    effective_radius_liquid,
    effective_radius_ice,
)
import jcm.constants as c

import rrtmgp
from rrtmgp.config import radiative_transfer
from rrtmgp import stretched_grid_util
from rrtmgp.rrtmgp import RRTMGP

# Cap on in-cloud condensate (kg/kg) handed to the cloud optics — the high end
# of realistic in-cloud water; bounds the cloud optical depth of thin clouds
# carrying large grid-mean condensate so the two-stream solver can't NaN. The
# faithful-radiation equivalent of ECHAM's optics inhomogeneity factor + r_eff
# table clamp. Applied in ``radiation_scheme_rrtmgp`` after ``in_cloud_path``.
_MAX_IN_CLOUD_CONDENSATE = 1.0e-2


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


def _to_3d_pressure_halo(
    pressure_1d: jnp.ndarray,
    dp_bottom: jnp.ndarray,
    dp_top: jnp.ndarray,
    nlev: int,
    halo: int = 1,
) -> jnp.ndarray:
    """Halo-pad the pressure profile so boundary-layer Δp comes out EXACT.

    jax-rrtmgp computes layer thickness with the centered difference
    ``dp[k] = 0.5·|p[k+1] − p[k−1]|``, so the halo values fully determine
    the Δp the heating rate sees in the top and bottom layers. Rather than
    approximating them (edge fill halves the Δp of a uniform grid; linear
    extrapolation ``2p[0] − p[1]`` is exact only for uniform spacing and
    on the hybrid L47 grid is ~75 % too thick at the surface and goes
    NEGATIVE at the log-spaced top), set the halo so the centered
    difference reproduces the model's TRUE half-level layer thickness:

        0.5·|p[k∓1] − halo| = Δp_true  →  halo = p[1] + 2·Δp_bottom
                                          halo = p[-2] − 2·Δp_top

    with ``Δp_bottom`` / ``Δp_top`` taken from ``pressure_interfaces``
    (``pressure_1d`` is surface→TOA here, so index 0 is the surface).
    The tiny positive floor on the top halo only binds if the top layer
    is thicker than half the distance to the level below — genuinely
    pathological — and keeps downstream log-pressure interpolation finite.
    """
    nzh = nlev + 2 * halo
    arr_3d = jnp.zeros((1, 1, nzh), dtype=pressure_1d.dtype)
    arr_3d = arr_3d.at[0, 0, halo : halo + nlev].set(pressure_1d)
    arr_3d = arr_3d.at[0, 0, 0].set(pressure_1d[1] + 2.0 * dp_bottom)
    arr_3d = arr_3d.at[0, 0, -1].set(
        jnp.maximum(pressure_1d[-2] - 2.0 * dp_top, 1e-3 * pressure_1d[-1])
    )
    return arr_3d


def _to_4d_per_gpoint(
    per_gpt_2d: jnp.ndarray, nlev: int, halo: int = 1,
) -> jnp.ndarray:
    """Halo-pad ``[n_gpt, nlev]`` → ``[n_gpt, 1, 1, nlev + 2*halo]``.

    Per-gpoint analogue of :func:`_to_3d_with_filled_halo` for the McICA
    cloud-path inputs to the rrtmgp library: edge-fill the halo by
    repeating the closest interior value. The library indexes into the
    leading g-point axis inside its g-point loop.
    """
    n_gpt = per_gpt_2d.shape[0]
    nzh = nlev + 2 * halo
    out = jnp.zeros((n_gpt, 1, 1, nzh), dtype=per_gpt_2d.dtype)
    out = out.at[:, 0, 0, halo : halo + nlev].set(per_gpt_2d)
    out = out.at[:, 0, 0, 0].set(per_gpt_2d[:, 0])
    out = out.at[:, 0, 0, -1].set(per_gpt_2d[:, -1])
    return out


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
    r_eff_liq_um: Optional[jnp.ndarray] = None,
    r_eff_ice_um: Optional[jnp.ndarray] = None,
) -> dict:
    """Convert ICON RadiationState to RRTMGP input dict.

    Handles vertical ordering, halo padding, stretched-grid mapping,
    water-variable conversions, and cloud effective radii.

    Args:
        icon_data: RadiationState with TOA-first profiles.
        layer_thickness: geometric layer thickness (m), TOA-first.
        cdnc_factor: aerosol CDNC scaling for the liquid r_eff fallback.
        surface_temperature: scalar surface temperature (K).
        land_fraction: land fraction for the liquid r_eff fallback.
        r_eff_liq_um: optional microphysical liquid effective radius (um),
            TOA-first (nlev,). Entries <= 0 mean "not provided" and fall
            back to the diagnostic ``effective_radius_liquid``.
        r_eff_ice_um: optional microphysical ice effective radius (um),
            TOA-first (nlev,). Entries <= 0 fall back to the Moss/Foot
            in-cloud-IWC formula (``effective_radius_ice``).

    """
    nlev = icon_data.temperature.shape[0]
    halo = 1

    if r_eff_liq_um is None:
        r_eff_liq_um = jnp.zeros((nlev,))
    if r_eff_ice_um is None:
        r_eff_ice_um = jnp.zeros((nlev,))

    to3d_nan = lambda a: _to_3d_with_nan_halo(a, nlev, halo)  # noqa: E731
    to3d_fill = lambda a: _to_3d_with_filled_halo(a, nlev, halo)  # noqa: E731

    rho = icon_data.pressure / (c.rd * icon_data.temperature)

    # Vertical ordering: ICON is TOA->surface, RRTMGP expects surface->TOA
    needs_reversal = _reverse_if_needed(icon_data.pressure)
    flip = lambda a: a[::-1]  # noqa: E731
    identity = lambda a: a  # noqa: E731

    layer_thickness = lax.cond(needs_reversal, flip, identity, layer_thickness)
    rho = lax.cond(needs_reversal, flip, identity, rho)
    temperature_1d = lax.cond(needs_reversal, flip, identity, icon_data.temperature)
    pressure_1d = lax.cond(needs_reversal, flip, identity, icon_data.pressure)
    # Surface-first half-level pressures, for the exact boundary-layer Δp
    # the pressure halo must encode (see _to_3d_pressure_halo).
    interfaces_1d = lax.cond(
        needs_reversal, flip, identity, icon_data.pressure_interfaces,
    )
    dp_bottom = interfaces_1d[0] - interfaces_1d[1]
    dp_top = interfaces_1d[-2] - interfaces_1d[-1]
    cwp_1d = lax.cond(needs_reversal, flip, identity, icon_data.cloud_water_path)
    cip_1d = lax.cond(needs_reversal, flip, identity, icon_data.cloud_ice_path)
    r_eff_liq_um = lax.cond(needs_reversal, flip, identity, r_eff_liq_um)
    r_eff_ice_um = lax.cond(needs_reversal, flip, identity, r_eff_ice_um)

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
    h2o_mass_mixing = icon_data.h2o_vmr * c.eps
    h2o_mass_mixing = lax.cond(needs_reversal, flip, identity, h2o_mass_mixing)
    total_water = h2o_mass_mixing + total_condensate

    # Cloud effective radii (microns -> metres). Microphysical values from
    # the clouds carry (ECHAM preffl/preffi, written by the 2M scheme) take
    # precedence where provided (> 0); otherwise fall back to the diagnostic
    # parameterisations. The ice fallback is ECHAM's Moss/Foot power law on
    # the IN-CLOUD ice water content in g/m3 — ``cip_1d`` is already the
    # in-cloud ice water path per layer (kg/m2; the caller divides the
    # grid-mean condensate by cloud fraction before building the state), so
    # IWC = path / dz with a kg -> g conversion and NO further cf division.
    # The jax-rrtmgp library clips both radii to its LUT bounds internally
    # (radius for liquid, 2*r as diameter for ice), so no clamp is applied
    # here.
    fallback_liq = jnp.broadcast_to(
        jnp.asarray(effective_radius_liquid(cdnc_factor, land_fraction)),
        (nlev,),
    )
    iwc_gm3 = cip_1d / jnp.maximum(layer_thickness, 1.0) * 1e3
    fallback_ice = effective_radius_ice(iwc_gm3)
    r_eff_liq = jnp.where(r_eff_liq_um > 0.0, r_eff_liq_um, fallback_liq)
    r_eff_ice = jnp.where(r_eff_ice_um > 0.0, r_eff_ice_um, fallback_ice)
    cloud_r_eff_liq = r_eff_liq * 1e-6
    cloud_r_eff_ice = r_eff_ice * 1e-6

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
        # Pressure halo encodes the model's exact half-level boundary-layer
        # thicknesses — see _to_3d_pressure_halo (edge fill / linear
        # extrapolation both misstate boundary Δp on the hybrid grid).
        "p_ref_xxc": _to_3d_pressure_halo(
            pressure_1d, dp_bottom, dp_top, nlev, halo,
        ),
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

    # Full flux profiles. RRTMGP returns shape (1, ngpt, nlev+1); we sum
    # over the ngpt (g-point) axis here — *before* the per-column vmap
    # bundles the result — so the vmapped diagnostic stays at
    # (ncols, nlev+1) instead of blowing up to (ncols, nlev+1, ngpt).
    # ngpt is 128 (LW) / 112 (SW), so this is a ~120× memory saving on
    # the radiation flux outputs. The downstream RadiationData consumer
    # (`echam_physics._apply_radiation_rrtmgp_inner`) already calls
    # `.sum(axis=-1)` on these, so the per-gpoint detail was being
    # discarded immediately anyway.
    sw_flux_up = rrtmgp_data["sw_flux_up_full"][0, :, :].sum(axis=0)
    sw_flux_down = rrtmgp_data["sw_flux_down_full"][0, :, :].sum(axis=0)
    lw_flux_up = rrtmgp_data["lw_flux_up_full"][0, :, :].sum(axis=0)
    lw_flux_down = rrtmgp_data["lw_flux_down_full"][0, :, :].sum(axis=0)

    sw_flux_up = lax.cond(needs_reversal, flip, identity, sw_flux_up)
    sw_flux_down = lax.cond(needs_reversal, flip, identity, sw_flux_down)
    lw_flux_up = lax.cond(needs_reversal, flip, identity, lw_flux_up)
    lw_flux_down = lax.cond(needs_reversal, flip, identity, lw_flux_down)

    diagnostics = RadiationData(
        # Match the grey scheme's shape convention so the downstream
        # vmap+squeeze(-1) in apply_radiation_rrtmgp resolves to (ncols,).
        # Grey emits cos_zenith with a trailing newaxis but passes the
        # surface scalars through bare; replicate exactly so the cached
        # branch in `_radiation_with_caching` matches our shape.
        cos_zenith=jnp.atleast_1d(cos_zenith),
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
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
        # The all-sky values come from the blended rrtmgp_data dict;
        # the caller (``radiation_scheme_rrtmgp``) overwrites the
        # clear-sky fields below with the actual clear-beam values via
        # ``.copy(...)``. Zero placeholders here keep the tree-shape
        # consistent in case someone calls ``prepare_icon_data``
        # outside the beam-split context.
        toa_sw_up_clear=jnp.zeros_like(toa_sw_up),
        toa_lw_up_clear=jnp.zeros_like(toa_lw_up),
        # ``step`` is owned by the enclosing ``RRTMGPRadiation`` carry —
        # the standalone scheme emits 0 and the term bumps the counter
        # after its compute-vs-cache cond.
        step=jnp.int32(0),
    )
    return tendencies, diagnostics


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
    solar,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    column_index: jnp.ndarray = jnp.int32(0),
    model_step: jnp.ndarray = jnp.int32(0),
    base_seed: int = 0,
    compute_cre: bool = True,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: Optional[jnp.ndarray] = None,
    ch4_vmr: Optional[jnp.ndarray] = None,
    n2o_vmr: Optional[jnp.ndarray] = None,
    r_eff_liq_um: Optional[jnp.ndarray] = None,
    r_eff_ice_um: Optional[jnp.ndarray] = None,
) -> Tuple[RadiationTendencies, RadiationData]:
    """RRTMGP radiation scheme — canonical McICA partial-cloud treatment.

    Partial-cloud handling: full McICA via the per-g-point cloud-path
    hook in the upstream rrtmgp library. ``mcica.generate_subcolumns``
    builds one stochastic binary cloud profile per g-point (LW and SW
    treated separately because their g-point counts differ); each
    g-point's RRTMGP solve sees only that sub-column's cloud condensate.
    Averaging across g-points naturally recovers the overlap-aware
    fluxes — at one RRTMGP call per radiation step rather than the
    previous 2-call beam-split's two.

    When ``compute_cre`` is True an extra clear-sky RRTMGP call (with
    zero condensate everywhere) populates ``toa_{sw,lw}_up_clear`` for
    the cloud radiative effect diagnostic. Costs 2× a McICA call;
    disable it (e.g. for production runs that only need the all-sky
    fluxes) for the 1× option.

    Args (additions over the previous beam-split signature):
        column_index: integer global index of the column being computed,
            used to seed the stochastic sub-column generator
            deterministically. Vmapped over the column axis upstream.
        model_step: scalar integer model step, also folded into the
            McICA seed so reruns are bit-exact reproducible.
        base_seed: term-level Python integer seed that the column +
            step indices fold into.
        compute_cre: if True, run an additional clear-sky RRTMGP call
            and populate ``toa_{sw,lw}_up_clear`` on the returned
            ``RadiationData``.
        r_eff_liq_um / r_eff_ice_um: optional microphysical effective
            radii (um, TOA-first (nlev,)) from the clouds carry (ECHAM
            preffl/preffi written by the 2M scheme; lagged one step by the
            carry). Levels <= 0 mean "not provided" and use the diagnostic
            fallbacks in ``prepare_rrtmgp_data``.

    """
    # CDNC factor from aerosol data
    if aerosol_data.cdnc_factor.ndim == 0:
        cdnc_factor = jnp.array(aerosol_data.cdnc_factor)
    else:
        cdnc_factor = aerosol_data.cdnc_factor

    # Solar geometry via jax_solar. `solar` is a `jcm.forcing.SolarGeometry`
    # precomputed by the Model; the radiation scheme stays date-free.
    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    # ``irrad`` must be the NORMAL-INCIDENCE (distance-corrected) solar
    # irradiance: jax-rrtmgp applies the cosine factor itself in the
    # direct-beam boundary condition (flux_down_direct_bc = irrad · µ0).
    # ``jax_solar.radiation_flux`` already includes ·µ0 (it returns flux on
    # a horizontal surface), so passing it here multiplied the cosine in
    # TWICE — TOA insolation ∝ µ0², a ~110 W/m² global-mean SW deficit
    # (hemispheric mean S0/6 instead of S0/4).
    direct_irradiance = direct_solar_irradiance(
        solar.orbital_phase, parameters.solar_constant
    )
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = sin_altitude  # cos(zenith) = sin(altitude)

    # In-cloud condensate (grid-mean / cf) is what each cloudy
    # sub-column sees; the binary McICA mask then re-imposes a
    # cloud-or-clear partitioning per g-point. ``in_cloud_path`` already
    # zeros the (essentially) clear cells (cf <= 2*eps; ECHAM mo_psrad).
    #
    # A *thin* but resolved cloud (cf ~ 0.01-0.05) carrying a lot of grid-mean
    # condensate still yields a very large in-cloud water (grid_mean / cf), and
    # the resulting extreme cloud optical depth NaNs the two-stream solver.
    # ECHAM bounds the radiative effect of such cells via the cloud-optics
    # sub-grid inhomogeneity factor (``zinhoml = LWP^-p``) and the r_eff table
    # clamp; we apply the equivalent guard as a direct cap on the in-cloud
    # condensate handed to the optics. ``_MAX_IN_CLOUD_CONDENSATE`` = 10 g/kg is
    # the high end of realistic in-cloud water, so genuine clouds are untouched
    # and only the pathological inflation is clipped.
    cloud_water_in_cloud = jnp.minimum(
        in_cloud_path(cloud_water, cloud_fraction), _MAX_IN_CLOUD_CONDENSATE
    )
    cloud_ice_in_cloud = jnp.minimum(
        in_cloud_path(cloud_ice, cloud_fraction), _MAX_IN_CLOUD_CONDENSATE
    )

    icon_state = prepare_radiation_state(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        pressure_interfaces=pressure_interfaces,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water_in_cloud,
        cloud_ice=cloud_ice_in_cloud,
        cloud_fraction=cloud_fraction,
        cos_zenith=cos_zenith,
        ozone_vmr=ozone_vmr,
    )

    # Stochastic sub-column generation (LW / SW use separate keys
    # because their g-point counts and resulting masks differ).
    rrtmgp_instance = _ensure_rrtmgp()
    n_gpt_lw = rrtmgp_instance.optics_lib.n_gpt_lw
    n_gpt_sw = rrtmgp_instance.optics_lib.n_gpt_sw

    # Frozen-step option for calibration: with mcica_freeze_step != 0
    # the mask key stops depending on the model step (see
    # RadiationParameters.mcica_freeze_step).
    model_step_eff = jnp.where(
        parameters.mcica_freeze_step != 0.0,
        jnp.zeros_like(model_step),
        model_step,
    )
    col_key = column_key(
        jax.random.PRNGKey(base_seed),
        model_step=model_step_eff, column_index=column_index,
    )
    key_lw, key_sw = jax.random.split(col_key)
    overlap_str = cloud_overlap_name(int(parameters.cloud_overlap))
    decorrelation_km = float(parameters.cloud_decorrelation_km)

    masks_lw = generate_subcolumns(
        cloud_fraction, layer_thickness,
        n_subcols=n_gpt_lw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, key=key_lw,
    )    # [n_gpt_lw, nlev], TOA-first
    masks_sw = generate_subcolumns(
        cloud_fraction, layer_thickness,
        n_subcols=n_gpt_sw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, key=key_sw,
    )    # [n_gpt_sw, nlev], TOA-first

    # Per-gpoint cloud paths in surface-first convention (the library's
    # internal expectation, see the flip in ``prepare_rrtmgp_data``).
    needs_reversal = _reverse_if_needed(icon_state.pressure)
    flip_per_gpt = lambda a: a[:, ::-1]  # noqa: E731
    identity = lambda a: a  # noqa: E731

    masks_lw_lib = lax.cond(
        needs_reversal, flip_per_gpt, identity, masks_lw,
    )
    masks_sw_lib = lax.cond(
        needs_reversal, flip_per_gpt, identity, masks_sw,
    )
    in_cloud_lwp_lib = lax.cond(
        needs_reversal, lambda a: a[::-1], identity,
        icon_state.cloud_water_path,
    )
    in_cloud_ipath_lib = lax.cond(
        needs_reversal, lambda a: a[::-1], identity,
        icon_state.cloud_ice_path,
    )

    nlev = icon_state.temperature.shape[0]
    halo = 1
    cpl_lw_4d = _to_4d_per_gpoint(
        masks_lw_lib * in_cloud_lwp_lib[jnp.newaxis, :], nlev, halo,
    )
    cpi_lw_4d = _to_4d_per_gpoint(
        masks_lw_lib * in_cloud_ipath_lib[jnp.newaxis, :], nlev, halo,
    )
    cpl_sw_4d = _to_4d_per_gpoint(
        masks_sw_lib * in_cloud_lwp_lib[jnp.newaxis, :], nlev, halo,
    )
    cpi_sw_4d = _to_4d_per_gpoint(
        masks_sw_lib * in_cloud_ipath_lib[jnp.newaxis, :], nlev, halo,
    )

    rrtmgp_input = prepare_rrtmgp_data(
        icon_state, layer_thickness, cdnc_factor, surface_temperature,
        r_eff_liq_um=r_eff_liq_um, r_eff_ice_um=r_eff_ice_um,
    )
    # The broadcast q_liq / q_ice are shadowed by the per-gpoint
    # arrays inside the g-point loop, so set them to zero. The clear-
    # sky branch (no per-gpoint args, just q_liq=0/q_ice=0) gives the
    # all-clear fluxes used for CRE.
    #
    # q_t must drop the condensate along with q_c: the library derives the
    # water-vapour VMR as (q_t − q_c)/(1 − q_t), so zeroing q_c while q_t
    # keeps ``vapor + in-cloud condensate`` makes gas optics absorb on the
    # condensate as if it were vapour — in every g-point (the cloud is
    # already represented by the per-gpoint McICA paths) and, worse, in the
    # clear-sky CRE call. Vapour-only q_t leaves the condensate's radiative
    # effect entirely to the cloud-optics paths.
    zero_3d = jnp.zeros_like(rrtmgp_input["q_liq"])
    rrtmgp_input["q_t"] = rrtmgp_input["q_t"] - rrtmgp_input["q_c"]
    rrtmgp_input["q_liq"] = zero_3d
    rrtmgp_input["q_ice"] = zero_3d
    rrtmgp_input["q_c"] = zero_3d

    # Per-cell gas concentrations (#483 + jax-rrtmgp PR #4). The library
    # merges this dict over the sounding-based defaults before calling
    # gas optics; ``h2o`` is always overridden internally from ``q_t``.
    # Each profile is shaped (1, 1, nz+2*halo) to match the rest of
    # rrtmgp_input.
    #
    # ORIENTATION: the library frame is surface-first (z index 0 = bottom;
    # see the flips in ``prepare_rrtmgp_data``), while jcm physics columns
    # are TOA-first. Every per-level profile handed to the library below
    # (gas VMRs and per-band aerosol) must flip under the SAME
    # ``needs_reversal`` as temperature/pressure/clouds. These two groups
    # previously skipped the flip: the chemistry ozone profile entered gas
    # optics upside down, and the per-band aerosol landed with its
    # surface-concentrated tau at the model top — the resulting spurious
    # top-level LW cooling from JAM's dust/BC LW bands grew a 2-grid
    # oscillation at 1 Pa that NaN'd coupled JAM runs by day ~10 (and the
    # same inversion put MACv2-SP's SW tau at the top of every RRTMGP run).
    flip_profile = lambda a: a[::-1]  # noqa: E731
    flip_per_band = lambda a: a[:, ::-1]  # noqa: E731
    identity_fn = lambda a: a  # noqa: E731

    def _oriented(profile_1d: jnp.ndarray) -> jnp.ndarray:
        """TOA-first (nlev,) profile → library surface-first orientation."""
        return lax.cond(needs_reversal, flip_profile, identity_fn, profile_1d)

    def _oriented_per_band(arr_2d: jnp.ndarray) -> jnp.ndarray:
        """TOA-first (n_bnd, nlev) → library surface-first orientation."""
        return lax.cond(needs_reversal, flip_per_band, identity_fn, arr_2d)

    halo = 1
    nlev = icon_state.temperature.shape[0]
    vmr_fields: dict[str, jnp.ndarray] = {}
    if ozone_vmr is not None:
        # ``ozone_vmr`` arrives as a (nlev,) TOA-first profile in mole
        # fraction; reorient before halo-padding.
        vmr_fields["o3"] = _to_3d_with_filled_halo(
            _oriented(ozone_vmr), nlev, halo,
        )
    if co2_vmr is not None:
        vmr_fields["co2"] = _to_3d_with_filled_halo(
            _oriented(jnp.broadcast_to(co2_vmr, (nlev,))), nlev, halo,
        )
    if ch4_vmr is not None:
        # CH4 is a real profile (methane-oxidation chemistry), not a
        # scalar — it needs the reorientation too.
        vmr_fields["ch4"] = _to_3d_with_filled_halo(
            _oriented(jnp.broadcast_to(ch4_vmr, (nlev,))), nlev, halo,
        )
    if n2o_vmr is not None:
        # Prescribed from ``forcing.n2o_vmr``; overrides RRTMGP's
        # ``vmr_global_means.json`` fallback for N2O.
        vmr_fields["n2o"] = _to_3d_with_filled_halo(
            _oriented(jnp.broadcast_to(n2o_vmr, (nlev,))), nlev, halo,
        )

    # Per-SW-band aerosol optics (Stevens 2017 wavelength scaling, jax-
    # rrtmgp PR #4). Each per-band field arrives as (n_bnd_sw, nlev)
    # from the column-vmap; reshape to (n_bnd_sw, 1, 1, nlev+2*halo)
    # for ``compute_heating_rate``. LW is omitted — MACv2-SP only models
    # SW aerosol effects per ``mo_bc_aeropt_splumes.f90``.
    def _to_4d_per_band(arr_2d: jnp.ndarray) -> jnp.ndarray:
        """TOA-first (n_bnd, nlev) → (n_bnd, 1, 1, nlev+2*halo), library frame.

        Reorients to the library's surface-first z axis first, then
        edge-fills the halos (halo 0 = below-surface copy of the surface
        layer, halo -1 = above-top copy of the top layer).
        """
        arr_2d = _oriented_per_band(arr_2d)
        n_bnd = arr_2d.shape[0]
        out = jnp.zeros((n_bnd, 1, 1, nlev + 2 * halo), dtype=arr_2d.dtype)
        out = out.at[:, 0, 0, halo:halo + nlev].set(arr_2d)
        out = out.at[:, 0, 0, 0].set(arr_2d[:, 0])
        out = out.at[:, 0, 0, -1].set(arr_2d[:, -1])
        return out

    aerosol_optics_sw: Optional[dict[str, jnp.ndarray]] = None
    if hasattr(aerosol_data, "aod_sw_per_band"):
        aerosol_optics_sw = {
            "optical_depth":   _to_4d_per_band(aerosol_data.aod_sw_per_band),
            "ssa":             _to_4d_per_band(aerosol_data.ssa_sw_per_band),
            "asymmetry_factor": _to_4d_per_band(aerosol_data.asy_sw_per_band),
        }

    # Per-LW-band aerosol optics from the JAM online-aerosol optics term (#495).
    # Zero/absent for MACv2-SP (SW-only) → ``aerosol_optics_lw=None``.
    aerosol_optics_lw: Optional[dict[str, jnp.ndarray]] = None
    if (hasattr(aerosol_data, "aod_lw_per_band")
            and aerosol_data.aod_lw_per_band.shape[0] > 0):
        aerosol_optics_lw = {
            "optical_depth":   _to_4d_per_band(aerosol_data.aod_lw_per_band),
            "ssa":             _to_4d_per_band(aerosol_data.ssa_lw_per_band),
            "asymmetry_factor": _to_4d_per_band(aerosol_data.asy_lw_per_band),
        }

    # Night columns are handled by the zenith clip (µ0 = 0 zeroes the direct
    # beam); the irradiance itself is strictly positive by construction.
    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0))
    irrad_val = direct_irradiance

    # Per-column surface boundary condition (jax-rrtmgp >= 0.2.1 hook —
    # previously the hardcoded AtmosphericStateCfg scalars 0.07/0.98 were
    # used for every column, severing the ice-albedo feedback and leaving
    # the column solve inconsistent with the surface scheme's absorbed
    # SW·(1−albedo_tile)). The library takes one BROADBAND SW albedo, so
    # blend the surface scheme's vis/nir pair with the ~0.46/0.54 split of
    # the TOA solar spectrum about 0.7 µm. A true per-band albedo (and the
    # direct/diffuse distinction ECHAM makes) needs a g-point→band albedo
    # map in the library — deferred to the cloud/surface optics overhaul.
    sfc_alb_broadband = 0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir

    rrtmgp_output = rrtmgp_instance.compute_heating_rate(
        zenith=zenith_angle, irrad=irrad_val,
        sfc_alb=sfc_alb_broadband, sfc_emis=surface_emissivity,
        cloud_path_liq_lw_per_gpt=cpl_lw_4d,
        cloud_path_ice_lw_per_gpt=cpi_lw_4d,
        cloud_path_liq_sw_per_gpt=cpl_sw_4d,
        cloud_path_ice_sw_per_gpt=cpi_sw_4d,
        vmr_fields=vmr_fields or None,
        aerosol_optics_sw=aerosol_optics_sw,
        aerosol_optics_lw=aerosol_optics_lw,
        **rrtmgp_input,
    )

    # Optional clear-sky call for the cloud radiative effect. With the
    # broadcast q_liq / q_ice already zero and no per-gpoint cloud
    # paths supplied, this collapses to a clear-sky calculation.
    # Aerosols are intentionally included on the clear-sky branch — CMIP
    # convention is that "clear-sky" means cloud-free, aerosols included.
    if compute_cre:
        rrtmgp_output_clear = rrtmgp_instance.compute_heating_rate(
            zenith=zenith_angle, irrad=irrad_val,
            sfc_alb=sfc_alb_broadband, sfc_emis=surface_emissivity,
            vmr_fields=vmr_fields or None,
            aerosol_optics_sw=aerosol_optics_sw,
            aerosol_optics_lw=aerosol_optics_lw,
            **rrtmgp_input,
        )
        toa_sw_up_clear = (
            rrtmgp_output_clear["toa_sw_flux_outgoing_2d_xy"][0, 0]
        )
        toa_lw_up_clear = (
            rrtmgp_output_clear["toa_lw_flux_outgoing_2d_xy"][0, 0]
        )
    else:
        toa_sw_up_clear = jnp.zeros_like(
            rrtmgp_output["toa_sw_flux_outgoing_2d_xy"][0, 0],
        )
        toa_lw_up_clear = jnp.zeros_like(
            rrtmgp_output["toa_lw_flux_outgoing_2d_xy"][0, 0],
        )

    tendencies, diagnostics = prepare_icon_data(
        rrtmgp_output, icon_state,
        surface_albedo_vis, surface_albedo_nir, surface_emissivity,
    )
    diagnostics = diagnostics.copy(
        toa_sw_up_clear=toa_sw_up_clear,
        toa_lw_up_clear=toa_lw_up_clear,
    )
    return tendencies, diagnostics


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

from flax import nnx  # noqa: E402

from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics.radiation import (  # noqa: E402
    cached_radiation_tendency,
    radiation_should_compute,
)
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


def _column_vector_rrtmgp(value: jnp.ndarray, ncols: int) -> jnp.ndarray:
    """Return a vmapped scalar diagnostic as one value per column."""
    return jnp.reshape(value, (ncols,))




def _maybe_chunked_vmap(fn, in_axes):
    """``jax.vmap`` over columns, optionally in rematerialized column blocks.

    The forward's per-column working set is small, so a single vmap over every
    column is both fastest and well within memory. The *backward* is a different
    matter: the reverse pass holds the per-g-point profiles of every column live
    at once (arrays of ``(ncols, n_gpt, nlev+2)``), so its peak grows linearly
    with the column count and reaches ~309 GiB at T63L47 -- far past a single 80
    GiB device -- while T21L47 needs only ~19 GiB.

    Setting ``JCM_RRTMGP_COL_CHUNKS=n`` splits the columns into ``n`` blocks and
    maps a ``jax.checkpoint``-ed vmap over them, so the backward materializes
    one block's intermediates at a time and recomputes the rest. Peak falls
    roughly like ``1/n`` at the cost of one extra forward evaluation of
    radiation. Unset (the default) reproduces the original single vmap exactly.
    """
    import os

    n_chunks = int(os.environ.get("JCM_RRTMGP_COL_CHUNKS", "0"))
    vmapped = jax.vmap(fn, in_axes=in_axes, out_axes=(0, 0))
    if n_chunks <= 1:
        return vmapped

    def run(*args):
        # Mapped arguments may be pytrees (the aerosol struct), so split with
        # tree_map rather than assuming an array.
        first = next(a for a, ax in zip(args, in_axes) if ax == 0)
        ncols = jax.tree_util.tree_leaves(first)[0].shape[0]
        if ncols % n_chunks:
            raise ValueError(f"{ncols} columns is not divisible by {n_chunks} chunks")
        size = ncols // n_chunks

        split = lambda x: x.reshape(n_chunks, size, *x.shape[1:])  # noqa: E731
        mapped = [
            jax.tree_util.tree_map(split, a) for a, ax in zip(args, in_axes) if ax == 0
        ]
        static = [a for a, ax in zip(args, in_axes) if ax is None]

        def body(block):
            block = list(block)
            rebuilt, statics = [], list(static)
            for ax in in_axes:
                rebuilt.append(block.pop(0) if ax == 0 else statics.pop(0))
            return vmapped(*rebuilt)

        tend, diag = lax.map(jax.checkpoint(body), tuple(mapped))
        merge = lambda x: x.reshape(ncols, *x.shape[2:])  # noqa: E731
        return jax.tree_util.tree_map(merge, (tend, diag))

    return run


class RRTMGPRadiation(PhysicsTerm):
    """RRTMGP full-spectrum radiation as a composable PhysicsTerm.

    A single ``jax.vmap`` over all columns (like the rest of the physics):
    the jax-rrtmgp shared-table fix keeps the per-column gas-optics working
    set tiny (~1.5 GB at T63L47), so the *forward* needs no chunking.

    The **reverse pass** is different: it holds per-g-point profiles of every
    column live at once, so its peak memory grows linearly with the column
    count (measured with ``jax.grad`` over a 2-step rollout: ~19 GiB at
    T21L47, ~77 GiB at T31L47, ~171 GiB at T63L47), and XLA's own
    rematerialization pass cannot reduce it. For gradient-based calibration at
    T63L47 set the environment variable ``JCM_RRTMGP_COL_CHUNKS`` (for example
    to ``8``, bringing the peak under 25 GiB); see ``_maybe_chunked_vmap``.
    Unset, the code path is exactly this single vmap and forward-only runs are
    unaffected.

    Reads pressure / height / density from the moist-air diagnostics
    dict, cloud fraction from ``diagnostics["clouds"]`` and
    pre-cloud-step cloud water / ice from state tracers, ozone / CO2 from
    ``"chemistry"``, aerosol from ``"aerosol"``, surface temperature
    from the legacy ``"surface"`` key, and surface albedos /
    emissivity from the public ``"radiation"`` key. Caches its own
    heating rates across radiation sub-steps via the previous step's
    ``RadiationData`` in ``diagnostics["radiation"]``.
    """

    name: ClassVar[str] = "rrtmgp_radiation"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "layer_thickness",
        "air_density", "chemistry", "aerosol",
        "radiation", "surface", "clouds",
    )
    provides: ClassVar[tuple[str, ...]] = ("radiation", "clouds")

    def __init__(
        self,
        params: RadiationParameters | None = None,
        base_seed: int = 0,
        compute_cre: bool = True,
    ):
        """Hold the scheme-native :class:`RadiationParameters`.

        Args:
            params: scheme-native ``RadiationParameters``.
            base_seed: McICA PRNG base seed. The generator folds this
                with ``model_step`` and the per-column index, so the
                same ``base_seed`` always produces the same stochastic
                sub-columns for a given run — bit-exact reruns.
            compute_cre: if True (default), do an extra clear-sky
                RRTMGP call per radiation step and populate
                ``toa_{sw,lw}_up_clear`` for the cloud radiative
                effect diagnostic. Set False for the 1× McICA-only
                cost when CRE isn't needed.

        """
        self.params = nnx.Param(params or RadiationParameters.default())
        # Plain Python attributes — these are static-at-trace-time so
        # the McICA seeding and the CRE branch fold into ``__call__``
        # without an extra pytree leaf.
        self._base_seed = int(base_seed)
        self._compute_cre = bool(compute_cre)
        self._coords_cached = False
        # Eagerly create the global RRTMGP instance now (loads netCDF
        # gas-optics + cloud-optics tables). Otherwise the first jit
        # trace of ``__call__`` triggers ``optics_factory`` from inside
        # the traced scope, and the lookup-table jnp arrays created by
        # ``load_data`` leak as ``UnexpectedTracerError`` — fatal for
        # ``output_averages=True`` runs in particular. Doing it here
        # forces a single non-traced load at term-construction time.
        _ensure_rrtmgp()

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (deg) for the radiation scheme."""
        lat, lon = column_lat_lon(coords.horizontal)
        self._lats = nnx.Variable(lat * 180.0 / jnp.pi)
        self._lons = nnx.Variable(lon * 180.0 / jnp.pi)
        self._coords_cached = True

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute or reuse cached RRTMGP heating rates."""
        params = self.params.get_value()
        radiation = diagnostics["radiation"]

        def _compute():
            tend, rad = self._compute_full(state, diagnostics, forcing, params)
            # Pin the compute branch to the carry's leaf dtypes: under
            # jax_enable_x64 (e.g. driving this scheme from a float64
            # dycore with float32 physics state) some strong table
            # constants promote a subset of the freshly-computed leaves
            # to float64, and the two lax.cond branches would fail to
            # type-check against the uniform-dtype cached carry.
            rad = jax.tree.map(lambda n, o: n.astype(o.dtype), rad, radiation)
            tend = jax.tree.map(
                lambda t: t.astype(state.temperature.dtype), tend)
            return tend, rad

        def _use_cached():
            tend = cached_radiation_tendency(
                radiation, state.temperature.shape,
            )
            # Same dtype pin as _compute: under x64 the cached heating ->
            # tendency arithmetic can promote through float64 scalars.
            tend = jax.tree.map(
                lambda t: t.astype(state.temperature.dtype), tend)
            return tend, radiation

        tendency, new_radiation = jax.lax.cond(
            radiation_should_compute(diagnostics, params),
            _compute, _use_cached,
        )
        # Advance the radiation-local step counter on every call (both
        # compute and cached paths). The McICA seed in ``_compute_full``
        # folds this counter in for per-(step, column) reproducibility,
        # so the increment must happen even on cached steps to keep the
        # sub-column sequence aligned with the chosen sub-stepping cadence.
        new_radiation = new_radiation.copy(step=radiation.step + 1)
        # Mirror the all-sky and clear-sky TOA fluxes onto the
        # ``"clouds"`` sub-struct for cloud-radiative-effect diagnostics.
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
        """Run the full RRTMGP scheme, return (tendency, RadiationData)."""
        nlev, ncols = state.temperature.shape

        latitudes = self._lats.get_value()
        longitudes = self._lons.get_value()
        solar = forcing.solar
        # Scalar radiation step counter + per-column index drive the
        # McICA seed via ``mcica.column_key`` inside
        # ``radiation_scheme_rrtmgp``. The counter lives on the
        # radiation carry slot, advanced in ``__call__`` after the
        # compute-vs-cache cond.
        model_step = diagnostics["radiation"].step
        column_indices = jnp.arange(ncols, dtype=jnp.int32)

        cloud_water, cloud_ice, cloud_fraction = radiation_cloud_fields(
            state, diagnostics,
        )
        # Microphysical effective radii from the clouds carry (ECHAM
        # preffl/preffi, written by the 2M microphysics; zero = not
        # provided, e.g. 1M or cold start). Lagged one step by the carry,
        # like the rest of the radiation's cloud inputs.
        clouds_in = diagnostics["clouds"]
        r_eff_liq_um = clouds_in.r_eff_liq.reshape(nlev, ncols)
        r_eff_ice_um = clouds_in.r_eff_ice.reshape(nlev, ncols)

        # Greenhouse-gas sourcing (all converted ppmv -> mole fraction here):
        #   O3, CH4  <- the chemistry diagnostic (O3 analytic/climatology; CH4
        #               forcing-seeded then evolved by the methane-loss scheme)
        #   CO2, N2O <- prescribed forcings read straight from ForcingData
        # Radiation never reads a GHG concentration from its own parameters, and
        # never lets a value be silently redeclared.
        chemistry = diagnostics["chemistry"]
        ozone_vmr = chemistry.ozone_vmr * 1e-6     # (nlev, ncols)
        ch4_vmr = chemistry.methane_vmr * 1e-6     # (nlev, ncols)
        co2_vmr = forcing.co2_vmr * 1e-6           # scalar
        n2o_vmr = forcing.n2o_vmr * 1e-6           # scalar

        surface_temperature_col = (
            diagnostics["surface"].surface_temperature.reshape(ncols)
        )
        radiation_in = diagnostics["radiation"]
        surface_albedo_vis_col = radiation_in.surface_albedo_vis.reshape(ncols)
        surface_albedo_nir_col = radiation_in.surface_albedo_nir.reshape(ncols)
        surface_emissivity_col = radiation_in.surface_emissivity.reshape(ncols)

        aerosol_in = diagnostics["aerosol"]
        # Per-SW-band fields are ``(n_bnd_sw, nlev, ncols)`` from MACv2-SP;
        # transpose to ``(ncols, n_bnd_sw, nlev)`` so the column axis is
        # leading (vmap-friendly).
        n_bnd_sw = aerosol_in.aod_sw_per_band.shape[0]
        n_bnd_lw = aerosol_in.aod_lw_per_band.shape[0]

        def _per_band_to_col(arr, n_bnd):
            """(n_bnd, nlev, ncols) → (ncols, n_bnd, nlev) for the column vmap."""
            return arr.reshape(n_bnd, nlev, ncols).transpose(2, 0, 1)

        aerosol_for_vmap = aerosol_in.copy(
            aod_profile=aerosol_in.aod_profile.reshape(nlev, ncols).T,
            ssa_profile=aerosol_in.ssa_profile.reshape(nlev, ncols).T,
            asy_profile=aerosol_in.asy_profile.reshape(nlev, ncols).T,
            cdnc_factor=aerosol_in.cdnc_factor.reshape(ncols),
            aod_total=aerosol_in.aod_total.reshape(ncols),
            aod_anthropogenic=aerosol_in.aod_anthropogenic.reshape(ncols),
            aod_background=aerosol_in.aod_background.reshape(ncols),
            angstrom=aerosol_in.angstrom.reshape(ncols),
            aod_sw_per_band=_per_band_to_col(aerosol_in.aod_sw_per_band, n_bnd_sw),
            ssa_sw_per_band=_per_band_to_col(aerosol_in.ssa_sw_per_band, n_bnd_sw),
            asy_sw_per_band=_per_band_to_col(aerosol_in.asy_sw_per_band, n_bnd_sw),
            aod_lw_per_band=_per_band_to_col(aerosol_in.aod_lw_per_band, n_bnd_lw),
            ssa_lw_per_band=_per_band_to_col(aerosol_in.ssa_lw_per_band, n_bnd_lw),
            asy_lw_per_band=_per_band_to_col(aerosol_in.asy_lw_per_band, n_bnd_lw),
        )

        base_seed = self._base_seed
        compute_cre = self._compute_cre

        # Column-leading inputs for the vmap. The jax-rrtmgp #8 fix keeps the
        # gas-optics tables shared operands under the column vmap, so the
        # per-column working set is tiny (~1.5 GB at T63L47) and no chunking is
        # needed — radiation now vmaps over columns like the rest of the
        # physics. ``aerosol_for_vmap`` is already column-leading; add Nccn.
        def lev_to_col(a):
            """(nlev, ncols) → (ncols, nlev) for the leading column vmap."""
            return a.T

        aerosol_col = aerosol_for_vmap.copy(Nccn=aerosol_in.Nccn.reshape(ncols))
        cols = dict(
            temperature=lev_to_col(state.temperature),
            specific_humidity=lev_to_col(state.specific_humidity),
            pressure_full=lev_to_col(diagnostics["pressure_full"]),
            pressure_half=lev_to_col(diagnostics["pressure_half"]),
            layer_thickness=lev_to_col(diagnostics["layer_thickness"]),
            air_density=lev_to_col(diagnostics["air_density"]),
            cloud_water=lev_to_col(cloud_water),
            cloud_ice=lev_to_col(cloud_ice),
            cloud_fraction=lev_to_col(cloud_fraction),
            surface_temperature=surface_temperature_col,
            surface_albedo_vis=surface_albedo_vis_col,
            surface_albedo_nir=surface_albedo_nir_col,
            surface_emissivity=surface_emissivity_col,
            latitudes=latitudes,
            longitudes=longitudes,
            aerosol=aerosol_col,
            column_indices=column_indices,
            ozone_vmr=lev_to_col(ozone_vmr),
            co2_vmr=lev_to_col(jnp.broadcast_to(co2_vmr, (nlev, ncols))),
            ch4_vmr=lev_to_col(jnp.broadcast_to(ch4_vmr, (nlev, ncols))),
            n2o_vmr=lev_to_col(jnp.broadcast_to(n2o_vmr, (nlev, ncols))),
            r_eff_liq_um=lev_to_col(r_eff_liq_um),
            r_eff_ice_um=lev_to_col(r_eff_ice_um),
        )

        # We tried a day/night split here (solve the dark ~half LW-only, skip
        # its shortwave) — bit-identical but ~25 % slower at T63L47, because
        # after the gas-optics gather fix the SW solve is no longer the
        # bottleneck and the gather/scatter outweighs the skipped work. A plain
        # single vmap over all columns is fastest.
        _in_axes = (
            0, 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            None, 0, 0,
            None, 0,
            0, None, None, None,  # col_index, model_step, base_seed, cre
            0, 0, 0, 0,          # ozone_vmr, co2_vmr, ch4_vmr, n2o_vmr
            0, 0,                # r_eff_liq_um, r_eff_ice_um
        )
        tendencies_vmapped, diagnostics_vmapped = _maybe_chunked_vmap(
            radiation_scheme_rrtmgp, _in_axes,
        )(
            cols["temperature"], cols["specific_humidity"],
            cols["pressure_full"], cols["pressure_half"],
            cols["layer_thickness"], cols["air_density"],
            cols["cloud_water"], cols["cloud_ice"], cols["cloud_fraction"],
            cols["surface_temperature"], cols["surface_albedo_vis"],
            cols["surface_albedo_nir"], cols["surface_emissivity"],
            solar, cols["latitudes"], cols["longitudes"],
            params, cols["aerosol"], cols["column_indices"],
            model_step, base_seed, compute_cre,
            cols["ozone_vmr"], cols["co2_vmr"], cols["ch4_vmr"], cols["n2o_vmr"],
            cols["r_eff_liq_um"], cols["r_eff_ice_um"],
        )

        # Per-gpoint flux profiles are summed over g-points inside the
        # vmapped per-column compute, so flux arrays are (ncols, nlev+1)
        # — only a transpose is needed (DO NOT use the grey path's
        # transpose+sum, the per-band axis is already gone).
        rad_out = RadiationData(
            cos_zenith=_column_vector_rrtmgp(diagnostics_vmapped.cos_zenith, ncols),
            surface_albedo_vis=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_albedo_vis, ncols,
            ),
            surface_albedo_nir=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_albedo_nir, ncols,
            ),
            surface_emissivity=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_emissivity, ncols,
            ),
            sw_flux_up=diagnostics_vmapped.sw_flux_up.T,
            sw_flux_down=diagnostics_vmapped.sw_flux_down.T,
            sw_heating_rate=tendencies_vmapped.shortwave_heating.T,
            lw_flux_up=diagnostics_vmapped.lw_flux_up.T,
            lw_flux_down=diagnostics_vmapped.lw_flux_down.T,
            lw_heating_rate=tendencies_vmapped.longwave_heating.T,
            surface_sw_down=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_sw_down, ncols,
            ),
            surface_lw_down=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_lw_down, ncols,
            ),
            surface_sw_up=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_sw_up, ncols,
            ),
            surface_lw_up=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_lw_up, ncols,
            ),
            toa_sw_up=_column_vector_rrtmgp(diagnostics_vmapped.toa_sw_up, ncols),
            toa_lw_up=_column_vector_rrtmgp(diagnostics_vmapped.toa_lw_up, ncols),
            toa_sw_down=_column_vector_rrtmgp(
                diagnostics_vmapped.toa_sw_down, ncols,
            ),
            toa_sw_up_clear=_column_vector_rrtmgp(
                diagnostics_vmapped.toa_sw_up_clear, ncols,
            ),
            toa_lw_up_clear=_column_vector_rrtmgp(
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
