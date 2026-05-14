"""RRTMGP-based radiation scheme for ECHAM physics.

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

import jax
import jax.numpy as jnp
from jax import lax

from jax_solar import OrbitalTime, radiation_flux, get_solar_sin_altitude
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
from jcm.constants import PhysicalConstants

import rrtmgp
from rrtmgp.config import radiative_transfer
from rrtmgp import stretched_grid_util
from rrtmgp.rrtmgp import RRTMGP

# ---------------------------------------------------------------------------
# Chunked-vmap configuration
# ---------------------------------------------------------------------------
#
# RRTMGP is ``vmap``'d over horizontal columns. Each column allocates
# ~150 intermediate arrays of shape (ngpt, nlev) inside ``compute_heating_rate``
# (gas-optics interpolation tables, planck source functions, optical depth,
# tridiagonal flux solver working memory). Vmapping all columns at once
# blows up GPU memory at high horizontal resolution, so we split the vmap
# into sequential chunks via ``lax.map``: ``n_chunks`` smaller batches that
# share the same JIT'd kernel.
#
# Empirical sweet spot on a single 80 GiB A100 (T63L47, ngpt=128, nlev=47):
#
#   chunk=18432 (1 chunk, no chunking): OOM at 67 GiB peak
#   chunk= 9216 (2 chunks)            : ~8.7 s/step
#   chunk= 4608 (4 chunks)            : ~15.2 s/step
#
# Two chunks is ~74 % faster than four because XLA has enough headroom
# to skip rematerialization. The per-cell cost (~3.6 MB) scales linearly
# with ``nlev``.
#
# ``chunk_budget(nlev)`` auto-detects the largest chunk that fits in the
# device's HBM (via ``jax.devices()[0].memory_stats()['bytes_limit']``) at
# 55 % of the budget — leaves room for XLA working memory. Override with
# :func:`set_chunk_size` (e.g. on shared GPUs with reduced free memory or
# to fix a chunk count for reproducible kernel launches).

_CHUNK_SIZE_OVERRIDE = None  # int | None


def set_chunk_size(chunk_size) -> None:
    """Override the RRTMGP chunked-vmap chunk size (cells per chunk).

    Set to a positive integer to fix the chunk count; pass ``None`` to
    revert to auto-detection from device HBM. Must be called BEFORE the
    first radiation call so the JIT'd radiation function picks up the
    new value (changing it after a JIT compile triggers a recompile on
    the next call).
    """
    global _CHUNK_SIZE_OVERRIDE
    _CHUNK_SIZE_OVERRIDE = chunk_size


def get_chunk_size_override():
    """Return the current chunk-size override (``None`` for auto)."""
    return _CHUNK_SIZE_OVERRIDE


def chunk_budget(nlev: int) -> int:
    """Return the RRTMGP chunk-size budget (cells/chunk) for this device.

    Uses :data:`_CHUNK_SIZE_OVERRIDE` if set, else queries the JAX
    device HBM and picks the largest chunk that fits at 55 % of the
    XLA bytes_limit. Falls back to 9216 if the device doesn't report
    HBM (e.g. CPU run).
    """
    if _CHUNK_SIZE_OVERRIDE is not None and _CHUNK_SIZE_OVERRIDE > 0:
        return int(_CHUNK_SIZE_OVERRIDE)
    bytes_per_cell = 3.6e6 * (nlev / 47.0)
    try:
        bytes_limit = jax.devices()[0].memory_stats().get('bytes_limit', 0)
    except Exception:
        bytes_limit = 0
    if bytes_limit > 0:
        return max(1, int(0.55 * bytes_limit / bytes_per_cell))
    return 9216


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
    toa_flux = radiation_flux(orbital_time, longitude, latitude, parameters.solar_constant)
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = sin_altitude  # cos(zenith) = sin(altitude)

    # In-cloud condensate (grid-mean / cf) is what each cloudy
    # sub-column sees; the binary McICA mask then re-imposes a
    # cloud-or-clear partitioning per g-point.
    cloud_water_in_cloud = in_cloud_path(cloud_water, cloud_fraction)
    cloud_ice_in_cloud = in_cloud_path(cloud_ice, cloud_fraction)

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

    col_key = column_key(
        jax.random.PRNGKey(base_seed),
        model_step=model_step, column_index=column_index,
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
    )
    # The broadcast q_liq / q_ice are shadowed by the per-gpoint
    # arrays inside the g-point loop, so set them to zero. The clear-
    # sky branch (no per-gpoint args, just q_liq=0/q_ice=0) gives the
    # all-clear fluxes used for CRE.
    zero_3d = jnp.zeros_like(rrtmgp_input["q_liq"])
    rrtmgp_input["q_liq"] = zero_3d
    rrtmgp_input["q_ice"] = zero_3d
    rrtmgp_input["q_c"] = zero_3d

    # Per-cell gas concentrations (#483 + jax-rrtmgp PR #4). The library
    # merges this dict over the sounding-based defaults before calling
    # gas optics; ``h2o`` is always overridden internally from ``q_t``.
    # Each profile is shaped (1, 1, nz+2*halo) to match the rest of
    # rrtmgp_input.
    halo = 1
    nlev = icon_state.temperature.shape[0]
    vmr_fields: dict[str, jnp.ndarray] = {}
    if ozone_vmr is not None:
        # ``ozone_vmr`` arrives as a (nlev,) profile in mole fraction.
        vmr_fields["o3"] = _to_3d_with_filled_halo(ozone_vmr, nlev, halo)
    if co2_vmr is not None:
        vmr_fields["co2"] = _to_3d_with_filled_halo(
            jnp.broadcast_to(co2_vmr, (nlev,)), nlev, halo,
        )
    if ch4_vmr is not None:
        vmr_fields["ch4"] = _to_3d_with_filled_halo(
            jnp.broadcast_to(ch4_vmr, (nlev,)), nlev, halo,
        )

    # Per-SW-band aerosol optics (Stevens 2017 wavelength scaling, jax-
    # rrtmgp PR #4). Each per-band field arrives as (n_bnd_sw, nlev)
    # from the column-vmap; reshape to (n_bnd_sw, 1, 1, nlev+2*halo)
    # for ``compute_heating_rate``. LW is omitted — MACv2-SP only models
    # SW aerosol effects per ``mo_bc_aeropt_splumes.f90``.
    def _to_4d_per_band(arr_2d: jnp.ndarray) -> jnp.ndarray:
        """(n_bnd, nlev) → (n_bnd, 1, 1, nlev+2*halo) with edge halos."""
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

    zenith_angle = jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0))
    irrad_val = jnp.maximum(toa_flux, 0.0)

    rrtmgp_output = rrtmgp_instance.compute_heating_rate(
        zenith=zenith_angle, irrad=irrad_val,
        cloud_path_liq_lw_per_gpt=cpl_lw_4d,
        cloud_path_ice_lw_per_gpt=cpi_lw_4d,
        cloud_path_liq_sw_per_gpt=cpl_sw_4d,
        cloud_path_ice_sw_per_gpt=cpi_sw_4d,
        vmr_fields=vmr_fields or None,
        aerosol_optics_sw=aerosol_optics_sw,
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
            vmr_fields=vmr_fields or None,
            aerosol_optics_sw=aerosol_optics_sw,
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


class RRTMGPRadiation(PhysicsTerm):
    """RRTMGP full-spectrum radiation as a composable PhysicsTerm.

    Uses ``jax.lax.map`` over chunks for memory-bounded vmap (RRTMGP
    allocates many g-point intermediates per column; running all columns
    of a T63L47 grid at once OOMs an 80 GiB A100). Chunk size is
    auto-detected from device HBM via :func:`chunk_budget`; override via
    ``RadiationParameters(rrtmgp_chunk_size=...)``.

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
        """Compute or reuse cached RRTMGP heating rates."""
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

        chemistry = diagnostics["chemistry"]
        ozone_vmr = chemistry.ozone_vmr * 1e-6     # (nlev, ncols)
        co2_vmr = chemistry.co2_vmr * 1e-6         # (nlev, ncols)
        ch4_vmr = chemistry.methane_vmr * 1e-6     # (nlev, ncols)

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
        aerosol_for_vmap = aerosol_in.copy(
            aod_profile=aerosol_in.aod_profile.reshape(nlev, ncols).T,
            ssa_profile=aerosol_in.ssa_profile.reshape(nlev, ncols).T,
            asy_profile=aerosol_in.asy_profile.reshape(nlev, ncols).T,
            cdnc_factor=aerosol_in.cdnc_factor.reshape(ncols),
            aod_total=aerosol_in.aod_total.reshape(ncols),
            aod_anthropogenic=aerosol_in.aod_anthropogenic.reshape(ncols),
            aod_background=aerosol_in.aod_background.reshape(ncols),
            angstrom=aerosol_in.angstrom.reshape(ncols),
            aod_sw_per_band=aerosol_in.aod_sw_per_band.reshape(
                n_bnd_sw, nlev, ncols).transpose(2, 0, 1),
            ssa_sw_per_band=aerosol_in.ssa_sw_per_band.reshape(
                n_bnd_sw, nlev, ncols).transpose(2, 0, 1),
            asy_sw_per_band=aerosol_in.asy_sw_per_band.reshape(
                n_bnd_sw, nlev, ncols).transpose(2, 0, 1),
        )

        # Auto-pick chunk size from device HBM (see chunk_budget()).
        budget = chunk_budget(nlev)
        if ncols <= budget:
            chunk_size = ncols
        else:
            n_chunks = -(-ncols // budget)  # ceil-div
            while ncols % n_chunks != 0:
                n_chunks += 1
            chunk_size = ncols // n_chunks
        n_chunks = ncols // chunk_size

        def split_lev_first(a):
            """Reshape (nz, ncols) → (n_chunks, chunk_size, nz)."""
            nz = a.shape[0]
            return a.reshape(nz, n_chunks, chunk_size).transpose(1, 2, 0)

        def split_col(a):
            """Reshape (ncols, ...) → (n_chunks, chunk_size, ...)."""
            return a.reshape(n_chunks, chunk_size, *a.shape[1:])

        chunked_inputs = dict(
            temperature=split_lev_first(state.temperature),
            specific_humidity=split_lev_first(state.specific_humidity),
            pressure_full=split_lev_first(diagnostics["pressure_full"]),
            pressure_half=split_lev_first(diagnostics["pressure_half"]),
            layer_thickness=split_lev_first(diagnostics["layer_thickness"]),
            air_density=split_lev_first(diagnostics["air_density"]),
            cloud_water=split_lev_first(cloud_water),
            cloud_ice=split_lev_first(cloud_ice),
            cloud_fraction=split_lev_first(cloud_fraction),
            surface_temperature=split_col(surface_temperature_col),
            surface_albedo_vis=split_col(surface_albedo_vis_col),
            surface_albedo_nir=split_col(surface_albedo_nir_col),
            surface_emissivity=split_col(surface_emissivity_col),
            latitudes=split_col(latitudes),
            longitudes=split_col(longitudes),
            column_indices=split_col(column_indices),
            ozone_vmr=split_lev_first(ozone_vmr),
            co2_vmr=split_lev_first(jnp.broadcast_to(co2_vmr, (nlev, ncols))),
            ch4_vmr=split_lev_first(jnp.broadcast_to(ch4_vmr, (nlev, ncols))),
            aerosol=aerosol_for_vmap.copy(
                aod_profile=split_col(aerosol_for_vmap.aod_profile),
                ssa_profile=split_col(aerosol_for_vmap.ssa_profile),
                asy_profile=split_col(aerosol_for_vmap.asy_profile),
                cdnc_factor=split_col(aerosol_for_vmap.cdnc_factor),
                aod_total=split_col(aerosol_for_vmap.aod_total),
                aod_anthropogenic=split_col(aerosol_for_vmap.aod_anthropogenic),
                aod_background=split_col(aerosol_for_vmap.aod_background),
                Nccn=split_col(aerosol_in.Nccn.reshape(ncols)),
                angstrom=split_col(aerosol_for_vmap.angstrom),
                aod_sw_per_band=split_col(aerosol_for_vmap.aod_sw_per_band),
                ssa_sw_per_band=split_col(aerosol_for_vmap.ssa_sw_per_band),
                asy_sw_per_band=split_col(aerosol_for_vmap.asy_sw_per_band),
            ),
        )

        base_seed = self._base_seed
        compute_cre = self._compute_cre

        def _vmap_one_chunk(chunk_inputs):
            return jax.vmap(
                radiation_scheme_rrtmgp,
                in_axes=(
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    None, 0, 0,
                    None, 0,
                    0, None, None, None,    # column_index, model_step, base_seed, compute_cre
                    0, 0, 0,                # ozone_vmr, co2_vmr, ch4_vmr
                ),
                out_axes=(0, 0),
                axis_size=chunk_size,
            )(
                chunk_inputs['temperature'], chunk_inputs['specific_humidity'],
                chunk_inputs['pressure_full'], chunk_inputs['pressure_half'],
                chunk_inputs['layer_thickness'], chunk_inputs['air_density'],
                chunk_inputs['cloud_water'], chunk_inputs['cloud_ice'],
                chunk_inputs['cloud_fraction'],
                chunk_inputs['surface_temperature'],
                chunk_inputs['surface_albedo_vis'],
                chunk_inputs['surface_albedo_nir'],
                chunk_inputs['surface_emissivity'],
                solar, chunk_inputs['latitudes'], chunk_inputs['longitudes'],
                params, chunk_inputs['aerosol'],
                chunk_inputs['column_indices'],
                model_step, base_seed, compute_cre,
                chunk_inputs['ozone_vmr'],
                chunk_inputs['co2_vmr'],
                chunk_inputs['ch4_vmr'],
            )

        chunked_results = jax.lax.map(_vmap_one_chunk, chunked_inputs)
        tendencies_chunked, diagnostics_chunked = chunked_results

        def merge(a):
            return a.reshape(ncols, *a.shape[2:])

        tendencies_vmapped = jax.tree_util.tree_map(merge, tendencies_chunked)
        diagnostics_vmapped = jax.tree_util.tree_map(merge, diagnostics_chunked)

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
