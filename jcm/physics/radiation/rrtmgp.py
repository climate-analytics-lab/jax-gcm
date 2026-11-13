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

"""

from pathlib import Path
from typing import Tuple, Optional
import contextvars
import dataclasses
import functools
import math
import warnings

import logging

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.coords_util import column_lat_lon
from jax import lax

from jax_solar import OrbitalTime, direct_solar_irradiance, get_solar_sin_altitude
from jcm.physics.chemistry.simple_chemistry import ppmv_to_mole_fraction
from jcm.physics.clouds.cloud_data import radiation_cloud_fields
from jcm.physics.radiation.radiation_types import (
    CLEAR_SKY_KEYS,
    RADIATION_OUTPUT_ATTRS,
    RadiationParameters,
    RadiationTendencies,
    RadiationData,
)
from jcm.physics.radiation.grey_two_stream.radiation_scheme import prepare_radiation_state
from jcm.physics.radiation.mcica import (
    column_key,
    effective_cloud_fraction,
    effective_cloud_fraction_with_mask_gradient,
    generate_subcolumns,
    generate_subcolumns_batch,
    in_cloud_condensate,
    in_cloud_condensate_with_mask_gradient,
    optically_thick_cover,
    resolvable_path,
)
from jcm.physics.radiation.radiation_types import (
    cloud_overlap_name,
    lagged_convection_type,
    liquid_inhomogeneity,
)
from jcm.physics.radiation.cloud_optics import (
    host_effective_radii,
    radiation_effective_radii,
)
import jcm.constants as c

import rrtmgp
from rrtmgp.config import radiative_transfer
from rrtmgp import stretched_grid_util
from rrtmgp.optics import gas_optics, optics_utils
from rrtmgp.rrtmgp import RRTMGP
from rrtmgp.rte import monochromatic_two_stream, rte_utils, two_stream


# ---------------------------------------------------------------------------
# Module-level RRTMGP instance (created once at import time)
# ---------------------------------------------------------------------------
_GLOBAL_RRTMGP_INSTANCE = None
_RRTMGP_2D_ZENITH_COMPAT_INSTALLED = False
_RRTMGP_TPU_OPTICS_COMPAT_INSTALLED = False
_ACTIVE_SAFE_ZENITH_2D: contextvars.ContextVar[Optional[jnp.ndarray]] = (
    contextvars.ContextVar("_ACTIVE_SAFE_ZENITH_2D", default=None)
)


logger = logging.getLogger(__name__)


def _install_rrtmgp_2d_zenith_compat() -> None:
    """Enable 2-D ``(nx, ny)`` solar zenith fields in ``jax-rrtmgp``'s SW solver.

    ``jax-rrtmgp`` <= 0.7.0 natively solves ``(nx, ny, nz)`` grids across gas
    optics, cloud optics, aerosol mixing, and longwave transport, while three
    shortwave sites assume a scalar ``zenith`` (multiplying ``cos(zenith)``
    with both 3-D ``optical_depth`` and 2-D ``toa_flux``, and masking both 3-D
    and 2-D flux outputs with ``night``). Upstream ``climate-analytics-lab/jax-rrtmgp#45``
    promotes ``zenith`` natively and sets ``two_stream._SUPPORTS_2D_ZENITH = True``;
    for older releases this idempotent dispatcher passes scalar ``zenith`` calls
    straight through untouched and promotes 2-D ``(nx, ny)`` zenith fields along
    the vertical axis only where needed.
    """
    global _RRTMGP_2D_ZENITH_COMPAT_INSTALLED
    if (
        _RRTMGP_2D_ZENITH_COMPAT_INSTALLED
        or getattr(two_stream, "_SUPPORTS_2D_ZENITH", False)
    ):
        _RRTMGP_2D_ZENITH_COMPAT_INSTALLED = True
        return

    orig_solve_sw = two_stream.solve_sw
    orig_sw_cell_properties = monochromatic_two_stream.sw_cell_properties
    orig_sw_cell_source = monochromatic_two_stream.sw_cell_source
    shift_down = monochromatic_two_stream._shift_down

    def _sw_cell_properties_compat(
        zenith, optical_depth, ssa, asymmetry_factor,
    ):
        z2d = _ACTIVE_SAFE_ZENITH_2D.get()
        if z2d is not None:
            return orig_sw_cell_properties(
                z2d[:, :, jnp.newaxis], optical_depth, ssa, asymmetry_factor,
            )
        return orig_sw_cell_properties(
            zenith, optical_depth, ssa, asymmetry_factor,
        )

    def _sw_cell_source_compat(
        t_dir, r_dir, optical_depth, toa_flux, sfc_albedo_direct, zenith,
        use_scan=False,
    ):
        z2d = _ACTIVE_SAFE_ZENITH_2D.get()
        if z2d is None:
            return orig_sw_cell_source(
                t_dir=t_dir, r_dir=r_dir, optical_depth=optical_depth,
                toa_flux=toa_flux, sfc_albedo_direct=sfc_albedo_direct,
                zenith=zenith, use_scan=use_scan,
            )
        optical_depth = jnp.maximum(optical_depth, 0.0)
        mu_2d = jnp.cos(z2d)
        t_noscat = jnp.exp(-optical_depth / mu_2d[:, :, jnp.newaxis])
        flux_down_direct_bc = toa_flux * mu_2d

        def op(carry, w):
            prod = w * carry
            return prod, prod

        _, flux_down_direct = rte_utils.recurrent_op_with_halos(
            op, flux_down_direct_bc, {"w": t_noscat},
            forward=False, use_scan=use_scan,
        )
        shifted = shift_down(flux_down_direct)
        halo_width = 1
        flux_down_sfc = flux_down_direct[:, :, halo_width]
        return {
            "src_up": r_dir * shifted,
            "src_down": t_dir * shifted,
            "flux_down_dir": flux_down_direct,
            "sfc_src": sfc_albedo_direct * flux_down_sfc,
        }

    def _solve_sw_compat(
        pressure, temperature, molecules, optics_lib, atmos_state,
        *args, **kwargs,
    ):
        zenith = atmos_state.zenith
        if (isinstance(zenith, jnp.ndarray)
                and zenith.ndim == 2
                and temperature.ndim == 3):
            night = zenith >= 0.5 * jnp.pi
            safe_zenith = jnp.where(night, jnp.zeros_like(zenith), zenith)
            atmos_state_scalar = dataclasses.replace(
                atmos_state, zenith=jnp.zeros((), dtype=zenith.dtype),
            )
            token = _ACTIVE_SAFE_ZENITH_2D.set(safe_zenith)
            try:
                fluxes = orig_solve_sw(
                    pressure, temperature, molecules, optics_lib,
                    atmos_state_scalar, *args, **kwargs,
                )
            finally:
                _ACTIVE_SAFE_ZENITH_2D.reset(token)
            night_3d = night[:, :, jnp.newaxis]
            return {
                key: jnp.where(
                    night_3d if val.ndim == 3 else night,
                    jnp.zeros_like(val),
                    val,
                )
                for key, val in fluxes.items()
            }
        return orig_solve_sw(
            pressure, temperature, molecules, optics_lib, atmos_state,
            *args, **kwargs,
        )

    monochromatic_two_stream.sw_cell_properties = _sw_cell_properties_compat
    monochromatic_two_stream.sw_cell_source = _sw_cell_source_compat
    two_stream.solve_sw = _solve_sw_compat
    _RRTMGP_2D_ZENITH_COMPAT_INSTALLED = True


def _install_rrtmgp_tpu_optics_compat() -> None:
    """Route ``jax-rrtmgp`` table interpolations to IEEE float32 MXU ops on TPU.

    Upstream ``climate-analytics-lab/jax-rrtmgp#45`` implements these TPU MXU
    contractions natively and sets ``gas_optics._SUPPORTS_TPU_MXU_ALL_GPTS = True``.
    For older ``jax-rrtmgp`` releases, every wrapper returns the upstream
    implementation untouched on GPU and CPU (``jax.default_backend() != "tpu"``).
    When tracing for TPU, multi-dimensional ``gather`` instructions on
    ``(nx, ny, nz)`` cells create rank-4/rank-5 buffers with size-1 minor slices
    that pad by 25x-1024x to ``(8, 128)`` physical tiles and serialize on the
    vector unit:

    1. 1-D ``optics_utils.interpolate`` calls (cloud Mie tables and Planck
       ``totplnk`` curves) contract on the MXU with ``lax.Precision.HIGHEST``
       without enabling multi-axis one-hot products on 3-D tables.
    2. ``all_gpts`` species and ``vmr_ref`` lookups with static band indices
       select between the lower- and upper-atmosphere species with ``jnp.where``
       and a 14-node temperature one-hot projection.
    3. ``_interpolate_rows_te`` (``14 * 9 = 126`` entries) contracts a flat
       one-hot weight matrix on the MXU, and ``_interpolate_rows_tpm`` maps a
       factored ``(n_t * n_eta = 126) -> (n_p = 60)`` MXU contraction over
       vertical levels so per-level intermediates fit in VMEM without inflating
       peak HBM.
    """
    global _RRTMGP_TPU_OPTICS_COMPAT_INSTALLED
    if (
        _RRTMGP_TPU_OPTICS_COMPAT_INSTALLED
        or getattr(gas_optics, "_SUPPORTS_TPU_MXU_ALL_GPTS", False)
    ):
        _RRTMGP_TPU_OPTICS_COMPAT_INSTALLED = True
        return

    orig_interpolate = optics_utils.interpolate
    orig_get_vmr = gas_optics.get_vmr
    orig_rel_abund = gas_optics._compute_relative_abundance_interpolant
    orig_rows_te = getattr(gas_optics, "_interpolate_rows_te", None)
    orig_rows_tpm = getattr(gas_optics, "_interpolate_rows_tpm", None)

    def _interpolate_compat(coeffs, interpolant_fns):
        if (jax.default_backend() == "tpu"
                and coeffs.ndim == 1
                and len(interpolant_fns) == 1):
            fn = next(iter(interpolant_fns.values()))
            interp = fn()
            n_nodes = coeffs.shape[0]
            cell_shape = (
                interp.idx.shape
                if isinstance(interp, optics_utils.IndexAndWeight)
                else interp.interp_high.idx.shape
            )
            ncells = math.prod(cell_shape)
            if isinstance(interp, optics_utils.IndexAndWeight):
                oh = interp.weight.reshape(1, ncells) * jax.nn.one_hot(
                    interp.idx.reshape(ncells),
                    n_nodes,
                    dtype=coeffs.dtype,
                    axis=0,
                )
            else:
                oh_lo = (
                    interp.interp_low.weight.reshape(1, ncells)
                    * jax.nn.one_hot(
                        interp.interp_low.idx.reshape(ncells),
                        n_nodes,
                        dtype=coeffs.dtype,
                        axis=0,
                    )
                )
                oh_hi = (
                    interp.interp_high.weight.reshape(1, ncells)
                    * jax.nn.one_hot(
                        interp.interp_high.idx.reshape(ncells),
                        n_nodes,
                        dtype=coeffs.dtype,
                        axis=0,
                    )
                )
                oh = oh_lo + oh_hi
            return jnp.einsum(
                "a,ac->c", coeffs, oh, precision=lax.Precision.HIGHEST,
            ).reshape(cell_shape)
        return orig_interpolate(coeffs, interpolant_fns)

    def _get_vmr_compat(
        lookup_gas_optics, vmr_lib, species_idx, vmr_fields=None,
    ):
        if (jax.default_backend() == "tpu"
                and isinstance(species_idx, (int, np.integer))):
            sp = int(species_idx)
            if vmr_fields is not None and sp in vmr_fields:
                return vmr_fields[sp]
            idx_gases = lookup_gas_optics.idx_gases
            vmr_gm = [0.0] * len(idx_gases)
            for k, v in vmr_lib.global_means.items():
                vmr_gm[idx_gases[k]] = v
            return jnp.asarray(
                vmr_gm[sp], dtype=lookup_gas_optics.vmr_ref.dtype,
            )
        return orig_get_vmr(
            lookup_gas_optics, vmr_lib, species_idx, vmr_fields,
        )

    def _rel_abund_compat(
        lookup_gas_optics,
        vmr_lib,
        troposphere_idx,
        temperature_idx,
        ibnd,
        scale_by_mixture,
        vmr_fields=None,
    ):
        if (jax.default_backend() != "tpu"
                or not isinstance(ibnd, (int, np.integer))
                or temperature_idx.ndim == 0):
            return orig_rel_abund(
                lookup_gas_optics, vmr_lib, troposphere_idx, temperature_idx,
                ibnd, scale_by_mixture, vmr_fields,
            )
        ks = np.asarray(lookup_gas_optics.key_species)[int(ibnd)]
        n_t = lookup_gas_optics.vmr_ref.shape[0]
        dtype = lookup_gas_optics.vmr_ref.dtype
        cell_shape = temperature_idx.shape
        ncells = math.prod(cell_shape)
        oh_t = jax.nn.one_hot(
            temperature_idx.reshape(ncells), n_t, dtype=dtype, axis=0,
        )
        is_tropo0 = troposphere_idx == 0
        sp00, sp10 = int(ks[0, 0]), int(ks[1, 0])
        sp01, sp11 = int(ks[0, 1]), int(ks[1, 1])
        vmr_for_interp = [
            jnp.where(
                is_tropo0,
                _get_vmr_compat(lookup_gas_optics, vmr_lib, sp00, vmr_fields),
                _get_vmr_compat(lookup_gas_optics, vmr_lib, sp10, vmr_fields),
            ),
            jnp.where(
                is_tropo0,
                _get_vmr_compat(lookup_gas_optics, vmr_lib, sp01, vmr_fields),
                _get_vmr_compat(lookup_gas_optics, vmr_lib, sp11, vmr_fields),
            ),
        ]
        ref_rows = jnp.stack(
            [
                lookup_gas_optics.vmr_ref[:, sp00, 0],
                lookup_gas_optics.vmr_ref[:, sp10, 1],
                lookup_gas_optics.vmr_ref[:, sp01, 0],
                lookup_gas_optics.vmr_ref[:, sp11, 1],
            ],
            axis=0,
        )
        ref_all = jnp.matmul(
            ref_rows, oh_t, precision=lax.Precision.HIGHEST,
        ).reshape(4, *cell_shape)
        vmr_ref_0 = jnp.where(is_tropo0, ref_all[0], ref_all[1])
        vmr_ref_1 = jnp.where(is_tropo0, ref_all[2], ref_all[3])
        vmr_ref_ratio = vmr_ref_0 / vmr_ref_1
        combined_vmr = vmr_for_interp[0] + vmr_ref_ratio * vmr_for_interp[1]
        is_present = combined_vmr > 0
        safe_combined_vmr = jnp.where(is_present, combined_vmr, 1.0)
        relative_abundance = jnp.where(
            is_present, vmr_for_interp[0] / safe_combined_vmr, 0.5,
        )
        interpolant = gas_optics._mixing_fraction_interpolant(
            relative_abundance, lookup_gas_optics.n_mixing_fraction,
        )
        if scale_by_mixture:
            interpolant.interp_low.weight *= combined_vmr
            interpolant.interp_high.weight *= combined_vmr
        return interpolant

    def _rows_te_compat(table, t_interp, mix_for_t, leading_idx=None):
        cell_shape = t_interp.interp_high.idx.shape
        if jax.default_backend() != "tpu" or len(cell_shape) == 0:
            return orig_rows_te(
                table, t_interp, mix_for_t, leading_idx=leading_idx,
            )
        ncells = math.prod(cell_shape)
        if leading_idx is None:
            n_t, n_eta, width = table.shape
            n_te = n_t * n_eta
            flat_table_t = table.reshape(n_te, width).T
            w_oh = None
            for eta_side in gas_optics._SIDES:
                for t_side in gas_optics._SIDES:
                    ti = getattr(t_interp, f"interp_{t_side}")
                    mi = getattr(mix_for_t[t_side], f"interp_{eta_side}")
                    idx = (ti.idx * n_eta + mi.idx).reshape(ncells)
                    w = (ti.weight * mi.weight).reshape(1, ncells)
                    oh = w * jax.nn.one_hot(
                        idx, n_te, dtype=table.dtype, axis=0,
                    )
                    w_oh = oh if w_oh is None else w_oh + oh
            return jnp.matmul(
                flat_table_t, w_oh, precision=lax.Precision.HIGHEST,
            ).T.reshape(*cell_shape, width)
        n_lead, n_t, n_eta, width = table.shape
        n_lte = n_lead * n_t * n_eta
        flat_table_t = table.reshape(n_lte, width).T
        lead_flat = leading_idx.reshape(ncells)
        w_oh = None
        for eta_side in gas_optics._SIDES:
            for t_side in gas_optics._SIDES:
                ti = getattr(t_interp, f"interp_{t_side}")
                mi = getattr(mix_for_t[t_side], f"interp_{eta_side}")
                idx = (
                    (lead_flat * n_t + ti.idx.reshape(ncells)) * n_eta
                    + mi.idx.reshape(ncells)
                )
                w = (ti.weight * mi.weight).reshape(1, ncells)
                oh = w * jax.nn.one_hot(
                    idx, n_lte, dtype=table.dtype, axis=0,
                )
                w_oh = oh if w_oh is None else w_oh + oh
        return jnp.matmul(
            flat_table_t, w_oh, precision=lax.Precision.HIGHEST,
        ).T.reshape(*cell_shape, width)

    def _tpm_block_plan_compat(ncells: int) -> tuple[int, int]:
        if ncells <= 4096:
            return 1, ncells
        if ncells % 128 == 0:
            for n_blocks in range(1, 65):
                if ncells % n_blocks != 0:
                    continue
                block_cells = ncells // n_blocks
                if block_cells % 128 == 0 and 12288 <= block_cells <= 32768:
                    return n_blocks, block_cells
        block_cells = 16384
        n_blocks = (ncells + block_cells - 1) // block_cells
        return n_blocks, block_cells

    def _rows_tpm_compat(table, t_interp, p_interp, mix_for_t):
        cell_shape = t_interp.interp_high.idx.shape
        if jax.default_backend() != "tpu" or len(cell_shape) == 0:
            return orig_rows_tpm(table, t_interp, p_interp, mix_for_t)
        ncells = math.prod(cell_shape)
        n_t, n_p, n_eta, width = table.shape
        n_te = n_t * n_eta
        table_pteg = jnp.transpose(table, (1, 0, 2, 3)).reshape(
            n_p, n_te, width,
        )
        n_blocks, block_cells = _tpm_block_plan_compat(ncells)
        padded_cells = n_blocks * block_cells
        pad = padded_cells - ncells

        def _pack_idx(arr):
            flat = arr.reshape(ncells)
            if pad:
                flat = jnp.pad(flat, (0, pad))
            return flat.reshape(n_blocks, block_cells)

        def _pack_w(arr):
            flat = arr.reshape(ncells)
            if pad:
                flat = jnp.pad(flat, (0, pad))
            return flat.reshape(n_blocks, 1, block_cells)

        p_lo_i_b = _pack_idx(p_interp.interp_low.idx)
        p_lo_w_b = _pack_w(p_interp.interp_low.weight)
        p_hi_i_b = _pack_idx(p_interp.interp_high.idx)
        p_hi_w_b = _pack_w(p_interp.interp_high.weight)

        te_idx_w = []
        for eta_side in gas_optics._SIDES:
            for t_side in gas_optics._SIDES:
                ti = getattr(t_interp, f"interp_{t_side}")
                mi = getattr(mix_for_t[t_side], f"interp_{eta_side}")
                te_idx_w.append((
                    _pack_idx(ti.idx * n_eta + mi.idx),
                    _pack_w(ti.weight * mi.weight),
                ))

        def _one_block(inputs):
            p_lo_i, p_lo_w, p_hi_i, p_hi_w, corners = inputs
            wp = (
                p_lo_w * jax.nn.one_hot(
                    p_lo_i, n_p, dtype=table.dtype, axis=0,
                )
                + p_hi_w * jax.nn.one_hot(
                    p_hi_i, n_p, dtype=table.dtype, axis=0,
                )
            )
            wte = None
            for idx_c, w_c in corners:
                oh = w_c * jax.nn.one_hot(
                    idx_c, n_te, dtype=table.dtype, axis=0,
                )
                wte = oh if wte is None else wte + oh
            mid = jnp.einsum(
                "pmg,mc->pgc",
                table_pteg,
                wte,
                precision=lax.Precision.HIGHEST,
            )
            return jnp.einsum(
                "pc,pgc->gc", wp, mid, precision=lax.Precision.HIGHEST,
            )

        if n_blocks == 1:
            first_corners = [(idx_b[0], w_b[0]) for idx_b, w_b in te_idx_w]
            out_gc = _one_block((
                p_lo_i_b[0],
                p_lo_w_b[0],
                p_hi_i_b[0],
                p_hi_w_b[0],
                first_corners,
            ))
            return out_gc.T[:ncells].reshape(*cell_shape, width)

        out_bgc = lax.map(
            _one_block, (p_lo_i_b, p_lo_w_b, p_hi_i_b, p_hi_w_b, te_idx_w),
        )
        out_flat = jnp.moveaxis(out_bgc, 1, 2).reshape(padded_cells, width)
        if pad:
            out_flat = out_flat[:ncells]
        return out_flat.reshape(*cell_shape, width)

    optics_utils.interpolate = _interpolate_compat
    gas_optics.get_vmr = _get_vmr_compat
    gas_optics._compute_relative_abundance_interpolant = _rel_abund_compat
    if orig_rows_te is not None:
        gas_optics._interpolate_rows_te = _rows_te_compat
    if orig_rows_tpm is not None:
        gas_optics._interpolate_rows_tpm = _rows_tpm_compat
    _RRTMGP_TPU_OPTICS_COMPAT_INSTALLED = True


def _greenhouse_gas_mole_fractions(chemistry, forcing):
    """Convert jcm's field-specific ppmv inputs to RRTMGP mol/mol values."""
    return (
        chemistry.ozone_mole_fraction(),
        chemistry.methane_mole_fraction(),
        ppmv_to_mole_fraction(forcing.co2_vmr),
        ppmv_to_mole_fraction(forcing.n2o_vmr),
    )


def _ensure_rrtmgp():
    """Lazily initialise the global RRTMGP instance on first use."""
    global _GLOBAL_RRTMGP_INSTANCE
    _install_rrtmgp_2d_zenith_compat()
    _install_rrtmgp_tpu_optics_compat()
    if _GLOBAL_RRTMGP_INSTANCE is not None:
        return _GLOBAL_RRTMGP_INSTANCE

    rrtmgp_root = Path(rrtmgp.__path__[0])
    rrtmgp_data_path = rrtmgp_root / "optics" / "rrtmgp_data"
    test_data_path = rrtmgp_root / "optics" / "test_data"

    # Constructed under scoped x64-off so the lookup tables load float32 even
    # on x64 hosts (pySES / MAM4-JAX) — see the compute_heating_rate call in
    # ``radiation_scheme_rrtmgp`` for the full rationale.
    with warnings.catch_warnings(), jax.enable_x64(False):
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
                # Emitted only when ``compute_cre`` asks the library for the
                # clear-sky fluxes (``clear_sky_fluxes=True``).
                "toa_sw_flux_outgoing_clearsky_2d_xy",
                "toa_lw_flux_outgoing_clearsky_2d_xy",
                # Emitted only when the aerosol-free companion is asked for
                # (``aerosol_free_fluxes=True``; the clear-sky pair needs
                # ``compute_cre`` too). The AeroCom ``*noa`` fluxes.
                "toa_sw_flux_outgoing_aerosolfree_2d_xy",
                "toa_lw_flux_outgoing_aerosolfree_2d_xy",
                "toa_sw_flux_outgoing_aerosolfree_clearsky_2d_xy",
                "toa_lw_flux_outgoing_aerosolfree_clearsky_2d_xy",
            ),
        )
        _sw_band_is_nir_mask(_GLOBAL_RRTMGP_INSTANCE)
    return _GLOBAL_RRTMGP_INSTANCE


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_3d_with_nan_halo(
    arr_1d: jnp.ndarray, nlev: int, halo: int = 1
) -> jnp.ndarray:
    """Convert 1D profile to 3D (1,1,nz+2*halo) with NaN halos (for temperature).

    The halo buffer is pinned to the profile's dtype: a dtype-less
    ``jnp.full`` defaults to float64 under ``jax_enable_x64`` (which pySES /
    MAM4-JAX hosts turn on process-wide), and a float64 temperature meeting
    the float32 ``vmr_fields`` inside the library's gas-optics ``lax.cond``
    branches is a trace-time TypeError — every other padder here already
    pins ``dtype=arr_1d.dtype``.
    """
    nzh = nlev + 2 * halo
    arr_3d = jnp.full((1, 1, nzh), jnp.nan, dtype=arr_1d.dtype)
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
    cloud-path inputs to the rrtmgp library, which indexes into the leading
    g-point axis inside its g-point loop. The halo cells hold no cloud: the
    library discards them, and a halo cell that repeated a cloudy surface or
    top layer carries the library's extrapolated halo temperature into a
    cloudy optical-depth computation whose derivative is not finite, although
    the fluxes are identical (measured: the lowest layer cloudy, the fluxes
    equal and the 19 non-finite derivative leaves of the term gone).
    """
    n_gpt = per_gpt_2d.shape[0]
    nzh = nlev + 2 * halo
    out = jnp.zeros((n_gpt, 1, 1, nzh), dtype=per_gpt_2d.dtype)
    return out.at[:, 0, 0, halo : halo + nlev].set(per_gpt_2d)


def _reverse_if_needed(pressure: jnp.ndarray) -> jnp.ndarray:
    """Return True if pressure increases with index (needs reversal for RRTMGP)."""
    return pressure[0] < pressure[-1]


#: Cached per-SW-band near-IR mask (jax-gcm#959), built once from the live
#: ``rrtmgp_instance``'s own band table -- see ``_sw_band_is_nir_mask``.
_SW_BAND_IS_NIR: Optional[np.ndarray] = None


def _sw_band_containing(rrtmgp_instance, wavelength_nm: float) -> int:
    """Index of the SW gas-optics band whose wavenumber range holds ``wavelength_nm``.

    Read off jax-rrtmgp's own band table (``bnd_lims_wn``, cm^-1) rather than
    a hardcoded index, like :func:`_sw_band_is_nir_mask`. At 550 nm
    (18182 cm^-1) this is the 16000-22650 cm^-1 band, the 11th of the 14
    RRTMGP SW bands (zero-based index 10).
    """
    wn = 1.0e7 / wavelength_nm
    lims = np.asarray(rrtmgp_instance.optics_lib.gas_optics_sw.bnd_lims_wn)
    hit = np.nonzero((lims[:, 0] <= wn) & (wn < lims[:, 1]))[0]
    if hit.size != 1:
        raise ValueError(
            f"{wavelength_nm} nm ({wn:.0f} cm^-1) is not in exactly one SW band"
        )
    return int(hit[0])


#: Wavelength at which the thresholded cloud cover evaluates the cloud optical
#: depth: the visible reference wavelength of cloud-mask and simulator
#: optical-depth thresholds.
COVER_TAU_WAVELENGTH_NM = 550.0


def _visible_cloud_optical_depth(
    rrtmgp_instance,
    r_eff_liq_um, r_eff_ice_um,
    in_cloud_liq_path, in_cloud_ice_path,
) -> jnp.ndarray:
    """Per-layer in-cloud liquid + ice extinction optical depth at 550 nm.

    Evaluated with the SAME jax-rrtmgp cloud-optics lookup the flux solve
    applies per g-point (``cloud_properties_fn`` for the band holding
    550 nm), on the same in-cloud paths and effective radii, once per layer
    rather than per g-point: a g-point's cloud path is ``mask * path``, so
    the optical depth of a sub-column is the mask-weighted sum of these
    per-layer values. This is the extinction optical depth BEFORE the solver's
    delta scaling (which removes forward-scattered energy for the two-stream
    transport) and WITHOUT the sub-grid inhomogeneity factors: a detector
    sees the physical optical depth, not the albedo-bias correction.

    Inputs and output are TOA-first ``(nlev,)`` (or multi-column
    ``(nlev, ncols)`` / ``(ncols, nlev)``); effective radii in um.
    """
    ibnd = _sw_band_containing(rrtmgp_instance, COVER_TAU_WAVELENGTH_NM)
    f32 = lambda a: jnp.asarray(a).astype(jnp.float32)  # noqa: E731
    if (
        jax.default_backend() == "tpu"
        and hasattr(two_stream, "_interp_1d_table_mxu")
    ):
        from rrtmgp import smooth_ops

        with jax.enable_x64(False):
            cld_lkp = rrtmgp_instance.optics_lib.cloud_optics_sw
            r_liq_m = f32(r_eff_liq_um) * 1e-6
            size_liq = jnp.clip(
                1.0e6 * r_liq_m,
                cld_lkp.radius_liq_lower,
                cld_lkp.radius_liq_upper,
            )
            grid_liq = jnp.linspace(
                cld_lkp.radius_liq_lower,
                cld_lkp.radius_liq_upper,
                cld_lkp.n_size_liq,
                dtype=cld_lkp.ext_liq.dtype,
            )
            ext_l = two_stream._interp_1d_table_mxu(
                cld_lkp.ext_liq[ibnd : ibnd + 1], size_liq, grid_liq,
            )[0]
            cld_l = f32(in_cloud_liq_path) * 1.0e3
            tau_l = ext_l * cld_l * smooth_ops.smooth_gate(cld_l, 0.0, 2.0e-6)

            r_ice_m = f32(r_eff_ice_um) * 1e-6
            size_ice = jnp.clip(
                2.0e6 * r_ice_m,
                cld_lkp.diameter_ice_lower,
                cld_lkp.diameter_ice_upper,
            )
            grid_ice = jnp.linspace(
                cld_lkp.diameter_ice_lower,
                cld_lkp.diameter_ice_upper,
                cld_lkp.n_size_ice,
                dtype=cld_lkp.ext_liq.dtype,
            )
            rough = cld_lkp.ice_roughness.value
            ext_i = two_stream._interp_1d_table_mxu(
                cld_lkp.ext_ice[rough][ibnd : ibnd + 1], size_ice, grid_ice,
            )[0]
            cld_i = f32(in_cloud_ice_path) * 1.0e3
            tau_i = ext_i * cld_i * smooth_ops.smooth_gate(cld_i, 0.0, 2.0e-6)
            return tau_l + tau_i
    props = rrtmgp_instance.optics_lib.cloud_properties_fn(ibnd, is_lw=False)
    with jax.enable_x64(False):
        out = props(
            f32(r_eff_liq_um) * 1e-6, f32(in_cloud_liq_path),
            f32(r_eff_ice_um) * 1e-6, f32(in_cloud_ice_path),
        )
    return out["optical_depth"]


def _sw_band_is_nir_mask(rrtmgp_instance) -> np.ndarray:
    """Return a ``(n_bnd_sw,)`` NumPy bool mask: True where the SW band is NIR.

    ECHAM's ``mo_surface`` hands the shortwave radiation a visible and a
    near-infrared surface albedo and applies the visible one below 0.7 um
    and the near-infrared one above it -- the TOA solar spectrum's split
    about that wavelength (the same split the RRTMGP wrapper's broadband
    blend, ``0.46*vis + 0.54*nir``, approximates with one value). This
    builds the per-band equivalent directly from jax-rrtmgp's own SW
    gas-optics band table (``bnd_lims_wn``, cm^-1 wavenumber limits) rather
    than a hardcoded band count or index list, so a change to the
    underlying k-distribution's band structure (e.g. a different
    ``*-gas-sw-g*.nc`` file) is picked up automatically. Band center
    wavelength follows ``band_config.py``'s own convention (``1e4 /
    (0.5*(wn_lo+wn_hi))`` in micrometres here, vs. that module's
    nanometres).

    A NumPy array, not a JAX one: ``bnd_lims_wn`` is a static table loaded at
    library-construction time, and the mask is used to build
    ``jnp.where(mask, nir, vis)`` per column, where NumPy broadcasts against
    the traced per-column scalars exactly as a JAX array would, with no
    tracing overhead for a value that can never change mid-run. Cached at
    module scope (keyed implicitly on the one process-global RRTMGP
    instance, ``_ensure_rrtmgp``) rather than recomputed every call.
    """
    global _SW_BAND_IS_NIR
    if _SW_BAND_IS_NIR is None:
        wn = np.asarray(rrtmgp_instance.optics_lib.gas_optics_sw.bnd_lims_wn)
        center_um = 1.0e4 / (0.5 * (wn[:, 0] + wn[:, 1]))
        _SW_BAND_IS_NIR = center_um > 0.7
    return _SW_BAND_IS_NIR


def _per_band_surface_albedo(
    vis: jnp.ndarray, nir: jnp.ndarray, nir_mask: np.ndarray,
) -> jnp.ndarray:
    """Per-column ``(vis, nir)`` scalars -> a ``(n_bnd_sw,)`` per-band spectrum.

    Every band below 0.7 um (``nir_mask`` False) takes ``vis``; every band at
    or above it takes ``nir`` (:func:`_sw_band_is_nir_mask`). Used for both
    the direct-beam and the diffuse albedo, so the two share the same band
    map applied to different (vis, nir) inputs.
    """
    return jnp.where(nir_mask, nir, vis)


def _flux_profiles(
    rrtmgp_data: dict, needs_reversal: jnp.ndarray, sky: str = "",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(sw_up, sw_down, lw_up, lw_down)`` interface profiles, ICON order.

    RRTMGP emits ``(1, ngpt, nlev+1)``; the g-point axis is summed out
    here — before the per-column vmap bundles the result — so the
    diagnostic stays ``(ncols, nlev+1)`` instead of blowing up to
    ``(ncols, nlev+1, ngpt)`` (ngpt is 128 LW / 112 SW, a ~120x memory
    saving), then flipped back to the caller's TOA-first ordering.

    Shared by the all-sky and clear-sky fluxes so both land on identical
    g-point handling and vertical orientation and can be differenced. ``sky``
    is ``""`` for the all-sky profiles and ``"clearsky_"`` for the clear-sky
    ones, which the library returns under ``sw_flux_up_clearsky_full`` etc.
    """
    flip = lambda a: a[::-1]  # noqa: E731
    identity = lambda a: a  # noqa: E731
    profiles = tuple(
        lax.cond(
            needs_reversal, flip, identity,
            rrtmgp_data[key][0, :, :].sum(axis=0),
        )
        for key in (f"sw_flux_up_{sky}full", f"sw_flux_down_{sky}full",
                    f"lw_flux_up_{sky}full", f"lw_flux_down_{sky}full")
    )
    return profiles


# ---------------------------------------------------------------------------
# Data conversion: ICON -> RRTMGP
# ---------------------------------------------------------------------------

def prepare_rrtmgp_data(
    icon_data,
    layer_thickness: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    r_eff_liq_um: jnp.ndarray,
    r_eff_ice_um: jnp.ndarray,
) -> dict:
    """Convert ICON RadiationState to RRTMGP input dict.

    Handles vertical ordering, halo padding, stretched-grid mapping,
    water-variable conversions, and cloud effective radii.

    Args:
        icon_data: RadiationState with TOA-first profiles.
        layer_thickness: geometric layer thickness (m), TOA-first.
        surface_temperature: scalar surface temperature (K).
        r_eff_liq_um / r_eff_ice_um: droplet / crystal effective radius
            (um), TOA-first (nlev,), used as given: the RRTMGP term forms
            them from the current state
            (:func:`~jcm.physics.radiation.cloud_optics.radiation_effective_radii`).
            The library clips both to its lookup-table range internally
            (radius for liquid, ``2 r`` as diameter for ice), so a clear
            layer's 0 is a valid input: it has no condensate path to weight.

    """
    nlev = icon_data.temperature.shape[0]
    halo = 1

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

    # The library wants q_t as a SPECIFIC humidity: it forms the vapour
    # mixing ratio itself as (q_t - q_c)/(1 - q_t). Reconstructing q from
    # h2o_vmr instead returned q/(1-q) -- because the grey scheme's
    # `h2o_vmr = q/(1-q)*1.608` and `1.608*eps = 1.0002` cancel -- so the
    # 1/(1-q) was applied twice and the H2O VMR reaching gas optics was
    # +2.1% at q = 20 g/kg (#678). Pass the specific humidity straight
    # through instead of round-tripping the grey convention.
    h2o_specific = lax.cond(
        needs_reversal, flip, identity, icon_data.specific_humidity,
    )
    total_water = h2o_specific + total_condensate

    # Cloud effective radii, microns -> metres (the library's unit).
    cloud_r_eff_liq = r_eff_liq_um * 1e-6
    cloud_r_eff_ice = r_eff_ice_um * 1e-6

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
    surface_albedo_vis_dir: jnp.ndarray,
    surface_albedo_vis_dif: jnp.ndarray,
    surface_albedo_nir_dir: jnp.ndarray,
    surface_albedo_nir_dif: jnp.ndarray,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Convert RRTMGP output dict back to ICON RadiationTendencies/RadiationData.

    ``surface_albedo_{vis,nir}_{dir,dif}`` (jax-gcm#959) are the per-band
    direct/diffuse quartet the solve actually used (``sfc_alb_dir``/
    ``sfc_alb_dif``, reduced back to the vis/nir pair for the published
    diagnostic, the same way ``surface_albedo_vis``/``_nir`` already publish
    the broadband value the solve used).
    """
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

    # Full flux profiles: g-points summed out and flipped back to ICON
    # order (see _flux_profiles). No consumer wants the per-gpoint detail.
    sw_flux_up, sw_flux_down, lw_flux_up, lw_flux_down = _flux_profiles(
        rrtmgp_data, needs_reversal,
    )

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
        surface_albedo_vis_dir=surface_albedo_vis_dir,
        surface_albedo_vis_dif=surface_albedo_vis_dif,
        surface_albedo_nir_dir=surface_albedo_nir_dir,
        surface_albedo_nir_dif=surface_albedo_nir_dif,
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
        lw_heating_rate=lw_heating,
        # Clear-sky profiles come from a separate solve the caller runs;
        # zero placeholders here, overwritten via ``.copy(...)`` below.
        sw_flux_up_clear=jnp.zeros_like(sw_flux_up),
        sw_flux_down_clear=jnp.zeros_like(sw_flux_down),
        lw_flux_up_clear=jnp.zeros_like(lw_flux_up),
        lw_flux_down_clear=jnp.zeros_like(lw_flux_down),
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
        toa_sw_up_noa=jnp.zeros_like(toa_sw_up),
        toa_lw_up_noa=jnp.zeros_like(toa_sw_up),
        toa_sw_up_clear_noa=jnp.zeros_like(toa_sw_up),
        noa_frac_toa_sw_up=jnp.zeros_like(toa_sw_up),
        noa_frac_toa_lw_up=jnp.zeros_like(toa_sw_up),
        noa_frac_toa_sw_up_clear=jnp.zeros_like(toa_sw_up),
        noa_frac_toa_lw_up_clear=jnp.zeros_like(toa_sw_up),
        toa_lw_up_clear_noa=jnp.zeros_like(toa_sw_up),
        total_cloud_cover=jnp.zeros_like(toa_sw_up),
        total_cloud_cover_tau=jnp.zeros_like(toa_sw_up),
        # Placeholder: this low-level helper never sees the cloud effective
        # radii (``radiation_scheme_rrtmgp``'s caller forms them from the
        # state, this function only assembles the library's flux output).
        # Both of this function's actual callers overwrite it immediately --
        # ``radiation_scheme_rrtmgp`` via its own ``diagnostics.copy(...)``
        # below (populated from its ``r_eff_liq_um``/``r_eff_ice_um``
        # arguments, the exact radii that call used, matching
        # ``radiation_scheme_emulated``'s identical pattern), and
        # ``RRTMGPRadiation._compute_full`` a second time on its own, final
        # ``RadiationData`` -- so the zero here is never read back by
        # anything, exactly like the clear-sky placeholders above.
        r_eff_liq_at_solve=jnp.zeros_like(sw_heating),
        r_eff_ice_at_solve=jnp.zeros_like(sw_heating),
        # ``step`` is owned by the enclosing ``RRTMGPRadiation`` carry —
        # the standalone scheme emits 0 and the term bumps the counter
        # after its compute-vs-cache cond.
        step=jnp.int32(0),
    )
    return tendencies, diagnostics


# ---------------------------------------------------------------------------
# Main entry point (ICON-compatible signature)
# ---------------------------------------------------------------------------


@jax.custom_jvp
def _zenith_from_mu0(mu0: jnp.ndarray) -> jnp.ndarray:
    """Return arccos(mu0) on [0, 1], with the overhead sun moved off zenith 0.

    See ``_solar_zenith_angle``. At mu0 = 1 the value is an angle whose cosine
    is still exactly 1 in mu0's dtype (half the square root of machine
    epsilon), so ``cos`` of it reproduces mu0 bit for bit; below 1 it is
    ``arccos`` itself.
    """
    below = mu0 < 1.0
    overhead = 0.5 * jnp.sqrt(jnp.finfo(jnp.result_type(mu0)).eps)
    return jnp.where(below, jnp.arccos(jnp.where(below, mu0, 0.0)), overhead)


@_zenith_from_mu0.defjvp
def _zenith_from_mu0_jvp(primals, tangents):
    # d(arccos mu)/dmu = -1/sin(arccos mu): arccos's own derivative wherever
    # mu0 < 1, and at the overhead value (sin > 0) the one that makes
    # d cos(zenith)/d mu0 exactly 1.
    (mu0,), (dmu0,) = primals, tangents
    zenith = _zenith_from_mu0(mu0)
    return zenith, -dmu0 / jnp.sin(zenith)


def _solar_zenith_angle(cos_zenith: jnp.ndarray) -> jnp.ndarray:
    """Return the solar zenith angle [rad] for ``cos_zenith``, clipped to [0, 1].

    Night columns (cos_zenith < 0) map to pi/2, where the library's mu0 = 0
    zeroes the direct beam.

    The library only ever uses cos(zenith), so what matters is the round
    trip mu0 -> zenith -> cos. Plain ``arccos`` breaks it at mu0 = 1, the sun
    overhead, which a float32 cos_zenith reaches anywhere within ~0.02
    degrees of the subsolar point: its derivative is infinite there and AD
    multiplies sin(0) = 0 by it, a NaN derivative with respect to the solar
    geometry. Zeroing that derivative instead would silently drop a real
    one, because inside the rounding band cos_zenith is below its maximum and
    still moves with the geometry. ``_zenith_from_mu0`` keeps the round trip
    the identity in value and in derivative: at mu0 = 1 it returns an angle
    whose cosine is exactly 1 in the working dtype, with the analytic
    derivative -1/sin(zenith). The upper clip is a ``where`` rather than a
    ``minimum``: at exactly 1 a ``minimum`` ties and would halve the
    derivative.
    """
    mu0 = jnp.maximum(cos_zenith, 0.0)
    mu0 = jnp.where(mu0 > 1.0, 1.0, mu0)
    return _zenith_from_mu0(mu0)


# Three performance options of the jax-rrtmgp solve, none of which changes the
# physics: where the gas optics is evaluated (``gas_optics_mode``), how many
# g-points its spectral loop solves per iteration (``gpt_chunk``), and whether
# the vertical recurrences run as a ``lax.scan`` over levels (``use_scan``)
# instead of an unrolled chain. ``all_gpts`` computes the gas absorption, Planck
# fraction and Rayleigh optical depth of every g-point in one pass ahead of the
# loop, with each k-table read a row gather over the g-points; ``per_gpt``
# computes them inside the loop one g-point (or chunk) at a time.
#
# The jcm defaults are the measured optimum on an A100 at T63L47
# (``t63-echam-rrtmgp``, one radiation call): 739 ms with a separate clear-sky
# solve, unrolled and ``per_gpt``; 350 ms with the clear sky shared and the
# recurrences fenced (both in the library); 202 ms with ``all_gpts`` and
# ``gpt_chunk=4``; 139 ms with the scan at ``gpt_chunk=64``, ~9.1 s/sim-day
# end to end. A scan
# step is a kernel over ``gpt_chunk`` x columns elements, so scanning loses at
# small chunks (each step is a launch) and wins once the chunk is large enough
# to amortise it, which needs the gas optics out of the loop. The price is
# memory: peak 12.5 GiB at T63L47 (+4.0 over the unrolled form) and 24.6 GiB at
# T63L95 (+8.0), inside a 40 GB card; ``gpt_chunk=16`` is the no-extra-memory
# setting (156.6 ms). The scanned and unrolled programs differ by at most 12
# float32 ulp on the fluxes (XLA fusion / FMA contraction), and the G=64 minimum
# is a property of the compiled program, to be re-measured when XLA or the
# library changes. The library's own defaults stay ``per_gpt`` / 1 /
# ``use_scan=False``, so a caller that does not opt in keeps the program it
# always had. On TPU, ``all_gpts`` contracts factored one-hot weights on the MXU
# with ``Precision.HIGHEST`` (``jax-rrtmgp#45``, mirrored for older releases by
# ``_install_rrtmgp_tpu_optics_compat``). See
# ``docs/source/design/rrtmgp_gpu_performance.md``.
GAS_OPTICS_MODES = ("per_gpt", "all_gpts")
DEFAULT_GAS_OPTICS_MODE = "all_gpts"
DEFAULT_GPT_CHUNK = 64
# On TPU, column-minor ``_radiation_col_minor_rrtmgp`` keeps the 128-lane
# column axis contiguous across gas optics, McICA, and two-stream scans;
# ``gpt_chunk=64`` (56 for 112-gpt SW) runs 2 spectral iterations per solver,
# and ``DEFAULT_TPU_COL_CHUNKS=2`` bounds full-model HBM well within 16 GiB.
DEFAULT_TPU_GPT_CHUNK = 64
DEFAULT_TPU_COL_CHUNKS = 2
DEFAULT_USE_SCAN = True


def _check_library_options(
    gas_optics_mode: str, gpt_chunk: int, use_scan: bool,
) -> tuple[int, bool]:
    """Validate the static library options; return ``(gpt_chunk, use_scan)``.

    All three select the traced program, so a bad value is rejected where the
    term is built rather than at the first trace.
    """
    if gas_optics_mode not in GAS_OPTICS_MODES:
        raise ValueError(
            f"gas_optics_mode must be one of {GAS_OPTICS_MODES}, "
            f"got {gas_optics_mode!r}.")
    if isinstance(gpt_chunk, bool) or int(gpt_chunk) != gpt_chunk or gpt_chunk < 1:
        raise ValueError(
            f"gpt_chunk must be a positive integer, got {gpt_chunk!r}.")
    if not isinstance(use_scan, (bool, np.bool_)):
        raise ValueError(f"use_scan must be a bool, got {use_scan!r}.")
    return int(gpt_chunk), bool(use_scan)


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
    column_index: jnp.ndarray = np.int32(0),  # numpy, not jnp: a jax default is built at import (#859)
    model_step: jnp.ndarray = np.int32(0),  # numpy, not jnp: a jax default is built at import (#859)
    base_seed: int = 0,
    compute_cre: bool = True,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: Optional[jnp.ndarray] = None,
    ch4_vmr: Optional[jnp.ndarray] = None,
    n2o_vmr: Optional[jnp.ndarray] = None,
    r_eff_liq_um: Optional[jnp.ndarray] = None,
    r_eff_ice_um: Optional[jnp.ndarray] = None,
    convection_type: jnp.ndarray = np.int32(0),  # numpy, not jnp: a jax default is built at import (#859)
    cloud_fraction_raw: Optional[jnp.ndarray] = None,
    # New optional parameters go at the END of the signature (mirroring the
    # jax-rrtmgp convention noted beside the library call below) so every
    # existing positional/kwarg caller keeps its bindings.
    surface_albedo_vis_dir: Optional[jnp.ndarray] = None,
    surface_albedo_vis_dif: Optional[jnp.ndarray] = None,
    surface_albedo_nir_dir: Optional[jnp.ndarray] = None,
    surface_albedo_nir_dif: Optional[jnp.ndarray] = None,
    lw_solver: str = "two_stream",
    gas_optics_mode: str = DEFAULT_GAS_OPTICS_MODE,
    gpt_chunk: int = DEFAULT_GPT_CHUNK,
    use_scan: bool = DEFAULT_USE_SCAN,
    solve_geometry: bool = False,
    aerosol_free: bool = False,
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

    When ``compute_cre`` is True the same RRTMGP call also returns the
    cloud-free fluxes (``clear_sky_fluxes=True``), which populate
    ``toa_{sw,lw}_up_clear`` for the cloud radiative effect diagnostic,
    plus the full ``{sw,lw}_flux_{up,down}_clear`` interface profiles used
    as NN emulator training labels. The library solves the clear sky inside
    the all-sky g-point loop, sharing the gas optics and Planck sources
    (the bulk of a g-point's cost), so it costs well under a second McICA
    call; disable it (e.g. for production runs that only need the all-sky
    fluxes) to drop even that.

    Args (additions over the previous beam-split signature):
        column_index: integer global index of the column being computed,
            used to seed the stochastic sub-column generator
            deterministically. Vmapped over the column axis upstream.
        model_step: scalar integer model step, also folded into the
            McICA seed so reruns are bit-exact reproducible.
        base_seed: term-level Python integer seed that the column +
            step indices fold into.
        compute_cre: if True, also take the clear-sky fluxes from the
            RRTMGP solve and populate ``toa_{sw,lw}_up_clear`` and the
            ``{sw,lw}_flux_{up,down}_clear`` profiles on the returned
            ``RadiationData``.
        r_eff_liq_um / r_eff_ice_um: droplet / crystal effective radii (um,
            TOA-first (nlev,)), REQUIRED despite the keyword default (the
            positional order predates them). The RRTMGP term forms them from
            the current step's state with ECHAM's
            ``mo_cloud_optics.f90::cloud_optics`` laws
            (:func:`~jcm.physics.radiation.cloud_optics.radiation_effective_radii`);
            they are used exactly as given, with no substitution.
        convection_type: the column's (previous-step) ECHAM ``ktype``, which
            selects the liquid inhomogeneity factor (see
            :func:`~jcm.physics.radiation.radiation_types.liquid_inhomogeneity`).
            0 (no convection) when not supplied.
        surface_albedo_vis_dir / _vis_dif / _nir_dir / _nir_dif: the per-band
            direct-beam / diffuse surface albedo quartet (jax-gcm#959),
            e.g. the open-water Fresnel direct albedo vs. its diffuse
            albedo (``jcm.physics.surface.echam.albedo.ocean_albedo``).
            Mapped to jax-rrtmgp 0.5.0's per-SW-band ``sfc_alb_dir``/
            ``sfc_alb_dif`` by wavelength (:func:`_sw_band_is_nir_mask`:
            bands below 0.7 um take the visible value, at or above it the
            near-infrared one, ECHAM's ``mo_surface`` split). ``None``
            (the default, every call site predating #959) falls back to
            the BLENDED BROADBAND scalar (``0.46*vis + 0.54*nir``) in all
            four slots -- not the raw ``vis``/``nir`` pair, which would
            give an unintended per-band split whenever ``vis != nir`` --
            reproducing today's single-albedo behaviour exactly, which is
            also the #959 regression pin (bit-identical fluxes whether
            ``vis == nir`` or not, as long as no quartet is passed).
        cloud_fraction_raw: the cover BEFORE ECHAM's condensate mask was
            applied to it (``cloud_fraction`` is the masked cover; see
            ``cloud_data.condensate_masked_cover``). Used only for the
            DERIVATIVE of the in-cloud condensate and the McICA cover with
            respect to condensate in a covered-but-empty cell (#973) — the
            forward computation is unaffected either way. ``None`` (the
            default, every call site predating #973) selects the PLAIN
            (zero-derivative-at-the-mask) functions outright, rather than
            substituting ``cloud_fraction`` for it: the ``_with_mask_gradient``
            surrogates divide by ``max(raw_cloud_fraction, eps)`` ungated, so
            passing the (zero, at a masked cell) masked value AS the raw one
            would not reproduce the plain functions' hard zero -- it would
            divide by ``eps`` instead, a large SPURIOUS derivative. Omitting
            the argument avoids that trap by skipping the surrogate
            entirely.
        solve_geometry: True when called by the radiation TERM, which stores
            the shortwave as transmissivities and re-applies the real sun
            every step (:func:`jcm.physics.radiation.apply_insolation`): the
            zenith is then ECHAM's solve zenith, curvature-corrected and
            floored at 0.1 (:func:`~jcm.physics.radiation.echam_solve_cos_zenith`),
            so night columns carry a finite transmissivity. False (default)
            keeps the geometric zenith, with night columns at zero flux, for a
            standalone call that reports the physical flux of ``solar``.
            Static.
        lw_solver: the library's longwave transport, ``"two_stream"`` (the
            default) or ``"noscat"`` (RTE's absorption-only solver; see
            :class:`RRTMGPRadiation`). Static: it selects the traced program.
        gas_optics_mode / gpt_chunk / use_scan: jax-rrtmgp's gas-optics
            placement, g-points per spectral-loop iteration and scanned
            vertical recurrences (see :class:`RRTMGPRadiation`). Static; they
            change the compiled program, not the physics.
        aerosol_free: also solve the aerosol-free twin of every problem (the
            all-sky one, and the clear-sky one under ``compute_cre``) inside
            the same library call and publish their TOA fluxes as
            ``toa_{sw,lw}_up_noa`` / ``toa_{sw,lw}_up_clear_noa`` (the
            AeroCom ``*noa`` diagnostics, jax-gcm#583). Static. The twins
            are the same gas optics, Planck sources and McICA cloud draw
            with no aerosol optics mixed in; the slots are zero when False.

    """
    if r_eff_liq_um is None or r_eff_ice_um is None:
        raise TypeError(
            "radiation_scheme_rrtmgp needs r_eff_liq_um and r_eff_ice_um: "
            "form them from the state with "
            "jcm.physics.radiation.cloud_optics.radiation_effective_radii "
            "(there is no fallback radius)."
        )

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
    if solve_geometry:
        cos_zenith = echam_solve_cos_zenith(cos_zenith)

    # In-cloud condensate (grid-mean / cf) is what each cloudy
    # sub-column sees; the binary McICA mask then re-imposes a
    # cloud-or-clear partitioning per g-point. ``in_cloud_path`` already
    # zeros the (essentially) clear cells (cf <= 2*eps; ECHAM mo_psrad).
    #
    # This is the PHYSICAL in-cloud condensate: the effective radii the caller
    # passes were formed from exactly this quantity (``in_cloud_condensate``
    # is shared with ``radiation_effective_radii``), and it is the condensate
    # ``q_c`` used in the vapour-VMR maths, so the sub-grid inhomogeneity
    # factor must NOT be baked in here -- that would shrink the inferred
    # crystal and partly undo the reduction (#678). The inhomogeneity factor
    # (ECHAM ``mo_cloud_optics.f90``: ``ztau = ztol*zinhoml + ztoi*zinhomi``)
    # is passed to jax-rrtmgp as ``cloud_tau_scale_liq``/``_ice`` alongside
    # the UNSCALED per-gpoint paths (``in_cloud_{lwp,ipath}_lib`` below, the
    # ``lib_in`` dict further down) rather than baked into those paths here,
    # so the library weights ssa/asymmetry by the unscaled per-phase τ as
    # ECHAM does (#958).
    #
    # Only the NaN guard (``_MAX_IN_CLOUD_CONDENSATE``) is applied: a *thin*
    # but resolved cloud (cf ~ 0.01-0.05) carrying large grid-mean condensate
    # yields a huge in-cloud water (grid_mean / cf) whose extreme optical depth
    # NaNs the two-stream solver. The cap = 10 g/kg is the high end of realistic
    # in-cloud water, so genuine clouds are untouched and only the pathological
    # inflation is clipped. It is a one-sided guard, NOT an inhomogeneity
    # scaling -- measured on T63L47 it binds in ~0.003% of cloudy cells (#678).
    # ``_with_mask_gradient``: forward value identical to the plain
    # ``in_cloud_condensate``/``effective_cloud_fraction`` (ECHAM's exact
    # condensate mask); derivative is the one-sided "if a trace of
    # condensate appeared it would radiate at the RAW diagnosed cover"
    # sensitivity, using ``cloud_fraction_raw`` (#973). ``cloud_fraction_raw
    # is None`` is a Python-level (trace-time) choice between the two
    # function pairs, not a value substitution -- see the argument's
    # docstring for why a substitution would be wrong.
    if cloud_fraction_raw is None:
        cloud_water_in_cloud = in_cloud_condensate(
            cloud_water, cloud_fraction, eps=parameters.cld_frac_min)
        cloud_ice_in_cloud = in_cloud_condensate(
            cloud_ice, cloud_fraction, eps=parameters.cld_frac_min)
    else:
        cloud_water_in_cloud = in_cloud_condensate_with_mask_gradient(
            cloud_water, cloud_fraction, cloud_fraction_raw,
            eps=parameters.cld_frac_min)
        cloud_ice_in_cloud = in_cloud_condensate_with_mask_gradient(
            cloud_ice, cloud_fraction, cloud_fraction_raw,
            eps=parameters.cld_frac_min)

    # The clear-cell threshold ``in_cloud_path`` uses to zero the condensate
    # (cf <= 2*cld_frac_min) must also gate the McICA sampler and the cover
    # diagnostic below: a cell whose in-cloud condensate was zeroed is
    # optically empty, so it must not be reported as cover nor bridge
    # maximum-random overlap through an empty layer (ECHAM ties condensate and
    # the icldlyr cloud flag together on the same test; see
    # ``effective_cloud_fraction``).
    if cloud_fraction_raw is None:
        cloud_fraction_rad = effective_cloud_fraction(
            cloud_fraction, eps=parameters.cld_frac_min,
        )
    else:
        cloud_fraction_rad = effective_cloud_fraction_with_mask_gradient(
            cloud_fraction, cloud_fraction_raw, eps=parameters.cld_frac_min,
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
        cloud_fraction=cloud_fraction_rad,
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
    # The decorrelation length stays an array: the sampler only uses it in
    # ``exp(-dz / L)``, and a Python ``float()`` here would make every
    # derivative with respect to the radiation parameters raise
    # (ConcretizationTypeError) instead of returning one. Its own derivative
    # here is zero — it reaches the fluxes only through the McICA draw's
    # ``y < alpha`` comparison — so it traces rather than calibrates; the
    # emulator's analytic expected cover does carry a derivative. The overlap
    # rule is an integer code that selects a code path, read as a Python int.
    decorrelation_km = parameters.cloud_decorrelation_km

    masks_lw = generate_subcolumns(
        cloud_fraction_rad, layer_thickness,
        n_subcols=n_gpt_lw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, key=key_lw,
    )    # [n_gpt_lw, nlev], TOA-first
    masks_sw = generate_subcolumns(
        cloud_fraction_rad, layer_thickness,
        n_subcols=n_gpt_sw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, key=key_sw,
    )    # [n_gpt_sw, nlev], TOA-first

    # Fraction of sub-columns (LW+SW draws pooled) with any cloudy layer.
    total_cloud_cover = (
        jnp.sum(jnp.any(masks_lw, axis=1)) + jnp.sum(jnp.any(masks_sw, axis=1))
    ) / (n_gpt_lw + n_gpt_sw)

    # Detection-thresholded cover of the SAME sub-columns: the fraction whose
    # column cloud optical depth at 550 nm exceeds ``tau_threshold``. Built
    # from the library's own cloud-optics lookup on the solve's in-cloud paths
    # and radii (one lookup per layer, not per g-point), so it adds no solver
    # work and feeds nothing back into the fluxes.
    tau_layer = _visible_cloud_optical_depth(
        rrtmgp_instance, r_eff_liq_um, r_eff_ice_um,
        resolvable_path(icon_state.cloud_water_path),
        resolvable_path(icon_state.cloud_ice_path),
    )
    total_cloud_cover_tau = optically_thick_cover(
        jnp.concatenate([masks_lw, masks_sw], axis=0), tau_layer,
        parameters.tau_threshold,
    )

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
    # Per-gpoint condensate paths carry the cloud optical depth (τ ∝ path at the
    # effective radius formed from the PHYSICAL in-cloud condensate). ECHAM's
    # sub-grid inhomogeneity factors scale the per-PHASE optical depth
    # (``mo_cloud_optics.f90``: ``ztau = ztol*zinhoml + ztoi*zinhomi``,
    # ``l_variable_inhoml = .FALSE.``) and then weight the combined ssa/asymmetry
    # by the UNSCALED ``ztol``/``ztoi``. jax-rrtmgp 0.5.0 exposes exactly that:
    # ``cloud_tau_scale_liq``/``_ice`` on ``compute_heating_rate`` (passed below),
    # applied after its own table lookup with ssa/asymmetry weighted by the
    # unscaled per-phase τ. So the paths handed to the library here are the
    # PHYSICAL (unscaled) per-phase paths -- scaling them ourselves, as a
    # pre-#958 version of this function did, would hand the library an
    # already-scaled τ that it would then weight ssa/asymmetry by directly,
    # recreating the same bug these two factors differ (shallow-convective
    # ``ktype == 4`` columns) that passing the factors separately fixes. The
    # liquid factor follows the column's convective type (``zinhoml`` 1/2/3,
    # :func:`liquid_inhomogeneity`); the ice factor is ``zinhomi``. Both are
    # per-COLUMN scalars (not per-level), so no orientation flip or halo
    # padding applies to them -- only the per-level paths below need it.
    zinhoml = liquid_inhomogeneity(convection_type, parameters)
    zinhomi = parameters.cloud_inhomogeneity_ice
    # The floor applies to the PHYSICAL (pre-scaling) path the library
    # receives: #958 moved the ECHAM inhomogeneity scaling from here into the
    # library's own ``cloud_tau_scale_{liq,ice}`` arguments, so this floor no
    # longer sees the scaled path. zinhoml/zinhomi are O(0.4-0.85) (never
    # below 0.4), so the floor's effective threshold on the SCALED path moves
    # by under a factor of ~2.5x either way -- far inside the "~9 orders of
    # magnitude of margin" the floor already has (see
    # ``mcica.NEGLIGIBLE_CLOUD_PATH_KG_M2``), so no recalibration is needed.
    in_cloud_lwp_lib = resolvable_path(lax.cond(
        needs_reversal, lambda a: a[::-1], identity,
        icon_state.cloud_water_path,
    ))
    in_cloud_ipath_lib = resolvable_path(lax.cond(
        needs_reversal, lambda a: a[::-1], identity,
        icon_state.cloud_ice_path,
    ))

    # Per-g-point path = mask * in-cloud path, as a plain product. A
    # surrogate on this product (making a covered-but-empty cell's
    # contribution carry a non-vanishing derivative, matching
    # ``in_cloud_condensate_with_mask_gradient`` above) was tried and
    # measured NOT to reach the flux: jax-rrtmgp 0.5.0's own cloud optics
    # (``rrtmgp.optics.cloud_optics``) multiplies this path by
    # ``smooth_ops.smooth_gate(path, 0, 2*EPSILON)``, a C1 smoothstep whose
    # VALUE AND FIRST DERIVATIVE are both exactly 0 at ``path = 0`` (by
    # construction, so the gate has no slope discontinuity at either end of
    # its ramp). ``d(path * gate(path))/dpath|_{path=0} = gate(0) + 0 =
    # 0``, for ANY tangent on ``path`` -- confirmed empirically: feeding this
    # exact function a non-vanishing tangent here still measures a zero
    # ``jax.grad`` of the flux, while the SAME column one ULP off the
    # degenerate point already has a large finite gradient. Closing this
    # needs jax-rrtmgp's gate itself to carry a one-sided slope at its
    # lower edge, which is outside jcm's RRTMGP backend; see jax-gcm#1093
    # (filed from #973) for the measurement and the upstream ask.
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
        icon_state, layer_thickness, surface_temperature,
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
    zenith_angle = _solar_zenith_angle(cos_zenith)
    irrad_val = direct_irradiance

    # Per-column surface boundary condition (jax-rrtmgp >= 0.2.1 hook —
    # previously the hardcoded AtmosphericStateCfg scalars 0.07/0.98 were
    # used for every column, severing the ice-albedo feedback and leaving
    # the column solve inconsistent with the surface scheme's absorbed
    # SW·(1−albedo_tile)). ``sfc_alb_broadband`` blends the surface scheme's
    # vis/nir pair with the ~0.46/0.54 split of the TOA solar spectrum about
    # 0.7 um; it is kept and passed as ``sfc_alb``, the library's broadband
    # fallback for a call that supplies no per-band albedo.
    #
    # jax-rrtmgp 0.5.0 also takes per-band direct and diffuse albedos
    # (``sfc_alb_dir``/``sfc_alb_dif``, jax-gcm#959): every band below
    # 0.7 um takes the visible value, every band at or above it the
    # near-infrared one (ECHAM's ``mo_surface`` split), read off
    # jax-rrtmgp's own SW band table rather than a hardcoded band count
    # (:func:`_sw_band_is_nir_mask`). The direct-beam and diffuse arrays
    # differ only over open water, whose Fresnel direct albedo is not its
    # diffuse albedo (``jcm.physics.surface.echam.albedo.ocean_albedo``);
    # land and sea ice repeat the same value in both. Absent inputs (every
    # call site predating #959) fall back to the BROADBAND BLEND
    # (``sfc_alb_broadband``) in all four slots -- not the raw vis/nir pair.
    # Falling back to raw vis/nir and then running them through the SAME
    # per-band split as a real quartet would give every band below 0.7 um
    # the visible value and every band at/above it the near-infrared one,
    # which is a per-band albedo an existing caller with vis != nir never
    # asked for and never saw before #959 (caught by Codex review: the
    # regression tests only exercised vis == nir, which hid this because
    # broadband == vis == nir there). Seeding every omitted slot from the
    # SAME blended scalar instead makes the per-band split below collapse
    # back to one uniform value across every band, for both beams --
    # exactly today's single-albedo behaviour, verified bit-identical to
    # the plain ``sfc_alb``-only call by
    # ``TestRRTMGPPerBandSurfaceAlbedo::test_vis_equal_nir_and_dir_equal_dif_matches_broadband_bit_for_bit``.
    sfc_alb_broadband = 0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir
    if surface_albedo_vis_dir is None:
        surface_albedo_vis_dir = sfc_alb_broadband
    if surface_albedo_vis_dif is None:
        surface_albedo_vis_dif = sfc_alb_broadband
    if surface_albedo_nir_dir is None:
        surface_albedo_nir_dir = sfc_alb_broadband
    if surface_albedo_nir_dif is None:
        surface_albedo_nir_dif = sfc_alb_broadband
    nir_mask = _sw_band_is_nir_mask(rrtmgp_instance)
    sfc_alb_dir = _per_band_surface_albedo(
        surface_albedo_vis_dir, surface_albedo_nir_dir, nir_mask)
    sfc_alb_dif = _per_band_surface_albedo(
        surface_albedo_vis_dif, surface_albedo_nir_dif, nir_mask)

    # The library call is float32 end-to-end regardless of the host's x64
    # state, in two coordinated steps (either alone is insufficient — the
    # first two derecho pySES JAM smoke runs failed on each half separately):
    #
    # 1. Every floating leaf crossing the library boundary is cast to
    #    float32. On ``jax_enable_x64`` hosts (pySES CAM-SE, MAM4-JAX)
    #    float64 leaks into *some* of these upstream (hybrid pressure
    #    tables, solar-geometry scalars), and one mixed-dtype leaf is a
    #    trace-time TypeError inside the library's gas-optics ``lax.cond``
    #    branches.
    # 2. The call itself runs under a scoped ``jax.enable_x64(False)``
    #    context so the library's own dtype-less internals (``jnp.float_``
    #    gas tables in ``get_vmr``, dtype-less literals) also come out
    #    float32 instead of float64.
    #
    # Together they reproduce exactly the float32 radiation the scheme is
    # validated with on non-x64 hosts, without touching the host's global
    # flag (upstream issue class: runs/UPSTREAM_ISSUE_jax-rrtmgp.md).
    def _f32_leaves(tree):
        return jax.tree_util.tree_map(
            lambda x: (x.astype(jnp.float32)
                       if isinstance(x, jnp.ndarray)
                       and jnp.issubdtype(x.dtype, jnp.floating) else x),
            tree,
        )

    lib_in = _f32_leaves(dict(
        zenith=zenith_angle, irrad=irrad_val,
        sfc_alb=sfc_alb_broadband, sfc_emis=surface_emissivity,
        sfc_alb_dir=sfc_alb_dir, sfc_alb_dif=sfc_alb_dif,
        cloud_path_liq_lw_per_gpt=cpl_lw_4d,
        cloud_path_ice_lw_per_gpt=cpi_lw_4d,
        cloud_path_liq_sw_per_gpt=cpl_sw_4d,
        cloud_path_ice_sw_per_gpt=cpi_sw_4d,
        vmr_fields=vmr_fields or None,
        aerosol_optics_sw=aerosol_optics_sw,
        aerosol_optics_lw=aerosol_optics_lw,
        # ECHAM's sub-grid inhomogeneity factors (#958): passed to the
        # library instead of pre-scaling the paths above, so it weights
        # ssa/asymmetry by the UNSCALED per-phase τ (``zinhoml``/``zinhomi``
        # are per-column scalars; see the comment above ``in_cloud_lwp_lib``).
        cloud_tau_scale_liq=zinhoml,
        cloud_tau_scale_ice=zinhomi,
        **rrtmgp_input,
    ))
    # ``compute_cre`` is a Python bool, so whether the library is asked for the
    # clear-sky fluxes is decided at trace time.
    # ``aerosol_free`` likewise: it asks the same call for the aerosol-free
    # twin of every problem it solves (the AeroCom ``*noa`` fluxes, jax-gcm#583).
    with jax.enable_x64(False):
        rrtmgp_output = rrtmgp_instance.compute_heating_rate(
            **lib_in, clear_sky_fluxes=compute_cre, lw_solver=lw_solver,
            gas_optics_mode=gas_optics_mode, gpt_chunk=gpt_chunk,
            use_scan=use_scan, aerosol_free_fluxes=aerosol_free,
        )

    # Aerosol-free (``*noa``) TOA fluxes: the all-sky and clear-sky problems
    # solved once more with no aerosol optics mixed in, inside the same
    # g-point loop (shared gas optics, Planck sources and cloud mix-in; only
    # the cell properties and the transport are repeated). The cloud field is
    # the same McICA draw by construction, so the difference to the fluxes
    # with aerosol is the instantaneous aerosol radiative effect (the ERFari
    # numerator), not an aerosol-cloud response. The clear-sky pair exists
    # only with ``compute_cre``; the slots are zero otherwise, as the
    # clear-sky fluxes themselves are.
    toa_sw_up_all = rrtmgp_output["toa_sw_flux_outgoing_2d_xy"][0, 0]
    if aerosol_free:
        toa_sw_up_noa = rrtmgp_output[
            "toa_sw_flux_outgoing_aerosolfree_2d_xy"][0, 0]
        toa_lw_up_noa = rrtmgp_output[
            "toa_lw_flux_outgoing_aerosolfree_2d_xy"][0, 0]
    else:
        toa_sw_up_noa = toa_lw_up_noa = jnp.zeros_like(toa_sw_up_all)
    if aerosol_free and compute_cre:
        toa_sw_up_clear_noa = rrtmgp_output[
            "toa_sw_flux_outgoing_aerosolfree_clearsky_2d_xy"][0, 0]
        toa_lw_up_clear_noa = rrtmgp_output[
            "toa_lw_flux_outgoing_aerosolfree_clearsky_2d_xy"][0, 0]
    else:
        toa_sw_up_clear_noa = toa_lw_up_clear_noa = jnp.zeros_like(
            toa_sw_up_all)

    # Optional clear-sky fluxes for the cloud radiative effect: the cloud-free
    # atmosphere, solved by the library inside the all-sky g-point loop so the
    # gas optics are computed once.
    # Aerosols are intentionally included on the clear-sky branch — CMIP
    # convention is that "clear-sky" means cloud-free, aerosols included — and
    # the clear-sky solve sees the same inputs minus the cloud condensate.
    if compute_cre:
        toa_sw_up_clear = (
            rrtmgp_output["toa_sw_flux_outgoing_clearsky_2d_xy"][0, 0]
        )
        toa_lw_up_clear = (
            rrtmgp_output["toa_lw_flux_outgoing_clearsky_2d_xy"][0, 0]
        )
        # Same g-point sum and vertical flip as the all-sky profiles, so
        # the clear-sky and all-sky interfaces line up element-wise.
        (sw_flux_up_clear, sw_flux_down_clear,
         lw_flux_up_clear, lw_flux_down_clear) = _flux_profiles(
            rrtmgp_output, _reverse_if_needed(icon_state.pressure),
            sky="clearsky_",
        )
    else:
        toa_sw_up_clear = jnp.zeros_like(
            rrtmgp_output["toa_sw_flux_outgoing_2d_xy"][0, 0],
        )
        toa_lw_up_clear = jnp.zeros_like(
            rrtmgp_output["toa_lw_flux_outgoing_2d_xy"][0, 0],
        )
        # (nlev+1,) zeros in the library's own dtype: one g-point slice of
        # the all-sky output has the shape and dtype of a summed profile.
        sw_flux_up_clear = jnp.zeros_like(
            rrtmgp_output["sw_flux_up_full"][0, 0, :],
        )
        sw_flux_down_clear = jnp.zeros_like(sw_flux_up_clear)
        lw_flux_up_clear = jnp.zeros_like(
            rrtmgp_output["lw_flux_up_full"][0, 0, :],
        )
        lw_flux_down_clear = jnp.zeros_like(lw_flux_up_clear)

    tendencies, diagnostics = prepare_icon_data(
        rrtmgp_output, icon_state,
        surface_albedo_vis, surface_albedo_nir, surface_emissivity,
        surface_albedo_vis_dir, surface_albedo_vis_dif,
        surface_albedo_nir_dir, surface_albedo_nir_dif,
    )
    diagnostics = diagnostics.copy(
        toa_sw_up_clear=toa_sw_up_clear,
        toa_lw_up_clear=toa_lw_up_clear,
        toa_sw_up_noa=toa_sw_up_noa,
        toa_lw_up_noa=toa_lw_up_noa,
        toa_sw_up_clear_noa=toa_sw_up_clear_noa,
        toa_lw_up_clear_noa=toa_lw_up_clear_noa,
        sw_flux_up_clear=sw_flux_up_clear,
        sw_flux_down_clear=sw_flux_down_clear,
        lw_flux_up_clear=lw_flux_up_clear,
        lw_flux_down_clear=lw_flux_down_clear,
        total_cloud_cover=total_cloud_cover,
        total_cloud_cover_tau=total_cloud_cover_tau,
        # The radii THIS call actually used -- its own ``r_eff_liq_um``/
        # ``r_eff_ice_um`` arguments, TOA-first and never reoriented by
        # ``prepare_rrtmgp_data`` (that flip is local to that function; see
        # its docstring) -- so they line up with ``sw_heating``'s ICON
        # orientation without any further flip. ``prepare_icon_data``
        # itself cannot fill these (it never receives the radii), hence the
        # zero placeholder there; populating them here, as
        # ``radiation_scheme_emulated`` already does for its own call,
        # keeps every direct caller of this canonical, ICON-signature
        # function correct -- e.g. ``tools/radiation_emulator/
        # generate_training_data.py``, which calls it directly rather than
        # through ``RRTMGPRadiation`` (jax-gcm#1135 review finding: this
        # function returned zeros here even though the term wrapper's own
        # final ``RadiationData`` always overwrote them again downstream).
        r_eff_liq_at_solve=r_eff_liq_um,
        r_eff_ice_at_solve=r_eff_ice_um,
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
    apply_insolation,
    echam_solve_cos_zenith,
    solve_solar_geometry,
    sw_solve_to_transmissivity,
    hold_land_albedo,
    radiation_should_compute,
    surface_optics_bands_for_solve,
    surface_optics_for_solve,
)
from jcm.physics.resolution_defaults import check_defaults_grid  # noqa: E402
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
    with the column count and reaches hundreds of GiB at T63L47 -- far past a
    single 80 GiB device -- while T21L47 needs only ~19 GiB. The placement of
    the gas optics sets the constant: XLA's backward scratch at T63L47 is 334
    GiB for ``per_gpt``/1/unrolled, 183 GiB for ``all_gpts``/4/unrolled and
    271 GiB for the jcm default ``all_gpts``/64/scanned (18.6, 10.2 and 15.1
    MiB per column; ``docs/source/design/rrtmgp_gpu_performance.md``).

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


def _grid_shape_for_columns(ncols: int) -> tuple[int, int]:
    """Choose a 2-D ``(nx, ny)`` horizontal grid layout for ``ncols`` columns.

    When ``ncols`` is a multiple of 128 (all standard T21/T31/T42/T63/T85/T127
    Gaussian grids), folding onto ``(ncols // 128, 128)`` aligns the minor
    horizontal axis with 128-lane vector registers and eliminates ``(1, 1)``
    dummy spatial dimensions that otherwise pad to ``(8, 128)`` physical tiles
    on TPU. Smaller or irregular column counts use ``(1, ncols)``.
    """
    if ncols >= 128 and ncols % 128 == 0:
        return ncols // 128, 128
    return 1, ncols


def _radiation_grid_rrtmgp(
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
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    parameters: RadiationParameters,
    aerosol_data,
    column_index: jnp.ndarray,
    model_step: jnp.ndarray,
    base_seed: int,
    compute_cre: bool,
    ozone_vmr: Optional[jnp.ndarray],
    co2_vmr: Optional[jnp.ndarray],
    ch4_vmr: Optional[jnp.ndarray],
    n2o_vmr: Optional[jnp.ndarray],
    r_eff_liq_um: jnp.ndarray,
    r_eff_ice_um: jnp.ndarray,
    convection_type: jnp.ndarray,
    cloud_fraction_raw: Optional[jnp.ndarray],
    surface_albedo_vis_dir: Optional[jnp.ndarray],
    surface_albedo_vis_dif: Optional[jnp.ndarray],
    surface_albedo_nir_dir: Optional[jnp.ndarray],
    surface_albedo_nir_dif: Optional[jnp.ndarray],
    *,
    lw_solver: str = "two_stream",
    gas_optics_mode: str = DEFAULT_GAS_OPTICS_MODE,
    gpt_chunk: int = DEFAULT_GPT_CHUNK,
    use_scan: bool = DEFAULT_USE_SCAN,
    solve_geometry: bool = False,
    aerosol_free: bool = False,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Multi-column RRTMGP solve on a native ``(nx, ny, nzh)`` 3-D grid.

    Computes the exact same physical quantities, McICA sub-column draws, and
    surrogate derivatives as ``jax.vmap(radiation_scheme_rrtmgp)`` over the
    leading ``ncols`` axis, while invoking ``jax-rrtmgp`` once on a dense
    ``(nx, ny, nlev + 2)`` domain instead of ``vmap``-ing single-column
    ``(1, 1, nlev + 2)`` solves. Avoiding ``(1, 1)`` spatial extents prevents
    128x-1024x physical-tile padding on TPU while preserving GPU vectorization.
    """
    ncols, nlev = temperature.shape
    nx, ny = _grid_shape_for_columns(ncols)
    halo = 1

    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    direct_irradiance = direct_solar_irradiance(
        solar.orbital_phase, parameters.solar_constant,
    )
    cos_zenith = get_solar_sin_altitude(orbital_time, longitude, latitude)
    if solve_geometry:
        cos_zenith = echam_solve_cos_zenith(cos_zenith)

    if cloud_fraction_raw is None:
        cloud_water_in_cloud = in_cloud_condensate(
            cloud_water, cloud_fraction, eps=parameters.cld_frac_min,
        )
        cloud_ice_in_cloud = in_cloud_condensate(
            cloud_ice, cloud_fraction, eps=parameters.cld_frac_min,
        )
        cloud_fraction_rad = effective_cloud_fraction(
            cloud_fraction, eps=parameters.cld_frac_min,
        )
    else:
        cloud_water_in_cloud = in_cloud_condensate_with_mask_gradient(
            cloud_water, cloud_fraction, cloud_fraction_raw,
            eps=parameters.cld_frac_min,
        )
        cloud_ice_in_cloud = in_cloud_condensate_with_mask_gradient(
            cloud_ice, cloud_fraction, cloud_fraction_raw,
            eps=parameters.cld_frac_min,
        )
        cloud_fraction_rad = effective_cloud_fraction_with_mask_gradient(
            cloud_fraction, cloud_fraction_raw, eps=parameters.cld_frac_min,
        )

    q_clipped = jnp.clip(specific_humidity, 0.0, 0.99)
    if ozone_vmr is None:
        p_mb = jnp.maximum(pressure_levels / 100.0, 1e-3)
        ozone_vmr = jnp.where(
            p_mb < 100.0,
            5e-6 * jnp.exp(-((jnp.log(p_mb) - jnp.log(30.0)) ** 2) / 2.0),
            1e-6,
        )
    cloud_water_path = (
        jnp.maximum(cloud_water_in_cloud, 0.0) * air_density * layer_thickness
    )
    cloud_ice_path = (
        jnp.maximum(cloud_ice_in_cloud, 0.0) * air_density * layer_thickness
    )

    rrtmgp_instance = _ensure_rrtmgp()
    n_gpt_lw = rrtmgp_instance.optics_lib.n_gpt_lw
    n_gpt_sw = rrtmgp_instance.optics_lib.n_gpt_sw

    model_step_eff = jnp.where(
        parameters.mcica_freeze_step != 0.0,
        jnp.zeros_like(model_step),
        model_step,
    )
    base_key = jax.random.PRNGKey(base_seed)
    col_keys = jax.vmap(
        lambda c: column_key(base_key, model_step=model_step_eff, column_index=c)
    )(column_index)
    split_keys = jax.vmap(jax.random.split)(col_keys)
    keys_lw, keys_sw = split_keys[:, 0], split_keys[:, 1]

    overlap_str = cloud_overlap_name(int(parameters.cloud_overlap))
    decorrelation_km = parameters.cloud_decorrelation_km
    cf_rad_lev = cloud_fraction_rad.T
    dz_lev = layer_thickness.T

    masks_lw = generate_subcolumns_batch(
        cf_rad_lev, dz_lev,
        n_subcols=n_gpt_lw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, keys=keys_lw,
    )  # [ncols, n_gpt_lw, nlev], TOA-first
    masks_sw = generate_subcolumns_batch(
        cf_rad_lev, dz_lev,
        n_subcols=n_gpt_sw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, keys=keys_sw,
    )  # [ncols, n_gpt_sw, nlev], TOA-first

    total_cloud_cover = (
        jnp.sum(jnp.any(masks_lw, axis=2), axis=1)
        + jnp.sum(jnp.any(masks_sw, axis=2), axis=1)
    ) / (n_gpt_lw + n_gpt_sw)

    tau_layer = _visible_cloud_optical_depth(
        rrtmgp_instance, r_eff_liq_um.T, r_eff_ice_um.T,
        resolvable_path(cloud_water_path).T,
        resolvable_path(cloud_ice_path).T,
    ).T
    col_tau_lw = jnp.sum(masks_lw * tau_layer[:, jnp.newaxis, :], axis=2)
    col_tau_sw = jnp.sum(masks_sw * tau_layer[:, jnp.newaxis, :], axis=2)
    total_cloud_cover_tau = (
        jnp.sum((col_tau_lw > parameters.tau_threshold).astype(masks_lw.dtype), axis=1)
        + jnp.sum((col_tau_sw > parameters.tau_threshold).astype(masks_sw.dtype), axis=1)
    ) / (n_gpt_lw + n_gpt_sw)

    needs_rev = pressure_levels[:, 0] < pressure_levels[:, -1]

    def _orient_2d(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(needs_rev[:, jnp.newaxis], arr[:, ::-1], arr)

    def _orient_3d(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            needs_rev[:, jnp.newaxis, jnp.newaxis], arr[:, :, ::-1], arr,
        )

    def _grid_nan_halo(arr_2d: jnp.ndarray) -> jnp.ndarray:
        g = arr_2d.reshape(nx, ny, nlev)
        pad = jnp.full((nx, ny, halo), jnp.nan, dtype=g.dtype)
        return jnp.concatenate([pad, g, pad], axis=2)

    def _grid_fill_halo(arr_2d: jnp.ndarray) -> jnp.ndarray:
        g = arr_2d.reshape(nx, ny, nlev)
        return jnp.concatenate([g[:, :, :1], g, g[:, :, -1:]], axis=2)

    def _grid_pressure_halo(
        p_2d: jnp.ndarray, dp_bot: jnp.ndarray, dp_top: jnp.ndarray,
    ) -> jnp.ndarray:
        g = p_2d.reshape(nx, ny, nlev)
        bot = (g[:, :, 1] + 2.0 * dp_bot.reshape(nx, ny))[:, :, jnp.newaxis]
        top = jnp.maximum(
            g[:, :, -2] - 2.0 * dp_top.reshape(nx, ny),
            1e-3 * g[:, :, -1],
        )[:, :, jnp.newaxis]
        return jnp.concatenate([bot, g, top], axis=2)

    def _grid_4d_zero_halo(per_gpt_3d: jnp.ndarray) -> jnp.ndarray:
        n_gpt = per_gpt_3d.shape[1]
        g = jnp.transpose(per_gpt_3d, (1, 0, 2)).reshape(n_gpt, nx, ny, nlev)
        pad = jnp.zeros((n_gpt, nx, ny, halo), dtype=g.dtype)
        return jnp.concatenate([pad, g, pad], axis=3)

    def _grid_4d_fill_halo(per_bnd_3d: jnp.ndarray) -> jnp.ndarray:
        oriented = _orient_3d(per_bnd_3d)
        n_bnd = oriented.shape[1]
        g = jnp.transpose(oriented, (1, 0, 2)).reshape(n_bnd, nx, ny, nlev)
        return jnp.concatenate([g[:, :, :, :1], g, g[:, :, :, -1:]], axis=3)

    masks_lw_lib = _orient_3d(masks_lw)
    masks_sw_lib = _orient_3d(masks_sw)

    zinhoml = liquid_inhomogeneity(convection_type, parameters).reshape(
        nx, ny, 1,
    )
    zinhomi = parameters.cloud_inhomogeneity_ice

    cwp_oriented = _orient_2d(cloud_water_path)
    cip_oriented = _orient_2d(cloud_ice_path)
    in_cloud_lwp_lib = resolvable_path(cwp_oriented)
    in_cloud_ipath_lib = resolvable_path(cip_oriented)

    cpl_lw_4d = _grid_4d_zero_halo(
        masks_lw_lib * in_cloud_lwp_lib[:, jnp.newaxis, :]
    )
    cpi_lw_4d = _grid_4d_zero_halo(
        masks_lw_lib * in_cloud_ipath_lib[:, jnp.newaxis, :]
    )
    cpl_sw_4d = _grid_4d_zero_halo(
        masks_sw_lib * in_cloud_lwp_lib[:, jnp.newaxis, :]
    )
    cpi_sw_4d = _grid_4d_zero_halo(
        masks_sw_lib * in_cloud_ipath_lib[:, jnp.newaxis, :]
    )

    rho_in = pressure_levels / (c.rd * temperature)
    dz_lib = _orient_2d(layer_thickness)
    rho_lib = _orient_2d(rho_in)
    temp_lib = _orient_2d(temperature)
    pres_lib = _orient_2d(pressure_levels)
    iface_lib = _orient_2d(pressure_interfaces)
    dp_bottom = iface_lib[:, 0] - iface_lib[:, 1]
    dp_top = iface_lib[:, -2] - iface_lib[:, -1]
    reff_liq_lib = _orient_2d(r_eff_liq_um) * 1e-6
    reff_ice_lib = _orient_2d(r_eff_ice_um) * 1e-6
    h2o_lib = _orient_2d(q_clipped)

    dz_3d = _grid_fill_halo(dz_lib)
    sg_map = {
        stretched_grid_util.hc_key(2): dz_3d,
        stretched_grid_util.hf_key(2): dz_3d,
    }
    cloud_water_mixing = cwp_oriented / (rho_lib * dz_lib)
    cloud_ice_mixing = cip_oriented / (rho_lib * dz_lib)
    total_condensate = cloud_water_mixing + cloud_ice_mixing
    total_water = h2o_lib + total_condensate

    q_t_3d = _grid_fill_halo(total_water) - _grid_fill_halo(total_condensate)
    zero_3d = jnp.zeros((nx, ny, nlev + 2 * halo), dtype=q_t_3d.dtype)

    rrtmgp_input = {
        "rho_xxc": _grid_fill_halo(rho_lib),
        "q_t": q_t_3d,
        "q_liq": zero_3d,
        "q_ice": zero_3d,
        "q_c": zero_3d,
        "cloud_r_eff_liq": _grid_fill_halo(reff_liq_lib),
        "cloud_r_eff_ice": _grid_fill_halo(reff_ice_lib),
        "temperature": _grid_nan_halo(temp_lib),
        "sfc_temperature": surface_temperature.reshape(nx, ny),
        "p_ref_xxc": _grid_pressure_halo(pres_lib, dp_bottom, dp_top),
        "sg_map": sg_map,
    }

    vmr_fields: dict[str, jnp.ndarray] = {
        "o3": _grid_fill_halo(_orient_2d(ozone_vmr)),
    }
    if co2_vmr is not None:
        vmr_fields["co2"] = _grid_fill_halo(
            _orient_2d(jnp.broadcast_to(co2_vmr, (ncols, nlev)))
        )
    if ch4_vmr is not None:
        vmr_fields["ch4"] = _grid_fill_halo(
            _orient_2d(jnp.broadcast_to(ch4_vmr, (ncols, nlev)))
        )
    if n2o_vmr is not None:
        vmr_fields["n2o"] = _grid_fill_halo(
            _orient_2d(jnp.broadcast_to(n2o_vmr, (ncols, nlev)))
        )

    aerosol_optics_sw: Optional[dict[str, jnp.ndarray]] = None
    if hasattr(aerosol_data, "aod_sw_per_band"):
        aerosol_optics_sw = {
            "optical_depth": _grid_4d_fill_halo(aerosol_data.aod_sw_per_band),
            "ssa": _grid_4d_fill_halo(aerosol_data.ssa_sw_per_band),
            "asymmetry_factor": _grid_4d_fill_halo(aerosol_data.asy_sw_per_band),
        }

    aerosol_optics_lw: Optional[dict[str, jnp.ndarray]] = None
    if (hasattr(aerosol_data, "aod_lw_per_band")
            and aerosol_data.aod_lw_per_band.shape[1] > 0):
        aerosol_optics_lw = {
            "optical_depth": _grid_4d_fill_halo(aerosol_data.aod_lw_per_band),
            "ssa": _grid_4d_fill_halo(aerosol_data.ssa_lw_per_band),
            "asymmetry_factor": _grid_4d_fill_halo(aerosol_data.asy_lw_per_band),
        }

    zenith_2d = _solar_zenith_angle(cos_zenith).reshape(nx, ny)
    sfc_alb_broadband = (
        0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir
    ).reshape(nx, ny)
    vis_dir = (
        sfc_alb_broadband if surface_albedo_vis_dir is None
        else surface_albedo_vis_dir.reshape(nx, ny)
    )
    vis_dif = (
        sfc_alb_broadband if surface_albedo_vis_dif is None
        else surface_albedo_vis_dif.reshape(nx, ny)
    )
    nir_dir = (
        sfc_alb_broadband if surface_albedo_nir_dir is None
        else surface_albedo_nir_dir.reshape(nx, ny)
    )
    nir_dif = (
        sfc_alb_broadband if surface_albedo_nir_dif is None
        else surface_albedo_nir_dif.reshape(nx, ny)
    )
    nir_mask = _sw_band_is_nir_mask(rrtmgp_instance)[jnp.newaxis, jnp.newaxis, :]
    sfc_alb_dir = jnp.where(
        nir_mask, nir_dir[:, :, jnp.newaxis], vis_dir[:, :, jnp.newaxis],
    )
    sfc_alb_dif = jnp.where(
        nir_mask, nir_dif[:, :, jnp.newaxis], vis_dif[:, :, jnp.newaxis],
    )

    def _f32_leaves(tree):
        return jax.tree_util.tree_map(
            lambda x: (x.astype(jnp.float32)
                       if isinstance(x, jnp.ndarray)
                       and jnp.issubdtype(x.dtype, jnp.floating) else x),
            tree,
        )

    lib_in = _f32_leaves(dict(
        zenith=zenith_2d,
        irrad=direct_irradiance,
        sfc_alb=sfc_alb_broadband,
        sfc_emis=surface_emissivity.reshape(nx, ny),
        sfc_alb_dir=sfc_alb_dir,
        sfc_alb_dif=sfc_alb_dif,
        cloud_path_liq_lw_per_gpt=cpl_lw_4d,
        cloud_path_ice_lw_per_gpt=cpi_lw_4d,
        cloud_path_liq_sw_per_gpt=cpl_sw_4d,
        cloud_path_ice_sw_per_gpt=cpi_sw_4d,
        vmr_fields=vmr_fields,
        aerosol_optics_sw=aerosol_optics_sw,
        aerosol_optics_lw=aerosol_optics_lw,
        cloud_tau_scale_liq=zinhoml,
        cloud_tau_scale_ice=zinhomi,
        **rrtmgp_input,
    ))

    with jax.enable_x64(False):
        rrtmgp_output = rrtmgp_instance.compute_heating_rate(
            **lib_in, clear_sky_fluxes=compute_cre, lw_solver=lw_solver,
            gas_optics_mode=gas_optics_mode, gpt_chunk=gpt_chunk,
            use_scan=use_scan, aerosol_free_fluxes=aerosol_free,
        )

    def _grid_flux_profiles(sky: str = ""):
        return tuple(
            _orient_2d(rrtmgp_output[key].reshape(ncols, nlev + 1))
            for key in (
                f"sw_flux_up_{sky}full",
                f"sw_flux_down_{sky}full",
                f"lw_flux_up_{sky}full",
                f"lw_flux_down_{sky}full",
            )
        )

    total_heating = _orient_2d(
        rrtmgp_output["rad_heat_src"][:, :, halo:halo + nlev].reshape(ncols, nlev)
    )
    lw_heating = _orient_2d(
        rrtmgp_output["rad_heat_lw_3d"][:, :, halo:halo + nlev].reshape(ncols, nlev)
    )
    sw_heating = _orient_2d(
        rrtmgp_output["rad_heat_sw_3d"][:, :, halo:halo + nlev].reshape(ncols, nlev)
    )

    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating,
    )

    surf_sw_down = rrtmgp_output["surf_sw_flux_down_2d_xy"].reshape(ncols)
    surf_sw_up = rrtmgp_output["surf_sw_flux_up_2d_xy"].reshape(ncols)
    surf_lw_down = rrtmgp_output["surf_lw_flux_down_2d_xy"].reshape(ncols)
    surf_lw_up = rrtmgp_output["surf_lw_flux_up_2d_xy"].reshape(ncols)
    toa_sw_down = rrtmgp_output["toa_sw_flux_incoming_2d_xy"].reshape(ncols)
    toa_sw_up = rrtmgp_output["toa_sw_flux_outgoing_2d_xy"].reshape(ncols)
    toa_lw_up = rrtmgp_output["toa_lw_flux_outgoing_2d_xy"].reshape(ncols)

    sw_flux_up, sw_flux_down, lw_flux_up, lw_flux_down = _grid_flux_profiles()

    if aerosol_free:
        toa_sw_up_noa = rrtmgp_output[
            "toa_sw_flux_outgoing_aerosolfree_2d_xy"
        ].reshape(ncols)
        toa_lw_up_noa = rrtmgp_output[
            "toa_lw_flux_outgoing_aerosolfree_2d_xy"
        ].reshape(ncols)
    else:
        toa_sw_up_noa = toa_lw_up_noa = jnp.zeros_like(toa_sw_up)

    if aerosol_free and compute_cre:
        toa_sw_up_clear_noa = rrtmgp_output[
            "toa_sw_flux_outgoing_aerosolfree_clearsky_2d_xy"
        ].reshape(ncols)
        toa_lw_up_clear_noa = rrtmgp_output[
            "toa_lw_flux_outgoing_aerosolfree_clearsky_2d_xy"
        ].reshape(ncols)
    else:
        toa_sw_up_clear_noa = toa_lw_up_clear_noa = jnp.zeros_like(toa_sw_up)

    if compute_cre:
        toa_sw_up_clear = rrtmgp_output[
            "toa_sw_flux_outgoing_clearsky_2d_xy"
        ].reshape(ncols)
        toa_lw_up_clear = rrtmgp_output[
            "toa_lw_flux_outgoing_clearsky_2d_xy"
        ].reshape(ncols)
        (sw_flux_up_clear, sw_flux_down_clear,
         lw_flux_up_clear, lw_flux_down_clear) = _grid_flux_profiles(
            sky="clearsky_",
        )
    else:
        toa_sw_up_clear = jnp.zeros_like(toa_sw_up)
        toa_lw_up_clear = jnp.zeros_like(toa_lw_up)
        sw_flux_up_clear = jnp.zeros_like(sw_flux_up)
        sw_flux_down_clear = jnp.zeros_like(sw_flux_down)
        lw_flux_up_clear = jnp.zeros_like(lw_flux_up)
        lw_flux_down_clear = jnp.zeros_like(lw_flux_down)

    broadband_for_pub = 0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir
    pub_vis_dir = (
        broadband_for_pub if surface_albedo_vis_dir is None
        else surface_albedo_vis_dir
    )
    pub_vis_dif = (
        broadband_for_pub if surface_albedo_vis_dif is None
        else surface_albedo_vis_dif
    )
    pub_nir_dir = (
        broadband_for_pub if surface_albedo_nir_dir is None
        else surface_albedo_nir_dir
    )
    pub_nir_dif = (
        broadband_for_pub if surface_albedo_nir_dif is None
        else surface_albedo_nir_dif
    )

    zero_col = jnp.zeros_like(toa_sw_up)
    diagnostics = RadiationData(
        cos_zenith=cos_zenith[:, jnp.newaxis],
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
        surface_albedo_vis_dir=pub_vis_dir,
        surface_albedo_vis_dif=pub_vis_dif,
        surface_albedo_nir_dir=pub_nir_dir,
        surface_albedo_nir_dif=pub_nir_dif,
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
        lw_heating_rate=lw_heating,
        sw_flux_up_clear=sw_flux_up_clear,
        sw_flux_down_clear=sw_flux_down_clear,
        lw_flux_up_clear=lw_flux_up_clear,
        lw_flux_down_clear=lw_flux_down_clear,
        surface_sw_down=surf_sw_down,
        surface_lw_down=surf_lw_down,
        surface_sw_up=surf_sw_up,
        surface_lw_up=surf_lw_up,
        toa_sw_up=toa_sw_up,
        toa_lw_up=toa_lw_up,
        toa_sw_down=toa_sw_down,
        toa_sw_up_clear=toa_sw_up_clear,
        toa_lw_up_clear=toa_lw_up_clear,
        toa_sw_up_noa=toa_sw_up_noa,
        toa_lw_up_noa=toa_lw_up_noa,
        toa_sw_up_clear_noa=toa_sw_up_clear_noa,
        toa_lw_up_clear_noa=toa_lw_up_clear_noa,
        noa_frac_toa_sw_up=zero_col,
        noa_frac_toa_lw_up=zero_col,
        noa_frac_toa_sw_up_clear=zero_col,
        noa_frac_toa_lw_up_clear=zero_col,
        total_cloud_cover=total_cloud_cover,
        total_cloud_cover_tau=total_cloud_cover_tau,
        r_eff_liq_at_solve=r_eff_liq_um,
        r_eff_ice_at_solve=r_eff_ice_um,
        step=jnp.zeros((ncols,), dtype=jnp.int32),
    )
    return tendencies, diagnostics


def _radiation_col_minor_rrtmgp(
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
    latitude: jnp.ndarray,
    longitude: jnp.ndarray,
    parameters: RadiationParameters,
    aerosol_sw: Optional[dict[str, jnp.ndarray]],
    aerosol_lw: Optional[dict[str, jnp.ndarray]],
    column_index: jnp.ndarray,
    model_step: jnp.ndarray,
    base_seed: int,
    compute_cre: bool,
    ozone_vmr: Optional[jnp.ndarray],
    co2_vmr: Optional[jnp.ndarray],
    ch4_vmr: Optional[jnp.ndarray],
    n2o_vmr: Optional[jnp.ndarray],
    r_eff_liq_um: jnp.ndarray,
    r_eff_ice_um: jnp.ndarray,
    convection_type: jnp.ndarray,
    cloud_fraction_raw: Optional[jnp.ndarray],
    surface_albedo_vis_dir: Optional[jnp.ndarray],
    surface_albedo_vis_dif: Optional[jnp.ndarray],
    surface_albedo_nir_dir: Optional[jnp.ndarray],
    surface_albedo_nir_dif: Optional[jnp.ndarray],
    *,
    lw_solver: str = "two_stream",
    gas_optics_mode: str = DEFAULT_GAS_OPTICS_MODE,
    gpt_chunk: int = DEFAULT_GPT_CHUNK,
    use_scan: bool = DEFAULT_USE_SCAN,
    solve_geometry: bool = False,
    aerosol_free: bool = False,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Column-minor ``(nlev, nc)`` TPU RRTMGP solve via ``compute_heating_rate_zc``.

    Eliminates all ``(nlev, nc) <-> (nc, nlev)`` transposes, McICA 3-D
    transposes, and ``(n_gpt, nx, ny, nlev + 2)`` haloed 4-D intermediates by
    keeping the 128-lane-aligned column axis ``nc`` contiguous throughout McICA,
    gas optics, cloud/Planck table lookups, and vertical two-stream scans.
    """
    nlev, nc = temperature.shape
    halo = 1

    orbital_time = OrbitalTime(
        orbital_phase=solar.orbital_phase,
        synodic_phase=solar.synodic_phase,
    )
    direct_irradiance = direct_solar_irradiance(
        solar.orbital_phase, parameters.solar_constant,
    )
    cos_zenith = get_solar_sin_altitude(orbital_time, longitude, latitude)
    if solve_geometry:
        cos_zenith = echam_solve_cos_zenith(cos_zenith)

    if cloud_fraction_raw is None:
        cloud_water_in_cloud = in_cloud_condensate(
            cloud_water, cloud_fraction, eps=parameters.cld_frac_min,
        )
        cloud_ice_in_cloud = in_cloud_condensate(
            cloud_ice, cloud_fraction, eps=parameters.cld_frac_min,
        )
        cloud_fraction_rad = effective_cloud_fraction(
            cloud_fraction, eps=parameters.cld_frac_min,
        )
    else:
        cloud_water_in_cloud = in_cloud_condensate_with_mask_gradient(
            cloud_water, cloud_fraction, cloud_fraction_raw,
            eps=parameters.cld_frac_min,
        )
        cloud_ice_in_cloud = in_cloud_condensate_with_mask_gradient(
            cloud_ice, cloud_fraction, cloud_fraction_raw,
            eps=parameters.cld_frac_min,
        )
        cloud_fraction_rad = effective_cloud_fraction_with_mask_gradient(
            cloud_fraction, cloud_fraction_raw, eps=parameters.cld_frac_min,
        )

    q_clipped = jnp.clip(specific_humidity, 0.0, 0.99)
    if ozone_vmr is None:
        p_mb = jnp.maximum(pressure_levels / 100.0, 1e-3)
        ozone_vmr = jnp.where(
            p_mb < 100.0,
            5e-6 * jnp.exp(-((jnp.log(p_mb) - jnp.log(30.0)) ** 2) / 2.0),
            1e-6,
        )
    cloud_water_path = (
        jnp.maximum(cloud_water_in_cloud, 0.0) * air_density * layer_thickness
    )
    cloud_ice_path = (
        jnp.maximum(cloud_ice_in_cloud, 0.0) * air_density * layer_thickness
    )

    rrtmgp_instance = _ensure_rrtmgp()
    n_gpt_lw = rrtmgp_instance.optics_lib.n_gpt_lw
    n_gpt_sw = rrtmgp_instance.optics_lib.n_gpt_sw

    model_step_eff = jnp.where(
        parameters.mcica_freeze_step != 0.0,
        jnp.zeros_like(model_step),
        model_step,
    )
    base_key = jax.random.PRNGKey(base_seed)
    col_keys = jax.vmap(
        lambda col_i: column_key(
            base_key, model_step=model_step_eff, column_index=col_i,
        )
    )(column_index)
    split_keys = jax.vmap(jax.random.split)(col_keys)
    keys_lw, keys_sw = split_keys[:, 0], split_keys[:, 1]

    overlap_str = cloud_overlap_name(int(parameters.cloud_overlap))
    decorrelation_km = parameters.cloud_decorrelation_km

    masks_lw = generate_subcolumns_batch(
        cloud_fraction_rad, layer_thickness,
        n_subcols=n_gpt_lw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, keys=keys_lw,
        col_minor_output=True,
    )  # [nlev, n_gpt_lw, nc], TOA-first
    masks_sw = generate_subcolumns_batch(
        cloud_fraction_rad, layer_thickness,
        n_subcols=n_gpt_sw, overlap=overlap_str,
        decorrelation_km=decorrelation_km, keys=keys_sw,
        col_minor_output=True,
    )  # [nlev, n_gpt_sw, nc], TOA-first

    total_cloud_cover = (
        jnp.sum(jnp.any(masks_lw, axis=0), axis=0)
        + jnp.sum(jnp.any(masks_sw, axis=0), axis=0)
    ) / (n_gpt_lw + n_gpt_sw)

    tau_layer = _visible_cloud_optical_depth(
        rrtmgp_instance, r_eff_liq_um, r_eff_ice_um,
        resolvable_path(cloud_water_path),
        resolvable_path(cloud_ice_path),
    )
    col_tau_lw = jnp.sum(masks_lw * tau_layer[:, jnp.newaxis, :], axis=0)
    col_tau_sw = jnp.sum(masks_sw * tau_layer[:, jnp.newaxis, :], axis=0)
    total_cloud_cover_tau = (
        jnp.sum((col_tau_lw > parameters.tau_threshold).astype(masks_lw.dtype), axis=0)
        + jnp.sum((col_tau_sw > parameters.tau_threshold).astype(masks_sw.dtype), axis=0)
    ) / (n_gpt_lw + n_gpt_sw)

    needs_rev = pressure_levels[0, :] < pressure_levels[-1, :]

    def _orient_zc(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(needs_rev[jnp.newaxis, :], arr[::-1, :], arr)

    def _orient_zgc(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            needs_rev[jnp.newaxis, jnp.newaxis, :], arr[::-1, :, :], arr,
        )

    def _orient_bzc(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            needs_rev[jnp.newaxis, jnp.newaxis, :], arr[:, ::-1, :], arr,
        )

    def _zc_nan_halo(arr_2d: jnp.ndarray) -> jnp.ndarray:
        pad = jnp.full((halo, nc), jnp.nan, dtype=arr_2d.dtype)
        return jnp.concatenate([pad, arr_2d, pad], axis=0)

    def _zc_fill_halo(arr_2d: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([arr_2d[:1, :], arr_2d, arr_2d[-1:, :]], axis=0)

    def _zc_pressure_halo(
        p_2d: jnp.ndarray, dp_bot: jnp.ndarray, dp_top: jnp.ndarray,
    ) -> jnp.ndarray:
        bot = (p_2d[1, :] + 2.0 * dp_bot)[jnp.newaxis, :]
        top = jnp.maximum(
            p_2d[-2, :] - 2.0 * dp_top,
            1e-3 * p_2d[-1, :],
        )[jnp.newaxis, :]
        return jnp.concatenate([bot, p_2d, top], axis=0)

    masks_lw_lib = _orient_zgc(masks_lw)
    masks_sw_lib = _orient_zgc(masks_sw)

    zinhoml = liquid_inhomogeneity(convection_type, parameters)[jnp.newaxis, :]
    zinhomi = parameters.cloud_inhomogeneity_ice

    cwp_oriented = _orient_zc(cloud_water_path)
    cip_oriented = _orient_zc(cloud_ice_path)
    in_cloud_lwp_lib = resolvable_path(cwp_oriented)
    in_cloud_ipath_lib = resolvable_path(cip_oriented)

    cpl_lw_zc = masks_lw_lib * in_cloud_lwp_lib[:, jnp.newaxis, :]
    cpi_lw_zc = masks_lw_lib * in_cloud_ipath_lib[:, jnp.newaxis, :]
    cpl_sw_zc = masks_sw_lib * in_cloud_lwp_lib[:, jnp.newaxis, :]
    cpi_sw_zc = masks_sw_lib * in_cloud_ipath_lib[:, jnp.newaxis, :]

    rho_in = pressure_levels / (c.rd * temperature)
    dz_lib = _orient_zc(layer_thickness)
    rho_lib = _orient_zc(rho_in)
    temp_lib = _orient_zc(temperature)
    pres_lib = _orient_zc(pressure_levels)
    iface_lib = _orient_zc(pressure_interfaces)
    dp_bottom = iface_lib[0, :] - iface_lib[1, :]
    dp_top = iface_lib[-2, :] - iface_lib[-1, :]
    reff_liq_lib = _orient_zc(r_eff_liq_um) * 1e-6
    reff_ice_lib = _orient_zc(r_eff_ice_um) * 1e-6
    h2o_lib = _orient_zc(q_clipped)

    cloud_water_mixing = cwp_oriented / (rho_lib * dz_lib)
    cloud_ice_mixing = cip_oriented / (rho_lib * dz_lib)
    total_condensate = cloud_water_mixing + cloud_ice_mixing
    total_water = h2o_lib + total_condensate

    q_t_zc = _zc_fill_halo(total_water) - _zc_fill_halo(total_condensate)

    rrtmgp_input = {
        "q_t": q_t_zc,
        "cloud_r_eff_liq": reff_liq_lib,
        "cloud_r_eff_ice": reff_ice_lib,
        "temperature": _zc_nan_halo(temp_lib),
        "sfc_temperature": surface_temperature,
        "p_ref_zc": _zc_pressure_halo(pres_lib, dp_bottom, dp_top),
    }

    vmr_fields: dict[str, jnp.ndarray] = {
        "o3": _zc_fill_halo(_orient_zc(ozone_vmr)),
    }
    if co2_vmr is not None:
        vmr_fields["co2"] = _zc_fill_halo(
            _orient_zc(jnp.broadcast_to(co2_vmr, (nlev, nc)))
        )
    if ch4_vmr is not None:
        vmr_fields["ch4"] = _zc_fill_halo(
            _orient_zc(jnp.broadcast_to(ch4_vmr, (nlev, nc)))
        )
    if n2o_vmr is not None:
        vmr_fields["n2o"] = _zc_fill_halo(
            _orient_zc(jnp.broadcast_to(n2o_vmr, (nlev, nc)))
        )

    aerosol_optics_sw: Optional[dict[str, jnp.ndarray]] = None
    if aerosol_sw is not None:
        aerosol_optics_sw = {
            k: _orient_bzc(v) for k, v in aerosol_sw.items()
        }

    aerosol_optics_lw: Optional[dict[str, jnp.ndarray]] = None
    if aerosol_lw is not None:
        aerosol_optics_lw = {
            k: _orient_bzc(v) for k, v in aerosol_lw.items()
        }

    zenith_1d = _solar_zenith_angle(cos_zenith)
    sfc_alb_broadband = 0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir
    vis_dir = (
        sfc_alb_broadband if surface_albedo_vis_dir is None
        else surface_albedo_vis_dir
    )
    vis_dif = (
        sfc_alb_broadband if surface_albedo_vis_dif is None
        else surface_albedo_vis_dif
    )
    nir_dir = (
        sfc_alb_broadband if surface_albedo_nir_dir is None
        else surface_albedo_nir_dir
    )
    nir_dif = (
        sfc_alb_broadband if surface_albedo_nir_dif is None
        else surface_albedo_nir_dif
    )
    nir_mask = _sw_band_is_nir_mask(rrtmgp_instance)[:, jnp.newaxis]
    sfc_alb_dir = jnp.where(
        nir_mask, nir_dir[jnp.newaxis, :], vis_dir[jnp.newaxis, :],
    )
    sfc_alb_dif = jnp.where(
        nir_mask, nir_dif[jnp.newaxis, :], vis_dif[jnp.newaxis, :],
    )

    def _f32_leaves(tree):
        return jax.tree_util.tree_map(
            lambda x: (x.astype(jnp.float32)
                       if isinstance(x, jnp.ndarray)
                       and jnp.issubdtype(x.dtype, jnp.floating) else x),
            tree,
        )

    lib_in = _f32_leaves(dict(
        zenith=zenith_1d,
        irrad=direct_irradiance,
        sfc_alb=sfc_alb_broadband,
        sfc_emis=surface_emissivity,
        sfc_alb_dir=sfc_alb_dir,
        sfc_alb_dif=sfc_alb_dif,
        cloud_path_liq_lw_per_gpt=cpl_lw_zc,
        cloud_path_ice_lw_per_gpt=cpi_lw_zc,
        cloud_path_liq_sw_per_gpt=cpl_sw_zc,
        cloud_path_ice_sw_per_gpt=cpi_sw_zc,
        vmr_fields=vmr_fields,
        aerosol_optics_sw=aerosol_optics_sw,
        aerosol_optics_lw=aerosol_optics_lw,
        cloud_tau_scale_liq=zinhoml,
        cloud_tau_scale_ice=zinhomi,
        **rrtmgp_input,
    ))

    del gas_optics_mode, use_scan
    with jax.enable_x64(False):
        rrtmgp_output = rrtmgp_instance.compute_heating_rate_zc(
            **lib_in, clear_sky_fluxes=compute_cre, lw_solver=lw_solver,
            gpt_chunk=gpt_chunk, aerosol_free_fluxes=aerosol_free,
        )

    def _zc_flux_profiles(sky: str = ""):
        return tuple(
            _orient_zc(rrtmgp_output[key])
            for key in (
                f"sw_flux_up_{sky}full",
                f"sw_flux_down_{sky}full",
                f"lw_flux_up_{sky}full",
                f"lw_flux_down_{sky}full",
            )
        )

    total_heating = _orient_zc(rrtmgp_output["rad_heat_src"][halo:halo + nlev, :])
    lw_heating = _orient_zc(rrtmgp_output["rad_heat_lw_3d"][halo:halo + nlev, :])
    sw_heating = _orient_zc(rrtmgp_output["rad_heat_sw_3d"][halo:halo + nlev, :])

    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating,
    )

    surf_sw_down = rrtmgp_output["surf_sw_flux_down_2d_xy"]
    surf_sw_up = rrtmgp_output["surf_sw_flux_up_2d_xy"]
    surf_lw_down = rrtmgp_output["surf_lw_flux_down_2d_xy"]
    surf_lw_up = rrtmgp_output["surf_lw_flux_up_2d_xy"]
    toa_sw_down = rrtmgp_output["toa_sw_flux_incoming_2d_xy"]
    toa_sw_up = rrtmgp_output["toa_sw_flux_outgoing_2d_xy"]
    toa_lw_up = rrtmgp_output["toa_lw_flux_outgoing_2d_xy"]

    sw_flux_up, sw_flux_down, lw_flux_up, lw_flux_down = _zc_flux_profiles()

    if aerosol_free:
        toa_sw_up_noa = rrtmgp_output["toa_sw_flux_outgoing_aerosolfree_2d_xy"]
        toa_lw_up_noa = rrtmgp_output["toa_lw_flux_outgoing_aerosolfree_2d_xy"]
    else:
        toa_sw_up_noa = toa_lw_up_noa = jnp.zeros_like(toa_sw_up)

    if aerosol_free and compute_cre:
        toa_sw_up_clear_noa = rrtmgp_output[
            "toa_sw_flux_outgoing_aerosolfree_clearsky_2d_xy"
        ]
        toa_lw_up_clear_noa = rrtmgp_output[
            "toa_lw_flux_outgoing_aerosolfree_clearsky_2d_xy"
        ]
    else:
        toa_sw_up_clear_noa = toa_lw_up_clear_noa = jnp.zeros_like(toa_sw_up)

    if compute_cre:
        toa_sw_up_clear = rrtmgp_output["toa_sw_flux_outgoing_clearsky_2d_xy"]
        toa_lw_up_clear = rrtmgp_output["toa_lw_flux_outgoing_clearsky_2d_xy"]
        (sw_flux_up_clear, sw_flux_down_clear,
         lw_flux_up_clear, lw_flux_down_clear) = _zc_flux_profiles(
            sky="clearsky_",
        )
    else:
        toa_sw_up_clear = jnp.zeros_like(toa_sw_up)
        toa_lw_up_clear = jnp.zeros_like(toa_lw_up)
        sw_flux_up_clear = jnp.zeros_like(sw_flux_up)
        sw_flux_down_clear = jnp.zeros_like(sw_flux_down)
        lw_flux_up_clear = jnp.zeros_like(lw_flux_up)
        lw_flux_down_clear = jnp.zeros_like(lw_flux_down)

    broadband_for_pub = 0.46 * surface_albedo_vis + 0.54 * surface_albedo_nir
    pub_vis_dir = (
        broadband_for_pub if surface_albedo_vis_dir is None
        else surface_albedo_vis_dir
    )
    pub_vis_dif = (
        broadband_for_pub if surface_albedo_vis_dif is None
        else surface_albedo_vis_dif
    )
    pub_nir_dir = (
        broadband_for_pub if surface_albedo_nir_dir is None
        else surface_albedo_nir_dir
    )
    pub_nir_dif = (
        broadband_for_pub if surface_albedo_nir_dif is None
        else surface_albedo_nir_dif
    )

    zero_col = jnp.zeros_like(toa_sw_up)
    diagnostics = RadiationData(
        cos_zenith=cos_zenith,
        surface_albedo_vis=surface_albedo_vis,
        surface_albedo_nir=surface_albedo_nir,
        surface_emissivity=surface_emissivity,
        surface_albedo_vis_dir=pub_vis_dir,
        surface_albedo_vis_dif=pub_vis_dif,
        surface_albedo_nir_dir=pub_nir_dir,
        surface_albedo_nir_dif=pub_nir_dif,
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
        lw_heating_rate=lw_heating,
        sw_flux_up_clear=sw_flux_up_clear,
        sw_flux_down_clear=sw_flux_down_clear,
        lw_flux_up_clear=lw_flux_up_clear,
        lw_flux_down_clear=lw_flux_down_clear,
        surface_sw_down=surf_sw_down,
        surface_lw_down=surf_lw_down,
        surface_sw_up=surf_sw_up,
        surface_lw_up=surf_lw_up,
        toa_sw_up=toa_sw_up,
        toa_lw_up=toa_lw_up,
        toa_sw_down=toa_sw_down,
        toa_sw_up_clear=toa_sw_up_clear,
        toa_lw_up_clear=toa_lw_up_clear,
        toa_sw_up_noa=toa_sw_up_noa,
        toa_lw_up_noa=toa_lw_up_noa,
        toa_sw_up_clear_noa=toa_sw_up_clear_noa,
        toa_lw_up_clear_noa=toa_lw_up_clear_noa,
        noa_frac_toa_sw_up=zero_col,
        noa_frac_toa_lw_up=zero_col,
        noa_frac_toa_sw_up_clear=zero_col,
        noa_frac_toa_lw_up_clear=zero_col,
        total_cloud_cover=total_cloud_cover,
        total_cloud_cover_tau=total_cloud_cover_tau,
        r_eff_liq_at_solve=r_eff_liq_um,
        r_eff_ice_at_solve=r_eff_ice_um,
        step=jnp.zeros((nc,), dtype=jnp.int32),
    )
    return tendencies, diagnostics


def _maybe_chunked_col_minor_solve(zc_fn, in_axes):
    """Run column-minor ``zc_fn`` over all columns or 128-aligned column blocks.

    Splits mapped operands along their trailing ``ncols`` axis while keeping the
    per-chunk column block ``size = ncols // n_chunks`` as the contiguous minor
    axis ``-1``.
    """
    import os

    env_val = os.environ.get("JCM_RRTMGP_COL_CHUNKS", "").strip()
    explicit_chunks = int(env_val) if env_val else None
    is_tpu = jax.default_backend() == "tpu"
    if explicit_chunks is not None and explicit_chunks <= 1:
        return zc_fn
    if explicit_chunks is None and not is_tpu:
        return zc_fn

    def run(*args):
        first = next(
            a for a, ax in zip(args, in_axes)
            if ax is not None and jax.tree_util.tree_leaves(a)
        )
        ncols = jax.tree_util.tree_leaves(first)[0].shape[-1]
        n_chunks = (
            explicit_chunks
            if explicit_chunks is not None
            else _default_tpu_col_chunks(ncols)
        )
        if n_chunks <= 1:
            return zc_fn(*args)
        if ncols % n_chunks:
            raise ValueError(
                f"{ncols} columns is not divisible by {n_chunks} chunks"
            )
        size = ncols // n_chunks

        def split(x):
            return jnp.moveaxis(
                x.reshape(*x.shape[:-1], n_chunks, size), -2, 0,
            )

        mapped = [
            jax.tree_util.tree_map(split, a)
            for a, ax in zip(args, in_axes)
            if ax is not None
        ]
        static = [a for a, ax in zip(args, in_axes) if ax is None]

        def body(block):
            block = list(block)
            rebuilt, statics = [], list(static)
            for ax in in_axes:
                rebuilt.append(
                    block.pop(0) if ax is not None else statics.pop(0)
                )
            return zc_fn(*rebuilt)

        tend, diag = lax.map(jax.checkpoint(body), tuple(mapped))

        def merge(y):
            return jnp.moveaxis(y, 0, -2).reshape(*y.shape[1:-1], ncols)

        return jax.tree_util.tree_map(merge, (tend, diag))

    return run


def _default_tpu_col_chunks(ncols: int) -> int:
    """Choose a 128-lane-aligned column block count for TPU grid solves.

    Splitting large GCM grids (such as T63's ``18,432 = 8 * 18 * 128`` columns)
    into ``DEFAULT_TPU_COL_CHUNKS`` blocks bounds RRTMGP's ``all_gpts`` scratch
    memory so multi-week simulations fit comfortably within single-chip 16 GiB
    TPU v5e HBM while preserving ``ny = 128`` physical-tile alignment.
    """
    for k in (DEFAULT_TPU_COL_CHUNKS, 4, 2):
        if ncols >= k * 256 and ncols % k == 0 and (ncols // k) % 128 == 0:
            return k
    return 1


def _maybe_chunked_grid_solve(grid_fn, in_axes):
    """Run ``grid_fn`` over all columns or in rematerialized column blocks.

    Honors ``JCM_RRTMGP_COL_CHUNKS`` when set explicitly, and defaults on TPU to
    :func:`_default_tpu_col_chunks` so both forward trajectories and
    ``jax.grad`` calibration keep dense ``(nx, 128, nzh)`` tiles within HBM.
    """
    import os

    env_val = os.environ.get("JCM_RRTMGP_COL_CHUNKS", "").strip()
    explicit_chunks = int(env_val) if env_val else None
    is_tpu = jax.default_backend() == "tpu"
    if explicit_chunks is not None and explicit_chunks <= 1:
        return grid_fn
    if explicit_chunks is None and not is_tpu:
        return grid_fn

    def run(*args):
        first = next(a for a, ax in zip(args, in_axes) if ax == 0)
        ncols = jax.tree_util.tree_leaves(first)[0].shape[0]
        n_chunks = (
            explicit_chunks
            if explicit_chunks is not None
            else _default_tpu_col_chunks(ncols)
        )
        if n_chunks <= 1:
            return grid_fn(*args)
        if ncols % n_chunks:
            raise ValueError(
                f"{ncols} columns is not divisible by {n_chunks} chunks"
            )
        size = ncols // n_chunks

        split = lambda x: x.reshape(n_chunks, size, *x.shape[1:])  # noqa: E731
        mapped = [
            jax.tree_util.tree_map(split, a)
            for a, ax in zip(args, in_axes)
            if ax == 0
        ]
        static = [a for a, ax in zip(args, in_axes) if ax is None]

        def body(block):
            block = list(block)
            rebuilt, statics = [], list(static)
            for ax in in_axes:
                rebuilt.append(block.pop(0) if ax == 0 else statics.pop(0))
            return grid_fn(*rebuilt)

        tend, diag = lax.map(jax.checkpoint(body), tuple(mapped))
        merge = lambda x: x.reshape(ncols, *x.shape[2:])  # noqa: E731
        return jax.tree_util.tree_map(merge, (tend, diag))

    return run


# The *noa spacing helpers live in ``aerosol_free`` so echam_physics() can
# validate the setting for emulated and custom-term configs too, without importing
# this module and its RRTMGP tables.
from jcm.physics.radiation.aerosol_free import (  # noqa: E402
    NOA_KEYS,
    check_cadence_precesses,
    hold_all,
    resolve_aerosol_free_interval,
    update_effect_fraction,
)

__all__ = ["RRTMGPRadiation", "LW_SOLVERS", "GAS_OPTICS_MODES"]

# The library's longwave transports (``rrtmgp.rte.two_stream.LW_SOLVERS``),
# listed here so a misspelt name fails at term construction rather than at
# the first trace.
LW_SOLVERS = ("two_stream", "noscat")


class RRTMGPRadiation(PhysicsTerm):
    """RRTMGP full-spectrum radiation as a composable PhysicsTerm.

    Solves all columns on a dense 3-D ``(nx, ny, nlev + 2)`` domain via
    :func:`_radiation_grid_rrtmgp` (choosing ``ny = 128`` whenever ``ncols`` is
    a multiple of 128 so TPU physical tiles are completely filled without
    ``(1, 1)`` spatial padding). On GPU and CPU the forward solve runs across
    all columns in one call; on TPU :func:`_maybe_chunked_grid_solve` defaults
    to :func:`_default_tpu_col_chunks` 128-lane-aligned column blocks so
    multi-week simulations fit within single-chip 16 GiB HBM.

    The **reverse pass** holds per-g-point profiles of every active column live
    at once, so its peak memory grows linearly with the block size (measured
    with ``jax.grad`` over a 2-step rollout: ~19 GiB at T21L47, ~77 GiB at
    T31L47, ~171 GiB at T63L47), and XLA's own rematerialization pass cannot
    reduce it. For gradient-based calibration on GPU at T63L47 set the
    environment variable ``JCM_RRTMGP_COL_CHUNKS`` (for example to ``8``: the
    term's backward then peaks at 15 GiB and takes 1.2 s per call with the
    default placement, 26 GiB and 7.5 s with ``per_gpt``); see
    :func:`_maybe_chunked_grid_solve`.

    Reads pressure / height / density from the moist-air diagnostics
    dict, cloud fraction from ``diagnostics["clouds"]`` and
    pre-cloud-step cloud water / ice from state tracers, ozone / CO2 from
    ``"chemistry"``, aerosol from ``"aerosol"``, surface temperature
    from the legacy ``"surface"`` key, and surface albedos /
    emissivity from the public ``"radiation"`` key. Caches its own
    heating rates across radiation sub-steps via the previous step's
    ``RadiationData`` in ``diagnostics["radiation"]``.

    The cloud effective radii are formed inside the term from the current
    step's state (``radiation_effective_radii``: step-start condensate, the
    2M number tracers or ECHAM's prescribed droplet profile, the land mask),
    as ECHAM's radiation forms them inside ``cloud_optics``. The term
    publishes them as the diagnostic ``clouds.r_eff_liq`` /
    ``clouds.r_eff_ice`` (um, 0 where a phase is absent), which it never
    reads back -- UNLESS constructed with ``use_host_effective_radii=True``
    (jax-gcm#1135), in which case the term reads them from the host's
    ``clouds.r_eff_liq`` / ``clouds.r_eff_ice`` instead of forming them
    itself; see :func:`~jcm.physics.radiation.cloud_optics.host_effective_radii`
    and the constructor docstring.
    """

    name: ClassVar[str] = "rrtmgp_radiation"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "layer_thickness",
        "air_density", "chemistry", "aerosol",
        "radiation", "surface", "clouds",
    )
    provides: ClassVar[tuple[str, ...]] = ("radiation", "clouds")
    # CF/units metadata for the ``radiation.*`` output fields (#740).
    output_attrs: ClassVar[dict[str, dict[str, str]]] = RADIATION_OUTPUT_ATTRS

    def __init__(
        self,
        params: RadiationParameters | None = None,
        base_seed: int = 0,
        compute_cre: bool = True,
        aerosol_free_interval: int | None = None,
        use_host_effective_radii: bool = False,
        *,
        params_are_defaults: bool = False,
        lw_solver: str = "two_stream",
        gas_optics_mode: str = DEFAULT_GAS_OPTICS_MODE,
        gpt_chunk: int = DEFAULT_GPT_CHUNK,
        use_scan: bool = DEFAULT_USE_SCAN,
    ):
        """Hold the scheme-native :class:`RadiationParameters`.

        Args:
            params: scheme-native ``RadiationParameters``.
            base_seed: McICA PRNG base seed. The generator folds this
                with ``model_step`` and the per-column index, so the
                same ``base_seed`` always produces the same stochastic
                sub-columns for a given run — bit-exact reruns.
            compute_cre: if True (default), also take the clear-sky
                fluxes from the RRTMGP solve (the library solves them inside
                the all-sky g-point loop, on the shared gas optics) and
                populate ``toa_{sw,lw}_up_clear`` for the cloud radiative
                effect diagnostic. Set False to drop that transport when CRE
                isn't needed.
            aerosol_free_interval: radiation steps between aerosol-free
                companion solves, producing the AeroCom ``*noa`` fluxes
                that ERFari is diagnosed from (jax-gcm#583). ``None``
                (default) means no ``*noa`` fluxes at all; ``1`` is the
                exact reference: the twins are solved inside the one
                radiation call on the shared gas optics, +48 % per
                call at T63L47 on an A100 (``aerosol_free`` in
                ``radiation_scheme_rrtmgp``); ``N > 1`` runs the companion every Nth
                step and holds the aerosol effect in between — a monotonic
                dial, cheaper and less accurate as N grows. The SIMULATION
                is unchanged to float32 rounding at every N: the fluxes
                with aerosol come out of a program that also solves the
                twins, which XLA fuses differently, so they differ from a
                run without the diagnostic by rounding (3e-4 W/m2 on a
                shortwave flux of ~500, 1e-7 K/s in the heating, one CPU
                column), and with N > 1 the companion calls run that
                program while the others run the plain one. It was
                bit-identical when the companion was a second solve; a
                fence does not restore that (XLA's CPU backend drops
                optimization barriers before fusion), so the diagnostic is
                a rounding-level perturbation of the trajectory, as any
                other change of the compiled program is. Only the
                diagnostic is approximated by N. N > 1 logs a warning quoting the measured
                error, since the choice is not recoverable from the output
                files afterwards.
            use_host_effective_radii: if True, read the droplet / crystal
                effective radii from the host's ``diagnostics["clouds"]
                .r_eff_liq`` / ``.r_eff_ice`` (um) instead of forming them
                from the current state's ``qnc``/``qni`` or ECHAM's
                prescribed profile (jax-gcm#1135, for jax-les: its liquid
                radius follows jcm's own
                ``cloud_utils.eff_liquid_droplet_radius`` +
                ``breadth_factor`` law from its prescribed/prognostic
                droplet number, and it has no ice-number tracer to drive
                the ``qnc``/``qni`` path at all). Default False, matching
                every prior release's behaviour exactly (bit-identical;
                see ``rrtmgp_test.py::TestRRTMGPHostEffectiveRadii
                .test_flag_off_matches_default_behaviour``).

                This is a CONSTRUCTOR flag, not a ``RadiationParameters``
                field, deliberately: ``RadiationParameters`` is a
                ``tree_math.struct``, which (unlike ``flax.struct
                .dataclass``) has no ``pytree_node=False`` mechanism, so a
                boolean held there would be a genuine pytree leaf that
                ``jax.jit`` traces into a 0-d array -- exactly the kind of
                value a Python ``if`` cannot branch on, defeating the point
                of a trace-time code-path selector (see
                ``RadiationParameters.defaults_truncation``'s docstring for
                the same constraint). A plain constructor attribute (this
                class's existing pattern for ``compute_cre`` and
                ``aerosol_free_interval``) IS genuinely static: it never
                becomes a pytree leaf, so ``_compute_full`` branches on it
                with an ordinary ``if`` with no retrace per call.

                An explicit flag was chosen over silently preferring
                ``clouds.r_eff_liq``/``r_eff_ice`` whenever they happen to be
                non-default: every jcm composition that writes them (RRTMGP
                itself, on the very step before this one) would then feed
                its own last-published radius back into itself the next
                solve, silently changing every existing run's results the
                moment it is composed -- the issue names both forms and
                rejects the silent one for exactly this reason.

                Fails loudly (``ValueError``, from
                :func:`~jcm.physics.radiation.cloud_optics.host_effective_radii`)
                if the host's ``clouds`` diagnostic lacks ``r_eff_liq``/
                ``r_eff_ice`` entirely, naming the actual remedy (construct
                with ``use_host_effective_radii=False``, or publish the
                fields) rather than a bare attribute-access traceback. Since
                the flag is a static, trace-time selector, this check runs
                on every ``__call__`` regardless of whether that particular
                step ends up solving or replaying a cached step
                (``jax.lax.cond`` traces both branches to build the
                conditional, and this branch is part of the trace whenever
                the flag is on) -- it cannot be silently skipped by landing
                on a cached step. It cannot detect a host that supplies the
                attributes but never actually fills them: the unfilled
                ``CloudData.zeros()`` default reads as the table's SMALLEST
                radius wherever condensate is present (the clamp raises any
                value at or below the lower bound up to it, 0 included), the
                most optically potent size, not an inert "cloud-free" one --
                a documented limit of the structural check, not an oversight
                (see :func:`~jcm.physics.radiation.cloud_optics.host_effective_radii`'s
                docstring).

                The radius is clamped to jax-rrtmgp's table range and
                zeroed where this term's own in-cloud condensate test finds
                no condensate, exactly the clamp
                :func:`~jcm.physics.radiation.cloud_optics.echam_cloud_effective_radii`
                applies to its own radius -- so the two paths are
                interchangeable from the RRTMGP solve's point of view. The
                ICE radius law is entirely the host's choice: with no
                ice-number tracer there is no ECHAM counterpart for jcm to
                defer to, so this flag does not second-guess it beyond that
                shared clamp.
            params_are_defaults: ``True`` when ``params`` are resolution
                defaults built by ``echam_physics()`` (with or without field
                overrides) rather than an object the caller supplied
                directly. Only then does :meth:`cache_coords` warn if they
                were built for another grid's truncation (#974), the same
                contract as ``SundqvistCloudFraction``.
            lw_solver: the longwave transport of the library solve.
                ``"two_stream"`` (default) is jax-rrtmgp's two-stream solver,
                which scatters cloud and aerosol. ``"noscat"`` is RTE's
                absorption-only solver (``lw_solver_noscat_oneangle``: one
                angle, secant 1/0.6096748751; cloud and aerosol enter through
                ``tau (1 - ssa)``), which is what ECHAM6's
                ``mo_lrtm_solver.f90`` does. It also solves the clear-sky
                (CRE) fluxes. The shortwave is unaffected. Selectable, not the
                default; see ``docs/source/science/radiation.md``.
            gas_optics_mode: where jax-rrtmgp evaluates the gas optics:
                ``"all_gpts"`` (default) for every g-point in one pass ahead of
                its spectral loop, ``"per_gpt"`` inside the loop. Changes no
                physics (the fluxes agree to a few float32 ulp, from XLA
                fusion). Default and ``gpt_chunk`` are the measured A100
                optimum; see the comment above :data:`GAS_OPTICS_MODES`.
            gpt_chunk: g-points the library solves per spectral-loop iteration
                (default 64, with ``use_scan``; 1 is the library's own default
                and the smallest memory footprint, 16 the largest chunk that
                adds no memory). Changes no physics.
            use_scan: run the vertical recurrences as a ``lax.scan`` over
                levels (default True) instead of an unrolled chain. Together
                with ``gpt_chunk=64`` and ``gas_optics_mode="all_gpts"`` this is
                the measured A100 optimum, at +4 GiB (T63L47) / +8 GiB (T63L95)
                of peak memory over the unrolled form; the fluxes differ by at
                most 12 float32 ulp. On TPU, ``all_gpts`` contracts factored
                one-hot weights on the MXU with ``Precision.HIGHEST`` and
                defaults ``gpt_chunk`` to :data:`DEFAULT_TPU_GPT_CHUNK` when
                left at :data:`DEFAULT_GPT_CHUNK`.

        """
        self._params_user_supplied = (params is not None
                                      and not params_are_defaults)
        self.params = nnx.Param(params or RadiationParameters.default())
        # Plain Python attributes — these are static-at-trace-time so
        # the McICA seeding and the CRE branch fold into ``__call__``
        # without an extra pytree leaf.
        self._base_seed = int(base_seed)
        self._compute_cre = bool(compute_cre)
        self._use_host_effective_radii = bool(use_host_effective_radii)
        if lw_solver not in LW_SOLVERS:
            raise ValueError(
                f"lw_solver must be one of {LW_SOLVERS}, got {lw_solver!r}.")
        # Static at trace time: it selects which transport program is built.
        self._lw_solver = lw_solver
        # Static at trace time, like ``lw_solver``: both select the program.
        self._gas_optics_mode = gas_optics_mode
        self._gpt_chunk, self._use_scan = _check_library_options(
            gas_optics_mode, gpt_chunk, use_scan)
        self._gpt_chunk_is_default = (self._gpt_chunk == DEFAULT_GPT_CHUNK)
        # AeroCom *noa fluxes (jax-gcm#583): the aerosol-free twin of each
        # problem, solved inside the one radiation call on the shared gas
        # optics (``radiation_scheme_rrtmgp(aerosol_free=True)``). What is
        # repeated per g-point is the cell properties and the transport, so
        # the companion costs a fraction of a solve rather than a second one;
        # still a cost, so opt-in only.
        #
        # Validated through the shared helper, which echam_physics() also
        # calls so any other radiation scheme is held to the same contract
        # rather than silently ignoring the argument.
        interval = resolve_aerosol_free_interval(aerosol_free_interval)
        # A cadence locked to the solar day never samples some columns in
        # daylight; see check_cadence_precesses.
        check_cadence_precesses(
            interval, (params or RadiationParameters.default())
            .radiation_interval)
        self._aerosol_free = interval is not None
        # Between companions the aerosol EFFECT is held, not the raw
        # aerosol-free flux: the effect varies slowly, the flux does not,
        # and holding the flux would leave rsut and rsutnoa sampling
        # different step sets (measured at 0.04-0.14 W/m2 for a 2x
        # subsample — the same size as the error this is meant to avoid).
        # The twins' transports are paid on every Nth call instead of
        # every call.
        self._aerosol_free_interval = interval or 1
        self._coords_cached = False
        # Eagerly create the global RRTMGP instance now (loads netCDF
        # gas-optics + cloud-optics tables). Otherwise the first jit
        # trace of ``__call__`` triggers ``optics_factory`` from inside
        # the traced scope, and the lookup-table jnp arrays created by
        # ``load_data`` leak as ``UnexpectedTracerError`` — fatal for
        # ``output_averages=True`` runs in particular. Doing it here
        # forces a single non-traced load at term-construction time.
        _ensure_rrtmgp()

    def legacy_surface_albedo_quartet(self, vis, nir):
        """``(vis, nir)`` -> the ``(vis_dir, vis_dif, nir_dir, nir_dif)`` this
        backend would have published for them before the quartet existed
        (jax-gcm#959's checkpoint migration, ``jcm.checkpoint._DERIVED_ON_MIGRATION``).

        RRTMGP's pre-#959 surface boundary condition was a single BLENDED
        broadband scalar (``0.46*vis + 0.54*nir``, ``sfc_alb_broadband`` in
        :func:`radiation_scheme_rrtmgp`) applied to every band, both beams
        -- a fresh solve's own ``RadiationData`` still publishes exactly
        this blend in all four quartet slots whenever the per-band
        direct/diffuse inputs are omitted (see that function's ``None``
        fallback). A migrated pre-#959 carry must match, not grey two-
        stream's raw-pair convention (Codex review on PR #1125).
        """
        blend = 0.46 * vis + 0.54 * nir
        return blend, blend, blend, blend

    def legacy_effective_radii_at_solve(self, r_eff_liq, r_eff_ice):
        """``(r_eff_liq, r_eff_ice)`` -> the ``(r_eff_liq_at_solve,
        r_eff_ice_at_solve)`` this backend would have held before the
        dedicated carry existed (jax-gcm#1135's checkpoint migration,
        ``jcm.checkpoint._DERIVED_ON_MIGRATION``).

        Before this PR added the dedicated carry, RRTMGP's ``__call__``
        already published the solve's own radius onto
        ``clouds.r_eff_liq``/``r_eff_ice`` every step (the
        ``clouds_in.copy(...)`` below) and no host-radii door existed to
        write anything else there, so the two were always identical -- the
        checkpoint migration this method serves simply carries that
        identity forward rather than guessing a formula.
        """
        return r_eff_liq, r_eff_ice

    def withheld_output_keys(self) -> tuple[str, ...]:
        """Hide the ``*noa`` and clear-sky slots when no companion solve runs.

        ``RadiationData`` carries the aerosol-free and clear-sky slots in
        every configuration. With a diagnostic off they stay at their
        zero default, and publishing that turns a downstream ERFari
        (``rsut - rsutnoa``) into the entire all-sky flux — ~240 W/m2
        rather than ~-1 — in a file that otherwise looks valid
        (jax-gcm#647). Absent is honest; present-and-zero is not.

        The same trap applies to the clear-sky fluxes without
        ``compute_cre``: a zero clear-sky flux reads as a CRE equal to
        the whole all-sky flux, and as an all-zero training label for
        anything fitted against these fields.
        """
        withheld = () if self._aerosol_free else tuple(
            f"radiation.{k}_noa" for k in NOA_KEYS
        ) + tuple(
            # The persisted ratios are placeholders too without the
            # companion solve; a published 0 reads as "aerosol removes
            # the entire flux" through the fraction-based ERFari path.
            f"radiation.noa_frac_{k}" for k in NOA_KEYS
        )
        if not self._compute_cre:
            withheld += tuple(f"radiation.{k}" for k in CLEAR_SKY_KEYS)
            if self._aerosol_free:
                # The clear-sky twins exist only with ``compute_cre``: with
                # the companion on but the clear sky off, those two slots
                # and their carried fractions are the same zero
                # placeholders, and a published zero rsutcsnoa reads as a
                # clear-sky ERFari equal to the whole clear-sky flux.
                withheld += tuple(
                    f"radiation.{k}_noa" for k in NOA_KEYS if "clear" in k
                ) + tuple(
                    f"radiation.noa_frac_{k}" for k in NOA_KEYS if "clear" in k
                )
            # __call__ mirrors the clear-sky TOA fluxes onto the clouds
            # sub-struct for CRE diagnostics; without the companion solve
            # those mirrors are the same zero placeholders (the all-sky
            # mirrors stay — they are real).
            withheld += ("clouds.toa_sw_up_clear", "clouds.toa_lw_up_clear")
        return withheld

    def cache_coords(self, coords) -> None:
        """Cache per-column lat/lon (deg); check the parameters' grid.

        The grid check (#974) mirrors ``SundqvistCloudFraction.cache_coords``:
        only warns, and only for resolution-defaulted parameters (never one
        the caller supplied directly), since the parameter pytree was fixed
        at construction and is not re-resolved here.
        """
        lat, lon = column_lat_lon(coords.horizontal)
        self._lats = nnx.Variable(lat * 180.0 / jnp.pi)
        self._lons = nnx.Variable(lon * 180.0 / jnp.pi)
        self._coords_cached = True
        if not self._params_user_supplied:
            check_defaults_grid(
                type(self).__name__,
                self.params.get_value().defaults_truncation, coords)

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
        clouds_in = diagnostics["clouds"]
        # The published radii are a DIAGNOSTIC of the radius this term
        # radiated with, never an input to it: a solve forms its radii from
        # the current state (``radiation_effective_radii``, or the host's
        # own law under ``use_host_effective_radii=True``), and a cached
        # step reports the radii of the solve its heating came from.
        #
        # That "last solve" value is read from ``radiation.r_eff_liq_at_solve``
        # / ``r_eff_ice_at_solve`` (the radiation term's OWN carry,
        # ``RadiationData``), never from ``clouds_in.r_eff_liq``/``r_eff_ice``:
        # in host mode a host writes a FRESH, un-clamped radius to the
        # ``clouds`` diagnostic every step as ITS OWN input channel,
        # regardless of whether this term is solving this step, so that
        # field no longer reliably holds "the last solve's own output"
        # (jax-gcm#1135 review finding). In the default (non-host) path
        # these two sources always agree -- nothing else ever writes
        # ``clouds.r_eff_liq``/``r_eff_ice`` -- so reading the dedicated
        # carry instead changes nothing there; it only fixes host mode.
        radii_at_last_solve = (radiation.r_eff_liq_at_solve,
                               radiation.r_eff_ice_at_solve)

        def _compute():
            _tend, rad, radii = self._compute_full(
                state, diagnostics, forcing, terrain, params)
            # Pin the compute branch to the carry's leaf dtypes: under
            # jax_enable_x64 (e.g. driving this scheme from a float64
            # dycore with float32 physics state) some strong table
            # constants promote a subset of the freshly-computed leaves
            # to float64, and the two lax.cond branches would fail to
            # type-check against the uniform-dtype cached carry.
            rad = jax.tree.map(lambda n, o: n.astype(o.dtype), rad, radiation)
            radii = jax.tree.map(
                lambda n, o: n.reshape(o.shape).astype(o.dtype),
                radii, radii_at_last_solve)
            return rad, radii

        def _use_cached():
            return radiation, radii_at_last_solve

        solved = radiation_should_compute(diagnostics, params)
        new_radiation, (r_eff_liq, r_eff_ice) = jax.lax.cond(
            solved, _compute, _use_cached,
        )
        # The shortwave of EVERY step, solve or replay, is the carried
        # transmissivity times the sun of this step (ECHAM ``zi0 * trsol``),
        # and the heating the temperature tendency applies is rebuilt from it
        # the same way -- one contract, no per-branch tendency.
        new_radiation = apply_insolation(
            new_radiation, forcing.solar, self._lons.get_value(),
            self._lats.get_value(), params.solar_constant)
        tendency = cached_radiation_tendency(
            new_radiation, state.temperature.shape)
        tendency = jax.tree.map(
            lambda t: t.astype(state.temperature.dtype), tendency)
        # Advance the radiation-local step counter on every call (both
        # compute and cached paths). The McICA seed in ``_compute_full``
        # folds this counter in for per-(step, column) reproducibility,
        # so the increment must happen even on cached steps to keep the
        # sub-column sequence aligned with the chosen sub-stepping cadence.
        new_radiation = new_radiation.copy(step=radiation.step + 1)
        # Mirror the all-sky and clear-sky TOA fluxes onto the
        # ``"clouds"`` sub-struct for cloud-radiative-effect diagnostics.
        clouds = clouds_in.copy(
            toa_sw_up_all=new_radiation.toa_sw_up,
            toa_sw_up_clear=new_radiation.toa_sw_up_clear,
            toa_lw_up_all=new_radiation.toa_lw_up,
            toa_lw_up_clear=new_radiation.toa_lw_up_clear,
            r_eff_liq=r_eff_liq,
            r_eff_ice=r_eff_ice,
        )
        return tendency, hold_land_albedo({
            **diagnostics, "radiation": new_radiation, "clouds": clouds,
        }, solved)

    def _compute_full(
        self, state, diagnostics, forcing, terrain, params,
    ):
        """Run the full RRTMGP scheme.

        Returns ``(tendency, RadiationData, (r_eff_liq_um, r_eff_ice_um))``,
        the radii ``(nlev, ncols)`` being the ones the solve used.
        """
        nlev, ncols = state.temperature.shape

        latitudes = self._lats.get_value()
        longitudes = self._lons.get_value()
        # The solve is made at the middle of the radiation interval, as ECHAM
        # does (``solve_solar_geometry``); every step's own sun is re-applied
        # to the transmissivities it yields (``apply_insolation``).
        solar = solve_solar_geometry(
            forcing.solar, diagnostics["_dt_seconds"], params)
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
        # The cover BEFORE ``radiation_cloud_fields``'s condensate mask, for
        # the mask's derivative only (#973) -- the SAME diagnosed cover
        # ``radiation_cloud_fields`` itself reads to build the masked
        # ``cloud_fraction`` above.
        cloud_fraction_raw = diagnostics["clouds"].cloud_fraction
        # Droplet and crystal effective radii from THIS step's state, formed
        # the way ECHAM's radiation forms them inside its own call
        # (mo_cloud_optics.f90::cloud_optics); see radiation_effective_radii
        # for the inputs and their time levels. ``self._use_host_effective_radii``
        # is a plain constructor attribute (never a pytree leaf, see its
        # docstring), so this ``if`` is a genuine trace-time branch, not a
        # per-call dispatch: a host (jax-les, jax-gcm#1135) supplies its own
        # radii on ``diagnostics["clouds"]`` instead of letting jcm form them
        # from qnc/qni or the prescribed profile.
        if self._use_host_effective_radii:
            r_eff_liq_um, r_eff_ice_um = host_effective_radii(
                diagnostics["clouds"],
                cloud_water, cloud_ice, cloud_fraction, params.cld_frac_min,
                cloud_fraction_raw=cloud_fraction_raw,
            )
        else:
            r_eff_liq_um, r_eff_ice_um = radiation_effective_radii(
                state, diagnostics, forcing, terrain,
                cloud_water, cloud_ice, cloud_fraction, params.cld_frac_min,
                cloud_fraction_raw=cloud_fraction_raw,
            )

        # Greenhouse-gas sourcing (all converted ppmv -> mole fraction here):
        #   O3, CH4  <- the chemistry diagnostic (O3 analytic/climatology; CH4
        #               forcing-seeded then evolved by the methane-loss scheme)
        #   CO2, N2O <- prescribed forcings read straight from ForcingData
        # Radiation never reads a GHG concentration from its own parameters, and
        # never lets a value be silently redeclared.
        chemistry = diagnostics["chemistry"]
        ozone_vmr, ch4_vmr, co2_vmr, n2o_vmr = (
            _greenhouse_gas_mole_fractions(chemistry, forcing)
        )

        surface_temperature_col = (
            diagnostics["surface"].surface_temperature.reshape(ncols)
        )
        (surface_albedo_vis_col, surface_albedo_nir_col,
         surface_emissivity_col) = surface_optics_for_solve(diagnostics, ncols)
        # Per-band direct/diffuse quartet (jax-gcm#959): falls back to the
        # broadband pair in both slots when the boundary-condition term
        # published no quartet (see ``surface_optics_bands_for_solve``).
        (surface_albedo_vis_dir_col, surface_albedo_vis_dif_col,
         surface_albedo_nir_dir_col,
         surface_albedo_nir_dif_col) = surface_optics_bands_for_solve(
            diagnostics, ncols)

        aerosol_in = diagnostics["aerosol"]
        n_bnd_sw = aerosol_in.aod_sw_per_band.shape[0]
        n_bnd_lw = aerosol_in.aod_lw_per_band.shape[0]

        base_seed = self._base_seed
        compute_cre = self._compute_cre
        effective_gpt_chunk = (
            DEFAULT_TPU_GPT_CHUNK
            if (self._gpt_chunk_is_default and jax.default_backend() == "tpu")
            else self._gpt_chunk
        )
        use_tpu_col_minor = (
            jax.default_backend() == "tpu"
            and getattr(two_stream, "_SUPPORTS_TPU_COL_MINOR_SOLVE", False)
            and self._gas_optics_mode == "all_gpts"
            and self._use_scan
            and effective_gpt_chunk > 1
        )

        if use_tpu_col_minor:
            aerosol_sw_zc = dict(
                optical_depth=aerosol_in.aod_sw_per_band.reshape(
                    n_bnd_sw, nlev, ncols,
                ),
                ssa=aerosol_in.ssa_sw_per_band.reshape(n_bnd_sw, nlev, ncols),
                asymmetry_factor=aerosol_in.asy_sw_per_band.reshape(
                    n_bnd_sw, nlev, ncols,
                ),
            )
            aerosol_lw_zc = (
                dict(
                    optical_depth=aerosol_in.aod_lw_per_band.reshape(
                        n_bnd_lw, nlev, ncols,
                    ),
                    ssa=aerosol_in.ssa_lw_per_band.reshape(
                        n_bnd_lw, nlev, ncols,
                    ),
                    asymmetry_factor=aerosol_in.asy_lw_per_band.reshape(
                        n_bnd_lw, nlev, ncols,
                    ),
                )
                if n_bnd_lw > 0
                else None
            )
            zc_scheme = functools.partial(
                _radiation_col_minor_rrtmgp,
                lw_solver=self._lw_solver,
                gas_optics_mode=self._gas_optics_mode,
                gpt_chunk=effective_gpt_chunk,
                use_scan=self._use_scan,
                solve_geometry=True,
            )
            _zc_in_axes = (
                -1, -1, -1, -1, -1,
                -1, -1, -1, -1,
                -1, -1, -1, -1,
                None, -1, -1,
                None, -1, -1,
                -1, None, None, None,
                -1, -1, -1, -1,
                -1, -1,
                -1,
                -1,
                -1, -1, -1, -1,
            )
            solve_args = (
                state.temperature, state.specific_humidity,
                diagnostics["pressure_full"], diagnostics["pressure_half"],
                diagnostics["layer_thickness"], diagnostics["air_density"],
                cloud_water, cloud_ice, cloud_fraction,
                surface_temperature_col, surface_albedo_vis_col,
                surface_albedo_nir_col, surface_emissivity_col,
                solar, latitudes, longitudes,
                params, aerosol_sw_zc, aerosol_lw_zc, column_indices,
                model_step, base_seed, compute_cre,
                ozone_vmr,
                jnp.broadcast_to(co2_vmr, (nlev, ncols)),
                jnp.broadcast_to(ch4_vmr, (nlev, ncols)),
                jnp.broadcast_to(n2o_vmr, (nlev, ncols)),
                r_eff_liq_um, r_eff_ice_um,
                lagged_convection_type(diagnostics, ncols),
                cloud_fraction_raw,
                surface_albedo_vis_dir_col, surface_albedo_vis_dif_col,
                surface_albedo_nir_dir_col, surface_albedo_nir_dif_col,
            )

            def _solve(aerosol_free: bool):
                return _maybe_chunked_col_minor_solve(
                    functools.partial(zc_scheme, aerosol_free=aerosol_free),
                    _zc_in_axes,
                )(*solve_args)
        else:
            # Per-SW-band fields are ``(n_bnd_sw, nlev, ncols)`` from MACv2-SP;
            # transpose to ``(ncols, n_bnd_sw, nlev)`` so the column axis is
            # leading (vmap-friendly).
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
                aod_sw_per_band=_per_band_to_col(
                    aerosol_in.aod_sw_per_band, n_bnd_sw,
                ),
                ssa_sw_per_band=_per_band_to_col(
                    aerosol_in.ssa_sw_per_band, n_bnd_sw,
                ),
                asy_sw_per_band=_per_band_to_col(
                    aerosol_in.asy_sw_per_band, n_bnd_sw,
                ),
                aod_lw_per_band=_per_band_to_col(
                    aerosol_in.aod_lw_per_band, n_bnd_lw,
                ),
                ssa_lw_per_band=_per_band_to_col(
                    aerosol_in.ssa_lw_per_band, n_bnd_lw,
                ),
                asy_lw_per_band=_per_band_to_col(
                    aerosol_in.asy_lw_per_band, n_bnd_lw,
                ),
            )

            # Column-leading inputs for the vmap. The jax-rrtmgp #8 fix keeps
            # the gas-optics tables shared operands under the column vmap, so
            # the per-column working set is tiny (~1.5 GB at T63L47) and no
            # chunking is needed — radiation now vmaps over columns like the
            # rest of the physics. ``aerosol_for_vmap`` is already
            # column-leading; add Nccn.
            def lev_to_col(a):
                """(nlev, ncols) → (ncols, nlev) for the leading column vmap."""
                return a.T

            aerosol_col = aerosol_for_vmap.copy(
                Nccn=aerosol_in.Nccn.reshape(ncols),
            )

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
                cloud_fraction_raw=lev_to_col(cloud_fraction_raw),
                surface_temperature=surface_temperature_col,
                surface_albedo_vis=surface_albedo_vis_col,
                surface_albedo_nir=surface_albedo_nir_col,
                surface_emissivity=surface_emissivity_col,
                surface_albedo_vis_dir=surface_albedo_vis_dir_col,
                surface_albedo_vis_dif=surface_albedo_vis_dif_col,
                surface_albedo_nir_dir=surface_albedo_nir_dir_col,
                surface_albedo_nir_dif=surface_albedo_nir_dif_col,
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
                # Previous step's convective type: selects the liquid
                # inhomogeneity factor (ECHAM radiation reads the lagged rtype).
                convection_type=lagged_convection_type(diagnostics, ncols),
            )

            grid_scheme = functools.partial(
                _radiation_grid_rrtmgp, lw_solver=self._lw_solver,
                gas_optics_mode=self._gas_optics_mode,
                gpt_chunk=effective_gpt_chunk, use_scan=self._use_scan,
                solve_geometry=True)
            _in_axes = (
                0, 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                None, 0, 0,
                None, 0,
                0, None, None, None,  # col_index, model_step, base_seed, cre
                0, 0, 0, 0,          # ozone_vmr, co2_vmr, ch4_vmr, n2o_vmr
                0, 0,                # r_eff_liq_um, r_eff_ice_um
                0,                   # convection_type
                0,                   # cloud_fraction_raw
                0, 0, 0, 0,          # per-band albedo dir/dif quartet (#959)
            )
            solve_args = (
                cols["temperature"], cols["specific_humidity"],
                cols["pressure_full"], cols["pressure_half"],
                cols["layer_thickness"], cols["air_density"],
                cols["cloud_water"], cols["cloud_ice"], cols["cloud_fraction"],
                cols["surface_temperature"], cols["surface_albedo_vis"],
                cols["surface_albedo_nir"], cols["surface_emissivity"],
                solar, cols["latitudes"], cols["longitudes"],
                params, cols["aerosol"], cols["column_indices"],
                model_step, base_seed, compute_cre,
                cols["ozone_vmr"], cols["co2_vmr"], cols["ch4_vmr"],
                cols["n2o_vmr"],
                cols["r_eff_liq_um"], cols["r_eff_ice_um"],
                cols["convection_type"],
                cols["cloud_fraction_raw"],
                cols["surface_albedo_vis_dir"], cols["surface_albedo_vis_dif"],
                cols["surface_albedo_nir_dir"], cols["surface_albedo_nir_dif"],
            )

            def _solve(aerosol_free: bool):
                """Run the 3-D grid solve, with or without the aerosol-free twins."""
                return _maybe_chunked_grid_solve(
                    functools.partial(grid_scheme, aerosol_free=aerosol_free),
                    _in_axes,
                )(*solve_args)

        # AeroCom *noa fluxes (jax-gcm#583) come out of the same solve as the
        # all-sky ones (``radiation_scheme_rrtmgp(aerosol_free=True)``: the
        # library solves the aerosol-free twins in the same g-point loop).
        # With a companion interval N > 1 the solve WITH the twins runs on
        # every Nth radiation call and the plain solve on the others, under a
        # ``lax.cond`` so only one runs per step; both programs are compiled.
        # Between companions the aerosol EFFECT is held as a fraction of the
        # all-sky flux (see below).
        if self._aerosol_free and self._aerosol_free_interval > 1:
            dt_s = diagnostics["_dt_seconds"]
            spc = jnp.where(
                params.radiation_interval > 0,
                jnp.int32(jnp.round(params.radiation_interval / dt_s)),
                jnp.int32(1),
            )
            rad_call = model_step // spc
            is_companion = jnp.mod(rad_call, self._aerosol_free_interval) == 0
            tendencies_vmapped, diagnostics_vmapped = jax.lax.cond(
                is_companion, lambda: _solve(True), lambda: _solve(False),
            )
        else:
            tendencies_vmapped, diagnostics_vmapped = _solve(self._aerosol_free)

        _fresh_toa = dict(
            toa_sw_up=_column_vector_rrtmgp(diagnostics_vmapped.toa_sw_up, ncols),
            toa_lw_up=_column_vector_rrtmgp(diagnostics_vmapped.toa_lw_up, ncols),
            toa_sw_up_clear=_column_vector_rrtmgp(
                diagnostics_vmapped.toa_sw_up_clear, ncols),
            toa_lw_up_clear=_column_vector_rrtmgp(
                diagnostics_vmapped.toa_lw_up_clear, ncols),
        )

        if self._aerosol_free:
            all_sky = _fresh_toa
            _KEYS = NOA_KEYS
            _FRAC_FIELDS = tuple(f"noa_frac_{k}" for k in _KEYS)
            # The twins of this call, from the solve above: the solved values
            # on a companion step, zeros on a held one.
            solved = tuple(
                _column_vector_rrtmgp(getattr(diagnostics_vmapped, f"{k}_noa"),
                                      ncols)
                for k in _KEYS)

            if self._aerosol_free_interval > 1:
                # Between companions carry the aerosol EFFECT forward and apply
                # it to the FRESH all-sky flux, so the pair always refers to
                # the same state and rsut/rsutnoa never sample different step
                # sets.
                fresh = tuple(_fresh_toa[k] for k in _KEYS)
                prev_frac = tuple(getattr(diagnostics["radiation"], f)
                                  for f in _FRAC_FIELDS)

                # The two branches must return identical dtypes, and mixing
                # the solve's flux dtype with the carried fraction's dtype
                # (which differ under jax_enable_x64: the solve returns the
                # fluxes in its own working precision, the fraction slot is
                # whatever the carry holds) promotes in one branch only. So
                # each *noa flux is pinned to the dtype of the fresh all-sky
                # flux it stands in for, and each fraction to the dtype of
                # its carried slot.
                def _companion():
                    """Take the solved twins; refresh the stored fraction."""
                    fracs = [
                        update_effect_fraction(_fresh_toa[k], noa_v,
                                               prev_frac[i]
                                               ).astype(prev_frac[i].dtype)
                        for i, (k, noa_v) in enumerate(zip(_KEYS, solved))
                    ]
                    vals = tuple(v.astype(f.dtype)
                                 for v, f in zip(solved, fresh))
                    return vals, tuple(fracs)

                def _held():
                    """Re-apply the stored fraction to the fresh all-sky flux.

                    The fraction passes through untouched, which is why it is
                    carried explicitly: deriving it back out of the flux slots
                    fails once those slots are zero, and a companion landing
                    on a dark column would then report no aerosol effect for
                    the rest of the interval — including after sunrise.
                    """
                    held = hold_all(fresh, prev_frac)
                    return (tuple(h.astype(f.dtype) for h, f in zip(held, fresh)),
                            prev_frac)

                noa_vals, new_frac = jax.lax.cond(
                    is_companion, _companion, _held,
                )
                frac_out = dict(zip(_FRAC_FIELDS, new_frac))
            else:
                noa_vals = solved
                # Only `paired` carries a fraction; every other mode
                # solves fresh, so leave the slots at zero.
                frac_out = {f: jnp.zeros((ncols,))
                            for f in (f"noa_frac_{k}" for k in NOA_KEYS)}

            noa = {f"{k}_noa": v for k, v in zip(_KEYS, noa_vals)}
            noa.update(frac_out)
        else:
            all_sky = _fresh_toa
            zero_col = jnp.zeros((ncols,))
            noa = dict(
                toa_sw_up_noa=zero_col, toa_lw_up_noa=zero_col,
                toa_sw_up_clear_noa=zero_col, toa_lw_up_clear_noa=zero_col,
                **{f"noa_frac_{k}": zero_col for k in NOA_KEYS},
            )

        # Per-gpoint flux profiles are summed over g-points inside the
        # vmapped per-column compute, so flux arrays are (ncols, nlev+1)
        # — only a transpose is needed (DO NOT use the grey path's
        # transpose+sum, the per-band axis is already gone).
        _prof = (lambda a: a) if use_tpu_col_minor else (lambda a: a.T)
        rad_out = RadiationData(
            **noa,
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
            surface_albedo_vis_dir=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_albedo_vis_dir, ncols,
            ),
            surface_albedo_vis_dif=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_albedo_vis_dif, ncols,
            ),
            surface_albedo_nir_dir=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_albedo_nir_dir, ncols,
            ),
            surface_albedo_nir_dif=_column_vector_rrtmgp(
                diagnostics_vmapped.surface_albedo_nir_dif, ncols,
            ),
            sw_flux_up=_prof(diagnostics_vmapped.sw_flux_up),
            sw_flux_down=_prof(diagnostics_vmapped.sw_flux_down),
            sw_heating_rate=_prof(tendencies_vmapped.shortwave_heating),
            lw_flux_up=_prof(diagnostics_vmapped.lw_flux_up),
            lw_flux_down=_prof(diagnostics_vmapped.lw_flux_down),
            lw_heating_rate=_prof(tendencies_vmapped.longwave_heating),
            sw_flux_up_clear=_prof(diagnostics_vmapped.sw_flux_up_clear),
            sw_flux_down_clear=_prof(diagnostics_vmapped.sw_flux_down_clear),
            lw_flux_up_clear=_prof(diagnostics_vmapped.lw_flux_up_clear),
            lw_flux_down_clear=_prof(diagnostics_vmapped.lw_flux_down_clear),
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
            **all_sky,
            toa_sw_down=_column_vector_rrtmgp(
                diagnostics_vmapped.toa_sw_down, ncols,
            ),
            total_cloud_cover=_column_vector_rrtmgp(
                diagnostics_vmapped.total_cloud_cover, ncols,
            ),
            total_cloud_cover_tau=_column_vector_rrtmgp(
                diagnostics_vmapped.total_cloud_cover_tau, ncols,
            ),
            # The radii THIS solve actually used (fresh from the state, or
            # the host, depending on ``use_host_effective_radii`` --
            # ``r_eff_liq_um``/``r_eff_ice_um`` are already the exact
            # values ``__call__`` publishes as ``clouds.r_eff_liq``/
            # ``r_eff_ice`` on this solve). Stored on the carry so a LATER
            # cached step can report them without re-reading
            # ``clouds.r_eff_liq``/``r_eff_ice`` -- a host may be writing
            # fresh, unclamped values there every step regardless of
            # whether this term is solving (jax-gcm#1135 review finding).
            r_eff_liq_at_solve=r_eff_liq_um,
            r_eff_ice_at_solve=r_eff_ice_um,
            # Placeholder — the enclosing ``__call__`` overwrites
            # ``step`` after the compute-vs-cache cond.
            step=jnp.int32(0),
        )

        tendency = PhysicsTendency(
            u_wind=jnp.zeros((nlev, ncols)),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=_prof(tendencies_vmapped.temperature_tendency),
            specific_humidity=jnp.zeros((nlev, ncols)),
            tracers={},
        )
        return tendency, sw_solve_to_transmissivity(rad_out), (
            r_eff_liq_um, r_eff_ice_um)
