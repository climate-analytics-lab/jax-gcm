from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

from jcm.rrtmgp.optics import constants
from jcm.rrtmgp.optics import dataloader_base

Array: TypeAlias = jax.Array

def _bytes_to_str(split_str):
	return str(split_str, 'utf-8').strip()


@dataclasses.dataclass(frozen=True)
class AbstractLookupGasOptics:
	"""Abstract class for loading and accessing tables of optical properties."""
	# Volume mixing ratio (vmr) array index for H2O.
	idx_h2o: int
	# Volume mixing ratio (vmr) array index for O3.
	idx_o3: int
	# Number of gases used in the lookup table.
	n_gases: int
	# Number of frequency bands.
	n_bnd: int
	# Number of `g-points`.
	n_gpt: int
	# Number of atmospheric layers (=2, lower and upper atmospheres).
	n_atmos_layers: int
	# Number of reference temperatures for absorption coefficient lookup table.
	n_t_ref: int
	# Number of reference pressures for absorption lookup table.
	n_p_ref: int
	# Number of reference binary mixing fractions, for absorption coefficient lookup table.
	n_mixing_fraction: int
	# Number of major absorbing gases.
	n_maj_absrb: int
	# Number of minor absorbing gases.
	n_minor_absrb: int
	# Number of minor absorbers in lower atmosphere.
	n_minor_absrb_lower: int
	# Number of minor absorbers in upper atmosphere.
	n_minor_absrb_upper: int
	# Number of minor contributors in the lower atmosphere.
	n_contrib_lower: int
	# Number of minor contributors in the upper atmosphere.
	n_contrib_upper: int
	# Reference pressure separating upper and lower atmosphere.
	p_ref_tropo: Array
	# Reference temperature.
	t_ref_absrb: Array
	# Reference pressure.
	p_ref_absrb: Array
	# Minimum pressure supported by RRTM lookup tables.
	p_ref_min: Array
	# Minimum temperature supported by RRTM lookup tables.
	temperature_ref_min: Array
	# Maximum temperature supported by RRTM lookup tables.
	temperature_ref_max: Array
	# Δt for reference temperature values (Δt is constant).
	dtemp: Array
	# Δ for log of reference pressure values (Δlog(p) is constant).
	dln_p: Array
	# Major absorbing species in each band `(n_bnd, n_atmos_layers, 2)`.
	key_species: Array
	# Major absorption coefficient `(n_t_ref, n_p_ref, n_η, n_gpt)`.
	kmajor: Array
	# Minor absorption coefficient in lower atmosphere `(n_t_ref, n_η, n_contrib_lower)`.
	kminor_lower: Array
	# Minor absorption coefficient in upper atmosphere `(n_t_ref, n_η, n_contrib_upper)`.
	kminor_upper: Array
	# Starting and ending `g-point` for each band `(n_bnd, 2)`.
	bnd_lims_gpt: Array
	# Starting and ending wavenumber for each band `(n_bnd, 2)`.
	bnd_lims_wn: Array
	# `g-point` limits for minor contributors in lower atmosphere `(
	# n_contrib_lower, 2)`.
	minor_lower_gpt_lims: Array
	# `g-point` limits for minor contributors in upper atmosphere `(
	# n_contrib_upper, 2)`.
	minor_upper_gpt_lims: Array
	# Map from `g-point` to band.
	g_point_to_bnd: Array
	# Band number for minor contributor in the lower atmosphere
	# `(n_contrib_lower)`.
	minor_lower_bnd: Array
	# Band number for minor contributor in the upper atmosphere
	# `(n_contrib_upper)`.
	minor_upper_bnd: Array
	# Starting index to `idx_gases_minor_lower` for each band `(n_bnd)`.
	minor_lower_bnd_start: Array
	# Starting index to `idx_gases_minor_upper` for each band `(n_bnd)`.
	minor_upper_bnd_start: Array
	minor_upper_bnd_end: Array
	minor_lower_bnd_end: Array
	# Shift in `kminor_lower` for each band `(n_min_absrb_lower)`.
	minor_lower_gpt_shift: Array
	# Shift in `kminor_upper` for each band `(n_min_absrb_upper)`.
	minor_upper_gpt_shift: Array
	# Indices for minor gases contributing to absorption in the lower atmosphere `(n_min_absrb_lower)`.
	idx_minor_gases_lower: Array
	# Indices for minor gases contributing to absorption in the upper atmosphere `(n_min_absrb_upper)`.
	idx_minor_gases_upper: Array
	# Indices for scaling gases in the lower atmosphere `(n_min_absrb_lower)`.
	idx_scaling_gases_lower: Array
	# Indices for scaling gases in the upper atmosphere `(n_min_absrb_upper)`.
	idx_scaling_gases_upper: Array
	# Minor gas (lower atmosphere) scales with density? `(n_minor_absrb_lower)`.
	minor_lower_scales_with_density: Array
	# Minor gas (upper atmosphere) scales with density? `(n_minor_absrb_upper)`.
	minor_upper_scales_with_density: Array
	# Minor gas (lower atmosphere) scales by compliment `(n_minor_absrb_lower)`.
	lower_scale_by_complement: Array
	# Minor gas (upper atmosphere) scales by compliment `(n_minor_absrb_upper)`.
	upper_scale_by_complement: Array
	# Reference pressures used by the lookup table `(n_p_ref)`.
	p_ref: Array
	# Reference temperatures used by the lookup table `(n_t_ref)`.
	t_ref: Array
	# Reference volume mixing ratios used by the lookup table `(n_t_ref, n_gases, 2)`.
	vmr_ref: Array
	# Mapping from gas name to index.
	idx_gases: dict[str, int]


def _create_rrtm_consistent_minor_gas_index(
    idx_gases: dict[str, int],
    gases_minor_arr: Sequence[Any],
    scaling_gases_arr: Sequence[Any],
) -> tuple[Array, Array]:
	"""Create a mapping from the minor and scaling absorber to the RRTM index.
	
	Args:
	idx_gases: A dictionary mapping gas name to its RRTM index.
	gases_minor_arr: An array of minor gas names.
	scaling_gases_arr: An array of scaling gas names. Some entries will be
	  empty, which implies that the corresponding minor absorber does not have a
	  scaling gas.
	
	Returns:
	A 2-tuple containing 1) a mapping from minor absorber index to RRTM gas
	index and 2) a mapping from minor absorber index to the corresponding
	scaling gas RRTM index.
	"""
	n_minor_absrb = len(gases_minor_arr)
	assert len(scaling_gases_arr) == n_minor_absrb, (
	  'The scaling gases array should have length equal to the number of'
	  ' minor absorbers.'
	)
	gases_minor_arr = [_bytes_to_str(b) for b in gases_minor_arr]
	scaling_gases_arr = [_bytes_to_str(b) for b in scaling_gases_arr]

	def idx_tensor(arr):
		idx = [-1] * len(arr)
		for i, g in enumerate(arr):
			if g:
				idx[i] = idx_gases[g]
				
		return jnp.array(idx, dtype=jnp.int32)

	idx_minor = idx_tensor(gases_minor_arr)
	idx_scale = idx_tensor(scaling_gases_arr)
	return idx_minor, idx_scale


def _minor_gas_mappings(
    g_point_to_bnd: np.ndarray,
    minor_gpt_limits: np.ndarray,
    n_bnd: int,
) -> tuple[Array, Array, Array, Array]:
	"""Compute useful mappings and offsets for the minor gases.
	
	Args:
	g_point_to_bnd: A 1D array mapping the g-point index to the unique frequency
	  band it belongs to.
	minor_gpt_limits: A 2D array mapping a minor absorber index to the two
	  endpoints of the g-point interval it contributes to.
	n_bnd: The total number of frequency bands.
	
	Returns:
	A 4-tuple containing 1) a mapping from minor absorber index to the
	frequency band it contributes to, 2) a mapping from the frequency band
	to the index of the first minor absorber that contributes to it (note that
	minor absorbers are grouped by band), 3) a mapping from the frequency band
	to the index of the last minor absorber that contributes to it, and 4) a
	mapping from the minor absorber index to its first index in the absorption
	coefficient table.
	"""
	n_minor_absrb = minor_gpt_limits.shape[0]
	# Map from the minor absorber index to the band it contributes to.
	minor_bnd = [
	    g_point_to_bnd[minor_gpt_limits[i, 0]] for i in range(n_minor_absrb)
	]
	minor_bnd_start = [n_minor_absrb] * n_bnd
	for bnd in set(minor_bnd):
		minor_bnd_start[bnd] = minor_bnd.index(bnd)
	minor_bnd_end = [n_minor_absrb] * n_bnd

	for bnd in set(minor_bnd):
		minor_bnd_end[bnd] = n_minor_absrb - 1 - minor_bnd[::-1].index(bnd)

	# Note that the `kminor` absorption coefficient table is indexed by all
	# combinations of minor absorbers and valid g-point, which is why the
	# following offset is useful. It allows easy access to the beginning of the
	# range of contributors associated with a given minor absorber. The g-point
	# will provide an additional offset to identify the specific `contributor`
	# in this range.
	minor_gpt_shift = [0] * n_minor_absrb
	for i in range(1, n_minor_absrb):
		minor_gpt_shift[i] = (
			minor_gpt_shift[i - 1]
			+ minor_gpt_limits[i - 1, 1]
			- minor_gpt_limits[i - 1, 0]
			+ 1
		)
	return (
		jnp.array(minor_bnd, dtype=jnp.int32),
		jnp.array(minor_bnd_start, dtype=jnp.int32),
		jnp.array(minor_bnd_end, dtype=jnp.int32),
		jnp.array(minor_gpt_shift, dtype=jnp.int32),
	)


def load_data(
    ds: nc.Dataset,
    tables: Mapping[str, Array],
    dims: Mapping[str, int],
) -> dict[str, Any]:
	"""Preprocesses the RRTMGP gas optics data.
	
	Args:
	ds: The original netCDF Dataset containing the RRTMGP optics data.
	tables: The extracted data as a dictionary of `Array`s.
	dims: A dictionary containing dimension information for the tables.
	
	Returns:
	A dictionary containing dimension information and the preprocessed RRTMGP
	data as `Array`s.
	"""
	p_ref = tables['press_ref']
	t_ref = tables['temp_ref']
	p_ref_min = jnp.min(p_ref)
	temperature_ref_min = jnp.min(t_ref)
	temperature_ref_max = jnp.max(t_ref)
	dtemp = t_ref[1] - t_ref[0]
	dln_p = jnp.log(p_ref[0]) - jnp.log(p_ref[1])
	gas_names_ds = ds['gas_names'][:].data
	gas_names = []
	for gas_name in gas_names_ds:
		gas_names.append(''.join([g_i.decode('utf-8') for g_i in gas_name]).strip())
	# Prepend a dry air key to the list of names so that the 0 index is reserved
	# for dry air and all the other names follow a 1-based index system,
	# consistent with the RRTMGP species indices.
	gas_names.insert(0, constants.DRY_AIR_KEY)
	idx_gases = dataloader_base.create_index(gas_names)
	# Map all h2o related species to the same index.
	idx_h2o = idx_gases['h2o']
	# water vapor - foreign
	idx_gases['h2o_frgn'] = idx_h2o
	# water vapor - self-continua
	idx_gases['h2o_self'] = idx_h2o
	# What follows are data structures required for handling minor gas optics.
	idx_minor_gases_lower, idx_scaling_gases_lower = (
		_create_rrtm_consistent_minor_gas_index(
			idx_gases,
			ds['minor_gases_lower'][:].data,
			ds['scaling_gas_lower'][:].data
		)
	)
	idx_minor_gases_upper, idx_scaling_gases_upper = (
		_create_rrtm_consistent_minor_gas_index(
			idx_gases,
			ds['minor_gases_upper'][:].data,
			ds['scaling_gas_upper'][:].data
		)
	)
	# Decrement indices since RRTMGP was originally developed in a 1-based index
	# system.
	bnd_limits_gpt = ds['bnd_limits_gpt'][:].data - 1
	g_point_to_bnd = np.asarray([None] * dims['gpt'])
	for i in range(dims['bnd']):
		g_point_to_bnd[bnd_limits_gpt[i, 0] : bnd_limits_gpt[i, 1] + 1] = i
	g_point_to_bnd = np.array(g_point_to_bnd, dtype=jnp.int32)
	minor_lower_gpt_lims = ds['minor_limits_gpt_lower'][:].data - 1
	(
		minor_lower_bnd,
		minor_lower_bnd_start,
		minor_lower_bnd_end,
		minor_lower_gpt_shift,
	) = _minor_gas_mappings(g_point_to_bnd, minor_lower_gpt_lims, dims['bnd'])
	
	minor_upper_gpt_lims = ds['minor_limits_gpt_upper'][:].data - 1
	(
		minor_upper_bnd,
		minor_upper_bnd_start,
		minor_upper_bnd_end,
		minor_upper_gpt_shift,
	) = _minor_gas_mappings(g_point_to_bnd, minor_upper_gpt_lims, dims['bnd'])
	
	return dict(
		idx_h2o=idx_h2o,
		idx_o3=idx_gases['o3'],
		idx_gases=idx_gases,
		n_gases=dims['absorber'],
		n_bnd=dims['bnd'],
		n_gpt=dims['gpt'],
		n_atmos_layers=dims['atmos_layer'],
		n_t_ref=dims['temperature'],
		n_p_ref=dims['pressure'],
		n_maj_absrb=dims['absorber'],
		n_minor_absrb=dims['minor_absorber'],
		n_minor_absrb_lower=dims['minor_absorber_intervals_lower'],
		n_minor_absrb_upper=dims['minor_absorber_intervals_upper'],
		n_contrib_lower=dims['contributors_lower'],
		n_contrib_upper=dims['contributors_upper'],
		n_mixing_fraction=dims['mixing_fraction'],
		p_ref_tropo=tables['press_ref_trop'],
		t_ref_absrb=tables['absorption_coefficient_ref_T'],
		p_ref_absrb=tables['absorption_coefficient_ref_P'],
		key_species=tables['key_species'],
		kmajor=tables['kmajor'],
		kminor_lower=tables['kminor_lower'],
		kminor_upper=tables['kminor_upper'],
		bnd_lims_gpt=tables['bnd_limits_gpt'] - 1,
		bnd_lims_wn=tables['bnd_limits_wavenumber'],
		g_point_to_bnd=jnp.array(g_point_to_bnd),
		minor_lower_bnd=jnp.array(minor_lower_bnd),
		minor_lower_bnd_start=jnp.array(minor_lower_bnd_start),
		minor_lower_bnd_end=jnp.array(minor_lower_bnd_end),
		minor_lower_gpt_shift=jnp.array(minor_lower_gpt_shift),
		minor_upper_bnd=jnp.array(minor_upper_bnd),
		minor_upper_bnd_start=jnp.array(minor_upper_bnd_start),
		minor_upper_bnd_end=jnp.array(minor_upper_bnd_end),
		minor_upper_gpt_shift=jnp.array(minor_upper_gpt_shift),
		idx_minor_gases_lower=jnp.array(idx_minor_gases_lower),
		idx_scaling_gases_lower=jnp.array(idx_scaling_gases_lower),
		idx_minor_gases_upper=jnp.array(idx_minor_gases_upper),
		idx_scaling_gases_upper=jnp.array(idx_scaling_gases_upper),
		minor_lower_gpt_lims=jnp.array(minor_lower_gpt_lims),
		minor_upper_gpt_lims=jnp.array(minor_upper_gpt_lims),
		minor_lower_scales_with_density=tables['minor_scales_with_density_lower'],
		minor_upper_scales_with_density=tables['minor_scales_with_density_upper'],
		lower_scale_by_complement=tables['scale_by_complement_lower'],
		upper_scale_by_complement=tables['scale_by_complement_upper'],
		p_ref=p_ref,
		t_ref=t_ref,
		p_ref_min=p_ref_min,
		temperature_ref_min=temperature_ref_min,
		temperature_ref_max=temperature_ref_max,
		dtemp=dtemp,
		dln_p=dln_p,
		vmr_ref=tables['vmr_ref'],
	)
