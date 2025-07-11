"""A base Dataclass for RRTMGP lookup tables."""

import collections
import dataclasses
import json
import csv
from typing import Callable, TypeAlias

import numpy as np
from absl import flags
from etils import epath
import jax
import jax.numpy as jnp

from jcm.rrtmgp.config import radiative_transfer
from jcm.rrtmgp.optics import constants
from jcm.rrtmgp.optics import optics_utils

Array: TypeAlias = jax.Array

def parse_csv_file(
    path: str,
) -> dict[str, np.ndarray]:
	"""Reads and returns the columns of the input csv file (with header row).
	
	Args:
	path: Path of the csv file.
	
	Returns:
	A dictionary mapping column names to an array of the column values.
	"""
	out = {}
	with epath.Path(path).open('r') as f:
		for i, row in enumerate(csv.DictReader(f)):
			for key, value in row.items():
				out.setdefault(key, []).append(float(value))
				
			assert all(
				len(values) == i + 1 for values in out.values()
			), f'Missing values in {path} while processing {row}.'
			
	return {
		key: np.array(values, dtype=np.float32) for key, values in out.items()
	}

_USE_RCEMIP_OZONE_PROFILE = flags.DEFINE_bool(
	'use_rcemip_ozone_profile',
	False,
	'If true, the analytic profile for ozone given for the RCEMIP configuration'
	' will be used, overriding the provided lookup table.  If no lookup table'
	' is provided, then ozone will not be included (as usual).',
	allow_override=True,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LookupVolumeMixingRatio:
	"""Lookup table of volume mixing ratio profiles of atmospheric gases."""
	
	# Volume mixing ratio (vmr) global mean of predominant atmospheric gas
	# species, keyed by chemical formula.
	global_means: dict[str, float]
	# Volume mixing ratio profiles, keyed by chemical formula.
	profiles: dict[str, Array] | None = None


def from_config(
    atmospheric_state_cfg: radiative_transfer.AtmosphericStateCfg,
) -> LookupVolumeMixingRatio:
	"""Instantiate a `LookupVolumeMixingRatio` object from config.
	
	The proto contains atmospheric conditions, the path to a json file
	containing globally averaged volume mixing ratio for various gas species,
	and the path to a file containing the volume mixing ratio sounding data for
	certain gas species. The gas species will be identified by their chemical
	formula in lowercase (e.g., 'h2o`, 'n2o', 'o3'). Each entry of the profile
	corresponds to the pressure level under 'p_ref', which is a required column.
	
	Args:
	atmospheric_state_cfg: The atmospheric state configuration.
	
	Returns:
	A `LookupVolumeMixingRatio` object.
	"""
	vmr_sounding_filepath = atmospheric_state_cfg.vmr_sounding_filepath
	if vmr_sounding_filepath:
		vmr_sounding = parse_csv_file(vmr_sounding_filepath)
	else:
		vmr_sounding = None
	
	profiles = None
	if vmr_sounding is not None:
		assert (
			'p_ref' in vmr_sounding
		), f'Missing p_ref column in sounding file {vmr_sounding_filepath}'
		profiles = {
			key: jnp.array(values, dtype=jnp.float32)
			for key, values in vmr_sounding.items()
		}
	
	# Dry air is a special case that always has a volume mixing ratio of 1
	# since, by definition, vmr is normalized by the number of moles of dry air.
	global_means = {
		constants.DRY_AIR_KEY: constants.DRY_AIR_VMR,
	}
	
	vmr_global_mean_filepath = atmospheric_state_cfg.vmr_global_mean_filepath
	if vmr_global_mean_filepath:
		with epath.Path(vmr_global_mean_filepath).open('r') as f:
			global_means.update(json.loads(f.read()))
	
	return LookupVolumeMixingRatio(global_means=global_means, profiles=profiles)


def _vmr_interpolant_fn(
    p_for_interp: Array,
    vmr_profile: Array,
) -> Callable[[Array], Array]:
	"""Create a volume mixing ratio interpolant for the given profile."""
	
	def interpolant_fn(p: Array) -> Array:
		interp = optics_utils.create_linear_interpolant(
			jnp.log(p), jnp.log(p_for_interp)
		)
		return optics_utils.interpolate(
			vmr_profile, collections.OrderedDict({'p': lambda: interp})
		)

	return interpolant_fn


def reconstruct_vmr_fields_from_pressure(
    lookup_volume_mixing_ratio: LookupVolumeMixingRatio,
    pressure: Array,
) -> dict[str, Array]:
	"""Reconstruct volume mixing ratio fields for a given pressure field.
	
	The volume mixing ratio fields are reconstructed for the gas species that
	have spatially variable profiles available from sounding data.
	
	Args:
	lookup_volume_mixing_ratio: An instance of `LookupVolumeMixingRatio`.
	pressure: The pressure field, in Pa.
	
	Returns:
	A dictionary keyed by chemical formula of volume mixing ratio fields
	interpolated to the 3D grid.
	"""
	if lookup_volume_mixing_ratio.profiles is None:
		return {}

	p_for_interp = lookup_volume_mixing_ratio.profiles['p_ref']

	output = {}
	for k, profile in lookup_volume_mixing_ratio.profiles.items():
		if k == 'p_ref':
			continue
	
	if k == 'o3' and _USE_RCEMIP_OZONE_PROFILE.value:
		def o3_from_p(p: Array) -> Array:
			"""The ozone analytic profile from RCEMIP-I; see Wing et al (2018)."""
			p_hpa = p / 100  # Convert from Pa to hPa.
			g1 = 3.6478
			g2 = 0.83209
			g3 = 11.3515
			o3 = g1 * p_hpa**g2 * jnp.exp(-p_hpa / g3)
			o3 = 1e-6 * o3  # Conve from ppm to vmr.
			return o3
		output[k] = o3_from_p(pressure)
	
	else:
		interpolant_fn = _vmr_interpolant_fn(p_for_interp, profile)
		output[k] = interpolant_fn(pressure)
	return output
